from pathlib import Path
import pickle
import json
from functools import partial
import gc

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from flax import serialization

from lenia.lenia import ConfigLenia, Lenia
from vae import VAE
from qdax.core.containers.unstructured_repertoire import UnstructuredRepertoire

import mediapy
from omegaconf import OmegaConf

# Reduce batch size to prevent OOM
BATCH_SIZE = 8

# Memory and precision configurations
jax.config.update('jax_default_matmul_precision', 'bfloat16')
jax.config.update('jax_platform_name', 'gpu')
jax.config.update('jax_enable_x64', False)

def plot_aurora_repertoire(config, repertoire, descriptors_3d):
    plt.clf()
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Move data to CPU before plotting
    fitnesses = jax.device_get(repertoire.fitnesses)
    
    sc = ax.scatter(descriptors_3d[:, 0], 
                   descriptors_3d[:, 1], 
                   descriptors_3d[:, 2], 
                   c=fitnesses, 
                   cmap="viridis")
    
    ax.set_title(f"AURORA repertoire {config.qd.fitness}")
    fig.colorbar(sc, shrink=0.5, aspect=5)
    
    return fig

@partial(jax.jit, static_argnums=(0,))
def process_vae_batch(vae, params, phenotypes, key):
    key1, key2 = jax.random.split(key)
    latent = jax.checkpoint(partial(vae.apply, method=vae.encode))(
        params, phenotypes, key1)
    return jax.checkpoint(partial(vae.apply, method=vae.generate))(
        params, latent, key2)

def save_batch_results(accum, accum_small, accum_medium, accum_large, 
                      phenotype_recon, batch_idx, dirs):
    # Move data to CPU once
    phenotypes = {
        'video': jax.device_get(accum.phenotype),
        'small': jax.device_get(accum_small.phenotype[:, -1]),
        'medium': jax.device_get(accum_medium.phenotype[:, -1]),
        'large': jax.device_get(accum_large.phenotype[:, -1]),
        'vae': jax.device_get(phenotype_recon[:, -1])
    }
    
    for j in range(len(phenotypes['video'])):
        idx = batch_idx * BATCH_SIZE + j
        
        # Save video
        mediapy.write_video(dirs['video'] / f"{idx:04d}.mp4", 
                          phenotypes['video'][j], fps=50)
        
        # Save images
        mediapy.write_image(dirs['small'] / f"{idx:04d}.png", 
                          phenotypes['small'][j])
        mediapy.write_image(dirs['medium'] / f"{idx:04d}.png", 
                          phenotypes['medium'][j])
        mediapy.write_image(dirs['large'] / f"{idx:04d}.png", 
                          phenotypes['large'][j])
        mediapy.write_image(dirs['vae'] / f"{idx:04d}.png", 
                          phenotypes['vae'][j])
    
    # Clear some memory after saving
    del phenotypes
    gc.collect()

def process_genotype_batch(lenia, init_carry, step_fns, vae, params, key, genotypes):
    # Express genotypes
    carries = jax.vmap(lambda g: lenia.express_genotype(init_carry, g))(genotypes)
    
    # Evaluate at different scales
    results = {}
    for name, step_fn in step_fns.items():
        _, accum = jax.vmap(lambda c: jax.lax.scan(
            step_fn, init=c, xs=jnp.arange(lenia._config.n_step)))(carries)
        results[name] = accum

    # Process VAE
    key, subkey = jax.random.split(key)
    phenotype_recon = process_vae_batch(
        vae, params, results['small'].phenotype, subkey)
    
    return results['full'], results['small'], results['medium'], results['large'], phenotype_recon

def visualize_aurora(run_dir):
    # Create directories
    run_dir = Path(run_dir)
    dirs = {
        'vis': run_dir / "visualization",
        'video': run_dir / "visualization" / "video",
        'small': run_dir / "visualization" / "phenotype_small",
        'medium': run_dir / "visualization" / "phenotype_medium",
        'large': run_dir / "visualization" / "phenotype_large",
        'vae': run_dir / "visualization" / "vae"
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    # Load config and initialize
    config = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    key = jax.random.PRNGKey(config.seed)
    
    # Initialize Lenia
    config_lenia = ConfigLenia(
        pattern_id=config.pattern_id,
        world_size=config.world_size,
        world_scale=config.world_scale,
        n_step=config.n_step,
        n_params_size=config.n_params_size,
        n_cells_size=config.n_cells_size,
    )
    lenia = Lenia(config_lenia)

    # Setup evaluation functions with different sizes
    step_fns = {
        'full': partial(lenia.step, phenotype_size=config.world_size, 
                       center_phenotype=False, record_phenotype=True),
        'small': partial(lenia.step, phenotype_size=config.phenotype_size, 
                        center_phenotype=True, record_phenotype=True),
        'medium': partial(lenia.step, phenotype_size=64, 
                         center_phenotype=True, record_phenotype=True),
        'large': partial(lenia.step, phenotype_size=config.world_size, 
                        center_phenotype=True, record_phenotype=True)
    }

    # Load initial state and repertoire
    init_carry, init_genotype, _ = lenia.load_pattern(lenia.pattern)
    _, reconstruction_fn = ravel_pytree(init_genotype)
    repertoire = UnstructuredRepertoire.load(
        reconstruction_fn=reconstruction_fn, 
        path=str(run_dir) + "/repertoire/")

    # Move repertoire data to CPU for t-SNE
    print("Computing t-SNE...")
    descriptors = jax.device_get(repertoire.descriptors)
    tsne = TSNE(n_components=3, perplexity=10., method='barnes_hut', n_jobs=-1)
    descriptors_3d = tsne.fit_transform(descriptors)

    # Plot and save repertoire
    fig = plot_aurora_repertoire(config, repertoire, descriptors_3d)
    fig.savefig(dirs['vis'] / "repertoire.pdf", bbox_inches="tight")
    plt.close()

    # Save descriptors for HTML visualization
    with open(dirs['vis'] / "descriptors_3d.json", "w") as f:
        json.dump(descriptors_3d.tolist(), f)

    # Initialize VAE
    key, subkey1, subkey2 = jax.random.split(key, 3)
    phenotype_fake = jnp.zeros((config.phenotype_size, 
                               config.phenotype_size, 
                               lenia.n_channel))
    vae = VAE(img_shape=phenotype_fake.shape, 
              latent_size=config.qd.hidden_size, 
              features=config.qd.features)
    params = vae.init(subkey1, phenotype_fake, subkey2)

    # Load VAE parameters
    with open(run_dir / "params.pickle", "rb") as f:
        params = serialization.from_state_dict(params, pickle.load(f))

    # Process genotypes in batches
    n_batches = repertoire.size // BATCH_SIZE
    genotypes = jax.device_get(repertoire.genotypes)  # Move to CPU once
    
    for i in range(n_batches):
        print(f"Processing batch {i+1}/{n_batches}")
        
        # Get batch of genotypes
        start_idx = i * BATCH_SIZE
        end_idx = start_idx + BATCH_SIZE
        genotypes_batch = jax.device_put(genotypes[start_idx:end_idx])  # Move batch to GPU
        
        # Process batch
        results = process_genotype_batch(
            lenia, init_carry, step_fns, vae, params, key, genotypes_batch)
        
        # Save results
        save_batch_results(*results, i, dirs)
        
        # Clear batch from GPU
        del results, genotypes_batch
        gc.collect()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        visualize_aurora(sys.argv[1])
    else:
        run_dirs = [
            "path/to/your/results/directory",
        ]
        for run_dir in run_dirs:
            visualize_aurora(run_dir)