# lenia_hpc.py
import os
import jax
import jax.numpy as jnp
import flax
import optax
import hydra
from omegaconf import DictConfig
import pickle
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from jax.flatten_util import ravel_pytree
from flax import serialization

# Import Lenia-specific modules
from lenia.lenia import ConfigLenia, Lenia
from vae import VAE
from qdax.core.containers.unstructured_repertoire import UnstructuredRepertoire

def main():
    # Print JAX device info
    print("JAX devices:", jax.devices())
    print("GPU available:", bool(jax.devices('gpu')))
    
    # Run the main evolutionary loop
    os.system("python main_aurora.py")
    
    # Create output directory
    output_dir = Path("output/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    def find_latest_output():
        output_paths = list(Path("output/aurora").glob("*/202**/metrics.pickle"))
        if not output_paths:
            print("No output directories found yet...")
            return None
        return max(output_paths, key=os.path.getctime)
    
    def plot_results():
        metrics_path = find_latest_output()
        if not metrics_path:
            return

        print(f"Loading metrics from: {metrics_path}")
        with open(metrics_path, "rb") as f:
            metrics = pickle.load(f)

        # Create plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Evolution Metrics', fontsize=16)

        # Plot metrics
        axes[0,0].plot(metrics['generation'], metrics['qd_score'], label='QD Score')
        axes[0,0].set_xlabel('Generation')
        axes[0,0].set_ylabel('Score')
        axes[0,0].set_title('Quality Diversity Score')

        axes[0,1].plot(metrics['generation'], metrics['coverage'], label='Coverage')
        axes[0,1].set_xlabel('Generation')
        axes[0,1].set_ylabel('Coverage')
        axes[0,1].set_title('Niche Coverage')

        axes[0,2].plot(metrics['generation'], metrics['max_fitness'], label='Max Fitness')
        axes[0,2].set_xlabel('Generation')
        axes[0,2].set_ylabel('Fitness')
        axes[0,2].set_title('Maximum Fitness')

        if 'variance' in metrics:
            axes[1,1].plot(metrics['generation'], metrics['variance'], label='Variance', color='green')
            axes[1,1].set_xlabel('Generation')
            axes[1,1].set_ylabel('Variance')
            axes[1,1].set_title('Population Variance')

        if 'n_elites' in metrics:
            cumulative_elites = np.cumsum(metrics['n_elites'])
            axes[1,2].plot(metrics['generation'], cumulative_elites, label='Number of Elites', color='orange')
            axes[1,2].set_xlabel('Generation')
            axes[1,2].set_ylabel('Count')
            axes[1,2].set_title('Cumulative Elite Solutions')

        plt.tight_layout()
        plt.savefig(output_dir / 'evolution_metrics.png')
        plt.close()

        # Print summary statistics to file
        with open(output_dir / 'summary_stats.txt', 'w') as f:
            print("\nSummary Statistics:", file=f)
            print(f"Final QD Score: {metrics['qd_score'][-1]:.2f}", file=f)
            print(f"Final Coverage: {metrics['coverage'][-1]:.2%}", file=f)
            print(f"Peak Fitness: {np.max(metrics['max_fitness']):.2f}", file=f)
            if 'variance' in metrics:
                print(f"Final Variance: {metrics['variance'][-1]:.2e}", file=f)
            if 'n_elites' in metrics:
                print(f"Total Elite Solutions: {cumulative_elites[-1]:.0f}", file=f)

    # Run visualization
    plot_results()

if __name__ == "__main__":
    main()
