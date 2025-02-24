import os
import time
import pickle
from functools import partial
import logging
logger = logging.getLogger(__name__) 

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import optax

from lenia.lenia import ConfigLenia, Lenia
from vae import VAE
from vae import loss as loss_vae
from qdax.core.aurora import AURORA
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.utils.metrics import CSVLogger, default_qd_metrics
from common import get_metric, repertoire_variance

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="configs/", config_name="aurora")
def main(config: DictConfig) -> None:
	logging.info("Starting AURORA...")

	# Init a random key
	key = jax.random.PRNGKey(config.seed)

	# Lenia
	logging.info("Initializing Lenia...")
	config_lenia = ConfigLenia(
		# Init pattern
		pattern_id=config.pattern_id,

		# Simulation
		world_size=config.world_size,
		world_scale=config.world_scale,
		n_step=config.n_step,

		# Genotype
		n_params_size=config.n_params_size,
		n_cells_size=config.n_cells_size,
	)
	lenia = Lenia(config_lenia)

	# Load pattern
	init_carry, init_genotype, other_asset = lenia.load_pattern(lenia.pattern)

	# VAE
	key, subkey_1, subkey_2 = jax.random.split(key, 3)
	phenotype_fake = jnp.zeros((config.phenotype_size, config.phenotype_size, lenia.n_channel))
	vae = VAE(img_shape=phenotype_fake.shape, latent_size=config.qd.hidden_size, features=config.qd.features)
	params = vae.init(subkey_1, phenotype_fake, subkey_2)

	params_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
	logging.info(f"VAE params count: {params_count}")

	# Create train state
	train_steps_per_epoch = config.qd.repertoire_size // config.qd.ae_batch_size
	train_steps_total = config.qd.n_generations * config.qd.train_ratio * train_steps_per_epoch
	learning_rate_fn = optax.linear_schedule(
		init_value=config.qd.lr_init_value,
		end_value=config.qd.lr_init_value,
		transition_steps=config.qd.lr_transition_steps,
		transition_begin=config.qd.lr_transition_begin,
	)
	tx = optax.chain(
		optax.clip_by_global_norm(1.0),
		optax.adam(learning_rate_fn),
	)
	train_state = TrainState.create(apply_fn=vae.apply, params=params, tx=tx)

	# Define the scoring function
	def latent_mean(observation, train_state, key):
		latents = vae.apply(train_state.params, observation.phenotype[-config.qd.n_keep:], key, method=vae.encode)
		return jnp.mean(latents, axis=-2)

	def latent_variance(observation, train_state, key):
		latents = vae.apply(train_state.params, observation.phenotype[-config.qd.n_keep:], key, method=vae.encode)
		latent_mean = jnp.mean(latents, axis=-2)
		return -jnp.mean(jnp.linalg.norm(latents - latent_mean[..., None, :], axis=-1), axis=-1)

	def homeodynamics(observation, train_state, key):
		latents = vae.apply(train_state.params, observation.phenotype[-config.qd.n_keep:], key, method=vae.encode)
		diffs = latents[1:] - latents[:-1]
		return -jnp.std(diffs, axis=0).mean()
	
	def calculate_average_mass(repertoire):
		"""Calculate average mass across valid solutions in repertoire"""
		valid_solutions = repertoire.fitnesses != -jnp.inf
		valid_masses = repertoire.observations.stats.mass[valid_solutions]
		valid_masses = jnp.mean(valid_masses, axis=1)  # Average across timesteps for each solution  
		return jnp.mean(valid_masses) if valid_masses.size > 0 else 0.0

	def compute_domination_count(objectives, archive_objectives):
		"""Count how many archive solutions dominate this solution"""
		# A solution dominates another if it's better in at least one objective
		# and not worse in all others
		dominates = jnp.all(archive_objectives >= objectives, axis=1) & \
					jnp.any(archive_objectives > objectives, axis=1)
		return jnp.sum(dominates)
	
	def compute_sparsity(descriptors, archive_descriptors, sigma=0.5):
		"""
		Compute sparsity score based on distance to archive solutions in descriptor space.
		Higher score = more sparse/novel area of descriptor space.
		
		Args:
			descriptors: Descriptors of current solution 
			archive_descriptors: Descriptors of solutions in archive
			sigma: Kernel width for density estimation
		"""
		# Compute distances to all archive solutions
		distances = jnp.linalg.norm(
			descriptors[:, None] - archive_descriptors[None, :], 
			axis=-1
		)
		
		# Convert distances to density estimate using RBF kernel
		density = jnp.sum(jnp.exp(-distances**2 / (2 * sigma**2)), axis=-1)
		
		# Return negative density (higher = more sparse)
		return -density

	def pareto_fitness(observation, train_state, key, repertoire):
		"""Calculate fitness using Pareto-based comparison against archive"""
		# Calculate individual objectives
		if repertoire is None or repertoire.fitnesses.size == 0:
			return latent_variance(observation, train_state, key)
		
		latent_mean = latent_mean(observation, train_state, key)

		objectives = jnp.array([
			latent_variance(observation, train_state, key),  # homeostasis
			jnp.linalg.norm(  # novelty
				latent_mean - 
				latent_mean.mean(axis=0), 
				axis=-1
			),
			compute_sparsity(  # sparsity in descriptor space
				descriptor_fn(observation, train_state, key),
				descriptor_fn(repertoire.observations, train_state, key)
			)
		])
		
		# Get archive objectives
		archive_objectives = jnp.stack([
			latent_variance(repertoire.observations, train_state, key),
			jnp.linalg.norm(
				latent_mean(repertoire.observations, train_state, key) - 
				latent_mean(repertoire.observations, train_state, key).mean(axis=0), 
				axis=-1
			),
			compute_sparsity(  # sparsity in descriptor space
				descriptor_fn(observation, train_state, key),
				descriptor_fn(repertoire.observations, train_state, key)
			)
		])

		# Calculate domination-based fitness
		domination_count = compute_domination_count(objectives, archive_objectives)
		
		# Return negative domination count (fewer dominating solutions = better fitness)
		return -domination_count

	def fitness_fn(observation, train_state, key, repertoire=None):
		# if config.qd.fitness == "unsupervised":

		# fitness = latent_variance(observation, train_state, key)
		# fitness = pareto_fitness(observation, train_state, key, repertoire)
		fitness = homeodynamics(observation, train_state, key)

		# else:
		# 	fitness = get_metric(observation, config.qd.fitness, config.qd.n_keep)
		# 	assert fitness.size == 1
		# 	fitness = jnp.squeeze(fitness)

		# if config.qd.secondary_fitness:
		# 	secondary_fitness = get_metric(observation, config.qd.secondary_fitness, config.qd.n_keep)
		# 	assert secondary_fitness.size == 1
		# 	secondary_fitness = jnp.squeeze(secondary_fitness)
		# 	fitness += config.qd.secondary_fitness_weight * secondary_fitness

		failed = jnp.logical_or(observation.stats.is_empty.any(), observation.stats.is_full.any())
		failed = jnp.logical_or(failed, observation.stats.is_spread.any())
		fitness = jnp.where(failed, -jnp.inf, fitness)
		return fitness

	def descriptor_fn(observation, train_state, key):
		descriptor_unsupervised = latent_mean(observation, train_state, key)
		return descriptor_unsupervised

	def evaluate(genotype, train_state, key, repertoire=None):
		carry = lenia.express_genotype(init_carry, genotype)
		lenia_step = partial(lenia.step, phenotype_size=config.phenotype_size, center_phenotype=config.center_phenotype, record_phenotype=config.record_phenotype)
		carry, accum = jax.lax.scan(lenia_step, init=carry, xs=jnp.arange(lenia._config.n_step))

		fitness = fitness_fn(accum, train_state, key, repertoire)
		descriptor = descriptor_fn(accum, train_state, key)
		accum = jax.tree.map(lambda x: x[-config.qd.n_keep_ae:], accum)
		return fitness, descriptor, accum

	def scoring_fn(genotypes, train_state, key, repertoire=None):
		batch_size = jax.tree.leaves(genotypes)[0].shape[0]
		key, *keys = jax.random.split(key, batch_size+1)
		fitnesses, descriptors, observations = jax.vmap(evaluate, in_axes=(0, None, 0, None))(genotypes, train_state, jnp.array(keys), repertoire)

		fitnesses_nan = jnp.isnan(fitnesses)
		descriptors_nan = jnp.any(jnp.isnan(descriptors), axis=-1)
		fitnesses = jnp.where(fitnesses_nan | descriptors_nan, -jnp.inf, fitnesses)

		return fitnesses, descriptors, {"observations": observations}, key

	# Define a metrics function
	metrics_fn = partial(default_qd_metrics, qd_offset=0.)

	# Define emitter
	variation_fn = partial(isoline_variation, iso_sigma=config.qd.iso_sigma, line_sigma=config.qd.line_sigma)
	mixing_emitter = MixingEmitter(
		mutation_fn=None,
		variation_fn=variation_fn,
		variation_percentage=1.0,
		batch_size=config.qd.batch_size
	)

	# Train
	if config.qd.use_data_augmentation:
		def data_augmentation(batch, key):
			# Flip
			batch_1, batch_2 = jnp.split(batch, 2)
			batch_2 = jnp.flip(batch_2, axis=1)
			batch = jnp.concatenate([batch_1, batch_2], axis=0)
			batch = jax.random.permutation(key, batch)

			# Rotate
			batch_1, batch_2, batch_3, batch_4 = jnp.split(batch, 4)
			batch_1 = jax.vmap(lambda x: jnp.rot90(x, k=0, axes=(0, 1)))(batch_1)
			batch_2 = jax.vmap(lambda x: jnp.rot90(x, k=1, axes=(0, 1)))(batch_2)
			batch_3 = jax.vmap(lambda x: jnp.rot90(x, k=2, axes=(0, 1)))(batch_3)
			batch_4 = jax.vmap(lambda x: jnp.rot90(x, k=3, axes=(0, 1)))(batch_4)
			batch = jnp.concatenate([batch_1, batch_2, batch_3, batch_4], axis=0)
			return batch
	else:
		def data_augmentation(batch, key):
			return batch

	@partial(jax.jit, static_argnames=("learning_rate_fn",))
	def train_step(train_state, batch, key, learning_rate_fn):

		def loss_fn(params):
			logits, mean, logvar = train_state.apply_fn(params, batch, key)
			return loss_vae(logits, batch, mean, logvar)

		(loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
		train_state = train_state.apply_gradients(grads=grads)

		learning_rate = learning_rate_fn(train_state.step)

		return train_state, {**aux, "learning_rate": learning_rate}

	def train_epoch(train_state, repertoire, key):
		steps_per_epoch = repertoire.size // config.qd.ae_batch_size

		key, subkey = jax.random.split(key)
		valid = repertoire.fitnesses != -jnp.inf
		indices = jax.random.choice(subkey, jnp.arange(repertoire.size), shape=(repertoire.size,), p=valid)
		indices = indices[:steps_per_epoch * config.qd.ae_batch_size]
		indices = indices.reshape((steps_per_epoch, config.qd.ae_batch_size))

		def scan_train_step(carry, x):
			train_state = carry
			batch_indices, key = x
			subkey_1, subkey_2, subkey_3 = jax.random.split(key, 3)
			step_indices = jax.random.randint(subkey_1, shape=(config.qd.ae_batch_size,), minval=0, maxval=config.qd.n_keep_ae)
			batch = repertoire.observations.phenotype[batch_indices, step_indices]
			batch = data_augmentation(batch, subkey_2)
			train_state, metrics = train_step(train_state, batch, subkey_3, learning_rate_fn)
			return train_state, metrics

		keys = jax.random.split(key, steps_per_epoch)
		train_state, metrics = jax.lax.scan(
			scan_train_step,
			train_state,
			(indices, keys),
			length=steps_per_epoch,
		)
		return train_state, metrics

	def train_fn(key, repertoire, train_state):
		def scan_train_epoch(carry, x):
			train_state = carry
			key = x
			train_state, metrics = train_epoch(train_state, repertoire, key)
			return train_state, metrics

		keys = jax.random.split(key, config.qd.train_ratio)
		train_state, metrics = jax.lax.scan(
			scan_train_epoch,
			train_state,
			keys,
			length=config.qd.train_ratio,
		)
		return train_state, metrics

	def fitness_fn_wrapper(obs, ts, key):
		"""Wrapper to make fitness function signature consistent for AURORA"""
		return fitness_fn(obs, ts, key, None)

	# Init AURORA
	aurora = AURORA(
		emitter=mixing_emitter,
    	scoring_fn=scoring_fn,
    	fitness_fn=fitness_fn_wrapper,
		descriptor_fn=descriptor_fn,
		train_fn=train_fn,
		metrics_fn=metrics_fn,
	)

	# Init step of the aurora algorithm
	logging.info("Initializing AURORA...")
	key, subkey = jax.random.split(key)
	init_genotypes = init_genotype[None, ...].repeat(config.qd.batch_size, axis=0)
	init_genotypes += jax.random.normal(subkey, shape=(config.qd.batch_size, lenia.n_gene)) * config.qd.iso_sigma
	repertoire, emitter_state, key = aurora.init(
		init_genotypes,
		train_state,
		config.qd.repertoire_size,
		key,
	)

	metrics = dict.fromkeys(["generation", "qd_score", "coverage", "max_fitness", "loss", "recon_loss", "kld_loss", "learning_rate", "n_elites", "variance", "time", "avg_mass"], jnp.array([]))
	csv_logger = CSVLogger("./log.csv", header=list(metrics.keys()))

	# Main loop
	logging.info("Starting main loop...")

	def aurora_scan(carry, unused):
		repertoire, train_state, key = carry

		# AURORA update
		repertoire, _, metrics, key = aurora.update(
			repertoire,
			None,
			key,
			train_state,
		)

		# AE training
		key, subkey = jax.random.split(key)
		repertoire, train_state, metrics_ae = aurora.train(
			repertoire, train_state, subkey
		)

		return (repertoire, train_state, key), (metrics, metrics_ae)

	for generation in range(0, config.qd.n_generations, config.qd.log_interval):
		start_time = time.time()
		(repertoire, train_state, key), (current_metrics, current_metrics_ae) = jax.lax.scan(
			aurora_scan,
			(repertoire, train_state, key),
			(),
			length=config.qd.log_interval,
		)
		timelapse = time.time() - start_time

		# Metrics
		current_metrics["generation"] = jnp.arange(1+generation, 1+generation+config.qd.log_interval, dtype=jnp.int32)
		current_metrics["n_elites"] = jnp.sum(current_metrics["is_offspring_added"], axis=-1)
		del current_metrics["is_offspring_added"]
		variance = repertoire_variance(repertoire)
		current_metrics["variance"] = jnp.repeat(variance, config.qd.log_interval)
		current_metrics["time"] = jnp.repeat(timelapse, config.qd.log_interval)
		avg_mass = calculate_average_mass(repertoire)
		current_metrics["avg_mass"] = jnp.repeat(avg_mass, config.qd.log_interval)

		current_metrics_ae = jax.tree_util.tree_map(lambda metric: jnp.repeat(metric[-1], config.qd.log_interval), current_metrics_ae)
		current_metrics |= current_metrics_ae

		metrics = jax.tree_util.tree_map(lambda metric, current_metric: jnp.concatenate([metric, current_metric], axis=0), metrics, current_metrics)

		# Log
		log_metrics = jax.tree_util.tree_map(lambda metric: metric[-1], metrics)  # log last value
		csv_logger.log(log_metrics)
		logging.info(log_metrics)

	# Metrics
	logging.info("Saving metrics...")
	with open("./metrics.pickle", "wb") as metrics_file:
		pickle.dump(metrics, metrics_file)

	# Repertoire
	logging.info("Saving repertoire...")
	os.mkdir("./repertoire/")
	repertoire.replace(observations=jnp.nan).save(path="./repertoire/")

	# Autoencoder
	logging.info("Saving autoencoder params...")
	with open("./params.pickle", "wb") as params_file:
		pickle.dump(train_state.params, params_file)


if __name__ == "__main__":
	main()