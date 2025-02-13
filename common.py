import pickle

import jax.numpy as jnp
import pandas as pd

from omegaconf import OmegaConf


def get_metric(observation, metric, n_keep):
    if not isinstance(metric, str) or "_" not in metric:
        raise ValueError(f"Invalid metric format: {metric}. Expected format: 'sign_metric_operator'")
    
    parts = metric.split("_")
    if len(parts) < 3:
        raise ValueError(f"Invalid metric format: {metric}. Expected at least 3 parts (sign_metric_operator)")
    
    sign = parts[0]
    operator = parts[-1]
    metric_name = "_".join(parts[1:-1])

    if operator == "avg":
        operator = jnp.mean
    elif operator == "var":
        operator = jnp.var
    elif operator == "max":
        operator = jnp.max
    else:
        raise NotImplementedError(f"Unsupported operator: {operator}")

    if sign == "pos":
        sign = 1.
    elif sign == "neg":
        sign = -1.
    else:
        raise NotImplementedError(f"Unsupported sign: {sign}")

    if metric_name == "mass":
        return sign * operator(observation.stats.mass[-n_keep:], keepdims=True)
    elif metric_name == "linear_velocity":  # equivalent to traveled distance from the origin
        return sign * jnp.sqrt(jnp.square(observation.stats.center_x[-1:] - observation.stats.center_x[-n_keep]) + jnp.square(observation.stats.center_y[-1:] - observation.stats.center_y[-n_keep]))
    elif metric_name == "angular_velocity":
        return sign * operator(observation.stats.angular_velocity[-n_keep:], keepdims=True)
    elif metric_name == "angle":
        return sign * operator(observation.stats.angle[-n_keep:], keepdims=True)
    elif metric_name == "center_x":
        return sign * operator(observation.stats.center_x[-n_keep:], keepdims=True)
    elif metric_name == "center_y":
        return sign * operator(observation.stats.center_y[-n_keep:], keepdims=True)
    elif metric_name == "color":
        return sign * operator(observation.phenotype[-n_keep:, ...], axis=(0, 1, 2))
    else:
        raise NotImplementedError


def get_config(run_dir):
	config = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
	return config


def get_metrics(run_dir):
	with open(run_dir / "metrics.pickle", "rb") as metrics_file:
		metrics = pickle.load(metrics_file)
	try:
		del metrics["loss"]
		del metrics["learning_rate"]
	except:
		pass
	return pd.DataFrame.from_dict(metrics)


def get_df(results_dir):
	metrics_list = []
	for fitness_dir in results_dir.iterdir():
		if fitness_dir.is_file():
			continue
		for run_dir in fitness_dir.iterdir():
			# Get config and metrics
			config = get_config(run_dir)
			metrics = get_metrics(run_dir)

			# Fitness
			try:
				metrics["fitness"] = config.qd.fitness
			except:
				metrics["fitness"] = "none"

			# Run
			metrics["run"] = run_dir.name

			# Number of Evaluations
			metrics["n_evaluations"] = metrics["generation"] * config.qd.batch_size

			# Coverage
			metrics["coverage"]

			metrics_list.append(metrics)
	return pd.concat(metrics_list, ignore_index=True)


def repertoire_variance(repertoire):
	is_occupied = (repertoire.fitnesses != -jnp.inf)
	var = jnp.var(repertoire.observations.phenotype[:, -1], axis=0, where=is_occupied[:, None, None, None])
	return jnp.mean(var)
