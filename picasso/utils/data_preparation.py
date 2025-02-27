from jax import Array
import jax.numpy as jnp
import jax.scipy.stats as jss


def transform_minmax(x: Array, minmaxs: Array):
    mins, maxs = minmaxs
    return (x - mins) / (maxs - mins)


def inv_transform_minmax(x: Array, minmaxs: Array):
    mins, maxs = minmaxs
    return x * (maxs - mins) + mins


def quantile_normalization(x: Array, dist=jss.norm):
    ranks = jnp.argsort(x, axis=0)
    sorted_ranks = jnp.argsort(ranks, axis=0)
    normalized = dist.ppf((sorted_ranks + 0.5) / x.shape[0])
    return normalized
