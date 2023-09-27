from jax import jit, Array
import jax.numpy as jnp
from functools import partial


@partial(jit, static_argnames=["stats"])
def azimuthal_profile(
    q: Array, r: Array, r_bins: Array, stats=(jnp.nanmean, jnp.nanstd)
) -> (Array, Array, Array):
    """Azimuthal profile of `q`, evaluated at radii `r`, with radial
    binning `r_bins`.

    Parameters
    ----------
    q : array
        Quantity of interest
    r : array
        Radii at which the quantity of interest is evaluated
    r_bins : array
        Radial bins to be used for the radial profile
    stat : str
        Which statistic to be used within annuli, "mean" or "median",
        defaults to "mean"

    Returns
    -------
    array
        center of 1d bins, shape=(n_bins - 1)
    array
        mean value of q in bins, shape=(n_bins - 1)
    array
        standard deviation of q in bins, shape=(n_bins - 1)
    """
    q, r = q.flatten(), r.flatten()

    mean = []
    std = []

    for i_r in range(len(r_bins) - 1):
        r_l, r_u = r_bins[i_r], r_bins[i_r + 1]
        q_i = jnp.where((r >= r_l) & (r < r_u), q, jnp.nan)
        mean.append(stats[0](q_i))
        std.append(stats[1](q_i))

    r_1d = r_bins[:-1] + 0.5 * jnp.ediff1d(r_bins)
    return r_1d, jnp.array(mean), jnp.array(std)
