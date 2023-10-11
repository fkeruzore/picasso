from jax import jit, value_and_grad, Array
import jax.numpy as jnp
import optax
from functools import partial
from typing import Callable


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


def optimize(
    loss_fn: Callable,
    start: Array,
    optimizer: optax.GradientTransformation = optax.adam(learning_rate=1e-3),
    n_steps: int = 10_000,
    loss_tol: float = 1e-16,
) -> Array:
    """Optimize a loss function and returns the gradient descent in
    parameter and loss space.

    Parameters
    ----------
    loss_fn : Callable
        The loss function
    start : Array
        Parameter space starting point for the gradient descent
    optimizer : optax.GradientTransformation, optional
        An `optax` minimizer, by default optax.adam(learning_rate=1e-3)
    n_steps : int, optional
        Maximum number of gradient descent steps, by default 10_000,
    loss_tol : float, optional
        Maximum relative loss change for convergence, by default 1e-16

    Returns
    -------
    Array
        The gradient descent results in parameter and loss space,
        shape=(# of steps, # of parameters + 1).
        The last column is the loss function values.
    """
    opt_state = optimizer.init(start)
    chain = []

    @jit
    def step(par, opt_state):
        loss_value, grads = value_and_grad(loss_fn)(par)
        updates, opt_state = optimizer.update(grads, opt_state, par)
        par = optax.apply_updates(par, updates)
        return par, opt_state, loss_value

    par = start.copy()
    old_loss_value = jnp.inf  # loss_fn(par)
    chain.append([*par, old_loss_value])

    for _ in range(n_steps):
        par, opt_state, loss_value = step(par, opt_state)
        chain.append([*par, loss_value])
        delta_loss = 1.0 - (loss_value / old_loss_value)
        if delta_loss < loss_tol:
            break
        old_loss_value = loss_value

    return jnp.array(chain)
