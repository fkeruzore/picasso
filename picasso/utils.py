from jax import jit, grad, value_and_grad, Array
import jax.numpy as jnp
import optax
from scipy.optimize import minimize
from functools import partial
from typing import Callable, Union


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
    try_bfgs: bool = True,
    return_chain: bool = False,
    n_steps: int = 10_000,
    backup_optimizer: optax.GradientTransformation = optax.adam(1e-3),
    backup_target_loss: float = 1e-8,
    backup_max_dloss: float = 1e-8,
) -> tuple[Array, float, int, Union[Array, None]]:
    """Optimize a loss function and returns the gradient descent in
    parameter and loss space.

    Parameters
    ----------
    loss_fn : Callable
        The loss function
    start : Array
        Parameter space starting point for the gradient descent
    try_bfgs : bool, optional
        If True, first tries minimizing the loss function using scipy
        optimize with methof `L-BFGS-B`, using jax's gradients, by
        default True
    backup_optimizer : optax.GradientTransformation, optional
        An `optax` minimizer to use as backup if BFGS doesn't converge
        (or if `try_bfgs=False`), by default optax.adam(1e-3)
    return_chain : bool, optional
        If True, return the gradient descent evolution in parameter and
        loss space, by default False
    n_steps : int, optional
        Maximum number of gradient descent steps, by default 10_000,
    backup_target_loss : float, optional
        Maximum loss value for convergence, by default 1e-8
    backup_max_delta_loss : float, optional
        Maximum relative loss change for convergence, by default 1e-8

    Returns
    -------
    best_par : Array
        Best-fit parameters
    best_loss : float
        Best-fit loss value
    status : int
        Success status. If 0, did not converge; if 1, converged when
        running BFGS; if 2, converged when running the backup optimizer
    chain : Array or None
        The gradient descent results in parameter and loss space,
        shape=(# of steps, # of parameters + 1).
        The last column is the loss function values.

    Notes
    -----
    * This function will always return a result, even if the target loss
    is not reached - please be mindful of the values of `status` (which
    will be 0 if neither BFGS nor the backup converged, 1 for BFGS convergence,
    2 for backup convergence) and `best_loss`.
    * n_steps will be used as a max number of steps for both BFGS and
    the backup optimizer
    * return_chain=True will significantly slow down the BFGS solver,
    as it requires twice as many calls to the loss function; but not
    the backup optimizer
    """

    status = 0
    if try_bfgs:
        if return_chain:
            chain = [[*start, loss_fn(start)]]

            def callback(par):  # Function to save params & loss for each step
                chain.append([*par, loss_fn(par)])

        else:
            chain = None
            callback = None

        jac = grad(loss_fn)
        res = minimize(
            loss_fn,
            x0=start,
            method="L-BFGS-B",
            jac=jac,
            callback=callback,
        )
        best_par = res.x
        best_loss = float(res.fun)

        if return_chain:
            chain = jnp.array(chain)
        if res.success:
            status = 1
            return best_par, best_loss, status, chain

    opt_state = backup_optimizer.init(start)
    chain = []

    @jit
    def step(par, opt_state):
        loss_value, grads = value_and_grad(loss_fn)(par)
        updates, opt_state = backup_optimizer.update(grads, opt_state, par)
        par = optax.apply_updates(par, updates)
        return par, opt_state, loss_value

    # Subscript are step indices; at step i, (im1: i-1), (ip1: i+1)

    # params & loss initialization, indices shifted by 1 for consistency
    # with the iteration notation
    par_ip1 = start.copy()  # that's really par_i
    loss_i = jnp.inf  # that's really loss_im1

    for _ in range(n_steps):  # _ is i
        par_i = par_ip1  # current params = next params of previous iteration
        loss_im1 = loss_i  # previous loss = current loss of previous iteration
        par_ip1, opt_state, loss_i = step(par_i, opt_state)  # step!
        if return_chain:
            chain.append([*par_i, loss_i])  # store

        dloss = loss_im1 - loss_i
        dloss_rel = dloss / loss_i

        if loss_i < backup_target_loss:
            status = 2
            break
        if jnp.abs(dloss_rel) < backup_max_dloss:
            status = 2
            break

    best_par = par_i
    best_loss = loss_i

    return (
        best_par,
        best_loss,
        status,
        (jnp.array(chain) if return_chain else None),
    )
