from jax import jit, grad, value_and_grad, Array
import jax.numpy as jnp
import optax
from scipy.optimize import minimize
from functools import partial
from typing import Callable, Union, Iterable, Tuple
import matplotlib.pyplot as plt
from astropy.cosmology import Cosmology
from astropy.constants import G

G = G.to("km2 Mpc Msun-1 s-2").value


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


class FitResults:
    """
    Wrapper class for gradient descent fit results.

    Attributes
    ----------
    bf : Array
        Best-fitting parameters
    bl : float
        Best-fit loss value
    status : int
        Fit status (0 for no success, 1 for BFGS success, 2 for backup
        optimizer success)
    bf_model : Array or None
        Best-fitting model, by default None
    history : Array or None
        Gradient descent history, by default None
    """

    def __init__(
        self, bf: Array, bl: float, status: int, bf_model=None, history=None
    ):
        self.bf = bf
        self.bl = bl
        self.status = status
        self.bf_model = bf_model
        self.history = history

    def decode_status(self):
        meanings = [
            "status=0: Gradient descent did not converge",
            "status=1: Gradient descent converged using L-BFGS-B",
            "status=2: Gradient descent converged using backup optimizer",
        ]
        return meanings[self.status]

    def __repr__(self):
        res = f"{self.decode_status()}\n"
        res += f"best fit parameters: {self.bf}\n"
        res += f"best fit loss value: {self.bl}"
        return res

    def __str__(self):
        return self.__repr__()

    def plot_history(self, param_names: list):
        """
        Plot gradient descent in parameter and loss space.

        Parameters
        ----------
        param_names : list
            Parameter names. Note that `loss` needs not be included.

        Returns
        -------
        fig, axs
        """
        assert self.history is not None, "Chain was not provided"
        param_names = [*param_names, "loss"]
        n = len(param_names)
        fig, axs = plt.subplots(n, 1)
        for i in range(n):
            axs[i].plot(self.history[:, i])
            axs[i].set_ylabel(param_names[i])
            if i < (n - 1):
                axs[i].set_xticklabels([])
        axs[-1].set_xlabel("Step number")
        return fig, axs


def optimize(
    loss_fn: Callable,
    start: Array,
    bounds: Union[None, Iterable[Tuple[float, float]]] = None,
    try_bfgs: bool = True,
    return_history: bool = False,
    n_steps: int = 10_000,
    backup_optimizer: optax.GradientTransformation = optax.adam(1e-3),
    backup_target_loss: float = 1e-8,
    backup_max_dloss: float = 1e-8,
) -> FitResults:
    """Optimize a loss function and returns the gradient descent in
    parameter and loss space.

    Parameters
    ----------
    loss_fn : Callable
        The loss function
    start : Array
        Parameter space starting point for the gradient descent
    bounds : sequence of 2-tuples, optional
        Bounds (min, max) for each parameter (only works for scipy's
        `L-BFGS-B` minimizer), by default None
    try_bfgs : bool, optional
        If True, first tries minimizing the loss function using scipy
        optimize with methof `L-BFGS-B`, using jax's gradients, by
        default True
    backup_optimizer : optax.GradientTransformation, optional
        An `optax` minimizer to use as backup if BFGS doesn't converge
        (or if `try_bfgs=False`), by default optax.adam(1e-3)
    return_history : bool, optional
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
    FitResults

    Notes
    -----
    * This function will always return a result, even if the target loss
    is not reached - please be mindful of the values of `status` (which
    will be 0 if neither BFGS nor the backup converged, 1 for BFGS convergence,
    2 for backup convergence) and `best_loss`.
    * n_steps will be used as a max number of steps for both BFGS and
    the backup optimizer
    * return_history=True will significantly slow down the BFGS solver,
    as it requires twice as many calls to the loss function; but not
    the backup optimizer
    """

    status = 0
    if try_bfgs:
        if return_history:
            history = [[*start, loss_fn(start)]]

            def callback(par):  # Function to save params & loss for each step
                history.append([*par, loss_fn(par)])

        else:
            history = None
            callback = None

        jac = grad(loss_fn)
        res = minimize(
            loss_fn,
            x0=start,
            method="L-BFGS-B",
            jac=jac,
            callback=callback,
            bounds=bounds,
        )
        best_par = res.x
        best_loss = float(res.fun)

        if return_history:
            history = jnp.array(history)
        if res.success:
            status = 1
            results = FitResults(
                best_par,
                best_loss,
                status,
                history=(jnp.array(history) if return_history else None),
            )
            return results

    opt_state = backup_optimizer.init(start)
    history = []

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
        if return_history:
            history.append([*par_i, loss_i])  # store

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

    results = FitResults(
        best_par,
        best_loss,
        status,
        history=(jnp.array(history) if return_history else None),
    )
    return results


class NFW:
    def __init__(
        self,
        MDelta: float,
        cDelta: float,
        Delta: str,
        z: float,
        cosmo: Cosmology,
    ):
        self.MDelta = MDelta
        self.cDelta = cDelta
        if Delta == "200c":
            mean_rho = 200 * cosmo.critical_density(z).to("Msun Mpc-3").value
        elif Delta == "500c":
            mean_rho = 500 * cosmo.critical_density(z).to("Msun Mpc-3").value
        else:
            raise ValueError(
                f"{Delta=} not supported yet, must be either '200c' or '500c'"
            )
        self.RDelta = (3 * MDelta / (4 * jnp.pi * mean_rho)) ** (1 / 3)
        self.Rs = self.RDelta / cDelta
        rho0_denum = 4 * jnp.pi * self.Rs**3
        rho0_denum *= jnp.log(1 + cDelta) - cDelta / (1 + cDelta)
        self.rho0 = MDelta / rho0_denum

    def density(self, r: Array) -> Array:
        """NFW density profile

        Parameters
        ----------
        r : Array [Mpc]
            Radius

        Returns
        -------
        Array [Msun Mpc-3]
            Density at radius `r`
        """
        return self.rho0 / (r / self.Rs * (1 + r / self.Rs) ** 2)

    def enclosed_mass(self, r: Array) -> Array:
        """Enclosed mass profile

        Parameters
        ----------
        r : Array [Mpc]
            Radius

        Returns
        -------
        Array [Msun]
            Enclosed mass at radius `r`
        """
        prefact = 4 * jnp.pi * self.rho0 * self.Rs**3
        return prefact * (jnp.log(1 + r / self.Rs) - r / (r + self.Rs))

    def potential(self, r: Array) -> Array:
        """Potential profile

        Parameters
        ----------
        r : Array [Mpc]
            Radius

        Returns
        -------
        Array [km2 s-2]
            Potential at radius `r`
        """
        # G = G.to("km2 Mpc Msun-1 s-2").value
        prefact = -4 * jnp.pi * G * self.rho0 * self.Rs**3
        return prefact * jnp.log(1 + r / self.Rs) / r
