import numpy as np
import jax.numpy as jnp
from jax import Array, jit
import optax
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Union
from .hacc import HACCCutoutPair
from .. import utils, polytrop, nonthermal


def fit_gas_profiles(
    cutout_pair: HACCCutoutPair,
    r_edges_R500: Array,
    which_P: str = "tot",
    fit_fnt: bool = False,
    try_bfgs: bool = True,
    backup_optimizer: optax.GradientTransformation = optax.adam(1e-3),
    return_history: bool = False,
) -> (dict, utils.FitResults, Union[utils.FitResults, None]):
    """
    Fit the polytropic gas model on a pair of halo matches, i.e. finds
    the best set of model parameters to infer the gas properties of a
    halo in a hydro simulation from its gravity-only gravitational
    potential distribution.

    Parameters
    ----------
    cutout_pair : HACCCutoutPair
        A pair of (hydro / gravity-only) halo particle cutouts.
    r_edges_R500 : Array
        Radial binning to be used to compute the loss function.
    which_P : str, optional
        Which pressure should be fitted with the model, either thermal
        ("th") or total (thermal+kinetic, "tot"); by default "tot"
    fit_fnt : bool, optional
        Wether or not to fit the non-thermal pressure fraction,
        by default False
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

    Returns
    -------
    dict
        Data
    picasso.utils.FitResults
        Polytropic fit results
    picasso.utils.FitResults or None
        Non-thermal fraction fit results
    """
    norms = {"rho": 1e14, "P": 1e20}
    data = cutout_pair.get_profiles(r_edges_R500, which_P=which_P, norms=norms)

    # Fit polytropic model
    def compute_model_pol(par):
        rho_0, P_0 = 10 ** par[:2]
        Gamma = par[2]
        theta_0 = 10 ** par[3]
        rho_mod_3d, P_mod_3d = polytrop.rho_P_g(
            data["phi"], rho_0, P_0, Gamma, theta_0
        )
        _, rho_mod_1d, _ = utils.azimuthal_profile(
            rho_mod_3d / norms["rho"], data["r_R500"], r_edges_R500
        )
        _, P_mod_1d, _ = utils.azimuthal_profile(
            P_mod_3d / norms["P"], data["r_R500"], r_edges_R500
        )
        return rho_mod_1d, P_mod_1d

    @jit
    def loss_fn_pol(par):
        rho_mod, P_mod = compute_model_pol(par)
        return 0.5 * (
            jnp.mean(((data["rho"] - rho_mod) / data["drho"]) ** 2)
            + jnp.mean(((data["P"] - P_mod) / data["dP"]) ** 2)
        )

    rho_0, P_0 = data["rho"][0] * norms["rho"], data["P"][0] * norms["P"]
    par_i = jnp.array(
        [jnp.log10(rho_0), jnp.log10(P_0), 1.2, jnp.log10(rho_0 / P_0 / 3)]
    )

    res_pol = utils.optimize(
        loss_fn_pol,
        par_i,
        bounds=[(None, None), (None, None), (1.0, 2.0), (None, None)],
        backup_optimizer=backup_optimizer,
        backup_target_loss=1e-2,
        return_history=return_history,
    )

    for k in ["rho", "P"]:
        data[k] *= norms[k]
        data[f"d{k}"] *= norms[k]

    if not fit_fnt:
        return data, res_pol

    # Fit non-thermal pressure fraction
    def compute_model_fnt(par):
        a, b, c = 10 ** par[0], 10 ** par[1], par[2]
        fnt_mod_3d = nonthermal.f_nt_generic(data["r_R500"], a, b, c)
        _, fnt_mod_1d, _ = utils.azimuthal_profile(
            fnt_mod_3d, data["r_R500"], r_edges_R500
        )
        return fnt_mod_1d

    @jit
    def loss_fn_fnt(par):
        fnt_mod = compute_model_fnt(par)
        return jnp.mean(((data["fnt"] - fnt_mod) / data["dfnt"]) ** 2)

    par_i = jnp.array([-1.0, -0.5, 0.75])

    res_fnt = utils.optimize(
        loss_fn_fnt,
        par_i,
        bounds=[(None, 0.0), (None, 0.0), (0.0, None)],
        backup_optimizer=backup_optimizer,
        backup_target_loss=1e-2,
        return_history=return_history,
    )

    return data, res_pol, res_fnt


def plot_fit_results(cutout_pair: HACCCutoutPair, data: dict, results: dict):
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(2, 3, height_ratios=[2, 1])
    axs_dat = [
        fig.add_subplot(gs[0]),
        fig.add_subplot(gs[1]),
        fig.add_subplot(gs[2]),
    ]
    axs_res = [
        fig.add_subplot(gs[3]),
        fig.add_subplot(gs[4]),
        fig.add_subplot(gs[5]),
    ]
    axs = axs_dat + axs_res
    r_R500 = data["r_edges_R500"][:-1] + 0.5 * np.ediff1d(data["r_edges_R500"])

    for ax_dat, ax_res, k in zip(axs_dat, axs_res, ["rho", "P", "fnt"]):
        ax_dat.errorbar(
            r_R500,
            data[k],
            xerr=None,
            yerr=data[f"d{k}"],
            fmt="o--",
            capsize=3,
            mec="w",
            zorder=1,
        )
        ax_dat.loglog(r_R500, results[k], lw=2, zorder=2)
        ax_res.errorbar(
            r_R500,
            (data[k] / data[k]) - 1,
            xerr=None,
            yerr=data[f"d{k}"] / data[k],
            fmt="o--",
            capsize=3,
            mec="w",
            zorder=1,
        )
        ax_res.semilogx(r_R500, (results[k] / data[k]) - 1, lw=2, zorder=2)

    for i, ax in enumerate(axs_dat):
        ax.set_xticklabels([])
        ax.set_ylabel(
            [
                "$\\rho \\; [h^2 M_\\odot {\\rm Mpc^{-3}}]$",
                "$P \\; [h^2 M_\\odot {\\rm Mpc^{-3} km^2 s^{-2}}]$",
                "$f_{\\rm nt} = P_{\\rm nt} / P_{\\rm tot}$",
            ][i]
        )

    for i, ax in enumerate(axs_res):
        ax.set_xlabel("$r / R_{500c}$")
        ax.set_yticks(np.arange(-0.8, 0.81, 0.2))
        ax.set_ylim(-0.7, 0.7)
        ax.set_ylabel(
            [
                "$\\Delta \\rho / \\rho$",
                "$\\Delta P / P$",
                "$\\Delta f_{\\rm nt} / f_{\\rm nt}$",
            ][i]
        )

    for ax in axs:
        ax.xaxis.set_ticks_position("both")
        ax.yaxis.set_ticks_position("both")
        ax.grid()
    fig.subplots_adjust(
        left=0.075,
        right=0.975,
        bottom=0.1,
        top=0.99,
        hspace=0.05,
        wspace=0.3,
    )
    fig.align_labels(axs)
    return fig, axs
