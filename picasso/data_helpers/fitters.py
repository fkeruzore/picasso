import jax.numpy as jnp
import optax
from typing import Union
from .hacc import HACCCutout, HACCSODProfiles
from .. import utils, polytrop, nonthermal


def fit_gas_profiles(
    cutout_go: HACCCutout,
    gas_profs: HACCSODProfiles,
    which_P: str = "tot",
    fit_fnt: bool = False,
    try_bfgs: bool = True,
    backup_optimizer: optax.GradientTransformation = optax.adam(1e-3),
    return_history: bool = False,
) -> (utils.FitResults, Union[utils.FitResults, None]):
    """
    Fit the polytropic gas model on a pair of halo matches, i.e. finds
    the best set of model parameters to infer the gas properties of a
    halo in a hydro simulation from its gravity-only gravitational
    potential distribution.

    Parameters
    ----------
    cutout_go : HACCCutout
        Cutout of grav-only halo particles.
    gas_profs : HACCSODProfiles
        Radial profiles of gas properties.
    which_P : str, optional
        Which pressure should be fitted with the polytropic model, one
        of 'th' (thermal pressure) or 'tot' (thermal+kinetic),
        by default 'tot'
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
    picasso.utils.FitResults
        Polytropic fit results
    picasso.utils.FitResults or None
        Non-thermal fraction fit results
    """

    # Pre-format data
    phi_pts, r_pts = cutout_go.parts["phi"], cutout_go.parts["r"]
    r_edges = gas_profs.r_edges
    rho_dat, drho_dat = gas_profs.rho_g, gas_profs.drho_g
    if which_P == "tot":
        P_dat, dP_dat = gas_profs.P_tot, gas_profs.dP_tot
    elif which_P == "th":
        P_dat, dP_dat = gas_profs.P_th, gas_profs.dP_th
    else:
        raise Exception(
            "`which_P` must be one of 'th' (thermal pressure) or 'tot'"
            + f"(thermal+kinetic), not '{which_P}'"
        )

    norms = {"rho": jnp.nanmean(rho_dat), "P": jnp.nanmean(P_dat)}
    rho_dat /= norms["rho"]
    drho_dat /= norms["rho"]
    P_dat /= norms["P"]
    dP_dat /= norms["P"]

    # Fit polytropic model
    def compute_model_pol(par):
        rho_0, P_0 = 10 ** par[:2]
        Gamma = par[2]
        theta_0 = 10 ** par[3]
        rho_mod_3d, P_mod_3d = polytrop.rho_P_g(
            phi_pts, rho_0, P_0, Gamma, theta_0
        )
        _, rho_mod_1d, _ = utils.azimuthal_profile(
            rho_mod_3d / norms["rho"], r_pts, r_edges
        )
        _, P_mod_1d, _ = utils.azimuthal_profile(
            P_mod_3d / norms["P"], r_pts, r_edges
        )
        return rho_mod_1d, P_mod_1d

    def loss_fn_pol(par):
        rho_mod, P_mod = compute_model_pol(par)
        lsq_rho = jnp.mean(((rho_dat - rho_mod) / drho_dat) ** 2)
        lsq_P = jnp.mean(((P_dat - P_mod) / dP_dat) ** 2)
        return (lsq_rho + lsq_P) / 2.0

    rho_0, P_0 = rho_dat[0] * norms["rho"], P_dat[0] * norms["P"]
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
    bf_model = list(compute_model_pol(res_pol.bf))
    bf_model[0] *= norms["rho"]
    bf_model[1] *= norms["P"]
    res_pol.bf_model = bf_model

    if not fit_fnt:
        return res_pol

    # Fit non-thermal pressure fraction
    def compute_model_fnt(par):
        a, b, c = 10 ** par[0], 10 ** par[1], par[2]
        fnt_mod_3d = nonthermal.f_nt_generic(
            r_pts / cutout_go.halo["sod_halo_R500c"], a, b, c
        )
        _, fnt_mod_1d, _ = utils.azimuthal_profile(fnt_mod_3d, r_pts, r_edges)
        return fnt_mod_1d

    def loss_fn_fnt(par):
        fnt_mod = compute_model_fnt(par)
        return jnp.mean((gas_profs.f_nt - fnt_mod) ** 2)

    par_i = jnp.array([-1.0, -0.5, 0.75])

    res_fnt = utils.optimize(
        loss_fn_fnt,
        par_i,
        bounds=[(None, 0.0), (None, 0.0), (0.0, None)],
        backup_optimizer=backup_optimizer,
        backup_target_loss=1e-2,
        return_history=return_history,
    )
    res_fnt.bf_model = compute_model_fnt(res_fnt.bf)

    return res_pol, res_fnt


# def plot_fit_results(cutout_pair: HACCCutoutPair, data: dict, results: dict):
#     fig = plt.figure(figsize=(12, 6))
#     gs = GridSpec(2, 3, height_ratios=[2, 1])
#     axs_dat = [
#         fig.add_subplot(gs[0]),
#         fig.add_subplot(gs[1]),
#         fig.add_subplot(gs[2]),
#     ]
#     axs_res = [
#         fig.add_subplot(gs[3]),
#         fig.add_subplot(gs[4]),
#         fig.add_subplot(gs[5]),
#     ]
#     axs = axs_dat + axs_res
#     r_R500 = data["r_edges_R500"][:-1] + np.ediff1d(data["r_edges_R500"]) / 2
#
#     for ax_dat, ax_res, k in zip(axs_dat, axs_res, ["rho", "P", "fnt"]):
#         ax_dat.errorbar(
#             r_R500,
#             data[k],
#             xerr=None,
#             yerr=data[f"d{k}"],
#             fmt="o--",
#             capsize=3,
#             mec="w",
#             zorder=1,
#         )
#         ax_dat.loglog(r_R500, results[k], lw=2, zorder=2)
#         ax_res.errorbar(
#             r_R500,
#             (data[k] / data[k]) - 1,
#             xerr=None,
#             yerr=data[f"d{k}"] / data[k],
#             fmt="o--",
#             capsize=3,
#             mec="w",
#             zorder=1,
#         )
#         ax_res.semilogx(r_R500, (results[k] / data[k]) - 1, lw=2, zorder=2)
#
#     for i, ax in enumerate(axs_dat):
#         ax.set_xticklabels([])
#         ax.set_ylabel(
#             [
#                 "$\\rho \\; [h^2 M_\\odot {\\rm Mpc^{-3}}]$",
#                 "$P \\; [h^2 M_\\odot {\\rm Mpc^{-3} km^2 s^{-2}}]$",
#                 "$f_{\\rm nt} = P_{\\rm nt} / P_{\\rm tot}$",
#             ][i]
#         )
#
#     for i, ax in enumerate(axs_res):
#         ax.set_xlabel("$r / R_{500c}$")
#         ax.set_yticks(np.arange(-0.8, 0.81, 0.2))
#         ax.set_ylim(-0.7, 0.7)
#         ax.set_ylabel(
#             [
#                 "$\\Delta \\rho / \\rho$",
#                 "$\\Delta P / P$",
#                 "$\\Delta f_{\\rm nt} / f_{\\rm nt}$",
#             ][i]
#         )
#
#     for ax in axs:
#         ax.xaxis.set_ticks_position("both")
#         ax.yaxis.set_ticks_position("both")
#         ax.grid()
#     fig.subplots_adjust(
#         left=0.075,
#         right=0.975,
#         bottom=0.1,
#         top=0.99,
#         hspace=0.05,
#         wspace=0.3,
#     )
#     fig.align_labels(axs)
#     return fig, axs
