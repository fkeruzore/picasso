import jax.numpy as jnp
import optax
from typing import Union, Tuple
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
    n_steps: int = 10_000,
) -> Tuple[utils.fitting.FitResults, Union[utils.fitting.FitResults, None]]:
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
    n_steps : int, optional
        Maximum number of gradient descent steps, by default 10_000,

    Returns
    -------
    picasso.utils.fitting.FitResults
        Polytropic fit results
    picasso.utils.fitting.FitResults or None
        Non-thermal fraction fit results
    """

    # Pre-format data
    phi_pts, r_pts = cutout_go.parts["phi"], cutout_go.parts["r"]
    r_edges = gas_profs.r_edges
    rho_dat, drho_dat = gas_profs.rho_g, gas_profs.rho_g
    if which_P == "tot":
        P_dat, dP_dat = gas_profs.P_tot, gas_profs.P_tot
    elif which_P == "th":
        P_dat, dP_dat = gas_profs.P_th, gas_profs.P_th
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

    res_pol = utils.fitting.optimize(
        loss_fn_pol,
        par_i,
        bounds=[(None, None), (None, None), (1.0, 2.0), (None, None)],
        backup_optimizer=backup_optimizer,
        backup_target_loss=1e-2,
        return_history=return_history,
        n_steps=n_steps,
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

    res_fnt = utils.fitting.optimize(
        loss_fn_fnt,
        par_i,
        bounds=[(None, 0.0), (None, 0.0), (0.0, None)],
        backup_optimizer=backup_optimizer,
        backup_target_loss=1e-2,
        return_history=return_history,
        n_steps=n_steps,
    )
    res_fnt.bf_model = compute_model_fnt(res_fnt.bf)

    return res_pol, res_fnt


def fit_gas_profiles_phi_prof(
    profs_go: HACCSODProfiles,
    profs_hy: HACCSODProfiles,
    which_P: str = "tot",
    fit_fnt: bool = False,
    try_bfgs: bool = True,
    backup_optimizer: optax.GradientTransformation = optax.adam(1e-3),
    return_history: bool = False,
    n_steps: int = 10_000,
) -> Tuple[utils.fitting.FitResults, Union[utils.fitting.FitResults, None]]:
    """
    Fit the polytropic gas model on a pair of halo matches, i.e. finds
    the best set of model parameters to infer the gas properties of a
    halo in a hydro simulation from its gravity-only gravitational
    potential distribution.

    Parameters
    ----------
    profs_go : HACCSODProfiles
        Radial profiles of grav-only halo.
    profs_hy : HACCSODProfiles
        Radial profiles of hydro halo.
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
    n_steps : int, optional
        Maximum number of gradient descent steps, by default 10_000,

    Returns
    -------
    picasso.utils.fitting.FitResults
        Polytropic fit results
    picasso.utils.fitting.FitResults or None
        Non-thermal fraction fit results
    """

    # Pre-format data
    phi, r = profs_go.phi_tot, profs_go.r
    rho_dat, drho_dat = profs_hy.rho_g, profs_hy.drho_g
    if which_P == "tot":
        P_dat, dP_dat = profs_hy.P_tot, profs_hy.dP_tot
    elif which_P == "th":
        P_dat, dP_dat = profs_hy.P_th, profs_hy.dP_th
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
        rho_mod_1d, P_mod_1d = polytrop.rho_P_g(
            phi, rho_0, P_0, Gamma, theta_0
        )
        return rho_mod_1d / norms["rho"], P_mod_1d / norms["P"]

    def loss_fn_pol(par):
        rho_mod, P_mod = compute_model_pol(par)
        lsq_rho = jnp.mean(((rho_dat - rho_mod) / drho_dat) ** 2)
        lsq_P = jnp.mean(((P_dat - P_mod) / dP_dat) ** 2)
        return (lsq_rho + lsq_P) / 2.0

    rho_0, P_0 = rho_dat[0] * norms["rho"], P_dat[0] * norms["P"]
    par_i = jnp.array(
        [jnp.log10(rho_0), jnp.log10(P_0), 1.2, jnp.log10(rho_0 / P_0 / 3)]
    )

    res_pol = utils.fitting.optimize(
        loss_fn_pol,
        par_i,
        bounds=[(None, None), (None, None), (1.0, 2.0), (None, None)],
        backup_optimizer=backup_optimizer,
        backup_target_loss=1e-2,
        return_history=return_history,
        n_steps=n_steps,
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
        fnt_mod_1d = nonthermal.f_nt_generic(
            r / profs_go.halo["sod_halo_R500c"], a, b, c
        )
        return fnt_mod_1d

    def loss_fn_fnt(par):
        fnt_mod = compute_model_fnt(par)
        return jnp.mean((profs_hy.f_nt - fnt_mod) ** 2)

    par_i = jnp.array([-1.0, -0.5, 0.75])

    res_fnt = utils.fitting.optimize(
        loss_fn_fnt,
        par_i,
        bounds=[(None, 0.0), (None, 0.0), (0.0, None)],
        backup_optimizer=backup_optimizer,
        backup_target_loss=1e-2,
        return_history=return_history,
        n_steps=n_steps,
    )
    res_fnt.bf_model = compute_model_fnt(res_fnt.bf)

    return res_pol, res_fnt
