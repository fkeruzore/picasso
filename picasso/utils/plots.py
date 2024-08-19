import jax.numpy as jnp
from jax import jit, Array
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from astropy.cosmology import FlatLambdaCDM, Cosmology
from astropy.constants import G

G = G.to("km2 Mpc Msun-1 s-2").value

from .. import nonthermal, polytrop


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


def plot_impact_model_params(
    n_curves: int, cmapname: str, force_central_color=None
):
    param_names = [
        "rho_0",
        "P_0",
        "Gamma_0",
        "c_gamma",
        "theta_0",
        "a",
        "b",
        "c",
    ]
    param_latex = [
        "$\\log_{10} \\rho_0$",
        "$\\log_{10} P_0$",
        "$\\Gamma_0$",
        "$c_\\gamma$",
        "$\\theta_0$",
        "$\\log_{10} A_{\\rm nt}$",
        "$\\log_{10} B_{\\rm nt}$",
        "$C_{\\rm nt}$",
    ]
    params_fix = {
        "rho_0": 3.0,
        "P_0": 2.5,
        "Gamma_0": 1.15,
        "c_gamma": 0.0,
        # "theta_0": -6.9,
        "theta_0": 1.0,
        "a": -1.5,
        "b": -0.75,
        "c": 1.0,
    }
    params_var = {
        "rho_0": np.linspace(2.5, 3.5, n_curves),
        "P_0": np.linspace(2.0, 4.0, n_curves),
        "Gamma_0": np.linspace(1.05, 1.25, n_curves),
        "c_gamma": np.linspace(-1.0, 1.0, n_curves),
        "theta_0": np.linspace(0.0, 2.0, n_curves),
        "a": np.linspace(-2.0, -1.0, n_curves),
        "b": np.linspace(-1.0, -0.5, n_curves),
        "c": np.linspace(0.25, 1.75, n_curves),
    }
    impacts = {
        "rho_0": [0],
        "P_0": [1, 3],
        "Gamma_0": [0, 1, 3],
        "c_gamma": [0, 1, 3],
        "theta_0": [0, 1, 3],
        "a": [2, 3],
        "b": [2, 3],
        "c": [2, 3],
    }
    n_p = len(param_names)

    nfw = NFW(3.5e14, 5.0, "500c", 0.0, FlatLambdaCDM(70.0, 0.3))
    r_R500c = jnp.logspace(-1.5, 0.5, 128)
    r = nfw.RDelta * r_R500c
    phi = nfw.potential(r)
    phi -= phi.min()

    fig = plt.figure(figsize=(13, 7))
    gs = mpl.gridspec.GridSpec(
        5, n_p, height_ratios=[0.05, 0.975, 0.975, 0.975, 0.975]
    )
    cb_axs = [fig.add_subplot(gs[i]) for i in range(n_p)]
    axs = np.array(
        [
            [fig.add_subplot(gs[i + 1 * n_p]) for i in range(n_p)],
            [fig.add_subplot(gs[i + 2 * n_p]) for i in range(n_p)],
            [fig.add_subplot(gs[i + 3 * n_p]) for i in range(n_p)],
            [fig.add_subplot(gs[i + 4 * n_p]) for i in range(n_p)],
        ]
    )

    par_vec_fix = params_fix.copy()
    for p in ["rho_0", "P_0", "a", "b"]:
        par_vec_fix[p] = 10 ** par_vec_fix[p]
    par_vec_fix["theta_0"] *= 1e-7

    @jit
    def compute_props(rho_0, P_0, Gamma_0, c_gamma, theta_0, a, b, c):
        rho, P_tot = polytrop.rho_P_g(
            phi, r_R500c, rho_0, P_0, Gamma_0, c_gamma, theta_0
        )
        f_nt = nonthermal.f_nt_generic(r_R500c / 2, a, b, c)
        return rho, P_tot, f_nt, P_tot * (1 - f_nt)

    fixed_props = compute_props(**par_vec_fix)

    for i_par_var, p_var in enumerate(param_names):
        norm = mpl.colors.Normalize(
            vmin=params_var[p_var].min(), vmax=params_var[p_var].max()
        )
        cmap = mpl.cm.ScalarMappable(norm=norm, cmap=plt.colormaps[cmapname])
        cmap.set_array([])

        par_vecs = {p: jnp.ones(n_curves) * params_fix[p] for p in param_names}
        par_vecs[p_var] = params_var[p_var]
        for p in ["rho_0", "P_0", "a", "b"]:
            par_vecs[p] = 10 ** par_vecs[p]
        par_vecs["theta_0"] = par_vecs["theta_0"] * 1e-7
        for i_vec in range(n_curves):
            props = compute_props(
                **{p: par_vecs[p][i_vec] for p in param_names}
            )
            for i_prop, (prop, ax) in enumerate(zip(props, axs[:, i_par_var])):
                if i_prop in impacts[p_var]:
                    ax.loglog(
                        r_R500c,
                        prop,
                        color=cmap.to_rgba(params_var[p_var][i_vec]),
                    )
                else:
                    if i_vec == 0:
                        ax.loglog(
                            r_R500c,
                            fixed_props[i_prop],
                            "--",
                            color=(
                                force_central_color
                                if force_central_color is not None
                                else plt.get_cmap(cmapname)(0.5)
                            ),
                        )

        cb = fig.colorbar(
            cmap,
            cax=cb_axs[i_par_var],
            orientation="horizontal",
            location="top",
        )  # , ticks=params_var[p_var][::2])
        cb_axs[i_par_var].set_xticks(params_var[p_var], minor=True)
        cb_axs[i_par_var].set_xticks(
            params_var[p_var][1::2],
            np.round(params_var[p_var][1::2], 2),
            fontsize=9,
        )
        cb.set_label(param_latex[i_par_var])

    for i_row, row in enumerate(axs):
        col_ref = [0, 1, 4, 1]
        ylim = jnp.array([ax.get_ylim() for ax in row]).flatten()
        for ax in row:
            if i_row == 2:
                ax.set_ylim(ylim.min(), ylim.max())
            else:
                ax.set_ylim(*row[col_ref[i_row]].get_ylim())
            if i_row == 3:
                ax.set_xlabel("$r / R_{500c}$")
            else:
                ax.set_xticklabels([])
            if i_row in [2, 3]:
                ax.text(
                    2.0,
                    0.1,
                    "$\\uparrow$",
                    fontsize=12,
                    ha="center",
                    va="center",
                    transform=ax.get_xaxis_transform(),
                )
        for ax in row[1:]:
            ax.set_yticklabels([])
    # for ax in axs.flatten():
    #     ax.xaxis.set_ticks_position("both")
    #     ax.yaxis.set_ticks_position("both")
    #     ax.grid(":", zorder=-5)

    axs[0, 0].set_ylabel("$\\rho_{\\rm g} / 500 \\rho_{\\rm crit}$")
    axs[1, 0].set_ylabel("$P_{\\rm tot} / P_{500}$")
    axs[2, 0].set_ylabel("$f_{\\rm nt}$")
    axs[3, 0].set_ylabel("$P_{\\rm th} / P_{500}$")
    fig.align_labels()
    fig.subplots_adjust(
        left=0.05, right=0.95, top=0.95, bottom=0.07, hspace=0.04, wspace=0.04
    )
    return fig


def plot_Gamma_r(Gamma_0: Array, c_gamma: Array, cmapname: str):
    compute_gamma = jit(polytrop.Gamma_r)

    r = jnp.logspace(-1.5, 0.5, 101)

    norm = mpl.colors.Normalize(vmin=c_gamma.min(), vmax=c_gamma.max())
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=plt.colormaps[cmapname])
    cmap.set_array([])

    fig, axs = plt.subplots(len(Gamma_0), 1)
    fig.subplots_adjust(hspace=0.05, top=0.95)
    for i, c in enumerate(c_gamma):
        color = cmap.to_rgba(c)
        for ax, g in zip(axs, Gamma_0):
            ax.semilogx(r, compute_gamma(r, g, c), color=color)
            ax.set_ylabel(
                f"$\\Gamma(r \\, | \\, \\Gamma_0 = {g:.3f}, c_\\gamma)$"
            )
    axs[-1].set_xlabel("$r / R_{500c}$")
    for ax in axs[:-1]:
        ax.set_xticklabels([])

    cb = fig.colorbar(cmap, ax=axs)
    cb.set_ticks(jnp.linspace(c_gamma.min(), c_gamma.max(), 5))
    cb.set_label("$c_\\gamma$")

    # for ax in axs:
    #     ax.xaxis.set_ticks_position("both")
    #     ax.yaxis.set_ticks_position("both")
    fig.align_labels()
    return fig
