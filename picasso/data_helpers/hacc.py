import numpy as np
import jax.numpy as jnp
from jax import Array, jit
import optax
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from astropy.cosmology import Cosmology
import h5py
import pygio
import os
from .. import utils, polytrop, nonthermal


def _comov2prop_v(v, x, a, adot):
    return a * v  # + adot * x


class HACCDataset:
    def __init__(self, cosmo: Cosmology):
        self.cosmo = cosmo


class HACCCutout(HACCDataset):
    def __init__(
        self,
        halo: dict,
        parts: dict,
        z: float,
        box_size: float,
        cosmo: Cosmology,
        is_hydro: bool = False,
    ):
        super().__init__(cosmo)
        self.is_hydro = is_hydro

        # General cosmic stuff
        self.halo = halo
        a = 1.0 / (1.0 + z)
        adot = (cosmo.H(z) * a).to("km s-1 Mpc-1").value / cosmo.h
        self.a, self.adot, self.z = a, adot, z

        # Keep only particles flagged as belonging to halo
        msk_parts = parts["fof_halo_tag"] == halo["fof_halo_tag"]
        parts = {k: v[msk_parts] for k, v in parts.items()}

        # Particles distance from center
        dxyz = {}
        for x in "xyz":
            dx = np.abs(parts[x] - halo[f"fof_halo_center_{x}"])
            dx = np.fmod(dx, box_size)
            dx[dx > box_size / 2] -= box_size
            dxyz[x] = dx
        parts["r"] = np.sqrt(
            (dxyz["x"] ** 2) + (dxyz["y"] ** 2) + (dxyz["z"] ** 2)
        )  # comoving distance from center, h-1 cMpc

        # Particle velocities
        xyz_h = {x: halo[f"fof_halo_center_{x}"] for x in "xyz"}
        vxyz_h = {x: halo[f"fof_halo_com_{x}"] for x in "xyz"}
        vxyz_h_p = {
            x: _comov2prop_v(vxyz_h[x], xyz_h[x], a, adot) for x in "xyz"
        }  # Proper halo velocity, km2 s-2
        vxyz_p_p = {
            x: _comov2prop_v(parts["vx"], parts["x"], a, adot) for x in "xyz"
        }  # Proper particle velocities, km2 s-2
        parts["v2_proper"] = (
            (vxyz_p_p["x"] - vxyz_h_p["x"]) ** 2
            + (vxyz_p_p["y"] - vxyz_h_p["y"]) ** 2
            + (vxyz_p_p["z"] - vxyz_h_p["z"]) ** 2
        )  # Squared proper particle velocities in halo frame, km2 s-2

        # Normalized potential
        parts["phi"] -= parts["phi"].min()

        # Is it a hydro sim? If yes, compute gas properties
        if is_hydro:
            parts["is_gas"] = is_gas = (  # part is ICM gas if part is not:
                (np.bitwise_and(parts["mask"], 2**2) > 0)  # CDM
                & (np.bitwise_and(parts["mask"], 2**3) == 0)  # AGN
                & (np.bitwise_and(parts["mask"], 2**4) == 0)  # star
                & (np.bitwise_and(parts["mask"], 2**5) == 0)  # wind
                & (np.bitwise_and(parts["mask"], 2**6) == 0)  # SFG
                & (np.bitwise_and(parts["mask"], 2**7) == 0)  # dead
                & (np.bitwise_and(parts["mask"], 2**8) == 0)  # accreted
                & (np.bitwise_and(parts["mask"], 2**9) == 0)  # merged
                & (np.bitwise_and(parts["mask"], 2**10) == 0)  # new AGN
            )

            # Thermal pressure
            parts["P_th"] = (
                (2.0 / 3.0) * (parts["rho"] * a**3) * (parts["uu"] * a**2)
            ) * (parts["is_gas"].astype(int))

            # Non-thermal pressure
            parts["P_nt"] = (
                (1.0 / 3.0) * (parts["rho"] * a**3) * parts["v2_proper"]
            ) * (parts["is_gas"].astype(int))

            # Total pressure
            parts["P_tot"] = parts["P_th"] + parts["P_nt"]
            self.gas_parts = {k: v[parts["is_gas"]] for k, v in parts.items()}

        self.parts = parts


class HACCCutoutPair:
    def __init__(
        self,
        halo_go: dict,
        halo_hy: dict,
        parts_go: dict,
        parts_hy: dict,
        z: float,
        box_size: float,
        cosmo: Cosmology,
    ):
        self.cutout_hy = HACCCutout(
            halo_hy, parts_hy, z, box_size, cosmo, is_hydro=True
        )
        self.cutout_go = HACCCutout(
            halo_go, parts_go, z, box_size, cosmo, is_hydro=False
        )

    def get_profiles(
        self,
        r_edges_R500: Array,
        which_P: str = "tot",
        norms: dict = {"rho": 1.0, "P": 1.0},
    ):
        h_go, h_hy = self.cutout_go.halo, self.cutout_hy.halo
        p_go = self.cutout_go.parts
        p_hy_g = self.cutout_hy.gas_parts

        _, rho_hy_1d, drho_hy_1d = utils.azimuthal_profile(
            p_hy_g["rho"] / norms["rho"],
            p_hy_g["r"] / h_hy["sod_halo_R500c"],
            r_edges_R500,
        )
        if which_P == "tot":
            P_3d = p_hy_g["P_th"] + p_hy_g["P_nt"]
        elif which_P == "th":
            P_3d = p_hy_g["P_th"]
        else:
            raise Exception(
                "`which_P` must be one of 'th' (thermal pressure)",
                "or 'tot' (thermal+kinetic),",
                f"not {which_P}",
            )
        _, P_hy_1d, dP_hy_1d = utils.azimuthal_profile(
            P_3d / norms["P"],
            p_hy_g["r"] / h_hy["sod_halo_R500c"],
            r_edges_R500,
        )
        _, fnt_hy_1d, dfnt_hy_1d = utils.azimuthal_profile(
            p_hy_g["P_nt"] / (p_hy_g["P_th"] + p_hy_g["P_nt"]),
            p_hy_g["r"] / h_hy["sod_halo_R500c"],
            r_edges_R500,
        )

        return {
            "r_R500": p_go["r"] / h_go["sod_halo_R500c"],
            "r_edges_R500": r_edges_R500,
            "phi": p_go["phi"],
            "rho": rho_hy_1d,
            "drho": drho_hy_1d,
            "P": P_hy_1d,
            "dP": dP_hy_1d,
            "fnt": fnt_hy_1d,
            "dfnt": dfnt_hy_1d,
            "norms": norms,
        }

    def fit_gas_profiles(
        self,
        r_edges_R500: Array,
        which_P: str = "tot",
        try_bfgs: bool = True,
        backup_optimizer: optax.GradientTransformation = optax.adam(1e-3),
        return_chain: bool = False,
    ):
        norms = {"rho": 1e14, "P": 1e20}
        data = self.get_profiles(r_edges_R500, which_P=which_P, norms=norms)

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

        bf_pol, bl_pol, s_pol, chain_pol = utils.optimize(
            loss_fn_pol,
            par_i,
            backup_optimizer=backup_optimizer,
            backup_target_loss=1e-2,
            return_chain=return_chain,
        )

        best_rho, best_P = compute_model_pol(bf_pol)
        results = {
            "pol_fit_output": {
                "bf": bf_pol,
                "loss": bl_pol,
                "status": s_pol,
                "chain": chain_pol,
            },
            "rho": best_rho * norms["rho"],
            "P": best_P * norms["P"],
            "which_P": which_P,
        }

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

        bf_fnt, bl_fnt, s_fnt, chain_fnt = utils.optimize(
            loss_fn_fnt,
            par_i,
            backup_optimizer=backup_optimizer,
            backup_target_loss=1e-2,
            return_chain=return_chain,
        )

        best_fnt = compute_model_fnt(bf_fnt)
        results = {
            **results,
            "fnt_fit_output": {
                "bf": bf_fnt,
                "loss": bl_fnt,
                "status": s_fnt,
                "chain": chain_fnt,
            },
            "fnt": best_fnt,
        }

        for k in ["rho", "P"]:
            data[k] *= norms[k]
            data[f"d{k}"] *= norms[k]
        return data, results

    def plot_fit_results(self, data: dict, results: dict):
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
        r_R500 = data["r_edges_R500"][:-1] + 0.5 * np.ediff1d(
            data["r_edges_R500"]
        )

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
