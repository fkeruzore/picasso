import numpy as np
import jax.numpy as jnp
from numpy.typing import NDArray
from astropy.cosmology import Cosmology
from astropy.units import Unit
from typing import Union
from .. import utils

units_P_sim2obs = Unit("Msun Mpc-3 km2 s-2").to("keV cm-3")
units_P_obs2sim = 1.0 / units_P_sim2obs


def _comov2prop_v(v, x, a, adot):
    return a * v + adot * x


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
        if all([f"v{_}" in parts for _ in "xyz"]) and is_hydro:
            _x = {x: parts[x] - halo[f"sod_halo_com_{x}_gas"] for x in "xyz"}
            _v = {
                x: parts[f"v{x}"] - halo[f"sod_halo_com_v{x}_gas"]
                for x in "xyz"
            }
            parts["v2_proper"] = (
                _comov2prop_v(_v["x"], _x["x"], a, adot) ** 2
                + _comov2prop_v(_v["y"], _x["y"], a, adot) ** 2
                + _comov2prop_v(_v["z"], _x["z"], a, adot) ** 2
            )

        # Normalized potential
        if "phi" in parts.keys():
            parts["phi"] -= parts["phi"].min()

        # Is it a hydro sim? If yes, compute gas properties
        if is_hydro:
            parts["is_gas"] = (  # part is ICM gas if part is not:
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


class HACCSODProfiles(HACCDataset):
    def __init__(
        self,
        r_edges: NDArray,
        halo: dict,
        z: float,
        cosmo: Cosmology,
        rho_tot: Union[NDArray, None] = None,
        phi_tot: Union[NDArray, None] = None,
        rho_g: Union[NDArray, None] = None,
        P_th: Union[NDArray, None] = None,
        P_nt: Union[NDArray, None] = None,
        is_hydro: bool = False,
    ):
        super().__init__(cosmo)
        self.is_hydro = is_hydro
        self.halo = halo
        self.z = z

        self.r_edges = r_edges
        self.r = (r_edges[:-1] + r_edges[1:]) / 2.0
        self.rho_tot = rho_tot
        self.phi_tot = phi_tot
        self.rho_g = rho_g
        self.P_th = P_th
        self.P_nt = P_nt
        if (P_nt is not None) and (P_th is not None):
            self.P_tot = P_nt + P_th
            self.f_nt = P_nt / self.P_tot
        else:
            self.P_tot, self.f_nt = None, None

        possible_profs = [
            "rho_tot",
            "phi_tot",
            "rho_g",
            "P_th",
            "P_nt",
            "P_tot",
            "f_nt",
        ]
        possible_profs = [*possible_profs, *[f"d{p}" for p in possible_profs]]
        self.profs = [
            p
            for p in possible_profs
            if (hasattr(self, p) and (getattr(self, p) is not None))
        ]

    @classmethod
    def from_sodpropertybins(
        cls,
        halo: dict,
        profs: dict,
        z: float,
        cosmo: Cosmology,
        is_hydro: bool = False,
        r_min: float = 0.0,
        r_max: float = np.inf,
    ):
        keys = [
            "fof_halo_bin_tag",
            "sod_halo_bin_radius",
            "sod_halo_bin_mass",
        ]
        if is_hydro:
            keys += [
                "sod_halo_bin_gas_fraction",
                "sod_halo_bin_gas_pthermal",
                "sod_halo_bin_gas_pkinetic",
            ]

        for key in keys:
            assert key in profs.keys(), f"`profs` missing required key {key}"

        msk_h = profs["fof_halo_bin_tag"] == halo["fof_halo_tag"]
        profs_h = {k: v[msk_h] for k, v in profs.items()}

        r_edges = np.concatenate(([0.0], profs_h["sod_halo_bin_radius"]))
        rho_tot = profs_h["sod_halo_bin_mass"] / (
            4.0 * np.pi * (r_edges[1:] ** 3 - r_edges[:-1] ** 3) / 3.0
        )
        if is_hydro:
            rho_g = profs_h["sod_halo_bin_gas_fraction"] * rho_tot
            P_th = profs_h["sod_halo_bin_gas_pthermal"]
            P_nt = profs_h["sod_halo_bin_gas_pkinetic"]
        else:
            rho_g, P_th, P_nt = None, None, None

        # Units: rho is h2 Msun cMpc-3, P is h2 keV ccm-3
        # -> P conversion needed
        if is_hydro:
            P_th *= units_P_obs2sim
            P_nt *= units_P_obs2sim

        # Cut radii
        r_ok = (r_edges[:-1] >= r_min) & (r_edges[1:] <= r_max)
        r_edges = r_edges[(r_edges >= r_min) & (r_edges <= r_max)]
        rho_tot = rho_tot[r_ok]
        if is_hydro:
            rho_g = rho_g[r_ok]
            P_th = P_th[r_ok]
            P_nt = P_nt[r_ok]

        inst = cls(
            r_edges,
            halo,
            z,
            cosmo,
            rho_tot=rho_tot,
            rho_g=rho_g,
            P_th=P_th,
            P_nt=P_nt,
            is_hydro=is_hydro,
        )

        inst.drho_tot = rho_tot
        if is_hydro:
            inst.drho_g = inst.rho_g
            inst.dP_th = inst.P_th
            inst.dP_nt = inst.P_nt
            inst.dP_tot = inst.P_tot

        return inst

    @classmethod
    def from_cutout(cls, cutout: HACCCutout, r_edges: NDArray):
        _, m_tot, _ = utils.azimuthal_profile(
            cutout.parts["mass"],
            cutout.parts["r"],
            r_edges,
            stats=(jnp.nansum, jnp.nanstd),
        )
        shell_vols = 4.0 * np.pi * (r_edges[1:] ** 3 - r_edges[:-1] ** 3) / 3.0
        rho_tot = m_tot / shell_vols

        _, phi_tot, _ = utils.azimuthal_profile(
            cutout.parts["phi"],
            cutout.parts["r"],
            r_edges,
            stats=(jnp.nanmean, jnp.nanstd),
        )

        if cutout.is_hydro:
            # Gas density = gas mass in shells, divided by shell volumes
            _, m_g, _ = utils.azimuthal_profile(
                cutout.gas_parts["mass"],
                cutout.gas_parts["r"],
                r_edges,
                stats=(jnp.nansum, jnp.nanstd),
            )
            rho_g = m_g / shell_vols

            # Gas thermal pressure
            _, kbT_g, _ = utils.azimuthal_profile(
                cutout.a * 2.0 * cutout.gas_parts["uu"] / 3.0,
                cutout.gas_parts["r"],
                r_edges,
            )
            P_th = rho_g * kbT_g

            # Gas non-thermal pressure
            _, v2_g, _ = utils.azimuthal_profile(
                cutout.gas_parts["v2_proper"] / cutout.a,
                cutout.gas_parts["r"],
                r_edges,
            )
            P_nt = rho_g * v2_g / 3.0
        else:
            rho_g, P_th, P_nt = None, None, None

        # Units: rho is h2 Msun cMpc-3, P is h2 keV ccm-3
        # -> no conversion needed

        inst = cls(
            r_edges,
            cutout.halo,
            cutout.z,
            cutout.cosmo,
            rho_tot=rho_tot,
            phi_tot=phi_tot,
            rho_g=rho_g,
            P_th=P_th,
            P_nt=P_nt,
            is_hydro=cutout.is_hydro,
        )

        return inst

    def rebin(self, r_edges):
        old_r_edges = self.r_edges
        old_r = (old_r_edges[:-1] + old_r_edges[1:]) / 2.0
        new_r_edges = r_edges
        new_r = (new_r_edges[:-1] + new_r_edges[1:]) / 2.0

        def interp_plaw(x, xp, fp, left=None, right=None):
            lx, lxp, lfp = np.log(x), np.log(xp), np.log(fp)
            lf = np.interp(lx, lxp, lfp, left=left, right=right)
            return np.exp(lf)

        self.r_edges = new_r_edges
        self.r = new_r

        for name in self.profs:
            setattr(self, name, interp_plaw(new_r, old_r, getattr(self, name)))
