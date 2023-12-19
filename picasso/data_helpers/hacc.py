import numpy as np
from numpy.typing import NDArray
from astropy.cosmology import Cosmology
from astropy.units import Unit
from typing import Union
from .. import utils

units_P_sim2obs = Unit("Msun Mpc-3 km2 s-2").to("keV cm-3")
units_P_obs2sim = 1.0 / units_P_sim2obs


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
        if all([f"v{_}" in parts for _ in "xyz"]):
            xyz_h = {x: halo[f"fof_halo_center_{x}"] for x in "xyz"}
            vxyz_h = {x: halo[f"fof_halo_com_{x}"] for x in "xyz"}
            vxyz_h_p = {
                x: _comov2prop_v(vxyz_h[x], xyz_h[x], a, adot) for x in "xyz"
            }  # Proper halo velocity, km2 s-2
            vxyz_p_p = {
                x: _comov2prop_v(parts["vx"], parts["x"], a, adot)
                for x in "xyz"
            }  # Proper particle velocities, km2 s-2
            parts["v2_proper"] = (
                (vxyz_p_p["x"] - vxyz_h_p["x"]) ** 2
                + (vxyz_p_p["y"] - vxyz_h_p["y"]) ** 2
                + (vxyz_p_p["z"] - vxyz_h_p["z"]) ** 2
            )  # Squared proper particle velocities in halo frame, km2 s-2

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
        rho_tot: NDArray,
        halo: dict,
        z: float,
        cosmo: Cosmology,
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
        self.rho_tot = rho_tot
        self.rho_g = rho_g
        self.P_th = P_th
        self.P_nt = P_nt
        if (P_nt is not None) and (P_th is not None):
            self.P_tot = P_nt + P_th
            self.f_nt = P_nt / self.P_tot
        else:
            self.P_tot, self.f_nt = None, None

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
            "sod_halo_bin_rho",
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
        rho_tot = profs_h["sod_halo_bin_rho"]
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
            rho_tot,
            halo,
            z,
            cosmo,
            rho_g=rho_g,
            P_th=P_th,
            P_nt=P_nt,
            is_hydro=is_hydro,
        )
        return inst

    @classmethod
    def from_cutout(cls, cutout: HACCCutout, r_edges: NDArray):
        _, rho_tot, drho_tot = utils.azimuthal_profile(
            cutout.parts["mass"], cutout.parts["r"], r_edges
        )
        rho_tot /= 4.0 * np.pi * (r_edges[1:] ** 3 - r_edges[:-1] ** 3) / 3.0
        drho_tot /= 4.0 * np.pi * (r_edges[1:] ** 3 - r_edges[:-1] ** 3) / 3.0

        if cutout.is_hydro:
            _, rho_g, drho_g = utils.azimuthal_profile(
                cutout.gas_parts["rho"], cutout.gas_parts["r"], r_edges
            )
            _, P_th, dP_th = utils.azimuthal_profile(
                cutout.gas_parts["P_th"], cutout.gas_parts["r"], r_edges
            )
            _, P_nt, dP_nt = utils.azimuthal_profile(
                cutout.gas_parts["P_nt"], cutout.gas_parts["r"], r_edges
            )
        else:
            rho_g, P_th, P_nt = None, None, None

        # Units: rho is h2 Msun cMpc-3, P is h2 keV ccm-3
        # -> no conversion needed

        inst = cls(
            r_edges,
            rho_tot,
            cutout.halo,
            cutout.z,
            cutout.cosmo,
            rho_g=rho_g,
            P_th=P_th,
            P_nt=P_nt,
            is_hydro=cutout.is_hydro,
        )

        inst.drho_tot = drho_tot
        if cutout.is_hydro:
            inst.drho_g = drho_g
            inst.dP_th = dP_th
            inst.dP_nt = dP_nt

        return inst
