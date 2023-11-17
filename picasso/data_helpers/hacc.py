import numpy as np
from numpy.typing import NDArray
from astropy.cosmology import Cosmology
from .. import utils


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
        r_edges_R500: NDArray,
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
