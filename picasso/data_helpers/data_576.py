import jax.numpy as jnp
import numpy as np
import h5py
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from .. import utils

cosmo = FlatLambdaCDM(67.66, 0.30964, Ob0=0.04897)


def read_hdf5(file_name, jax=False):
    arr_func = jnp.array if jax else np.array
    with h5py.File(file_name, "r") as f:
        grids = {k: arr_func(v) for k, v in f.items()}
    return grids


path = "/Users/fkeruzore/Data/HACC/576/LIGHTCONE"


def get_sim_products(step):
    halos_ad = read_hdf5(f"{path}/AD/halos_{step}.h5")
    halos_go = read_hdf5(f"{path}/GO/halos_{step}.h5")
    pairs = Table.read(f"{path}/PAIRS/halos_{step}.h5", format="hdf5")
    parts_ad = read_hdf5(f"{path}/AD/sodbighaloparticles_{step}.h5")
    parts_go = read_hdf5(f"{path}/GO/sodbighaloparticles_{step}.h5")
    return [
        halos_go,
        halos_ad,
        pairs,
        parts_go,
        parts_ad,
    ]


def get_sim_products_1halo(
    i_pair, pairs, halos_go, halos_ad, parts_go, parts_ad, verbose=False
):
    # Halo properties
    pair = pairs[i_pair]
    i_go = np.where((halos_go["fof_halo_tag"] == pair["GO"]))[0][0]
    i_ad = np.where((halos_ad["fof_halo_tag"] == pair["AD"]))[0][0]
    h_go = {k: v[i_go] for k, v in halos_go.items()}
    h_ad = {k: v[i_ad] for k, v in halos_ad.items()}

    a = h_go["fof_halo_center_a"]
    z = 1.0 / a - 1.0
    adot = (cosmo.H(z) * a).to("km s-1 Mpc-1").value / cosmo.h

    if verbose:
        print("====   HALO PROPERTIES  ====")
        print(f"{z=:.3f} ({a=:.3f})")
        print("-------   GRAV-ONLY  -------")
        print(f"M500 = {h_go['sod_halo_M500c']:.2e} h-1 Msun")
        print(f"R500 = {h_go['sod_halo_R500c']:.3f} h-1 cMpc")
        print("-------   ADIABATIC  -------")
        print(f"M500 = {h_ad['sod_halo_M500c']:.2e} h-1 Msun")
        print(f"R500 = {h_ad['sod_halo_R500c']:.3f} h-1 cMpc")

    # Gravity-only particles
    wp_go = parts_go["fof_halo_tag"] == pair["GO"]
    p_go = {k: v[wp_go] for k, v in parts_go.items()}

    p_go["r"] = np.sqrt(
        np.fmod((576.0 + p_go["x"] - h_go["fof_halo_center_x"]), 576.0) ** 2
        + np.fmod((576.0 + p_go["y"] - h_go["fof_halo_center_y"]), 576.0) ** 2
        + np.fmod((576.0 + p_go["z"] - h_go["fof_halo_center_z"]), 576.0) ** 2
    )  # comoving distance from center, h-1 cMpc
    p_go["v2"] = (
        (
            (a * p_go["vx"] + adot * p_go["x"])
            - (a * h_go["fof_halo_com_vx"] + adot * h_go["fof_halo_center_x"])
        )
        ** 2
        + (
            (a * p_go["vy"] + adot * p_go["y"])
            - (a * h_go["fof_halo_com_vy"] + adot * h_go["fof_halo_center_y"])
        )
        ** 2
        + (
            (a * p_go["vz"] + adot * p_go["z"])
            - (a * h_go["fof_halo_com_vz"] + adot * h_go["fof_halo_center_z"])
        )
        ** 2
    )  # proper squared velocity, km2 s-2
    p_go["phi"] -= p_go["phi"].min()
    p_go["mass"] = np.ones_like(p_go["x"]) * 1.3428e09  # h-1 Msun

    # Adiabatic particles
    wp_ad = parts_ad["fof_halo_tag"] == pair["AD"]
    p_ad = {k: v[wp_ad] for k, v in parts_ad.items()}
    p_ad["r"] = np.sqrt(
        np.fmod((576.0 + p_ad["x"] - h_ad["fof_halo_center_x"]), 576.0) ** 2
        + np.fmod((576.0 + p_ad["y"] - h_ad["fof_halo_center_y"]), 576.0) ** 2
        + np.fmod((576.0 + p_ad["z"] - h_ad["fof_halo_center_z"]), 576.0) ** 2
    )  # comoving distance from center, h-1 cMpc
    p_ad["v2"] = (
        (
            (a * p_ad["vx"] + adot * p_ad["x"])
            - (a * h_ad["fof_halo_com_vx"] + adot * h_ad["fof_halo_center_x"])
        )
        ** 2
        + (
            (a * p_ad["vy"] + adot * p_ad["y"])
            - (a * h_ad["fof_halo_com_vy"] + adot * h_ad["fof_halo_center_y"])
        )
        ** 2
        + (
            (a * p_ad["vz"] + adot * p_ad["z"])
            - (a * h_ad["fof_halo_com_vx"] + adot * h_ad["fof_halo_center_z"])
        )
        ** 2
    )  # proper squared velocity, km2 s-2
    p_ad["is_gas"] = (  # part is gas if part:
        (np.bitwise_and(p_ad["mask"], 2**2) > 0)  # is not CDM
        & (np.bitwise_and(p_ad["mask"], 2**3) == 0)  # is not AGN
        & (np.bitwise_and(p_ad["mask"], 2**4) == 0)  # is not star
        & (np.bitwise_and(p_ad["mask"], 2**5) == 0)  # is not wind
    )
    p_ad["P_th"] = (2 / 3) * p_ad["rho"] * p_ad["uu"]
    p_ad["P_nt"] = (1 / 3) * p_ad["rho"] * p_ad["v2"]
    for p in ["P_th", "P_nt"]:
        p_ad[p] = np.where(p_ad["is_gas"], p_ad[p], np.nan)

    return [h_go, h_ad, p_go, p_ad]


def get_data_vector(
    h_go, h_ad, p_go, p_ad, r_edges_R500, norms={"rho": 1.0, "P": 1.0}
):
    p_ad_g = {k: v[p_ad["is_gas"]] for k, v in p_ad.items()}

    _, rho_ad_1d, drho_ad_1d = utils.azimuthal_profile(
        p_ad_g["rho"] / norms["rho"],
        p_ad_g["r"] / h_ad["sod_halo_R500c"],
        r_edges_R500,
    )
    _, P_ad_1d, dP_ad_1d = utils.azimuthal_profile(
        (p_ad_g["P_th"] + p_ad_g["P_nt"]) / norms["P"],
        p_ad_g["r"] / h_ad["sod_halo_R500c"],
        r_edges_R500,
    )

    return {
        "r": p_go["r"] / h_go["sod_halo_R500c"],
        "phi": p_go["phi"],
        "rho": rho_ad_1d,
        "drho": drho_ad_1d,
        "P": P_ad_1d,
        "dP": dP_ad_1d,
    }
