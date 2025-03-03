import jax.numpy as jnp
from jax import Array
from typing import Tuple, Dict


def compute_halo_shapes(
    halos: dict, use_sod: bool = True, use_reduced: bool = True
) -> Tuple[Array, Array, Array, Array, Array, Array]:
    """
    Compute shape parameters (semi-axes, ellipticity, prolateness,
    triaxiality) from inertia tensor eigenvectors as stored in a
    HACC haloproperties catalog.

    Parameters
    ----------
    halos : dict
        HACC haloproperties catalog
    use_sod : bool, optional
        Use the SOD inertia tensor (as opposed to the FoF),
        by default True
    use_reduced : bool, optional
        Use the reduced inertia tensor, by default True

    Returns
    -------
    Array
        Semi-major axis
    Array
        Semi-intermediate axis
    Array
        Semi-minor axis
    Array
        Ellipticity
    Array
        Prolateness
    Array
        Triaxiality
    """
    sod = "sod" if use_sod else "fof"
    red = "R" if use_reduced else "S"
    l1 = jnp.sum(
        jnp.array([halos[f"{sod}_halo_eig{red}1{_x}"] ** 2 for _x in "XYZ"]),
        axis=0,
    )
    l2 = jnp.sum(
        jnp.array([halos[f"{sod}_halo_eig{red}2{_x}"] ** 2 for _x in "XYZ"]),
        axis=0,
    )
    l3 = jnp.sum(
        jnp.array([halos[f"{sod}_halo_eig{red}3{_x}"] ** 2 for _x in "XYZ"]),
        axis=0,
    )

    a, b, c = l3**0.5, l2**0.5, l1**0.5

    L = 1 + ((b / a) ** 2) + ((c / a) ** 2)
    e = (1 - (c / a) ** 2) / (2 * L)

    p = (1 - (2 * (b / a) ** 2) + (c / a) ** 2) / (2 * L)

    T = 0.5 * (1 + (p / e))

    return a, b, c, e, p, T


def compute_fof_com_offset(halos: dict) -> Array:
    """
    Compute the offset between the FoF halo center and the
    center of mass of its particles.

    Parameters
    ----------
    halos : dict
        HACC haloproperties catalog

    Returns
    -------
    NDArray
        Offset between FoF halo center and center of mass
    """
    return jnp.sqrt(
        (halos["fof_halo_com_x"] - halos["fof_halo_center_x"]) ** 2
        + (halos["fof_halo_com_y"] - halos["fof_halo_center_y"]) ** 2
        + (halos["fof_halo_com_z"] - halos["fof_halo_center_z"]) ** 2
    )


def build_input_dict(
    halos: dict,
    input_keys: list,
    shapes_sod: bool = True,
    shapes_reduced: bool = True,
) -> Dict:
    """
    Builds a dictionnary of halo features from a HACC halo catalog and
    a list of feature names.

    Parameters
    ----------
    halos : dict
        HACC haloproperties catalog
    input_keys : list
        List of halo feature names.
    shapes_sod : bool, optional
        Use the SOD inertia tensor (instead of FoF) to compute halo
        shapes, by default True
    shapes_reduced : bool, optional
        Use the reduced inertia tensor (instead of unreduced) to compute
        halo shapes, by default True

    Returns
    -------
    Dict
        Dictionnary of arrays containing halo features.
    """
    input_dict = {
        "log M200": jnp.log10(halos["sod_halo_mass"]),
        "log M500": jnp.log10(halos["sod_halo_M500c"]),
        "log Mfof": jnp.log10(halos["fof_halo_mass"]),
        "c200": halos["sod_halo_cdelta"],
        "cacc": halos["sod_halo_c_acc_mass"],
        "cpeak": halos["sod_halo_c_peak_mass"],
        "log sigmav": jnp.log10(halos["sod_halo_1D_vel_disp"]),
        "log vmax": jnp.log10(halos["sod_halo_max_cir_vel"]),
        "a25": halos["mah_halo_a25"],
        "a50": halos["mah_halo_a50"],
        "a75": halos["mah_halo_a75"],
        "almm": halos["mah_halo_a_lmm"],
        "mdot": halos["mah_halo_mass_acc_rate"],
    }
    input_dict["cacc/c200"] = input_dict["cacc"] / input_dict["c200"]
    input_dict["cpeak/c200"] = input_dict["cpeak"] / input_dict["c200"]
    input_dict["log dx/R200c"] = jnp.log10(
        compute_fof_com_offset(halos) / halos["sod_halo_radius"]
    )
    shapes = compute_halo_shapes(
        halos, use_sod=shapes_sod, use_reduced=shapes_reduced
    )
    input_dict["c/a"] = shapes[2] / shapes[0]
    input_dict["b/a"] = shapes[1] / shapes[0]
    input_dict["e"] = shapes[3]
    input_dict["p"] = shapes[4]
    input_dict["t"] = shapes[5]
    return {k: input_dict[k] for k in input_keys}


def build_input_vector(
    halos: dict,
    input_keys: list,
    shapes_sod: bool = True,
    shapes_reduced: bool = True,
) -> Array:
    """
    Builds an array of halo features from a HACC halo catalog and
    a list of feature names.

    Parameters
    ----------
    halos : dict
        HACC haloproperties catalog
    input_keys : list
        List of halo feature names.
    shapes_sod : bool, optional
        Use the SOD inertia tensor (instead of FoF) to compute halo
        shapes, by default True
    shapes_reduced : bool, optional
        Use the reduced inertia tensor (instead of unreduced) to compute
        halo shapes, by default True

    Returns
    -------
    Array
        Array of halo features (shape = (n_features, n_halos))
    """
    input_dict = build_input_dict(
        halos, input_keys, shapes_sod=shapes_sod, shapes_reduced=shapes_reduced
    )
    return jnp.array([input_dict[k] for k in input_keys])
