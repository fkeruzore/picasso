import numpy as np
from scipy.spatial import KDTree


def match_halo_catalogs(
    halos_go: dict,
    halos_hy: dict,
    box_size: float,
    d_R_max: float = 1.0,
    d_M_max: float = 0.2,
    key_R: str = "sod_halo_R500c",
    key_M: str = "sod_halo_M500c",
    key_tag: str = "fof_halo_tag",
    key_center_prefix: str = "fof_halo_center_",
    suffix_go: str = "_GO",
    suffix_hy: str = "_HY",
    verbose: bool = True,
) -> dict:
    """Matches hydro counterparts to a gravity-only halo catalog using
    spatial separation and mass difference.

    Parameters
    ----------
    halos_go : dict
        Gravity-only halo catalog to find counterparts for
    halos_hy : dict
        Hydro halo catalog in which counterparts are to be found
    box_size : float
        Box size of the simulation, in the same units as distances
        (radius, coordinates) in the halo catalogs
    d_R_max : float, optional
        Maximum distance between halo centers to be accepted as a pair,
        in units of `key_R`; by default 1.0
    d_M_max : float, optional
        Maximum mass difference between halos to be accepted as a pair,
        in units of `key_M`; by default 0.2
    key_R : str, optional
        Catalog key to be used for halo radii,
        by default "sod_halo_R500c"
    key_M : str, optional
        Catalog key to be used for halo masses,
        by default "sod_halo_M500c"
    key_tag : str, optional
        Catalog key to be used for halo tags, by default "fof_halo_tag"
    key_center_prefix : str, optional
        Prefix (before 'x', 'y', 'z') of the key to be used for halo
        coordinates, by default "fof_halo_center_"
    verbose : bool, optional
        Print matching summary, by default True

    Returns
    -------
    dict
        Dictionary with keys from `halos_go` and `halos_hy` and values
        matched to each other, with respective suffixes.
    """
    for k in [key_R, key_M, key_tag] + [
        f"{key_center_prefix}{x}" for x in "xyz"
    ]:
        assert k in halos_go.keys(), f"{k} is not a key in `halos_go`"
        assert k in halos_hy.keys(), f"{k} is not a key in `halos_hy`"

    n_halos = len(halos_go[key_tag])

    halos_matched = {
        **{f"{k}{suffix_go}": v for k, v in halos_go.items()},
        **{
            f"{k}{suffix_hy}": np.zeros(n_halos, v.dtype) - 1
            for k, v in halos_hy.items()
        },
    }

    xyz_go = np.array([halos_go[f"{key_center_prefix}{x}"] for x in "xyz"]).T
    xyz_go = np.fmod(xyz_go + box_size, box_size)
    xyz_hy = np.array([halos_hy[f"{key_center_prefix}{x}"] for x in "xyz"]).T
    xyz_hy = np.fmod(xyz_hy + box_size, box_size)

    dists_best_n, i_best_n = KDTree(xyz_hy, boxsize=box_size).query(
        xyz_go, k=5
    )
    for i_go in range(n_halos):
        halo = {k: v[i_go] for k, v in halos_go.items()}
        M_i, R_i = halo[key_M], halo[key_R]
        cands = {k: v[i_best_n[i_go]] for k, v in halos_hy.items()}
        msk_cands = np.logical_and(
            dists_best_n[i_go] <= (d_R_max * R_i),
            np.abs(cands[key_M] - M_i) < (d_M_max * M_i),
        )
        success = np.any(msk_cands)
        if success:
            i_hy = i_best_n[i_go][msk_cands][0]
            for k, v in halos_hy.items():
                halos_matched[f"{k}{suffix_hy}"][i_go] = v[i_hy]

    if verbose:
        n_matched = np.sum(halos_matched[f"{key_tag}{suffix_hy}"] != -1)
        print(
            f"Matched {n_matched} of {n_halos} GO halos to hydro halos",
            f"({n_matched / n_halos:.2%})",
        )

    return halos_matched
