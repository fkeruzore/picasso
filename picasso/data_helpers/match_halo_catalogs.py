import numpy as np
from numpy.typing import NDArray
from astropy.table import Table


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
    verbose: bool = True,
) -> (Table, NDArray, NDArray):
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
    Table
        Table containing matches. Two columns: `key_tag`_GO for
        gravity-only tags, `key_tag`_HY for hydro halo tags.
        Rows are halos in `halos_go`. If no match was found, the hydro
        tag is set to -1.
    array
        Tags of hydro halo matches for GO halos (shape=n_halos_go).
        If no match was found, the tag is set to -1.
    array
        Tags of GO halo matches for hydro halos (shape=n_halos_hydro).
        If no match was found, the tag is set to -1.
    """
    for k in [key_R, key_M, key_tag] + [
        f"{key_center_prefix}{x}" for x in "xyz"
    ]:
        assert k in halos_go.keys(), f"{k} is not a key in `halos_go`"
        assert k in halos_hy.keys(), f"{k} is not a key in `halos_hy`"

    tags_hy_match_in_go = -1 * np.ones_like(halos_go[key_tag])
    tags_go_match_in_hy = -1 * np.ones_like(halos_hy[key_tag])

    tags_matches = []
    n_halos = len(halos_go[key_tag])
    for i_go in range(n_halos):
        halo = {k: v[i_go] for k, v in halos_go.items()}
        xyz = {x: halo[f"{key_center_prefix}{x}"] for x in "xyz"}
        M_i, R_i = halo[key_M], halo[key_R]
        tag = halo[key_tag]
        tags_matches_i = {f"{key_tag}_GO": tag, f"{key_tag}_HY": -1}

        dx, dy, dz = [
            np.fmod((halos_hy[f"{key_center_prefix}{x}"] - xyz[x]), 576.0)
            for x in "xyz"
        ]
        dists = np.sqrt(dx**2 + dy**2 + dz**2)

        # Limit search to halos matching distance criterion (="cands")
        msk_cands = dists <= (d_R_max * R_i)
        dists = dists[msk_cands]
        cands = {k: v[msk_cands] for k, v in halos_hy.items()}

        # Sort candidates by distance to halo center
        dsort = np.argsort(dists)
        cands_dsorted = {k: v[dsort] for k, v in cands.items()}

        success = False
        for j_hy in range(len(dsort)):
            M_j = cands_dsorted[key_M][j_hy]
            if np.abs(M_j - M_i) < (d_M_max * M_i):
                tag_hy = cands_dsorted[key_tag][j_hy]
                tags_matches_i[f"{key_tag}_HY"] = tag_hy
                success = True
                break

        if success:
            tags_matches.append(tags_matches_i)
            tags_hy_match_in_go[i_go] = tag_hy
            i_hy = np.where(halos_hy[key_tag] == tag_hy)[0][0]
            tags_go_match_in_hy[i_hy] = tag

    tags_matches = Table(tags_matches)
    if verbose:
        n_nomatch = (tags_matches[f"{key_tag}_HY"] == -1).sum()
        f_nomatch = n_nomatch / n_halos
        print(f"-> {n_nomatch}/{n_halos} ({100*f_nomatch:.2f}%) unmatched")
    return tags_matches, tags_hy_match_in_go, tags_go_match_in_hy
