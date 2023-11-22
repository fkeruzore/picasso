import numpy as np
from picasso.data_helpers import match_halo_catalogs
import pytest


@pytest.mark.parametrize("test_unmatched", [True, False])
def test_match_halo_catalogs(test_unmatched):
    n_go = 10
    np.random.seed(1118)
    halos_go = {
        "tag": np.arange(n_go),
        "center_x": np.random.uniform(0.0, 50.0, n_go),
        "center_y": np.random.uniform(0.0, 50.0, n_go),
        "center_z": np.random.uniform(0.0, 50.0, n_go),
        "R": np.random.uniform(0.5, 1.0, n_go),
    }
    halos_go["M"] = halos_go["R"] ** 3

    halos_hy = {
        "tag": halos_go["tag"] + 20,
        "R": halos_go["R"] * np.random.uniform(0.99, 1.01, n_go),
    }
    halos_hy["M"] = halos_hy["R"] ** 3
    for x in "xyz":
        k = f"center_{x}"
        halos_hy[k] = halos_go[k] + np.random.uniform(0.0, 0.1, n_go)

    matches, tags_hy_in_go, tags_go_in_hy = match_halo_catalogs(
        halos_go,
        halos_hy,
        75.0 if test_unmatched else 50.0,
        d_R_max=1.0,
        d_M_max=0.5,
        key_R="R",
        key_M="M",
        key_tag="tag",
        key_center_prefix="center_",
    )
    assert np.all(matches["tag_HY"] != -1), "Found unmatched halos"
    assert np.all(
        matches["tag_GO"] == matches["tag_HY"] - 20
    ), "Not all halos got matched to the correct counterpart"
    assert np.all(
        tags_hy_in_go == halos_go["tag"] + 20
    ), "Hydro match tags are not correctly attributed"
    assert np.all(
        tags_go_in_hy == halos_hy["tag"] - 20
    ), "GO match tags are not correctly attributed"

    if test_unmatched:  # Add a GO halo with no match
        halos_go_b = {}
        halos_go_b["tag"] = np.append(halos_go["tag"], [999])
        halos_go_b["center_x"] = np.append(halos_go["center_x"], [70.0])
        halos_go_b["center_y"] = np.append(halos_go["center_y"], [70.0])
        halos_go_b["center_z"] = np.append(halos_go["center_z"], [70.0])
        halos_go_b["R"] = np.append(halos_go["R"], [1.0])
        halos_go_b["M"] = halos_go_b["R"] ** 3

        matches_b, tags_hy_in_go_b, tags_go_in_hy_b = match_halo_catalogs(
            halos_go_b,
            halos_hy,
            75.0,
            d_R_max=1.0,
            d_M_max=0.5,
            key_R="R",
            key_M="M",
            key_tag="tag",
            key_center_prefix="center_",
        )
        assert np.all(matches_b[:-1] == matches)
        assert np.all(tags_go_in_hy_b == tags_go_in_hy)
        assert np.all(tags_hy_in_go_b[:-1] == tags_hy_in_go)
        assert np.all(tags_hy_in_go_b[-1] == -1), (
            "The unmatched GO halo was tagged as a match to halo "
            + f"{tags_hy_in_go_b[-1]}"
        )
