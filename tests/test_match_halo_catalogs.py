import numpy as np
from picasso.data_helpers import match_halo_catalogs


def test_match_halo_catalogs():
    n_go = 10
    halos_go = {
        "tag": np.arange(n_go),
        "center_x": np.random.uniform(0.0, 50.0, n_go),
        "center_y": np.random.uniform(0.0, 50.0, n_go),
        "center_z": np.random.uniform(0.0, 50.0, n_go),
        "R": np.random.uniform(0.5, 1.0, n_go),
    }
    halos_go["M"] = halos_go["R"] ** 3

    halos_hy = {
        "tag": halos_go["tag"],
        "R": halos_go["R"] * np.random.uniform(0.99, 1.01, n_go),
    }
    halos_hy["M"] = halos_hy["R"] ** 3
    for x in "xyz":
        k = f"center_{x}"
        halos_hy[k] = halos_go[k] + np.random.uniform(0.0, 0.1, n_go)

    matches = match_halo_catalogs(
        halos_go,
        halos_hy,
        50.0,
        d_R_max=1.0,
        d_M_max=0.5,
        key_R="R",
        key_M="M",
        key_tag="tag",
        key_center_prefix="center_",
    )

    assert np.all(matches["tag_HY"] != -1), "Found unmatched halos"
    assert np.all(
        matches["tag_GO"] == matches["tag_HY"]
    ), "Not all halos got matched to the correct counterpart"


if __name__ == "__main__":
    test_match_halo_catalogs()
