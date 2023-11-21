import numpy as np
import jax.numpy as jnp
import h5py
import os
from astropy.cosmology import FlatLambdaCDM
from picasso.data_helpers import hacc, fitters
import pytest

cosmo = FlatLambdaCDM(67.66, 0.30964, Ob0=0.04897)

here = os.path.dirname(os.path.abspath(__file__))
d = {}
with h5py.File(f"{here}/data/cutout_data.hdf5", "r") as f:
    for k, v in f.items():  # halo_ad, halo_go, parts_ad, parts_go
        d[k] = {}
        for k2, v2 in v.items():
            d[k][k2] = np.array(v2)
d_nov = {k: v for k, v in d.items()}
for x in "xyz":
    _ = d_nov["parts_go"].pop(f"v{x}")


@pytest.mark.parametrize("has_velocities", [True, False])
def test_hacc_coutout_pair(has_velocities):
    test_data = d if has_velocities else d_nov
    _ = hacc.HACCCutoutPair(
        test_data["halo_go"],
        test_data["halo_hy"],
        test_data["parts_go"],
        test_data["parts_hy"],
        0.0,
        576.0,
        cosmo,
    )


def test_fit_hacc_cutouts():
    cutouts = hacc.HACCCutoutPair(
        d["halo_go"],
        d["halo_hy"],
        d["parts_go"],
        d["parts_hy"],
        0.0,
        576.0,
        cosmo,
    )
    _ = fitters.fit_gas_profiles(cutouts, jnp.logspace(-1, 0, 10))
