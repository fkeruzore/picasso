import numpy as np
import jax.numpy as jnp
import h5py
import os
from astropy.cosmology import FlatLambdaCDM
from picasso.data_helpers import hacc

cosmo = FlatLambdaCDM(67.66, 0.30964, Ob0=0.04897)

here = os.path.dirname(os.path.abspath(__file__))
d = {}
with h5py.File(f"{here}/data/cutout_data.hdf5", "r") as f:
    for k, v in f.items():
        d[k] = {}
        for k2, v2 in v.items():
            d[k][k2] = np.array(v2)


def test_hacc_coutout_pair():
    cutouts = hacc.HACCCutoutPair(
        d["halo_go"],
        d["halo_hy"],
        d["parts_go"],
        d["parts_hy"],
        0.0,
        576.0,
        cosmo,
    )
