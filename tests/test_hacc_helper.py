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
    for k, v in f.items():  # h, h_ad, parts_ad, parts_go, profs_ad
        d[k] = {}
        for k2, v2 in v.items():
            d[k][k2] = np.array(v2)
R500 = float(d["h"]["sod_halo_R500c"])


@pytest.mark.parametrize("case", ["grav-only", "hydro"])
def test_hacc_cutout(case):
    attrs = ["halo", "a", "z", "parts"]
    if case == "grav-only":
        cutout = hacc.HACCCutout(
            d["h"], d["parts_go"], 0.0, 576.0, cosmo, is_hydro=False
        )
    elif case == "hydro":
        cutout = hacc.HACCCutout(
            d["h_ad"], d["parts_ad"], 0.0, 576.0, cosmo, is_hydro=True
        )
        attrs += ["gas_parts"]

    for attr in attrs:
        assert hasattr(cutout, attr), f"Missing cutout attribute: {attr}"


@pytest.mark.parametrize("case", ["from profs", "from particles"])
def test_hacc_profile(case):
    if case == "from profs":
        profs_hy = hacc.HACCSODProfiles.from_sodpropertybins(
            d["h_ad"], d["profs_ad"], 0.0, cosmo, is_hydro=True
        )
    elif case == "from particles":
        cutout = hacc.HACCCutout(
            d["h_ad"], d["parts_ad"], 0.0, 576.0, cosmo, is_hydro=True
        )
        profs_hy = hacc.HACCSODProfiles.from_cutout(
            cutout, jnp.logspace(-1.2, 0.3, 16) * R500
        )
    profs_hy.rebin(jnp.logspace(-1.0, 0.1, 8) * R500)


def test_fit_gas_profiles():
    cutout_go = hacc.HACCCutout(
        d["h"], d["parts_go"], 0.0, 576.0, cosmo, is_hydro=False
    )
    profs_hy = hacc.HACCSODProfiles.from_sodpropertybins(
        d["h_ad"], d["profs_ad"], 0.0, cosmo, is_hydro=True
    )
    res_pol = fitters.fit_gas_profiles(
        cutout_go,
        profs_hy,
        which_P="tot",
        fit_fnt=False,
        return_history=True,
    )
    assert res_pol.bl < 1.0, f"Large loss value: {res_pol.bl:.2e}"


if __name__ == "__main__":
    import time

    ti = time.time()
    cutout_go = hacc.HACCCutout(
        d["h"], d["parts_go"], 0.0, 576.0, cosmo, is_hydro=False
    )
    profs_hy = hacc.HACCSODProfiles.from_sodpropertybins(
        d["h_ad"], d["profs_ad"], 0.0, cosmo, is_hydro=True
    )
    res_pol = fitters.fit_gas_profiles(
        cutout_go,
        profs_hy,
        which_P="tot",
        fit_fnt=False,
        return_history=True,
    )
    print(time.time() - ti)
