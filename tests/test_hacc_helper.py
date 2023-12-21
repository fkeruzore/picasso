import numpy as np
import jax.numpy as jnp
import h5py
import os
from astropy.cosmology import FlatLambdaCDM
from picasso.data_helpers import hacc, fitters
import pytest

cosmo = FlatLambdaCDM(67.66, 0.30964, Ob0=0.04897)

# Use full files if available, down-sampled if not
here = os.path.dirname(os.path.abspath(__file__))
path = f"{here}/data"
files = {s: f"{path}/data_1halo_{s}_full.hdf5" for s in ["624", "498"]}
if not all([os.path.isfile(f) for f in files.values()]):
    files = {s: f"{path}/data_1halo_{s}_1pct.hdf5" for s in ["624", "498"]}


d = {"624": {}, "498": {}}
for snap in d.keys():
    with h5py.File(files[snap], "r") as f:
        for k, v in f.items():  # h, h_ad, parts_ad, parts_go, profs_ad
            d[snap][k] = {}
            for k2, v2 in v.items():
                d[snap][k][k2] = np.array(v2)


@pytest.mark.parametrize("z", [0.0, 0.5])
def test_hacc_cutouts_profiles(z):
    if z == 0.0:
        snap = "624"
    elif z == 0.5:
        snap = "498"
    profs_hy_p = hacc.HACCSODProfiles.from_sodpropertybins(
        d[snap]["h_ad"], d[snap]["profs_ad"], z, cosmo, is_hydro=True
    )
    cutout_hy = hacc.HACCCutout(
        d[snap]["h_ad"], d[snap]["parts_ad"], z, 576.0, cosmo, is_hydro=True
    )
    profs_hy_c = hacc.HACCSODProfiles.from_cutout(
        cutout_hy, profs_hy_p.r_edges
    )
    for prop in ["rho_tot", "rho_g", "P_th", "f_nt"]:
        p, c = profs_hy_p.__dict__[prop], profs_hy_c.__dict__[prop]
        w = profs_hy_p.r > 0.05
        chi2 = np.nanmean(((p - c) / p)[w] ** 2)
        assert chi2 < 0.05, (
            f"{prop} profiles from SODpropertybins and cutout "
            + f"incompatible: chi2={chi2:.3f}"
        )


@pytest.mark.parametrize(
    ["case", "z"],
    [
        ("from profiles", 0.0),
        ("from particles", 0.0),
        ("from profiles", 0.5),
        ("from particles", 0.5),
    ],
)
def test_fit_hacc_profiles(case, z):
    if z == 0.0:
        snap = "624"
    elif z == 0.5:
        snap = "498"
    cutout_go = hacc.HACCCutout(
        d[snap]["h"], d[snap]["parts_go"], z, 576.0, cosmo, is_hydro=False
    )
    if case == "from profiles":
        profs_hy = hacc.HACCSODProfiles.from_sodpropertybins(
            d[snap]["h_ad"], d[snap]["profs_ad"], z, cosmo, is_hydro=True
        )
        profs_hy.rebin(jnp.logspace(-1.25, 0.25, 8))
    elif case == "from particles":
        cutout_hy = hacc.HACCCutout(
            d[snap]["h_ad"],
            d[snap]["parts_ad"],
            z,
            576.0,
            cosmo,
            is_hydro=True,
        )
        profs_hy = hacc.HACCSODProfiles.from_cutout(
            cutout_hy, jnp.logspace(-1.25, 0.25, 8)
        )

    res_pol = fitters.fit_gas_profiles(
        cutout_go,
        profs_hy,
        which_P="tot",
        fit_fnt=False,
        return_history=True,
    )
    assert res_pol.bl < 1.0, f"Large loss value: {res_pol.bl:.2e}"
