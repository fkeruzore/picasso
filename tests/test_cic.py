import numpy as np
import scipy.stats as ss
from scipy.interpolate import interp1d
from astropy.cosmology import FlatLambdaCDM
from picasso import cic, utils
import pytest


@pytest.mark.parametrize("which", ["gaussian", "nfw"])
def test_cic(which):
    np.random.seed(1118)
    n_parts = int(1e5)
    center = np.zeros(3)

    # Randomly sample multivariate gaussian and deposit using CIC
    if which == "gaussian":
        xyz = np.random.multivariate_normal(center, np.eye(3), n_parts)
        m_part = 1.0

        box_size = 15.0
        n_cells = 101
        m_grid_cic = cic.cic_3d_nojax(xyz, center, box_size, n_cells)

    # Randomly sample NFW profile and deposit using CIC
    elif which == "nfw":
        M500c = 1e15  # h-1 Msun
        c500c = 10.0
        z = 0
        cosmo = FlatLambdaCDM(70.0, 0.3)

        nfw = utils.NFW(M500c / 0.7, c500c, "500c", z, cosmo)
        R500c = nfw.RDelta
        m_part = M500c / n_parts  # h-1 Msun

        # theta, phi
        theta = 2 * np.pi * np.random.uniform(0, 1, n_parts)
        phi = np.arccos(2 * np.random.uniform(0, 1, n_parts) - 1)

        # r via inverse-cdf sampling
        _r = np.linspace(0, 2 * R500c, 100_000)
        _cdf = nfw.enclosed_mass(_r)
        _cdf /= _cdf.max()
        interp = interp1d(_cdf, _r)

        cdf = np.random.uniform(0, 1, n_parts)
        r = interp(cdf)

        # x, y, z
        xyz = np.array(
            [
                r * np.sin(phi) * np.cos(theta),
                r * np.sin(phi) * np.sin(theta),
                r * np.cos(phi),
            ]
        ).T

        box_size = 5.0 * R500c
        n_cells = 101
        cell_size = box_size / n_cells
        m_grid_cic = cic.cic_3d_nojax(
            xyz, center, box_size, n_cells, weights=m_part * np.ones(n_parts)
        )

    cell_size = box_size / n_cells
    rho_grid_cic = m_grid_cic / (cell_size**3)

    xyz_grid = np.array(
        np.meshgrid(
            np.linspace(-box_size / 2, box_size / 2, n_cells) + center[0],
            np.linspace(-box_size / 2, box_size / 2, n_cells) + center[1],
            np.linspace(-box_size / 2, box_size / 2, n_cells) + center[2],
        )
    )

    if which == "gaussian":
        rho_grid_pred = n_parts * ss.multivariate_normal(
            np.zeros(3), np.eye(3)
        ).pdf(xyz_grid.T.reshape(-1, 3)).reshape(n_cells, n_cells, n_cells)
    elif which == "nfw":
        r_grid = np.sqrt(np.sum(xyz_grid**2, axis=0))
        rho_grid_pred = nfw.density(r_grid)

    rel_diff = (rho_grid_cic / rho_grid_pred) - 1.0
    rel_diff_ok = rel_diff[m_grid_cic > (10 * m_part)]
    mu = 100 * np.mean(rel_diff_ok)
    assert mu < 10, f"Mean density biased by {mu:.2f}% (>10%)"
