import numpy as np
import scipy.stats as ss
from halotools.empirical_models import NFWPhaseSpace
from astropy.cosmology import FlatLambdaCDM
from picasso import cic
import pytest


@pytest.mark.parametrize("which", ["gaussian", "nfw"])
def test_cic(which):
    np.random.seed(1811)
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
        M_vir = 1e15  # h-1 Msun
        c_vir = 10.0
        z = 0
        cosmo = FlatLambdaCDM(70.0, 0.3)

        np.random.seed(1811)
        nfw = NFWPhaseSpace(cosmology=cosmo, redshift=z, mdef="vir")
        R_vir = nfw.halo_mass_to_halo_radius(M_vir)  # h-1 Mpc
        m_part = M_vir / n_parts  # h-1 Msun

        p = nfw.mc_generate_nfw_phase_space_points(
            Ngals=int(n_parts), conc=c_vir, mass=M_vir
        )
        parts = {c: np.array(p[c]) for c in p.colnames}

        parts["mass"] = m_part * np.ones(n_parts)

        xyz = np.array([parts[_] for _ in "xyz"]).T
        box_size = 3.0 * R_vir
        n_cells = 71
        cell_size = box_size / n_cells
        m_grid_cic = cic.cic_3d_nojax(
            xyz, center, box_size, n_cells, weights=parts["mass"]
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
        rho_grid_pred = nfw.mass_density(r_grid, M_vir, c_vir)

    rel_diff = (rho_grid_cic / rho_grid_pred) - 1.0
    rel_diff_ok = rel_diff[m_grid_cic > (10 * m_part)]
    mu = 100 * np.mean(rel_diff_ok)
    assert mu < 10, f"Mean density biased by {mu:.2f}% (>10%)"
