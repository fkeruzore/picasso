import numpy as np
from picasso import cic


def test_cic_gaussian():
    import scipy.stats as ss

    np.random.seed(1811)
    center = np.zeros(3)
    n_parts = int(1e5)
    xyz = np.random.multivariate_normal(center, np.eye(3), n_parts)

    box_size = 15.0
    n_cells = 101
    m_grid = cic.cic_3d_nojax(xyz, np.zeros(3), box_size, n_cells)

    cell_size = box_size / n_cells
    rho_grid = m_grid / (cell_size**3)

    xyz_grid = np.array(
        np.meshgrid(
            np.linspace(-box_size / 2, box_size / 2, n_cells) + center[0],
            np.linspace(-box_size / 2, box_size / 2, n_cells) + center[1],
            np.linspace(-box_size / 2, box_size / 2, n_cells) + center[2],
        )
    )

    grid_pred = n_parts * ss.multivariate_normal(np.zeros(3), np.eye(3)).pdf(
        xyz_grid.T.reshape(-1, 3)
    )

    rel_diff = (rho_grid / grid_pred.reshape(n_cells, n_cells, n_cells)) - 1.0
    rel_diff_ok = rel_diff[m_grid > 10]
    mu = 100 * np.mean(rel_diff_ok)
    assert mu < 10, f"Mean density biased by {mu:.2f}% (>10%)"


def test_cic_nfw():
    from halotools.empirical_models import NFWPhaseSpace
    from astropy.cosmology import FlatLambdaCDM

    num_pts = int(1e5)
    M_vir = 1e15  # h-1 Msun
    c_vir = 10.0
    z = 0
    cosmo = FlatLambdaCDM(70.0, 0.3)

    np.random.seed(1811)
    nfw = NFWPhaseSpace(cosmology=cosmo, redshift=z, mdef="vir")
    R_vir = nfw.halo_mass_to_halo_radius(M_vir)  # h-1 Mpc
    m_part = M_vir / num_pts  # h-1 Msun

    p = nfw.mc_generate_nfw_phase_space_points(
        Ngals=int(num_pts), conc=c_vir, mass=M_vir
    )
    parts = {c: np.array(p[c]) for c in p.colnames}

    parts["mass"] = m_part * np.ones(num_pts)

    xyz = np.array([parts[_] for _ in "xyz"]).T
    box_size = 3.0 * R_vir
    n_cells = 71
    cell_size = box_size / n_cells
    m_grid = cic.cic_3d_nojax(
        xyz,
        np.zeros(3),
        box_size,
        n_cells,
        weights=parts["mass"],
        verbose=False,
    )
    rho_grid = m_grid / (cell_size**3)

    xyz_grid = np.array(
        np.meshgrid(
            np.linspace(-box_size / 2, box_size / 2, n_cells),
            np.linspace(-box_size / 2, box_size / 2, n_cells),
            np.linspace(-box_size / 2, box_size / 2, n_cells),
        )
    )
    r_grid = np.sqrt(np.sum(xyz_grid**2, axis=0))
    nfw_rho_grid = nfw.mass_density(r_grid, M_vir, c_vir)

    rel_diff = (rho_grid / nfw_rho_grid) - 1
    rel_diff_ok = rel_diff[m_grid > 50 * m_part]
    mu = 100 * np.mean(rel_diff_ok)
    assert mu < 10, f"Mean density biased by {mu:.2f}% (>10%)"
