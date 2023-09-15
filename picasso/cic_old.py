from math import floor
from jax import jit, numpy as jnp


@jit
def cic3d_nonperiodic_jax(
    grid_n,
    part_xyz,
    n_cells,
    periodic_mask,
    shift=False,
    other_grids=[],
    other_props=[],
):
    """
    Cloud-in-cell particle population and total velocity
    estimation from particle data.

    Parameters
    ==========
    grid_n : ndarray (n, n, n)
        3d grid to be filled with the number of particles at
        each point.
    part_xyz : ndarray (k, 3)
        (x, y, z) positions of each particle, normalized to the
        box size (i.e. only particles between 0 and 1 will be
        projected)
    n_cells : ndarray (3)
        Number of cells on the grid in (x, y, z) directions.
        #UNTESTED with different numbers in each dir!
    periodic_mask : ndarray (3)
        Mask to periodically wrap particles at boundaries.
        #UNTESTED
    shift : bool, default=False
        Shift particles by half a cell.
        #UNTESTED

    Notes
    =====
    - Input notations assume the grid is (n, n, n), and you have
        k particles to project.
    - The arrays provided as `grid_n` and `grid_v2` will be filled,
        the function itself does not return anything.
    - After execution, `grid_n` will contain the number of particles
        at each grid point, and `grid_v2` the sum of the quadratic
        velocities at each point.
    - Interesting quantities can be computed as follows:
        * rho = grid_n * particle_mass / cell_volume
        * <v2> = grid_v2 / grid_n
        * rho<v2> = grid_v2 * particle_mass / cell_volume

    Authors
    =======
    - Michael Buehlmann,
    - Florian Keruzore, fkeruzore@anl.gov
    """
    x = jnp.empty(3, dtype=jnp.float32)
    for i in range(len(part_xyz)):
        x[:] = part_xyz[i] * n_cells
        # phi = part_phi[i]
        if not shift:
            # we want x to be the lower left corner of the cube which we
            # are depositing
            x -= 0.5

        # correct periodicity for periodic axes
        x[0] = (
            periodic_mask[0] * jnp.fmod(n_cells[0] + x[0], n_cells[0])
            + (not periodic_mask[0]) * x[0]
        )
        x[1] = (
            periodic_mask[1] * jnp.fmod(n_cells[1] + x[1], n_cells[1])
            + (not periodic_mask[1]) * x[1]
        )
        x[2] = (
            periodic_mask[2] * jnp.fmod(n_cells[2] + x[2], n_cells[2])
            + (not periodic_mask[2]) * x[2]
        )

        ix = floor(x[0])
        iy = floor(x[1])
        iz = floor(x[2])

        ix1 = ix + 1
        iy1 = iy + 1
        iz1 = iz + 1

        # 1d depositis
        dx = x[0] - ix
        dy = x[1] - iy
        dz = x[2] - iz

        tx = (1 - dx) * (periodic_mask[0] or (ix >= 0 and ix < n_cells[0]))
        ty = (1 - dy) * (periodic_mask[1] or (iy >= 0 and iy < n_cells[1]))
        tz = (1 - dz) * (periodic_mask[2] or (iz >= 0 and iz < n_cells[2]))

        # zero-out deposits if outside grid_n
        dx *= periodic_mask[0] or (ix1 >= 0 and ix1 < n_cells[0])
        dy *= periodic_mask[1] or (iy1 >= 0 and iy1 < n_cells[1])
        dz *= periodic_mask[2] or (iz1 >= 0 and iz1 < n_cells[2])

        # make everything valid grid_n indices
        ix %= n_cells[0]
        iy %= n_cells[1]
        iz %= n_cells[2]

        ix1 %= n_cells[0]
        iy1 %= n_cells[1]
        iz1 %= n_cells[2]

        # deposit particle counts
        grid_n[ix, iy, iz] += tx * ty * tz
        grid_n[ix, iy, iz1] += tx * ty * dz
        grid_n[ix, iy1, iz] += tx * dy * tz
        grid_n[ix, iy1, iz1] += tx * dy * dz
        grid_n[ix1, iy, iz] += dx * ty * tz
        grid_n[ix1, iy, iz1] += dx * ty * dz
        grid_n[ix1, iy1, iz] += dx * dy * tz
        grid_n[ix1, iy1, iz1] += dx * dy * dz

        # deposit other things
        if len(other_grids) > 0:
            for other_grid, other_prop in zip(other_grids, other_props):
                prop = other_prop[i]
                other_grid[ix, iy, iz] += (tx * ty * tz) * prop
                other_grid[ix, iy, iz1] += (tx * ty * dz) * prop
                other_grid[ix, iy1, iz] += (tx * dy * tz) * prop
                other_grid[ix, iy1, iz1] += (tx * dy * dz) * prop
                other_grid[ix1, iy, iz] += (dx * ty * tz) * prop
                other_grid[ix1, iy, iz1] += (dx * ty * dz) * prop
                other_grid[ix1, iy1, iz] += (dx * dy * tz) * prop
                other_grid[ix1, iy1, iz1] += (dx * dy * dz) * prop
