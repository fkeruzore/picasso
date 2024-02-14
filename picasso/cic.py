from jax import Array
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray


def cic_3d(
    positions: Array,
    center: Array,
    box_size: float,
    n_cells: int,
    weights=None,
) -> Array:
    """Interpolate particle properties on a regulat cubic grid using
    Cloud In Cell (CIC) deposition.

    Parameters
    ----------
    positions : Array
        (x, y, z) positions of the N particles (shape=(N, 3)).
    center : Array
        (x, y, z) position of the center (shape=(3,))
    box_size : float
        Size of the cube along the (x, y, z) axes.
    n_cells : int
        Number of cells along the (x, y, z) axes.
    weights : Array or None, optional
        Particle property to be deposited.
        If None (default), each particle will be attributed a weight 1,
        and the resulting grid is the interpolated particle count.

    Returns
    -------
    Array
        Interpolated field property.
    """

    n_parts = positions.shape[0]
    center = jnp.array(center)[jnp.newaxis, :]
    box_size = box_size * jnp.ones(3, dtype=jnp.float32)[jnp.newaxis, :]

    # Normalize positions between 0 and 1
    positions = (positions - center) / box_size + 0.5
    grid = jnp.zeros((n_cells, n_cells, n_cells), dtype=jnp.float32)
    if weights is None:
        weights = jnp.ones(n_parts)

    for i_p in range(n_parts):
        w = weights[i_p]
        xyz = positions[i_p, :] * n_cells - 0.5  # 3d position in cell space
        i = jnp.floor(xyz).astype(jnp.int16)  # position indices of cell
        j = i + 1  # position indices of next cell
        d = xyz - i  # 1d distance to next cell edge
        t = 1.0 - d  # 1d distance to cell edge

        grid.at[i[0], i[1], i[2]].add(w * t[0] * t[1] * t[2])
        grid.at[i[0], i[1], j[2]].add(w * t[0] * t[1] * d[2])
        grid.at[i[0], j[1], i[2]].add(w * t[0] * d[1] * t[2])
        grid.at[i[0], j[1], j[2]].add(w * t[0] * d[1] * d[2])
        grid.at[j[0], i[1], i[2]].add(w * d[0] * t[1] * t[2])
        grid.at[j[0], i[1], j[2]].add(w * d[0] * t[1] * d[2])
        grid.at[j[0], j[1], i[2]].add(w * d[0] * d[1] * t[2])
        grid.at[j[0], j[1], j[2]].add(w * d[0] * d[1] * d[2])

    return grid


def cic_3d_nojax(
    positions: NDArray,
    center: NDArray,
    box_size: float,
    n_cells: int,
    weights=None,
    verbose=False,
) -> NDArray:
    """Interpolate particle properties on a regulat cubic grid using
    Cloud In Cell (CIC) deposition.

    Parameters
    ----------
    positions : Array
        (x, y, z) positions of the N particles (shape=(N, 3)).
    center : Array
        (x, y, z) position of the center (shape=(3,))
    box_size : float
        Size of the cube along the (x, y, z) axes.
    n_cells : int
        Number of cells along the (x, y, z) axes.
    weights : Array or None, optional
        Particle property to be deposited.
        If None (default), each particle will be attributed a weight 1,
        and the resulting grid is the interpolated particle count.

    Returns
    -------
    Array
        Interpolated field property.
    """

    n_parts = positions.shape[0]
    center = np.array(center)[np.newaxis, :]
    box_size = box_size * np.ones(3, dtype=np.float32)[np.newaxis, :]
    if weights is None:
        weights = np.ones(n_parts)

    # Normalize positions between 0 and 1
    positions = (positions - center) / box_size + 0.5

    # Remove particles outside the box
    in_box = np.all((positions > 0) & (positions < 1), axis=1)
    positions = positions[in_box, :]
    weights = weights[in_box]

    grid = np.zeros((n_cells, n_cells, n_cells), dtype=np.float32)

    for i_p in range(in_box.sum()):
        w = weights[i_p]
        xyz = positions[i_p, :] * n_cells - 0.5  # 3d position in cell space
        i = np.floor(xyz).astype(np.int16)  # position indices of cell
        j = i + 1  # position indices of next cell
        d = xyz - i  # 1d distance to next cell edge
        t = 1.0 - d  # 1d distance to cell edge
        if verbose:
            print(i, j, d, t)

        grid[i[0], i[1], i[2]] += w * t[0] * t[1] * t[2]
        grid[i[0], i[1], j[2]] += w * t[0] * t[1] * d[2]
        grid[i[0], j[1], i[2]] += w * t[0] * d[1] * t[2]
        grid[i[0], j[1], j[2]] += w * t[0] * d[1] * d[2]
        grid[j[0], i[1], i[2]] += w * d[0] * t[1] * t[2]
        grid[j[0], i[1], j[2]] += w * d[0] * t[1] * d[2]
        grid[j[0], j[1], i[2]] += w * d[0] * d[1] * t[2]
        grid[j[0], j[1], j[2]] += w * d[0] * d[1] * d[2]

    return grid
