import jax
import jax.numpy as jnp
from scipy.spatial import KDTree
from functools import partial
from typing import Sequence
from jax import Array


def _sph_kernel_cubic_spline(x, h):
    q = x / (h / 2)
    W = jnp.where((q <= 1), 1 - 1.5 * q**2 + 0.75 * q**3, 0.0)
    W = jnp.where((q > 1) & (q < 2), 0.25 * (2 - q) ** 3, W)
    return W * 1 / (jnp.pi * h**3)


def _sph_kernel_wendland(x, h):
    q = x / h
    W = (
        495
        / (32 * jnp.pi * h**3)
        * ((1 - q) ** 6)
        * (1 + 6 * q + 35 / 3 * q**2)
    )
    return jnp.where((q < 1), W, 0.0)


sph_kernels = {
    "cubic_spline": _sph_kernel_cubic_spline,
    "wendland": _sph_kernel_wendland,
}


@partial(jax.jit, static_argnums=(2,))
def _n_nearest_dists_indices_bruteforce_i(xyz_i, xyz_j, n):
    dists = jnp.sqrt(jnp.sum((xyz_j - xyz_i) ** 2, axis=1))
    dists_best_n, i_best_n = jax.lax.top_k(-dists, n)
    return -dists_best_n, i_best_n


def n_nearest_dists_indices_bruteforce(xyz, n):
    return jax.vmap(
        _n_nearest_dists_indices_bruteforce_i, in_axes=(0, None, None)
    )(xyz, xyz, n)


def n_nearest_dists_indices_kdtree(xyz, n):
    tree = KDTree(xyz)
    dists_best_n, i_best_n = tree.query(xyz, k=n)
    return jnp.array(dists_best_n), jnp.array(i_best_n)


nnd_funcs = {
    "bruteforce": n_nearest_dists_indices_bruteforce,
    "kdtree": n_nearest_dists_indices_kdtree,
}


def sph_radii_volumes(
    xyz: Array,
    n: int = 32,
    kernel: str = "wendland",
    kind: str = "kdtree",
    return_dists_and_indices: bool = False,
):
    sph_kernel = sph_kernels[kernel]
    nnd_func = nnd_funcs[kind]
    r_ij, j = nnd_func(xyz, n)
    h = r_ij[:, -1]
    V = 1 / jnp.sum(sph_kernel(r_ij, r_ij[:, -1, None]), axis=1)

    if return_dists_and_indices:
        return h, V, r_ij, j
    else:
        return h, V


def sph_convolve(
    r_ij: Array,
    h_i: Array,
    V_j: Array,
    arrs: Sequence[Array],
    kernel: str = "wendland",
) -> Sequence[Array]:
    """
    Perform convolution of arrays using the Smoothed Particle
    Hydrodynamics (SPH) method, on N particles with K neighbors.

    Parameters
    ----------
    r_ij : Array
        The distance between particles i and j, shape: (N, K)
    h_i : Array
        The smoothing length for particle i, shape: (N,)
    V_j : Array
        The volume of particle j, shape: (N, K)
    arrs : Sequence[Array]
        The arrays to be convolved, each of shape: (N, K)
    kernel : str, optional
        Name of the kernel function to use for convolution.
        Defaults to "wendland".

    Returns
    -------
    Sequence[Array]
        The convolved arrays, each of shape: (N,)

    """
    sph_kernel = sph_kernels[kernel]
    VjWij = V_j * sph_kernel(r_ij, h_i[:, None])
    return [jnp.sum(arr * VjWij, axis=1) for arr in arrs]


if __name__ == "__main__":
    N_PARTS = 15_000

    keys = jax.random.split(jax.random.PRNGKey(0), 5)

    parts_i = {
        "x": jax.random.normal(keys[0], (N_PARTS,)),
        "y": jax.random.normal(keys[1], (N_PARTS,)),
        "z": jax.random.normal(keys[2], (N_PARTS,)),
    }

    xyz = jnp.array([parts_i["x"], parts_i["y"], parts_i["z"]]).T
    parts_i["r"] = jnp.sqrt(jnp.sum(xyz**2, axis=-1))
    parts_i["P_th"] = 10.0 * jnp.exp(-0.5 * parts_i["r"] ** 2)
    parts_i["P_th"] += 0.5 * jax.random.normal(keys[3], (N_PARTS,))
    parts_i["rho_g"] = 1.0 * jnp.exp(-0.5 * parts_i["r"] ** 2)
    parts_i["rho_g"] += 0.05 * jax.random.normal(keys[4], (N_PARTS,))

    h, V, r, j = sph_radii_volumes(
        xyz,
        n=32,
        kernel="wendland",
        kind="kdtree",
        return_dists_and_indices=True,
    )
