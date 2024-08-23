import jax
import jax.numpy as jnp
from jax.scipy.integrate import trapezoid
import pytest

from picasso.utils import sph

N_PARTS = 14_999
NN_SPH = 32
keys = jax.random.split(jax.random.PRNGKey(0), 5)

parts_i = {
    "x": jax.random.normal(keys[0], (N_PARTS,)),
    "y": jax.random.normal(keys[1], (N_PARTS,)),
    "z": jax.random.normal(keys[2], (N_PARTS,)),
}

xyz = jnp.array([parts_i["x"], parts_i["y"], parts_i["z"]]).T
parts_i["r"] = jnp.sqrt(jnp.sum(xyz**2, axis=-1))
parts_i["arr1"] = 10.0 * jnp.exp(-0.5 * parts_i["r"] ** 2)
parts_i["arr1"] += 0.5 * jax.random.normal(keys[3], (N_PARTS,))
parts_i["arr2"] = 1.0 * jnp.exp(-0.5 * parts_i["r"] ** 2)
parts_i["arr2"] += 0.05 * jax.random.normal(keys[4], (N_PARTS,))


@pytest.mark.parametrize("kernel", ["cubic_spline", "wendland"])
def test_kernel_integral(kernel):
    w = sph.sph_kernels[kernel](x := jnp.linspace(0.0, 1.55, 10_000), 1.5)
    integ = 4 * jnp.pi * trapezoid(w * x**2, x)
    assert jnp.isclose(
        integ, 1.0, rtol=1e-3
    ), f"Kernel `{kernel}` does not integrate to 1: {integ}"


@pytest.mark.parametrize(
    ["backend", "batches"],
    [
        ("kdtree", "single_batch"),
        ("kdtree", "multi_batch"),
        ("bruteforce", "single_batch"),
        ("bruteforce", "multi_batch"),
    ],
)
def test_radii_volumes_batching(backend, batches):
    if batches == "multi_batch":
        N_PARTS_MAX_BATCH = 5000
        N_BATCHES = 1 + N_PARTS // N_PARTS_MAX_BATCH
        iter_batches = [
            *jnp.arange(N_BATCHES * N_PARTS_MAX_BATCH).reshape(
                N_BATCHES, N_PARTS_MAX_BATCH
            )
        ]
        iter_batches[-1] = iter_batches[-1][iter_batches[-1] < N_PARTS]

        results = [
            sph.sph_radii_volumes(
                xyz[batch],
                xyz,
                n=NN_SPH,
                kernel="wendland",
                kind=backend,
                return_dists_and_indices=True,
            )
            for batch in iter_batches
        ]
        h = jnp.concatenate([_[0] for _ in results])
        V = jnp.concatenate([_[1] for _ in results])
        r = jnp.concatenate([_[2] for _ in results])
        j = jnp.concatenate([_[3] for _ in results])

    elif batches == "single_batch":
        h, V, r, j = sph.sph_radii_volumes(
            xyz,
            xyz,
            n=NN_SPH,
            kernel="wendland",
            kind=backend,
            return_dists_and_indices=True,
        )

    assert h.size == N_PARTS, f"Size of h: {h.size} (should be {N_PARTS})"
    assert V.size == N_PARTS, f"Size of V: {h.size} (should be {N_PARTS})"

    assert r.shape == (
        N_PARTS,
        NN_SPH,
    ), f"Shape of r: {r.shape} (should be {(N_PARTS, NN_SPH)})"
    assert j.shape == (
        N_PARTS,
        NN_SPH,
    ), f"Shape of j: {j.shape} (should be {(N_PARTS, NN_SPH)})"


def test_sph_convolve():
    h, V, r, j = sph.sph_radii_volumes(
        xyz,
        xyz,
        n=NN_SPH,
        kernel="wendland",
        kind="kdtree",
        return_dists_and_indices=True,
    )
    arr1_conv, arr2_conv = sph.sph_convolve(
        r, h, V[j], [parts_i["arr1"][j], parts_i["arr2"][j]], kernel="wendland"
    )
    assert arr1_conv.shape == parts_i["arr1"].shape
    assert arr2_conv.shape == parts_i["arr2"].shape
