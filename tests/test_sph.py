import jax
import jax.numpy as jnp
from jax.scipy.integrate import trapezoid
import pytest

from picasso.utils import sph


@pytest.mark.parametrize("kernel", ["cubic_spline", "wendland"])
def test_kernel_integral(kernel):
    w = sph.sph_kernels[kernel](x := jnp.linspace(0.0, 1.55, 10_000), 1.5)
    integ = 4 * jnp.pi * trapezoid(w * x**2, x)
    assert jnp.isclose(
        integ, 1.0, rtol=1e-3
    ), f"Kernel `{kernel}` does not integrate to 1: {integ}"


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

    h, V, r, j = sph.sph_radii_volumes(
        xyz,
        n=32,
        kernel="wendland",
        kind="kdtree",
        return_dists_and_indices=True,
    )
