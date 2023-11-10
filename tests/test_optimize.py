import numpy as np
import jax.numpy as jnp
import optax
from picasso.utils import optimize


def _optimize_linear_regression(
    scatter=False,
    try_bfgs=True,
    return_chain=True,
    backup_optimizer=optax.adam(learning_rate=1e-2),
):
    np.random.seed(42)
    truth = jnp.array([1.0, 1.0])
    if scatter:
        xerr, yerr, sigma_int = 0.01, 0.01, 0.02
        loss_tol = np.sqrt(yerr**2 + sigma_int**2)
    else:
        xerr, yerr, sigma_int = 0.0, 0.0, 0.0
        loss_tol = 1e-8

    X = np.random.normal(0.0, 1, 100)
    Y = np.random.normal(truth[0] + truth[1] * X, sigma_int)

    x = jnp.array(np.random.normal(X, xerr))
    y = jnp.array(np.random.normal(Y, yerr))

    def model_fn(x_, alpha, beta):
        return alpha + beta * x_

    def loss_fn(par):
        pred = model_fn(x, *par)
        return jnp.nanmean((y - pred) ** 2)

    res = optimize(
        loss_fn,
        jnp.array([0.5, 1.5]),
        try_bfgs=try_bfgs,
        backup_optimizer=backup_optimizer,
        return_chain=return_chain,
    )
    return res, truth, loss_tol


def _assert_accurate(par, truth, rtol=1e-2):
    assert jnp.allclose(
        par, truth, rtol=rtol
    ), "Recovered parameters not close to truth: {par} != {truth}"


def _assert_converged(loss, loss_tol):
    assert (
        loss < loss_tol
    ), f"Target loss not reached: {loss:.3e} > {loss_tol:.3e}"


def test_optimize_linear_regression_noscatter():
    res_bfgs, truth, loss_tol = _optimize_linear_regression(
        try_bfgs=True, scatter=False, backup_optimizer=optax.adam(1e-3)
    )
    res_adam, truth, loss_tol = _optimize_linear_regression(
        try_bfgs=False, scatter=False, backup_optimizer=optax.adam(1e-3)
    )
    bf_bfgs, bl_bfgs = res_bfgs.bf, res_bfgs.bl
    bf_adam, bl_adam = res_adam.bf, res_adam.bl

    _assert_accurate(bf_bfgs, truth)
    _assert_accurate(bf_adam, truth)
    _assert_converged(bl_bfgs, loss_tol)
    _assert_converged(bl_adam, loss_tol)

    assert jnp.allclose(
        bf_bfgs, bf_adam, rtol=1e-2
    ), f"BFGS & adam find different parameters: {bf_bfgs=} != {bf_adam=}"
    assert jnp.allclose(
        bl_bfgs, bl_adam, rtol=1e-2
    ), f"BFGS & adam find different loss minimum: {bl_bfgs=} != {bl_adam=}"


def test_optimize_linear_regression_scatter():
    res, truth, loss_tol = _optimize_linear_regression(
        scatter=True,
        try_bfgs=True,
        backup_optimizer=optax.adam(learning_rate=1e-2),
        return_chain=True,
    )

    assert res.status != -1, "status=-1: unconverged optimization"
    _assert_accurate(res.bf, truth)
    _assert_converged(res.bl, loss_tol)


if __name__ == "__main__":
    test_optimize_linear_regression_noscatter()
    test_optimize_linear_regression_scatter()
