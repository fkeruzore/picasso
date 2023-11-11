import numpy as np
import jax.numpy as jnp
import optax
from picasso.utils import optimize
import pytest


def _optimize_linear_regression(
    scatter=False,
    try_bfgs=True,
    return_history=True,
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
        return_history=return_history,
    )
    return res, truth, loss_tol


def _assert_accurate(par, truth, rtol=1e-2):
    assert jnp.allclose(
        par, truth, rtol=rtol
    ), "Recovered parameters not close to truth: {par} != {truth}"


def _assert_converged(loss, loss_tol, status):
    assert status != 0, "status=0: gradient descent did not converge"
    assert (
        loss < loss_tol
    ), f"Target loss not reached: {loss:.3e} > {loss_tol:.3e}"


@pytest.mark.parametrize("scatter", [True, False])
def test_optimize_linear_regression(scatter):
    res, truth, loss_tol = _optimize_linear_regression(
        try_bfgs=True, scatter=scatter, backup_optimizer=optax.adam(1e-3)
    )

    _assert_accurate(res.bf, truth)
    _assert_converged(res.bl, loss_tol, res.status)


if __name__ == "__main__":
    test_optimize_linear_regression()
