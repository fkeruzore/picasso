import numpy as np
import jax.numpy as jnp
import optax
from picasso.utils.fitting import optimize
import pytest


def _optimize_linear_regression(
    scatter=False,
    try_bfgs=True,
    bounds=None,
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
        bounds=bounds,
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


def _assert_in_bounds(par, bounds):
    for i, (p, b) in enumerate(zip(par, bounds)):
        b_low = b[0] if b[0] is not None else -np.inf
        b_upp = b[1] if b[1] is not None else +np.inf
        assert (p >= b_low) and (
            p <= b_upp
        ), f"param {i} ({p}) is not within bounds ({b_low}, {b_upp})"


@pytest.mark.parametrize(
    ["scatter", "bounds"],
    [
        (False, None),
        (False, [(0.0, 2.0), (0.0, 2.0)]),
        (False, [(0.0, None), (0.0, None)]),
        (True, None),
    ],
)
def test_optimize_linear_regression(scatter, bounds):
    res, truth, loss_tol = _optimize_linear_regression(
        try_bfgs=True,
        scatter=scatter,
        bounds=bounds,
        backup_optimizer=optax.adam(1e-3),
    )

    _assert_accurate(res.bf, truth)
    _assert_converged(res.bl, loss_tol, res.status)
    if bounds is not None:
        _assert_in_bounds(res.bf, bounds)
