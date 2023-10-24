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

    best_par, best_loss, status, chain = optimize(
        loss_fn,
        jnp.array([0.5, 1.5]),
        try_bfgs=try_bfgs,
        backup_optimizer=backup_optimizer,
        return_chain=return_chain,
    )
    return best_par, best_loss, status, chain, truth, loss_tol


def _assert_accurate(par, truth, rtol=1e-2):
    assert jnp.allclose(
        par, truth, rtol=rtol
    ), "Recovered parameters not close to truth: {par} != {truth}"


def _assert_converged(loss, loss_tol):
    assert (
        loss < loss_tol
    ), f"Target loss not reached: {loss:.3e} > {loss_tol:.3e}"


def test_optimize_linear_regression_noscatter_bfgs():
    res = _optimize_linear_regression(
        try_bfgs=True,
        scatter=False,
    )
    best_par, best_loss, status, chain, truth, loss_tol = res

    assert status == 1, f"{status=} != 1: BFGS did not converge"
    _assert_accurate(best_par, truth)
    _assert_converged(best_loss, loss_tol)


def test_optimize_linear_regression_noscatter_adam():
    res = _optimize_linear_regression(
        try_bfgs=False, scatter=False, backup_optimizer=optax.adam(1e-3)
    )
    best_par, best_loss, status, chain, truth, loss_tol = res

    assert status == 2, f"{status=} != 2: adam did not converge"
    _assert_accurate(best_par, truth)
    _assert_converged(best_loss, loss_tol)


def test_optimize_linear_regression_noscatter_bfgs_vs_adam():
    res_bfgs = _optimize_linear_regression(
        try_bfgs=False, scatter=False, backup_optimizer=optax.adam(1e-3)
    )
    par_bfgs, loss_bfgs, _, _, _, _ = res_bfgs
    res_adam = _optimize_linear_regression(
        try_bfgs=False, scatter=False, backup_optimizer=optax.adam(1e-3)
    )
    par_adam, loss_adam, _, _, _, _ = res_adam

    assert jnp.allclose(
        par_bfgs, par_adam
    ), f"BFGS & adam find different parameters: {par_bfgs=} != {par_adam=}"
    assert jnp.allclose(
        loss_bfgs, loss_adam
    ), f"BFGS & adam find different loss minimum: {loss_bfgs=} != {loss_adam=}"


def test_optimize_linear_regression_scatter():
    res = _optimize_linear_regression(
        scatter=True,
        try_bfgs=True,
        backup_optimizer=optax.adam(learning_rate=1e-2),
        return_chain=True,
    )
    best_par, best_loss, status, chain, truth, loss_tol = res

    assert status != -1, "status=-1: unconverged optimization"
    _assert_accurate(best_par, truth)
    _assert_converged(best_loss, loss_tol)


if __name__ == "__main__":
    test_optimize_linear_regression_noscatter_bfgs()
    test_optimize_linear_regression_noscatter_adam()
    test_optimize_linear_regression_noscatter_bfgs_vs_adam()
    test_optimize_linear_regression_scatter()
