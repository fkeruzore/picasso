import numpy as np
import jax.numpy as jnp
from jax import Array
import optax
from picasso.utils import optimize


def optimize_linear_regression(
    scatter=False,
    try_bfgs=True,
    return_chain=True,
    optimizer=optax.adam(learning_rate=1e-2),
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
        optimizer=optimizer,
        n_steps=1000,
        return_chain=return_chain,
        loss_tol=loss_tol,
        dloss_tol=1e-8,
    )
    return best_par, best_loss, status, chain, truth, loss_tol


def test_optimize_linear_regression_noscatter():
    for return_chain in [True, False]:
        for try_bfgs in [True, False]:
            res = optimize_linear_regression(
                try_bfgs=try_bfgs,
                return_chain=return_chain,
                scatter=False,
                optimizer=optax.adam(learning_rate=1e-2),
            )
            best_par, best_loss, status, chain, truth, loss_tol = res

            print(f"{return_chain=}, {try_bfgs=}")
            assert status != -1, "status=-1: unconverged optimization"
            assert best_loss < loss_tol, "loss > loss_tol"
            assert jnp.allclose(best_par, truth, rtol=1e-2), (
                "Recovered parameters not close to truth: "
                + f"{best_par} != {truth}"
            )
            if return_chain:
                assert isinstance(
                    chain, Array
                ), "Returned chain is not an array"
            else:
                assert chain is None, "Returned chain is not None"


def test_optimize_linear_regression_scatter():
    res = optimize_linear_regression(
        scatter=True,
        try_bfgs=True,
        optimizer=optax.adam(learning_rate=1e-2),
        return_chain=True,
    )
    best_par, best_loss, status, chain, truth, loss_tol = res

    assert status != -1, "status=-1: unconverged optimization"
    assert best_loss < loss_tol, "loss > loss_tol"
    assert jnp.allclose(
        best_par, truth, rtol=1e-2
    ), f"Recovered parameters not close to truth: {best_par} != {truth}"


if __name__ == "__main__":
    test_optimize_linear_regression_noscatter()
    test_optimize_linear_regression_scatter()
