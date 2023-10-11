import numpy as np
import jax.numpy as jnp
import optax
from picasso.utils import optimize


def test_optimize_linear_regression():
    np.random.seed(42)
    truth = jnp.array([1.0, 1.0])

    X = np.random.normal(0.0, 1, 100)
    Y = np.random.normal(truth[0] + truth[1] * X, 0.02)

    x = jnp.array(np.random.normal(X, 0.01))
    y = jnp.array(np.random.normal(Y, 0.01))

    def model_fn(x_, alpha, beta):
        return alpha + beta * x_

    def loss_fn(par):
        pred = model_fn(x, *par)
        return jnp.nanmean((y - pred) ** 2)

    chain = optimize(
        loss_fn,
        jnp.array([0.5, 1.5]),
        optimizer=optax.adam(learning_rate=1e-2),
        n_steps=1000,
        loss_tol=1e-6,
    )

    best_par = chain[-1, :-1]
    success = jnp.allclose(best_par, truth, rtol=1e-2)
    assert (
        success
    ), f"Recovered parameters not close to truth: {best_par} != {truth}"


if __name__ == "__main__":
    test_optimize_linear_regression()
