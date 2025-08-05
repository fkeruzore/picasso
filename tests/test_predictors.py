import jax
import jax.numpy as jnp
import os
import pytest

from picasso import predictors
from picasso.test_data import halos, profs


def test_flax_reg_mlp():
    # Define input and output dimensions
    X_DIM = 10
    Y_DIM = 5

    # Create an instance of FlaxRegMLP
    mlp = predictors.FlaxRegMLP(
        X_DIM=X_DIM,
        Y_DIM=Y_DIM,
        hidden_features=(16, 16),
        activations=["selu", "selu", "selu", "sigmoid"],
    )

    # Generate random input
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (1, X_DIM))

    default_net_par = mlp.init(rng, x[0])

    # Perform forward pass
    y = mlp.apply(default_net_par, x)

    # Check output shape
    assert y.shape == (1, Y_DIM)

    # Check output values are within valid range
    assert jnp.all(y >= 0.0) and jnp.all(y <= 1.0)


@pytest.mark.parametrize("transform", ["transforms", "no transforms"])
def test_predictor_conversion_and_io(transform):
    X_DIM, Y_DIM = 12, 8
    mlp = predictors.FlaxRegMLP(X_DIM, Y_DIM)
    net_par = mlp.init(jax.random.PRNGKey(66), jnp.empty(X_DIM))

    if transform:
        transform_x = "minmax"
        transform_y = "inv_minmax"
        args_transform_x = jnp.array([jnp.zeros(X_DIM), jnp.ones(X_DIM)])
        args_transform_y = jnp.array([jnp.zeros(Y_DIM), jnp.ones(Y_DIM)])
    else:
        transform_x = None
        transform_y = None
        args_transform_x = None
        args_transform_y = None

    pred = predictors.PicassoPredictor(
        mlp,
        transform_x=transform_x,
        transform_y=transform_y,
        args_transform_x=args_transform_x,
        args_transform_y=args_transform_y,
    )
    pred_t = predictors.PicassoTrainedPredictor.from_predictor(pred, net_par)
    filename = "./toto.hdf5"
    pred_t.save(filename)
    pred_t_rest = predictors.load(filename)

    # Single halo
    x = jnp.ones(X_DIM)
    y = pred.predict_model_parameters(x, net_par)
    y2 = pred_t.predict_model_parameters(x)
    y3 = pred_t_rest.predict_model_parameters(x)

    assert y.shape == (Y_DIM,), (
        f"Wrong shape for prediction: {y.shape} (should be {(Y_DIM,)})"
    )

    assert jnp.allclose(y, y2), (
        "Failed to recover same results when converting"
        + "`PicassoPredictor` to `PicassoTrainedPredictor`"
    )
    assert jnp.allclose(y, y3), (
        "Failed to recover same results after writing/loading"
        + "`PicassoTrainedPredictor`"
    )

    # Multi halo
    N = 5
    x = jnp.ones((N, X_DIM))
    y = pred.predict_model_parameters(x, net_par)
    y2 = pred_t.predict_model_parameters(x)
    y3 = pred_t_rest.predict_model_parameters(x)

    assert y.shape == (
        N,
        Y_DIM,
    ), f"Wrong shape for prediction: {y.shape} (should be {(N, Y_DIM)})"

    assert jnp.allclose(y, y2), (
        "Failed to recover same results when converting"
        + "`PicassoPredictor` to `PicassoTrainedPredictor`"
    )
    assert jnp.allclose(y, y3), (
        "Failed to recover same results after writing/loading"
        + "`PicassoTrainedPredictor`"
    )

    os.remove(filename)


@pytest.mark.parametrize("jit", ["jit", "nojit"])
def test_benchmark_predict_model_parameters(jit, benchmark):
    predictor = predictors.nonradiative_Gamma_r_576
    x = jnp.array([halos[k] for k in predictor.input_names]).T

    predict_func = predictor.predict_model_parameters
    if jit == "jit":  # jit the function and call it once to compile
        predict_func = jax.jit(predict_func)
        _ = predict_func(x)

    # benchmark function (jitted or not)
    _ = benchmark(predict_func, x)


@pytest.mark.parametrize("jit", ["jit", "nojit"])
def test_benchmark_predict_gas_model(jit, benchmark):
    predictor = predictors.nonradiative_Gamma_r_576
    x = jnp.array([halos[k] for k in predictor.input_names]).T

    predict_func = predictor.predict_gas_model
    if jit == "jit":  # jit the function and call it once to compile
        predict_func = jax.jit(predict_func)
        _ = predict_func(
            x,
            profs["phi"],
            profs["r_R500"],
            profs["r_R500"] / 2,
        )

    # benchmark function (jitted or not)
    _ = benchmark(
        predict_func,
        x,
        profs["phi"],
        profs["r_R500"],
        profs["r_R500"] / 2,
    )


@pytest.mark.parametrize(
    "predictor_name", predictors.available_predictors.keys()
)
def test_pretrained_predictors(predictor_name):
    predictor = predictors.available_predictors[predictor_name]
    x = jnp.array([halos[k] for k in predictor.input_names]).T
    n_halos = len(halos["log M200"])
    y_pred = predictor.predict_model_parameters(x)
    assert y_pred.shape == (n_halos, 8)
    assert jnp.all(jnp.isfinite(y_pred))

    n_r_bins = profs["r_R500"].shape[1]
    profs_pred = predictor.predict_gas_model(
        x,
        profs["phi"],
        profs["r_R500"],
        profs["r_R500"] / 2,
    )
    assert jnp.array(profs_pred).shape == (4, n_halos, n_r_bins)
    assert jnp.all(jnp.isfinite(jnp.array(profs_pred)))
