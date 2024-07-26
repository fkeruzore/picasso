import jax
import jax.numpy as jnp
from picasso import predictors, utils
import pickle
from functools import partial
import os
import pytest

here = os.path.dirname(os.path.abspath(__file__))
path = f"{here}/data"
with open(f"{path}/test_data_predictor.pkl", "rb") as f:
    data_predictor = pickle.load(f)


def test_flax_reg_mlp():
    # Define input and output dimensions
    X_DIM = 10
    Y_DIM = 5

    # Create an instance of FlaxRegMLP
    mlp = predictors.FlaxRegMLP(
        X_DIM=X_DIM,
        Y_DIM=Y_DIM,
        hidden_features=(16, 16),
        activations=["selu", "selu", "selu", "soft_clip"],
        extra_args_output_activation=[jnp.zeros(Y_DIM), jnp.ones(Y_DIM)],
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
        minmax_x = jnp.array([jnp.zeros(X_DIM), jnp.ones(X_DIM)])
        minmax_y = jnp.array([jnp.zeros(Y_DIM), jnp.ones(Y_DIM)])
        transfom_x = partial(
            utils.transform_minmax, mins=minmax_x[0], maxs=minmax_x[1]
        )
        transfom_y = partial(
            utils.inv_transform_minmax, mins=minmax_y[0], maxs=minmax_y[1]
        )
    else:  # This violates Flake8(E731), but this is what I want to do
        transform_x = lambda x: x  # noqa: E731, F841
        transform_y = lambda y: y  # noqa: E731, F841

    pred = predictors.PicassoPredictor(
        mlp, transfom_x=transfom_x, transfom_y=transfom_y
    )
    x = jnp.ones(X_DIM)
    y = pred.predict_model_parameters(x, net_par)

    tpred = predictors.PicassoTrainedPredictor.from_predictor(pred, net_par)
    y2 = tpred.predict_model_parameters(x)
    assert jnp.allclose(y, y2), (
        "Failed to recover same results when converting"
        + "`PicassoPredictor` to `PicassoTrainedPredictor`"
    )

    tpred.save("./toto.pkl")
    tpred_2 = predictors.PicassoTrainedPredictor.load("./toto.pkl")
    y3 = tpred_2.predict_model_parameters(x)
    assert jnp.allclose(y, y3), (
        "Failed to recover same results after writing/loading"
        + "`PicassoTrainedPredictor`"
    )

    os.remove("./toto.pkl")


def test_picasso_predictor():
    # Number of (halos, points) for which to predict
    n_halos = 2
    n_pts = 5

    # Define input and output dimensions
    X_DIM = 10
    Y_DIM = 8

    # Create an instance of FlaxRegMLP
    mlp = predictors.FlaxRegMLP(
        X_DIM=X_DIM,
        Y_DIM=Y_DIM,
        hidden_features=(16, 16),
        activations=["selu", "selu", "selu", "soft_clip"],
        extra_args_output_activation=[jnp.zeros(Y_DIM), jnp.ones(Y_DIM)],
    )

    # Generate random input
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (n_halos, X_DIM))

    default_net_par = mlp.init(rng, x[0])

    # Create an instance of PicassoPredictor
    predictor = predictors.PicassoTrainedPredictor(mlp, default_net_par)

    # Test predict_model_parameters method
    y_pred = predictor.predict_model_parameters(x)
    assert y_pred.shape == (n_halos, Y_DIM)

    # Test predict_gas_model method
    phi = jnp.ones((n_halos, n_pts))
    r_R500 = jnp.ones((n_halos, n_pts))
    gas_model = predictor.predict_gas_model(x, phi, r_R500)
    assert len(gas_model) == 4
    assert gas_model[0].shape == (n_halos, n_pts)
    assert gas_model[1].shape == (n_halos, n_pts)
    assert gas_model[2].shape == (n_halos, n_pts)


@pytest.mark.parametrize("jit", ["jit", "nojit"])
def test_predict_model_params_pretrained(jit, benchmark):

    predict_func = predictors.net1.predict_model_parameters
    if jit == "jit":  # jit the function and call it once to compile
        predict_func = jax.jit(predict_func)
        _ = predict_func(data_predictor["x"][0])

    # benchmark function (jitted or not)
    y_pred = benchmark(predict_func, data_predictor["x"])
    assert y_pred.shape == (data_predictor["x"].shape[0], 7)
    assert jnp.all(jnp.isfinite(y_pred))


@pytest.mark.parametrize("jit", ["jit", "nojit"])
def test_predict_gas_properties_pretrained(jit, benchmark):
    predict_func = predictors.net1.predict_gas_model
    if jit == "jit":  # jit the function and call it once to compile
        predict_func = jax.jit(predict_func)
        _ = predict_func(
            data_predictor["x"][0],
            data_predictor["phi"][0],
            data_predictor["r_R500"][0],
        )

    # benchmark function (jitted or not)
    profs_pred = benchmark(
        predict_func,
        data_predictor["x"],
        data_predictor["phi"],
        data_predictor["r_R500"],
    )
    profs_pred = jnp.array(profs_pred)
    assert profs_pred.shape == (
        4,
        data_predictor["x"].shape[0],
        data_predictor["phi"].shape[1],
    )
    assert jnp.all(jnp.isfinite(profs_pred))
