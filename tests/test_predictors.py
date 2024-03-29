import jax
import jax.numpy as jnp
from picasso.predictors import FlaxRegMLP


def test_flax_reg_mlp():
    # Define input and output dimensions
    X_DIM = 10
    Y_DIM = 5

    # Create an instance of FlaxRegMLP
    mlp = FlaxRegMLP(
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
