import jax.numpy as jnp
import pytest

from picasso import predictors
from picasso.utils import hacc
from picasso.test_data import hacc_halos, hacc_halos_inputs


@pytest.mark.parametrize("inputs", ["compact", "minimal"])
def test_build_input_dict(inputs):
    keys = {
        "compact": predictors.compact_576,
        "minimal": predictors.minimal_576,
    }[inputs].input_names

    hacc_halos_inputs_test = hacc.build_input_dict(hacc_halos, keys)
    allgood = [
        jnp.allclose(hacc_halos_inputs[k], hacc_halos_inputs_test[k])
        for k in keys
    ]
    assert jnp.all(jnp.array(allgood)), "Wrong derived halo properties"
