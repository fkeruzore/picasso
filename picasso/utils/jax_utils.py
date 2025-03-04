import jax.numpy as jnp
import numpy as np
import h5py


def save_pytree_h5(pytree, filename):
    """Save a JAX pytree to an HDF5 file."""

    # Helper function to save nested structures
    def save_node(group, key, value):
        if isinstance(value, dict):
            # Create a subgroup for dict
            subgroup = group.create_group(key)
            for k, v in value.items():
                save_node(subgroup, k, v)
        elif isinstance(value, jnp.ndarray):
            # Save array directly
            group.create_dataset(key, data=np.array(value))
        else:
            # Save other types as attributes
            group.attrs[key] = value

    with h5py.File(filename, "w") as f:
        for k, v in pytree.items():
            save_node(f, k, v)


def load_pytree_h5(filename):
    """Load a JAX pytree from an HDF5 file."""

    # Helper function to load nested structures
    def load_node(group):
        result = {}

        # Load datasets (arrays)
        for k in group.keys():
            if isinstance(group[k], h5py.Group):
                result[k] = load_node(group[k])
            else:
                result[k] = jnp.array(group[k][()])

        # Load attributes (non-array values)
        for k in group.attrs:
            result[k] = group.attrs[k]

        return result

    with h5py.File(filename, "r") as f:
        return load_node(f)
