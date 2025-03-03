import jax.numpy as jnp
import h5py
import os

_path = f"{os.path.dirname(os.path.abspath(__file__))}/data"
with h5py.File(f"{_path}/halos.hdf5", "r") as f:
    halos = {
        k.replace("_ov_", "/"): jnp.array(v) for k, v in f["halos"].items()
    }
    profs = {k: jnp.array(v) for k, v in f["profs"].items()}

hacc_halos = {k: v for k, v in jnp.load(f"{_path}/halos_576.npz").items()}
hacc_halos_inputs = {
    k: v for k, v in jnp.load(f"{_path}/inputs_compact_576.npz").items()
}
