import jax
import jax.numpy as jnp
import flax.linen as nn
import os

from jax import Array
from typing import Sequence, Callable, Iterable, Optional

from . import polytrop, nonthermal
from .utils import jax_utils, data_preparation


_avail_activ = {
    "selu": nn.selu,
    "relu": nn.relu,
    "tanh": nn.tanh,
    "sigmoid": nn.sigmoid,
    "clip": jnp.clip,
    "linear": lambda x: x,
}

_avail_trans = {
    "minmax": data_preparation.transform_minmax,
    "inv_minmax": data_preparation.inv_transform_minmax,
    "none": lambda x, *args: x,
}


class FlaxRegMLP(nn.Module):
    """
    A wrapper class around `flax.linen.Module` to generate fully
    connected multi-layer perceptrons, that can be instantiated easily
    with list of strings and integers for the layer sizes and activation
    functions.

    Parameters
    ----------
    X_DIM : int
        Number of features on the input layer.
    Y_DIM : int
        Number of features on the output layer.
    hidden_features : Sequence[int]
        Number of features for each hidden layer, defaults to [16, 16].
    activations : Sequence[str]
        Name of the activation functions to be used for each layer,
        including input and output. Accepted names are ["selu", "relu",
        "tanh", "sigmoid", "clip", "linear]. Defaults to ["selu",
        "selu", "selu", "linear"].

    See also
    --------
    :external+flax:doc:`flax.linen.Module <api_reference/flax.linen/module>`
    """

    X_DIM: int
    Y_DIM: int
    hidden_features: Sequence[int] = (16, 16)
    activations: Sequence[str] = ("selu", "selu", "selu", "linear")

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.X_DIM, name="input")(x)
        x = _avail_activ[self.activations[0]](x)
        for i, _features in enumerate(self.hidden_features):
            x = nn.Dense(_features, name=f"dense{i + 1}")(x)
            x = _avail_activ[self.activations[i + 1]](x)
        x = nn.Dense(self.Y_DIM, name="output")(x)
        x = _avail_activ[self.activations[-1]](x)
        return x


def _gas_par2gas_props_broken_plaw(gas_par, phi_tot, r_pol, r_fnt):
    rho_g, P_tot = polytrop.rho_P_g(phi_tot, r_pol, *gas_par[:5])
    f_nth = nonthermal.f_nt_generic(r_fnt, *gas_par[5:])
    P_th = P_tot * (1 - f_nth)
    return jnp.array([rho_g, P_tot, P_th, f_nth])


def _gas_par2gas_props_nelson(gas_par, phi_tot, r_pol, r_fnt):
    rho_g, P_tot = polytrop.rho_P_g(phi_tot, r_pol, *gas_par[:5])
    f_nth = nonthermal.f_nt_nelson14(r_fnt, *gas_par[5:])
    P_th = P_tot * (1 - f_nth)
    return jnp.array([rho_g, P_tot, P_th, f_nth])


_gas_par2gas_props = {
    "broken_plaw": _gas_par2gas_props_broken_plaw,
    "nelson14": _gas_par2gas_props_nelson,
}
_gas_par2gas_props_v = {
    k: jax.vmap(v, out_axes=1) for k, v in _gas_par2gas_props.items()
}


class PicassoPredictor:
    """
    A wrapper class to predict picasso model parameters and gas
    properties from an input halo properties vector and network
    parameters.

    Parameters
    ----------
    mlp : FlaxRegMLP
        Predictor for model parameters from halo properties.
    transform_x : Callable, optional
        Transformation to be applied to input vector,
        by default lambda x: x
    transform_y : Callable, optional
        Transformation to be applied to output vector,
        by default lambda y: y
    fix_params : dict, optional
        List and values of parameters to be fixed, formatted as
        {parameter name: fixed value}, by default {}
    f_nt_model : str, optional
        Non-thermal pressure fraction model to be used, one of
        ["broken_plaw", "Nelson14"], by default "broken_plaw"
    input_names : list[str], optional
        The names of input parameters. This is never used, only stored
        to be accessed by one to remind oneself of what the inputs are,
        bu default the inputs of the baseline model
    name : str, optional
        A name for the model, by default "model"
    """

    def __init__(
        self,
        mlp: FlaxRegMLP,
        transform_x: Optional[str] = None,
        transform_y: Optional[str] = None,
        args_transform_x: Optional[Array] = None,
        args_transform_y: Optional[Array] = None,
        fix_params: dict = {},
        f_nt_model: str = "broken_plaw",
        input_names: Iterable[str] = [
            "log M200",
            "c200",
            "cacc/c200",
            "cpeak/c200",
            "log dx/R200c",
            "e",
            "p",
            "a25",
            "a50",
            "a75",
            "almm",
            "mdot",
        ],
        name="model",
    ):
        self.mlp = mlp

        if transform_x is None:
            transform_x = "none"
        self._transform_x = transform_x
        self.args_transform_x = args_transform_x

        if transform_y is None:
            transform_y = "none"
        self._transform_y = transform_y
        self.args_transform_y = args_transform_y

        self.name = name
        self.input_names = input_names
        self.param_indices = {
            "log10 rho_0": "0",
            "log10 P_0": "1",
            "Gamma_0": "2",
            "c_Gamma": "3",
            "theta_0": "4",
            "log10 A_nt": "5",
            "log10 B_nt": "6",
            "C_nt": "7",
        }
        self.fix_params = fix_params
        self._fix_params = {}
        for k, v in fix_params.items():
            self._fix_params[self.param_indices[k]] = jnp.array(v)

        self.f_nt_model = f_nt_model
        self._gas_par2gas_props = _gas_par2gas_props[f_nt_model]
        self._gas_par2gas_props_v = _gas_par2gas_props_v[f_nt_model]

    def transform_x(self, x: Array) -> Array:
        return _avail_trans[self._transform_x](x, self.args_transform_x)

    def transform_y(self, y: Array) -> Array:
        # First make the output the right shape to be able to apply the
        # y scaling regardless of fixed parameters
        for k in self._fix_params.keys():
            y = jnp.insert(y, int(k), 0.0, axis=-1)
        # Apply the y scaling
        y_out = _avail_trans[self._transform_y](y, self.args_transform_y)
        # Fix parameters that need to be fixed
        for k, v in self._fix_params.items():
            y_out = y_out.at[..., int(k)].set(v)
        return y_out

    def predict_model_parameters(self, x: Array, net_par: dict) -> Array:
        """
        Predicts the gas model parameters based on halo properties.

        Parameters
        ----------
        x : Array
            Halo properties.
        net_par: dict
            Parameters of the MLP to be used for the prediction.

        Returns
        -------
        Array
            Gas model parameters.
        """

        x_ = self.transform_x(x)
        y_ = self.mlp.apply(net_par, x_)
        return self.transform_y(y_)

    def predict_gas_model(
        self, x: Array, phi: Array, r_pol: Array, r_fnt: Array, net_par: dict
    ) -> Sequence[Array]:
        """
        Predicts the gas properties from halo properties ant potential
        values.

        Parameters
        ----------
        x : Array
            Halo properties.
        phi : Array
            Potential values.
        r_pol : Array
            Normalized radii to be used for the polytropic model.
        r_fnt : Array
            Normalized radii to be used for the non-thermal pressure
            fraction model.
        net_par: dict
            Parameters of the MLP to be used for the prediction.

        Returns
        -------
        Sequence[Array]
            A sequence of arrays containing the predicted gas model
            parameters:

            - rho_g : Array
                The predicted gas density.
            - P_tot : Array
                The predicted total pressure.
            - P_th : Array
                The predicted thermal pressure.
            - f_nth : Array
                The predicted non-thermal pressure fraction.
        """
        gas_par = self.predict_model_parameters(x, net_par)
        # Careful with the log scales!
        gas_par = gas_par.at[..., 0].set(10 ** gas_par[..., 0])
        gas_par = gas_par.at[..., 1].set(10 ** gas_par[..., 1])
        gas_par = gas_par.at[..., 4].set(1e-6 * gas_par[..., 4])
        gas_par = gas_par.at[..., 5].set(10 ** gas_par[..., 5])
        gas_par = gas_par.at[..., 6].set(10 ** gas_par[..., 6])

        if len(gas_par.shape) == 1:
            rho_g, P_tot, P_th, f_nth = self._gas_par2gas_props(
                gas_par, phi, r_pol, r_fnt
            )
        else:
            rho_g, P_tot, P_th, f_nth = self._gas_par2gas_props_v(
                gas_par, phi, r_pol, r_fnt
            )
        return (rho_g, P_tot, P_th, f_nth)

    def save(self, filename):
        """
        Serializes the object to a Pytree and saves it to disk in
        HDF5 format.

        Parameters
        ----------
        filename : str
            File to save the model to.
        """

        tree = {
            "X_DIM": self.mlp.X_DIM,
            "Y_DIM": self.mlp.Y_DIM,
            "hidden_features": self.mlp.hidden_features,
            "activations": self.mlp.activations,
            "transform_x": self._transform_x,
            "transform_y": self._transform_y,
            "args_transform_x": self.args_transform_x,
            "args_transform_y": self.args_transform_y,
            "fix_params": self.fix_params,
            "f_nt_model": self.f_nt_model,
            "input_names": self.input_names,
            "name": self.name,
        }
        if hasattr(self, "net_par"):
            tree["net_par"] = self.net_par
        jax_utils.save_pytree_h5(tree, filename)


class PicassoTrainedPredictor(PicassoPredictor):
    """
    A wrapper class to predict picasso model parameters and gas
    properties from an input halo properties vector, with a fixed set
    of network parameters.

    Parameters
    ----------
    net_par: dict
        Parameters of the MLP to be used for the predictions.
    mlp : FlaxRegMLP
        Predictor for model parameters from halo properties.
    transform_x : Callable, optional
        Transformation to be applied to input vector,
        by default lambda x: x
    transform_y : Callable, optional
        Transformation to be applied to output vector,
        by default lambda y: y
    fix_params : dict, optional
        List and values of parameters to be fixed, formatted as
        {parameter name: fixed value}, by default {}
    f_nt_model : str, optional
        Non-thermal pressure fraction model to be used, one of
        ["broken_plaw", "Nelson14"], by default "broken_plaw"
    """

    def __init__(self, net_par: dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net_par = net_par

    @classmethod
    def from_predictor(cls, predictor: PicassoPredictor, net_par: dict):
        """
        Instantiate a trained predictor from an untrained predictor and
        network parameters.

        Parameters
        ----------
        predictor : PicassoPredictor
            Untrained predictor.
        net_par : dict
            Trained network parameters.

        Returns
        -------
        PicassoTrainedPredictor
            The predictor with pre-trained parameters.
        """
        trained_predictor = cls.__new__(cls)
        trained_predictor.__dict__.update(predictor.__dict__)
        trained_predictor.net_par = net_par
        return trained_predictor

    def predict_gas_model(
        self, x: Array, phi: Array, r_pol: Array, r_fnt: Array, *args
    ) -> Sequence[Array]:
        """
        Predicts the gas properties from halo properties ant potential
        values.

        Parameters
        ----------
        x : Array
            Halo properties.
        phi : Array
            Potential values.
        r_pol : Array
            Normalized radii to be used for the polytropic model.
        r_fnt : Array
            Normalized radii to be used for the non-thermal pressure
            fraction model.

        Returns
        -------
        Sequence[Array]
            A sequence of arrays containing the predicted gas model
            parameters:

            - rho_g : Array
                The predicted gas density.
            - P_tot : Array
                The predicted total pressure.
            - P_th : Array
                The predicted thermal pressure.
            - f_nth : Array
                The predicted non-thermal pressure fraction.
        """
        return super().predict_gas_model(x, phi, r_pol, r_fnt, self.net_par)

    def predict_model_parameters(self, x: Array, *args) -> Array:
        """
        Predicts the gas model parameters based on halo properties.

        Parameters
        ----------
        x : Array
            Halo properties.

        Returns
        -------
        Array
            Gas model parameters.
        """
        return super().predict_model_parameters(x, self.net_par)


def load(filename):
    """
    Reads a model object from disk.

    Parameters
    ----------
    filename : str
        File to read the saved model from.

    Returns
    -------
    PicassoPredictor
        The saved model.
    """
    tree = jax_utils.load_pytree_h5(filename)
    mlp = FlaxRegMLP(
        tree["X_DIM"],
        tree["Y_DIM"],
        tree["hidden_features"],
        tree["activations"],
    )
    kwargs = {
        "transform_x": tree["transform_x"],
        "transform_y": tree["transform_y"],
        "args_transform_x": tree["args_transform_x"],
        "args_transform_y": tree["args_transform_y"],
        "fix_params": tree["fix_params"],
        "f_nt_model": tree["f_nt_model"],
        "input_names": tree["input_names"],
        "name": tree["name"],
    }
    predictor = PicassoPredictor(mlp, **kwargs)
    if "net_par" in tree.keys():
        return PicassoTrainedPredictor.from_predictor(
            predictor, tree["net_par"]
        )
    return predictor


def draw_mlp(mlp: FlaxRegMLP, colors=["k", "w"], alpha_line=1.0):
    import matplotlib.pyplot as plt

    key = jax.random.PRNGKey(0)
    par = mlp.init(key, jnp.zeros((1, mlp.X_DIM)))["params"]
    layer_sizes = [layer["bias"].size for layer in par.values()]
    top, bottom, left, right = 0.95, 0.05, 0.05, 0.95
    v_spacing = (top - bottom) / float(max(layer_sizes))
    h_spacing = (right - left) / float(len(layer_sizes) - 1)

    fig, ax = plt.subplots(figsize=(4, 3))
    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing * (layer_size - 1) / 2.0 + (top + bottom) / 2.0
        for m in range(layer_size):
            circle = plt.Circle(
                (n * h_spacing + left, layer_top - m * v_spacing),
                v_spacing / 4.0,
                color=colors[1],
                ec=colors[0],
                zorder=4,
            )
            ax.add_artist(circle)
    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(
        zip(layer_sizes[:-1], layer_sizes[1:])
    ):
        layer_top_a = (
            v_spacing * (layer_size_a - 1) / 2.0 + (top + bottom) / 2.0
        )
        layer_top_b = (
            v_spacing * (layer_size_b - 1) / 2.0 + (top + bottom) / 2.0
        )
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D(
                    [n * h_spacing + left, (n + 1) * h_spacing + left],
                    [
                        layer_top_a - m * v_spacing,
                        layer_top_b - o * v_spacing,
                    ],
                    c=colors[0],
                    alpha=alpha_line,
                )
                ax.add_artist(line)

    ax.set_xticks([])
    ax.set_yticks([])
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig, ax


_path = f"{os.path.dirname(os.path.abspath(__file__))}/trained_models"

available_predictors = [
    baseline_576 := load(f"{_path}/576/baseline.hdf5"),
    compact_576 := load(f"{_path}/576/compact.hdf5"),
    minimal_576 := load(f"{_path}/576/minimal.hdf5"),
    subgrid_576 := load(f"{_path}/576/subgrid.hdf5"),
    nonradiative_Gamma_r_576 := load(f"{_path}/576/nonradiative_Gamma_r.hdf5"),
    subgrid_Gamma_r_576 := load(f"{_path}/576/subgrid_Gamma_r.hdf5"),
]
