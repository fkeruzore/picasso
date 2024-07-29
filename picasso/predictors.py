import jax
import jax.numpy as jnp
import flax.linen as nn
import os
import dill

from jax import Array
from typing import Sequence, Callable, Iterable

from . import polytrop, nonthermal


possible_activations = {
    "selu": nn.selu,
    "relu": nn.relu,
    "tanh": nn.tanh,
    "sigmoid": nn.sigmoid,
    "clip": jnp.clip,
    "linear": lambda x: x,
}


class FlaxRegMLP(nn.Module):
    X_DIM: int
    Y_DIM: int
    hidden_features: Sequence[int] = (16, 16)
    activations: Sequence[str] = ("selu", "selu", "selu", "linear")
    extra_args_output_activation: Iterable[Array] = ()

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.X_DIM, name="input")(x)
        x = possible_activations[self.activations[0]](x)
        for i, _features in enumerate(self.hidden_features):
            x = nn.Dense(_features, name=f"dense{i + 1}")(x)
            x = possible_activations[self.activations[i + 1]](x)
        x = nn.Dense(self.Y_DIM, name="output")(x)
        x = possible_activations[self.activations[-1]](
            x, *self.extra_args_output_activation
        )
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
    transfom_x : Callable, optional
        Transformation to be applied to input vector,
        by default lambda x: x
    transfom_y : Callable, optional
        Transformation to be applied to output vector,
        by default lambda y: y
    fix_params : dict, optional
        List and values of parameters to be fixed, formatted as
        {parameter name: fixed value}, by default {}
    f_nt_model : str, optional
        Non-thermal pressure fraction model to be used, one of
        ["broken_plaw", "Nelson14"], by default "broken_plaw"
    input_params : list[str], optional
        The names of input parameters. This is never used, only stored
        to be accessed by one to remind oneself of what the inputs are,
        bu default the inputs of the baseline model
    name : str, optional
        A name for the model, by default "model"
    """

    def __init__(
        self,
        mlp: FlaxRegMLP,
        transfom_x: Callable = lambda x: x,
        transfom_y: Callable = lambda y: y,
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
        self._transfom_x = transfom_x
        self._transfom_y = transfom_y
        self.name = name
        self.input_names = input_names
        self.param_indices = {
            "rho_0": 0,
            "P_0": 1,
            "Gamma_0": 2,
            "c_Gamma": 3,
            "theta_0": 4,
            "A_nt": 5,
            "B_nt": 6,
            "C_nt": 7,
        }
        self.fix_params = {}
        for k, v in fix_params.items():
            self.fix_params[self.param_indices[k]] = jnp.array(v)
        self._gas_par2gas_props = _gas_par2gas_props[f_nt_model]
        self._gas_par2gas_props_v = _gas_par2gas_props_v[f_nt_model]

    def transfom_x(self, x: Array) -> Array:
        return self._transfom_x(x)

    def transfom_y(self, y: Array) -> Array:
        # First make the output the right shape to be able to apply the
        # y scaling regardless of fixed parameters
        for k in self.fix_params.keys():
            y = jnp.insert(y, k, 0.0, axis=-1)
        # Apply the y scaling
        y_out = self._transfom_y(y)
        # Fix parameters that need to be fixed
        for k, v in self.fix_params.items():
            y_out = y_out.at[..., k].set(v)
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

        x_ = self.transfom_x(x)
        y_ = self.mlp.apply(net_par, x_)
        return self.transfom_y(y_)

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
        Serializes the object using `dill` and saves it to disk.

        Parameters
        ----------
        filename : str
            File to save the model to.
        """
        with open(filename, "wb") as f:
            f.write(dill.dumps(self))

    @classmethod
    def load(cls, filename):
        """
        Reads a model object from disk using `dill`.

        Parameters
        ----------
        filename : str
            File to read the saved model from.

        Returns
        -------
        PicassoPredictor
            The saved model.
        """
        with open(filename, "rb") as f:
            inst = dill.load(f)
        return inst


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
    transfom_x : Callable, optional
        Transformation to be applied to input vector,
        by default lambda x: x
    transfom_y : Callable, optional
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
        return super().predict_gas_model(x, phi, r_pol, r_fnt, self.net_par)

    def predict_model_parameters(self, x: Array, *args) -> Array:
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
        return super().predict_model_parameters(x, self.net_par)


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
_load = PicassoTrainedPredictor.load

baseline_576 = _load(f"{_path}/576/baseline.pkl")
minimal_576 = _load(f"{_path}/576/minimal.pkl")
subgrid_576 = _load(f"{_path}/576/subgrid.pkl")
adiabatic_Gamma_r_576 = _load(f"{_path}/576/adiabatic_Gamma_r.pkl")
subgrid_Gamma_r_576 = _load(f"{_path}/576/subgrid_Gamma_r.pkl")
