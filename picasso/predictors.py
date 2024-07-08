import jax
import jax.numpy as jnp
import flax.linen as nn
import jax.scipy.stats as jss
import pickle
import os

from jax import Array
from typing import Sequence, Callable, Iterable
from functools import partial

from . import polytrop, nonthermal


def soft_clip(x, a, b, c=1.0):
    return (
        x
        - jnp.log(1 + jnp.exp(c * (x - b))) / c
        + jnp.log(1 + jnp.exp(-c * (x - a))) / c
    )


possible_activations = {
    "selu": nn.selu,
    "relu": nn.relu,
    "tanh": nn.tanh,
    "sigmoid": nn.sigmoid,
    "clip": jnp.clip,
    "soft_clip": soft_clip,
    "linear": lambda x: x,
}


def transform_minmax(x: Array, mins: Array, maxs: Array):
    return (x - mins) / (maxs - mins)


def inv_transform_minmax(x: Array, mins: Array, maxs: Array):
    return x * (maxs - mins) + mins


def transform_y(y: Array):
    y_trans = 10**y
    y_trans = y_trans.at[..., 2].add(1.0)
    y_trans = y_trans.at[..., 3].multiply(1e-6)
    return y_trans


def quantile_normalization(x: Array, dist=jss.norm):
    ranks = jnp.argsort(x, axis=0)
    sorted_ranks = jnp.argsort(ranks, axis=0)
    normalized = dist.ppf((sorted_ranks + 0.5) / x.shape[0])
    return normalized


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
    def __init__(
        self,
        mlp: FlaxRegMLP,
        transfom_x: Callable = lambda x: x,
        transfom_y: Callable = lambda y: y,
        fix_params: dict = {},
        f_nt_model: str = "broken_plaw",
    ):
        self.mlp = mlp
        self._transfom_x = transfom_x
        self._transfom_y = transfom_y
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
        y_out = self._transfom_y(y)
        for k, v in self.fix_params.items():
            y_out = jnp.insert(y_out, k, v, axis=-1)
        return y_out

    def predict_model_parameters(self, x: Array, net_par: dict) -> Array:
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
        r_R500 : Array
            Radii in units of R500c.

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


class PicassoTrainedPredictor(PicassoPredictor):
    def __init__(
        self,
        mlp: FlaxRegMLP,
        net_par: dict,
        transfom_x: Callable = lambda x: x,
        transfom_y: Callable = lambda y: y,
        fix_params: dict = {},
        f_nt_model: str = "nelson14",
    ):
        super().__init__(mlp, transfom_x, transfom_y, fix_params, f_nt_model)
        self.net_par = net_par

    def predict_gas_model(
        self, x: Array, phi: Array, r_pol: Array, r_fnt: Array, *args
    ) -> Sequence[Array]:
        return super().predict_gas_model(x, phi, r_pol, r_fnt, self.net_par)

    def predict_model_parameters(self, x: Array, *args) -> Array:
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


_here = os.path.dirname(os.path.abspath(__file__))


def load_trained_net(pkl_file: str):
    with open(f"{_here}/trained_networks/{pkl_file}", "rb") as f:
        _params = pickle.load(f)
    if not ("f_nt_model" in _params):
        _params["f_nt_model"] = "nelson14"

    return PicassoTrainedPredictor(
        FlaxRegMLP(
            _params["X_DIM"],
            _params["Y_DIM"],
            _params["hidden_features"],
            _params["activations"],
            _params["extra_args_output_activation"],
        ),
        _params["net_par"],
        transfom_x=partial(
            transform_minmax,
            mins=_params["minmax_x"][0],
            maxs=_params["minmax_x"][1],
        ),
        transfom_y=transform_y,
        f_nt_model=_params["f_nt_model"],
    )


net12 = load_trained_net("net12.pkl")

net1 = load_trained_net("net1.pkl")
