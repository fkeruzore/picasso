from jax import Array
import jax.numpy as jnp
from typing import Tuple


def theta(phi: Array, theta_0: Array) -> Array:
    """
    Re-parametrized polytropic variable.

    Parameters
    ----------
    phi : Array
        Normalized isolated gravitational potential (see Notes)
    theta_0 : Array
        Potential prefactor (see Notes)

    Returns
    -------
    Array
        Polytropic variable for each phi
    """
    t = 1 - (theta_0 * phi)
    t = jnp.where(t >= 0, t, 0)
    return t


def _Gamma_r_pos(r_norm, Gamma_0, c_Gamma):
    return 1 + (Gamma_0 - 1) * (1 / (1 + jnp.exp(-r_norm / c_Gamma)))


def _Gamma_r_neg(r_norm, Gamma_0, c_Gamma):
    return Gamma_0 + (Gamma_0 - 1) * (
        1 - (1 / (1 + jnp.exp(r_norm / c_Gamma)))
    )


def Gamma_r(r_norm: Array, Gamma_0: Array, c_Gamma: Array):
    """
    Compute the radius-dependent polytropic index Gamma(r)

    Parameters
    ----------
    r_norm : Array
        Normalized radii
    Gamma_0 : Array
        Asymptotic outer value of the polytropic index
    c_Gamma : Array
        Polytropic concentration value

    Returns
    -------
    Array
        Gamma values at specified radii
    """
    # Note that jnp.where or jax.lax.select cannot be used naively here
    # because # c_Gamma == 0 can create NaNs in one of the branches,
    # which would propagate to the *gradients* even if not called;
    # see jax.numpy.where documentation.

    # Gamma_0 everywhere
    g = jnp.ones_like(r_norm) * Gamma_0
    # Gamma_r where c_Gamma > 0, rubbish elsewhere
    gp = _Gamma_r_pos(r_norm, Gamma_0, jnp.where(c_Gamma > 0, c_Gamma, 1.0))
    # Gamma_r where c_Gamma < 0, rubbish elsewhere
    gn = _Gamma_r_neg(r_norm, Gamma_0, jnp.where(c_Gamma < 0, c_Gamma, -1.0))

    g = jnp.where(c_Gamma > 0, gp, g)
    g = jnp.where(c_Gamma < 0, gn, g)

    return g


def P_g(
    phi: Array,
    r_norm: Array,
    P_0: Array,
    Gamma_0: Array,
    c_Gamma: Array,
    theta_0: Array,
) -> Array:
    """
    Polytropic gas pressure (total pressure, therm. + kin.).

    Parameters
    ----------
    phi : Array
        Normalized isolated gravitational potential (see Notes)
    r_norm : Array
        Normalized radii to be used for the polytropic model.
    P_0 : Array
        Central gas pressure (total pressure, therm. + kin.)
    Gamma_0 : Array
        Central value of the polytropic index
    c_Gamma : Array
        Polytropic concentration value
    theta_0 : Array
        Potential prefactor (see Notes)

    Returns
    -------
    Array
        Gas pressure (total pressure, therm. + kin.) for each phi
    """
    t = theta(phi, theta_0)
    g = Gamma_r(r_norm, Gamma_0, c_Gamma)
    return P_0 * (t ** (g / (g - 1)))


def rho_g(
    phi: Array,
    r_norm: Array,
    rho_0: Array,
    Gamma_0: Array,
    c_Gamma: Array,
    theta_0: Array,
) -> Array:
    """
    Polytropic gas density.

    Parameters
    ----------
    phi : Array
        Normalized isolated gravitational potential (see Notes)
    r_norm : Array
        Normalized radii to be used for the polytropic model.
    rho_0 : Array
        Central gas density
    Gamma_0 : Array
        Central value of the polytropic index
    c_Gamma : Array
        Polytropic concentration value
    theta_0 : Array
        Potential prefactor (see Notes)

    Returns
    -------
    Array
        Gas density for each phi
    """
    t = theta(phi, theta_0)
    g = Gamma_r(r_norm, Gamma_0, c_Gamma)
    return rho_0 * (t ** (1 / (g - 1)))


def rho_P_g(
    phi: Array,
    r_norm: Array,
    rho_0: Array,
    P_0: Array,
    Gamma_0: Array,
    c_Gamma: Array,
    theta_0: Array,
) -> Tuple[Array, Array]:
    """
    Polytropic gas density and pressure (total pressure, therm. + kin.).

    Parameters
    ----------
    phi : Array
        Normalized isolated gravitational potential (see Notes)
    r_norm : Array
        Normalized radii to be used for the polytropic model.
    rho_0 : Array
        Central gas density
    P_0 : Array
        Central gas pressure (total pressure, therm. + kin.)
    Gamma_0 : Array
        Central value of the polytropic index
    c_Gamma : Array
        Polytropic concentration value
    theta_0 : Array
        Potential prefactor (see Notes)

    Returns
    -------
    Array
        Gas density for each phi
    Array
        Gas pressure (total pressure, therm. + kin.) for each phi
    """
    t = theta(phi, theta_0)
    g = Gamma_r(r_norm, Gamma_0, c_Gamma)
    rho = rho_0 * (t ** (1 / (g - 1)))
    P = P_0 * (t ** (g / (g - 1)))
    return rho, P


notes = """
    Notes
    -----
"""

notes_pot = """
    * The potential :math:`\\phi` is to be normalized to be zero at the
      bottom of the well, and positive everywhere else. This definition
      makes it equivalent to :math:`(\\phi - \\phi_0)` in the Ostriker
      model.
"""

notes_gamma = """
    * A fixed value of the polytropic index (e.g. 1.2) can be achieved
      with `Gamma = Gamma_r(r, 1.2, 0)`
"""


Gamma_r.__doc__ += notes + notes_gamma
theta.__doc__ += notes + notes_pot
P_g.__doc__ += notes + notes_pot + notes_gamma
rho_g.__doc__ += notes + notes_pot + notes_gamma
rho_P_g.__doc__ += notes + notes_pot + notes_gamma
