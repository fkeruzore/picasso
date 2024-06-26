from jax import Array
import jax.numpy as jnp
from typing import Tuple

notes = """
    Notes
    -----
    * The potential :math:`\\phi` is to be normalized to be zero at the
      bottom of the well, and positive everywhere else. This definition
      makes it equivalent to :math:`(\\phi - \\phi_0)` in the Ostriker
      model.

    * :math:`\\theta_0` is the pre-factor to multiply this potential
      with. In the Ostriker model,

      .. math:: \\theta_0 = \\frac{\\Gamma - 1}{\\Gamma}
        \\times \\frac{\\rho_0}{P_0}
"""


def theta(phi: Array, theta_0: float) -> Array:
    """
    Re-parametrized polytropic variable.

    Parameters
    ----------
    phi : Array
        Normalized isolated gravitational potential (see Notes)
    theta_0 : float
        Potential prefactor (see Notes)

    Returns
    -------
    Array
        Polytropic variable for each phi
    """
    t = 1 - (theta_0 * phi)
    t = jnp.where(t >= 0, t, 0)
    return t


def Gamma(r_norm, Gamma_0, Gamma_1):
    g = Gamma_0 * (1 + r_norm) ** Gamma_1
    g = jnp.where(g <= 1.0, 1.0, g)
    return g


def P_g(
    phi: Array,
    r_norm: float,
    P_0: float,
    Gamma_0: float,
    Gamma_1: float,
    theta_0: float,
) -> Array:
    """
    Polytropic gas pressure (total pressure, therm. + kin.).

    Parameters
    ----------
    phi : Array
        Normalized isolated gravitational potential (see Notes)
    P_0 : float
        Central gas pressure (total pressure, therm. + kin.)
    Gamma : float
        Gas polytropic index
    theta_0 : float
        Potential prefactor (see Notes)

    Returns
    -------
    Array
        Gas pressure (total pressure, therm. + kin.) for each phi
    """
    t = theta(phi, theta_0)
    g = Gamma(r_norm, Gamma_0, Gamma_1)
    return P_0 * (t ** (g / (g - 1)))


def rho_g(
    phi: Array,
    r_norm: float,
    rho_0: float,
    Gamma_0: float,
    Gamma_1: float,
    theta_0: float,
) -> Array:
    """
    Polytropic gas density.

    Parameters
    ----------
    phi : Array
        Normalized isolated gravitational potential (see Notes)
    rho_0 : float
        Central gas density
    Gamma : float
        Gas polytropic index
    theta_0 : float
        Potential prefactor (see Notes)

    Returns
    -------
    Array
        Gas density for each phi
    """
    t = theta(phi, theta_0)
    g = Gamma(r_norm, Gamma_0, Gamma_1)
    return rho_0 * (t ** (1 / (g - 1)))


def rho_P_g(
    phi: Array,
    r_norm: float,
    rho_0: float,
    P_0: float,
    Gamma_0: float,
    Gamma_1: float,
    theta_0: float,
) -> Tuple[Array, Array]:
    """
    Polytropic gas density and pressure (total pressure, therm. + kin.).

    Parameters
    ----------
    phi : Array
        Normalized isolated gravitational potential (see Notes)
    rho_0 : float
        Central gas density
    P_0 : float
        Central gas pressure (total pressure, therm. + kin.)
    Gamma : float
        Gas polytropic index
    theta_0 : float
        Potential prefactor (see Notes)

    Returns
    -------
    Array
        Gas density for each phi
    Array
        Gas pressure (total pressure, therm. + kin.) for each phi
    """
    t = theta(phi, theta_0)
    g = Gamma(r_norm, Gamma_0, Gamma_1)
    rho = rho_0 * (t ** (1 / (g - 1)))
    P = P_0 * (t ** (g / (g - 1)))
    return rho, P


theta.__doc__ += notes
P_g.__doc__ += notes
rho_g.__doc__ += notes
rho_P_g.__doc__ += notes
