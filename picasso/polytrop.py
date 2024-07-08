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

    * A fixed value of the polytropic index (e.g. 1.2) can be achieved
      with `Gamma = Gamma_r(r, 1.2, 1e-6)` (note that `c_Gamma` cannot
      be zero by definition).
"""


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


def Gamma_r(r: Array, Gamma_0: Array, c_Gamma: Array):
    """
    Compute the radius-dependent polytropic index Gamma(r), following
    Komatsu & Seljak (2001).

    Parameters
    ----------
    r : Array
        Normalized radii
    Gamma_0 : Array
        Central value of the polytropic index
    c_Gamma : Array
        Polytropic concentration value

    Returns
    -------
    Array
        Gamma values at specified radii
    """
    x = c_Gamma * r
    return Gamma_0 + ((x + 1) * jnp.log(x + 1) - x) / (
        (3 * x + 1) * jnp.log(x + 1)
    )


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


theta.__doc__ += notes
P_g.__doc__ += notes
rho_g.__doc__ += notes
rho_P_g.__doc__ += notes
