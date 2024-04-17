from jax import Array
import jax.numpy as jnp


def theta(phi: Array, Gamma: float, theta_0: float) -> Array:
    """
    Re-parametrized polytropic variable.

    Parameters
    ----------
    phi : array
        Normalized isolated gravitational potential (see Notes)
    Gamma : float
        Gas polytropic index
    theta_0 : float
        Potential prefactor (see Notes)

    Returns
    -------
    array
        Polytropic variable for each phi

    Notes
    -----
    The potential `phi` is to be normalized to be zero at the bottom of
        the well, and positive everywhere else. This definition males it
        equivalent to (phi - phi_0) in the Ostriker model.
    theta_0 is the pre-factor to multiply this potential with. In the
        Ostriker model, theta_0 = ((Gamma - 1) / Gamma) * (rho_0 / P_0).
    """
    # t = 1 - ((Gamma - 1) / Gamma) * theta_0 * phi
    t = 1 - (theta_0 * phi)
    t = jnp.where(t >= 0, t, 0)
    return t


def P_g(phi: Array, P_0: float, Gamma: float, theta_0: float) -> Array:
    """
    Polytropic gas pressure (total pressure, therm. + kin.).

    Parameters
    ----------
    phi : array
        Normalized isolated gravitational potential (see Notes)
    P_0 : float
        Central gas pressure (total pressure, therm. + kin.)
    Gamma : float
        Gas polytropic index
    theta_0 : float
        Potential prefactor (see Notes)

    Returns
    -------
    array
        Gas pressure (total pressure, therm. + kin.) for each phi

    Notes
    -----
    The potential `phi` is to be normalized to be zero at the bottom of
        the well, and positive everywhere else. This definition males it
        equivalent to (phi - phi_0) in the Ostriker model.
    theta_0 is the pre-factor to multiply this potential with. In the
        Ostriker model, theta_0 = ((Gamma - 1) / Gamma) * (rho_0 / P_0).
    """
    t = theta(phi, Gamma, theta_0)
    return P_0 * (t ** (Gamma / (Gamma - 1)))


def rho_g(phi: Array, rho_0: float, Gamma: float, theta_0: float) -> Array:
    """
    Polytropic gas density.

    Parameters
    ----------
    phi : array
        Normalized isolated gravitational potential (see Notes)
    rho_0 : float
        Central gas density
    Gamma : float
        Gas polytropic index
    theta_0 : float
        Potential prefactor (see Notes)

    Returns
    -------
    array
        Gas density for each phi

    Notes
    -----
    The potential `phi` is to be normalized to be zero at the bottom of
        the well, and positive everywhere else. This definition males it
        equivalent to (phi - phi_0) in the Ostriker model.
    theta_0 is the pre-factor to multiply this potential with. In the
        Ostriker model, theta_0 = ((Gamma - 1) / Gamma) * (rho_0 / P_0).
    """
    t = theta(phi, Gamma, theta_0)
    return rho_0 * (t ** (1 / (Gamma - 1)))


def rho_P_g(
    phi: Array, rho_0: float, P_0: float, Gamma: float, theta_0: float
) -> (Array, Array):
    """
    Polytropic gas density and pressure (total pressure, therm. + kin.).

    Parameters
    ----------
    phi : array
        Normalized isolated gravitational potential (see Notes)
    rho_0 : float
        Central gas density
    P_0 : float
        Central gas pressure (total pressure, therm. + kin.)
    Gamma : float
        Gas polytropic index
    theta_0 : float
        Potential prefactor (see Notes)

    Notes
    -----
    The potential `phi` is to be normalized to be zero at the bottom of
        the well, and positive everywhere else. This definition males it
        equivalent to (phi - phi_0) in the Ostriker model.
    theta_0 is the pre-factor to multiply this potential with. In the
        Ostriker model, theta_0 = ((Gamma - 1) / Gamma) * (rho_0 / P_0).

    Returns
    -------
    array-like
        Gas density for each phi
    array-like
        Gas pressure (total pressure, therm. + kin.) for each phi
    """
    t = theta(phi, Gamma, theta_0)
    rho = rho_0 * (t ** (1 / (Gamma - 1)))
    P = P_0 * (t ** (Gamma / (Gamma - 1)))
    return rho, P
