from jax import Array
import jax.numpy as jnp


def theta(phi: Array, rho_0: float, P_0: float, Gamma: float) -> Array:
    """
    Polytropic variable as defined in e.g. Ostriker+05

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

    Returns
    -------
    array
        Polytropic variable for each phi

    Notes
    -----
    The potential `phi` is to be normalized to be zero at the bottom of
        the well, and positive everywhere else. This definition males it
        equivalent to (phi - phi_0) in the Ostriker model.
    """
    t = 1.0 - ((Gamma - 1.0) / Gamma) * (rho_0 / P_0) * phi
    t = jnp.where(t >= 0.0, t, 0.0)
    return t


def P_g(phi: Array, rho_0: float, P_0: float, Gamma: float) -> Array:
    """
    Polytropic gas pressure (total pressure, therm. + kin.)
    as defined in e.g. Ostriker+05

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

    Returns
    -------
    array
        Gas pressure (total pressure, therm. + kin.) for each phi

    Notes
    -----
    The potential `phi` is to be normalized to be zero at the bottom of
        the well, and positive everywhere else. This definition males it
        equivalent to (phi - phi_0) in the Ostriker model.
    """
    t = theta(phi, rho_0, P_0, Gamma)
    return P_0 * (t ** (Gamma / (Gamma - 1.0)))


def rho_g(phi: Array, rho_0: float, P_0: float, Gamma: float) -> Array:
    """
    Polytropic gas density as defined in e.g. Ostriker+05

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

    Returns
    -------
    array
        Gas density for each phi

    Notes
    -----
    The potential `phi` is to be normalized to be zero at the bottom of
        the well, and positive everywhere else. This definition males it
        equivalent to (phi - phi_0) in the Ostriker model.
    """
    t = theta(phi, rho_0, P_0, Gamma)
    return rho_0 * (t ** (1.0 / (Gamma - 1.0)))


def rho_P_g(
    phi: Array, rho_0: float, P_0: float, Gamma: float
) -> (Array, Array):
    """
    Polytropic gas density and pressure (total pressure, therm. + kin.)
    as defined in e.g. Ostriker+05

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

    Returns
    -------
    array-like
        Gas density for each phi
    array-like
        Gas pressure (total pressure, therm. + kin.) for each phi

    Notes
    -----
    The potential `phi` is to be normalized to be zero at the bottom of
        the well, and positive everywhere else. This definition males it
        equivalent to (phi - phi_0) in the Ostriker model.
    """
    t = theta(phi, rho_0, P_0, Gamma)
    rho = rho_0 * (t ** (1.0 / (Gamma - 1.0)))
    P = P_0 * (t ** (Gamma / (Gamma - 1.0)))
    return rho, P
