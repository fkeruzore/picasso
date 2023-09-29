from jax import Array
import jax.numpy as jnp


def theta(phi: Array, Gamma: float) -> Array:
    """
    Re-parametrized polytropic variable.

    Parameters
    ----------
    phi : array
        Normalized isolated gravitational potential (see Notes)
    Gamma : float
        Gas polytropic index

    Returns
    -------
    array
        Polytropic variable for each phi
    """
    t = 1.0 - ((Gamma - 1.0) / Gamma) * phi
    t = jnp.where(t >= 0.0, t, 0.0)
    return t


def P_g(phi: Array, P_0: float, Gamma: float) -> Array:
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

    Returns
    -------
    array
        Gas pressure (total pressure, therm. + kin.) for each phi
    """
    t = theta(phi, Gamma)
    return P_0 * (t ** (Gamma / (Gamma - 1.0)))


def rho_g(phi: Array, rho_0: float, Gamma: float) -> Array:
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

    Returns
    -------
    array
        Gas density for each phi
    """
    t = theta(phi, Gamma)
    return rho_0 * (t ** (1.0 / (Gamma - 1.0)))


def rho_P_g(
    phi: Array, rho_0: float, P_0: float, Gamma: float
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

    Returns
    -------
    array-like
        Gas density for each phi
    array-like
        Gas pressure (total pressure, therm. + kin.) for each phi
    """
    t = theta(phi, Gamma)
    rho = rho_0 * (t ** (1.0 / (Gamma - 1.0)))
    P = P_0 * (t ** (Gamma / (Gamma - 1.0)))
    return rho, P


def f_nt_shaw10(
    r_R500: Array, z: float, alpha_0=0.18, beta=0.5, n_nt=0.8
) -> Array:
    """
    Non-thermal pressure fraction computed from the Shaw+10 model

    Parameters
    ----------
    r_R500 : array-like
        Radii normalized to R500c
    alpha_0 : float, optional
        Non-thermal pressure fraction at z=0 and r=R500c,
        by default 0.18
    beta : float, optional
        Redshift evolution rate, by default 0.5
    n_nt : float, optional
        Power-law radial dependence, by default 0.8

    Returns
    -------
    array-like
        Non-thermal pressure fraction for each radius
    """
    f_max = 1.0 / (alpha_0 * (4.0**n_nt))
    alpha = alpha_0 * jnp.min(
        [(1.0 + z) ** beta, f_max * jnp.tanh(beta * z) + 1.0]
    )
    return alpha * (r_R500**n_nt)


def f_nt_generic(r_R500: Array, a: float, b: float, c: float) -> Array:
    """
    Generic expression for non-thermal pressure fraction: a power law
    evolution with radius, plus a constant plateau (see Notes)


    Parameters
    ----------
    r_R500 : array-like
        Radii normalized to R500c
    a : float
        Non-thermal pressure fraction in the cluster center
    b : float
        Non-thermal pressure fraction at r=2*R500c
    c : float
        Power law radial dependence of f_nt

    Returns
    -------
    array-like
        Non-thermal pressure fraction for each radius

    Notes
    -----
    The model is computed as:

    .. math:: f_{nt} = a + (b-a) \left(\frac{r}{2R_{500c}}\right)^c
    """
    return a + (b - a) * ((r_R500 / 2.0) ** c)
