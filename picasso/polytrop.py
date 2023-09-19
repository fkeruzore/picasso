import jax.numpy as jnp


def theta(phi, rho_0, P_0, Gamma):
    """
    Polytropic variable as defined in e.g. Ostriker+05

    Parameters
    ----------
    phi : array-like
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


def P_g(phi, rho_0, P_0, Gamma):
    """
    Polytropic gas pressure (total pressure, therm. + kin.)
    as defined in e.g. Ostriker+05

    Parameters
    ----------
    phi : array-like
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
        Gas pressure (total pressure, therm. + kin.) for each phi

    Notes
    -----
    The potential `phi` is to be normalized to be zero at the bottom of
        the well, and positive everywhere else. This definition males it
        equivalent to (phi - phi_0) in the Ostriker model.
    """
    t = theta(phi, rho_0, P_0, Gamma)
    return P_0 * (t ** (Gamma / (Gamma - 1.0)))


def rho_g(phi, rho_0, P_0, Gamma):
    """
    Polytropic gas density as defined in e.g. Ostriker+05

    Parameters
    ----------
    phi : array-like
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

    Notes
    -----
    The potential `phi` is to be normalized to be zero at the bottom of
        the well, and positive everywhere else. This definition males it
        equivalent to (phi - phi_0) in the Ostriker model.
    """
    t = theta(phi, rho_0, P_0, Gamma)
    return rho_0 * (t ** (1.0 / (Gamma - 1.0)))


def rho_P_g(phi, rho_0, P_0, Gamma):
    """
    Polytropic gas density and pressure (total pressure, therm. + kin.)
    as defined in e.g. Ostriker+05

    Parameters
    ----------
    phi : array-like
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


def f_nt_shaw10(r_R500, z, alpha_0=0.18, beta=0.5, n_nt=0.8):
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


def f_nt_generic(r_R500, alpha, beta, gamma):
    """
    Generic expression for non-thermal pressure fraction: a power law
    evolution with radius, plus a constant plateau (see Notes)


    Parameters
    ----------
    r_R500 : array-like
        Radii normalized to R500c
    alpha : float
        Log10 of the non-thermal pressure fraction in the cluster
        center
    beta : float
        Log10 of the non-thermal pressure fraction at r=2*R500c
    gamma : float
        Power law radial dependence of f_nt

    Returns
    -------
    array-like
        Non-thermal pressure fraction for each radius

    Notes
    -----
    The model is computed as:

    .. math:: f_{nt} = 10^\alpha + (10^\beta - 10^\alpha)
              \left(\frac{r}{2R_{500c}}\right)^\gamma
    """
    return (10**alpha) + ((10**beta) - (10**alpha)) * (
        (r_R500 / 2.0) ** gamma
    )
