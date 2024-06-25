from jax import Array
import jax.numpy as jnp


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
    f_max = 1 / (alpha_0 * (4**n_nt))
    alpha = alpha_0 * jnp.min(
        [(1 + z) ** beta, f_max * jnp.tanh(beta * z) + 1]
    )
    return alpha * (r_R500**n_nt)


def f_nt_nelson14(r_RDelta: Array, A: float, B: float, C: float) -> Array:
    """
    Non-thermal pressure fraction computed from an adapted version of
    the Nelson+14 model (see Notes)

    Parameters
    ----------
    r_RDelta : array-like
        Radii normalized to RDelta
    A : float
        :math:`A` parameter values
    B : float
        :math:`B` parameter values
    C : float
        :math:`C` parameter values

    Returns
    -------
    array-like
        Non-thermal pressure fraction for each radius

    Notes
    -----
    The model is computed as:

    .. math:: f_{nt} = 1 - A \\times \\left\\{ 1 + \\exp \\left[
        -\\left( \\frac{r}{B \\times R_\\Delta} \\right)^{C} \\right]
        \\right\\}.

    This is a modified version of the original model (see eq. 7 in
    Nelson+14), where the radius is expressed in units of RDelta instead
    of R200m.
    """
    return 1 - A * (1 + jnp.exp(-((r_RDelta / B) ** C)))


def f_nt_generic(r_RDelta: Array, a: float, b: float, c: float) -> Array:
    """
    Generic expression for non-thermal pressure fraction: a power law
    evolution with radius, plus a constant plateau (see Notes)


    Parameters
    ----------
    r_RDelta : array-like
        Radii normalized to RDelta
    A : float
        Non-thermal pressure fraction in the cluster center
    B : float
        Non-thermal pressure fraction at r=RDelta
    C : float
        Power law radial dependence of f_nt

    Returns
    -------
    array-like
        Non-thermal pressure fraction for each radius

    Notes
    -----
    The model is computed as:

    .. math:: f_{nt} = A + (B-A) \\left(\\frac{r}{R_\\Delta}\\right)^C
    """
    f = a + (b - a) * (r_RDelta**c)
    f = jnp.where(f < 1.0, f, 1.0)
    return f
