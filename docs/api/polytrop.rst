polytrop: Polytropic gas model
==============================

.. currentmodule:: picasso.polytrop

The polytropic gas model is written as:

    .. math:: \rho(\phi) = \rho_0 \theta^{\Gamma(r) / (\Gamma(r) - 1)}(\phi), \\[10pt]
        P(\phi) = P_0 \theta^{1 / (\Gamma(r) - 1)}(\phi),

where :math:`\phi` is the halo's gravitational potential, and

    .. math:: \theta(\phi) = 1 - \theta_0 (\phi - \phi_0).

The gas polytropic index, :math:`\Gamma`, is allowed to vary with radius as:

    .. math:: \Gamma(r) = 
        \begin{cases}
        \begin{aligned}
            & \; 1 + (\Gamma_0 - 1) \frac{1}{1 + e^{-x}} & c_\Gamma \geqslant 0; \\
            & \; \Gamma_0 + (\Gamma_0 - 1) \left(1 - \frac{1}{1 + e^{x}}\right) & c_\Gamma < 0, \\
        \end{aligned}
        \end{cases}

with :math:`x \equiv r / (c_\gamma R_{500c})`.
The model has five parameters: :math:`(\rho_0, P_0)` are the central value of gas density and pressure, :math:`\Gamma_0` is the asymptotic value of the polytropic index as :math:`r \rightarrow \infty`, :math:`c_\gamma` is the polytropic concentration (:math:`c_\gamma = 0` implies :math:`\Gamma(r) = \Gamma_0`), and :math:`\theta_0` is a shape parameter.
In the Ostriker model,

    .. math:: \theta_0 = \frac{\Gamma - 1}{\Gamma}
        \times \frac{\rho_0}{P_0}

We further write the fraction of non-thermal pressure as a power-law of radius, plus a constant plateau:

    .. math:: f_{\rm nt}(r) = a_{\rm nt} + (b_{\rm nt} - a_{\rm nt}) \left(\frac{r}{2r_{500c}}\right)^{c_{\rm nt}}

This adds three parameters to our gas model: :math:`a_{\rm nt}` is the central value of non-thermal pressure fraction, :math:`b_{\rm nt}` is the non-thermal pressure fraction at :math:`r=2r_{500c}`, and :math:`c_{\rm nt}` is the power law evolution with radius.

.. autosummary::
    rho_P_g
    rho_g
    P_g
    theta
    Gamma_r

.. autofunction:: rho_P_g
.. autofunction:: rho_g
.. autofunction:: P_g
.. autofunction:: theta
.. autofunction:: Gamma_r