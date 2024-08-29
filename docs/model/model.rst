What is the picasso model?
==========================

The ``picasso`` model is the combination of two different parts:

* A polytropic gas model, mapping intracluster gas thermodynamics to a gravitational potential distribution, given a set of model parameters :math:`\vartheta_{\rm gas}`;
* A neural network predicting a vector of model parameters :math:`\vartheta_{\rm gas}` from a vector of halo properties :math:`\vartheta_{\rm halo}`.

The gas model
-------------

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

.. math:: \theta_0 = \frac{\Gamma - 1}{\Gamma} \times \frac{\rho_0}{P_0}

We further write the fraction of non-thermal pressure as a power-law of radius, plus a constant plateau:

.. math:: f_{\rm nt}(r) = A_{\rm nt} + (B_{\rm nt} - A_{\rm nt}) \left(\frac{r}{2R_{500c}}\right)^{C_{\rm nt}}

This adds three parameters to our gas model: :math:`A_{\rm nt}` is the central value of non-thermal pressure fraction, :math:`B_{\rm nt}` is the non-thermal pressure fraction at :math:`r=2R_{500c}`, and :math:`C_{\rm nt}` is the power law evolution with radius.

Therefore, for a potential distribution :math:`\phi(r)`, gas thermodynamics are fully specified by a vector :math:`\vartheta_{\rm gas} = (\rho_0, \, P_0, \, \Gamma_0, \, c_\gamma, \, \theta_0, \, A_{\rm nt}, \, B_{\rm nt}, \, C_{\rm nt})`.

See also:

* :doc:`../notebooks/plot_model_params`, for a visual representation of the impact of each parameter of the model on gas thermodynamics;
* :doc:`../api/polytrop` and :doc:`../api/nonthermal`, for the documentation of the functions providing the numerical implementation of the model above;
* :doc:`../notebooks/demo_model`, for code examples.

Neural network predictions
--------------------------

To pick values for the gas model components, ``picasso`` uses a fully-connected neural network that is trained to predict the :math:`\vartheta_{\rm gas}` vector given a set of halo properties :math:`\vartheta_{\rm halo}`.
The key assumptions are that, for a given halo, the complexity of the gas distribution can be captured by the parameters of :math:`\vartheta_{\rm gas}`, and that halo properties contain enough information to robustly predict the values of these parameters.
The neural network is trained to reproduce a given target data, in general the properties of halos found in hydrodynamic simulations.
The components of :math:`\vartheta_{\rm halo}` can vary, depending on the properties available during the training and on how widely usable a trained model aims to be.

See also:

* :doc:`../model/trained_models`, for a list of the available pre-trained models;
* :doc:`../api/predictors`, for the documentation of the prediction functions available in ``picasso``;
* :doc:`../notebooks/make_predictions`, for code examples.