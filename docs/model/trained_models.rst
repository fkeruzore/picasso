Available pre-trained predictors
================================

The ``picasso`` package comes with several pre-trained models, allowing direct inference of gas thermodynamics from halo properties and potential.
They are all accessible in the ``picasso.predictors`` module; a list can be found in ``picasso.predictors.available_predictors``.

576 runs
--------

The six models described in `Kéruzoré et al. (2024) <https://arxiv.org/abs/2408.17445>`_.

* Underlying cosmology: Planck 2020 (:math:`\Omega_{\rm cdm} = 0.26067`, :math:`\omega_{\rm b} = 0.02242`, :math:`h = 0.6766`, :math:`\sigma_8 = 0.8102`, :math:`n_s = 0.9665`, :math:`w = −1`), :math:`\Omega_k = 0`, :math:`\Sigma m_\nu = 0`
* :math:`L_{\rm box} = 576 h^{−1} \; {\rm Mpc}`, :math:`L_{\rm box} = 576 \; h^{−1}{\rm Mpc}`, :math:`2304^3` particles (Gravity-only particle mass: :math:`m_m = 1.34 \times 10^9 \; h^{-1}M_\odot`).
* Trained on :math:`M_{500c} > 10^{13.5} \; h^{-1}M_\odot`
* Full input vector (see details in table): :math:`\log_{10} (M_{200c} / 10^{14} \; h^{-1}M_\odot)`, :math:`c_{200c}`, :math:`\Delta x / R_{200c}`, :math:`c_{\rm acc.}/c_{200c}`, :math:`c_{\rm peak}/c_{200c}`, :math:`e`, :math:`p`, :math:`a_{\rm \rm lmm}`, :math:`a_{25}`, :math:`a_{50}`, :math:`a_{75}`, :math:`\dot{M} / (M_\odot / {\rm GYr})`.

.. list-table::
   :widths: 20 30 50
   :header-rows: 1

   * - Model name
     - In-code reference
     - Description
   * - baseline
     - ``predictors.baseline_576``
     - Uses full input vector, trained to reproduce non-radiative hydrodynamics profiles, with fixed :math:`c_\gamma = 0`
   * - compact
     - ``predictors.compact_576``
     - Uses full input vector minus mass assembly history, trained to reproduce non-radiative hydrodynamics profiles, with fixed :math:`c_\gamma = 0`
   * - minimal
     - ``predictors.minimal_576``
     - Uses halo mass and concentration, trained to reproduce non-radiative hydrodynamics profiles, with fixed :math:`c_\gamma = 0`
   * - subgrid
     - ``predictors.subgrid_576``
     - Uses full input vector, trained to reproduce full-physics hydrodynamics profiles, with fixed :math:`c_\gamma = 0`
   * - compact_subgrid
     - ``predictors.compact_subgrid_576``
     - Uses compact input vector, trained to reproduce full-physics hydrodynamics profiles, with fixed :math:`c_\gamma = 0`
   * - NR + :math:`\Gamma(r)`
     - ``predictors.nonradiative_Gamma_r_576``
     - Uses full input vector, trained to reproduce non-radiative hydrodynamics profiles, with variable :math:`c_\gamma`
   * - SG + :math:`\Gamma(r)`
     - ``predictors.subgrid_Gamma_r_576``
     - Uses full input vector, trained to reproduce full-physics hydrodynamics profiles, with variable :math:`c_\gamma`

See also:

* :doc:`../api/predictors`, for the documentation of the prediction functions available in ``picasso``;
* :doc:`../notebooks/make_predictions`, for code examples.
