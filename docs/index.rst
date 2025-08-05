.. picasso documentation master file, created by
   sphinx-quickstart on Thu Apr 18 14:08:35 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

*******
picasso
*******

.. div:: sd-text-left sd-font-italic

   Painting intracluster gas on gravity-only simulations

----

``picasso`` is a model that makes predictions for the thermodynamic properties of the gas in massive dark matter halos from gravity-only cosmological simulations.
It combines an analytical model of gas properties as a function of gravitational potential with a neural network predicting the parameters of said model.
It is released here as a Python package, combining an implementation of the gas model based on `JAX <https://jax.readthedocs.io/en/latest/>`_ and `flax <https://flax.readthedocs.io/en/latest/index.html>`_, and models that have been pre-trained to reproduce gas properties from hydrodynamic simulations.

Why use picasso?
^^^^^^^^^^^^^^^^

``picasso`` presents a few advantages that make it particularly interesting to predict gas properties from gravity-only halos:

.. grid::

   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Robustness
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            By combining neural networks and physical models, ``picasso`` can make fast, accurate and precise predictions of intracluster gas thermodynamics.

   .. grid-item::
      :columns: 12 12 12 6

      .. card:: JAX under the hood
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            Thanks to the use of ``JAX`` and ``flax`` in its numerical implementation, ``picasso`` can make predictions that can be compiled just-in-time, accelerated on GPU/TPU, and are automatically differentiable.

   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Flexibility
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            ``picasso`` models can be trained to make predictions from extensive data inputs (*e.g.*, from the full N-body particle distribution of a dark matter halo) or from minimal information (*e.g.*, a halo catalog with only halo mass and concentration).

   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Trained models
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            The ``picasso`` library includes pre-trained models that can reasily be used to make predictions from various inputs.

----

Installation
^^^^^^^^^^^^

``picasso`` can be install via ``pip``:

.. code-block:: bash

   pip install picasso-cosmo[jax]

Alternatively, if you already have JAX and flax installed, you may use

.. code-block:: bash

   pip install picasso-cosmo

The latter option will not install or upgrade any package relying on JAX, which can be useful to avoid messing up an existing install.
To install JAX on your system, see `JAX's installation page <https://github.com/google/jax#installation>`_.


Testing and benchmarking
^^^^^^^^^^^^^^^^^^^^^^^^

``picasso`` uses `uv <https://docs.astral.sh/uv/>`_ to manage dependencies.
To test your installation of ``picasso``, you can install the ``tests`` dependency group and run ``pytest``:

.. code-block:: bash

   git clone git@github.com:fkeruzore/picasso.git
   cd picasso
   uv python install
   uv sync --all-groups --all-extras
   uv run pytest

Some of the test also include basic benchmarking of model predictions using `pytest-benchmark <https://pytest-benchmark.readthedocs.io/en/latest/>`_:

.. code-block:: bash

   uv run pytest --benchmark-enable

----

Learn more
^^^^^^^^^^

.. toctree::
   :maxdepth: 1
   :caption: The picasso model

   model/model
   model/trained_models
   notebooks/plot_model_params.ipynb

.. toctree::
   :maxdepth: 1
   :caption: Example gallery

   notebooks/make_predictions.ipynb
   notebooks/demo_model.ipynb
   notebooks/train_model.ipynb

.. toctree::
   :caption: Reference
   :maxdepth: 1

   api/predictors
   api/polytrop
   api/nonthermal

----

Citation
^^^^^^^^

If you use ``picasso`` for your research, please cite the ``picasso`` `original paper <https://astro.theoj.org/article/127486-the-picasso-gas-model-painting-intracluster-gas-on-gravity-only-simulations>`_:

.. code-block:: bibtex

   @ARTICLE{2024OJAp....7E.116K,
          author = {{K{\'e}ruzor{\'e}}, Florian and {Bleem}, L.~E. and {Frontiere}, N. and {Krishnan}, N. and {Buehlmann}, M. and {Emberson}, J.~D. and {Habib}, S. and {Larsen}, P.},
           title = "{The picasso gas model: Painting intracluster gas on gravity-only simulations}",
         journal = {The Open Journal of Astrophysics},
        keywords = {Astrophysics - Cosmology and Nongalactic Astrophysics},
            year = 2024,
           month = dec,
          volume = {7},
             eid = {116},
           pages = {116},
             doi = {10.33232/001c.127486},
   archivePrefix = {arXiv},
          eprint = {2408.17445},
    primaryClass = {astro-ph.CO},
}



----

Indices and tables
^^^^^^^^^^^^^^^^^^

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
