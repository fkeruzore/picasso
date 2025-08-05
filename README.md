<div align="center">
<img src="https://raw.githubusercontent.com/fkeruzore/fkeruzore.github.io/master/images/picasso_header.png" alt="logo"></img>
</div>

![Tests](https://github.com/fkeruzore/picasso/actions/workflows/python-tests.yml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/picasso-cosmo/badge/?version=latest)](https://picasso-cosmo.readthedocs.io/en/latest/?badge=latest)
[![arXiv](https://img.shields.io/badge/arXiv-2408.17445-b31b1b.svg)](https://arxiv.org/abs/2408.17445)
![PyPI - Version](https://img.shields.io/pypi/v/picasso-cosmo)

# picasso

*Painting intracluster gas on gravity-only simulations*

`picasso` is a model that allows making predictions for the thermodynamic properties of the gas in massive dark matter halos from gravity-only cosmological simulations.
It combines an analytical model of gas properties as a function of gravitational potential with a neural network predicting the parameters of said model.
It is released here as a Python package, combining an implementation of the gas model based on [JAX](https://jax.readthedocs.io/en/latest/) and [Flax](https://flax.readthedocs.io/en/latest/index.html), and models that have been pre-trained to reproduce gas properties from hydrodynamic simulations.

## [Documentation](https://picasso-cosmo.readthedocs.io/en/latest/)

*See also [Kéruzoré et al. (2024)](https://arxiv.org/abs/2408.17445).*

## Installation

`picasso` can be install via `pip`:

```sh
pip install picasso-cosmo[jax]
```

Alternatively, if you already have JAX and flax installed, you may use

```sh
pip install picasso-cosmo
```

The latter option will not install or upgrade any package relying on JAX, which can be useful to avoid messing up an existing install.
To install JAX on your system, see [JAX's installation page](https://github.com/google/jax#installation).

## Testing and benchmarking

`picasso` uses [uv](https://docs.astral.sh/uv/) to manage dependencies.
To test your installation of `picasso`, you can install the `tests` dependency group and run `pytest`:

```sh
  git clone git@github.com:fkeruzore/picasso.git
  cd picasso
  uv python install
  uv sync --all-groups --all-extras
  uv run pytest
```

Some of the test also include basic benchmarking of model predictions using [pytest-benchmark](https://pytest-benchmark.readthedocs.io/en/latest/):

```sh
uv run pytest --benchmark-enable
```

## Citation

If you use `picasso` for your research, please cite the `picasso` [original paper](https://astro.theoj.org/article/127486-the-picasso-gas-model-painting-intracluster-gas-on-gravity-only-simulations):

```bib
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
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024OJAp....7E.116K},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
