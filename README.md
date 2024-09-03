<div align="center">
<img src="https://raw.githubusercontent.com/fkeruzore/fkeruzore.github.io/master/images/picasso_header.png" alt="logo"></img>
</div>

![Tests](https://github.com/fkeruzore/picasso/actions/workflows/python-tests.yml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/picasso-cosmo/badge/?version=latest)](https://picasso-cosmo.readthedocs.io/en/latest/?badge=latest)
[![arXiv](https://img.shields.io/badge/arXiv-2408.17445-b31b1b.svg)](https://arxiv.org/abs/2408.17445)

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
pip install -e "git+https://github.com/fkeruzore/picasso.git#egg=picasso[jax]"
```

Alternatively, if you already have JAX and flax installed, you may use

```sh
pip install -e "git+https://github.com/fkeruzore/picasso.git#egg=picasso"
```

The latter option will not install or upgrade any package relying on JAX, which can be useful to avoid messing up an existing install.
To install JAX on your system, see [JAX's installation page](https://github.com/google/jax#installation).

## Testing and benchmarking

`picasso` uses [Poetry](https://python-poetry.org) to manage dependencies.
To test your installation of `picasso`, you can install the `tests` dependency group and run `pytest`:

```sh
git clone git@github.com:fkeruzore/picasso.git
cd picasso
poetry install --with=tests
poetry run pytest
```

Some of the test also include basic benchmarking of model predictions using [pytest-benchmark](https://pytest-benchmark.readthedocs.io/en/latest/):

```sh
poetry run pytest --benchmark-enable
```

## Citation

If you use `picasso` for your research, please cite the `picasso` [original paper](https://arxiv.org/abs/2408.17445):

```bib
@article{keruzore_picasso_2024,
  title={The picasso gas model: Painting intracluster gas on gravity-only simulations}, 
  author={F. Kéruzoré and L. E. Bleem and N. Frontiere and N. Krishnan and M. Buehlmann and J. D. Emberson and S. Habib and P. Larsen},
  year={2024},
  eprint={2408.17445},
  archivePrefix={arXiv},
  primaryClass={astro-ph.CO},
  url={https://arxiv.org/abs/2408.17445}, 
}
```
