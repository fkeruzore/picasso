<div align="center">
<img src="https://raw.githubusercontent.com/fkeruzore/fkeruzore.github.io/master/images/picasso_header.png" alt="logo"></img>
</div>

# picasso

*Painting intracluster gas on gravity-only simulations*

`picasso` is a model that allows making predictions for the thermodynamic properties of the gas in massive dark matter halos from gravity-only cosmological simulations.
It combines an analytical model of gas properties as a function of gravitational potential with a neural network predicting the parameters of said model.
It is released here as a Python package, combining an implementation of the gas model based on [JAX](https://jax.readthedocs.io/en/latest/) and [Flax](https://flax.readthedocs.io/en/latest/index.html), and models that have been pre-trained to reproduce gas properties from hydrodynamic simulations.

## [Documentation](https://github.com/fkeruzore/picasso)

*See also [Kéruzoré et al. (2024)](https://arxiv.org/abs/2306.13807).*

## Installation

`picasso` uses [Poetry](https://python-poetry.org) to install dependencies:

```sh
git clone git@github.com:fkeruzore/picasso.git
cd picasso
poetry install
# or, if you already have JAX and flax installed,
poetry install --without=jax
```

The latter option will not install or upgrade any package relying on JAX, which can be useful to avoid messing up an existing install.
To install JAX on your system, see [JAX's installation page](https://github.com/google/jax#installation).

## Citation

If you use `picasso` for your research, please cite the `picasso` [original paper](https://arxiv.org/abs/2306.13807):

```bib
@article{keruzore_picasso_2024,
  title={Optimization and Quality Assessment of Baryon Pasting for Intracluster Gas using the Borg Cube Simulation}, 
  author={F. Kéruzoré and L. E. Bleem and M. Buehlmann and J. D. Emberson and N. Frontiere and S. Habib and K. Heitmann and P. Larsen},
  year={2023},
  eprint={2306.13807},
  archivePrefix={arXiv},
  primaryClass={astro-ph.CO},
  doi={https://doi.org/10.21105/astro.2306.13807},
  url={https://arxiv.org/abs/2306.13807}, 
}
```
