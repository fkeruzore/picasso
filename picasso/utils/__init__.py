__author__ = "Florian Kéruzoré"
__email__ = "florian.keruzore@gmail.com"

__all__ = [
    "sph",
    "transform_minmax",
    "inv_transform_minmax",
    "quantile_normalization",
]

from . import sph
from .data_preparation import (
    transform_minmax,
    inv_transform_minmax,
    quantile_normalization,
)
