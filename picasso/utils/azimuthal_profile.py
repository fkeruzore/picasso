from jax import Array, vmap
import jax.numpy as jnp
from typing import Tuple


def azimuthal_profile(
    q: Array,
    r: Array,
    r_bin_edges: Array,
    statistics: Tuple[str] = ("mean", "std"),
) -> Tuple[dict, Array]:
    """
    Compute binned statistics for a set of data.

    Parameters
    ----------
    q : Array
        The values on which to compute the statistic.
    r : Array
        A sequence of values to be binned.
    r_edges : Array
        Edges of the radial bins.
    statistics : tuple of str, optional
        The statistics to compute (default is ['mean', 'std']).
        The following statistics are available:
            - 'mean' : compute the mean of values for points within
              each bin.
            - 'std' : compute the standard deviation of values for
              points within each bin.
            - 'sum' : compute the sum of values for points within
              each bin.
            - 'count' : compute the count of points within each bin.

    Returns
    -------
    results : dict
        A dictionary where keys are the statistics and values are
        arrays of the computed statistic in each bin.
    r_bin_centers : Array
        The centers of the bins.
    """

    # Digitize the x values to find out which bin they belong to
    binnumber = jnp.digitize(r, r_bin_edges) - 1

    # Exclude values that fall outside the bin range
    binnumber = jnp.where(
        (binnumber >= 0) & (binnumber < len(r_bin_edges) - 1), binnumber, -1
    )
    bin_indices = jnp.arange(len(r_bin_edges) - 1)

    def compute_statistic_per_bin(stat, bin_index, values, binnumber):
        bin_mask = binnumber == bin_index
        bin_values = jnp.where(bin_mask, values, 0)

        if stat == "mean":
            bin_count = jnp.sum(bin_mask)
            bin_sum = jnp.sum(bin_values)
            return bin_sum / jnp.where(bin_count > 0, bin_count, 1)
        elif stat == "sum":
            return jnp.sum(bin_values)
        elif stat == "count":
            return jnp.sum(bin_mask)
        elif stat == "std":
            bin_count = jnp.sum(bin_mask)
            bin_mean = jnp.sum(bin_values) / jnp.where(
                bin_count > 0, bin_count, 1
            )
            bin_sq_diff = jnp.sum(
                jnp.where(bin_mask, (values - bin_mean) ** 2, 0)
            )
            return jnp.sqrt(
                bin_sq_diff / jnp.where(bin_count > 0, bin_count, 1)
            )
        else:
            raise ValueError("Unsupported statistic")

    # Create a function to compute all statistics for a given bin index
    def compute_all_statistics(bin_index, values, binnumber):
        return [
            compute_statistic_per_bin(stat, bin_index, values, binnumber)
            for stat in statistics
        ]

    # Use vmap to vectorize the computation over bin indices
    compute_all = vmap(
        lambda bin_index: compute_all_statistics(bin_index, q, binnumber)
    )
    all_statistics = compute_all(bin_indices)

    # Organize the results in a tuple
    results = (
        jnp.array(
            [
                all_statistics[i][bin_index]
                for bin_index in range(len(bin_indices))
            ]
        )
        for i, stat in enumerate(statistics)
    )

    return 0.5 * (r_bin_edges[1:] + r_bin_edges[:-1]), *results
