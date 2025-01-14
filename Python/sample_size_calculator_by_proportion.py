#!/usr/bin/env python3
def get_sample_size(pool_size, z_score, pop_proportion, error_margin, pop_size=None):
    """
    Calculate the sample size required to achieve a desired confidence level
    given the size of the population, the population proportion, the desired
    error margin, and the confidence interval.

    The formula used is Cochran's sample size formula, also known as the
    "finite population correction" formula.

    :param pool_size: The size of the population.
    :param z_score: The Z-score for the desired confidence level.
    :param pop_proportion: The proportion of the population expected to have a
        certain characteristic.
    :param error_margin: The desired error margin.
    :param pop_size: The size of the population. If not provided, pool_size is
        used.

    :return: The sample size required to achieve the desired confidence level.
    :rtype: int
    """
    if not pop_size:
        pop_size = pool_size
    return pop_size / (
        1
        + (
            ((z_score**2) * (pop_proportion * (1 - pop_proportion)))
            / ((error_margin**2) * pop_size)
        )
    )


print(get_sample_size(100000, 2.575, 0.38, 0.005, 100000))
print(get_sample_size(100000, 2.575, 0.38, 0.005))
print(get_sample_size(100000, 2.575, 0.38, 0.005, 1000000))
