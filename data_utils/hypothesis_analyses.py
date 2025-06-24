#!/usr/bin/env python3
import numpy as np
import scipy.stats as sp_stats

try:
    import fireducks.pandas as pd
except ImportError:
    import pandas as pd
    pd.set_option("mode.copy_on_write", True)
from statsmodels.stats.power import TTestIndPower


def calc_sample_size(
    min_detect_effect,
    std_dev_val,
    effect_size=0.25,
    alpha_val=0.05,
    power=0.80,
    messages=True,
):
    """
    Calculate the required sample size for a two-sample t-test to detect a specified effect size.

    Parameters
    ----------
    min_detect_effect : float
        The minimum detectable effect size as a difference in means.
    std_dev_val : float
        The standard deviation of the data.
    effect_size : float, optional
        The expected effect size as a fraction of the standard deviation. Default is 0.25.
    alpha_val : float, optional
        The significance level for the test. Default is 0.05.
    power : float, optional
        The desired statistical power of the test. Default is 0.80.
    messages : bool, optional
        If True, print the sample size to the console. Default is True.

    Returns
    -------
    sample_size : int
        The required sample size.

    Notes
    -----
    The calculation is based on the formula given in [1]_, which is an approximation of the sample size required for a two-sample t-test to detect a specified effect size.
    """
    from statsmodels.stats.power import TTestIndPower

    analyser = TTestIndPower()

    if min_detect_effect and std_dev_val:
        if not effect_size:
            effect_size = min_detect_effect / std_dev_val

    sample_size = analyser.solve_power(
        effect_size=effect_size,
        alpha=alpha_val,
        power=power,
        ratio=1,
        alternative="two-sided",
    )

    if messages:
        print(
            f"The required sample size is {sample_size} to get an effect size of {effect_size}."
        )

    return sample_size


def calc_t_test(arr_1, arr_2, alpha_val=0.05, equal_var=False, num_dp=4, messages=True):
    """
    Calculate a two-sample t-test between two arrays.

    Parameters
    ----------
    arr_1 : array_like
        The first array of data.
    arr_2 : array_like
        The second array of data.
    alpha_val : float, optional
        The significance level for the test. Default is 0.05.
    equal_var : bool, optional
        If True, the two arrays are assumed to have equal variances. Default is False.
    num_dp : int, optional
        The number of decimal places to round the results to. Default is 4.
    messages : bool, optional
        If True, print the results to the console. Default is True.

    Returns
    -------
    t_stat : float
        The t-statistic.
    p_value_2_tailed : float
        The p-value of the two-tailed test.

    Notes
    -----
    The t-test is calculated using scipy.stats.ttest_ind.
    """
    try:
        import fireducks.pandas as pd
    except ImportError:
        import pandas as pd
    import scipy.stats as sp_stats

    t_stat, p_value_2_tailed = sp_stats.ttest_ind(arr_1, arr_2, equal_var=equal_var)

    if messages:
        if p_value_2_tailed < alpha_val:
            print(
                f"Significant two-tailed T-statistic: stat = {round(t_stat, num_dp)}, p-value = {round(p_value_2_tailed, num_dp)}."
            )
        else:
            print(
                f"Two-tailed T-statistic: stat = {round(t_stat, num_dp)}, p-value = {round(p_value_2_tailed, num_dp)}."
            )

    return t_stat, p_value_2_tailed


def calc_t_test_extended(
    arr_1,
    arr_2,
    alpha_val=0.05,
    equal_var=False,
    confidence_level=0.95,
    num_dp=4,
    messages=True,
):
    """
    Perform an extended two-sample t-test between two arrays and calculate the confidence interval for the mean difference.

    Parameters
    ----------
    arr_1 : array_like
        The first array of data.
    arr_2 : array_like
        The second array of data.
    alpha_val : float, optional
        The significance level for the test. Default is 0.05.
    equal_var : bool, optional
        If True, the two arrays are assumed to have equal variances. Default is False.
    confidence_level : float, optional
        The confidence level for the interval. Default is 0.95.
    num_dp : int, optional
        The number of decimal places to round the results to. Default is 4.
    messages : bool, optional
        If True, print the results to the console. Default is True.

    Returns
    -------
    t_stat : float
        The t-statistic.
    p_value_2_tailed : float
        The p-value of the two-tailed test.
    mean_diff : float
        The difference in means between the two arrays.
    confidence_interval : list of float
        The lower and upper bounds of the confidence interval for the mean difference.

    Notes
    -----
    The t-test is calculated using scipy.stats.ttest_ind. The confidence interval is derived using the Welch-Satterthwaite equation for degrees of freedom.
    """
    import numpy as np

    try:
        import fireducks.pandas as pd
    except ImportError:
        import pandas as pd
    import scipy.stats as sp_stats

    # Calculate the t-test and p-value
    t_stat, p_value_2_tailed = sp_stats.ttest_ind(arr_1, arr_2, equal_var=equal_var)

    # Get the sizes of both samples
    arr_len_1 = len(arr_1)
    arr_len_2 = len(arr_2)

    # Get the means of both samples
    x_bar_1 = np.mean(arr_1)
    x_bar_2 = np.mean(arr_2)

    # Get the standard deviations for both groups
    std_dev_1 = np.std(arr_1, ddof=1)
    std_dev_2 = np.std(arr_2, ddof=1)

    # Difference in means
    mean_diff = x_bar_1 - x_bar_2

    # Standard error of the differences
    se_diff = np.sqrt((std_dev_1**2 / arr_len_1) + (std_dev_2**2 / arr_len_2))

    # Degree of freedom (Welch-Satterthwaite equation)
    dof_val = (se_diff**4) / (
        (
            (std_dev_1**2 / arr_len_1) ** 2 / (arr_len_1 - 1)
            + ((std_dev_2**2 / arr_len_2) ** 2 / (arr_len_2 - 1))
        )
    )

    # Get the 2-sided t-critical value for the desired confidence interval
    t_crit = sp_stats.t.ppf(1 - (1 - confidence_level) / 2, dof_val)

    # Calculate the upper and lower bounds for the above confidence interval
    ci_lower = mean_diff - (t_crit * se_diff)
    ci_upper = mean_diff + (t_crit * se_diff)

    if messages:
        if p_value_2_tailed < alpha_val:
            print(
                f"Significant two-tailed T-statistic: stat = {round(t_stat, num_dp)}, p-value = {round(p_value_2_tailed, num_dp)}, ..."
            )
            print(
                f"... mean difference = {round(mean_diff, num_dp)}, {confidence_level}% confidence interval = {round(ci_lower, num_dp)}, {round(ci_upper, num_dp)}."
            )
        else:
            print(
                f"Two-tailed T-statistic: stat = {round(t_stat, num_dp)}, p-value = {round(p_value_2_tailed, num_dp)}."
            )

    return t_stat, p_value_2_tailed, mean_diff, [ci_lower, ci_upper]
