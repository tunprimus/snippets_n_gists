#!/usr/bin/env python3
# Adapted from: Statistical Methods for Anomaly Detection: A Comprehensive Guide -> https://toxigon.com/statistical-methods-for-anomaly-detection
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import mahalanobis
from statsmodels.stats.stattools import test_esd

def calculate_z_score(data):
    import numpy as np
    import pandas as pd

    x_bar = np.mean(data)
    std_dev = np.std(data)
    z_scores = [(x - x_bar) / std_dev for x in data]
    return z_scores


def modified_z_score(data):
    import numpy as np
    import pandas as pd

    x_bar = np.mean(data)
    std_dev = np.std(data)
    mean_abs_deviation = np.mean([np.abs(x - x_bar) for x in data])

    median = np.median(data)
    median_abs_deviation = np.median([np.abs(x - median) for x in data])

    if median_abs_deviation == 0:
        modified_z_scores = [((x - median) / (1.253314 * mean_abs_deviation)) for x in data]
    else:
        modified_z_scores = [((x - median) / (1.486 * median_abs_deviation)) for x in data]
    return modified_z_scores


def calculate_iqr(data):
    import numpy as np
    import pandas as pd

    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    lo_bound = q1 - 1.5 * iqr
    hi_bound = q3 + 1.5 * iqr
    return lo_bound, hi_bound


def grubbs_test(data):
    import numpy as np
    import pandas as pd
    from scipy import stats

    n = len(data)
    x_bar = np.mean(data)
    std_dev = np.std(data)
    numerator = np.max([np.abs(x - x_bar) for x in data])
    denominator = std_dev
    test_statistic = numerator / denominator
    critical_value = stats.t.ppf(1 - 0.05 / (2 * n), n - 2)
    return test_statistic, critical_value


def dixons_q_test(data):
    sorted_data = sorted(data)
    q_value = (sorted_data[1] - sorted_data[0]) / (sorted_data[-1] - sorted_data[0])
    return q_value


def tukeys_fences(data, k=1.5):
    import numpy as np
    import pandas as pd

    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    lo_bound = q1 - k * iqr
    hi_bound = q3 + k * iqr
    return lo_bound, hi_bound


def chauvenets_criterion(data, criterion_val=0.5):
    import numpy as np
    import pandas as pd
    from scipy import stats

    n = len(data)
    x_bar = np.mean(data)
    std_dev = np.std(data)
    z_scores = [(x - x_bar) / std_dev for x in data]
    probabilities = [stats.norm.cdf(z) for z in z_scores]
    expected_count = n * (1 - probabilities)
    return expected_count < criterion_val


def generalised_esd_test(data, alpha=0.05):
    import numpy as np
    import pandas as pd
    from statsmodels.stats.stattools import test_esd

    test_result = test_esd(data, alpha=alpha)
    return test_result


def calculate_mahalanobis_distance(data):
    import numpy as np
    import pandas as pd
    from scipy.spatial.distance import mahalanobis

    x_bar = np.mean(data, axis=0)
    cov_matrix = np.cov(data, rowvar=False)
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    distances = [mahalanobis(x, x_bar, inv_cov_matrix) for x in data]
    return distances


