#!/usr/bin/env python3
# Adapted from: Statistical Methods for Anomaly Detection: A Comprehensive Guide -> https://toxigon.com/statistical-methods-for-anomaly-detection , Top 5 Statistical Techniques to Detect and Handle Outliers in Data -> https://www.statology.org/top-5-statistical-techniques-detect-handle-outliers-data/ , Grubbs’ Test: A Comprehensive Guide to Detecting Outliers -> https://diogoribeiro7.github.io/statistics/grubbs_test_comprehensive_guide_detecting_outliers/
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import mahalanobis
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


def detect_outliers_z_score(data, threshold=3, messages=True):
    import numpy as np
    import pandas as pd

    # Compute z-scores
    x_bar = np.mean(data)
    std_dev = np.std(data)
    # z_scores = [(x - x_bar) / std_dev for x in data]
    z_scores = (data - x_bar) / std_dev
    # Find values and indices of outliers
    outlier_indices = np.where(np.abs(z_scores) > threshold)[0]
    outlier_values = data[outlier_indices]
    if messages:
        print(f"{len(outlier_indices)} Outliers Detected")
        print("Z-Score Outliers (Index, Value):")
        for idx, val in zip(outlier_indices, outlier_values):
            print(f"Index: {idx}, Value: {val}")
    return outlier_values, outlier_indices


def detect_outliers_modified_z_score(data, threshold=3.5, messages=True):
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
    outlier_indices = np.where(np.abs(modified_z_scores) > threshold)[0]
    outlier_values = data[outlier_indices]
    if messages:
        print(f"{len(outlier_indices)} Outliers Detected")
        print("Z-Score Outliers (Index, Value):")
        for idx, val in zip(outlier_indices, outlier_values):
            print(f"Index: {idx}, Value: {val}")
    return outlier_values, outlier_indices


def detect_outliers_iqr(data, messages=True):
    import numpy as np
    import pandas as pd

    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    lo_bound = q1 - 1.5 * iqr
    hi_bound = q3 + 1.5 * iqr

    outlier_indices = np.where((data < lo_bound) | (data > hi_bound))[0]
    outlier_values = data[outlier_indices]
    if messages:
        print(f"{len(outlier_indices)} Outliers Detected")
        print("Z-Score Outliers (Index, Value):")
        for idx, val in zip(outlier_indices[0], outlier_values):
            print(f"Index: {idx}, Value: {val}")
    return outlier_values, outlier_indices


def detect_outlier_grubbs_test(data, alpha=0.05, messages=True):
    import numpy as np
    import pandas as pd
    from scipy import stats

    n = len(data)
    x_bar = np.mean(data)
    std_dev = np.std(data)

    # Find the maximum absolute deviation from the mean
    abs_deviation = np.abs(data - x_bar)
    max_deviation = np.max([np.abs(x - x_bar) for x in data])
    outlier_value = data[np.argmax(abs_deviation)]

    # Calculate the Grubbs' test statistic
    test_statistic = max_deviation / std_dev

    # Calculate the critical value using the t-distribution
    test_crit = stats.t.ppf(1 - alpha / (2 * n), n - 2)
    critical_value = ((n - 1) / np.sqrt(n)) * np.sqrt(test_crit ** 2 / (n - 2 + test_crit ** 2))

    if messages:
        if outlier_value:
            print(f"Outlier detected: {outlier_value}.\nRemove this value and check again for another outlier.")
        else:
            print("No outlier detected")
    # Compare the test statistic with the critical value
    if test_statistic > critical_value:
        return outlier_value, test_statistic, critical_value
    else:
        return None, test_statistic, critical_value



def dixons_q_test(data, sig_level=0.05, messages=True):
    Q_CRITICAL_TABLE = {
        3: {0.90: 0.941, 0.95: 0.970, 0.99: 0.994,},
        4: {0.90: 0.765, 0.95: 0.829, 0.99: 0.926,},
        5: {0.90: 0.642, 0.95: 0.710, 0.99: 0.821,},
        6: {0.90: 0.560, 0.95: 0.625, 0.99: 0.740,},
        7: {0.90: 0.507, 0.95: 0.568, 0.99: 0.680,},
        8: {0.90: 0.468, 0.95: 0.526, 0.99: 0.634,},
        9: {0.90: 0.437, 0.95: 0.493, 0.99: 0.598,},
        10: {0.90: 0.412, 0.95: 0.466, 0.99: 0.568,},
        11: {0.90: 0.392, 0.95: 0.444, 0.99: 0.542,},
        12: {0.90: 0.376, 0.95: 0.426, 0.99: 0.522,},
        13: {0.90: 0.361, 0.95: 0.410, 0.99: 0.503,},
        14: {0.90: 0.349, 0.95: 0.396, 0.99: 0.488,},
        15: {0.90: 0.338, 0.95: 0.384, 0.99: 0.475,},
        16: {0.90: 0.329, 0.95: 0.374, 0.99: 0.463,},
        17: {0.90: 0.320, 0.95: 0.365, 0.99: 0.452,},
        18: {0.90: 0.313, 0.95: 0.356, 0.99: 0.442,},
        19: {0.90: 0.306, 0.95: 0.349, 0.99: 0.433,},
        20: {0.90: 0.300, 0.95: 0.342, 0.99: 0.425,},
        21: {0.90: 0.295, 0.95: 0.337, 0.99: 0.418,},
        22: {0.90: 0.290, 0.95: 0.331, 0.99: 0.411,},
        23: {0.90: 0.285, 0.95: 0.326, 0.99: 0.404,},
        24: {0.90: 0.281, 0.95: 0.321, 0.99: 0.399,},
        25: {0.90: 0.277, 0.95: 0.317, 0.99: 0.393,},
        26: {0.90: 0.273, 0.95: 0.312, 0.99: 0.388,},
        27: {0.90: 0.269, 0.95: 0.308, 0.99: 0.384,},
        28: {0.90: 0.266, 0.95: 0.305, 0.99: 0.380,},
        29: {0.90: 0.263, 0.95: 0.301, 0.99: 0.376,},
        30: {0.90: 0.260, 0.95: 0.290, 0.99: 0.372,},
    }

    length_data = len(data)
    if (length_data < 3) or (length_data > 30):
        raise ValueError("Data size must be between 3 and 30")
    if sig_level == 0.05:
        Q_critical = Q_CRITICAL_TABLE[length_data][0.95]
    elif sig_level == 0.10:
        Q_critical = Q_CRITICAL_TABLE[length_data][0.90]
    elif sig_level == 0.01:
        Q_critical = Q_CRITICAL_TABLE[length_data][0.99]
    else:
        raise ValueError("Significance level must be 0.01, 0.05 or 0.10")

    # Sort data in ascending order
    sorted_data = np.sort(data)
    gap_low = abs(sorted_data[1] - sorted_data[0])
    gap_high = abs(sorted_data[-1] - sorted_data[-2])
    data_range = sorted_data[-1] - sorted_data[0]

    # Calculate Q-value
    q_low = gap_low / data_range
    q_high = gap_high / data_range

    # Compare Q statistics with the critical value
    if q_high > Q_critical:
        if messages:
            print(f"Highest value: {sorted_data[-1]} is an outlier.\nRemove this value only.")
        return sorted_data[-1], q_high, Q_critical
    elif q_low > Q_critical:
        if messages:
            print(f"Lowest value: {sorted_data[0]} is an outlier.\nRemove this value only.")
        return sorted_data[0], q_low, Q_critical
    else:
        if messages:
            print("No outliers detected.")
        return None, max(q_low, q_high), Q_critical


def detect_outliers_tukeys_fences(data, k=1.5, messages=True):
    import numpy as np
    import pandas as pd

    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    lo_bound = q1 - k * iqr
    hi_bound = q3 + k * iqr

    outlier_indices = np.where((data < lo_bound) | (data > hi_bound))[0]
    outlier_values = data[outlier_indices]
    if messages:
        print(f"{len(outlier_indices)} Outliers Detected")
        print("Tukey's fences Outliers (Index, Value):")
        for idx, val in zip(outlier_indices, outlier_values):
            print(f"Index: {idx}, Value: {val}")
    return outlier_values, outlier_indices


def detect_outliers_chauvenets_criterion(data, criterion_val=0.1, return_filtered_data=False, messages=True):
    import numpy as np
    import pandas as pd
    from scipy import stats

    num_data_points = len(data)
    x_bar = np.mean(data)
    std_dev = np.std(data)

    if not criterion_val:
        criterion_val = 1.0 / (2 * num_data_points)

    # Compute z-scores
    z_scores = np.abs((data - x_bar) / std_dev)

    # Calculate the corresponding probabilities (two-tailed)
    probabilities = 1 - stats.norm.cdf(z_scores)

    # Detect outliers: points where the probability is less than the criterion value
    outlier_indices = np.where(probabilities < criterion_val)[0]
    outlier_values = data[outlier_indices]
    try:
        filtered_data = np.delete(data, outlier_indices)
    except Exception:
        filtered_data = data[probabilities >= criterion_val]
    if messages:
        print(f"{len(outlier_indices)} Outliers Detected")
        print("Chauvenet's Outliers (Index, Value):")
        for idx, val in zip(outlier_indices, outlier_values):
            print(f"Index: {idx}, Value: {val}")
    if return_filtered_data:
        return outlier_values, outlier_indices, filtered_data
    else:
        return outlier_values, outlier_indices


def detect_outliers_mahalanobis_distance(data, alpha=0.05, messages=True):
    import numpy as np
    import pandas as pd
    from scipy.spatial.distance import mahalanobis

    if np.ndim(data) < 2:
        raise ValueError("Data must be at least  2D array.")

    x_bar = np.mean(data, axis=0)
    cov_matrix = np.cov(data, rowvar=False)
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    distances = [mahalanobis(x, x_bar, inv_cov_matrix) for x in data]

    # Calculate the threshold for identifying outliers
    threshold = np.quantile(distances, 1 - alpha)
    outlier_indices = np.where(distances > threshold)[0]
    outlier_values = data[outlier_indices]
    if messages:
        print(f"{len(outlier_indices)} Outliers Detected")
        print("Mahalanobis Distance Outliers (Index, Value):")
        for idx, val in zip(outlier_indices, outlier_values):
            print(f"Index: {idx}, Value: {val}")
    return outlier_indices, outlier_values


def detect_outliers_lof(data, num_neighbours=19, messages=True):
    import numpy as np
    import pandas as pd
    from sklearn.neighbors import LocalOutlierFactor

    # Reshape the 1D array for LOF (it expects a 2D array as input)
    reshaped_data = data.reshape(-1, 1)

    # Create an instance of LocalOutlierFactor
    lof = LocalOutlierFactor(n_neighbors=num_neighbours)

    # Compute the local outlier factor for each data point
    lof_scores = lof.fit_predict(reshaped_data)

    # Get the indices of the outliers and the outlier values
    outlier_indices = np.where(lof_scores == -1)[0]
    outlier_values = data[outlier_indices]
    if messages:
        print(f"{len(outlier_indices)} Outliers Detected")
        print("Local Outlier Factor Outliers (Index, Value):")
        for idx, val in zip(outlier_indices, outlier_values):
            print(f"Index: {idx}, Value: {val}")
    return outlier_indices, outlier_values


def detect_outliers_iso_forest(data, contamination=0.05, messages=True):
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import IsolationForest

    # Reshape the 1D array for LOF (it expects a 2D array as input)
    reshaped_data = data.reshape(-1, 1)

    iso = IsolationForest(contamination=contamination, random_state=42)

    # Compute the local outlier factor for each data point
    iso_scores = iso.fit_predict(reshaped_data)

    # Get the indices of the outliers and the outlier values
    outlier_indices = np.where(iso_scores == -1)[0]
    outlier_values = data[outlier_indices]
    if messages:
        print(f"{len(outlier_indices)} Outliers Detected")
        print("Isolation Forest Outliers (Index, Value):")
        for idx, val in zip(outlier_indices, outlier_values):
            print(f"Index: {idx}, Value: {val}")
    return outlier_indices, outlier_values


#-----------------#
#      Tests      #
#-----------------#
np.random.seed(42)

# Generate trimodal data
low_spending = np.random.normal(1, 1073, 17)
normal_spending = np.random.normal(10000, 1500, 500)
high_spending = np.random.normal(20000, 1073, 17)
spending_data = np.concatenate([normal_spending, low_spending, high_spending])

# Plot the data
# plt.hist(spending_data, bins=50)
plt.scatter(spending_data, np.zeros(len(spending_data)), c="k", s=1)
plt.xlabel("Spending")
plt.ylabel("Frequency")
plt.title("Spending Data")
plt.show()


detect_outliers_z_score(spending_data)

modified_z_score(spending_data)

detect_outliers_iqr(spending_data)

detect_outliers_grubbs_test(spending_data)

detect_outlier_grubbs_test(spending_data)

dixons_q_test(spending_data[17:45])

detect_outliers_tukeys_fences(spending_data)

detect_outliers_chauvenets_criterion(spending_data)

detect_outliers_mahalanobis_distance(spending_data.reshape)
detect_outliers_mahalanobis_distance(spending_data.reshape(-1, 2))

detect_outliers_lof(spending_data)

detect_outliers_iso_forest(spending_data)



