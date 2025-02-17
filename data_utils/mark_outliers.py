#!/usr/bin/env python3
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from os.path import realpath as realpath
from sklearn.neighbors import LocalOutlierFactor
# Monkey patching NumPy for compatibility with newer versions
np.float = np.float64
np.int = np.int_
np.object = np.object_
np.bool = np.bool_

pd.set_option("mode.copy_on_write", True)

GOLDEN_RATIO = 1.618033989
FIG_WIDTH = 20
FIG_HEIGHT = FIG_WIDTH / GOLDEN_RATIO
FIG_SIZE = (FIG_WIDTH, FIG_HEIGHT)
FIG_DPI = 72

# rcParams for Plotting
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = FIG_SIZE
plt.rcParams["figure.dpi"] = FIG_DPI


def plot_binary_outliers(dataset, col, outlier_col, reset_index):
    """ Plot outliers in case of a binary outlier score. Here, the col specifies the real data
    column and outlier_col the columns with a binary value (outlier or not).

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): Column that you want to plot
        outlier_col (string): Outlier column marked with true/false
        reset_index (bool): whether to reset the index for plotting
    """
    dataset = dataset.dropna(axis=0, subset=[col, outlier_col])
    dataset[outlier_col] = dataset[outlier_col].astype("bool")
    if reset_index:
        dataset = dataset.reset_index()
    fig, ax = plt.subplots()
    plt.xlabel("samples")
    plt.ylabel("value")
    # Plot non outliers in default colour
    ax.plot(
        dataset.index[~dataset[outlier_col]],
        dataset[col][~dataset[outlier_col]],
        "+",
    )
    # Plot data points that are outliers in red
    ax.plot(
        dataset.index[dataset[outlier_col]],
        dataset[col][dataset[outlier_col]],
        "r+",
    )
    plt.legend(
        ["normal " + col, "outlier " + col],
        loc="upper center",
        ncol=2,
        fancybox=True,
        shadow=True,
    )
    plt.show()


# Mark with Interquartile range (IQR) function - (distribution based)
def mark_outliers_iqr(dataset, col):
    """Function to mark values as outliers using the IQR method.

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want to apply outlier detection to

    Returns:
        pd.DataFrame: The original DataFrame with an extra boolean column
        indicating whether the value is an outlier or not.
    """
    dataset = dataset.copy()
    Q1 = dataset[col].quantile(0.25)
    Q3 = dataset[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    dataset[col + "_outlier"] = (dataset[col] < lower_bound) | (
        dataset[col] > upper_bound
    )

    return dataset


# Mark with Chauvenet's criterion function - (distribution based)
def mark_outliers_chauvenet(dataset, col, C=2):
    """Finds outliers in the specified column of DataTable and adds a binary column with
    the same name extended with '_outlier' that expresses the result per data point.

    Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/Chapter3/OutlierDetection.py

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want to apply outlier detection to
        C (int, optional): Degree of certainty for the identification of outliers given the assumption
                           of a normal distribution, typically between 1 - 10. Defaults to 2.

    Returns:
        pd.DataFrame: The original DataFrame with an extra boolean column
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()
    # Compute the mean and standard deviation.
    x_bar = dataset[col].mean()
    std = dataset[col].std()
    sample_size = len(dataset.index)
    criterion = 1.0 / (C * sample_size)

    # Consider the deviation for the data points.
    deviation = abs(dataset[col] - x_bar) / std

    # Express the upper and lower bounds.
    low = -deviation / math.sqrt(C)
    high = deviation / math.sqrt(C)
    probability = []
    mask = []

    # Pass all rows in the dataset.
    for i in range(0, sample_size):
        # Determine the probability of observing the point
        probability.append(
            1.0 - 0.5 * (scipy.special.erf(high[i]) - scipy.special.erf(low[i]))
        )
        # And mark as an outlier when the probability is below our criterion.
        mask.append(probability[i] < criterion)
    dataset[col + "_outlier"] = mask
    return dataset


# Mark with Local outlier factor (LOF) function - (distance based)
def mark_outliers_lof(dataset, columns, num_neighbours=20):
    """Mark values as outliers using LOF

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want to apply outlier detection to
        n (int, optional): n_neighbours. Defaults to 20.

    Returns:
        pd.DataFrame: The original DataFrame with an extra boolean column
        indicating whether the value is an outlier or not.
    """
    dataset = dataset.copy()

    lof = LocalOutlierFactor(n_neighbors=num_neighbours)
    data = dataset[columns]
    outliers = lof.fit_predict(data)
    X_scores = lof.negative_outlier_factor_

    dataset["outlier_lof"] = outliers == -1
    return dataset, outliers, X_scores


def plot_marked_outliers(dataset, columns, target_outlier_col, reset_index, chauvenet_val=2, num_neighbours=20):
    pass

