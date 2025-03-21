#!/usr/bin/env python3
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os.path import realpath as realpath
from scipy.special import erf as ss_erf
from scipy.stats import zscore
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

def plot_outliers_by_iqr(data_to_use, target_col):
    marked_dataset = mark_outliers_iqr(dataset=data_to_use, col=target_col)
    plot_binary_outliers(dataset=marked_dataset, col=target_col, outlier_col=target_col + "_outlier", reset_index=True)


# Mark with Quantile function - (distribution based)
def mark_outliers_quantile(dataset, col, lower_q=0.025, upper_q=0.975):
    dataset = dataset.copy()
    lower_value = dataset[col].quantile(lower_q)
    upper_value = dataset[col].quantile(upper_q)

    dataset[col + "_outlier"] = (dataset[col] < lower_value) | (
        dataset[col] > upper_value
    )
    return dataset

def plot_outliers_by_quantile(data_to_use, target_col):
    marked_dataset = mark_outliers_quantile(dataset=data_to_use, col=target_col)
    plot_binary_outliers(dataset=marked_dataset, col=target_col, outlier_col=target_col + "_outlier", reset_index=True)



def mark_outliers_by_z_score(dataset, col, threshold=3):
    dataset["z_score"] = zscore(dataset[col])
    dataset[col + "_outlier"] = dataset[dataset["z_score"].abs() > threshold]
    return dataset

def plot_outliers_by_z_score(data_to_use, target_col):
    marked_dataset = mark_outliers_by_z_score(dataset=data_to_use, col=target_col)
    plot_binary_outliers(dataset=marked_dataset, col=target_col, outlier_col=target_col + "_outlier", reset_index=True)



# Mark with univariate Chauvenet's criterion function - (distribution based)
def mark_outliers_univariate_chauvenet(dataset, col, chauvenet_cutoff=[2,3,4,5,6,7,8,9,10]):
    dataset = dataset.copy()
    # Compute the mean and standard deviation.
    x_bar = dataset[col].mean()
    std = dataset[col].std()
    sample_size = len(dataset.index)
    if chauvenet_cutoff is None:
        criterion = 1.0 / (2 * sample_size)
    elif isinstance(chauvenet_cutoff, List) and len(chauvenet_cutoff) == 1:
        criterion = 1.0 / (chauvenet_cutoff * sample_size)

        # Consider the deviation for the data points.
        deviation = np.abs(dataset[col] - x_bar) / std

        # Express the upper and lower bounds.
        low = -deviation / math.sqrt(C)
        high = deviation / math.sqrt(C)
        probability = []
        mask = []

        # Pass all rows in the dataset.
        for i in range(0, sample_size):
            # Determine the probability of observing the point
            probability.append(
                1.0 - 0.5 * (ss_erf(high[i]) - ss_erf(low[i]))
            )
            # And mark as an outlier when the probability is below our criterion.
            mask.append(probability[i] < criterion)
        dataset[col + "_outlier"] = mask
    elif isinstance(chauvenet_cutoff, List) and len(chauvenet_cutoff) > 1:
        for cc_val in chauvenet_cutoff:
            criterion = 1.0 / (cc_val * sample_size)

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
                    1.0 - 0.5 * (ss_erf(high[i]) - ss_erf(low[i]))
                )
                # And mark as an outlier when the probability is below our criterion.
                mask.append(probability[i] < criterion)
            dataset[col + "_outlier" + f"_{cc_val}"] = mask
    return dataset

def plot_outliers_by_univariate_chauvenet(data_to_use, target_col, chauvenet_val=2):
    marked_dataset = mark_outliers_univariate_chauvenet(dataset=data_to_use, col=target_col)
    plot_binary_outliers(dataset=marked_dataset, col=target_col, outlier_col=target_col + "_outlier", reset_index=True)

# Mark with Local outlier factor (LOF) function - (distance based)
def mark_outliers_lof(dataset, col, num_neighbours=[5, 7, 13, 19, 23]):
    dataset = dataset.copy()

    if not num_neighbours:
        num_neighbours = 20
    elif isinstance(num_neighbours, List) and len(num_neighbours) == 1:
        lof = LocalOutlierFactor(n_neighbors=num_neighbours[0])
        data = dataset[col]
        outliers = lof.fit_predict(data)
        X_scores = lof.negative_outlier_factor_
        dataset["outlier_lof"] = outliers == -1
    elif isinstance(num_neighbours, List) and len(num_neighbours) > 1:
        for nn_val in num_neighbours:
            lof = LocalOutlierFactor(n_neighbors=nn_val)
            outliers = lof.fit_predict(data)
            X_scores = lof.negative_outlier_factor_
            dataset[col + "_outlier_lof" + f"_{nn_val}"]
    return dataset, outliers, X_scores

def plot_outliers_by_lof(data_to_use, target_col):
    marked_dataset = mark_outliers_lof(dataset=data_to_use, columns=target_col)
    plot_binary_outliers(dataset=marked_dataset, col=target_col, outlier_col=target_col + "_outlier", reset_index=True)


def plot_marked_outliers(dataset, list_of_cols_to_check, reset_index, chauvenet_val=[2], num_neighbours=20):
    if not list_of_cols_to_check:
        all_columns = dataset.columns
        list_of_cols_to_check = [all_columns[item] for item in all_columns if "outlier" in str(all_columns[item])]
    for col in list_of_cols_to_check:
        plot_outliers_by_iqr(dataset, col)
        plot_outliers_by_quantile(dataset, col)
        plot_outliers_by_z_score(dataset, col)
        plot_outliers_by_univariate_chauvenet(dataset, col)
        plot_outliers_by_lof(dataset, col)

