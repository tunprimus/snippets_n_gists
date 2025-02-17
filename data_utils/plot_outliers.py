#!/usr/bin/env python3
# Modified from https://sustainabilitymethods.org/index.php/Outlier_Detection_in_Python
import numpy as np
import pandas as pd
import seaborn as sns
from os.path import realpath as realpath
from scipy.special import erfc as ss_erfc
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
plt.style.use("seaborn")
plt.rcParams["figure.figsize"] = FIG_SIZE
plt.rcParams["figure.dpi"] = FIG_DPI


def plot_outliers_by_quantile(dataset, col, lower_q=0.025, upper_q=0.975):
    dataset = dataset.copy()
    lower_value = dataset[col].quantile(lower_q)
    upper_value = dataset[col].quantile(upper_q)
    mask = (lower_value <= dataset[col]) & (dataset[col] <= upper_value)
    fig, ax = plt.subplots(figsize=(FIG_SIZE), dpi=FIG_DPI)
    fig.suptitle("Quantile-Based Outlier Detection", size=20)
    sns.scatterplot(data=dataset, y=col, hue=np.where(mask, "No Outlier", "Outlier"), ax=ax)
    plt.tight_layout()
    plt.show()


def plot_outliers_by_univariate_chauvenet(dataset, criterion=None):
    list_of_chauvenet_criteria = [0.01, 0.005, 0.001]
    dataset = dataset.copy()
    dataset_size = len(dataset)
    x_bar = dataset.mean()
    std = dataset.std()
    if not criterion:
        criterion = 1 / (2 * dataset_size)
    deviation = np.abs(dataset - x_bar) / std
    probabilities = ss_erfc(deviation)
    # mask = probabilities < criterion
    fig, axs = plt.subplots(ncols=3, figsize=FIG_SIZE, dpi=FIG_DPI)
    fig.suptitle("Univariate Chauvenet Criterion", size=20)
    for i, c in enumerate(list_of_chauvenet_criteria):
        mask = probabilities < c
        sns.scatterplot(data=dataset, y=col, hue=np.where(mask, "Outlier", "No Outlier"), hue_order=["No Outlier", "Outlier"], ax=axs[i])
        axs[i].set(title=f"Criterion = {c} with {sum(mask)} Outliers")
    plt.tight_layout()
    plt.show()

