#!/usr/bin/env python3
from matplotlib import rcParams

# Define constants
GOLDEN_RATIO = 1.618033989
FIG_WIDTH = 12
FIG_HEIGHT = FIG_WIDTH / GOLDEN_RATIO
FIG_SIZE = (FIG_WIDTH, FIG_HEIGHT)
FIG_DPI = 72
RANDOM_SAMPLE_SIZE = 13
RANDOM_SEED = 42
ALPHA_VALUE = 0.05

# Plotting parameters
rcParams["figure.dpi"] = FIG_DPI
rcParams["savefig.format"] = "svg"


def plot_bullet_graph(actual_sale, target_sale, acceptable_range):
    """
    Plots a bullet graph representing actual sales against target sales with an acceptable range.

    Parameters
    ----------
    actual_sale : int, float, list, or numpy.ndarray
        The actual sales value(s). Can be a single number or a list/array of numbers.
    target_sale : int, float, list, or numpy.ndarray
        The target sales value(s). Must be the same type and length as `actual_sale`.
    acceptable_range : list or tuple
        A two-element sequence defining the minimum and maximum acceptable sales range.

    Raises
    ------
    TypeError
        If `acceptable_range` is not a list or tuple, or if `actual_sale` and `target_sale`
        are not numbers or lists/arrays of numbers.
    ValueError
        If `acceptable_range` does not contain exactly two values, or if the second value
        is not greater than the first. Also raised if `actual_sale` and `target_sale` lists/arrays
        are of different lengths.

    Notes
    -----
    The function generates a bullet graph for either a single pair of sales values or multiple pairs.
    The graph visualises actual sales against target sales and highlights the acceptable sales range.

    Example
    -------
    >>> actual_sale01 = 250
    >>> target_sale01 = 300
    >>> acceptable_range = [200, 350]
    >>> plot_bullet_graph(actual_sale01, target_sale01, acceptable_range)
    >>>
    >>> actual_sale02 = [250, 500, 400, 100]
    >>> target_sale02 = [300, 300, 300, 300]
    >>> plot_bullet_graph(actual_sale02, target_sale02, acceptable_range)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    if not isinstance(acceptable_range, (list, tuple)):
        raise TypeError("Acceptable range must be a list or tuple.")
    if len(acceptable_range) != 2:
        raise ValueError(
            "Acceptable range must contain exactly two values: min and max."
        )
    if acceptable_range[0] >= acceptable_range[1]:
        raise ValueError(
            "The second value in acceptable range must be greater than the first value."
        )

    if isinstance(actual_sale, (list, np.ndarray)) or isinstance(
        target_sale, (list, np.ndarray)
    ):
        if len(actual_sale) != len(target_sale):
            raise ValueError(
                "The length of actual_sales and target_sales must be the same."
            )

        n_cols = max(int(np.sqrt(len(actual_sale))), 1)
        n_rows = int(np.ceil(len(actual_sale) / n_cols))
        fig, axs = plt.subplots(
            nrows=n_rows,
            ncols=n_cols,
            figsize=(FIG_WIDTH, FIG_HEIGHT / n_rows),
            dpi=FIG_DPI,
        )
        axs = axs.flatten() if n_rows * n_cols > 1 else [axs]

        for i, (act_sale, tar_sale) in enumerate(zip(actual_sale, target_sale)):
            axs[i].barh(0, act_sale, color="blue", label="Actual Sale")
            axs[i].plot(tar_sale, 0, "ro", markersize=8, label="Target Sale")
            axs[i].barh(
                0,
                acceptable_range[1] - acceptable_range[0],
                color="lightgrey",
                left=acceptable_range[0],
                label="Acceptable Range",
            )

            if i == len(axs) - 1:
                axs[i].set_xlabel("Sales Value")

            axs[i].set_yticks([])
            left, right = axs[i].get_xlim()
            right = max(right, acceptable_range[1] * 1.15)
            axs[i].set_xlim(left, right)

            if i == 0:
                axs[i].legend(loc="upper left")

    else:
        if not isinstance(actual_sale, (int, float)) or not isinstance(
            target_sale, (int, float)
        ):
            raise TypeError("Actual sale and target sale must be numbers.")

        fig, ax = plt.subplots(figsize=(8, 2.5), dpi=FIG_DPI)

        ax.barh(0, actual_sale, color="blue", label="Actual Sale")
        ax.plot(target_sale, 0, "ro", markersize=8, label="Target Sale")
        ax.barh(
            0,
            acceptable_range[1] - acceptable_range[0],
            color="lightgrey",
            left=acceptable_range[0],
            label="Acceptable Range",
        )

        ax.set_xlabel("Sale Value")
        ax.set_yticks([])
        left, right = ax.get_xlim()
        right = max(right, acceptable_range[1] * 1.15)
        ax.set_xlim(left, right)
        ax.legend(loc="upper left")

    plt.suptitle("Sales Performance Against Targets", fontsize=14, fontweight="bold")
    plt.show()
