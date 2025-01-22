#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Monkey patching NumPy for compatibility for versions>= 1.24
np.float = np.float64
np.int = np.int_
np.object = np.object_
np.bool = np.bool_

RANDOM_SEED = 42
GOLDEN_RATIO = 1.618033989
FIG_WIDTH = 30
FIG_HEIGHT = FIG_WIDTH / GOLDEN_RATIO
FIG_DPI = 72
FONT_SIZE = 30
QUART_LINE_OFFSET_VAL = 0.2
MEDIAN_LINEWIDTH = 5.0
QUARTILE_LINEWIDTH = 3.0


def calculate_quartiles(group):
    """
    Calculate the first, second (median), and third quartiles of a given group.

    Parameters
    ----------
    group : array_like
        The data to calculate the quartiles from.

    Returns
    -------
    array_like
        A 3-element array with the 25th, 50th (median), and 75th percentiles of the input data.
    """
    return np.percentile(group, [25, 50, 75])


def calculate_deciles(group):
    """
    Calculate the 10th, 20th, 30th, 40th, 50th (median), 60th, 70th, 80th, and 90th percentiles of a given group.

    Parameters
    ----------
    group : array_like
        The data to calculate the deciles from.

    Returns
    -------
    array_like
        A 9-element array with the 10th, 20th, 30th, 40th, 50th (median), 60th, 70th, 80th, and 90th percentiles of the input data.
    """
    return np.percentile(group, [10, 20, 30, 40, 50, 60, 70, 80, 90])


def calculate_2SD(group):
    """
    Calculate the 2.5th and 97.5th percentiles (lower and upper normal limits) of a given group.

    Parameters
    ----------
    group : array_like
        The data to calculate the percentiles from.

    Returns
    -------
    array_like
        A 2-element array with the 2nd and 98th percentiles of the input data.
    """
    return np.percentile(group, [2.5, 97.5])


def generate_violin_with_quartiles(
    data,
    category,
    target_values,
    quartiles_colour=["yellow", "green", "blue"],
    quartiles_linestyle=["dashdot", "solid", "dotted"],
):
    """
    Generates a violin plot of the given data, with customised quartiles
    drawn as horizontal lines.

    Parameters
    ----------
    data : pandas.DataFrame
        The data to be plotted.
    category : str
        The column name of the data to be used for the x-axis.
    target_values : str
        The column name of the data to be used for the y-axis.
    quartiles_colour : list of str, optional
        A list of 3 colours for the quartiles. Defaults to ["yellow", "green", "blue"].
    quartiles_linestyle : list of str, optional
        A list of 3 linestyles for the quartiles. Defaults to ["dashdot", "solid", "dotted"].

    Returns
    -------
    None
    """
    # Get category label
    category_label = category
    # Get the values of the quartiles
    quartiles = data.groupby(f"{category}")[f"{target_values}"].apply(
        calculate_quartiles
    )

    # Setup and plot figure without default quartile lines
    plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=FIG_DPI)
    sns.violinplot(
        x=category,
        y=target_values,
        data=data,
        order=np.sort(data[category].unique()),
        inner=None,
        palette="muted",
    )

    # Add customised quartile lines
    for i, category in enumerate(data[category].unique()):
        q1, q2, q3 = quartiles[category]
        plt.hlines(
            y=q1,
            xmin=i - QUART_LINE_OFFSET_VAL,
            xmax=i + QUART_LINE_OFFSET_VAL,
            colors=quartiles_colour[0],
            linestyles=quartiles_linestyle[0],
            linewidth=QUARTILE_LINEWIDTH,
        )
        plt.hlines(
            y=q2,
            xmin=i - QUART_LINE_OFFSET_VAL,
            xmax=i + QUART_LINE_OFFSET_VAL,
            colors=quartiles_colour[1],
            linestyles=quartiles_linestyle[1],
            linewidth=MEDIAN_LINEWIDTH,
        )
        plt.hlines(
            y=q3,
            xmin=i - QUART_LINE_OFFSET_VAL,
            xmax=i + QUART_LINE_OFFSET_VAL,
            colors=quartiles_colour[2],
            linestyles=quartiles_linestyle[2],
            linewidth=QUARTILE_LINEWIDTH,
        )

    # Add titles, labels and legends
    plt.tick_params(axis="both", which="major", labelsize=FONT_SIZE)
    plt.title(
        f"Violin Plot of {category_label.title()} vs {target_values.title()} with Custom Quartiles as Horizontal Lines",
        fontsize=(FONT_SIZE + 10),
    )
    plt.xlabel(f"{category_label}", fontsize=FONT_SIZE)
    plt.ylabel(f"{target_values}", fontsize=FONT_SIZE)
    plt.show()


def generate_violin_with_quartiles_plus_extremes(
    data,
    category,
    target_values,
    quartiles_colour=["yellow", "green", "blue"],
    quartiles_linestyle=["dashdot", "solid", "dotted"],
):
    """
    Plots a violin plot of the given data with the given category and target values, with customised quartiles and SD cut-offs as horizontal lines.

    Parameters
    ----------
    data : pandas.DataFrame
        The data to plot.
    category : str
        The category to plot.
    target_values : str
        The target values to plot.
    quartiles_colour : list of str, optional
        A list of 3 colours for the quartiles. Defaults to ["yellow", "green", "blue"].
    quartiles_linestyle : list of str, optional
        A list of 3 linestyles for the quartiles. Defaults to ["dashdot", "solid", "dotted"].

    Returns
    -------
    None
    """
    # Get category label
    category_label = category
    # Get the values of the quartiles
    quartiles = data.groupby(f"{category}")[f"{target_values}"].apply(
        calculate_quartiles
    )
    extremes = data.groupby(f"{category}")[f"{target_values}"].apply(calculate_2SD)

    # Setup and plot figure without default quartile lines
    plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=FIG_DPI)
    sns.violinplot(
        x=category,
        y=target_values,
        data=data,
        order=np.sort(data[category].unique()),
        inner=None,
        palette="muted",
    )

    # Add customised quartile lines
    for i, category in enumerate(data[category].unique()):
        q1, q2, q3 = quartiles[category]
        minus_2SD, plus_2SD = extremes[category]
        plt.hlines(
            y=minus_2SD,
            xmin=i - QUART_LINE_OFFSET_VAL,
            xmax=i + QUART_LINE_OFFSET_VAL,
            colors="red",
            linestyles="solid",
            linewidth=QUARTILE_LINEWIDTH,
        )
        plt.hlines(
            y=q1,
            xmin=i - QUART_LINE_OFFSET_VAL,
            xmax=i + QUART_LINE_OFFSET_VAL,
            colors=quartiles_colour[0],
            linestyles=quartiles_linestyle[0],
            linewidth=QUARTILE_LINEWIDTH,
        )
        plt.hlines(
            y=q2,
            xmin=i - QUART_LINE_OFFSET_VAL,
            xmax=i + QUART_LINE_OFFSET_VAL,
            colors=quartiles_colour[1],
            linestyles=quartiles_linestyle[1],
            linewidth=MEDIAN_LINEWIDTH,
        )
        plt.hlines(
            y=q3,
            xmin=i - QUART_LINE_OFFSET_VAL,
            xmax=i + QUART_LINE_OFFSET_VAL,
            colors=quartiles_colour[2],
            linestyles=quartiles_linestyle[2],
            linewidth=QUARTILE_LINEWIDTH,
        )
        plt.hlines(
            y=plus_2SD,
            xmin=i - QUART_LINE_OFFSET_VAL,
            xmax=i + QUART_LINE_OFFSET_VAL,
            colors="red",
            linestyles="solid",
            linewidth=QUARTILE_LINEWIDTH,
        )

    # Add titles, labels and legends
    plt.tick_params(axis="both", which="major", labelsize=FONT_SIZE)
    plt.title(
        f"Violin Plot of {category_label.title()} vs {target_values.title()} with Custom Quartiles and SD Cut-offs as Horizontal Lines",
        fontsize=(FONT_SIZE + 10),
    )
    plt.xlabel(f"{category_label}", fontsize=FONT_SIZE)
    plt.ylabel(f"{target_values}", fontsize=FONT_SIZE)
    plt.show()


def generate_violin_plot_with_boxplot(data, category, target_values):
    """
    Generates a violin plot with a superimposed boxplot of the given data.

    Parameters
    ----------
    data : pandas.DataFrame
        The data to be plotted.
    category : str
        The column name of the data to be used for the x-axis.
    target_values : str
        The column name of the data to be used for the y-axis.

    Returns
    -------
    None
    """
    # Get category label
    category_label = category

    # Setup and plot figure without default quartile lines
    plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=FIG_DPI)
    ax = sns.violinplot(
        x=category,
        y=target_values,
        data=data,
        order=np.sort(data[category].unique()),
        inner=None,
        palette="muted",
    )
    sns.boxplot(
        x=category,
        y=target_values,
        data=data,
        order=np.sort(data[category].unique()),
        palette="rocket",
        saturation=0.5,
        width=0.4,
        boxprops={"zorder": 2},
        ax=ax,
    )

    # Add titles, labels and legends
    plt.tick_params(axis="both", which="major", labelsize=FONT_SIZE)
    plt.title(
        f"Violin Plot of {category_label.title()} vs {target_values.title()} with Superimposed Boxplot",
        fontsize=(FONT_SIZE + 10),
    )
    plt.xlabel(f"{category_label}", fontsize=FONT_SIZE)
    plt.ylabel(f"{target_values}", fontsize=FONT_SIZE)
    plt.show()


np.random.seed(RANDOM_SEED)
data_to_use = pd.DataFrame(
    {
        "Group": np.random.choice(["A", "B", "C", "D"], size=200),
        "Values": np.random.randn(200),
    }
)
print(data_to_use)

generate_violin_with_quartiles(
    data=data_to_use, category="Group", target_values="Values"
)
generate_violin_with_quartiles_plus_extremes(
    data=data_to_use, category="Group", target_values="Values"
)
generate_violin_plot_with_boxplot(
    data=data_to_use, category="Group", target_values="Values"
)
