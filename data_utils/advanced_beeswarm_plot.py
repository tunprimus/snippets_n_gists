#!/usr/bin/env python3

# Define constants
GOLDEN_RATIO = 1.618
FIG_WIDTH = 20
FIG_HEIGHT = FIG_WIDTH / GOLDEN_RATIO
FIG_SIZE = (FIG_WIDTH, FIG_HEIGHT)
FIG_DPI = 72

# Function to generate IQR region
def iqr_region_highlighter(data_to_use=None, median_line_colour="green", q_line_colour="red", iqr_colour="red", iqr_alpha=0.2):
    """
    Function to generate IQR region highlighter

    Parameters
    ----------
    data_to_use : array_like
        The data to be plotted
    median_line_colour : str, optional
        The colour of the median line. Defaults to "green".
    q_line_colour : str, optional
        The colour of the Q1 and Q3 lines. Defaults to "red".
    iqr_colour : str, optional
        The colour of the IQR region fill. Defaults to "red".
    iqr_alpha : float, optional
        The alpha (transparency) value of the IQR region fill. Defaults to 0.2.

    Returns
    -------
    None

    Notes
    -----
    This function will generate 3 horizontal lines, one for Q1, one for the median, and one for Q3.
    It will also generate a filled region between Q1 and Q3, with the given colour and alpha value.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    
    # generate the iqr boundary values
    q1 = np.percentile(data_to_use, 25)
    median_of_data = np.median(data_to_use)
    q3 = np.percentile(data_to_use, 75)
    # highlight the IQR region
    plt.axhline(y=q1, color=q_line_colour, linestyle="dashed", label="Q1")
    plt.axhline(y=q3, color=q_line_colour, linestyle="dashed", label="Q3")
    plt.axhline(y=median_of_data, color=median_line_colour, linestyle="-", label="Median")
    plt.fill_between([-0.5, 0.5], q1, q3, color=iqr_colour, alpha=iqr_alpha)

# Function to plot advanced beeswarm plots
def advanced_beeswarm_plot(data_list=None, titles_list=None, swarmplot_colour="blue", plot_fig_size=FIG_SIZE):
    """
    Function to plot advanced beeswarm plots

    Parameters
    ----------
    data_list : array_like or list of array_likes
        The data to be plotted
    titles_list : list of str
        The titles of each subplot
    swarmplot_colour : str, optional
        The colour of the swarmplot. Defaults to "blue".
    plot_fig_size : tuple of int, optional
        The size of the figure. Defaults to (20, 10).

    Returns
    -------
    None

    Notes
    -----
    This function will generate a beeswarm plot of the given data, with IQR region highlighter.
    The IQR region highlighter will generate 3 horizontal lines, one for Q1, one for the median, and one for Q3.
    It will also generate a filled region between Q1 and Q3.
    """
    plt.subplots_adjust(hspace=0.23, wspace=0.23)
    if data_list:
        # n_cols = 3 or int(len(data_list)**0.5)
        # n_rows = len(data_list) // n_cols + (len(data_list) % n_cols > 0)
        n_cols = int(np.sqrt(len(data_list)))
        n_rows = int(np.ceil(len(data_list) / n_cols))
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=True, sharey=True, figsize=FIG_SIZE, dpi=FIG_DPI)
    for i, data in enumerate(data_list):
        ax = plt.subplot(n_rows, n_cols, i+1)
        sns.swarmplot(ax=ax, y=data, size=3)
        iqr_region_highlighter(data_to_use=data)
        ax.set_title(f"Beeswarm Plot: {titles_list[i]}")
    plt.tight_layout
    plt.show()


# Tests
np.random.seed(42)

# Normal Distribution
normal_dist = np.random.normal(5, 1, 1000)

# Bimodal Distribution
bimodal_dist = np.concatenate([np.random.normal(3, 0.5, 500), np.random.normal(7, 0.5, 500)])

# Exponential Distribution
exponential_dist = np.random.exponential(scale=1, size=1000)

# Uniform Distribution
uniform_dist = np.random.uniform(0, 10, 1000)

# Skewed Distribution with Outliers
skewed_main = np.random.chisquare(3, 900)
outliers = [15, 16, 17, 18, 19]
skewed_dist = np.concatenate([skewed_main, outliers])

datasets = [normal_dist, bimodal_dist, exponential_dist, uniform_dist, skewed_dist]
titles = ["Normal Distribution", "Bimodal Distribution", "Exponential Distribution", "Uniform Distribution", "Skewed Distribution with Outliers"]

# Plotting
advanced_beeswarm_plot(data_list=datasets, titles_list=titles)
