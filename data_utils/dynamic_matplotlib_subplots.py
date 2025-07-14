#!/usr/bin/env python3

def dynamic_subplot(data_to_use, categories_list, figsize=(16.18, 10), figure_title=None, xlabel_title=None, ylabel_title=None):
    """
    Create a dynamic subplot matrix using matplotlib.

    Parameters
    ----------
    data_to_use : list
        A list of lists, each containing the data to plot.
    categories_list : list
        A list of strings, each containing the title of the subplots.
    figsize : tuple, optional
        The size of the figure in inches. Default is (16.18, 10).
    figure_title : str, optional
        The title of the figure. Default is None.
    xlabel_title : str, optional
        The title of the x-axis. Default is None.
    ylabel_title : str, optional
        The title of the y-axis. Default is None.

    Returns
    -------
    None
    """
    import matplotlib.pyplot as plt

    length_data = len(data_to_use)
    length_categories = len(categories_list)
    nrows = -int((-(length_data ** 1/2)) // 1)
    ncols = length_categories

    fig, axs = plt.subplots(nrows, ncols, figsize)
    fig.suptitle(f"{figure_title}")

    for i, item in enumerate(categories_list):
        ax = axs[i // nrows, i % ncols]
        ax.plot(data_to_use[i], marker="o")
        ax.set_title(item)
        ax.set_xlabel(f"{xlabel_title}")
        ax.set_ylabel(f"{ylabel_title}")

    plt.tight_layout()
    plt.show()

def dynamic_subplot_extended(data_to_use, categories_list, figsize=(16.18, 10), figure_title=None, xlabel_title=None, ylabel_title=None):
    """
    Create a grid of subplots with dynamic titles and labels.

    Parameters
    ----------
    data_to_use : list
        A list of data series, where each series will be plotted in a separate subplot.
    categories_list : list
        A list of strings, each representing the title of a subplot corresponding to the data series.
    figsize : tuple, optional
        The size of the entire figure in inches. Default is (16.18, 10).
    figure_title : str, optional
        An optional title for the entire figure. Default is None.
    xlabel_title : str, optional
        An optional label for the x-axis of each subplot. Default is None.
    ylabel_title : str, optional
        An optional label for the y-axis of each subplot. Default is None.

    Returns
    -------
    None
    """
    import matplotlib.pyplot as plt
    import numpy as np

    length_data = len(data_to_use)
    ncols = len(categories_list)
    nrows = int(np.ceil(length_data / ncols))

    fig, axs = plt.subplots(nrows, ncols, figsize)
    fig.suptitle(figure_title if figure_title else "", fontsize=16)

    for i, (data, item) in enumerate(zip(data_to_use, categories_list)):
        ax = axs.flat[i]
        ax.plot(data, marker="o")
        ax.set_title(item)
        ax.set_xlabel(xlabel_title if xlabel_title else "")
        ax.set_ylabel(ylabel_title if ylabel_title else "")

    plt.tight_layout()
    plt.show()
