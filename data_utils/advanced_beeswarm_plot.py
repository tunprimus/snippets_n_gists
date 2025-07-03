#!/usr/bin/env python3

# Define constants
GOLDEN_RATIO = 1.618033989
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
def advanced_beeswarm_plot(data=None, x=None, y=None, order=None, titles_list=None, swarmplot_colour="blue", plot_fig_size=FIG_SIZE):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns

    plt.subplots_adjust(hspace=0.23, wspace=0.23)

    # Get the numerical column names from the DataFrame
    numerical_columns = [col for col in data.columns if data[col].dtype.kind in 'bifc']

    n_cols = int(np.sqrt(len(numerical_columns)))
    n_rows = int(np.ceil(len(numerical_columns) / n_cols))

    # Create a figure with subplots for each numerical column
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols)

    # Flatten the axs array for easier iteration
    axs = axs.flatten()

    # Iterate over each numerical column and create a beeswarm plot
    for i, col in enumerate(numerical_columns):
        sns.swarmplot(y=col, data=data, ax=axs[i], color=swarmplot_colour, size=2.3)

        # Set the title for the subplot
        if titles_list:
            axs[i].set_title(titles_list[i])
        else:
            axs[i].set_title(col)

    # Remove any unused subplots
    for i in range(len(numerical_columns), len(axs)):
        axs[i].axis('off')

    # Layout so plots do not overlap
    fig.tight_layout()
    plt.show()

    return axs
