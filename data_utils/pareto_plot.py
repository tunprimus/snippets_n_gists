#!/usr/bin/env python3
# Adapted from: Pareto Plot With Matplotlib and Others -> https://tylermarrs.com/posts/pareto-plot-with-matplotlib/
# https://www.statology.org/pareto-chart-python/, https://github.com/HasanYahya101/Pareto-Tutorial-Python, https://stackoverflow.com/a/73954823
import pandas as pd
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
rcParams["figure.figsize"] = FIG_SIZE
rcParams["figure.dpi"] = FIG_DPI
rcParams["savefig.format"] = "svg"


def pareto_plot(
    data,
    cat_var=None,
    num_var=None,
    threshold=0.80,
    title=None,
    x_label=None,
    y_label=None,
    pct_format="{0:.0%}",
):
    """
    Create a Pareto Plot for a given set of data.

    Parameters
    ----------
    data: pandas DataFrame, pandas Series, or numpy array
        The data to be plotted.
    cat_var: str, optional
        The categorical variable to be plotted on the x-axis.
    num_var: str, optional
        The numerical variable to be plotted on the y-axis.
    threshold: float, optional
        The cumulative percentage threshold for the Pareto Plot.
        The default is 0.80.
    title: str, optional
        The title of the Pareto Plot.
    x_label: str, optional
        The label for the x-axis.
    y_label: str, optional
        The label for the y-axis.
    pct_format: str, optional
        The format string for the cumulative percentage values.
        The default is "{0:.0%}".

    Returns
    -------
    None

    Notes
    -----
    The Pareto Plot is a bar chart with a secondary y-axis that shows the cumulative percentage of the values.
    The x-axis shows the categorical variable, and the y-axis shows the numerical variable.
    The secondary y-axis shows the cumulative percentage of the values, and the threshold line is drawn at the specified threshold value.
    The function also annotates the cumulative percentage values on the plot.

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame({"category": ["A", "B", "C", "D", "E"], "value": [10, 20, 30, 40, 50]})
    >>> pareto_plot(data, cat_var="category", num_var="value", threshold=0.80)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.ticker import PercentFormatter

    if not isinstance(data, (pd.DataFrame, pd.Series, np.ndarray)):
        raise TypeError(
            "Data must be a pandas DataFrame, pandas Series, or numpy array"
        )
    if title is None:
        title = f"Pareto Plot for {cat_var} vs {num_var}"
    if x_label is None:
        x_label = cat_var
    if y_label is None:
        y_label = num_var

    tmp = data.sort_values(num_var, ascending=False)
    x = tmp[cat_var].values
    y = tmp[num_var].values
    weights = y / y.sum()
    cum_sum = weights.cumsum()
    cum_sum_pct = cum_sum * 100

    fig, ax1 = plt.subplots()
    ax1.bar(x, y, color="skyblue", alpha=0.80)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)

    ax2 = ax1.twinx()
    ax2.plot(x, cum_sum_pct, "-go", alpha=0.5)
    ax2.yaxis.set_major_formatter(PercentFormatter())
    ax2.set_ylabel("", color="r")
    ax2.tick_params(axis="y", colors="r")

    formatted_weights = [pct_format.format(v) for v in cum_sum]
    for i, txt in enumerate(formatted_weights):
        ax2.annotate(txt, (x[i], cum_sum_pct[i]), fontweight="demi")

    # Rotate x-axis labels for better readability
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    if threshold:
        plt.axhline(y=threshold * 100, color="darkorange", linestyle="--")

    if title:
        plt.title(title)

    plt.tight_layout()
    plt.show()


def pareto_plot_alt(df, cat_var, num_var):
    """
    Creates a Pareto plot of the given categorical variable (cat_var) against the given numerical variable (num_var) in the given DataFrame.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame containing the categorical and numerical variables to be used in the Pareto plot.
    cat_var : str
        Name of the categorical variable column in the DataFrame.
    num_var : str
        Name of the numerical variable column in the DataFrame.

    Returns
    -------
    None

    Notes
    -----
    The Pareto plot is displayed in a new window.

    Examples
    --------
    >>> df = pd.DataFrame({"Category": ["A", "B", "C", "D"], "Value": [10, 20, 30, 40]})
    >>> pareto_plot_alt(df, "Category", "Value")
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.ticker import PercentFormatter

    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    if cat_var not in df.columns or num_var not in df.columns:
        raise ValueError(
            "cat_var and num_var must be valid column names in the DataFrame"
        )

    df_temp = df.sort_values(num_var, ascending=False).copy(deep=True)
    df_temp["cum_sum_pct"] = df_temp[num_var].cumsum() / df_temp[num_var].sum() * 100

    fig, ax1 = plt.subplots()
    bars = ax1.bar(df_temp[cat_var], df_temp[num_var], color="teal")
    ax2 = ax1.twinx()
    (line,) = ax2.plot(
        df_temp[cat_var],
        df_temp["cum_sum_pct"],
        marker="o",
        linestyle="-",
        color="darkorange",
    )
    ax2.yaxis.set_major_formatter(PercentFormatter())

    for bar, val, cum_pct in zip(bars, df_temp[num_var], df_temp["cum_sum_pct"]):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val}",
            ha="center",
            va="bottom",
        )
        idx = df_temp[df_temp[cat_var] == bar.get_x()].index
        if not idx.empty:
            ax2.text(
                line.get_xdata()[idx[0]],
                cum_pct,
                f"{cum_pct:.1f}%",
                ha="center",
                va="bottom",
            )

    # Rotate x-axis labels for better readability
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    plt.title(f"Pareto Chart of {cat_var} vs {num_var}")
    plt.show()
