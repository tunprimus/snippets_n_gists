#!/usr/bin/env python3
# Adapted from VanWestendorp_PriceSensitivityMeter.py -> https://github.com/vivianamarquez/Van-Westendorp-Price-Sensitivity-Meter/blob/master/VanWestendorp_PriceSensitivityMeter.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import seaborn as sns
try:
    import fireducks.pandas as pd
except ImportError:
    import pandas as pd
    pd.set_option("mode.copy_on_write", True)
from functools import reduce
from matplotlib import rcParams
from os.path import realpath, expanduser
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


# Monkey patching NumPy >= 1.24 in order to successfully import model from sklearn and other libraries
np.float = np.float64
np.int = np.int_
np.object = np.object_
np.bool = np.bool_

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


# Remove intransitive price preferences
def validate(df, price_cols, messages=True):
    """
    Remove intransitive price preferences from a dataframe.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame containing price columns.
    price_cols : list of str
        List of column names for price columns.
    messages : bool, optional
        Whether to print out the number of cases kept/dropped (default is True).

    Returns
    -------
    pandas DataFrame
        DataFrame with intransitive price preferences removed.
    """
    try:
        import fireducks.pandas as pd
    except ImportError:
        import pandas as pd
        pd.set_option("mode.copy_on_write", True)

    orig_size = df.shape[0]
    validations = []
    for val01, val02 in zip(price_cols, price_cols[1:]):
        validations.append(df[val01] < df[val02])
    df = df[validations[0] & validations[1] & validations[2]]
    new_size = df.shape[0]
    # return df[reduce(lambda x, y: x & y, validations)], orig_size
    if messages:
        print(
            f"Total dataset contains {orig_size} cases, {new_size} cases kept as transitive price preferences."
        )
    return df


# Calculate cumulative frequencies
def cum_density_freq(df, col):
    """
    Calculate the cumulative density frequency for a specific column in a DataFrame.

    This function computes the cumulative distribution function (CDF) for the values
    in the specified column of the DataFrame. It returns a DataFrame with the unique
    values of the column and their corresponding CDF values.

    Parameters
    ----------
    df : pandas DataFrame
        The input DataFrame containing the data.
    col : str
        The column name within the DataFrame for which the cumulative density
        frequency is to be calculated.

    Returns
    -------
    pandas DataFrame
        A DataFrame with columns 'price' and the CDF values for the specified column.
        The DataFrame contains the unique values of the specified column and their
        corresponding cumulative density frequency values.
    """
    try:
        import fireducks.pandas as pd
    except ImportError:
        import pandas as pd
        pd.set_option("mode.copy_on_write", True)

    # Frequency
    stats_df = (
        df.groupby(col)[col]
        .agg("count")
        .pipe(pd.DataFrame)
        .rename(columns={col: f"{col}_freq"})
    )
    # PDF
    stats_df[f"{col}_pdf"] = stats_df[f"{col}_freq"] / stats_df[f"{col}_freq"].sum(
        min_count=1
    )
    # CDF
    stats_df[f"{col}_cdf"] = stats_df[f"{col}_pdf"].cumsum()
    stats_df.reset_index(inplace=True)
    stats_df.drop([f"{col}_freq", f"{col}_pdf"], axis=1, inplace=True)
    stats_df.rename(columns={col: "price", f"{col}_cdf": col}, inplace=True)

    return stats_df


##  Re-creating R's function output$data_vanwestendorp
def cum_density_freq_table(df, price_cols, interpolate=True, messages=True):
    """
    Re-create R's function output$data_vanwestendorp

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame containing price columns.
    price_cols : list of str
        List of column names for price columns.
    interpolate : bool, optional
        Whether to create a continuous function by interpolation (default is True).
    messages : bool, optional
        Whether to print out the table (default is True).

    Returns
    -------
    pandas DataFrame
        DataFrame with cumulative density frequency values for the price columns.
    """
    try:
        import fireducks.pandas as pd
    except ImportError:
        import pandas as pd
        pd.set_option("mode.copy_on_write", True)
    from functools import reduce

    df.rename(
        columns={
            price_cols[0]: "too_cheap",
            price_cols[1]: "cheap",
            price_cols[2]: "expensive",
            price_cols[3]: "too_expensive",
        },
        inplace=True,
    )
    cdfs = [
        cum_density_freq(df, "too_cheap"),
        cum_density_freq(df, "cheap"),
        cum_density_freq(df, "expensive"),
        cum_density_freq(df, "too_expensive"),
    ]
    cdfs = reduce(
        lambda x, y: pd.merge(x, y, on="price", how="outer"), cdfs
    ).sort_values("price")
    cdfs = cdfs.fillna(method="ffill").fillna(0)
    cdfs["too_cheap"] = 1 - cdfs["too_cheap"]
    cdfs["cheap"] = 1 - cdfs["cheap"]
    cdfs["not_cheap"] = 1 - cdfs["cheap"]
    cdfs["not_expensive"] = 1 - cdfs["expensive"]
    cdfs = cdfs.clip(lower=0)

    if interpolate:
        low = cdfs["price"].min()
        high = cdfs["price"].max()
        cdfs = pd.merge(
            pd.DataFrame(list(np.arange(low, high, 0.01)), columns=["price"]),
            cdfs,
            how="outer",
        ).sort_values("price")
        cdfs["price"] = cdfs["price"].round(2)
        cdfs.drop_duplicates_subset_name("price", keep="last", inplace=True)
        cdfs = cdfs.interpolate(method="linear", limit_direction="forward")
        cdfs["too_cheap"] = cdfs["too_cheap"].fillna(1)
        cdfs["cheap"] = cdfs["cheap"].fillna(0)
        cdfs["expensive"] = cdfs["expensive"].fillna(0)
        cdfs["too_expensive"] = cdfs["too_expensive"].fillna(0)
        cdfs["not_cheap"] = cdfs["not_cheap"].fillna(0)
        cdfs["not_expensive"] = cdfs["not_expensive"].fillna(1)
        cdfs.reset_index(inplace=True)
        cdfs.drop("index", axis=1, inplace=True)

    if messages:
        print(cdfs)

    return cdfs


def plot_price_sensitivity_via_plt_n_sns(
    cdfs,
    point_of_marginal_cheapness,
    pmc_height,
    point_of_marginal_expensiveness,
    pme_height,
    indifference_price_point,
    ipp_height,
    optimal_price_point,
    opp_height,
    plot_title="",
    currency_symbol="$",
    figsize=FIG_SIZE,
    dpi=FIG_DPI,
):
    """
    Plot the results of the Van Westendorp Price Sensitivity Meter using Matplotlib
    and Seaborn.

    Parameters
    ----------
    cdfs : pandas DataFrame
        The cumulative density frequency table.
    point_of_marginal_cheapness : float
        The point of marginal cheapness.
    pmc_height : float
        The height of the point of marginal cheapness on the y-axis.
    point_of_marginal_expensiveness : float
        The point of marginal expensiveness.
    pme_height : float
        The height of the point of marginal expensiveness on the y-axis.
    indifference_price_point : float
        The indifference price point.
    ipp_height : float
        The height of the indifference price point on the y-axis.
    optimal_price_point : float
        The optimal price point.
    opp_height : float
        The height of the optimal price point on the y-axis.
    plot_title : str, optional
        The title of the plot (default is '').
    currency_symbol : str, optional
        The currency symbol to use in the plot (default is '$').
    figsize : tuple of ints, optional
        The figure size (default is (12, 6)).
    dpi : int, optional
        The dots per inch (default is 72).

    Returns
    -------
    None

    Notes
    -----
    The plot shows the cumulative density frequency curves for the four price
    categories. The points of marginal cheapness, marginal expensiveness,
    indifference and optimal price are marked with a scatter plot.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    sns.set()
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    sns.lineplot(
        x=cdfs["price"].values,
        y=cdfs["too_expensive"].values,
        label="too_expensive",
        color="red",
    )
    sns.lineplot(
        x=cdfs["price"].values,
        y=cdfs["not_expensive"].values,
        label="not_expensive",
        color="orange",
    )
    sns.lineplot(
        x=cdfs["price"].values,
        y=cdfs["not_cheap"].values,
        label="not_cheap",
        color="blue",
    )
    sns.lineplot(
        x=cdfs["price"].values,
        y=cdfs["too_cheap"].values,
        label="too_cheap",
        color="green",
    )

    ax.scatter(
        [point_of_marginal_cheapness],
        [pmc_height],
        label=f"Point of Marginal Cheapness: {currency_symbol}{point_of_marginal_cheapness:.2f}",
        color="blue",
        s=50,
    )
    ax.scatter(
        [point_of_marginal_expensiveness],
        [pme_height],
        label=f"Point of Marginal Expensiveness: {currency_symbol}{Point_of_Marginal_Expensiveness:.2f}",
        color="red",
        s=50,
    )
    ax.scatter(
        [indifference_price_point],
        [ipp_height],
        label=f"Indifference Price Point: {currency_symbol}{indifference_price_point:.2f}",
        color="orange",
        s=50,
    )
    ax.scatter(
        [optimal_price_point],
        [opp_height],
        label=f"Optimal Price Point: {currency_symbol}{optimal_price_point:.2f}",
        color="green",
        s=50,
    )

    ax.set_title(f"Van Westendorp's Price Sensitivity Meter\n{plot_title}")
    ax.set_xlabel(f"Price ({currency_symbol})")
    ax.set_ylabel("% of Participants")
    ax.set_xlim(cdfs["price"].min() - 5, cdfs["price"].max() + 5)
    ax.set_ylim(-0.1, 1.1)
    ax.legend()

    plt.show()

    return None


def plot_price_sensitivity_via_plotly(
    cdfs,
    point_of_marginal_cheapness,
    pmc_height,
    point_of_marginal_expensiveness,
    pme_height,
    indifference_price_point,
    ipp_height,
    optimal_price_point,
    opp_height,
    plot_title="",
    currency_symbol="$",
    figsize=FIG_SIZE,
    dpi=FIG_DPI,
):
    """
    Plot a Van Westendorp's Price Sensitivity Meter via Plotly.

    Parameters
    ----------
    cdfs : pandas DataFrame
        The cumulative density frequency table.
    point_of_marginal_cheapness : float
        The point of marginal cheapness.
    pmc_height : float
        The height of the point of marginal cheapness on the y-axis.
    point_of_marginal_expensiveness : float
        The point of marginal expensiveness.
    pme_height : float
        The height of the point of marginal expensiveness on the y-axis.
    indifference_price_point : float
        The indifference price point.
    ipp_height : float
        The height of the indifference price point on the y-axis.
    optimal_price_point : float
        The optimal price point.
    opp_height : float
        The height of the optimal price point on the y-axis.
    plot_title : str, optional
        The title of the plot (default is '').
    currency_symbol : str, optional
        The currency symbol to use in the plot (default is '$').
    figsize : tuple of ints, optional
        The figure size (default is (12, 6)).
    dpi : int, optional
        The dots per inch (default is 72).

    Returns
    -------
    None

    Notes
    -----
    The plot shows the cumulative density frequency curves for the four price
    categories. The points of marginal cheapness, marginal expensiveness,
    indifference and optimal price are marked with a scatter plot.
    """
    import pandas as pd
    import plotly.graph_objs as go
    from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

    line_width = 1
    marker_size = 3

    var = "too_expensive"
    trace_01 = go.Scatter(
        x=cdfs["price"].values,
        y=cdfs[var].values,
        text=[
            f"{var}<br>Price: {currency_symbol}{price:.2f}<br>Participants: {val*100:.2f}%"
            for (price, val) in zip(cdfs["price"].values, cdfs[var].values)
        ],
        mode="lines",
        opacity=0.8,
        marker={"size": marker_size, "color": "red"},
        hoverinfo="text",
        line={"color": "red", "width": line_width},
        name=var,
    )

    var = "not_expensive"
    trace_02 = go.Scatter(
        x=cdfs["price"].values,
        y=cdfs[var].values,
        text=[
            f"{var}<br>Price: {currency_symbol}{price:.2f}<br>Participants: {val*100:.2f}%"
            for (price, val) in zip(cdfs["price"].values, cdfs[var].values)
        ],
        mode="lines",
        opacity=0.8,
        marker={"size": marker_size, "color": "orange"},
        hoverinfo="text",
        line={"color": "orange", "width": line_width},
        name=var,
    )

    var = "not_cheap"
    trace_03 = go.Scatter(
        x=cdfs["price"].values,
        y=cdfs[var].values,
        text=[
            f"{var}<br>Price: {currency_symbol}{price:.2f}<br>Participants: {val*100:.2f}%"
            for (price, val) in zip(cdfs["price"].values, cdfs[var].values)
        ],
        mode="lines",
        opacity=0.8,
        marker={"size": marker_size, "color": "blue"},
        hoverinfo="text",
        line={"color": "blue", "width": line_width},
        name=var,
    )

    var = "too_cheap"
    trace_04 = go.Scatter(
        x=cdfs["price"].values,
        y=cdfs[var].values,
        text=[
            f"{var}<br>Price: {currency_symbol}{price:.2f}<br>Participants: {val*100:.2f}%"
            for (price, val) in zip(cdfs["price"].values, cdfs[var].values)
        ],
        mode="lines",
        opacity=0.8,
        marker={"size": marker_size, "color": "green"},
        hoverinfo="text",
        line={"color": "green", "width": line_width},
        name=var,
    )

    point_01 = go.Scatter(
        x=[point_of_marginal_cheapness],
        y=[pmc_height],
        text=[
            f"Point of Marginal Cheapness: {currency_symbol}{point_of_marginal_cheapness:.2f}<br>Participants: {pmc_height*100:.2f}%"
        ],
        mode="markers",
        opacity=1,
        marker={"size": 7, "color": "blue"},
        hoverinfo="text",
        name=f"<br>Point of Marginal Cheapness<br>{currency_symbol}{point_of_marginal_cheapness:.2f}",
    )

    point_02 = go.Scatter(
        x=[point_of_marginal_expensiveness],
        y=[pme_height],
        text=[
            f"Point of Marginal Expensiveness: {currency_symbol}{point_of_marginal_expensiveness:.2f}<br>Participants: {pme_height*100:.2f}%"
        ],
        mode="markers",
        opacity=1,
        marker={"size": 7, "color": "red"},
        hoverinfo="text",
        name=f"Point of Marginal Expensiveness<br>${point_of_marginal_expensiveness:.2f}",
    )

    point_03 = go.Scatter(
        x=[indifference_price_point],
        y=[ipp_height],
        text=[
            f"Indifference Price Point: {currency_symbol}{indifference_price_point:.2f}<br>Participants: {ipp_height*100:.2f}%"
        ],
        mode="markers",
        opacity=1,
        marker={"size": 7, "color": "orange"},
        hoverinfo="text",
        name=f"Indifference Price Point<br>{currency_symbol}{indifference_price_point:.2f}",
    )

    point_04 = go.Scatter(
        x=[optimal_price_point],
        y=[opp_height],
        text=[
            f"Optimal Price Point: {currency_symbol}{optimal_price_point:.2f}<br>Participants: {opp_height*100:.2f}%"
        ],
        mode="markers",
        opacity=1,
        marker={"size": 7, "color": "green"},
        hoverinfo="text",
        name=f"Optimal Price Point<br>{currency_symbol}{optimal_price_point:.2f}",
    )

    data = [
        trace_01,
        trace_02,
        trace_03,
        trace_04,
        point_01,
        point_02,
        point_03,
        point_04,
    ]

    layout = go.Layout(
        title=f"Van Westendorp's Price Sensitivity Meter<br>{plot_title}",
        xaxis=dict(
            title=f"{currency_symbol} Price",
            range=(cdfs["price"].min() - 5, cdfs["price"].max() + 5),
        ),
        yaxis=dict(title="% of Participants", range=(-0.1, 1.1)),
        template="plotly_white",
    )

    fig = go.Figure(data=data, layout=layout)
    iplot(fig)

    return None


# Get results
def psm_results(
    df, price_cols, to_plot=True, plot_title="", currency_symbol="$", figsize=FIG_SIZE, dpi=FIG_DPI, messages=True
):
    """
    Get results from the Van Westendorp Price Sensitivity Meter.

    Parameters
    ----------
    df : pandas DataFrame
        The DataFrame with the survey results.
    price_cols : list of str
        The column names of the price columns.
    to_plot : bool, optional
        Whether to plot the results (default is True).
    plot_title : str, optional
        The title of the plot (default is '').
    currency_symbol : str, optional
        The currency symbol to use in the plot (default is '$').
    figsize : tuple of ints, optional
        The figure size (default is (12, 6)).
    dpi : int, optional
        The dots per inch (default is 72).
    messages : bool, optional
        Whether to print the results (default is True).

    Returns
    -------
    result_df : pandas DataFrame
        A DataFrame with the results.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import plotly.graph_objs as go
    import seaborn as sns
    from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

    df = validate(df, price_cols, messages=messages)
    cdfs = cum_density_freq_table(df, price_cols)

    # Get point parameters
    point_of_marginal_cheapness = cdfs.iloc[
        np.argwhere(np.diff(np.sign(cdfs["too_cheap"] - cdfs["not_cheap"]))).flatten()
        + 1
    ]["price"].values[0]
    point_of_marginal_expensiveness = cdfs.iloc[
        np.argwhere(
            np.diff(np.sign(cdfs["too_expensive"] - cdfs["not_expensive"]))
        ).flatten() + 1
    ]["price"].values[0]
    indifference_price_point = cdfs.iloc[
        np.argwhere(
            np.diff(np.sign(cdfs["not_cheap"] - cdfs["not_expensive"]))
        ).flatten() + 1
    ]["price"].values[0]
    optimal_price_point = cdfs.iloc[
        np.argwhere(
            np.diff(np.sign(cdfs["too_expensive"] - cdfs["too_cheap"]))
        ).flatten() + 1
    ]["price"].values[0]

    # Get values for the plot
    pmc_height = (
        cdfs.iloc[
            np.argwhere(
                np.diff(np.sign(cdfs["too_cheap"] - cdfs["not_cheap"]))
            ).flatten() + 1
        ]["too_cheap", "not_cheap"]
        .mean(axis=1)
        .values[0]
    )
    pme_height = (
        cdfs.iloc[
            np.argwhere(
                np.diff(np.sign(cdfs["too_expensive"] - cdfs["not_expensive"]))
            ).flatten() + 1
        ]["too_expensive", "not_expensive"]
        .mean(axis=1)
        .values[0]
    )
    ipp_height = (
        cdfs.iloc[
            np.argwhere(
                np.diff(np.sign(cdfs["not_cheap"] - cdfs["not_expensive"]))
            ).flatten() + 1
        ]["not_cheap", "not_expensive"]
        .mean(axis=1)
        .values[0]
    )
    opp_height = (
        cdfs.iloc[
            np.argwhere(
                np.diff(np.sign(cdfs["too_expensive"] - cdfs["too_cheap"]))
            ).flatten() + 1
        ]["too_expensive", "too_cheap"]
        .mean(axis=1)
        .values[0]
    )

    if messages:
        print(
            f"Accepted Price Range: {currency_symbol}{point_of_marginal_cheapness:.2f} to {currency_symbol}{point_of_marginal_expensiveness:.2f}"
        )
        print(
            f"Indifference Price Point: {currency_symbol}{indifference_price_point:.2f}"
        )
        print(f"Optimal Price Point: {currency_symbol}{optimal_price_point:.2f}")

    if to_plot:
        plot_price_sensitivity_via_plt_n_sns(
            cdfs,
            point_of_marginal_cheapness,
            pmc_height,
            point_of_marginal_expensiveness,
            pme_height,
            indifference_price_point,
            ipp_height,
            optimal_price_point,
            opp_height,
            plot_title=plot_title,
            figsize=FIG_SIZE,
            dpi=FIG_DPI,
            currency_symbol=currency_symbol,
        )

        plot_price_sensitivity_via_plotly(
            cdfs,
            point_of_marginal_cheapness,
            pmc_height,
            point_of_marginal_expensiveness,
            pme_height,
            indifference_price_point,
            ipp_height,
            optimal_price_point,
            opp_height,
            plot_title=plot_title,
            figsize=FIG_SIZE,
            dpi=FIG_DPI,
            currency_symbol=currency_symbol,
        )

    result_df = pd.DataFrame(
        {
            "point_of_marginal_cheapness": point_of_marginal_cheapness,
            "point_of_marginal_expensiveness": point_of_marginal_expensiveness,
            "indifference_price_point": indifference_price_point,
            "optimal_price_point": optimal_price_point,
            "pmc_height": pmc_height,
            "pme_height": pme_height,
            "ipp_height": ipp_height,
            "opp_height": opp_height,
        },
        index=[0],
    )

    return result_df
