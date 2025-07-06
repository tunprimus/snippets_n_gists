#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import pingouin as pg
import plotly.graph_objects as go
import statsmodels.api as sm
from scipy.stats import linregress
from numpy.random import random

SD_LIMIT = 1.96
GOLDEN_RATIO = 1.618033989
FIG_WIDTH = 12
FIG_HEIGHT = FIG_WIDTH / GOLDEN_RATIO
FIG_DPI = 72
Y_LIM_OFFSET = 3.5
RANDOM_SEED = 42
LINSPACE_SAMPLE_VAL = 10

np.random.seed(RANDOM_SEED)


def bland_altman_plot_self(data01, data02, *args, **kwargs):
    """
    Bland-Altman Plot for comparing two sets of data.

    Parameters
    ----------
    data01 : numpy array
        The first set of data to compare.
    data02 : numpy array
        The second set of data to compare.
    *args : tuple
        Arguments to pass to matplotlib.pyplot.scatter.
    **kwargs : dict
        Keyword arguments to pass to matplotlib.pyplot.scatter.

    Returns
    -------
    None

    Notes
    -----
    This function generates a Bland-Altman Plot, which is a statistical graphic
    that is used to compare two sets of data. The x-axis represents the mean of
    the two sets of data, and the y-axis represents the difference between the
    two sets of data.
    """
    for data in (data01, data02):
        if not isinstance(data, np.ndarray) or data is None or data.size == 0:
            raise ValueError("Data must be non-empty numpy arrays")
    if data01.shape != data02.shape:
        raise ValueError("Data must have the same shape")
    if np.isnan(data01).any() or np.isnan(data02).any():
        raise ValueError("Data cannot contain NaN values")
    data01 = np.asarray(data01)
    data02 = np.asarray(data02)
    if np.isnan(data01).any() or np.isnan(data02).any():
        x_bar = np.nanmean([data01, data02], axis=0)
    else:
        x_bar = np.mean([data01, data02], axis=0)
    diff = data01 - data02
    mean_diff = np.nanmean(diff) if np.isnan(diff).any() else np.mean(diff)
    sd_diff = np.std(diff, axis=0)
    conf_interv_low = mean_diff - (SD_LIMIT * sd_diff)
    conf_interv_high = mean_diff + (SD_LIMIT * sd_diff)
    x_out_plot = np.min(x_bar) + ((np.max(x_bar) - np.min(x_bar)) * 1.14)
    plt.scatter(x_bar, diff, *args, **kwargs)
    plt.axhline(mean_diff, color="black", linestyle="-", linewidth=2)
    plt.axhline(
        conf_interv_high,
        color="grey",
        linestyle="--",
        linewidth=1,
        c="orange",
        label=r"+ 1.96$\sigma$",
    )
    plt.axhline(
        conf_interv_low,
        color="grey",
        linestyle="--",
        linewidth=1,
        c="magenta",
        label=r"- 1.96$\sigma$",
    )
    plt.title(r"$\mathbf{Bland-Altman}$" + " " + r"$\mathbf{Plot}$")
    plt.xlabel("Mean")
    plt.ylabel("Difference")
    plt.ylim(mean_diff - (Y_LIM_OFFSET * sd_diff), mean_diff + (Y_LIM_OFFSET * sd_diff))
    plt.text(
        x_out_plot,
        conf_interv_low,
        r"-1.96SD:" + "\n" + "%.2f" % conf_interv_low,
        ha="center",
        va="center",
    )
    plt.text(
        x_out_plot,
        conf_interv_high,
        r"+1.96SD:" + "\n" + "%.2f" % conf_interv_high,
        ha="center",
        va="center",
    )
    plt.text(
        x_out_plot,
        mean_diff,
        r"Mean:" + "\n" + "%.2f" % mean_diff,
        ha="center",
        va="center",
    )
    plt.subplots_adjust(right=0.85)
    plt.show()


def bland_altman_plot_plotly(
    data01,
    data02,
    data01_name="A",
    data02_name="B",
    subgroups=None,
    plotly_template=None,
    annotation_offset=0.05,
    plot_trendline=True,
    n_sd=SD_LIMIT,
    *args,
    **kwargs,
):
    """
    Bland-Altman Plot for comparing two sets of data with optional trendline and standard deviation intervals.

    Parameters
    ----------
    data01 : numpy array
        The first set of data to compare.
    data02 : numpy array
        The second set of data to compare.
    data01_name : str
        Name of the first dataset.
    data02_name : str
        Name of the second dataset.
    subgroups : list or numpy array
        Optional list or numpy array of subgroups. If given, the points will be coloured according to the subgroups.
    plotly_template : str
        Optional plotly template to use.
    annotation_offset : float
        Offset for the annotations of the mean and standard deviation intervals.
    plot_trendline : bool
        Whether to plot a trendline.
    n_sd : float
        Number of standard deviations to show in the intervals.
    *args : tuple
        Arguments to pass to plotly.graph_objects.Scatter.
    **kwargs : dict
        Keyword arguments to pass to plotly.graph_objects.Scatter.

    Returns
    -------
    plotly.graph_objects.Figure
        The generated Bland-Altman Plot.
    """
    for data in (data01, data02):
        if not isinstance(data, np.ndarray) or data is None or data.size == 0:
            raise ValueError("Data must be non-empty numpy arrays")
    if data01.shape != data02.shape:
        raise ValueError("Data must have the same shape")
    if np.isnan(data01).any() or np.isnan(data02).any():
        raise ValueError("Data cannot contain NaN values")
    data01 = np.asarray(data01)
    data02 = np.asarray(data02)
    if np.isnan(data01).any() or np.isnan(data02).any():
        x_bar = np.nanmean([data01, data02], axis=0)
    else:
        x_bar = np.mean([data01, data02], axis=0)
    diff = data01 - data02
    mean_diff = np.nanmean(diff) if np.isnan(diff).any() else np.mean(diff)
    sd_diff = np.std(diff, axis=0)
    fig = go.Figure()
    if plot_trendline:
        slope, intercept, r_value, p_value, std_err = linregress(x_bar, diff)
        trendline_x = np.linspace(np.min(x_bar), np.max(x_bar), LINSPACE_SAMPLE_VAL)
        # fig.add_trace(go.Scatter(x=trendline_x, y=slope * trendline_x + intercept, mode="lines", name="Trendline"), line=dict(width=4, dash="dot"))
        fig.add_trace(
            go.Scatter(
                x=trendline_x,
                y=slope * trendline_x + intercept,
                mode="lines",
                name="Trendline",
            )
        )
    if subgroups is None:
        fig.add_trace(
            go.Scatter(x=x_bar, y=diff, mode="markers", name="Data", *args, **kwargs)
        )
    else:
        for group_name in np.unique(subgroups):
            group_mask = np.where(np.array(subgroups) == group_name)
            fig.add_trace(
                go.Scatter(
                    x=x_bar[group_mask],
                    y=diff[group_mask],
                    mode="markers",
                    name=str(group_name),
                    *args,
                    **kwargs,
                )
            )
    # horizontal line
    fig.add_shape(
        type="line",
        xref="paper",
        x0=0,
        y0=mean_diff,
        x1=1,
        y1=mean_diff,
        line=dict(color="black", width=6, dash="dashdot"),
        name=f"Mean {mean_diff:.2f}",
    )
    # borderless rectangle
    fig.add_shape(
        type="rect",
        xref="paper",
        x0=0,
        y0=mean_diff - n_sd * sd_diff,
        x1=1,
        y1=mean_diff + n_sd * sd_diff,
        line=dict(color="SeaGreen", width=2),
        fillcolor="LightSkyBlue",
        opacity=0.4,
        name=f"±{n_sd} Standard Deviations",
    )
    # Edit the layout
    fig.update_layout(
        title=f"Bland-Altman Plot for {data01_name} and {data02_name}",
        xaxis_title=f"Average of {data01_name} and {data02_name}",
        yaxis_title=f"{data01_name} Minus {data02_name}",
        template=plotly_template,
        annotations=[
            dict(
                x=1,
                y=mean_diff,
                xref="paper",
                yref="y",
                text=f"Mean {mean_diff:.2f}",
                showarrow=True,
                arrowhead=7,
                ax=50,
                ay=0,
            ),
            dict(
                x=1,
                y=n_sd * sd_diff + mean_diff + annotation_offset,
                xref="paper",
                yref="y",
                text=f"+{n_sd} SD",
                showarrow=False,
                arrowhead=0,
                ax=0,
                ay=-20,
            ),
            dict(
                x=1,
                y=mean_diff - n_sd * sd_diff + annotation_offset,
                xref="paper",
                yref="y",
                text=f"-{n_sd} SD",
                showarrow=False,
                arrowhead=0,
                ax=0,
                ay=20,
            ),
            dict(
                x=1,
                y=mean_diff + n_sd * sd_diff - annotation_offset,
                xref="paper",
                yref="y",
                text=f"Mean {mean_diff:.2f}",
                showarrow=False,
                arrowhead=0,
                ax=0,
                ay=20,
            ),
            dict(
                x=1,
                y=mean_diff - n_sd * sd_diff - annotation_offset,
                xref="paper",
                yref="y",
                text=f"Mean {mean_diff:.2f}",
                showarrow=False,
                arrowhead=0,
                ax=0,
                ay=20,
            ),
        ],
    )
    return fig


def bland_altman_plot_josesho(
    data01,
    data02,
    sd_limit=SD_LIMIT,
    ax=None,
    scatter_kwds=None,
    mean_line_kwds=None,
    limit_lines_kwds=None,
):
    for data in (data01, data02):
        if not isinstance(data, np.ndarray) or data is None or data.size == 0:
            raise ValueError("Data must be non-empty numpy arrays")
    if data01.shape != data02.shape:
        raise ValueError("Data must have the same shape")
    if np.isnan(data01).any() or np.isnan(data02).any():
        raise ValueError("Data cannot contain NaN values")
    if sd_limit < 0:
        raise ValueError(f"sd_limit {sd_limit} is less than 0")
    data01 = np.asarray(data01)
    data02 = np.asarray(data02)
    if np.isnan(data01).any() or np.isnan(data02).any():
        x_bar = np.nanmean([data01, data02], axis=0)
    else:
        x_bar = np.mean([data01, data02], axis=0)
    diff = data01 - data02
    mean_diff = np.nanmean(diff) if np.isnan(diff).any() else np.mean(diff)
    sd_diff = np.std(diff, axis=0)
    if ax is None:
        ax = plt.gca()
    scatter_kwds = scatter_kwds or {}
    if "s" not in scatter_kwds:
        scatter_kwds["s"] = 20
    mean_line_kwds = mean_line_kwds or {}
    limit_lines_kwds = limit_lines_kwds or {}
    for kwds in [mean_line_kwds, limit_lines_kwds]:
        if "color" not in kwds:
            kwds["color"] = "grey"
        if "linewidth" not in kwds:
            kwds["linewidth"] = 1
        if "linestyle" not in mean_line_kwds:
            kwds["linestyle"] = "--"
        if "linestyle" not in limit_lines_kwds:
            kwds["linestyle"] = ":"
    ax.scatter(x_bar, diff, **scatter_kwds)
    ax.axhline(mean_diff, **mean_line_kwds)
    # Annotate mean line with mean difference
    ax.annotate(
        f"{mean_diff:.2f}",
        xy=(0.99, 0.5),
        horizontalalignment="right",
        verticalalignment="center",
        fontsize=14,
        xycoords="axes fraction",
    )
    if sd_limit > 0:
        half_ylim = (1.5 * sd_limit) * sd_diff
        ax.set_ylim(mean_diff - half_ylim, mean_diff + half_ylim)
        limit_of_agreement = sd_limit * sd_diff
        lower_limit = mean_diff - limit_of_agreement
        upper_limit = mean_diff + limit_of_agreement
        for j, lim in enumerate([lower_limit, upper_limit]):
            ax.axhline(lim, **limit_lines_kwds)
        ax.annotate(
            f"-SD{sd_limit}: {lower_limit:.2f}",
            xy=(0.99, 0.07),
            horizontalalignment="right",
            verticalalignment="bottom",
            fontsize=14,
            xycoords="axes fraction",
        )
        ax.annotate(
            f"+SD{sd_limit}: {upper_limit:.2f}",
            xy=(0.99, 0.92),
            horizontalalignment="right",
            fontsize=14,
            xycoords="axes fraction",
        )
    elif sd_limit == 0:
        half_ylim = 3 * sd_diff
        ax.set_ylim(mean_diff - half_ylim, mean_diff + half_ylim)
    plt.title("Bland-Altman Plot", fontsize=18)
    ax.set_ylabel("Difference", fontsize=15)
    ax.set_xlabel("Means", fontsize=15)
    ax.tick_params(labelsize=13)
    plt.tight_layout()
    return ax


def bland_altman_plot_statsmodels(data01, data02, *args, **kwargs):
    """
    Generates a Bland-Altman Plot using the statsmodels library.

    Parameters
    ----------
    data01 : numpy array
        The first set of data to compare.
    data02 : numpy array
        The second set of data to compare.
    *args : tuple
        Arguments to pass to matplotlib.pyplot.scatter.
    **kwargs : dict
        Keyword arguments to pass to matplotlib.pyplot.scatter.

    Returns
    -------
    None

    Notes
    -----
    This function generates a Bland-Altman Plot, which is a statistical graphic
    that is used to compare two sets of data. The x-axis represents the mean of
    the two sets of data, and the y-axis represents the difference between the
    two sets of data.
    """
    for data in (data01, data02):
        if not isinstance(data, np.ndarray) or data is None or data.size == 0:
            raise ValueError("Data must be non-empty numpy arrays")
    if data01.shape != data02.shape:
        raise ValueError("Data must have the same shape")
    if np.isnan(data01).any() or np.isnan(data02).any():
        raise ValueError("Data cannot contain NaN values")
    plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=FIG_DPI)
    sm.graphics.mean_diff_plot(data01, data02)
    plt.title("Bland-Altman Plot")
    plt.xlabel("Mean")
    plt.ylabel("Difference")
    plt.show()


def bland_altman_plot_pingouin(data01, data02):
    """
    Generates a Bland-Altman Plot using the pingouin library.

    Parameters
    ----------
    data01 : numpy array
        The first set of data to compare.
    data02 : numpy array
        The second set of data to compare.

    Returns
    -------
    None

    Notes
    -----
    This function generates a Bland-Altman Plot, which is a statistical graphic
    that is used to compare two sets of data. The x-axis represents the mean of
    the two sets of data, and the y-axis represents the difference between the
    two sets of data.
    """
    for data in (data01, data02):
        if not isinstance(data, np.ndarray) or data is None or data.size == 0:
            raise ValueError("Data must be non-empty numpy arrays")
    if data01.shape != data02.shape:
        raise ValueError("Data must have the same shape")
    if np.isnan(data01).any() or np.isnan(data02).any():
        raise ValueError("Data cannot contain NaN values")
    # plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=FIG_DPI)
    ax = pg.plot_blandaltman(data01, data02)
    if ax is None:
        raise RuntimeError("Plot failed to generate")
    plt.tight_layout()
    plt.title("Bland-Altman Plot")
    plt.xlabel("Mean")
    plt.ylabel("Difference")
    plt.show()


bland_altman_plot_self(random(10), random(10))
bland_altman_plot_josesho(random(10), random(10))
bland_altman_plot_plotly(random(10), random(10))
bland_altman_plot_statsmodels(random(10), random(10))
bland_altman_plot_pingouin(random(10), random(10))
