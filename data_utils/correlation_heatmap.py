#!/usr/bin/env python3
# Adapted from: klib -> https://github.com/akanz1/klib/blob/main/src/klib/describe.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from screeninfo import get_monitors
from screeninfo import ScreenInfoError

def _corr_selector(corr, split=None, threshold=0.0, messages=False):
    """
    Select correlations based on a split option.

    Parameters
    ----------
    corr : pd.DataFrame
        The correlation matrix.
    split : str, optional
        The split option. Options are 'pos', 'neg', 'high', 'low'. The default is None.
    threshold : float, optional
        The threshold for the split. The default is 0.0.
    messages : bool, optional
        If True, prints messages to the console. The default is False.

    Returns
    -------
    pd.DataFrame
        The selected correlation matrix.
    """
    if split == "pos":
        corr = corr.where((corr >= threshold) & (corr > 0))
        if messages:
            print("Displaying positive correlations. Specify a positive `threshold` to `limit the results further`.")
    elif split == "neg":
        corr = corr.where((corr <= threshold) & (corr < 0))
        if messages:
            print("Displaying negative correlations. Specify a negative `threshold` to `limit the results further`.")
    elif split == "high":
        threshold = 0.3 if threshold <= 0 else threshold
        corr = corr.where(np.abs(corr) >= threshold)
        if messages:
            print(f"Displaying absolute correlations above the threshold ({threshold}). Specify a positive `threshold` to limit the results further.")
    elif split == "low":
        threshold = 0.3 if threshold <= 0 else threshold
        corr = corr.where(np.abs(corr) <= threshold)
        if messages:
            print(f"Displaying absolute correlations below the threshold ({threshold}). Specify a negative `threshold` to limit the results further.")

    return corr


def corr_mat(df, split=None, threshold=0.0, target=None, method="pearson", coloured=True, messages=False):
    """
    Computes the correlation matrix of a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame for which the correlation matrix is computed.
    split : str, optional
        Specifies the type of correlation to display. Options are 'pos', 'neg', 'high', 'low'.
        Default is None.
    threshold : float, optional
        Threshold value used for filtering correlations based on the `split` option. Default is 0.0.
    target : str, list, pd.Series, or np.ndarray, optional
        If specified, computes the correlation of all columns with the target.
        Can be a column name, list, Series, or ndarray. Default is None.
    method : str, optional
        Method of correlation: 'pearson', 'kendall', 'spearman'. Default is 'pearson'.
    coloured : bool, optional
        If True, applies color formatting to negative correlations. Default is True.
    messages : bool, optional
        If True, prints messages to the console. Default is False.

    Returns
    -------
    pd.DataFrame or pd.Styler
        A correlation matrix as a DataFrame. If `coloured` is True, returns a styled DataFrame.
    """
    def colour_negative_red(val):
        colour = "#FF3344" if val < 0 else None
        return f"color: {colour}"

    data = pd.DataFrame(df)

    if isinstance(target, (str, list, pd.Series, np.ndarray)):
        target_data = []
        if isinstance(target, str):
            target_data = data[target]
            data = data.drop(target, axis=1)

        elif isinstance(target, (list, pd.Series, np.ndarray)):
            target_data = pd.Series(target)
            target = target_data.name

        corr = pd.DataFrame(data.corrwith(target_data, method=method, numeric_only=True),)
        corr = corr.sort_values(corr.columns[0], ascending=False)
        corr.columns = [target]

    else:
        corr = data.corr(method=method, numeric_only=True)

    corr = _corr_selector(corr, split=split, threshold=threshold)

    if coloured:
        return corr.style.applymap(colour_negative_red).format("{:.2f}", na_rep="-")

    # Return pd.DataFrame | pd.Styler
    return corr


def corr_plot(df, split=None, threshold=0.0, target=None, method="pearson", cmap="BrBG", figsize=(20, 12.4), annot=True, dev=False, messages=False, **kwargs):
    """
    Computes the correlation matrix of a DataFrame and visualises it as a heatmap.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame for which the correlation matrix is computed.
    split : str, optional
        Specifies the type of correlation to display. Options are 'pos', 'neg', 'high', 'low'.
        Default is None.
    threshold : float, optional
        Threshold value used for filtering correlations based on the `split` option. Default is 0.0.
    target : str, list, pd.Series, or np.ndarray, optional
        If specified, computes the correlation of all columns with the target.
        Can be a column name, list, Series, or ndarray. Default is None.
    method : str, optional
        Method of correlation: 'pearson', 'kendall', 'spearman'. Default is 'pearson'.
    cmap : str, optional
        Colour map to use for the heatmap. Default is 'BrBG'.
    figsize : tuple, optional
        Figure size in inches. Default is (20, 12.4).
    annot : bool, optional
        If True, annotates the heatmap with the correlation values. Default is True.
    dev : bool, optional
        If True, adds a title with the developer settings used for the heatmap. Default is False.
    messages : bool, optional
        If True, prints messages to the console. Default is False.
    **kwargs
        Additional keyword arguments passed to seaborn's heatmap.

    Returns
    -------
    matplotlib Axes object
        The Axes object with the heatmap.
    """
    data = pd.DataFrame(df)

    corr = corr_mat(data, split=split, threshold=threshold, target=target, method=method, coloured=False,)

    mask = np.zeros_like(corr, dtype=bool)

    if target is None:
        mask = np.triu(np.ones_like(corr, dtype=bool))

    vmax = np.round(np.nanmax(corr.where(~mask)) - 0.05, 2)
    vmin = np.round(np.nanmin(corr.where(~mask)) + 0.05, 2)

    fig, ax = plt.subplots(figsize=figsize)

    # Specify kwargs for the heatmap
    kwargs = {
        "mask": mask,
        "cmap": cmap,
        "annot": annot,
        "vmax": vmax,
        "vmin": vmin,
        "linewidths": 0.5,
        "annot_kws": {"size": 10},
        "cbar_kws": {"shrink": 0.95, "aspect": 30},
        **kwargs,
    }

    # Draw heatmap with mask and default settings
    sns.heatmap(corr, center=0, fmt=".2f", **kwargs)
    ax.set_title(f"Feature-correlation ({method})", fontdict={"fontsize": 18})

    # Settings
    if dev:
        fig.suptitle(
            f"\
            Settings (dev-mode): \n\
            - split-mode: {split} \n\
            - threshold: {threshold} \n\
            - method: {method} \n\
            - annotations: {annot} \n\
            - cbar: \n\
                - vmax: {vmax} \n\
                - vmin: {vmin} \n\
            - linewidths: {kwargs['linewidths']} \n\
            - annot_kws: {kwargs['annot_kws']} \n\
            - cbar_kws: {kwargs['cbar_kws']}",
            fontsize=12,
            color="grey",
            x=0.35,
            y=0.85,
            ha="left",
        )

    # Return matplotlib Axes object
    return ax


def corr_interactive_plot(df, split=None, threshold=0.0, target=None, method="pearson", cmap="BrBG", figsize=(20, 12.4), annot=True, messages=False, **kwargs):
    """
    Computes and visualises an interactive correlation matrix heatmap using Plotly.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame for which the correlation matrix is computed.
    split : str, optional
        Specifies the type of correlation to display. Options are 'pos', 'neg', 'high', 'low'.
        Default is None.
    threshold : float, optional
        Threshold value used for filtering correlations based on the `split` option. Default is 0.0.
    target : str, list, pd.Series, or np.ndarray, optional
        If specified, computes the correlation of all columns with the target.
        Can be a column name, list, Series, or ndarray. Default is None.
    method : str, optional
        Method of correlation: 'pearson', 'kendall', 'spearman'. Default is 'pearson'.
    cmap : str, optional
        Colour scale to use for the heatmap. Default is 'BrBG'.
    figsize : tuple, optional
        Figure size in inches. Default is (20, 12.4).
    annot : bool, optional
        If True, includes annotation text in the heatmap. Default is True.
    messages : bool, optional
        If True, prints messages to the console. Default is False.
    **kwargs
        Additional keyword arguments passed to Plotly's Heatmap.

    Returns
    -------
    plotly.graph_objs.Figure
        Interactive heatmap figure object created using Plotly.
    """
    data = pd.DataFrame(df).iloc[:, ::-1]

    corr = corr_mat(data, split=split, threshold=threshold, target=target, method=method, coloured=False,)

    mask = np.zeros_like(corr, dtype=bool)

    if target is None:
        mask = np.triu(np.ones_like(corr, dtype=bool))
        np.fill_diagonal(corr.to_numpy(), np.nan)
        corr = corr.where(mask == 1)
    else:
        corr = corr.iloc[::-1, :]

    vmax = np.round(np.nanmax(corr) - 0.05, 2)
    vmin = np.round(np.nanmin(corr) + 0.05, 2)

    vmax = -vmin if split == "neg" else vmax
    vmin = -vmax if split == "pos" else vmin

    vtext = corr.round(2).fillna("") if annot else None

    corr_columns = corr.columns
    corr_index = corr.index

    if isinstance(corr_columns, pd.MultiIndex):
        corr_columns = ["-".join(col) for col in corr.columns]

    if isinstance(corr_index, pd.MultiIndex):
        corr_index = ["-".join(idx) for idx in corr.index]

    # Specify kwargs for the heatmap
    kwargs = {
        "colorscale": cmap,
        "zmax": vmax,
        "zmin": vmin,
        "text": vtext,
        "texttemplate": "%{text}",
        "textfont": {"size": 13},
        "x": corr_columns,
        "y": corr_index,
        "z": corr,
        **kwargs,
    }

    # Draw heatmap with masked corr and default settings
    heatmap = go.Figure(data=go.Heatmap(hoverongaps=False, xgap=1, ygap=1, **kwargs,),)

    dpi = None
    try:
        for monitor in get_monitors():
            if monitor.is_primary:
                if monitor.width_mm is None or monitor.height_mm is None:
                    continue
                dpi = monitor.width / (monitor.width_mm / 25.4)
                break

        if dpi is None:
            monitor = get_monitors()[0]
            if monitor.width_mm is None or monitor.height_mm is None:
                dpi = 96  # more or less arbitrary default value
            else:
                dpi = monitor.width / (monitor.width_mm / 25.4)
    except ScreenInfoError:
        dpi = 96

    heatmap.update_layout(
        title=f"Feature-correlation ({method})",
        title_font={"size": 24},
        title_x=0.5,
        autosize=True,
        width=figsize[0] * dpi,
        height=(figsize[1] + 1) * dpi,
        xaxis={"autorange": "reversed"},
    )

    # Return Plotly plotly.graph_objs._figure.Figure interactive object
    return heatmap

