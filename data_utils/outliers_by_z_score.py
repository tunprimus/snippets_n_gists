#!/usr/bin/env python3
import pandas as pd
from scipy.stats import zscore

pd.set_option("mode.copy_on_write", True)

def outliers_by_z_score(df, target_col=None, threshold=3):
    """
    Find outliers in a DataFrame column based on the Z-score method.

    This function calculates the Z-score for the specified column in the DataFrame,
    and returns a DataFrame with an additional column for the Z-score and a DataFrame
    containing the outliers where the absolute Z-score is greater than the specified
    threshold.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to process.
    target_col : str, optional
        Column name to calculate the Z-score for. If not specified, the first column
        in the DataFrame is used.
    threshold : float, optional
        The absolute Z-score threshold for outliers. The default is 3.

    Returns
    -------
    pandas.DataFrame, pandas.DataFrame
        The first DataFrame contains the original DataFrame with an additional column
        for the Z-score and the second DataFrame contains the outliers.
    """
    df["z_score"] = zscore(df[target_col])
    outliers = df[df["z_score"].abs() > threshold]
    print(outliers)
    return df, outliers

def remove_outliers_by_z_score(df, target_col=None, threshold=3):
    """
    Remove outliers from a DataFrame column based on the Z-score method.

    This function calculates the Z-score for the specified column in the DataFrame,
    identifies outliers based on the provided threshold, and removes them.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - target_col (str, optional): The column name to check for outliers. If None, no operation is performed.
    - threshold (float, optional): The Z-score threshold to identify outliers. Default is 3.

    Returns:
    - pd.DataFrame: A DataFrame with outliers removed from the specified column.
    """

    df, outliers = outliers_by_z_score(df, target_col, threshold)
    try:
        df = df.drop(outliers.index)
        return df
    except Exception as exc:
        print(exc)
        df = df[df["z_score"].abs() <= threshold]
        return df
