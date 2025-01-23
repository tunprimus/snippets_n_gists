#!/usr/bin/env python3
import pandas as pd

pd.set_option("mode.copy_on_write", True)

def min_max_normalisation(df, target_col):
    """
    Perform min-max normalisation on a pandas DataFrame column.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the column to be normalised.
    target_col : str
        Name of the column to be normalised.

    Returns
    -------
    pandas.DataFrame
        DataFrame with the normalised column.
    """
    df[target_col] = (df[target_col] - df[target_col].min()) / (df[target_col].max() - df[target_col].min())

