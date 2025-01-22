import pandas as pd
import numpy as np
import scipy.stats.gmean as gmean
import scipy.special.agm as ss_agm

np.float = np.float64
np.int = np.int_
np.object = np.object_
np.bool = np.bool_

pd.set_option("mode.copy_on_write", True)


def columnar_arithmetic_geometric_mean(df, target_col):
    """
    Calculate the arithmetic-geometric mean of a given column in a pandas DataFrame.

    The arithmetic-geometric mean is the limit of the sequence of values obtained by
    repeatedly computing the arithmetic mean and geometric mean of the previous two
    values in the sequence, starting from the arithmetic and geometric mean of the
    given input vector.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the column to be processed.
    target_col : str
        Name of the column in the DataFrame for which the arithmetic-geometric mean
        is to be calculated.

    Returns
    -------
    agm : float
        Arithmetic-geometric mean of the target column in the DataFrame.
    """
    # Get arithmetic mean of the target column
    ari_mean = df[target_col].mean()
    # Get the geometric mean of the target column
    try:
        geo_mean = gmean(df[target_col], nan_policy="omit")
    except Exception as exc:
        geo_mean = gmean(df.loc[:, target_col], nan_policy="omit")
    # Get the arithmetic-geometric mean of the target column
    agm = ss_agm(ari_mean, geo_mean)
    return agm


def apply_agm_to_nan(df, target_col):
    """
    Fill NaN values in a DataFrame column with the AGM of the respective column.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the column to be processed.
    target_col : str
        Name of the column in the DataFrame for which the NaNs are to be filled.

    Returns
    -------
    None
    """
    df[target_col] = np.where(np.isnan(df[target_col]), columnar_arithmetic_geometric_mean(df, target_col), df[target_col])
