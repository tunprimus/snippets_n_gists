#!/usr/bin/env python3
import pandas as pd
import numpy as np
from scipy.stats import zscore
from scipy.special import agm
from sklearn.preprocessing import LabelEncoder
from os.path import realpath as realpath

np.float = np.float64
np.int = np.int_
np.object = np.object_
np.bool = np.bool_

pd.set_option("mode.copy_on_write", True)

def drop_duplicates(df, subset_name=None):
    """
    Drop duplicates of a dataframe based on a subset of columns.

    Parameters
    ----------
    df : DataFrame
        The DataFrame to remove duplicates from.
    subset_name : str
        The name of the column to consider when dropping duplicates.

    Returns
    -------
    df : DataFrame
        The DataFrame with duplicates removed.
    """
    df.drop_duplicates(subset=[subset_name], inplace=True)
    return df


def transform_data_types(df, col_types):
    """
    Transform the data types of a DataFrame according to a dictionary of column names to data types.

    Parameters
    ----------
    df : DataFrame
        The DataFrame to transform.
    col_types : dict
        A dictionary of column names to data types.

    Returns
    -------
    df : DataFrame
        The DataFrame with the transformed data types.
    """
    for col, dtype in col_types.items():
        df[col] = df[col].astype(dtype)
    return df


def encode(df, col_to_encode):
    """
    Encode a categorical column in a DataFrame using a LabelEncoder.

    Parameters
    ----------
    df : DataFrame
        The DataFrame with the categorical column to encode.
    col_to_encode : str
        The name of the column to encode.

    Returns
    -------
    df : DataFrame
        The DataFrame with the encoded column.
    """
    le = LabelEncoder()
    df[col_to_encode] = le.fit_transform(df[col_to_encode])
    return df


def select_outliers_by_z_score(df, target_col=None, threshold=3):
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

def handle_outliers_by_z_score(df, target_col=None, threshold=3):
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

    df, outliers = select_outliers_by_z_score(df, target_col, threshold)
    try:
        df = df.drop(outliers.index)
        return df
    except Exception as exc:
        print(exc)
        df = df[df["z_score"].abs() <= threshold]
        return df


def handle_outliers_with_iqr(df, col_with_outliers):
    """
    Remove outliers from a column in a DataFrame based on the Interquartile Range (IQR) method.

    Parameters
    ----------
    df : DataFrame
        The DataFrame with the column containing outliers.
    col_with_outliers : str
        The name of the column containing outliers.

    Returns
    -------
    df : DataFrame
        The DataFrame with outliers removed.
    """
    q1 = df[col_with_outliers].quantile(0.25)
    q3 = df[col_with_outliers].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    # remove outliers
    df = df[(df[col_with_outliers] > lower_bound) & (df[col_with_outliers] < upper_bound)]
    return df


def date_formatting(df, col_with_date):
    """
    Convert a column in a DataFrame to datetime format.

    Parameters
    ----------
    df : DataFrame
        The DataFrame with the column to convert.
    col_with_date : str
        The name of the column to convert.

    Returns
    -------
    df : DataFrame
        The DataFrame with the converted column.
    """
    df[col_with_date] = pd.to_datetime(df[col_with_date])
    return df


def handle_missing_values(df, method="drop", fill_value=None):
    """
    Handle missing values in a DataFrame.

    Parameters
    ----------
    df : DataFrame
        The DataFrame to handle missing values from.
    method : str, optional
        The method to use for handling missing values. Must be one of ['drop', 'fill', 'mean', 'median', 'mode', 'agm', 'impute', 'interpolate'].
    fill_value : float, optional
        The value to fill missing values with when method is 'fill'.

    Returns
    -------
    df : DataFrame
        The DataFrame with missing values handled.

    Notes
    -----
    * When method is 'drop', rows with missing values are removed.
    * When method is 'fill', missing values are replaced with fill_value.
    * When method is 'mean', 'median', 'mode', or 'agm', missing values are replaced with the mean, median, mode, or arithmetic-geometric mean of the column, respectively.
    * When method is 'impute', missing values are imputed using backwards and forwards filling.
    * When method is 'interpolate', missing values are interpolated.
    """
    if method == "drop":
        # find missing values
        missing_values = df.isnull().sum()
        # remove rows with missing values
        df = df.dropna()
        # print number of missing values removed
        print(f"Removed {missing_values.sum()} missing values")
    elif method == "fill":
        return df.fillna(fill_value)
    elif method == "mean":
        return df.fillna(df.mean())
    elif method == "median":
        return df.fillna(df.median())
    elif method == "mode":
        return df.fillna(df.mode().iloc[0])
    elif method == "agm":
        return df.fillna(agm(df))
    elif method == "impute":
        return df.fillna(method="bfill").fillna(method="ffill")
    elif method == "interpolate":
        return df.interpolate()
    else:
        raise ValueError(f"Invalid method: {method} provided. Must be one of ['drop', 'fill', 'mean', 'median', 'mode', 'agm', 'impute', 'interpolate']")
    return df


def data_cleaning_pipeline(df_path, missing_val_method="agm", fill_value=None, duplication_subset=None, col_to_encode=None, col_with_outliers=None, col_with_date=None, col_types=None):
    """
    Applies a series of cleaning functions to the data loaded from the path.

    Parameters
    ----------
    df_path : str
        The path to the data file.
    duplication_subset : list of str
        The columns to consider when finding duplicates.
    col_to_encode : str
        The column to encode with the LabelEncoder.
    col_with_outliers : str
        The column to handle outliers for with the IQR method.
    col_with_date : str
        The column to format as a datetime.

    Returns
    -------
    df : DataFrame
        The cleaned DataFrame.
    """
    real_path_to_df = realpath(df_path)
    # Load the data
    try:
        df = pd.read_csv(real_path_to_df)
    except Exception as exc:
        print(exc)
        df = pd.read_excel(real_path_to_df)
    # Apply functions to clean data
    try:
        df_no_duplicates = drop_duplicates(df=df, subset_name=duplication_subset)
        df_encoded = encode(df=df_no_duplicates, col_to_encode=col_to_encode)
        df_no_outliers = handle_outliers_with_iqr(df=df_encoded, col_with_outliers=col_with_outliers) or handle_outliers_with_iqr(df=df_encoded, col_with_outliers=col_with_outliers)
        df_date_formatted = date_formatting(df=df_no_outliers, col_with_date=col_with_date)
        df_no_nulls = handle_missing_values(df=df_date_formatted, method=missing_val_method, fill_value=fill_value)
        if col_types:
            df_no_nulls = df_no_nulls.astype(col_types)
        return df_no_nulls
    except Exception as exc:
        print(exc)
        return None


