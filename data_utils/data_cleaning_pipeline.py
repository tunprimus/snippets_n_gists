#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from os.path import realpath as realpath

np.float = np.float64
np.int = np.int_
np.object = np.object_
np.bool = np.bool_

pd.set_option("mode.copy_on_write", True)

def drop_duplicates(df, subset_name):
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


def remove_missing_values(df):
    """
    Remove rows with missing values from the DataFrame.

    Parameters
    ----------
    df : DataFrame
        The DataFrame from which to remove rows with missing values.

    Returns
    -------
    df : DataFrame
        The DataFrame with rows containing missing values removed.
    """
    # find missing values
    missing_values = df.isnull().sum()
    # remove rows with missing values
    df = df.dropna()
    # print number of missing values removed
    print(f"Removed {missing_values.sum()} missing values")
    return df


def data_cleaning_pipeline(df_path, duplication_subset, col_to_encode, col_with_outliers, col_with_date):
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
        df_no_duplicates = drop_duplicates(df, duplication_subset)
        df_encoded = encode(df_no_duplicates, col_to_encode)
        df_no_outliers = handle_outliers_with_iqr(df_encoded, col_with_outliers)
        df_date_formatted = date_formatting(df_no_outliers, col_with_date)
        df_no_nulls = remove_missing_values(df_date_formatted)
        return df_no_nulls
    except Exception as exc:
        print(exc)
        return None


