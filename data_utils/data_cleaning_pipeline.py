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
    df.drop_duplicates(subset=[subset_name], inplace=True)
    return df


def encode(df, col_to_encode):
    le = LabelEncoder()
    df[col_to_encode] = le.fit_transform(df[col_to_encode])
    return df


def handle_outliers_with_iqr(df, col_with_outliers):
    q1 = df[col_with_outliers].quantile(0.25)
    q3 = df[col_with_outliers].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    # remove outliers
    df = df[(df[col_with_outliers] > lower_bound) & (df[col_with_outliers] < upper_bound)]
    return df


def date_formatting(df, col_with_date):
    df[col_with_date] = pd.to_datetime(df[col_with_date])
    return df


def remove_missing_values(df):
    # find missing values
    missing_values = df.isnull().sum()
    # remove rows with missing values
    df = df.dropna()
    # print number of missing values removed
    print(f"Removed {missing_values.sum()} missing values")
    return df


def data_cleaning_pipeline(df_path, duplication_subset, col_to_encode, col_with_outliers, col_with_date):
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


