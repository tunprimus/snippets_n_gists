#!/usr/bin/env python3
import pandas as pd
from os.path import realpath as realpath

pd.set_option("mode.copy_on_write", True)


def get_pandas_schema(file_path):
    """Return a dictionary of column names and data types from the given file."""
    try:
        df = pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        df = pd.read_excel(file_path)
    return dict(zip(df.columns.tolist(), df.dtypes.tolist()))
