#!/usr/bin/env python3
import sqlite3
import sys
import uuid
try:
    import fireducks.pandas as pd
except ImportError:
    import pandas as pd
from os.path import realpath, expanduser
from ulid import ULID
from uuid_extensions import uuid7, uuid7str

sys.path.append(realpath(expanduser("~/zzz_personal/snippets_n_gists/data_utils")))

PATH_TO_COMBO_DB = (
    "../../Data_Science_Analytics/000_common_dataset/combo-dataset-in-sqlite.db"
)

from standardise_column_names import standardise_column_names
from dynamic_import_spreadsheet_into_sqlite import (
    create_table,
    insert_into_db,
    process_csv_into_sqlite,
    process_spreadsheet_into_sqlite,
)


def transform_tabular_data_into_sqlite(
    file_path, sheet_name=None, path_to_database="./test.db"
):
    """
    Take a file path and a database path, and copy the data from the file into
    the database. If the file is a CSV or TSV, use process_csv_into_sqlite. If the file
    is a spreadsheet, use process_spreadsheet_into_sqlite.

    Parameters
    ----------
    file_path : str
        The file path to the file to be inserted into the database.

    sheet_name : str, optional
        The name of the sheet in the Excel spreadsheet. Defaults to None.

    path_to_database : str, optional
        The path to the SQLite database. Defaults to "./test.db".
    """
    real_path_to_file = realpath(expanduser(file_path))
    real_path_to_database = realpath(expanduser(path_to_database))

    try:
        process_csv_into_sqlite(real_path_to_file, real_path_to_database)
    except:
        process_spreadsheet_into_sqlite(
            real_path_to_file, sheet_name, real_path_to_database
        )


# transform_tabular_data_into_sqlite(
#     file_path="../../../?", sheet_name=None, path_to_database=PATH_TO_COMBO_DB
# )


if __name__ == "__main__":
    transform_tabular_data_into_sqlite(sys.argv[1], sys.argv[2], sys.argv[3])
