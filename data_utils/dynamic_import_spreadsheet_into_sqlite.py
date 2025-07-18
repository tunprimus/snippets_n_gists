#!/usr/bin/env python3
# Modified from https://medium.com/@ccpythonprogramming/dynamically-import-csv-files-into-sqlite-with-python-3c9ba07fe559
import pandas as pd
import sqlite3
import sys
import uuid
from os.path import expanduser, realpath
from ulid import ULID
from uuid_extensions import uuid7, uuid7str

pd.set_option("mode.copy_on_write", True)

sys.path.append(realpath(expanduser("~/zzz_personal/snippets_n_gists/data_utils")))

from standardise_column_names import standardise_column_names


# Create UUID/ULID/UUIDv7 functions that would be registered with sqlite3.
# Adapted from: https://stackoverflow.com/a/39690541 , https://docs.python.org/2/library/sqlite3.html#sqlite3.Connection.create_function and https://github.com/manufaktor/articles/blob/main/using-uuid-in-sqlite.md
def uuid4_func():
    return str(uuid.uuid4())


# Create a ULID function.
def ulid_func():
    return str(ULID())


# Create a UUIDv7 function.
def uuid7_func():
    return uuid7str()


def create_table(cur, table_name, columns):
    """
    Create a table in SQLite with the given table name and column names.

    Parameters
    ----------
    cur : sqlite3.Cursor
        The database connection cursor.

    table_name : str
        The name of the table to create.

    columns : list of str
        The column names of the table to create.

    Returns
    -------
    None
    """
    # pre_col_defn = "id INTEGER PRIMARY KEY AUTOINCREMENT, ulid_uuidv7 DEFAULT NOT NULL, "
    # pre_col_defn = "id INT AUTO_INCREMENT PRIMARY KEY, ulid_uuidv7 DEFAULT NOT NULL, "
    # pre_col_defn = f"ulid_uuidv7 UUID DEFAULT (uuid4()) NOT NULL, "
    # pre_col_defn = f"ulid_uuidv7 UUID DEFAULT (uuid7()) NOT NULL, "
    pre_col_defn = f"ulid_uuidv7 UUID DEFAULT (ulid()) NOT NULL, "
    buffer_col_defn = (
        ", ".join(f"{col} TEXT" for col in columns)
        if isinstance(columns, (pd.core.indexes.base.Index, pd.core.frame.DataFrame))
        else columns
    )
    # post_col_defn = ", created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL, updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP, deleted_at DATETIME DEFAULT NULL"
    post_col_defn = ", created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL, updated_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL, deleted_at DATETIME DEFAULT NULL"
    columns_defn = f"{pre_col_defn}{buffer_col_defn}{post_col_defn}"

    table_create_stmt = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_defn})"
    cur.execute(table_create_stmt)

    # Create trigger for auto-updating columns later
    # Adapted from https://stackoverflow.com/a/1964534 and https://stackoverflow.com/a/65237351
    table_trigger_stmt = f"""
    CREATE TRIGGER {table_name}_trig
    AFTER UPDATE ON {table_name}
        BEGIN
            update {table_name} SET updated_at = datetime('now') WHERE ulid_uuidv7 = NEW.ulid_uuidv7;
        END;
    """
    cur.execute(table_trigger_stmt)


def insert_into_db(
    table_name,
    df,
    path_to_database="./test.db",
    use_uuid4=False,
    use_ulid=True,
    use_uuid7=False,
):
    """
    Insert data from a pandas DataFrame into a SQLite database.

    Parameters
    ----------
    table_name : str
        The name of the table to insert into.

    df : pandas.DataFrame
        The DataFrame containing the data to insert.

    path_to_database : str, optional (default: "./test.db")
        The path to the SQLite database file.

    use_uuid4 : bool, optional (default: False)
        Whether to use the uuid4 function to generate a UUID for the "ulid_uuidv7" column.

    use_ulid : bool, optional (default: True)
        Whether to use the ulid function to generate a ULID for the "ulid_uuidv7" column.

    use_uuid7 : bool, optional (default: False)
        Whether to use the uuid7 function to generate a UUIDv7 for the "ulid_uuidv7" column.

    Returns
    -------
    None
    """
    real_path_to_database = realpath(expanduser(path_to_database))
    col_length = len(df.columns)

    with sqlite3.connect(real_path_to_database) as conn:
        # Register the UUID/ULID/UUID7 functions with SQLite. Only one can work at a time!

        if use_uuid4:
            conn.create_function("uuid4", 0, uuid4_func)

        if use_ulid:
            conn.create_function("ulid", 0, ulid_func)

        if use_uuid7:
            conn.create_function("uuid7", 0, uuid7_func)

        cur = conn.cursor()

        # Enable foreign key constraints
        cur.execute("PRAGMA foreign_keys = ON;")

        # Generate sanitised column labels
        col_labels = (
            ", ".join(df.columns).replace("/", "_").replace("-", "_").replace(" ", "")
        )

        # Create table based on DataFrame columns
        create_table(cur, table_name, col_labels)

        # Insert data into the table
        try:
            for _, row in df.iterrows():
                cur.execute(
                    f"INSERT OR REPLACE INTO {table_name} ({col_labels}) VALUES ({', '.join(['?'] * (col_length))})",
                    tuple(row),
                )
        except:
            df.to_sql(table_name, conn, if_exists="append", index=False)
        finally:
            conn.commit()


def process_csv_into_sqlite(file_path, path_to_database="./test.db"):
    """
    Insert data from a CSV file into a table within an SQLite database.

    Parameters
    ----------
    file_path : str
        The file path to the CSV file to be inserted into the database.

    path_to_database : str, optional
        The file path to the SQLite database. Defaults to "./test.db".

    Returns
    -------
    None
    """
    real_path_to_file = realpath(file_path)
    real_path_to_database = realpath(expanduser(path_to_database))

    if (real_path_to_file.endswith(".csv")) or (real_path_to_file.endswith(".tsv")):
        df = pd.read_csv(real_path_to_file)

    df = standardise_column_names(df)

    table_name = (
        real_path_to_file.split("/")[-1]
        .replace(".csv", "")
        .replace(".tsv", "")
        .replace("-", "_")
        .replace("/", "")
    )
    insert_into_db(table_name, df, real_path_to_database)


def process_spreadsheet_into_sqlite(
    file_path, sheet_name=None, path_to_database="./test.db"
):
    """
    Insert data from a spreadsheet into a table within an SQLite database.

    Parameters
    ----------
    file_path : str
        The file path to the spreadsheet file to be inserted into the database.

    sheet_name : str, optional
        The name of the sheet in the spreadsheet. Defaults to None.

    path_to_database : str, optional
        The file path to the SQLite database. Defaults to "./test.db".

    Returns
    -------
    None
    """
    real_path_to_file = realpath(file_path)
    real_path_to_database = realpath(expanduser(path_to_database))

    if (
        (real_path_to_file.endswith(".xlsx"))
        or (real_path_to_file.endswith(".xls"))
        or (real_path_to_file.endswith(".ods"))
        or (real_path_to_file.endswith(".xlsm"))
        or (real_path_to_file.endswith(".xlsb"))
    ):
        df = pd.read_excel(real_path_to_file)

    df = standardise_column_names(df)

    table_name = (
        sheet_name.lower()
        if sheet_name
        else real_path_to_file.split("/")[-1]
        .replace(".xlsx", "")
        .replace(".xls", "")
        .replace(".ods", "")
        .replace(".xlsm", "")
        .replace(".xlsb", "")
        .replace("-", "_")
        .replace("/", "")
        .lower()
    )
    insert_into_db(table_name, df, real_path_to_database)
