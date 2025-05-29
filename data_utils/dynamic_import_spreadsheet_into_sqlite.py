#!/usr/bin/env python3
# Modified from https://medium.com/@ccpythonprogramming/dynamically-import-csv-files-into-sqlite-with-python-3c9ba07fe559
import sqlite3
import pandas as pd
from os.path import expanduser, realpath

pd.set_option("mode.copy_on_write", True)


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
    columns_defn = ", ".join(f"{col} TEXT" for col in columns) if isinstance(columns, (pd.core.indexes.base.Index, pd.core.frame.DataFrame)) else columns
    columns_defn = columns_defn + ",added_on DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL"

    table_create_stmt = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_defn})"
    cur.execute(table_create_stmt)


def insert_into_db(table_name, df, path_to_database="./test.db"):
    """
    Insert data from a DataFrame into a specified table within an SQLite database.

    Parameters
    ----------
    table_name : str
        The name of the table to insert data into.

    df : pandas.DataFrame
        The DataFrame containing data to be inserted into the table.

    path_to_database : str, optional
        The file path to the SQLite database. Defaults to "./test.db".

    Returns
    -------
    None
    """
    real_path_to_database = realpath(expanduser(path_to_database))

    with sqlite3.connect(real_path_to_database) as conn:
        cur = conn.cursor()

        # Generate sanitised column labels
        col_labels = ', '.join(df.columns).replace('/', '_').replace('-', '_').replace(' ', '')

        # Create table based on DataFrame columns
        create_table(cur, table_name, col_labels)

        # Insert data into the table
        try:
            for _, row in df.iterrows():
                cur.execute(
                    f"INSERT OR REPLACE INTO {table_name} ({col_labels}) VALUES ({', '.join(['?'] * len(df.columns))})",
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

    df = pd.read_csv(real_path_to_file)
    table_name = real_path_to_file.split("/")[-1].replace(".csv", "").replace("-", "_").replace("/", "")
    insert_into_db(table_name, df, real_path_to_database)


def process_spreadsheet_into_sqlite(file_path, sheet_name=None, path_to_database="./test.db"):
    real_path_to_file = realpath(file_path)
    real_path_to_database = realpath(expanduser(path_to_database))

    df = pd.read_excel(real_path_to_file)
    table_name = (
        sheet_name if sheet_name else real_path_to_file.split("/")[-1].replace(".xlsx", "").replace("-", "_").replace("/", "")
    )
    insert_into_db(table_name, df, real_path_to_database)
