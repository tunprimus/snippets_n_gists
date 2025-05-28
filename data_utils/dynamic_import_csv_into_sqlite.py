#!/usr/bin/env python3
# Modified from https://medium.com/@ccpythonprogramming/dynamically-import-csv-files-into-sqlite-with-python-3c9ba07fe559
import sqlite3
import pandas as pd
from os.path import expanduser, realpath

pd.set_option("mode.copy_on_write", True)


# Path to Database
path_to_database = "./test.db"
real_path_to_database = realpath(expanduser(path_to_database))


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
    columns_defn = ", ".join(f"{col} TEXT" for col in columns)
    table_create_stmt = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_defn})"
    cur.execute(table_create_stmt)


def insert_into_db(table_name, df):
    """
    Insert data from a pandas DataFrame into a SQLite database table.

    Parameters
    ----------
    table_name : str
        The name of the table to insert into.
    df : pandas.DataFrame
        The DataFrame with data to be inserted.

    Returns
    -------
    None
    """
    with sqlite3.connect(real_path_to_db) as conn:
        cur = conn.cursor()
        # Create table based on DataFrame columns
        create_table(cur, table_name, df.columns)
        # Insert data into the table
        try:
            for _, row in df.iterrows():
                cur.execute(
                    f"INSERT INTO {table_name} ({', '.join(df.columns)}) VALUES ({', '.join(['?'] * len(df.columns))})",
                    tuple(row),
                )
        except:
            df.to_sql(table_name, conn, if_exists="append", index=False)
        finally:
            conn.commit()


def process_csv_into_sqlite(file_path):
    """
    Read a CSV file into a pandas DataFrame and insert the data into a SQLite database table.

    Parameters
    ----------
    file_path : str
        The path to the CSV file to read.

    Returns
    -------
    None
    """
    real_path_to_file_path = realpath(file_path)
    df = pd.read_csv(real_path_to_file_path)
    table_name = file_path.split("/")[-1].replace(".csv", "")
    insert_into_db(table_name, df)


def process_spreadsheet_into_sqlite(file_path, sheet_name=None):
    """
    Read an Excel spreadsheet into a pandas DataFrame and insert the data into a SQLite database table.

    Parameters
    ----------
    file_path : str
        The path to the Excel file to read.
    sheet_name : str, optional
        The name of the sheet to read. If not provided, the first sheet is used and the table name is derived
        from the file name.

    Returns
    -------
    None
    """
    real_path_to_file_path = realpath(file_path)
    df = pd.read_excel(real_path_to_file_path)
    table_name = (
        sheet_name if sheet_name else file_path.split("/")[-1].replace(".xlsx", "")
    )
    insert_into_db(table_name, df)

