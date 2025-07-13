#!/usr/bin/env python3
import sqlite3
from pandas import read_sql_query, read_sql_table
from os.path import expanduser, realpath

# Parsing a sqlite database into a dictionary of DataFrames without knowing the table names

def read_all_sqlite_tables(path_to_db):
    """
    Reads in a SQLite database from a specified file path and returns a dictionary of
    Pandas DataFrames, where each key is the name of a table in the database.

    Parameters
    ----------
    path_to_db : str
        The file path to the SQLite database.

    Returns
    -------
    dict of pandas.DataFrame
        A dictionary of DataFrames, where each key is the name of a table in the database.
    """
    import sqlite3
    from pandas import read_sql_query, read_sql_table
    from os.path import expanduser, realpath

    real_path_to_db = realpath(expanduser(path_to_db))

    with sqlite3.connect(real_path_to_db) as db_connection:
        db_tables = read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", db_connection)['name']
        out_dict = {tbl: read_sql_query(f"SELECT * FROM {tbl}", db_connection) for tbl in db_tables}

    return out_dict
