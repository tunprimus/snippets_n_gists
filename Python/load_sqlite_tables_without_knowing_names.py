import sqlite3
from pandas import read_sql_query, read_sql_table
from os.path import realpath as realpath

# Parsing a sqlite database into a dictionary of DataFrames without knowing the table names

def read_sqlite(path_to_db):
    real_path_to_db = realpath(path_to_db)

    with sqlite3.connect(real_path_to_db) as db_connection:
        db_tables = list(read_sql_table("SELECT name FROM sqlite_master WHERE type = 'table';", db_connection)['name'])
        out_dict = {tbl: read_sql_query(f"SELECT * FROM {tbl}", db_connection) for tbl in db_tables}
    
    return out_dict