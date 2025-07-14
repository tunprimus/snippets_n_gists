#!/usr/bin/env python3
import oracledb
import pyodbc
import psycopg2
import pymysql
import sqlite3
from os.path import expanduser, realpath
from sqlalchemy import create_engine, text

def create_sql_connection(db_path="./test.db", db_type="sqlite", host_name=None, user_name=None, user_password=None, db_port=None, db_name=None, messages=True):
    """
    Establishes a connection to a database based on the specified database type.

    Parameters
    ----------
    db_path : str, optional
        The file path to the SQLite database. Defaults to "./test.db".

    db_type : str, optional
        The type of the database to connect to. Options include 'sqlite', 'mysql', 'postgres', 'oracle', 'mssql'.

    host_name : str, optional
        The hostname of the database server for non-SQLite databases.

    user_name : str, optional
        The username to use for authentication for non-SQLite databases.

    user_password : str, optional
        The password to use for authentication for non-SQLite databases.

    db_port : int, optional
        The port number to connect to for non-SQLite databases.

    db_name : str, optional
        The name of the database to connect to for non-SQLite databases.

    messages : bool, optional
        If True, print success or error messages. Defaults to True.

    Returns
    -------
    sqlalchemy.engine.base.Engine or sqlite3.Connection
        A SQLAlchemy engine or SQLite connection object depending on the database type.

    Raises
    ------
    ValueError
        If an invalid database type is provided.
    """
    real_path_to_db = realpath(expanduser(db_path))
    conn_engine = None

    # Try to connect to SQLite DB
    if db_type == "sqlite":
        conn_str = f"sqlite:///{real_path_to_db}"
        try:
            conn_engine = create_engine(conn_str, connect_args={"check_same_thread": False})
            if messages:
                print("Connection to SQLite DB using SQLAlchemy successful!")
        except Exception:
            try:
                conn_engine = sqlite3.connection(real_path_to_db)
                if messages:
                    print("Connection to SQLite DB using sqlite3 successful!")
            except sqlite3.Error as err:
                print(f"The error '{err}' occurred.")
        # Return connection for SQLite
        return conn_engine

    # Try to connect to MySQL DB
    elif db_type == "mysql":
        conn_str = f"mysql+pymysql://{user_name}:{user_password}@{host_name}:{db_port}/{db_name}"
        try:
            conn_engine = create_engine(conn_str)
            if messages:
                print("Connection to MySQL DB using SQLAlchemy successful!")
        except Exception:
            try:
                conn_engine = pymysql.connect(host=host_name, user=user_name, password=user_password, port=db_port, db=db_name)
                if messages:
                    print("Connection to MySQL DB using pymysql successful!")
            except pymysql.Error as err:
                print(f"The error '{err}' occurred.")
        # Return connection for MySQL
        return conn_engine

    # Try to connect to PostgreSQL DB
    elif db_type == "postgres":
        conn_str = f"postgresql://{user_name}:{user_password}@{host_name}:{db_port}/{db_name}"
        try:
            conn_engine = create_engine(conn_str)
            if messages:
                print("Connection to PostgreSQL DB using SQLAlchemy successful!")
        except Exception:
            try:
                conn_engine = psycopg2.connect(host=host_name, user=user_name, password=user_password, port=db_port, dbname=db_name)
                if messages:
                    print("Connection to PostgreSQL DB using psycopg2 successful!")
            except psycopg2.Error as err:
                print(f"The error '{err}' occurred.")
        # Return connection for PostgreSQL
        return conn_engine

    # Try to connect to Oracle DB
    elif db_type == "oracle":
        conn_str = f"oracle://{user_name}:{user_password}@{host_name}:{db_port}/{db_name}"
        try:
            conn_engine = create_engine(conn_str)
            if messages:
                print("Connection to Oracle DB using SQLAlchemy successful!")
        except Exception:
            try:
                conn_engine = oracledb.connect(user=user_name, password=user_password, dsn=f"{host_name}:{db_port}/{db_name}")
                if messages:
                    print("Connection to Oracle DB using cx_Oracle successful!")
            except cx_Oracle.Error as err:
                print(f"The error '{err}' occurred.")
        # Return connection for Oracle
        return conn_engine

    # Try to connect to Microsoft SQL Server DB
    elif db_type == "mssql":
        conn_str = f"mssql+pymssql://{user_name}:{user_password}@{host_name}:{db_port}/{db_name}"
        try:
            conn_engine = create_engine(conn_str)
            if messages:
                print("Connection to Microsoft SQL Server DB using SQLAlchemy successful!")
        except Exception:
            try:
                conn_engine = pyodbc.connect(f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={host_name};PORT={db_port};DATABASE={db_name};UID={user_name};PWD={user_password}")
                if messages:
                    print("Connection to Microsoft SQL Server DB using pyodbc successful!")
            except pyodbc.Error as err:
                print(f"The error '{err}' occurred.")
        # Return connection for Microsoft SQL Server
        return conn_engine

    # Invalid database type
    else:
        raise ValueError("Invalid database type. Please choose 'sqlite', 'mysql', 'postgres', 'oracle' or 'mssql'.")


def execute_on_database(db_path="./test.db", db_type="sqlite", host_name=None, user_name=None, user_password=None, db_port=None, db_name=None, sql_query=None, messages=True):
    """
    Create and execute a SQL query on a specified database using SQLAlchemy.

    Parameters
    ----------
    db_path : str, optional
        The file path to the SQLite database. Defaults to "test.db".

    db_type : str, optional
        The type of the database to connect to. Options include 'sqlite', 'mysql', 'postgres', 'oracle', 'mssql'.
        Defaults to "sqlite".

    host_name : str, optional
        The hostname of the database server for non-SQLite databases.

    user_name : str, optional
        The username for authentication for non-SQLite databases.

    user_password : str, optional
        The password for authentication for non-SQLite databases.

    db_port : int, optional
        The port number for connecting to non-SQLite databases.

    db_name : str, optional
        The name of the database to connect to for non-SQLite databases.

    sql_query : str, optional
        The SQL query to be executed on the database.

    messages : bool, optional
        If True, print success or error messages. Defaults to True.

    Returns
    -------
    sqlalchemy.engine.ResultProxy
        The result of the executed SQL query.

    Raises
    ------
    ValueError
        If an invalid database type is provided.
    """
    import sqlalchemy

    sqlalchemy_version = sqlalchemy.__version__
    sqlalchemy_version = float(sqlalchemy_version.split(".")[0] + "." + sqlalchemy_version.split(".")[1])

    conn_engine = create_sql_connection(db_path=db_path, db_type=db_type, host_name=host_name, user_name=user_name, user_password=user_password, db_port=db_port, db_name=db_name, messages=messages)

    if (sqlalchemy_version >= 1.4):
        with conn_engine.connect() as conn:
            result = conn.execute(text(sql_query))
            if messages:
                print("Database operation successful!")
            return result
    else:
        with conn_engine.connect() as conn:
            result = conn.execute(text(sql_query))
            conn.commit()
            if messages:
                print("Database operation successful!")
            return result

# if __name__ == "__main__":
#     execute_on_database(db_path="test.db", db_type="sqlite", sql_query="CREATE TABLE IF NOT EXISTS test (id INTEGER PRIMARY KEY, username TEXT NOT NULL, surname TEXT NOT NULL, firstname TEXT NOT NULL, email TEXT NOT NULL, created_on DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL)")

