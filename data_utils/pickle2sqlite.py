#!/usr/bin/env python3
import sqlite3
import pickle
import codecs
import sys

from os.path import expanduser, realpath


python_version = sys.version.split(" ")[0].replace(".", "_")

"""
Adapted from https://gist.github.com/molpopgen/aa6225a18466591213880d320748f9bc
"""
def to_pickle_encoded(filename):
    """
    Read a file and encode it using base64 for storage in a sqlite database.

    Parameters
    ----------
    filename : str
        The path to the file to read.

    Returns
    -------
    str
        The base64 encoded string.
    """
    filename = realpath(expanduser(filename))

    with open(filename, "rb") as fin:
        to_dump = pickle.dumps(fin, -1)
        filename_encoded = codecs.encode(to_dump, "base64").decode()
        filename_encoded = f"{filename_encoded}-{python_version}"

    return filename_encoded


def save_pickle_to_sqlite(pickled_file, db_name="test.db", id_val=1):
    """
    Save a pickled file to a sqlite database.

    Parameters
    ----------
    pickled_file : str
        The base64 encoded string to store in the database.
    db_name : str
        The name of the database to store the file in.
    id_val : int
        The id to use for the insert statement.

    Returns
    -------
    int
        The id of the inserted row.
    """
    real_path_to_db = realpath(expanduser(db_name))

    with sqlite3.connect(real_path_to_db) as conn:
        conn.execute("create table pickle_store (id integer, pickled text, python_version text)")
        conn.execute(
            'insert into pickle_store values ({}, "{}", "{}")'.format(id_val, pickled_file, python_version)
        )
        conn.commit()
        print(id_val)
        return id_val


def load_pickle_from_sqlite(db_name="test.db", id_val=1):
    """
    Load a pickled file from a sqlite database.

    Parameters
    ----------
    db_name : str
        The name of the database to load the file from.
    id_val : int
        The id to use for the select statement.

    Returns
    -------
    object
        The unpickled object.
    """
    real_path_to_db = realpath(expanduser(db_name))

    with sqlite3.connect(real_path_to_db) as conn:
        cursor = conn.cursor()
        cursor.execute("select pickled from pickle_store where id == id_val")
        row = cursor.fetchone()
        print(row)
        result = pickle.loads(codecs.decode(row[0].encode(), "base64"))
        print(result)
        return result

