#!/usr/bin/env python3
# From https://pythonforthelab.com/blog/storing-data-with-sqlite/
import sqlite3
import numpy as np
import io


def adapt_array(arr):
    """
    https://stackoverflow.com/a/18622264/4467480
    Converts a numpy array into binary format suitable for storing in an SQLite database.

    :param arr: numpy array to be converted
    :return: binary representation of the numpy array
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def convert_array(text):
    """
    Converts binary data read from sqlite database back into a numpy array.
    :param text: a bytes object read from sqlite database
    :return: a numpy array
    """
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("array", convert_array)

# It is important to note that for the above code to work when you start the connection with the database, you should add the following:
#
# conn = sqlite3.connect("database_name.db", detect_types=sqlite3.PARSE_DECLTYPES)
# with sqlite3.connect("database_name.db", detect_types=sqlite3.PARSE_DECLTYPES) as conn:
#
