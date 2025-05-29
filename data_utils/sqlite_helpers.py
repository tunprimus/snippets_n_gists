#!/usr/bin/env python3
import sqlite3
import time
from os.path import expanduser, realpath
try:
    from ulid import ULID
except ImportError:
    print("ulid module not found! Install with `pip3 install python-ulid[pydantic]` ")


def generate_ulid():
    ulid_obj = ULID().generate()
    return ulid_obj


def explain_query(search_name, path_to_db):
    """Explain the query execution plan for a SELECT query without an index."""

    real_path_to_db = realpath(expanduser(path_to_db))

    with sqlite3.connect(real_path_to_db) as connection:
        cursor = connection.cursor()

        # Create query statement
        query_stmt_01 = '''
        EXPLAIN QUERY PLAN
        SELECT * FROM Students WHERE name = ?;
        '''

        # Use EXPLAIN QUERY PLAN to analyse how the query is executed
        cursor.execute(query_stmt_01, (search_name,))

        # Fetch and display the query plan
        query_plan = cursor.fetchall()

        print("Query Plan:")
        for step in query_plan:
            print(step)


def create_index(path_to_db, table_name, column_name, index_name):
    """Create an index on the name column of a selected table."""

    real_path_to_db = realpath(expanduser(path_to_db))

    try:
        with sqlite3.connect(real_path_to_db) as connection:
            cursor = connection.cursor()

            # Create query statement
            create_index_query = '''
                CREATE INDEX IF NOT EXISTS index_name ON table_name (column_name);
                '''

            # Measure the start time
            start_time = time.perf_counter_ns()

            # Execute the SQL command to create the index
            cursor.execute(create_index_query)

            # Measure the end time
            end_time = time.perf_counter_ns()

            # Commit the changes
            connection.commit()

            print("Index on 'column_name' column created successfully!")

            # Calculate the total time taken
            elapsed_time = (end_time - start_time) / 1000

            # Display the results and the time taken
            print(f"Query completed in {elapsed_time:.5f} microseconds.")
    except sqlite3.IntegrityError as err:
        print(f"Error: Integrity constraint violated - {err}")
    except sqlite3.OperationalError as err:
        print(f"Error: Operational issue - {err}")
    except Exception as err:
        print(f"An unexpected error occurred: {err}")



def query_with_index(path_to_db, table_name, column_name, index_name, target_name):
    """Query the specified table using an index on the given column."""

    real_path_to_db = realpath(expanduser(path_to_db))

    with sqlite3.connect(real_path_to_db) as connection:
        cursor = connection.cursor()

        # SQL command to select a student by name
        select_query = 'SELECT * FROM table_name WHERE column_name = ?;'

        # Measure the execution time
        start_time = time.perf_counter_ns()

        # Execute the query with the provided student name
        cursor.execute(select_query, (target_name,))
        result = cursor.fetchall()

        end_time = time.perf_counter_ns()

        # Calculate the elapsed time in microseconds
        execution_time = (end_time - start_time) / 1000

        # Display results and execution time
        print(f"Query result: {result}")
        print(f"Execution time with index: {execution_time:.5f} microseconds")

