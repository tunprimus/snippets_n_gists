#!/usr/bin/env python3
# Adapted from: Convert an Excel dataset into a SQL insert statement
# https://dev.to/smuniraj/convert-excel-dataset-into-sql-insert-statement-38k4
import pandas as pd
from os.path import expanduser, realpath


def csv_to_sql_insert(path_to_csv, path_to_sql_output=".", table_name="Test_Table_Name"):
    real_path_to_csv = realpath(expanduser(path_to_csv))
    real_path_to_sql_output = realpath(expanduser(path_to_sql_output))
    # Generate SQL insert statements
    table_name = table_name
    sql_statements = []

    df = pd.read_csv(real_path_to_csv)

    for index, row in df.iterrows():
        columns = ", ".join(row.index)
        values = ", ".join([f"'{str(value)}'" for value in row.values])
        sql_statements.append(f"INSERT INTO {table_name} ({columns}) VALUES ({values});")

    # Save the SQL statements to a file
    with open(f"{real_path_to_sql_output}/insert_statements.sql", "w") as fop:
        for statement in sql_statements:
            fop.write(statement + "\n")

if __name__ == "__main__":
    csv_to_sql_insert()

