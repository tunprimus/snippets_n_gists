#!/usr/bin/env python3
# Adapted from: Top 5 Ways to Move Columns in a Pandas DataFrame -> https://sqlpey.com/python/move-columns-in-a-pandas-dataframe-python/

def reorder_pandas_columns(df, pop_n_insert=False, col_to_pop=None, index_to_insert=None, reindex=False, new_cols_for_reindex=[], custom_reorder=False, custom_first_cols=[], custom_last_cols=[], custom_drop_cols=[]):
    """
    Reorders columns in a Pandas DataFrame using one of three methods.

    Parameters:
    df (Pandas DataFrame): The DataFrame to reorder.
    pop_n_insert (bool): If True, pop the column specified by col_to_pop and insert it at the index specified by index_to_insert.
    col_to_pop (str): The column to pop.
    index_to_insert (int): The index to insert the popped column at.
    reindex (bool): If True, reindex the DataFrame with the columns specified by new_cols_for_reindex.
    new_cols_for_reindex (list): The columns to reindex the DataFrame with.
    custom_reorder (bool): If True, reorder the columns by first adding custom_first_cols, then adding any columns not in custom_first_cols, custom_last_cols, or custom_drop_cols, and finally adding custom_last_cols.
    custom_first_cols (list): The columns to add first.
    custom_last_cols (list): The columns to add last.
    custom_drop_cols (list): The columns to drop.

    Returns:
    Pandas DataFrame: The reordered DataFrame.
    """
    import pandas as pd
    pd.set_option("mode.copy_on_write", True)

    if pop_n_insert and col_to_pop and index_to_insert:
        col_to_move = df.pop(col_to_pop)
        df.insert(index_to_insert, col_to_pop, col_to_move)
    elif reindex and new_cols_for_reindex:
        old_cols = df.columns.values
        new_cols = new_cols_for_reindex
        df = df.reindex(columns=new_cols)
    elif custom_reorder:
        n_columns = df.columns.tolist()
        n_columns = list(set(n_columns) - set(custom_first_cols))
        n_columns = list(set(n_columns) - set(custom_drop_cols))
        n_columns = list(set(n_columns) - set(custom_last_cols))
        new_order = custom_first_cols + n_columns + custom_last_cols
        df = df[new_order]

    return df
