#!/usr/bin/env python3
# Adapted from @georgerichardson -> https://gist.github.com/georgerichardson/db66b686b4369de9e7196a65df45fc37
def standardise_column_names(df, remove_punct=True):
    """
    Standardises column names in a pandas DataFrame. By default, removes punctuation, replaces spaces with underscores and removes trailing underscores.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to standardise
    remove_punct : bool, optional
        Whether to remove punctuation from column names. Defaults to True.

    Returns
    -------
    pandas.DataFrame
        DataFrame with standardised column names
    Example
    -------
    >>> df = pd.DataFrame({'Column With Spaces': [1,2,3,4,5],
                            'Column-With-Hyphens&Others/': [6,7,8,9,10],
                            'Too    Many Spaces': [11,12,13,14,15],
                            })
    >>> df = standardise_column_names(df)
    >>> print(df.columns)
    Index(['column_with_spaces',
            'column_with_hyphens_others',
            'too_many_spaces'], dtype='object')
    """
    import re
    import string
    try:
        import fireducks.pandas as pd
    except ImportError:
        import pandas as pd

    # Handle CamelCase
    for i, col in enumerate(df.columns):
        matches = re.findall(re.compile("[a-z][A-Z]"), col)
        column = col
        for match in matches:
            column = column.replace(match, f"{match[0]}_{match[1]}")
            df = df.rename(columns={df.columns[i]: column})
    # Main cleaning
    translator = str.maketrans(string.punctuation, " " * len(string.punctuation))
    for c in df.columns:
        c_mod = c.lower()
        if remove_punct:
            c_mod = c_mod.translate(translator)
        c_mod = "_".join(c_mod.split(" "))
        if c_mod.endswith("_") or c_mod[-1] == "_":
            c_mod = c_mod[:-1]
        c_mod = re.sub(r"\_+", "_", c_mod)
        df.rename({c: c_mod}, inplace=True, axis=1)
    # Further cleaning of unusual languages
    df.columns = (
        df.columns.str.replace("\n", "_", regex=False)
        .str.replace("(", "_", regex=False)
        .str.replace(")", "_", regex=False)
        .str.replace("'", "_", regex=False)
        .str.replace('"', "_", regex=False)
        .str.replace(".", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace(r"[!?:;/]", "_", regex=True)
        .str.replace("+", "_plus_", regex=False)
        .str.replace("*", "_times_", regex=False)
        .str.replace("<", "_smaller", regex=False)
        .str.replace(">", "_larger_", regex=False)
        .str.replace("=", "_equal_", regex=False)
        .str.replace("ä", "ae", regex=False)
        .str.replace("ö", "oe", regex=False)
        .str.replace("ü", "ue", regex=False)
        .str.replace("ß", "ss", regex=False)
        .str.replace("%", "_pct_", regex=False)
        .str.replace("$", "_dollar_", regex=False)
        .str.replace("€", "_euro_", regex=False)
        .str.replace("@", "_at_", regex=False)
        .str.replace("#", "_hash_", regex=False)
        .str.replace("&", "_and_", regex=False)
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
        .str.lower()
    )
    return df


