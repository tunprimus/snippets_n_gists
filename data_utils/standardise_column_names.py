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

    translator = str.maketrans(string.punctuation, " "*len(string.punctuation))
    for c in df.columns:
        c_mod = c.lower()
        if remove_punct:
            c_mod = c_mod.translate(translator)
        c_mod = "_".join(c_mod.split(" "))
        if c_mod.endswith("_") or c_mod[-1] == "_":
            c_mod = c_mod[:-1]
        c_mod = re.sub(r"\_+", "_", c_mod)
        df.rename({c: c_mod}, inplace=True, axis=1)
    return df

