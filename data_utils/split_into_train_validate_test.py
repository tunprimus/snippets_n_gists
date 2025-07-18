#!/usr/bin/env python3
import pandas as pd


def split_into_train_validate_test(
    df_input,
    stratify_col_name="target",
    use_last_col_as_target=False,
    train_prop=0.74,
    test_prop=0.13,
    validate_prop=0.13,
    random_state=42,
    drop_stratify_col=True,
    return_y=True
):
    """
    Split a Pandas DataFrame into three pairs: train, validate and test.
    Adapted from: https://stackoverflow.com/a/60804119

    Parameters
    ----------
    df_input : Pandas DataFrame
        The (combined) input dataframe to be split.

    stratify_col_name : str, optional
        The name of a column in the dataframe to use as target and for stratified splitting. By default "target".

    use_last_col_as_target : bool, optional
        To use the last column in the dataframe as target. By default False.

    train_prop : float, optional
        The proportion of the input dataframe to use for the training set. By default 0.74.

    test_prop : float, optional
        The proportion of the input dataframe to use for the test set. By default 0.13.

    validate_prop : float, optional
        The proportion of the input dataframe to use for the validation set. By default 0.13.

    random_state : int, optional
        The seed used by the random number generator. By default 42.

    drop_stratify_col : bool, optional
        If True, the stratify_col_name column is dropped from the input dataframe. By default True.

    return_y : bool, optional
        If True, y_train, y_validate and y_test dataframes are returned. By default True.

    Returns
    -------
    df_train : Pandas DataFrame
        The subset of the input dataframe to use for training.

    df_validate : Pandas DataFrame
        The subset of the input dataframe to use for validation.

    df_test : Pandas DataFrame
        The subset of the input dataframe to use for testing.

    y_train : Pandas DataFrame, Series or numpy.ndarray
        The subset of the input dataframe to use for training.

    y_validate : Pandas DataFrame, Series or numpy.ndarray
        The subset of the input dataframe to use for validation.

    y_test : Pandas DataFrame, Series or numpy.ndarray
        The subset of the input dataframe to use for testing.

    Example
    -------
    df = pd.DataFrame({ "A": list(range(0, 1000)),
                    "B": list(range(1000, 0, -1)),
                    "label": ["foo"] * 750 + ["bar"] * 150 + ["baz"] * 100 })
    >>> df.shape
    (1000, 3)

    >>> df.label.value_counts()
    foo    750
    bar    150
    baz    100
    Name: label, dtype: int64

    >>> df_train, df_validate, df_test, y_train, y_validate, y_test = split_into_train_validate_test(df_input, stratify_col_name="y", train_prop=0.74, test_prop=0.13, validate_prop=0.13, random_state=42)

    >>> df_train.shape
    (740, 3)

    >>> df_validate.shape
    (130, 3)

    >>> df_test.shape
    (130, 3)

    >>> df_train["label"].value_counts()
    foo    555
    bar    111
    baz     74
    Name: label, dtype: int64

    >>> df_validate["label"].value_counts()
    foo    98
    bar    19
    baz    13
    Name: label, dtype: int64

    >>> df_test["label"].value_counts()
    foo    97
    bar    20
    baz    13
    Name: label, dtype: int64
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import StratifiedShuffleSplit

    # Validate total proportion
    try:
        assert train_prop + test_prop + validate_prop == 1
    except:
        raise ValueError("train_prop + test_prop + validate_prop must be equal to 1")
    # Validate stratify_col_name is in df_input
    try:
        assert stratify_col_name in df_input.columns
    except:
        try:
            assert use_last_col_as_target
        except:
            raise ValueError("stratify_col_name must be in df_input! Check spelling or case.\nOr set use_last_col_as_target=True")

    df_columns = df_input.columns
    if use_last_col_as_target:
        stratify_col_name = df_columns[-1]

    X = df_input  # contains all columns.
    y = df_input[
        [stratify_col_name]
    ]  # Dataframe of just the column on which to stratify.

    # Split original dataframe into train and temp DataFrames.
    df_train, df_temp, y_train, y_temp = train_test_split(
        X,
        y,
        stratify=y,
        test_size=(1 - train_prop),
        random_state=random_state,
        shuffle=True,
    )

    # Split the df_test_plus_validate DataFrame into validate and test DataFrames.
    df_validate, df_test, y_validate, y_test = train_test_split(
        df_temp,
        y_temp,
        stratify=y_temp,
        test_size=(test_prop / (test_prop + validate_prop)),
        random_state=random_state,
        shuffle=True,
    )

    # Validate no rows were lost.
    try:
        assert len(df_train) + len(df_validate) + len(df_test) == len(df_input)
    except:
        raise ValueError("Some rows were lost in the split.")
    # Validate no rows were duplicated.
    try:
        assert len(df_train) + len(df_validate) + len(df_test) == len(df_train) + len(df_temp)
    except:
        raise ValueError("Some rows were duplicated in the split.")

    # Check if to drop the stratify column
    if drop_stratify_col:
        df_train = df_train.drop(stratify_col_name, axis=1)
        df_validate = df_validate.drop(stratify_col_name, axis=1)
        df_test = df_test.drop(stratify_col_name, axis=1)

    # Return DataFrames
    if return_y:
        return df_train, df_validate, df_test, y_train, y_validate, y_test
    else:
        return df_train, df_validate, df_test


if __name__ == "__main__":
    df = pd.DataFrame(
        {
            "A": list(range(0, 1000)),
            "B": list(range(1000, 0, -1)),
            "label": ["foo"] * 750 + ["bar"] * 150 + ["baz"] * 100,
        }
    )

    df_train, df_validate, df_test, y_train, y_validate, y_test = split_into_train_validate_test(
        df,
        stratify_col_name="label",
        use_last_col_as_target=False,
        train_prop=0.74,
        test_prop=0.13,
        validate_prop=0.13,
        random_state=42,
        drop_stratify_col=False,
        return_y=True
    )
