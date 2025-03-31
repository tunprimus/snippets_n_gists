#!/usr/bin/env python3
import pandas as pd


def split_into_six_train_validate_test(
    df_X,
    df_y=None,
    target_col_name="target",
    train_prop=0.74,
    test_prop=0.13,
    validate_prop=0.13,
    random_state=42,
    drop_target_col_name=False,
):
    """
    Split a Pandas dataframe into three pairs of DataFrames: train, validate and test.
    Adapted from: https://stackoverflow.com/a/60804119

    Parameters
    ----------
    df_X : Pandas DataFrame
        The input dataframe to be split.
    df_y : Pandas DataFrame, optional
        The target dataframe to use also for split.
    target_col_name : str, optional
        The name of a column in the dataframe to use for stratified splitting. By default "target".
    train_prop : float, optional
        The proportion of the input dataframe to use for the training set. By default 0.74.
    test_prop : float, optional
        The proportion of the input dataframe to use for the test set. By default 0.13.
    validate_prop : float, optional
        The proportion of the input dataframe to use for the validation set. By default 0.13.
    random_state : int, optional
        The seed used by the random number generator. By default 42
    drop_target_col_name : bool, optional
        To drop target_col_name from rest of DataFrame. By default False.

    Returns
    -------
    X_train : Pandas DataFrame
        The subset of the input dataframe to use for training.
    X_validate : Pandas DataFrame
        The subset of the input dataframe to use for validation.
    X_test : Pandas DataFrame
        The subset of the input dataframe to use for testing.
    y_train : Pandas DataFrame
        The subset of the input dataframe to use for training.
    y_validate : Pandas DataFrame
        The subset of the input dataframe to use for validation.
    y_test : Pandas DataFrame
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

    >>> X_train, X_validate, X_test, y_train, y_validate, y_test = split_into_six_train_validate_test(df_input, target_col_name="y", train_prop=0.74, test_prop=0.13, validate_prop=0.13, random_state=42)

    >>> X_train.shape
    (740, 3)

    >>> X_validate.shape
    (130, 3)

    >>> X_test.shape
    (130, 3)

    >>> X_train.label.value_counts()
    foo    555
    bar    111
    baz     74
    Name: label, dtype: int64

    >>> X_validate.label.value_counts()
    foo    98
    bar    19
    baz    13
    Name: label, dtype: int64

    >>> X_test.label.value_counts()
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

    X = df_X  # can contain all columns.
    if (df_y is None) or df_y.empty:
        try:
            assert target_col_name in df_X.columns
        except:
            raise ValueError("target_col_name must be in df_X.columns")
        else:
            y = df_X[[target_col_name]] # Dataframe of just the target_col_name column.
            if drop_target_col_name:
                X.drop([target_col_name], axis=1, inplace=True)
    else:
        y = df_y

    # Split original dataframe into train and temp DataFrames.
    X_train, df_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=(1 - train_prop),
        random_state=random_state,
        shuffle=True,
        stratify=y,
    )

    # Split the X_test_plus_validate DataFrame into validate and test DataFrames.
    X_validate, X_test, y_validate, y_test = train_test_split(
        df_temp,
        y_temp,
        test_size=(test_prop / (test_prop + validate_prop)),
        random_state=random_state,
        shuffle=True,
        stratify=y_temp,
    )

    # Validate no rows were lost.
    try:
        assert len(X_train) + len(X_validate) + len(X_test) == len(df_X)
    except:
        raise ValueError("Some rows were lost in the split.")
    # Validate no rows were duplicated.
    try:
        assert len(X_train) + len(X_validate) + len(X_test) == len(X_train) + len(X_validate) + len(X_test)
    except:
        raise ValueError("Some rows were duplicated in the split.")

    # Return DataFrames
    return X_train, X_validate, X_test, y_train, y_validate, y_test


if __name__ == "__main__":
    df = pd.DataFrame(
        {
            "A": list(range(0, 1000)),
            "B": list(range(1000, 0, -1)),
            "label": ["foo"] * 750 + ["bar"] * 150 + ["baz"] * 100,
        }
    )

    X_train, X_validate, X_test, y_train, y_validate, y_test = split_into_six_train_validate_test(
        df,
        target_col_name="label",
        train_prop=0.74,
        test_prop=0.13,
        validate_prop=0.13,
        random_state=42,
        drop_target_col_name=True,
    )
    print(X_train, X_validate, X_test)
    print(y_train, y_validate, y_test)

