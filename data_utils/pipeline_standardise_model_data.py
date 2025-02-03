#!/usr/bin/env python3
from pandas import read_csv
import numpy as np
from os.path import realpath as realpath

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

np.float = np.float64
np.int = np.int_
np.object = np.object_
np.bool = np.bool_

# pd.set_option("mode.copy_on_write", True)

RANDOM_SEED = 42
NUM_KFOLD_SPLITS = 13

def pipeline_standardise_model_data(path_to_csv, list_of_columns=None, target_column=None):
    """
    Function to load a CSV file, split the data into X and y, create a pipeline
    to standardise the data, perform LDA and evaluate the pipeline using
    cross-validation.

    Parameters
    ----------
    path_to_csv : str, required
        Path to the CSV file to load.

    list_of_columns : list of str, optional
        List of column names to use. If None, uses the first row of the CSV
        file as the column names.

    target_column : str, optional
        Name of the column to use as the target variable. If None, uses the
        last column of the CSV file as the target variable.

    Returns
    -------
    results : numpy array
        Evaluation results from cross-validation of the pipeline.

    """
    #Load data
    real_path_to_csv = realpath(path_to_csv)
    df = read_csv(real_path_to_csv, names=list_of_columns)
    arr_to_use = df.values
    if target_column:
        mask = arr_to_use.loc[target_column]
        X = arr_to_use[~mask]
        y = arr_to_use[mask]
    else:
        X, y = arr_to_use[:, :-1], arr_to_use[:, -1]
    #Create pipeline
    estimators = []
    estimators.append(("standardise", StandardScaler()))
    estimators.append(("lda", LinearDiscriminantAnalysis()))
    pipeline = Pipeline(estimators)
    #Evaluate pipeline
    kfold = KFold(n_splits=NUM_KFOLD_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    results = cross_val_score(pipeline, X, y, cv=kfold)
    print(results.mean())
    return results
