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

def pipeline_standardise_model_data(path_to_csv, list_of_columns=None, target_column=None):
    real_path_to_csv = realpath(path_to_csv)
    df = read_csv(real_path_to_csv, names=list_of_columns)
    arr_to_use = df.values
    # X, y = arr_to_use[:, :-1], arr_to_use[:, -1]
    mask = arr_to_use.loc[target_column]
    X = arr_to_use[~mask]
    y = arr_to_use[mask]
    estimators = []
    estimators.append(("standardise", StandardScaler()))
    estimators.append(("lda", LinearDiscriminantAnalysis()))
    pipeline = Pipeline(estimators)
    kfold = KFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
    results = cross_val_score(pipeline, X, y, cv=kfold)
    return results
