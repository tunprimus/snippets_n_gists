#!/usr/bin/env python3
import pandas as pd
from scipy.io import arff
from os.path import realpath as realpath


def load_arff(filepath):
    real_path_to_file = realpath(filepath)
    arff_file = arff.loadarff(real_path_to_file)
    df, meta = pd.DataFrame(arff_file)
    print(meta)
    print(df.head())
    return df
