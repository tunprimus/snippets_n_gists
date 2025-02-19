#!/usr/bin/env python3
import pandas as pd
import numpy as np
from os.path import realpath as realpath
np.float = np.float64
np.int = np.int_
np.object = np.object_
np.bool = np.bool_

pd.set_option("mode.copy_on_write", True)

def download_sklearn_dataset_to_csv(dataset_name, path_to_save):
    """
    Downloads the given sklearn dataset and saves it to a csv file.

    Args:
        dataset_name (str): The name of the dataset to download.
        path_to_save (str): The path where the csv file should be saved.

    Returns:
        None

    Raises:
        ValueError: If the dataset_name is not recognised.
    """
    if path_to_save:
        real_path_to_save = realpath(path_to_save)
    else:
        real_path_to_save = realpath(".")
    from sklearn import datasets
    if dataset_name.lower() == "california housing":
        data = datasets.california_housing.fetch_california_housing()
        calf_hous_df = pd.DataFrame(data= data.data, columns=data.feature_names)
        calf_hous_df.to_csv(real_path_to_save, index=False)
    elif dataset_name.lower() == "diabetes":
        data = datasets.load_diabetes()
        diabetes_df = pd.DataFrame(data= data.data, columns=data.feature_names)
        diabetes_df.to_csv(real_path_to_save, index=False)
    elif dataset_name.lower() == "digits":
        data = datasets.load_digits()
        digits_df = pd.DataFrame(data= data.data, columns=data.feature_names)
        digits_df.to_csv(real_path_to_save, index=False)
    elif dataset_name.lower() == "iris":
        data = datasets.load_iris()
        iris_df = pd.DataFrame(data= data.data, columns=data.feature_names)
        iris_df.to_csv(real_path_to_save, index=False)
    elif dataset_name.lower() == "wine":
        data = datasets.load_wine()
        wine_df = pd.DataFrame(data= data.data, columns=data.feature_names)
        wine_df.to_csv(real_path_to_save, index=False)
    else:
        print("Dataset not found")
