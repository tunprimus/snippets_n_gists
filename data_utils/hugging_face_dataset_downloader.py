#!/usr/bin/env python3
from datasets import load_dataset
from os.path import realpath as realpath

def get_hugging_face_dataset(target_name, path_to_save=None):
    dataset = f"{target_name}_dataset"
    dataset = load_dataset(target_name)
    if path_to_save is not None:
        real_path_to_save = realpath(path_to_save)

    for key, value in dataset.items():
        value.to_csv(real_path_to_save + f"/{key}.csv")
    print(f"Dataset {target_name} saved to ", real_path_to_save)
