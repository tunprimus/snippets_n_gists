#!/usr/bin/env python3
import numpy as np

def random_numpy_sampler(arr, size=13):
    if (np.shape(arr)[0] < size):
        size = (np.shape(arr)[0] // 3)[0]
    return arr[np.random.choice(len(arr), size=size, replace=False)]

