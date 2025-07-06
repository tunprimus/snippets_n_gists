#!/usr/bin/env python3
import numpy as np

def random_numpy_sampler(arr, size=13):
    """
    Randomly sample a given numpy array.

    Parameters
    ----------
    arr : numpy.array
        The numpy array to sample.
    size : int, optional
        The size of the sample. Defaults to 13.

    Returns
    -------
    numpy.array
        A subset of the input array with the specified size.
    """
    import numpy as np

    if (np.shape(arr)[0] < size):
        size = (np.shape(arr)[0] // 3)[0]
    return arr[np.random.choice(len(arr), size=size, replace=False)]

