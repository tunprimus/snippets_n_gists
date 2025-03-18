#!/usr/bin/env python3
# Adapted from aboood40091/dbscanpp-py -> https://github.com/aboood40091/dbscanpp-py
import enum
import numpy as np
import numpy.typing as npt
from math import ceil
from scipy.spatial import KDTree
from scipy.spatial.distance import sqeuclidean
from sklearn.cluster import KMeans
from typing import Optional

# Monkey patching NumPy >= 1.24 in order to successfully import model from sklearn and other libraries
np.float = np.float64
np.int = np.int_
np.object = np.object_
np.bool = np.bool_


class SamplingType(enum.Enum):
    Linspace = enum.auto()
    Uniform = enum.auto()
    KCentre = enum.auto()
    KMeansPP = enum.auto()


def k_centres(m: int, x: npt.NDArray) -> npt.NDArray[np.int_]:
    """
    Selects m points from x to serve as cluster centres by k-centres++ algorithm.

    Parameters
    ----------
    m : int
        Number of cluster centres to select.
    x : array-like of shape (n_samples, n_features)
        The data points from which to select the cluster centres.

    Returns
    -------
    result : array-like of shape (m,)
        The indices of the selected cluster centres in x.
    """
    result = np.empty(m, dtype=np.int_)

    # Initialise the first centre to index 0
    centre_id = 0
    result[0] = centre_id

    # Precompute squared norms of all data points
    norms_sq = np.einsum("ij,ij->i", x, x)

    # Compute squared distances from all data points to the first centre
    closest_dist_sq = norms_sq + norms_sq[centre_id] - 2 * np.dot(x, x[centre_id])
    for c in range(1, m):
        # Select the point that is farthest from its closest centre
        centre_id = np.argmax(closest_dist_sq)
        result[c] = centre_id

        # Compute squared distances from all data points to the new centre
        dist_sq_new_centre = (
            norms_sq + norms_sq[centre_id] - 2 * np.dot(x, x[centre_id])
        )

        # Update closest distances
        np.minimum(closest_dist_sq, dist_sq_new_centre, out=closest_dist_sq)
    return result


def k_means_pp(m: int, x: npt.NDArray) -> npt.NDArray[np.int_]:
    """
    Selects m points from x to serve as cluster centres by k-means++ algorithm.

    Parameters
    ----------
    m : int
        Number of cluster centres to select.
    x : array-like of shape (n_samples, n_features)
        The data points from which to select the cluster centres.

    Returns
    -------
    result : array-like of shape (m,)
        The indices of the selected cluster centres in x.
    """
    kmeans = KMeans(n_clusters=m, init="k-means++", random_state=42).fit(x)
    centres = kmeans.cluster_centers_
    f_compute_distance = (
        sqeuclidean  # TODO: Replace this with euclidean for better accuracy?
    )
    # f_compute_distance = lambda p, c: np.sqrt(np.sum((p-c)**2))
    return np.fromiter(
        (
            np.argmin([f_compute_distance(point, centre) for point in x])
            for centre in centres
        ),
        dtype=np.int_,
        count=len(centres),
    )


class DBSCANPP:
    eps: float
    min_samples: int

    def __init__(self, eps: float, min_samples: int) -> None:
        self.eps = eps
        self.min_samples = min_samples

    def fit_predict(
        self,
        x: npt.NDArray,
        *,
        ratio: Optional[float] = None,
        p: Optional[float] = None,
        sampling_type: SamplingType = SamplingType.Linspace
    ) -> npt.NDArray[np.int_]:
        """
        Performs clustering on the input data using the DBSCAN++ algorithm.

        Parameters
        ----------
        x : npt.NDArray
            The input data points, provided as an array of shape (n_samples, n_features).
        ratio : Optional[float], default=None
            The ratio of the total number of points to be sampled as potential cluster centres.
            If specified, overrides the parameter `p`.
        p : Optional[float], default=None
            Parameter used to determine the number of points to sample based on the input data
            dimensionality. Ignored if `ratio` is specified.
        sampling_type : SamplingType, default=SamplingType.Linspace
            The method used for sampling initial cluster centres. Options include Linspace, Uniform,
            KCentre, and KMeansPP.

        Returns
        -------
        npt.NDArray[np.int_]
            An array of shape (n_samples,) containing cluster labels for each input data point.
            Noise points are labeled as -1.
        """
        # Ensure x is a contiguous array of doubles
        x = np.ascontiguousarray(x, dtype=np.double)
        n, d = x.shape
        assert n > 0
        if ratio is not None:
            m = int(ceil(ratio * n))
        else:
            assert p is not None
            m = int(p * n ** (d / (d + 4)))
        # Clamp m between 1 and n
        m = min(max(m, 1), n)

        if m == n:
            subset_indices = np.arange(n, dtype=np.int_)
            is_in_subset = np.ones(n, dtype=np.bool_)
        else:
            if sampling_type == SamplingType.Uniform:
                subset_indices = np.random.choice(
                    np.arange(n, dtype=np.int_), size=m, replace=False
                )
            elif sampling_type == SamplingType.KCentre:
                subset_indices = k_centres(m, x)
            elif sampling_type == SamplingType.KMeansPP:
                subset_indices = k_means_pp(m, x)
            else:
                subset_indices = np.linspace(0, n - 1, m, dtype=np.int_)
            is_in_subset = np.zeros(n, dtype=np.bool_)
            is_in_subset[subset_indices] = True

        CONST_UNEXPLORED = -2
        CONST_NOISE = -1
        CONST_CLUSTER_START = 0

        eps = self.eps
        min_samples = self.min_samples

        # Initialise labels
        labels = np.full(n, CONST_UNEXPLORED, dtype=np.int_)
        cluster = CONST_CLUSTER_START

        # Create a KD-tree for efficient neighbour lookup within radius `eps`
        kdtree = KDTree(x)
        for i in subset_indices:
            if labels[i] != CONST_UNEXPLORED:
                continue

            # Get neighbours within radius `eps` using KD-tree
            neighbours: list[int] = kdtree.query_ball_point(x[i], eps)
            if len(neighbours) >= min_samples:
                # Start a new cluster
                labels[i] = cluster
                # Initialise a position index
                pos = 0

                while pos < len(neighbours):
                    neighbour_i = neighbours[pos]
                    pos += 1
                    if labels[neighbour_i] == CONST_NOISE:
                        labels[neighbour_i] = cluster
                    elif labels[neighbour_i] == CONST_UNEXPLORED:
                        labels[neighbour_i] = cluster

                        # Expand neighbours only if it is a core point
                        if is_in_subset[neighbour_i]:
                            extended_neighbours = kdtree.query_ball_point(
                                x[neighbour_i], eps
                            )
                            if len(extended_neighbours) >= min_samples:
                                neighbours.extend(extended_neighbours)

                # Increment cluster label after finishing this cluster
                cluster += 1
            else:
                labels[i] = CONST_NOISE

        # Convert all remaining CONST_UNEXPLORED data points to CONST_NOISE
        labels[labels == CONST_UNEXPLORED] = CONST_NOISE

        # Return labels
        return labels
