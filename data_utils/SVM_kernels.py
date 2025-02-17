#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles
from sklearn.svm import LinearSVC
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler

# Monkey patching NumPy for compatibility for versions >= 1.24
np.float = np.float64
np.int = np.int_
np.object = np.object_
np.bool = np.bool_

RANDOM_SEED = 42
GOLDEN_RATIO = 1.618
FIG_WIDTH = 30
FIG_HEIGHT = FIG_WIDTH / GOLDEN_RATIO
FIG_DPI = 72
NUM_SAMPLES = 1500
NOISE_FACTOR = 0.1
ALPHA_2D_PLOT = 0.3
ALPHA_3D_PLOT = 0.6
REG_PARAM = 1.0
MAX_ITER = 5000
SEP_OFFSET = 0.2
LIN_SPACE_SAMPLE_NUM = 50
ELEVATION_ANGLE = 20
AZIMUTH = 45
TITLE_2D = "2D Circle Dataset"
TITLE_3D = "3D Transformed Dataset with Linear Separator"
CLASS_0_COLOUR = "blue"
CLASS_1_COLOUR = "red"
CLASS_0_MARKER = "o"
CLASS_1_MARKER = "^"
CLASS_0_LABEL = "Class 0"
CLASS_1_LABEL = "Class 1"

# Generate Non-linear Data
def generate_circle_data(n_samples=NUM_SAMPLES, noise=NOISE_FACTOR, random_state=RANDOM_SEED):
    """
    Generates a 2D circle dataset with given number of samples, noise and random state using
    scikit-learn's make_circles.

    Parameters
    ----------
    n_samples : int
        The total number of points.
    noise : float
        Standard deviation of Gaussian noise added to the data.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    X : array of shape [n_samples, 2]
        The generated samples.
    y : array of shape [n_samples]
        The integer labels (0 or 1) for class membership of each sample.
    """
    return make_circles(n_samples=n_samples, noise=noise, random_state=random_state)


# Plot 2D Data
def plot_2d_data(X_data, y_data, title=TITLE_2D):
    """
    Plots the 2D data in a scatter plot with a given title.

    Parameters
    ----------
    X_data : array of shape [n_samples, 2]
        The feature data to be plotted.
    y_data : array of shape [n_samples]
        The target data to be plotted.
    title : str
        The title for the plot.
    """
    plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=FIG_DPI)
    plt.scatter(X_data[:, 0], X_data[:, 1], c=y_data, marker=".", cmap="viridis")
    plt.title(title)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.colorbar(label="Class")
    plt.grid(True, alpha=ALPHA_2D_PLOT)
    plt.show()


# Transform into a Higher-Dimension Space
def transform_to_3d(data_arr):
    """
    Transforms the given 2D data array into a 3D array, by using a modified
    transformation X3 = X1^2 + X2^2 to create better separability.

    Parameters
    ----------
    data_arr : array-like of shape [n_samples, 2]
        The 2D feature array to be transformed into a 3D array.

    Returns
    -------
    X_3d : array-like of shape [n_samples, 3]
        The transformed 3D array.
    """
    X1 = data_arr[:, 0].reshape(-1, 1)
    X2 = data_arr[:, 1].reshape(-1, 1)
    # Modified transformtion to create better separaton
    # X3 = X1**2 + X2**2
    X3 = np.square(X1) + np.square(X2)
    return np.hstack((X1, X2, X3))


# Plot the 3D Transformation
def plot_3d_transformation_with_separator(X_transformed_data, y_data, title=TITLE_3D):
    """
    Plots the 3D transformation of the 2D circle dataset with a given title.

    Parameters
    ----------
    X_transformed_data : array-like of shape [n_samples, 3]
        The 3D feature array to be plotted.
    y_data : array-like of shape [n_samples]
        The target data to be plotted.
    title : str
        The title for the plot.

    Notes
    -----
    The 3D transformation is done using a modified transformation X3 = X1^2 + X2^2
    to create better separability.
    """
    # Scale the transformed features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_transformed_data)

    # Fit linear SVM with adjusted parameters for better separation
    svm = LinearSVC(C=REG_PARAM, max_iter=MAX_ITER)
    svm.fit(X_scaled, y_data)

    # Create the 3D plot
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=FIG_DPI)
    ax = fig.add_subplot(111, projection="3d")

    # Plot the two classes with different colours and markers for clarity
    class_0 = y_data == 0
    class_1 = y_data == 1

    ax.scatter(X_transformed_data[class_0, 0], X_transformed_data[class_0, 1], X_transformed_data[class_0, 2], c=CLASS_0_COLOUR, marker=CLASS_0_MARKER, label=CLASS_0_LABEL, alpha=ALPHA_3D_PLOT)

    ax.scatter(X_transformed_data[class_1, 0], X_transformed_data[class_1, 1], X_transformed_data[class_1, 2], c=CLASS_1_COLOUR, marker=CLASS_1_MARKER, label=CLASS_1_LABEL, alpha=ALPHA_3D_PLOT)

    # Create a grid for the separator plane
    x_min, x_max = X_transformed_data[:, 0].min() - SEP_OFFSET, X_transformed_data[:, 0].max() + SEP_OFFSET
    y_min, y_max = X_transformed_data[:, 1].min() - SEP_OFFSET, X_transformed_data[:, 1].max() + SEP_OFFSET

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, num=LIN_SPACE_SAMPLE_NUM), np.linspace(y_min, y_max, num=LIN_SPACE_SAMPLE_NUM))

    # Get the separating plane coefficients
    sep_plane_coef = svm.coef_[0]
    sep_plane_intercept = svm.intercept_[0]

    # Calculate z coordinates of the plane
    grid_points = np.c_[xx.ravel(), yy.ravel(), np.zeros(xx.ravel().shape[0])]
    scaled_grid = scaler.transform(grid_points)

    # Calculate the separator plane
    z = ((-sep_plane_coef[0] * scaled_grid[:, 0]) - (sep_plane_coef[1] * scaled_grid[:, 1]) - sep_plane_intercept) / sep_plane_coef[2]
    z = z.reshape(xx.shape)
    z = scaler.inverse_transform(np.c_[xx.ravel(), yy.ravel(), z.ravel()])[:, 2].reshape(xx.shape)

    # Plot the separating plane with adjusted transparency
    surface = ax.plot_surface(xx, yy, z, alpha=ALPHA_2D_PLOT, cmap="coolwarm")

    # Customise the plot
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("X1^2 + X2^2")
    ax.set_title(title)

    # Add legend
    ax.legend()

    # Adjust the viewing angle for better visualisation
    ax.view_init(elev=ELEVATION_ANGLE, azim=AZIMUTH)

    # Add text descripion
    ax.text2D(0.05, 0.95, "Polynomial Kernel Transformation:\n(X1, X2) -> (X1, X2, X1^2 + X2^2)\n\nClasses are linearly separable\nin transformed space", transform=ax.transAxes, bbox=dict(facecolor="white", alpha=0.8))

    plt.show()


# The Main Function
def main():
    # Generate and plot the dataset
    X, y = generate_circle_data()
    # Transform and plot 3D data with clear separator
    X_transformed = transform_to_3d(X)
    plot_3d_transformation_with_separator(X_transformed, y)

if __name__ == "__main__":
    main()
