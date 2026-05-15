#Basic
import os, sys

# Typing
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Tuple, Optional, Sequence, List, Literal

# Basics
from enum import Enum

# Numpy
import numpy as np 

# Visualization
import matplotlib.pyplot as plt 

from .utils import analytic_single_dense_hessian, hessian_metrics_np

# ---------------------------
class DataType(Enum):
    """Enumeration of dataset input/output types for visualization and processing.

    Each enum value represents the structure and nature of the data:
    - IMAGE_CLASS: 2D image with class labels (e.g., segmentation map).
    - IMAGE_FLOAT: 2D image with continuous pixel values.
    - D1_CLASS: 1D feature input/output with classification labels.
    - D1_FLOAT: 1D feature input/output with continuous values (regression).
    - D2_CLASS: 2D feature space with classification (e.g., one-hot labels).
    - D2_FLOAT: 2D feature space with continuous targets.
    """
    IMAGE_CLASS = "image-class"
    IMAGE_FLOAT = "image-float"
    D1_CLASS = "1d-class"
    D1_FLOAT = "1d-float"
    D2_CLASS = "2d-class"
    D2_FLOAT = "2d-float"



@dataclass
class BaseDataset:
    """Base class for datasets used in ENABOL training and testing.

    Attributes
    ----------
    X : np.ndarray
        Input features of the dataset.
    Y : np.ndarray
        Target outputs corresponding to X.
    """

    X: np.ndarray = field(init=False)
    Y: np.ndarray = field(init=False)
    num_samples: int = 1000
    input_type: DataType = field(init=False)
    output_type: DataType = field(init=False)
    seed: Optional[int] = None

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """Returns the shape of the input features X."""
        return self.X.shape
    
    @property
    def output_shape(self) -> Tuple[int, ...]:
        """Returns the shape of the target outputs Y."""
        return self.Y.shape

    @property
    @abstractmethod
    def reference_weight_matrix(self) -> np.ndarray:
        """Returns the reference weight matrix for the dataset.

        This method should be implemented by subclasses to provide a reference
        weight matrix that can be used for comparison or validation.
        
        Returns
        -------
        np.ndarray
            The reference weight matrix.
        """
        pass

    @property
    @abstractmethod
    def reference_bias_vector(self) -> np.ndarray:
        """Returns the reference bias vector for the dataset.

        This method should be implemented by subclasses to provide a reference
        bias vector that can be used for comparison or validation.
        
        Returns
        -------
        np.ndarray
            The reference bias vector.
        """
        pass

    def get(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the dataset as a tuple of (X, Y).

        Returns
        -------
        tuple of np.ndarray
            The input features and target outputs.
        """
        return self.X, self.Y

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the dataset as NumPy arrays.

        Returns
        -------
        tuple of np.ndarray
            X and Y arrays.
        """
        return self.get()

    def to_txt(self, prefix: str = "dataset") -> None:
        """Saves the dataset to text files using NumPy's `savetxt`.

        Parameters
        ----------
        prefix : str, optional
            Prefix for the output filenames, by default "dataset".

        Notes
        -----
        Two files will be saved:
        - <prefix>_X.txt
        - <prefix>_Y.txt
        """
        np.savetxt(f"{prefix}_X.txt", self.X)
        np.savetxt(f"{prefix}_Y.txt", self.Y)
    
    def to_dat(self, prefix: str = "dataset") -> None:
        """Saves the dataset to binary files using NumPy's `save`.

        Parameters
        ----------
        prefix : str, optional
            Prefix for the output filenames, by default "dataset".

        Notes
        -----
        Two files will be saved:
        - <prefix>_X.dat
        - <prefix>_Y.dat
        """
        os.makedirs(prefix, exist_ok=True)

        np.savetxt(os.path.join(prefix, "tb_input_features.dat"), self.X.reshape(-1, np.prod(self.X.shape[1:])))
        np.savetxt(os.path.join(prefix, "tb_output_predictions.dat"), self.Y)

    def plot(self, max_points: int = 100) -> None:
        """Visualize the dataset based on input and output tags.

        Parameters
        ----------
        max_points : int, optional
            Max number of points to plot for large datasets.
        """

        X = self.X[:max_points]
        Y = self.Y[:max_points]

        if self.input_type == DataType.D1_FLOAT and self.output_type == DataType.D1_FLOAT:
            plt.scatter(X, Y, alpha=0.7)
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.title("1D Float Regression")
            plt.grid(True)
            plt.show()
        elif self.input_type == DataType.D1_FLOAT and self.output_type == DataType.D1_CLASS:
            # Histogram for classification
            Y = np.argmax(Y, axis=1)
            num_classes = Y.shape[1] if Y.ndim > 1 else np.max(Y) + 1
            plt.hist(Y, bins=np.arange(num_classes + 1) - 0.5, alpha=0.7, edgecolor='black')
            plt.xticks(np.arange(num_classes))
            plt.xlabel("Class")
            plt.ylabel("Frequency")
            plt.title("1D Classification Histogram")
            plt.grid(True)
            plt.show()
        elif self.input_type == DataType.IMAGE_FLOAT and self.output_type == DataType.D2_FLOAT:
            # Imshow of images
            num_images = min(9, self.X.shape[0])
            # Arrange in a grid
            nrows = int(np.ceil(np.sqrt(num_images)))
            ncols = int(np.ceil(num_images / nrows))

            fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 5*nrows))
            axs = axs.flatten()
            for i in range(num_images):
                axs[i].imshow(self.X[i], cmap='gray')
                axs[i].set_title(f"Label: {self.Y[i]}")
                axs[i].axis("off")
                axs[i].set_aspect('auto')
            # Delete unused axs
            for j in range(num_images, len(axs)):
                fig.delaxes(axs[j])
            plt.show()
        elif self.input_type == DataType.IMAGE_FLOAT and self.output_type == DataType.IMAGE_FLOAT:
            # This is a cAE basically.
            num_images = min(9, self.X.shape[0])
            # Arrange in a grid
            nrows = int(np.ceil(np.sqrt(num_images)))
            ncols = int(np.ceil(num_images / nrows))

            fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 5*nrows))
            axs = axs.flatten()
            for i in range(num_images):
                axs[i].imshow(self.X[i], cmap='gray')
                axs[i].axis("off")
                axs[i].set_aspect('auto')
            # Delete unused axs
            for j in range(num_images, len(axs)):
                fig.delaxes(axs[j])
            plt.show()
        elif self.input_type == DataType.IMAGE_FLOAT and self.output_type == DataType.D1_CLASS:
            # like mnist classification task 
            num_images = min(9, self.X.shape[0])
            # Arrange in a grid
            nrows = int(np.ceil(np.sqrt(num_images)))
            ncols = int(np.ceil(num_images / nrows))

            fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 5*nrows))
            axs = axs.flatten()
            for i in range(num_images):
                axs[i].imshow(self.X[i], cmap='gray')
                axs[i].set_title(f"Label: {self.Y[i]}")
                axs[i].axis("off")
                axs[i].set_aspect('auto')
            # Delete unused axs
            for j in range(num_images, len(axs)):
                fig.delaxes(axs[j])
            plt.show()

            # also plot distribution of classes
            num_classes = self.Y.shape[1] if self.Y.ndim > 1 else np.max(self.Y) + 1
            Y_classes = np.argmax(self.Y, axis=1)
            plt.figure()
            plt.hist(Y_classes, bins=np.arange(num_classes + 1) - 0.5, alpha=0.7, edgecolor='black')
            plt.xticks(np.arange(num_classes))
            plt.xlabel("Class")
            plt.ylabel("Frequency")
            plt.title("Classification Histogram")
            plt.grid(True)
            plt.show()
        elif self.input_type == DataType.D2_FLOAT and self.output_type == DataType.D2_FLOAT:
            # pairplot for each feature and target dimension
            num_features = self.X.shape[1]
            num_targets = self.Y.shape[1]
            fig, axs = plt.subplots(num_targets, num_features, figsize=(5*num_features, 5*num_targets), sharex=True, sharey=True)
            # Make sure axs is 2D
            if num_features == 1 and num_targets == 1:
                axs = np.array([[axs]])
            elif num_features == 1:
                axs = axs[np.newaxis, :]    # type: ignore
            elif num_targets == 1:
                axs = axs[:, np.newaxis]    # type: ignore
                
            hlegs = []
            hlabs = []
            for i in range(num_targets):
                for j in range(num_features):
                    axs[i, j].scatter(self.X[:, j], self.Y[:, i], alpha=0.7, color=f'C{i*num_features+j}', label=f"Y[{i}] vs X[{j}]")   # type: ignore
                    if i==num_targets-1: axs[i, j].set_xlabel(f"X[{j}]")    # type: ignore
                    if j==0: axs[i, j].set_ylabel(f"Y[{i}]")                # type: ignore
                    #if i==0: axs[i, j].set_title(f"Feature {j} vs Target {i}")
                    axs[i, j].grid(True)                                    # type: ignore
                    # Get legend handles and labels
                    handles, labels = axs[i, j].get_legend_handles_labels() # type: ignore
                    hlegs.extend(handles)
                    hlabs.extend(labels)
            # Add a single legend for the entire figure at the bottom spanning multiple columns
            fig.legend(hlegs, hlabs, loc='lower center', ncol=num_features, bbox_to_anchor=(0.5, -0.05))
            plt.tight_layout()

        else:
            print("Plotting not implemented for this data type combination.")
    

    def plot_histogram(self, bins: Optional[int] = None) -> None:
        """Plot histograms of the input features and target outputs."""
        
        if self.input_type in {DataType.D1_FLOAT, DataType.D2_FLOAT}:
            num_features = self.X.shape[1] if self.X.ndim > 1 else 1
            num_targets = self.Y.shape[1] if self.Y.ndim > 1 else 1

            num_cols = max(num_features, num_targets)

            fig, axs = plt.subplots(2, num_cols, figsize=(5*num_cols, 5*2), sharey=True)

            # Inputs X:
            for i in range(num_cols):
                if i >= num_features:
                    # If there are fewer features than columns, hide the extra subplots
                    ax = axs[0, i] if num_cols > 1 else axs[0]
                    ax.axis('off')
                    continue
                ax = axs[0, i] if num_cols > 1 else axs[0]
                ax.hist(self.X[:, i] if self.X.ndim > 1 else self.X, bins=bins, alpha=0.7, edgecolor='black', color=f'C{i}')
                ax.set_title(f"Histogram of X[{i}]")
                ax.set_xlabel(f"X[{i}] values")
                if i==0: ax.set_ylabel("Frequency")
                ax.grid(True)

            # Outputs Y:
            for i in range(num_cols):
                if i >= num_targets:
                    # If there are fewer targets than columns, hide the extra subplots
                    ax = axs[1, i] if num_cols > 1 else axs[1]
                    ax.axis('off')
                    continue
                ax = axs[1, i] if num_cols > 1 else axs[1]
                ax.hist(self.Y[:, i] if self.Y.ndim > 1 else self.Y, bins=bins, alpha=0.7, edgecolor='black', color=f'C{num_features+i}')
                ax.set_title(f"Histogram of Y[{i}]")
                ax.set_xlabel(f"Y[{i}] values")
                if i==0: ax.set_ylabel("Frequency")
                ax.grid(True)

        else:
            print("Histogram plotting is only implemented for float data types.")

    @abstractmethod
    def __repr__(self) -> str:
        """String representation of the dataset for easy visualization."""
        pass


@dataclass
class AffineDataset(BaseDataset):
    """Dataset for affine transformations of the form y = Ax + b.

    This dataset is designed to test the ability of ENABOL to learn linear
    transformations and biases. The reference weight matrix A and bias vector b
    are provided for validation.

    Attributes
    ----------
    A : np.ndarray
        The reference weight matrix for the affine transformation.
        Default is [[1.25, -0.75, 0.50, 0.20], [-0.40, 0.90, 1.10, -0.60]].
    b : np.ndarray
        The reference bias vector for the affine transformation.
            Default is [0.35, -0.25].
    """

    A: np.ndarray = field(default_factory=lambda: np.array([[1.25, -0.75, 0.50, 0.20], [-0.40, 0.90, 1.10, -0.60]]))
    b: np.ndarray = field(default_factory=lambda: np.array([0.35, -0.25]))
    use_bias: bool = True

    def __post_init__(self):
        # If A and B were not initialized, set the to default values
        if self.A is None:
            self.A = np.array([[1.25, -0.75, 0.50, 0.20], [-0.40, 0.90, 1.10, -0.60]])
        if self.b is None:
            self.b = np.array([0.35, -0.25])
        if not self.use_bias:
            self.b = np.zeros(self.A.shape[0])

        # Make sure that A and b have matching shapes for the affine transformation
        if self.A.shape[0] != self.b.shape[0]:
            raise ValueError("The number of rows in A must match the length of b.")
        
        # Generate data 
        rng = np.random.default_rng(self.seed)

        self.X = rng.uniform(0, 1, size=(self.num_samples, self.A.shape[1]))  # 1000 samples, input dimension matches A's columns
        self.Y = self.X @ self.A.T + self.b  # Apply the affine transformation

        self.input_type = DataType.D1_FLOAT if self.A.shape[1] == 1 else DataType.D2_FLOAT
        self.output_type = DataType.D1_FLOAT if self.A.shape[0] == 1 else DataType.D2_FLOAT

        # Init H_nom and lam_nom for the analytic Hessian and its spectral norm
        self.H_nom = None
        self.lam_nom = None

    @property
    def reference_weight_matrix(self) -> np.ndarray:
        """Returns the reference weight matrix A."""
        return self.A

    @property
    def reference_bias_vector(self) -> np.ndarray:
        """Returns the reference bias vector b."""
        return self.b
    
    @property 
    def analytic_hessian(self) -> dict[str, np.ndarray | float]:
        """Returns the analytic Hessian for the affine dataset."""
        
        if self.H_nom is None:
            H_nom = analytic_single_dense_hessian(
                self.X,
                d_out=self.Y.shape[1],
                keras_mse_scaling=False,
            )
            self.H_nom = H_nom
        
        if self.lam_nom is None:
            self.lam_nom = hessian_metrics_np(self.H_nom)["hessian_lambda_max"]
        
        self.eta_max_nom = 2.0 / self.lam_nom

        #print(f'Lambda nominal: {self.lam_nom}, eta_max nominal: {self.eta_max_nom}')
        return {"hessian": self.H_nom, "lambda_max": self.lam_nom, "eta_max": self.eta_max_nom}

    
    def __repr__(self) -> str:
        s  = f"AffineDataset(\n"
        s += f" [Input] X: \n"
        s += f"    Shape: {self.X.shape}\n"
        s += f"    Dtype: {self.input_type}\n"
        s += f"    X <- Uniform(-1, 1)\n"
        s += f" [Output] Y: \n"
        s += f"    Shape: {self.Y.shape}\n"
        s += f"    Dtype: {self.output_type}\n"
        s += f"    Y <- X @ A.T + b\n"
        s += f" ---------\n"
        s += f"  A = "
        sA = str(self.A).replace("\n", "\n      ")
        s += f"{sA}\n"
        s += f"  b = {str(self.b)}\n"
        s += f"----------\n"
        hessian = self.analytic_hessian
        s += f"Analytic Hessian:\n"
        s += f"  Lambda max: {hessian['lambda_max']:.4f}\n"
        s += f"  Eta max: {hessian['eta_max']:.4f}\n"
        s += ")"
        return s
    




