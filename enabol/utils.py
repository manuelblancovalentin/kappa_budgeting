import numpy as np
import tensorflow as tf
from typing import Sequence, Optional, Dict, Any


def flatten_tensors(tensors: Sequence[tf.Tensor]) -> tf.Tensor:
    """Flatten and concatenate a list of tensors.

    Parameters
    ----------
    tensors : sequence of tf.Tensor
        Tensors to flatten.

    Returns
    -------
    tf.Tensor
        One-dimensional concatenated tensor.
    """
    flats = []
    for t in tensors:
        if t is None:
            continue
        flats.append(tf.reshape(tf.cast(t, tf.float32), [-1]))

    if not flats:
        return tf.constant([], dtype=tf.float32)

    return tf.concat(flats, axis=0)


def tensor_l2_norm(x: tf.Tensor, eps: float = 1e-12) -> tf.Tensor:
    """Compute stable L2 norm."""
    return tf.sqrt(tf.reduce_sum(tf.square(x)) + eps)


def safe_cosine(a: tf.Tensor, b: tf.Tensor, eps: float = 1e-12) -> tf.Tensor:
    """Cosine similarity between two flattened vectors."""
    denom = tensor_l2_norm(a, eps) * tensor_l2_norm(b, eps) + eps
    return tf.reduce_sum(a * b) / denom


def spectral_norm_np(W: np.ndarray) -> float:
    """Compute matrix spectral norm safely.

    Returns NaN if the matrix contains non-finite values or if SVD fails.
    """
    W = np.asarray(W)

    if not np.all(np.isfinite(W)):
        return np.nan

    if W.ndim != 2:
        return float(np.linalg.norm(W.reshape(-1), ord=2))

    try:
        return float(np.linalg.svd(W, compute_uv=False)[0])
    except np.linalg.LinAlgError:
        return np.nan

def matrix_norms_np(W: np.ndarray) -> dict[str, float]:
    """Common matrix norms for logging, safely."""
    W = np.asarray(W)

    if not np.all(np.isfinite(W)):
        return {
            "fro": np.nan,
            "l1": np.nan,
            "linf": np.nan,
            "spectral": np.nan,
        }

    return {
        "fro": float(np.linalg.norm(W, ord="fro")),
        "l1": float(np.linalg.norm(W, ord=1)),
        "linf": float(np.linalg.norm(W, ord=np.inf)),
        "spectral": spectral_norm_np(W),
    }

def analytic_single_dense_hessian(
    X_batch: np.ndarray,
    d_out: int,
    keras_mse_scaling: bool = True,
) -> np.ndarray:
    """Analytic Hessian for one Dense layer without bias.

    Model
    -----
    Yhat = X @ W.T

    Parameters
    ----------
    X_batch : np.ndarray
        Batch input of shape (N, d_in).
    d_out : int
        Output dimension.
    keras_mse_scaling : bool
        If True, assumes tf.keras.losses.MSE-like scaling:
        mean over batch and output dimensions.
        If False, assumes L = (1 / (2N)) ||Yhat - Y||_F^2.

    Returns
    -------
    np.ndarray
        Hessian with respect to vec(W), shape
        (d_out * d_in, d_out * d_in).
    """
    X_batch = np.asarray(X_batch, dtype=np.float64)
    N, d_in = X_batch.shape

    sigma_x = (X_batch.T @ X_batch) / N

    if keras_mse_scaling:
        scale = 2.0 / d_out
    else:
        scale = 1.0

    H = scale * np.kron(np.eye(d_out), sigma_x)
    return H


def hessian_metrics_np(H: np.ndarray) -> Dict[str, float]:
    """Compute Hessian eigen/spectral-radius metrics."""
    H = np.asarray(H, dtype=np.float64)

    # Symmetrize defensively.
    Hs = 0.5 * (H + H.T)

    eigvals = np.linalg.eigvalsh(Hs)
    lam_max = float(np.max(eigvals))
    lam_min = float(np.min(eigvals))
    h_norm_2 = float(np.max(np.abs(eigvals)))

    return {
        "hessian_lambda_max": lam_max,
        "hessian_lambda_min": lam_min,
        "hessian_spectral_norm": h_norm_2,
    }


def stability_metrics_from_hessian(
    H: np.ndarray,
    eta: float,
    alpha: float = 1.0,
) -> Dict[str, float]:
    """Compute local discrete-time stability metrics.

    For SGD:
        theta_{t+1} = theta_t - alpha * eta * grad

    Local linearized map:
        I - alpha * eta * H
    """
    H = np.asarray(H, dtype=np.float64)
    n = H.shape[0]
    M = np.eye(n) - alpha * eta * H

    eig_M = np.linalg.eigvals(M)
    rho = float(np.max(np.abs(eig_M)))

    Hm = hessian_metrics_np(H)
    lam_max = Hm["hessian_lambda_max"]
    h_norm = Hm["hessian_spectral_norm"]

    return {
        "stability_margin_lambda": float(alpha * eta * lam_max),
        "stability_margin_norm": float(alpha * eta * h_norm),
        "spectral_radius_update_map": rho,
    }

def half_mse_batch_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Loss L = (1 / (2N)) ||Yhat - Y||_F^2.

    This keeps the Hessian formula clean:
        H = I_dout kron Sigma_X
    """
    batch_size = tf.cast(tf.shape(y_true)[0], tf.float32)
    residual = y_pred - y_true
    return 0.5 * tf.reduce_sum(tf.square(residual)) / batch_size