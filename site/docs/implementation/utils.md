---
sidebar_label: "🧰 utils.py"
status:
  - valid
  - inprogress
tags:
  - implementation
  - utils
  - stability
last_modified: 2026-05-16
source: "enabol/utils.py"
source_url: "https://github.com/manuelblancovalentin/kappa_budgeting/blob/master/enabol/utils.py"
---
# 🧰 Utils Module Reference
<PageMeta />
---

<TBox type="summary" title="What this page covers">
This page is a coder-facing reference for `enabol/utils.py`: tensor flattening, stable norms, update-direction metrics, matrix norms, analytic Hessian helpers, local stability metrics, and the clean half-MSE loss used by the first ablations.
</TBox>

## Module Map

| Group | Functions | Used by |
|---|---|---|
| Tensor geometry | `flatten_tensors`, `tensor_l2_norm`, `safe_cosine` | `BaseModel.train_instrumented(...)` |
| Matrix diagnostics | `spectral_norm_np`, `matrix_norms_np` | gain/norm logging and future κ analysis |
| Hessian diagnostics | `analytic_single_dense_hessian`, `hessian_metrics_np`, `stability_metrics_from_hessian` | EXP-000A / one-layer sanity tests |
| Loss helpers | `half_mse_batch_loss` | clean Hessian experiments |

<TBox type="todo" title="Utils TODOs">

- [ ] Keep pure NumPy diagnostics and TensorFlow training helpers clearly separated as this file grows.
- [ ] Move plotting or experiment-specific metrics out of `utils.py`; this file should stay reusable.
- [ ] Add tests for edge cases such as empty tensor lists, non-finite matrices, and singular Hessians.

</TBox>

## Imports And Dependencies

```python
import numpy as np
import tensorflow as tf
from typing import Sequence, Optional, Dict, Any
```

`utils.py` mixes TensorFlow helpers and NumPy diagnostics. That is acceptable for the current small harness, but if the module grows, split candidates are:

| Future file | Contents |
|---|---|
| `tensor_utils.py` | TensorFlow flattening, norms, cosine helpers. |
| `linear_diagnostics.py` | Matrix norms, Hessian helpers, stability metrics. |
| `losses.py` | Small custom losses used by ablations. |

## Tensor Geometry Functions

#### `flatten_tensors(tensors)`

```python
def flatten_tensors(tensors: Sequence[tf.Tensor]) -> tf.Tensor:
    ...
```

Flattens a sequence of tensors and concatenates them into one global vector. `None` entries are skipped.

This is central to the closed-loop controller because the controller treats all trainable parameters as one vector:

```math
\theta = \operatorname{flatten}(W_1, b_1, W_2, b_2, \ldots)
```

and all gradients as one vector:

```math
G = \operatorname{flatten}\left(\frac{\partial L}{\partial W_1}, \frac{\partial L}{\partial b_1}, \ldots\right).
```

Usage:

```python
theta = flatten_tensors(model.trainable_variables)
grad_vec = flatten_tensors(grads)
```

If all inputs are `None` or the list is empty, it returns an empty `tf.float32` tensor.

#### `tensor_l2_norm(x, eps=1e-12)`

```python
def tensor_l2_norm(x: tf.Tensor, eps: float = 1e-12) -> tf.Tensor:
    return tf.sqrt(tf.reduce_sum(tf.square(x)) + eps)
```

Computes a stable L2 norm. The small `eps` prevents exact zero inside the square root.

Usage:

```python
grad_norm = tensor_l2_norm(grad_vec)
update_norm = tensor_l2_norm(delta_actual)
```

#### `safe_cosine(a, b, eps=1e-12)`

```python
def safe_cosine(a: tf.Tensor, b: tf.Tensor, eps: float = 1e-12) -> tf.Tensor:
    denom = tensor_l2_norm(a, eps) * tensor_l2_norm(b, eps) + eps
    return tf.reduce_sum(a * b) / denom
```

Computes cosine similarity between two flattened vectors.

Current use:

```python
cos_actual = safe_cosine(delta_actual, -grad_vec)
```

Interpretation:

| Value | Meaning |
|---:|---|
| near `1` | actual update follows the intended descent direction. |
| near `0` | actual update is mostly orthogonal to the intended descent direction. |
| below `0` | actual update is partly anti-descent. |

<TBox type="warning" title="Cosine is a diagnostic, not necessarily a hardware controller signal">
The update cosine is useful in software because we can compare the raw intended update against the post-quantization applied update. In real hardware, the same register quantization may make that comparison unavailable or expensive. Treat it as an analysis metric unless a hardware-feasible estimator is explicitly designed.
</TBox>

## Matrix Diagnostic Functions

#### `spectral_norm_np(W)`

```python
def spectral_norm_np(W: np.ndarray) -> float:
    ...
```

Computes the matrix spectral norm using SVD. If the input contains non-finite values or SVD fails, it returns `np.nan` rather than crashing the experiment.

Behavior:

| Input | Output |
|---|---|
| finite 2D matrix | largest singular value. |
| finite non-2D array | vector L2 norm after flattening. |
| non-finite array | `np.nan`. |
| failed SVD | `np.nan`. |

#### `matrix_norms_np(W)`

```python
def matrix_norms_np(W: np.ndarray) -> dict[str, float]:
    ...
```

Returns common matrix norms in one dictionary:

```python
{
    "fro": ...,
    "l1": ...,
    "linf": ...,
    "spectral": ...,
}
```

These are the natural diagnostics for future κ/gain studies:

```math
\lVert W_l \rVert_F,
\qquad
\lVert W_l \rVert_1,
\qquad
\lVert W_l \rVert_\infty,
\qquad
\lVert W_l \rVert_2.
```

If any value in `W` is non-finite, every returned norm is `np.nan`.

## Hessian And Stability Functions

#### `analytic_single_dense_hessian(...)`

```python
def analytic_single_dense_hessian(
    X_batch: np.ndarray,
    d_out: int,
    keras_mse_scaling: bool = True,
) -> np.ndarray:
    ...
```

Analytic Hessian for one Dense layer without bias:

```math
\hat{Y} = X W^T.
```

The input covariance is:

```math
\Sigma_X = \frac{X^T X}{N}.
```

For the clean half-MSE loss:

```math
L = \frac{1}{2N}\lVert \hat{Y} - Y \rVert_F^2,
```

this gives:

```math
H = I_{d_{out}} \otimes \Sigma_X.
```

If `keras_mse_scaling=True`, the helper uses the scaling associated with Keras-style mean squared error:

```math
H = \frac{2}{d_{out}} I_{d_{out}} \otimes \Sigma_X.
```

Usage:

```python
H = analytic_single_dense_hessian(X_batch, d_out=2, keras_mse_scaling=False)
metrics = hessian_metrics_np(H)
```

#### `hessian_metrics_np(H)`

```python
def hessian_metrics_np(H: np.ndarray) -> Dict[str, float]:
    ...
```

Computes core Hessian metrics after symmetrizing defensively:

```python
{
    "hessian_lambda_max": lam_max,
    "hessian_lambda_min": lam_min,
    "hessian_spectral_norm": h_norm_2,
}
```

These are used to compare the cheap online curvature proxy against true local curvature in toy experiments.

#### `stability_metrics_from_hessian(H, eta, alpha=1.0)`

```python
def stability_metrics_from_hessian(
    H: np.ndarray,
    eta: float,
    alpha: float = 1.0,
) -> Dict[str, float]:
    ...
```

Computes local discrete-time stability metrics for the SGD-like update:

```math
\theta_{t+1} = \theta_t - \alpha \eta \nabla L(\theta_t).
```

The local linearized update map is:

```math
M = I - \alpha \eta H.
```

The function returns:

```python
{
    "stability_margin_lambda": alpha * eta * lambda_max,
    "stability_margin_norm": alpha * eta * hessian_spectral_norm,
    "spectral_radius_update_map": rho(I - alpha * eta * H),
}
```

For a convex quadratic, the familiar gradient-descent stability condition is approximately:

```math
0 < \alpha\eta\lambda_i(H) < 2.
```

So the scalar margin:

```math
\alpha\eta\lambda_{max}(H)
```

should remain below `2`.

## Loss Helpers

#### `half_mse_batch_loss(y_true, y_pred)`

```python
def half_mse_batch_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    ...
```

Computes:

```math
L = \frac{1}{2N}\lVert \hat{Y} - Y \rVert_F^2.
```

This loss is intentionally used in the first ablations because it makes the one-layer Hessian formula clean:

```math
H = I_{d_{out}} \otimes \Sigma_X.
```

Usage:

```python
with tf.GradientTape() as tape:
    y_pred = model(x_batch, training=True)
    loss = half_mse_batch_loss(y_batch, y_pred)
```

## Extension Notes

Keep this module boring and reusable. If a helper starts depending on a specific experiment, plot layout, or documentation artifact, it probably belongs in a notebook, `history.py`, or a future experiment-specific module instead of `utils.py`.
