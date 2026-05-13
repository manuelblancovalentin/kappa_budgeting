# Models / NN Module Reference

Source: [`kappa/nn.py`](../../../kappa/nn.py)

This page documents the model and training-loop code block by block. It is written for people extending the ablation harness.

## Imports

```python
from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Union, Optional, Tuple

import numpy as np
import tensorflow as tf
```

Local dependencies:

```python
from .dataset import BaseDataset
from .history import FitHistory
from .precision import PrecisionDict, ensure_precision_dict
from .quantization import quantize_tensor, rail_stats
from .utils import ...
```

Important utilities:

| Utility | Purpose |
|---|---|
| `flatten_tensors` | Convert trainable variables or gradients into one global vector. |
| `half_mse_batch_loss` | Loss used by analytic Hessian experiments. |
| `tensor_l2_norm` | Stable tensor norm for logging. |
| `safe_cosine` | Update direction diagnostic. |
| `analytic_single_dense_hessian` | Exact Hessian for one-layer no-bias regression. |
| `stability_metrics_from_hessian` | Computes `eta * lambda_max` and update-map spectral radius. |

## `BaseModel`

```python
@dataclass(eq=False)
class BaseModel(ABC):
    dataset: BaseDataset
    loss: Union[str, tf.keras.losses.Loss] = "mse"
    optimizer: Union[str, tf.keras.optimizers.Optimizer] = "sgd"
    metrics: list = field(default_factory=lambda: [])
    model: Optional[tf.keras.Model] = field(init=False, default=None)
    input_shape: Optional[Tuple[int, ...]] = field(init=False, default=None)
    output_shape: Optional[Tuple[int, ...]] = field(init=False, default=None)
    name: str = "BaseModel"
    verbose: bool = False
    seed: Optional[int] = None
```

`BaseModel` owns the dataset reference, the Keras model object, and the custom training loop.

Subclasses are responsible for constructing `self.model`.

## `__post_init__()`

```python
def __post_init__(self):
    self.input_shape = self.dataset.input_shape[1:]
    self.output_shape = self.dataset.output_shape[1:]
```

Removes the batch dimension from dataset shapes.

Example:

```python
dataset.X.shape == (1000, 4)
model.input_shape == (4,)
```

## `_compile(...)`

```python
def _compile(self, optimizer=None, loss=None, metrics=None, **kwargs):
    ...
```

Thin wrapper over `self.model.compile(...)`.

Used by the simple `train()` method. The instrumented ablation loop does not use Keras optimizer application because it needs direct control over every update.

## `summary()`

```python
def summary(self) -> None:
    ...
```

Prints Keras model summary if the model exists.

Usage:

```python
model.summary()
```

## `reinitialize_weights()`

```python
def reinitialize_weights(self):
    ...
```

Loops over Keras layers and reassigns kernels/biases using the layer initializers.

Usage:

```python
model.reinitialize_weights()
```

Used before comparing baseline and controller runs so they start from comparable initial conditions.

## `train(...)`

```python
def train(self, X, Y, epochs=10, batch_size=32) -> dict[str, np.ndarray]:
    ...
```

Simple Keras-style training loop:

1. Builds a `tf.data.Dataset`.
2. Calls `_compile()`.
3. Uses the compiled optimizer and loss.
4. Applies gradients with `optimizer.apply_gradients`.
5. Returns only loss history.

This is mostly a convenience method. Ablation work should use `train_instrumented()`.

## `train_instrumented(...)`

```python
def train_instrumented(
    self,
    X,
    Y,
    epochs=10,
    batch_size=32,
    learning_rate=0.05,
    shuffle=True,
    loss_mode="half_mse",
    curvature_ema_rho=0.05,
    chi=1.5,
    eps=1e-12,
    use_controller=False,
    compute_analytic_hessian=True,
    reference_A=None,
    precision_dict=None,
) -> FitHistory:
    ...
```

This is the main ablation loop.

### Inputs

| Argument | Purpose |
|---|---|
| `X`, `Y` | NumPy training arrays. |
| `learning_rate` | Base SGD learning rate `eta`. |
| `loss_mode` | `"half_mse"` for Hessian-clean experiments or `"keras_mse"`. |
| `curvature_ema_rho` | EMA smoothing factor for curvature proxy. |
| `chi` | Target stability margin for the throttle. |
| `use_controller` | If true, applies `alpha_t`. If false, only logs would-be `alpha_t`. |
| `reference_A` | Teacher matrix for one-layer weight error. |
| `precision_dict` | Optional `PrecisionDict`; `None` means full floating point. |

### Setup Block

The method:

1. Converts `precision_dict` using `ensure_precision_dict`.
2. Validates precision names against the Keras model.
3. Converts `X` and `Y` to `np.float32`.
4. Selects the loss function.
5. Creates a batched `tf.data.Dataset`.
6. Allocates the `history` dictionary.

Backward compatibility rule:

```python
precision_dict=None
```

must preserve the original Experiment 000 behavior.

### Main Step Block

At each batch:

```text
theta_before = flatten(model.trainable_variables)

with GradientTape:
    y_pred = forward(...)
    loss = loss_fn(y_batch, y_pred)

grads = tape.gradient(loss, trainable_vars)
grad_flat = flatten(grads)
```

If `precision_dict` is present, the forward path uses `_forward_with_precision()` and gradients are passed through `_quantize_gradients()`.

### Curvature Proxy Block

```python
dG = grad_flat - prev_grad
dtheta = theta_before - prev_theta
C = ||dG|| / (||dtheta|| + eps)
```

This estimates directional update-field sensitivity.

EMA:

```python
curvature_ema = (1-rho) * curvature_ema + rho * curvature_proxy
```

Control signal:

```python
curvature_for_control = max(curvature_proxy, curvature_ema)
```

### Controller Block

```python
alpha_would = min(1.0, chi / (eta * (curvature_for_control + eps)))
alpha = alpha_would if use_controller else 1.0
eta_eff = alpha * eta
```

If `use_controller=False`, the loop still logs `alpha_would` so we can see whether the controller would have intervened.

### Update Block

Floating-point path:

```python
var.assign_sub(eta_eff * grad)
```

Quantized path:

```python
delta = -eta_eff * grad
delta = quantize_tensor(delta, update_dtype, ste=False)
var.assign_add(delta)
self._quantize_variable_storage(var, precision)
```

This is where update quantization and stored-weight quantization happen.

### Metric Block

The loop logs:

```text
loss
rmse
theta_norm
grad_norm
raw_update_norm
actual_update_norm
update_cosine
update_angle_rad
update_radius_ratio
curvature_proxy
curvature_ema
alpha
alpha_would
eta_eff
forward_gain_spectral
hessian_lambda_max
stability_margin_lambda_raw
stability_margin_lambda_ctrl
spectral_radius_raw
spectral_radius_ctrl
weight_error_fro
finite/divergence flags
```

With quantization enabled, it also logs:

```text
weight_saturation_fraction_max
weight_near_rail_fraction_max
gradient_saturation_fraction_max
gradient_near_rail_fraction_max
update_saturation_fraction_max
update_underflow_fraction_max
```

### Update Geometry Diagnostics

Two metrics are especially important for the quantized global-throttle ablation:

```text
actual_update_norm
update_cosine
update_angle_rad
update_radius_ratio
```

`raw_update_norm` is the norm of the intended update before storage effects:

```math
\Delta\theta_{\mathrm{raw}}
=
-\alpha_t\eta G_t.
```

`actual_update_norm` is computed after the update has been applied and after any quantized variable storage has been enforced:

```math
\Delta\theta_{\mathrm{actual}}
=
\theta_{t+1}-\theta_t.
```

This catches silent learning death. A run can look numerically stable because the applied update has underflowed to zero; that is not a successful controller result.

`update_cosine` compares the applied update to the intended update:

```math
\cos_t
=
\frac{
\left\langle
\Delta\theta_{\mathrm{actual}},
\Delta\theta_{\mathrm{raw}}
\right\rangle
}{
\left\|\Delta\theta_{\mathrm{actual}}\right\|_2
\left\|\Delta\theta_{\mathrm{raw}}\right\|_2
+\varepsilon
}.
```

For pure global throttling, this should stay close to 1 because the throttle scales the full update vector uniformly. If quantization, clipping, or a future row/column projection changes the update direction, this cosine should fall.

The phase-distortion diagnostics are derived from the same quantities:

```math
\beta_t
=
\arccos(\cos_t)
```

```math
r_t
=
\frac{
\left\|\Delta\theta_{\mathrm{actual}}\right\|_2
}{
\left\|\Delta\theta_{\mathrm{raw}}\right\|_2
+\varepsilon
}.
```

`update_angle_rad` stores `beta_t`; `update_radius_ratio` stores `r_t`.

These are offline software diagnostics. They are meant to visualize and compare numerical update distortion during ablations. They are not currently proposed as hardware controller inputs because the high-precision raw update may not exist as a physical hardware signal.

### Return Value

```python
return FitHistory(**{k: np.asarray(v) for k, v in history.items()})
```

Use:

```python
h = model.train_instrumented(...)
h.plot_results()
```

## `_forward_with_precision(...)`

```python
def _forward_with_precision(self, x, precision, *, training):
    ...
```

Manual forward path with fake quantization hooks.

Current behavior:

1. Quantize input with `precision.dtype("input", "value")`.
2. For each Dense layer:
   - quantize kernel as `"weight"`,
   - matrix multiply,
   - quantize dot-product as `"accumulator"`,
   - add quantized bias if present,
   - quantize layer output as `"activation"`.
3. For non-Dense layers:
   - call layer normally,
   - quantize output as `"activation"`.

The quantizers use STE inside `GradientTape`, so the forward value is quantized but gradients still flow.

Extension point:

If Conv layers are added, implement their explicit forward path here.

## `_quantize_gradients(...)`

```python
def _quantize_gradients(self, grads, trainable_vars, precision):
    ...
```

For each gradient:

1. Find the owning layer with `_layer_and_field_for_variable`.
2. Lookup `precision.dtype(layer_name, "gradient")`.
3. Quantize the gradient if a dtype exists.

Missing gradient dtype means floating point gradient.

## `_quantize_variable_storage(...)`

```python
def _quantize_variable_storage(self, var, precision):
    ...
```

After the update is applied, this quantizes the stored variable:

- Dense kernel uses `"weight"`,
- Dense bias uses `"bias"`,
- other trainable variables use `"value"`.

This models fixed-point storage.

## `_layer_and_field_for_variable(...)`

```python
def _layer_and_field_for_variable(self, var) -> tuple[str, str]:
    ...
```

Maps a Keras variable to:

```python
(layer_name, field_name)
```

Examples:

```text
dense0/kernel -> ("dense0", "weight")
dense0/bias   -> ("dense0", "bias")
```

This function is central to `PrecisionDict` integration. If a new layer type has special trainable variables, update this mapping.

## `_same_variable(a, b)`

```python
@staticmethod
def _same_variable(a, b) -> bool:
    ...
```

Defensive Keras variable comparison helper. It first checks identity, then tries `path`, then falls back to `name`.

This avoids fragile behavior across Keras/TensorFlow versions.

## `_rail_max_for_variables(...)`

```python
def _rail_max_for_variables(self, vars_, precision, *, fields):
    ...
```

Computes max saturation and near-rail fractions across trainable variables for selected fields.

Used for:

```text
weight_saturation_fraction_max
weight_near_rail_fraction_max
```

## `_rail_max_for_tensors(...)`

```python
def _rail_max_for_tensors(self, tensors, trainable_vars, precision, *, field):
    ...
```

Computes max rail pressure across non-variable tensors, currently gradients.

Used for:

```text
gradient_saturation_fraction_max
gradient_near_rail_fraction_max
```

## `LinearBlockModel`

```python
@dataclass
class LinearBlockModel(BaseModel):
    num_hidden: list = field(default_factory=lambda: [64, 64])
    activation: Optional[Union[str, tf.keras.layers.Activation]] = None
    use_batchnorm: bool = False
    use_bias: bool = True
    name: str = "LinearBlockModel"
```

Dense-block model used by the first ablations.

Each block is:

```text
Dense -> optional Activation -> optional BatchNorm
```

## `LinearBlockModel.__post_init__()`

```python
def __post_init__(self):
    super().__post_init__()
    self.model = self._build_model(...)
```

Builds the Keras model immediately after dataclass initialization.

## `_build_model(input_shape, output_shape, verbose=True)`

```python
def _build_model(self, input_shape, output_shape, verbose=True) -> tf.keras.Model:
    ...
```

Build process:

1. Create Keras input named `model_input`.
2. Add Dense blocks using stable names:

```text
dense0
dense1
...
```

3. Add optional activations:

```text
activation0
activation1
...
```

4. Add optional batchnorm:

```text
batchnorm0
batchnorm1
...
```

5. If the last Dense output dimension does not match the dataset output, add a final Dense layer.

Usage:

```python
model = kappa.LinearBlockModel(
    dataset=dataset,
    num_hidden=[8, 2],
    activation="relu",
    use_batchnorm=False,
    use_bias=True,
)
```

## Extension Checklist

When adding a new model class:

1. Subclass `BaseModel`.
2. Build `self.model` in `__post_init__()`.
3. Use stable layer names.
4. Avoid reserved layer names `input` and `loss`.
5. If the model has non-Dense trainable variables, update `_layer_and_field_for_variable`.
6. If fake quantization must happen inside special ops, update `_forward_with_precision`.

When adding a new optimizer:

1. Keep the flattened global update available for logging.
2. Log intended update and actual applied update.
3. Keep `update_cosine` meaningful.
4. Add precision fields for optimizer state, such as `momentum`, `adam_m`, or `adam_v`.
