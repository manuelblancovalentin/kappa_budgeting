---
sidebar_label: "📦 nn.py"
status:
  - valid
  - inprogress
tags:
  - implementation
  - model
last_modified: 2026-05-15
author: mbvalentin
source: "enabol/nn.py"
source_url: "https://github.com/manuelblancovalentin/kappa_budgeting/blob/master/enabol/nn.py"
---
# 📦 Models / NN Module Reference
<PageMeta />
---

<TBox type="summary" title="What this page covers">
This page is a coder-facing reference for `enabol/nn.py`: the shared `BaseModel`, the current dense-only `LinearBlockModel`, quantization hooks, and the custom instrumented training loop. It also tracks model families from `old_enabol/nn.py` that have not been ported yet.
</TBox>

## Module Map

| Group | Objects | Purpose |
|---|---|---|
| Base class | `BaseModel` | Owns dataset linkage, Keras model object, simple training, and the instrumented online loop. |
| BaseModel public methods | `summary`, `reinitialize_weights`, `train`, `train_instrumented` | User-facing model utilities and training entry points. |
| BaseModel precision helpers | `_forward_with_precision`, `_quantize_gradients`, `_quantize_variable_storage`, rail helpers | Internal fake-quantization and diagnostic hooks. |
| Implemented model | `LinearBlockModel` | Dense-only model family used by the current ablation experiments. |

## Model Coverage

This table tracks active model classes and legacy model families that are still missing from the current `enabol` module. Priority is intentionally blank for implemented models.

| Model family | Class / source | Status | Priority | Notes / Documentation |
|---|---|---|---|---|
| Base model and training harness | `BaseModel` | <Badge status="valid" /> |  | Internal parent class documented below. |
| Dense linear blocks | `LinearBlockModel` | <Badge status="valid" /> |  | [`MDL-DENSE1-LINEAR-NOBIAS-000`](../models/dense1-linear-nobias-000.md) |
| Generic MLP | `MLPModel` | <Badge status="missing" /> | <Badge status="priority-high" /> | Needed for two-layer and deeper dense ablations. |
| Bounded activation layers | `ClippedReLU`, `ClippedReLUAdaptive` | <Badge status="missing" /> | <Badge status="priority-medium" /> | Needed when activation rails become part of the controller study. |
| Fusion CNN family | `FusionModel`, `TinyFusionModel`, `NanoFusionModel`, `NanoFusionModel16` | <Badge status="missing" /> | <Badge status="priority-medium" /> | Useful for TinyML image-like regression once dense ablations are stable. |
| Generic CNN classifier | `CNN` | <Badge status="missing" /> | <Badge status="priority-medium" /> | Useful once MNIST/Fashion-MNIST/CIFAR datasets are ported. |
| Jet tagging MLP | `JetTaggingModel` | <Badge status="missing" /> | <Badge status="priority-medium" /> | Relevant to hls4ml/physics workflows. |
| Autoencoders | `AEModel`, `cAEModel` | <Badge status="missing" /> | <Badge status="priority-low" /> | Port only if reconstruction experiments return. |

<TBox type="todo" title="Model TODOs">

- [ ] Port a minimal `MLPModel` or extend `LinearBlockModel` enough to support two-layer dense experiments.
- [ ] Decide whether bounded activations belong in `nn.py` or a separate `activations.py`.
- [ ] Add model registry pages only when a model is used by an experiment.

</TBox>

## Imports And Dependencies

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

## Classes

### `BaseModel`

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

#### `__post_init__()`

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

#### `_compile(...)`

```python
def _compile(self, optimizer=None, loss=None, metrics=None, **kwargs):
    ...
```

Thin wrapper over `self.model.compile(...)`.

Used by the simple `train()` method. The instrumented ablation loop does not use Keras optimizer application because it needs direct control over every update.

#### `summary()`

```python
def summary(self) -> None:
    ...
```

Prints Keras model summary if the model exists.

Usage:

```python
model.summary()
```

#### `reinitialize_weights()`

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

#### `train(...)`

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

#### `train_instrumented(...)`

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

##### Inputs

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

##### Training Loop Structure

The instrumented loop is easiest to understand as a top-level online training loop with clickable internal blocks. The explorer starts at `train_instrumented(...)`; click a block to inspect its local flowchart and pseudo-code, then use the back arrow to return to the parent loop.

<TrainInstrumentedExplorer />

##### Metric Block

The loop returns a `FitHistory` whose keys are grouped by what they diagnose. The exact set of populated metrics depends on whether `reference_A`, analytic Hessian logging, and `precision_dict` are enabled.

| Metric | Symbol / expression | What it measures | Notes |
|---|---|---|---|
| `loss` | $L_t$ | Current batch loss. | Usually `half_mse` in clean Hessian experiments. |
| `rmse` | $\sqrt{\operatorname{MSE}}$ | Output error scale. | Easier to read than raw loss. |
| `theta_norm` | $\lVert \theta_t \rVert_2$ | Global parameter magnitude. | Detects growth or collapse. |
| `grad_norm` | $\lVert G_t \rVert_2$ | Global gradient magnitude. | Detects exploding or vanishing gradients. |
| `raw_update_norm` | $\lVert \Delta\theta_{\mathrm{raw}} \rVert_2$ | Intended update magnitude before storage effects. | Uses the throttled raw update direction. |
| `actual_update_norm` | $\lVert \theta_{t+1}-\theta_t \rVert_2$ | Applied update magnitude after quantization/storage. | Detects silent update underflow. |
| `update_cosine` | $\cos(\Delta\theta_{\mathrm{actual}},\Delta\theta_{\mathrm{raw}})$ | Direction preservation. | Near `1` means little update rotation. |
| `update_angle_rad` | $\beta_t=\arccos(\cos_t)$ | Angular distortion in radians. | Used by phase-style plots. |
| `update_radius_ratio` | $r_t=\lVert\Delta\theta_{\mathrm{actual}}\rVert_2/(\lVert\Delta\theta_{\mathrm{raw}}\rVert_2+\varepsilon)$ | Applied-vs-intended update radius. | Values near `0` indicate update death. |
| `curvature_proxy` | $C_t=\lVert G_t-G_{t-1}\rVert/(\lVert\theta_t-\theta_{t-1}\rVert+\varepsilon)$ | Online curvature/update-field sensitivity proxy. | Controller input candidate. |
| `curvature_ema` | $S_t$ | Smoothed curvature proxy. | Reduces step-to-step noise. |
| `alpha` | $\alpha_t$ | Applied global throttle. | `1` in baseline mode. |
| `alpha_would` | $\alpha^{\mathrm{would}}_t$ | Controller value that would be applied. | Logged even when controller is disabled. |
| `eta_eff` | $\eta^{\mathrm{eff}}_t=\alpha_t\eta$ | Effective learning rate. | Compare against stability bounds. |
| `forward_gain_spectral` | approximate $\prod_l \lVert W_l\rVert_2$ | Forward gain proxy. | Static gain diagnostic, not the closed-loop condition. |
| `hessian_lambda_max` | $\lambda_{\max}(H_t)$ | True local curvature when available. | Exact only for supported toy cases. |
| `stability_margin_lambda_raw` | $\eta\lambda_{\max}(H_t)$ | Raw SGD stability margin. | Quadratic stability boundary is near `2`. |
| `stability_margin_lambda_ctrl` | $\alpha_t\eta\lambda_{\max}(H_t)$ | Throttled stability margin. | Should stay below the boundary in stable controller runs. |
| `spectral_radius_raw` | $\rho(I-\eta H_t)$ | Raw local update-map spectral radius. | Stable if below `1` in the quadratic local model. |
| `spectral_radius_ctrl` | $\rho(I-\alpha_t\eta H_t)$ | Throttled local update-map spectral radius. | Controller should reduce this when curvature rises. |
| `weight_error_fro` | $\lVert W_t-A\rVert_F$ | Teacher-weight error. | Only meaningful when `reference_A` is supplied. |
| finite/divergence flags | boolean indicators | Numerical run health. | Used to stop or label unstable trajectories. |

With quantization enabled, the loop also logs rail-pressure maxima:

| Metric | What it measures | Interpretation |
|---|---|---|
| `weight_saturation_fraction_max` | Max fraction of weights at fixed-point rails. | High values mean storage rails are active. |
| `weight_near_rail_fraction_max` | Max fraction of weights near rails. | Early warning before saturation. |
| `gradient_saturation_fraction_max` | Max fraction of gradients at rails. | Indicates gradient clipping by dtype. |
| `gradient_near_rail_fraction_max` | Max fraction of gradients near rails. | Early warning for gradient rail pressure. |
| `update_saturation_fraction_max` | Max fraction of updates at rails. | Indicates update dtype is too tight or learning rate too high. |
| `update_underflow_fraction_max` | Max fraction of nonzero updates below half a quantum. | Indicates learning may silently die. |

##### Update Geometry Diagnostics

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

##### Return Value

```python
return FitHistory(**{k: np.asarray(v) for k, v in history.items()})
```

Use:

```python
h = model.train_instrumented(...)
h.plot_results()
```

#### `_forward_with_precision(...)`

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

#### `_quantize_gradients(...)`

```python
def _quantize_gradients(self, grads, trainable_vars, precision):
    ...
```

For each gradient:

1. Find the owning layer with `_layer_and_field_for_variable`.
2. Lookup `precision.dtype(layer_name, "gradient")`.
3. Quantize the gradient if a dtype exists.

Missing gradient dtype means floating point gradient.

#### `_quantize_variable_storage(...)`

```python
def _quantize_variable_storage(self, var, precision):
    ...
```

After the update is applied, this quantizes the stored variable:

- Dense kernel uses `"weight"`,
- Dense bias uses `"bias"`,
- other trainable variables use `"value"`.

This models fixed-point storage.

#### `_layer_and_field_for_variable(...)`

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

#### `_same_variable(a, b)`

```python
@staticmethod
def _same_variable(a, b) -> bool:
    ...
```

Defensive Keras variable comparison helper. It first checks identity, then tries `path`, then falls back to `name`.

This avoids fragile behavior across Keras/TensorFlow versions.

#### `_rail_max_for_variables(...)`

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

#### `_rail_max_for_tensors(...)`

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

### `LinearBlockModel`

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

#### `__post_init__()`

```python
def __post_init__(self):
    super().__post_init__()
    self.model = self._build_model(...)
```

Builds the Keras model immediately after dataclass initialization.

#### `_build_model(input_shape, output_shape, verbose=True)`

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
model = enabol.LinearBlockModel(
    dataset=dataset,
    num_hidden=[8, 2],
    activation="relu",
    use_batchnorm=False,
    use_bias=True,
)
```

## Extension Notes

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
