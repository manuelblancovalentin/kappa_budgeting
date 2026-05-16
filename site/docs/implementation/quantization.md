---
sidebar_label: "🧮 quantization.py"
status:
  - valid
  - inprogress
tags:
  - implementation
  - quantization
  - precision
last_modified: 2026-05-16
source: "enabol/quantization.py"
source_url: "https://github.com/manuelblancovalentin/kappa_budgeting/blob/main/enabol/quantization.py"
---
# 🧮 Quantization Module Reference
<PageMeta />
---

<TBox type="summary" title="What this page covers">
This page is a coder-facing reference for `enabol/quantization.py`: NumPy quantization, TensorFlow fake quantization with a straight-through estimator, rail/underflow statistics, and the TensorFlow implementations of HLS-style rounding, clipping, and wrapping.
</TBox>

## Module Map

| Group | Objects | Purpose |
|---|---|---|
| Statistics | `RailStats`, `rail_stats(...)` | Record saturation, near-rail pressure, and underflow pressure. |
| Public quantizers | `quantize_np(...)`, `quantize_tensor(...)` | Apply an `HLSDataType` to NumPy values or TensorFlow tensors. |
| TensorFlow internals | `_quantize_tensor_value(...)`, `_round_tensor(...)`, `_clip_tensor(...)`, `_wrap_tensor(...)` | Implement HLS-style quantization behavior for tensors. |

<TBox type="todo" title="Quantization TODOs">

- [ ] Add focused tests comparing NumPy dtype quantization against TensorFlow tensor quantization for every supported `QMODE` / `OMODE` pair.
- [ ] Decide whether wraparound modes are useful for ablations or should be avoided in favor of saturation-first experiments.
- [ ] Add optional richer rail statistics if future controllers use hardware-feasible saturation proxies.

</TBox>

## Imports And Dependencies

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import tensorflow as tf

from .dtypes import HLSDataType, ap_fixed, ap_int, ap_ufixed, ap_uint
```

`quantization.py` bridges the dtype descriptions in [`dtypes.py`](./dtypes.md) and the TensorFlow training path in [`nn.py`](./nn.md).

## Data Classes

#### `RailStats`

```python
@dataclass(frozen=True)
class RailStats:
    min_value: float
    max_value: float
    saturation_fraction: float
    near_rail_fraction: float
    underflow_fraction: float
```

`RailStats` is the small diagnostics object returned by `rail_stats(...)`.

| Field | Meaning |
|---|---|
| `min_value` | Lower representable rail for the dtype. |
| `max_value` | Upper representable rail for the dtype. |
| `saturation_fraction` | Fraction of values at or beyond either rail. |
| `near_rail_fraction` | Fraction of values whose magnitude is near the maximum rail magnitude. |
| `underflow_fraction` | Fraction of nonzero values smaller than half a quantization step. |

These values are used as software observables for quantized ablations. They are also candidates for hardware-feasible controller proxies later, but only after we decide which statistics are cheap enough to implement on edge.

## Public Functions

#### `quantize_np(x, dtype)`

```python
def quantize_np(x: Any, dtype: HLSDataType | None) -> Any:
    if dtype is None:
        return x
    return dtype(x)
```

Applies a dtype object to a NumPy-like value. If `dtype is None`, the input is returned unchanged, which is the default floating-point path.

Usage:

```python
from enabol import dtypes
from enabol.quantization import quantize_np

weight_dtype = dtypes.ap_fixed(12, 4, "AP_RND", "AP_SAT")
W_q = quantize_np(W, weight_dtype)
```

This path delegates to the dtype implementation in `dtypes.py`, so it is the reference behavior for NumPy-side checks.

#### `quantize_tensor(x, dtype, *, ste=True)`

```python
def quantize_tensor(x: tf.Tensor, dtype: HLSDataType | None, *, ste: bool = True) -> tf.Tensor:
    ...
```

Applies fake quantization to a TensorFlow tensor.

If `dtype is None`, the tensor is returned unchanged. Otherwise, the forward value is quantized by `_quantize_tensor_value(...)`.

With the default `ste=True`, the function uses a straight-through estimator:

```python
return x + tf.stop_gradient(q - x)
```

This means:

- forward pass value: quantized,
- backward pass gradient: identity through the quantizer.

That is the right default for fake quantization inside `tf.GradientTape` because hard rounding has zero or undefined gradients almost everywhere.

Usage:

```python
x_q = quantize_tensor(x, activation_dtype, ste=True)
```

<TBox type="warning" title="STE is a software approximation">
The straight-through estimator is useful for ablation studies, but it is not the same thing as hardware arithmetic. It lets us study the effect of quantized forward/update values while still allowing TensorFlow to produce gradients.
</TBox>

#### `rail_stats(x, dtype, *, near_ratio=0.95)`

```python
def rail_stats(x: Any, dtype: HLSDataType | None, *, near_ratio: float = 0.95) -> RailStats:
    ...
```

Computes saturation and rail-pressure statistics for an array-like value.

If the array is empty or `dtype is None`, it returns neutral values:

```python
RailStats(np.nan, np.nan, 0.0, 0.0, 0.0)
```

For a real dtype, it computes:

```math
\text{saturation fraction}
=
\frac{\#\{x_i \leq x_{min} \;\lor\; x_i \geq x_{max}\}}{N}.
```

The near-rail fraction is:

```math
\frac{\#\{|x_i| \geq r \cdot \max(|x_{min}|, |x_{max}|)\}}{N},
```

where `r = near_ratio`.

The underflow fraction is measured when the dtype exposes a `quantum`:

```math
\frac{\#\{x_i \neq 0 \;\land\; |x_i| < \frac{1}{2}\Delta\}}{N},
```

where `\Delta` is the quantization step.

Usage:

```python
stats = rail_stats(weights, weight_dtype)
print(stats.saturation_fraction)
print(stats.near_rail_fraction)
```

## TensorFlow Internal Functions

These helpers are implementation details. They are documented because anyone adding new HLS dtype behavior must keep the TensorFlow path aligned with `dtypes.py`.

#### `_quantize_tensor_value(x, dtype)`

```python
def _quantize_tensor_value(x: tf.Tensor, dtype: HLSDataType) -> tf.Tensor:
    ...
```

Converts a TensorFlow tensor into the scaled integer domain, rounds it, applies overflow behavior, and scales it back.

For fixed-point types:

```math
x_{scaled} = x \cdot 2^F,
```

where:

```math
F = WL - IWL.
```

Then:

```math
q = \frac{\operatorname{overflow}(\operatorname{round}(x_{scaled}))}{2^F}.
```

Supported dtype families:

| Dtype | Scale | Signed? | Default qmode/omode behavior |
|---|---:|---|---|
| `ap_fixed` | `2^fractional_bits` | yes | uses dtype `QMODE` / `OMODE` |
| `ap_ufixed` | `2^fractional_bits` | no | uses dtype `QMODE` / `OMODE` |
| `ap_int` | `1` | yes | `AP_TRN`, `AP_WRAP` |
| `ap_uint` | `1` | no | `AP_TRN`, `AP_WRAP` |

Unsupported dtype classes raise `TypeError`.

#### `_round_tensor(x, qmode)`

```python
def _round_tensor(x: tf.Tensor, qmode: str) -> tf.Tensor:
    ...
```

TensorFlow implementation of supported HLS-style rounding modes.

| QMODE | Behavior |
|---|---|
| `AP_TRN`, `AP_TRN_ZERO` | Truncate toward zero. |
| `AP_RND`, `AP_RND_CONV` | TensorFlow round-to-nearest. |
| `AP_RND_INF` | Round half toward positive infinity. |
| `AP_RND_MIN_INF` | Round half toward negative infinity. |
| `AP_RND_ZERO` | Round to nearest, ties toward zero. |

Unsupported modes raise `NotImplementedError`.

#### `_clip_tensor(x, WL, *, signed)`

```python
def _clip_tensor(x: tf.Tensor, WL: int, *, signed: bool) -> tf.Tensor:
    ...
```

Clips integer-domain values to the representable range.

Signed:

```math
[-2^{WL-1}, 2^{WL-1}-1].
```

Unsigned:

```math
[0, 2^{WL}-1].
```

This is used for `AP_SAT`, `AP_SAT_ZERO`, and `AP_SAT_SYM` in the current implementation.

<TBox type="warning" title="Saturation modes are currently grouped">
`AP_SAT`, `AP_SAT_ZERO`, and `AP_SAT_SYM` currently use the same TensorFlow clipping path. If future experiments depend on the subtle differences between these modes, update this function and add tests against the NumPy dtype behavior.
</TBox>

#### `_wrap_tensor(x, WL, *, signed)`

```python
def _wrap_tensor(x: tf.Tensor, WL: int, *, signed: bool) -> tf.Tensor:
    ...
```

Wraps integer-domain values modulo `2^WL`.

Unsigned values use:

```math
x \bmod 2^{WL}.
```

Signed values use the same modulo operation, then reinterpret values at or above the sign bit as negative:

```math
x_{signed} =
\begin{cases}
x_{wrapped} - 2^{WL}, & x_{wrapped} \geq 2^{WL-1} \\
x_{wrapped}, & \text{otherwise}.
\end{cases}
```

Wraparound is hardware-realistic, but it can make early ablation plots harder to interpret. Saturation should remain the first-line setting for stability experiments unless wrap behavior is the object being tested.

## Integration Points

`quantization.py` is used by `BaseModel.train_instrumented(...)` through three main paths:

| Training-loop path | Quantization role |
|---|---|
| `_forward_with_precision(...)` | Quantizes inputs, weights, biases, and activations during the forward pass. |
| `_quantize_gradients(...)` | Quantizes gradients before the update is formed. |
| `_quantize_variable_storage(...)` | Quantizes stored trainable variables after applying the update. |

The precision choices are supplied by [`PrecisionDict`](./precision.md). For example:

```python
from enabol import PrecisionDict, dtypes

precision = PrecisionDict({
    "dense0": {
        "weight": dtypes.ap_fixed(12, 4, "AP_RND", "AP_SAT"),
        "gradient": dtypes.ap_fixed(16, 6, "AP_RND", "AP_SAT"),
        "update": dtypes.ap_fixed(16, 4, "AP_RND", "AP_SAT"),
    },
})
```

## Extension Notes

When adding a new dtype behavior, keep these three paths synchronized:

1. NumPy dtype behavior in [`dtypes.py`](./dtypes.md),
2. TensorFlow fake-quantization behavior in this module,
3. documentation and tests that compare both implementations on the same values.

If those paths diverge, the software ablations can stop representing the intended hardware arithmetic.
