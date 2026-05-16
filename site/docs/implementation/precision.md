---
sidebar_label: "🧮 precision.py"
status:
  - valid
  - inprogress
tags:
  - implementation
  - precision
  - quantization
last_modified: 2026-05-15
source: "enabol/precision.py"
source_url: "https://github.com/manuelblancovalentin/kappa_budgeting/blob/main/enabol/precision.py"
---
# 🧮 PrecisionDict Module Reference
<PageMeta />
---

<TBox type="summary" title="What this page covers">
This page is a coder-facing reference for `enabol/precision.py`: the explicit layer-indexed `PrecisionDict`, reserved precision scopes, dtype parsing, model validation, and usage patterns for per-layer quantization studies.
</TBox>

## Imports

```python
from collections.abc import Mapping
from typing import Any

import tensorflow as tf

from .dtypes import HLSDataType
```

The only package-level dependency is `HLSDataType`, which parses strings and validates dtype objects.

## Reserved Names

```python
RESERVED_NAMES = {"loss", "input", "__default__"}
```

These names are semantic precision scopes, not ordinary Keras layers.

Meaning:

| Name | Purpose |
|---|---|
| `input` | Precision for raw input tensors. |
| `loss` | Precision for loss value or loss-side signals. |
| `__default__` | Optional fallback precision fields. |

No Keras layer should be named `input` or `loss`.

## `PrecisionDict`

```python
class PrecisionDict(dict[str, dict[str, HLSDataType | None]]):
    ...
```

The top-level key is a layer or semantic scope:

```text
dense0
activation0
input
loss
```

The second-level key is a tensor family:

```text
weight
bias
activation
accumulator
gradient
update
value
```

Example:

```python
from enabol import dtypes, PrecisionDict

precisions = PrecisionDict({
    "input": {
        "value": "ap_fixed<12,4,AP_RND,AP_SAT>",
    },
    "dense0": {
        "weight": dtypes.ap_fixed(12, 4, "AP_RND", "AP_SAT"),
        "gradient": dtypes.ap_fixed(16, 6, "AP_RND", "AP_SAT"),
        "update": dtypes.ap_fixed(16, 4, "AP_RND", "AP_SAT"),
    },
    "loss": {
        "value": "ap_fixed<24,12,AP_RND,AP_SAT>",
    },
})

print(precisions)
```

should print:
<Terminal
  title="precision map"
  content={`PrecisionDict(
    input:
        value: ap_fixed<12,4,AP_RND,AP_SAT>
    dense0:
        weight: ap_fixed<12,4,AP_RND,AP_SAT>
        gradient: ap_fixed<16,6,AP_RND,AP_SAT>
        update: ap_fixed<16,4,AP_RND,AP_SAT>
    loss:
        value: ap_fixed<24,12,AP_RND,AP_SAT>
)`}
/>

Strings and dtype objects can be mixed. `None` means explicitly float/no quantization.

## `__init__(data)`

```python
def __init__(self, data: Mapping[str, Mapping[str, Any]] | None = None):
    ...
```

Builds the nested dictionary and parses all dtype entries.

Input:

```python
{
    "dense0": {
        "weight": "ap_fixed<12,4,AP_RND,AP_SAT>",
        "bias": None,
    }
}
```

Stored internally as:

```python
{
    "dense0": {
        "weight": ap_fixed(...),
        "bias": None,
    }
}
```

## `_parse_dtype(dtype)`

```python
@staticmethod
def _parse_dtype(dtype: Any) -> HLSDataType | None:
    ...
```

Rules:

- `None` stays `None`,
- dtype objects pass through,
- strings are parsed with `HLSDataType.from_dtype()`.

This is why notebook configs can be concise:

```python
"weight": "ap_fixed<12,4,AP_RND,AP_SAT>"
```

## `dtype(layer_name, field, default=None)`

```python
def dtype(self, layer_name: str, field: str, default=None):
    ...
```

Main lookup method.

Lookup order:

1. Exact layer and field:

```python
precisions["dense0"]["weight"]
```

2. Default field:

```python
precisions["__default__"]["weight"]
```

3. Provided default, usually `None`.

Usage:

```python
weight_t = precisions.dtype("dense0", "weight")
update_t = precisions.dtype("dense0", "update")
loss_t = precisions.dtype("loss", "value")
```

Missing fields default to floating point in the training loop because the returned dtype is `None`.

## `has(layer_name, field)`

```python
def has(self, layer_name: str, field: str) -> bool:
    return self.dtype(layer_name, field) is not None
```

Boolean convenience method.

Usage:

```python
if precisions.has("dense0", "update"):
    ...
```

## `layers()`

```python
def layers(self) -> list[str]:
    return [name for name in self.keys() if name != "__default__"]
```

Returns the non-default scopes.

Example:

```python
["input", "dense0", "loss"]
```

## `fields(layer_name)`

```python
def fields(self, layer_name: str) -> list[str]:
    return list(self.get(layer_name, {}).keys())
```

Useful for debugging:

```python
print(precisions.fields("dense0"))
```

## `validate_model(model, allow_missing=True)`

```python
def validate_model(self, model: tf.keras.Model, *, allow_missing: bool = True) -> None:
    ...
```

Checks that:

1. No Keras layer is named `loss`.
2. No Keras layer is named `input`.
3. Non-reserved precision entries match model layer names.
4. Optionally, every model layer has a precision entry.

Usage:

```python
precisions.validate_model(model.model)
```

For ablations, keep:

```python
allow_missing=True
```

because staged experiments intentionally quantize only selected paths.

Use:

```python
allow_missing=False
```

only when you want a fully specified hardware-style precision map.

## `describe()`

```python
def describe(self) -> str:
    ...
```

Returns a readable multi-line summary:

```python
print(precisions.describe())
```

Useful in notebooks so the exact precision map appears next to the plots.

## `ensure_precision_dict(precision)`

```python
def ensure_precision_dict(precision):
    ...
```

Accepts:

- `None`,
- an existing `PrecisionDict`,
- a nested mapping.

Returns either:

- `None`, or
- a `PrecisionDict`.

This is used by `train_instrumented()` so callers can pass either:

```python
precision_dict=precisions
```

or:

```python
precision_dict={
    "dense0": {
        "weight": "ap_fixed<12,4,AP_RND,AP_SAT>",
    },
}
```

## Extension Notes

Add new fields freely when the trainer supports them:

```python
"momentum"
"adam_m"
"adam_v"
"batchnorm_mean"
"batchnorm_variance"
```

The object is intentionally schema-light. Enforcement should live in the training/firmware path, not here, because different optimizers and layers require different fields.
