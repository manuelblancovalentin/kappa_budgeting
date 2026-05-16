---
sidebar_label: "📚 dataset.py"
status:
  - valid
  - inprogress
tags:
  - implementation
  - dataset
last_modified: 2026-05-15
source: "enabol/dataset.py"
source_url: "https://github.com/manuelblancovalentin/kappa_budgeting/blob/main/enabol/dataset.py"
---
# 📚 Dataset Module Reference
<PageMeta />
---

<TBox type="summary" title="What this page covers">
This page is a coder-facing reference for `enabol/dataset.py`: the shared `BaseDataset` interface, plotting/export helpers, and the currently implemented synthetic affine dataset. The experiment-facing mathematical description of the affine dataset lives in the dataset registry page, so this page focuses on code structure and extension points.
</TBox>

## Module Map

| Group | Objects | Purpose |
|---|---|---|
| Enum tags | `DataType` | Describe whether inputs/outputs are image, 1D, 2D, class, or float data. |
| Base class | `BaseDataset` | Common dataset interface used by model wrappers and export helpers. |
| Implemented dataset | `AffineDataset` | Synthetic affine regression data used by the current sanity experiments. |

## Dataset Coverage

This table tracks what exists in the active `enabol` module and what still needs to be ported from `old_enabol/dataset.py`. Priority is intentionally blank for implemented datasets.

| Dataset family | Class / source | Status | Priority | Notes / Documentation |
|---|---|---|---|---|
| Synthetic affine regression | `AffineDataset` | <Badge status="valid" /> |  | [`DS-AFFINE-LINEAR-000`](../datasets/affine-linear-000.md) |
| Simple synthetic regression | `LinearDataset`, `PolynomialDataset` | <Badge status="missing" /> | <Badge status="priority-high" /> | Useful for controlled ablations before image/audio datasets. |
| Simple classification | `LinearClassificationDataset` | <Badge status="missing" /> | <Badge status="priority-medium" /> | Useful once classification losses are added to the harness. |
| MNIST / Fashion-MNIST | `MNISTDataset`, `FashionMNISTDataset` | <Badge status="missing" /> | <Badge status="priority-medium" /> | Good first image-classification benchmarks. |
| CIFAR-10 | `CIFAR10Dataset` | <Badge status="missing" /> | <Badge status="priority-medium" /> | Higher-dimensional image benchmark after MNIST path is stable. |
| TinyML ToyADMOS / MSWC-style audio | `TinyMLToyADMOSDataset`; future MSWC loader | <Badge status="missing" /> | <Badge status="priority-medium" /> | Important for TinyML realism; needs careful preprocessing docs. |
| Jet tagging | `JetTaggingDataset` | <Badge status="missing" /> | <Badge status="priority-medium" /> | Relevant to hls4ml/physics workflows. |
| MedMNIST | `MedMNISTDataset` | <Badge status="missing" /> | <Badge status="priority-low" /> | Useful later for medical-image style examples. |
| H5 / Fusion / ImageNet loaders | `H5Dataset`, `FusionDataset`, `ImageNetDataset` | <Badge status="missing" /> | <Badge status="priority-low" /> | Port only when a concrete experiment needs them. |

<TBox type="todo" title="Dataset TODOs">

- [ ] Port the simplest synthetic datasets first: linear, polynomial, and linear classification.
- [ ] Decide whether image/audio datasets belong in core `enabol` or optional extras with heavier dependencies.
- [ ] Add registry pages only when a dataset is actually used by an experiment.

</TBox>

## Imports And Dependencies

`dataset.py` uses:

```python
import os
from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

from .utils import analytic_single_dense_hessian, hessian_metrics_np
```

The important dependency is `utils.py`, which provides the analytic Hessian helper used by `AffineDataset`.

## Classes

### `DataType`

```python
class DataType(Enum):
    IMAGE_CLASS = "image-class"
    IMAGE_FLOAT = "image-float"
    D1_CLASS = "1d-class"
    D1_FLOAT = "1d-float"
    D2_CLASS = "2d-class"
    D2_FLOAT = "2d-float"
```

`DataType` is a lightweight tag used by plotting and export helpers. It does not change the numeric data. It tells helper methods how to interpret `X` and `Y`.

Current usage:

```python
dataset.input_type
dataset.output_type
```

When adding a new dataset, set these tags in `__post_init__()` after generating `X` and `Y`.

### `BaseDataset`

```python
@dataclass
class BaseDataset:
    X: np.ndarray = field(init=False)
    Y: np.ndarray = field(init=False)
    num_samples: int = 1000
    input_type: DataType = field(init=False)
    output_type: DataType = field(init=False)
    seed: Optional[int] = None
```

`BaseDataset` is the common interface expected by the models. Subclasses must generate:

- `self.X`,
- `self.Y`,
- `self.input_type`,
- `self.output_type`.

#### `input_shape`

```python
@property
def input_shape(self) -> Tuple[int, ...]:
    return self.X.shape
```

Used by `BaseModel.__post_init__()` to infer the Keras input shape.

#### `output_shape`

```python
@property
def output_shape(self) -> Tuple[int, ...]:
    return self.Y.shape
```

Used by `LinearBlockModel` to decide whether it needs a final output Dense layer.

#### `reference_weight_matrix`

```python
@property
@abstractmethod
def reference_weight_matrix(self) -> np.ndarray:
    pass
```

Subclasses should return the teacher matrix used to generate targets. This is used by experiment plots and `train_instrumented(..., reference_A=...)`.

#### `reference_bias_vector`

```python
@property
@abstractmethod
def reference_bias_vector(self) -> np.ndarray:
    pass
```

Subclasses should return the teacher bias. For no-bias tests this should be zeros.

#### `get()`

```python
def get(self) -> Tuple[np.ndarray, np.ndarray]:
    return self.X, self.Y
```

Primary notebook API:

```python
X, Y = dataset.get()
```

#### `to_numpy()`

```python
def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
    return self.get()
```

Alias for `get()`. Useful if we later support datasets backed by another object and want an explicit conversion call.

#### `to_txt(prefix)`

```python
def to_txt(self, prefix: str = "dataset") -> None:
    np.savetxt(f"{prefix}_X.txt", self.X)
    np.savetxt(f"{prefix}_Y.txt", self.Y)
```

Writes plain-text arrays:

```python
dataset.to_txt("affine_nominal")
```

Outputs:

```text
affine_nominal_X.txt
affine_nominal_Y.txt
```

#### `to_dat(prefix)`

```python
def to_dat(self, prefix: str = "dataset") -> None:
    os.makedirs(prefix, exist_ok=True)
    np.savetxt(os.path.join(prefix, "tb_input_features.dat"), ...)
    np.savetxt(os.path.join(prefix, "tb_output_predictions.dat"), ...)
```

Writes files using hls4ml-style testbench names:

```text
tb_input_features.dat
tb_output_predictions.dat
```

Use this when exporting a synthetic dataset toward firmware/HLS tests.

#### `plot(max_points)`

```python
def plot(self, max_points: int = 100) -> None:
    ...
```

Dispatches plotting based on `input_type` and `output_type`.

Important current path:

```python
DataType.D2_FLOAT -> DataType.D2_FLOAT
```

This creates pairwise scatter plots between each input feature and each output target.

Usage:

```python
dataset.plot(max_points=200)
```

When adding a new dataset type, either reuse existing tags or add a new branch here.

#### `plot_histogram(bins)`

```python
def plot_histogram(self, bins: Optional[int] = None) -> None:
    ...
```

Plots marginal histograms for float datasets. This is useful for confirming drift or quantization ranges.

Usage:

```python
dataset.plot_histogram(bins=40)
```

#### `__repr__()`

```python
@abstractmethod
def __repr__(self) -> str:
    pass
```

Subclasses should provide a compact experiment-readable summary.

### `AffineDataset`

```python
@dataclass
class AffineDataset(BaseDataset):
    A: np.ndarray = field(default_factory=...)
    b: np.ndarray = field(default_factory=...)
    use_bias: bool = True
```

`AffineDataset` is the only dataset currently implemented in the active module. Its experiment-level definition, default teacher matrix, plots, drift model, and usage in EXP-000A / EXP-000B are documented in [`DS-AFFINE-LINEAR-000`](../datasets/affine-linear-000.md).

This implementation page only records the code responsibilities:

- generate `X` and `Y`,
- set `input_type` and `output_type`,
- expose `reference_weight_matrix` and `reference_bias_vector`,
- expose `analytic_hessian` for the one-layer no-bias sanity tests,
- provide an experiment-readable `__repr__()`.

#### `__post_init__()`

Responsibilities:

1. Fill default `A` or `b` if missing.
2. Replace `b` with zeros when `use_bias=False`.
3. Validate that `A.shape[0] == b.shape[0]`.
4. Create a seeded NumPy random generator.
5. Generate `X`.
6. Generate `Y = X @ A.T + b`.
7. Set `input_type` and `output_type`.
8. Initialize Hessian cache fields.

Usage:

```python
dataset = enabol.AffineDataset(num_samples=1000, use_bias=False, seed=1)
X, Y = dataset.get()
```

#### `reference_weight_matrix`

```python
@property
def reference_weight_matrix(self) -> np.ndarray:
    return self.A
```

Used for:

```python
h = model.train_instrumented(..., reference_A=dataset.reference_weight_matrix)
```

#### `reference_bias_vector`

```python
@property
def reference_bias_vector(self) -> np.ndarray:
    return self.b
```

Use this once bias terms are reintroduced into the ablations.

#### `analytic_hessian`

```python
@property
def analytic_hessian(self) -> dict[str, np.ndarray | float]:
    ...
```

For the current one-layer no-bias linear regression tests, this returns:

```python
{
    "hessian": H_nom,
    "lambda_max": lam_nom,
    "eta_max": 2.0 / lam_nom,
}
```

This is the diagnostic anchor for [`EXP-000A`](../experiments/exp-000a-global-throttle-float-lin1.md).

#### `__repr__()`

Returns a multi-line summary:

```python
print(dataset)
```

Includes shape, data tags, teacher parameters, analytic Hessian `lambda_max`, and nominal maximum stable learning rate.

## Extension Notes

When adding a new dataset:

1. Subclass `BaseDataset`.
2. Generate `self.X` and `self.Y` in `__post_init__()`.
3. Set `self.input_type` and `self.output_type`.
4. Implement `reference_weight_matrix`.
5. Implement `reference_bias_vector`.
6. Implement `__repr__()`.
7. Add analytic diagnostics only if they are actually valid.

Example skeleton:

```python
@dataclass
class MyDataset(BaseDataset):
    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self.X = ...
        self.Y = ...
        self.input_type = DataType.D2_FLOAT
        self.output_type = DataType.D1_FLOAT

    @property
    def reference_weight_matrix(self):
        return ...

    @property
    def reference_bias_vector(self):
        return ...

    def __repr__(self):
        return "MyDataset(...)"
```
