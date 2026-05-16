---
sidebar_label: "📈 history.py"
status:
  - valid
  - inprogress
tags:
  - implementation
  - history
last_modified: 2026-05-16
source: "enabol/history.py"
source_url: "https://github.com/manuelblancovalentin/kappa_budgeting/blob/main/enabol/history.py"
---
# 📈 History Module Reference
<PageMeta />
---

<TBox type="summary" title="What this page covers">
This page is a coder-facing reference for `enabol/history.py`. The module currently contains the `FitHistory` container used by instrumented training loops and one default plotting routine for the global-throttle sanity experiments.
</TBox>

## Module Map

| Object | Kind | Purpose |
|---|---|---|
| `FitHistory` | class | Dictionary-like container mapping metric names to NumPy arrays. |
| `FitHistory.__repr__()` | method | Prints metric names and array shapes. |
| `FitHistory.plot_results(...)` | method | Creates the default six-panel diagnostic plot. |

<TBox type="todo" title="History TODOs">

- [ ] Split plotting presets by experiment family once EXP-001 and multi-layer tests exist.
- [ ] Add optional save/export helpers so notebooks can write figures consistently.
- [ ] Consider moving plotting code into `plots.py` if `FitHistory` starts carrying too many display responsibilities.

</TBox>

## Imports And Dependencies

```python
import numpy as np
import matplotlib.pyplot as plt
```

`history.py` intentionally has no TensorFlow dependency. It receives arrays after the training loop has already converted logs into NumPy values.

## Classes

### `FitHistory`

```python
class FitHistory(dict[str, np.array]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
```

`FitHistory` is a thin `dict` subclass. Keys are metric names and values are arrays with one entry per logged training step.

Typical construction happens at the end of `BaseModel.train_instrumented(...)`:

```python
return FitHistory(**{k: np.asarray(v) for k, v in history.items()})
```

Typical notebook usage:

```python
history = model.train_instrumented(X, Y, use_controller=True)
print(history)
history.plot_results(title="EXP-000A controller run")
```

Expected keys depend on the training loop. Current one-layer ablations commonly include:

| Key | Meaning |
|---|---|
| `loss` | Training loss per step. |
| `weight_error_fro` | Frobenius error against the teacher matrix, when a reference matrix is supplied. |
| `grad_norm` | Global gradient norm. |
| `curvature_proxy` | Online gradient-change curvature proxy. |
| `curvature_ema` | EMA-smoothed curvature signal used by the controller. |
| `hessian_lambda_max` | Analytic or logged maximum Hessian eigenvalue when available. |
| `stability_margin_lambda_raw` | Raw `eta * lambda_max(H)` margin. |
| `alpha_would` | Controller throttle that would be applied, even in baseline runs. |

## Class Methods

### `__repr__()`

```python
def __repr__(self) -> str:
    s = "FitHistory:\n"
    for k, v in self.items():
        s += f"  {k}: {v.shape}\n"
    return s
```

Returns a compact shape summary. This is useful in notebooks because it confirms which metrics were recorded without printing full arrays.

Example output:

<Terminal
  title="FitHistory repr"
  content={`FitHistory:
  loss: (1200,)
  grad_norm: (1200,)
  curvature_proxy: (1200,)
  alpha_would: (1200,)`}
/>

### `plot_results(title=None)`

```python
def plot_results(self, title=None):
    fig, axs = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    ...
```

Creates the default six-panel plot used by the first global-throttle experiments:

| Panel | Metric |
|---|---|
| Top left | `loss` |
| Top right | `weight_error_fro` |
| Middle left | `grad_norm` |
| Middle right | `curvature_proxy`, `curvature_ema`, `hessian_lambda_max` |
| Bottom left | `stability_margin_lambda_raw` with a horizontal stability limit at `2.0` |
| Bottom right | `alpha_would` |

Usage:

```python
history.plot_results(title="Baseline run")
```

<TBox type="warning" title="Plot preset is not universal">
`plot_results(...)` assumes the history contains the keys above. It is correct for the early one-layer throttle experiments, but future experiments with quantization, phase diagnostics, or multi-layer metrics may need separate plotting presets.
</TBox>

## Extension Notes

Keep `FitHistory` lightweight. Its job is to carry logged arrays and provide simple convenience plots. If the project needs richer plotting, prefer adding explicit plot helpers or experiment-specific notebook functions rather than making `FitHistory` responsible for every visualization.
