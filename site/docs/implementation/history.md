---
sidebar_label: "📈 history.py"
status:
  - valid
  - inprogress
tags:
  - implementation
  - history
last_modified: 2026-05-16
author: mbvalentin
source: "enabol/history.py"
source_url: "https://github.com/manuelblancovalentin/kappa_budgeting/blob/master/enabol/history.py"
---
# 📈 History Module Reference
<PageMeta />
---

<TBox type="summary" title="What this page covers">
This page is a coder-facing reference for `enabol/history.py`. The module contains the `FitHistory` container used by instrumented software training loops and the first hls4ml trainable CSIM trace reader.
</TBox>

## Module Map

| Object | Kind | Purpose |
|---|---|---|
| `FitHistory` | class | Dictionary-like container mapping metric names to NumPy arrays. |
| `FitHistory.__repr__()` | method | Prints metric names and array shapes. |
| `FitHistory.plot_results(...)` | method | Creates the default six-panel diagnostic plot. |
| `FitHistory.from_dir(...)` | classmethod | Loads hls4ml trainable traces from an output directory, `tb_data`, or `tb_data/training`. |
| `FitHistory.from_trainable_dir(...)` | classmethod | Alias for `from_dir(...)` when the caller wants a more explicit name. |
| `FitHistory.plot_training(...)` | method | Plots the hls4ml trainable loss/alpha panel with metadata and epoch/global-step axes. |

<TBox type="todo" title="History TODOs">

- [ ] Split plotting presets by experiment family once EXP-001 and multi-layer tests exist.
- [ ] Add optional save/export helpers so notebooks can write figures consistently.
- [ ] Consider moving plotting code into `plots.py` if `FitHistory` starts carrying too many display responsibilities.

</TBox>

## Imports And Dependencies

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

`history.py` intentionally has no TensorFlow dependency. It receives arrays or `.dat` traces after the training loop has already materialized logs.

## Classes

### `FitHistory`

```python
class FitHistory(dict[str, np.array]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
```

`FitHistory` is a `dict` subclass. Keys are metric names and values are arrays with one entry per logged training step. For hls4ml trainable traces it also carries:

| Attribute | Meaning |
|---|---|
| `frame` | A pandas dataframe indexed by `epoch`, `sample`, `global_step`, and `sample_index`. |
| `metadata` | Metadata parsed from the first trace file. |
| `metadata_by_trace` | Metadata dictionaries keyed by trace name, such as `loss` or `alpha`. |
| `source_dir` | The resolved `tb_data/training` directory. |

Typical construction happens at the end of `BaseModel.train_instrumented(...)`:

```python
return FitHistory(**{k: np.asarray(v) for k, v in history.items()})
```

Typical notebook usage:

```python
from enabol import Controller

history = model.train_instrumented(X, Y, controller=Controller.from_str("gt-order-0"))
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

#### `__repr__()`

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

#### `plot_results(title=None)`

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

### hls4ml Trainable Traces

The generated hls4ml trainable testbench writes comma-delimited `.dat` traces under:

```text
<hls4ml-output>/tb_data/training/
```

The first files are:

| File | Data column |
|---|---|
| `loss.dat` | `loss` |
| `alpha.dat` | `alpha` |

Each file repeats the same index columns:

```text
epoch,sample,global_step,sample_index,loss
```

Use `FitHistory.from_dir(...)` to read either the hls4ml output directory, the `tb_data` directory, or the final `tb_data/training` directory:

```python
from enabol import FitHistory

history = FitHistory.from_dir(hls_model.config.get_output_dir())
history.frame.head()
```

The merged dataframe is indexed by `epoch`, `sample`, `global_step`, and `sample_index`, so shuffled CSIM runs can still be mapped back to the original `.dat` sample rows.

### `plot_training(...)`

```python
history.plot_training(metrics=("loss", "alpha"), window_size=30)
```

This creates the default hls4ml trainable panel:

| Panel | Content |
|---|---|
| Metadata | Compact comment metadata from the trace files. |
| Loss | Rolling mean and quantile bands on a log scale. |
| Alpha | Rolling mean and quantile bands for the controller output. |

The bottom x-axis is `global_step`. The top x-axis of the first metric panel marks epoch starts. `epoch` and `sample` are one-based in the trace files, while `global_step` and `sample_index` remain zero-based.

Use `show=False` in tests or scripts that need the figure object:

```python
fig, axes = history.plot_training(show=False)
```

## Extension Notes

Keep `FitHistory` lightweight. Its job is to carry logged arrays and provide simple convenience plots. The hls4ml trainable reader is part of this module because it returns the same object, but richer plot families should still remain explicit methods rather than hidden notebook-only code.

## Change Log

| Task | Date | Files | Summary |
|---|---|---|---|
| [ENB-024](/docs/status/tasks?query=ENB-024) | 2026-05-23 | `enabol/history.py`, `tests/test_history.py` | Added `FitHistory.from_dir(...)`, `FitHistory.from_trainable_dir(...)`, metadata parsing, merged trace dataframe storage, and `plot_training(...)` for hls4ml trainable CSIM traces. |
