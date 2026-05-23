---
sidebar_label: "🧪 testbench.py"
status:
  - valid
  - inprogress
tags:
  - implementation
  - hls4ml
  - csim
  - testbench
last_modified: 2026-05-23
author: mbvalentin
source: "enabol/testbench.py"
source_url: "https://github.com/manuelblancovalentin/kappa_budgeting/blob/master/enabol/testbench.py"
---
# 🧪 Testbench Module Reference
<PageMeta />
---

<TBox type="summary" title="What this page covers">
`enabol/testbench.py` reads and plots outputs produced by generated hls4ml CSIM testbenches. It is separate from `FitHistory`, which remains the software-training history object.
</TBox>

## Module Map

| Object | Kind | Purpose |
|---|---|---|
| `TestbenchData` | class | Container for hls4ml testbench outputs and trainable CSIM traces. |
| `TestbenchData.from_dir(...)` | classmethod | Loads traces from an hls4ml output directory, `tb_data`, or `tb_data/training`. |
| `TestbenchData.from_trainable_dir(...)` | classmethod | Alias for explicit trainable-trace loading. |
| `TestbenchData.plot_training(...)` | method | Plots hls4ml trainable loss/alpha traces with metadata and epoch/global-step axes. |

## Why This Is Not `FitHistory`

`FitHistory` is the result of software training inside ENABOL/Keras/TensorFlow. `TestbenchData` is the result of reading generated files from a CSIM testbench. The data sources and future responsibilities are different:

| Object | Source | Responsibility |
|---|---|---|
| `FitHistory` | ENABOL software training loop | Logged NumPy arrays from Python training. |
| `TestbenchData` | hls4ml `tb_data` directory | File parsing, metadata, sample-index reconstruction, CSIM statistics, and firmware trace plots. |

`TestbenchData` may later create a `FitHistory` view if a comparison needs that shape, but it should remain the object that understands the testbench directory layout.

## Trainable Trace Layout

The generated trainable testbench writes comma-delimited `.dat` traces under:

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

The repeated index columns are intentional. Later traces can be sparse, for example logging loss every sample while logging weights only once per epoch.

## Usage

```python
from enabol import TestbenchData

tb = TestbenchData.from_dir(hls_model.config.get_output_dir())
tb.frame.head()
tb.plot_training(window_size=30)
```

`from_dir(...)` accepts:

- the hls4ml output directory
- the `tb_data` directory
- the final `tb_data/training` directory

The merged dataframe is indexed by:

```text
epoch, sample, global_step, sample_index
```

`epoch` and `sample` are one-based because they are human-facing training coordinates. `global_step` and `sample_index` are zero-based because they are counters and indexes back into the original `.dat` dataset rows.

## Plotting

`plot_training(...)` creates:

| Panel | Content |
|---|---|
| Metadata | Compact run metadata parsed from trace comments. |
| Loss | Rolling mean and quantile bands on a log scale. |
| Alpha | Rolling mean and quantile bands for the controller output. |

The bottom x-axis is `global_step`. The top x-axis of the first metric panel marks epoch starts.

Use `show=False` in tests or scripts that need the figure object:

```python
fig, axes = tb.plot_training(show=False)
```

## Change Log

| Task | Date | Files | Summary |
|---|---|---|---|
| [ENB-024](/docs/status/tasks?query=ENB-024) | 2026-05-23 | `enabol/testbench.py`, `tests/test_history.py` | Added `TestbenchData.from_dir(...)`, metadata parsing, merged trace dataframe storage, and `plot_training(...)` for hls4ml trainable CSIM traces. |
