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
| `TestbenchData.plot_training(...)` | method | Plots hls4ml trainable traces with metadata and epoch/global-step axes. |

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

| File | Cadence | Data columns |
|---|---|
| `loss.dat` | every logged training sample | `loss` |
| `alpha.dat` | every logged training sample | `alpha` |
| `<layer>/weights.dat` | once per epoch | one column per flattened matrix element, named `weight_<input>_<output>` |
| `<layer>/biases.dat` | once per epoch | one column per bias element, named `bias_<output>` |

Each file repeats the same index columns:

```text
epoch,sample,global_step,sample_index,loss
```

The repeated index columns are intentional. Traces are allowed to have different cadences. For example, `loss.dat` and `alpha.dat` are dense per-sample traces, while per-layer parameter traces are written only once per epoch.

Dense weights use the same row-major flattening convention as hls4ml dense kernels: `weight_<input>_<output>` maps to flat index `input * n_out + output`.

`TestbenchData.from_dir(...)` recursively discovers trace files and outer-merges them on the shared index. Sparse traces therefore become columns with `NaN` values on steps where that trace did not write a row. Top-level traces keep their natural metric names such as `loss` and `alpha`; nested traces are namespaced by path, for example `dense0.weights.weight_0_0` and `dense0.biases.bias_0`. `plot_training(...)` drops missing values per metric before computing the rolling mean and quantile bands, so the rolling window is expressed in available samples for that metric, not in global train steps. This keeps dense loss plots and sparse weight plots valid in the same object.

## Usage

```python
from enabol import TestbenchData

tb = TestbenchData.from_dir(hls_model.config.get_output_dir())
tb.frame.head()
tb.plot_training(window_size=30)
print(tb)
```

`from_dir(...)` accepts:

- the hls4ml output directory
- the `tb_data` directory
- the final `tb_data/training` directory

The merged dataframe is stored at `tb.frame` and indexed by:

```text
epoch, sample, global_step, sample_index
```

`epoch` and `sample` are one-based because they are human-facing training coordinates. `global_step` and `sample_index` are zero-based because they are counters and indexes back into the original `.dat` dataset rows.

Trace comments are parsed into dictionaries:

| Attribute | Meaning |
|---|---|
| `tb.metadata` | Metadata from the first trace file, normally enough for display. |
| `tb.metadata_by_trace` | Metadata keyed by trace name, for example `loss`, `alpha`, `dense0/weights`, and `dense0/biases`. |

The `TestbenchData` representation prints a compact table with the directory, traces, metrics, and key run metadata:

```text
TestbenchData
+------------+--------------------------------+
| Directory  | .../tb_data/training           |
| Rows       | 2002                           |
| Traces     | alpha, dense0/biases, dense0/weights, loss |
| Metrics    | alpha, dense0.weights.weight_0_0, loss     |
+------------+--------------------------------+
```

## Plotting

`plot_training(...)` creates:

| Panel | Content |
|---|---|
| Metadata | Compact run metadata parsed from trace comments. |
| Loss | Rolling mean and quantile bands on a log scale. |
| Alpha | Rolling mean and quantile bands for the controller output. |
| Parameter metrics | Rolling mean and quantile bands for selected scalar parameter elements, after dropping missing sparse-trace rows. |

The bottom x-axis is `global_step`. The top x-axis of the first metric panel marks epoch starts. By default, `plot_training()` plots every loaded scalar metric. Pass `metrics=[...]` to inspect one weight or a smaller subset:

```python
tb.plot_training(metrics=["loss", "dense0.weights.weight_0_0"], window_size=1)
```

For larger networks, scalar traces are the data backend. The more useful visual views will be layered on top: per-layer mean/std envelopes, before/after histograms, and weight-matrix heatmaps.

Use `show=False` in tests or scripts that need the figure object:

```python
fig, axes = tb.plot_training(show=False)
```

## Change Log

| Task | Date | Files | Summary |
|---|---|---|---|
| [ENB-024](/docs/status/tasks?query=ENB-024) | 2026-05-23 | `enabol/testbench.py`, `tests/test_history.py` | Added `TestbenchData.from_dir(...)`, metadata parsing, merged trace dataframe storage, and `plot_training(...)` for hls4ml trainable CSIM traces. |
| [HLS4ML-037](/docs/status/tasks?query=HLS4ML-037) | 2026-05-23 | `hls4ml/writer/vivado_writer.py`, `enabol/testbench.py`, `tests/test_history.py` | Added per-layer epoch-level parameter traces, sparse nested-trace loading, default all-metric plotting, and a table-style `TestbenchData` representation. |
