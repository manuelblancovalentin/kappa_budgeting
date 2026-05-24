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
| `TestbenchLayerData` | class | Per-layer container for raw parameter traces and parameter summary statistics. |
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
| `controller.dat` | every logged training sample | `dtheta_sq`, `dgrad_sq`, `lhs_sq`, `rhs_sq`, `alpha_feasible`, `alpha_state` |
| `<layer>/weights.dat` | once per epoch | one column per flattened matrix element, named `weight_<input>_<output>` |
| `<layer>/biases.dat` | once per epoch | one column per bias element, named `bias_<output>` |

Each file repeats the same index columns:

```text
epoch,sample,global_step,sample_index,loss
```

The repeated index columns are intentional. Traces are allowed to have different cadences. For example, `loss.dat` and `alpha.dat` are dense per-sample traces, while per-layer parameter traces are written only once per epoch.

Dense weights use the same row-major flattening convention as hls4ml dense kernels: `weight_<input>_<output>` maps to flat index `input * n_out + output`.

`TestbenchData.from_dir(...)` recursively discovers trace files, but it does not put raw parameter matrices into the main dataframe. Top-level traces such as `loss.dat`, `alpha.dat`, and `controller.dat` are outer-merged into `tb.frame`. Per-layer parameter traces are loaded into `tb.layers[layer_name]` when `load_weights` allows that layer.

`controller.dat` is the firmware-side controller diagnostic trace. For GT-0 and GT-1, `dtheta_sq` and `dgrad_sq` are the global summed curvature-sensor quantities, `lhs_sq` and `rhs_sq` are the inequality terms used by the division-free candidate search, `alpha_feasible` is the largest candidate accepted by the raw feasibility check, and `alpha_state` is the final controller state applied to the SGD update. For `CTRL-NONE`, the same columns are emitted with zero curvature terms and unit alpha-state values so downstream analysis can use a stable schema.

Parameter traces have their own raw dataframes and summary statistics. For example, `tb.layers["dense0"].weights` is the raw weights dataframe, and `tb.layers["dense0"].stats` is a dataframe of scalar summaries computed from weights and biases. Sparse traces therefore remain sparse at the layer level, while `tb.stats_frame` exposes scalar summaries with names such as `dense0.weights.mean` and `dense0.biases.norm_l2`.

## Usage

```python
from enabol import TestbenchData

tb = TestbenchData.from_dir(hls_model.config.get_output_dir(), load_weights=True)
tb.frame.head()
tb.layers["dense0"].weights.head()
tb.layers["dense0"].stats.head()
tb.plot_training(window_size=30)
print(tb)
```

`from_dir(...)` accepts:

- the hls4ml output directory
- the `tb_data` directory
- the final `tb_data/training` directory

`load_weights` controls whether per-layer parameter traces are loaded:

| Value | Behavior |
|---|---|
| `True` | Load all layer parameter traces. |
| `False` | Load only top-level traces such as `loss` and `alpha`. |
| `["dense0", "dense1"]` | Load only those layer parameter traces. |

The top-level merged dataframe is stored at `tb.frame` and indexed by:

```text
epoch, sample, global_step, sample_index
```

`epoch` and `sample` are one-based because they are human-facing training coordinates. `global_step` and `sample_index` are zero-based because they are counters and indexes back into the original `.dat` dataset rows.

Trace comments are parsed into dictionaries:

| Attribute | Meaning |
|---|---|
| `tb.metadata` | Metadata from the first trace file, normally enough for display. |
| `tb.metadata_by_trace` | Metadata keyed by trace name, for example `loss`, `alpha`, `controller`, `dense0/weights`, and `dense0/biases`. |
| `tb.layers` | Mapping from layer name to `TestbenchLayerData`. |
| `tb.stats_frame` | Scalar summaries computed from loaded layer parameter traces. |
| `tb.scalar_frame` | `tb.frame` plus `tb.stats_frame`, used by `plot_training(...)`. |

The `TestbenchData` representation prints a compact table with the directory, traces, metrics, and key run metadata:

```text
TestbenchData
+------------+--------------------------------+
| Directory  | .../tb_data/training           |
| Rows       | 2002                           |
| Traces     | alpha, dense0/biases, dense0/weights, loss |
| Metrics    | alpha, loss                    |
| Layers     | dense0                         |
| Layer Stats | dense0.weights.mean, dense0.weights.std, ... |
+------------+--------------------------------+
```

## Parameter Statistics

For each loaded parameter trace, ENABOL computes:

```text
mean, std, min, max, median, q05, q25, q75, q95,
norm_l2, norm_inf, sparsity_fraction,
saturation_fraction, near_rail_fraction, underflow_fraction
```

The rail and underflow fractions are present now but return `NaN` until the reader receives explicit rail/underflow thresholds from precision metadata. This keeps the dataframe schema stable while avoiding fake hardware-safety numbers.

## Plotting

`plot_training(...)` creates:

| Panel | Content |
|---|---|
| Metadata | Compact run metadata parsed from trace comments. |
| Loss | Rolling mean and quantile bands on a log scale. |
| Alpha | Rolling mean and quantile bands for the controller output. |
| Controller norms | Rolling mean and quantile bands for `||Delta G||` and `||Delta theta||` on twin y-axes. |
| Parameter metrics | Rolling mean and quantile bands for loaded parameter summary statistics, after dropping missing sparse-trace rows. |

The controller norm panel is derived from `controller.dat`: `dgrad_norm = sqrt(dgrad_sq)` and `dtheta_norm = sqrt(dtheta_sq)`. The raw squared metrics remain available in `tb.frame`, while the derived norms are exposed through `tb.scalar_frame`.

The bottom x-axis is `global_step`. The top x-axis of the first metric panel marks epoch starts. By default, `plot_training()` promotes the expected firmware-training panels first: loss, alpha, controller norms, and then loaded parameter statistics. Pass `metrics=[...]` to inspect a smaller subset:

```python
tb.plot_training(metrics=["loss", "dense0.weights.mean", "dense0.weights.norm_l2"], window_size=1)
```

Controller diagnostics can be plotted with the standardized controller panel by requesting both norms:

```python
tb.plot_training(metrics=["loss", "alpha", "dgrad_norm", "dtheta_norm"], window_size=1)
```

Use `scales={...}` to override y-axis scales. `loss` defaults to `log`; all other metrics default to Matplotlib's linear scale unless specified:

```python
tb.plot_training(
    metrics=["loss", "alpha", "dgrad_norm", "dtheta_norm"],
    scales={"dgrad_norm": "log", "dtheta_norm": "log"},
    window_size=30,
)
```

Raw individual parameter values remain available through `tb.layers[layer].weights` and `tb.layers[layer].biases`. For larger networks, scalar summaries are the default plot backend. More specialized views can be layered on top later: selected element line plots, before/after histograms, and weight-matrix heatmaps.

Use `show=False` in tests or scripts that need the figure object:

```python
fig, axes = tb.plot_training(show=False)
```

## Change Log

| Task | Date | Files | Summary |
|---|---|---|---|
| [ENB-024](/docs/status/tasks?query=ENB-024) | 2026-05-23 | `enabol/testbench.py`, `tests/test_history.py` | Added `TestbenchData.from_dir(...)`, metadata parsing, merged trace dataframe storage, and `plot_training(...)` for hls4ml trainable CSIM traces. |
| [HLS4ML-037](/docs/status/tasks?query=HLS4ML-037) | 2026-05-23 | `hls4ml/writer/vivado_writer.py`, `enabol/testbench.py`, `tests/test_history.py` | Added per-layer epoch-level parameter traces, sparse nested-trace loading, default all-metric plotting, and a table-style `TestbenchData` representation. |
| [ENB-025](/docs/status/tasks?query=ENB-025) | 2026-05-23 | `enabol/testbench.py`, `tests/test_history.py` | Moved raw parameter traces into per-layer objects, added `load_weights`, and computed per-layer parameter summary dataframes for plotting. |
| [HLS4ML-038](/docs/status/tasks?query=HLS4ML-038) | 2026-05-23 | `hls4ml/writer/vivado_writer.py`, `enabol/testbench.py`, `tests/test_history.py` | Added `controller.dat` as a top-level scalar trace for controller curvature diagnostics and alpha-state analysis. |
| [ENB-027](/docs/status/tasks?query=ENB-027) | 2026-05-23 | `enabol/testbench.py`, `tests/test_history.py` | Standardized the training panel with derived controller norms, a twin-axis controller norm panel, and per-metric y-axis scale overrides. |
