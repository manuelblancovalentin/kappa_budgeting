---
title: "Phase 5: ENABOL Bridge and Validation"
sidebar_label: "🚧 Phase 5: Bridge + Validation"
status:
  - planned
tags:
  - hls4ml
  - implementation
  - training
last_modified: 2026-05-22
author: mbvalentin
---
# Phase 5: ENABOL Bridge and Validation
<PageMeta />
---

<TBox type="summary" title="Goal">

Make current ENABOL produce trainable hls4ml configurations and reference traces, then use CSIM to compare firmware behavior against software behavior.

</TBox>

## Linked Tasks

- [HLS4ML-012](/docs/status/tasks?query=HLS4ML-012): ENABOL bridge.
- [HLS4ML-013](/docs/status/tasks?query=HLS4ML-013): one-layer Dense CSIM.
- [HLS4ML-014](/docs/status/tasks?query=HLS4ML-014): two-Dense validation.

## Notebook Entry Point

The first working notebook is:

```text
/Users/mbvalentin/scripts/ENABOL/workspace/compilation/pipe000_first_compilation_pipeline.ipynb
```

It should become the small, reproducible path from current ENABOL model construction to hls4ml config export and compilation.

## Bridge Responsibilities

The ENABOL bridge should:

- build a simple model and dataset
- apply a requested toolchain profile through `enabol.toolchain.toolchain_environment()`
- compute precision recommendations
- emit `Model.Training`
- emit per-layer trainable precision overrides
- export ground-truth inputs for the testbench
- export reference traces for forward output, loss, gradients, alpha, and updated parameters

## Validation Policy

Start with local CSIM and software trace comparison. Vitis/Vivado synthesis can happen later on the server; it should not block the first config and codegen work on the Mac.

Toolchain setup is tracked separately in [Toolchain Profiles](/docs/implementation/toolchain). The notebook should not set compiler paths directly; it should pass a profile name into the compile bridge once `enabol.compile` is implemented.

## Compile API

The first bridge entry point is:

```python
from enabol.compile import compile

hls_model, hls_config = compile(
    model=model,
    dataset=dataset,
    backend="Vitis",
    toolchain="auto",
    output_dir="workspace/compilation/pipe000_first_compilation_pipeline",
    project_name="pipe000_dense",
    part="xcku035-fbva676-2-e",
    io_type="io_parallel",
    strategy="Latency",
    reuse_factor=1,
    trainable=True,
    optimizer="sgd",
    learning_rate=0.01,
    batch_size=1,              # rounded up to a power of two in Model.Training.BatchSize
    epochs=2,                  # CSIM trainable testbench epochs
    shuffle=True,              # shuffle samples once per epoch in CSIM
    shuffle_seed=13,           # deterministic bring-up shuffling
    log_every=1,               # console logging cadence in global train steps
    loss="half_mse",
    controller="ctrl-gt-order-0",
    precision=precision_dict,
    write=True,
    compile_cpp=False,
    build=False,
)
```

`toolchain="auto"` is applied only when `build=True`. This lets Mac notebooks generate and inspect the hls4ml project without failing on Kona-only paths. On Kona, setting `build=True` activates the default 2024.1 toolchain profile before hls4ml calls `vitis-run`.

`build=True` also forces a project write before launching Vitis. This is necessary because ENABOL stages dataset files under the output directory before conversion; that makes the directory exist, so hls4ml's native `ModelGraph.build()` will not auto-write `build_prj.tcl` by directory-existence alone.

For the first notebook trace, use an explicit `PrecisionDict`:

```python
from enabol.precision import PrecisionDict

precision_dict = PrecisionDict({
    "input": {
        "value": "ap_fixed<16,8,AP_RND,AP_SAT>",
    },
    "dense0": {
        "weight": "ap_fixed<16,6,AP_RND,AP_SAT>",
        "activation": "ap_fixed<16,8,AP_RND,AP_SAT>",
        "gradient": "ap_fixed<20,8,AP_RND,AP_SAT>",
        "update": "ap_fixed<20,6,AP_RND,AP_SAT>",
        "accumulator": "ap_fixed<28,14,AP_RND,AP_SAT>",
    },
    "loss": {
        "value": "ap_fixed<32,16,AP_RND,AP_SAT>",
    },
})
```

The bridge expands these semantic fields into hls4ml trainable fields:

| ENABOL field | hls4ml trainable fields |
|---|---|
| `loss.value` | `Model.Training.Precision.loss` |
| `dense0.gradient` | `grad_in`, `grad_out`, `weight_grad`, `bias_grad`, `loss_grad` |
| `dense0.update` | `raw_update`, `update`, `optimizer_state` |
| `dense0.accumulator` | `gradient_accum`, `controller_metric` |

If no `PrecisionDict` is passed, the bridge emits conservative placeholder trainable defaults. Those defaults are only for bring-up; automatic precision inference is tracked separately.

Batch size is emitted as a power-of-two hardware cadence. `enabol.compile.build_hls_config(...)` rounds the requested `batch_size` up, writes the rounded value to `Model.Training.BatchSize`, records the original value in `Model.Training.BatchSizeRequested`, and writes `Model.Training.BatchSizeLog2` for shift-based gradient averaging in trainable kernels.

Static learning rate is also part of the generated trainable configuration. For the first path, ENABOL emits `Model.Training.Optimizer.LearningRate`, hls4ml writes it as `trainable_configN::learning_rate` in `parameters.h`, and the SGD call casts that config value into `learning_rate_t`. We are intentionally not using a top-level learning-rate port yet; dynamic learning-rate IO remains available through `LearningRateInput` but is deferred until the fixed path passes CSIM.

## Trainable CSIM Testbench

When `Model.Training.Trainable` is true, the Vivado/Vitis writer now emits a trainable-specific `*_test.cpp` through the normal hls4ml testbench path. This means `hls_model.build(csim=True)` still uses the standard project script, but the generated C++ testbench trains instead of running a single inference pass.

The generated trainable testbench:

- loads `tb_data/tb_input_features.dat` and `tb_data/tb_output_predictions.dat`
- indexes all samples once before training
- shuffles sample order once per epoch when `Model.Training.Shuffle` is true
- runs `Model.Training.Epochs` epochs
- computes `reset_accumulators` and `batch_end` from the rounded power-of-two `BatchSize`
- writes predictions to the normal CSIM result log
- writes loss to `tb_data/training/loss.dat`
- writes alpha to `tb_data/training/alpha.dat`
- writes controller diagnostics to `tb_data/training/controller.dat`
- writes trainable weights and biases once per epoch to per-layer files under `tb_data/training/<layer>/`
- prints compact progress every `Model.Training.LogEvery` global train steps

The current logging is intentionally testbench-side. The firmware kernels stay clean except for the optional `HLS4ML_TRAINABLE_TRACE` hooks already fenced out of synthesis.

Trace files are comma-delimited `.dat` files. They intentionally repeat the trace index columns in every file:

```text
# ----
# Trace: loss
# Generated by enabol+hls4ml-trainable
# ENABOL version: 0.1.0
# hls4ml-trainable version: 0.0.0a
# User: ...
# Host: ...
# Date: ...
# Project: ...
# Controller: ...
# Optimizer: ...
# LearningRate: ...
# BatchSize: ...
# Epochs: ...
# ---
epoch,sample,global_step,sample_index,loss
1,1,0,37,2.02597
```

The controller trace uses the same index columns and then appends the global controller metrics:

```text
epoch,sample,global_step,sample_index,dtheta_sq,dgrad_sq,lhs_sq,rhs_sq,alpha_feasible,alpha_state
1,1,0,37,0.25,0.5,0.00005,0.5625,0.875,0.9875
```

For GT-0 and GT-1, `dtheta_sq` and `dgrad_sq` are the globally summed curvature-sensor terms. `lhs_sq` and `rhs_sq` are the two sides of the division-free candidate-search inequality. `alpha_feasible` is the raw accepted candidate, and `alpha_state` is the final alpha emitted by the controller. For `CTRL-NONE`, the file still exists with zero curvature terms and unit alpha-state values.

Layer-local parameter traces live below the global traces:

```text
tb_data/training/dense0/weights.dat
tb_data/training/dense0/biases.dat
```

Dense weight columns are named `weight_<input>_<output>` and follow the same row-major flattening as hls4ml dense kernels: flat index `input * n_out + output`.

`epoch` and `sample` are one-based for readability. `global_step` and `sample_index` remain zero-based because they are counters and indexes into the original `.dat` dataset files. Repeating these columns in every trace file is deliberate: traces can be sparse. The current path logs loss and alpha every sample, and logs parameter traces once per epoch with `sample_index=-1` because that row describes epoch-end model state rather than a single dataset row.

The generated console output is similarly compact:

```text
Epoch [1/2] - sample 1/128: loss 2.02597, alpha: 1
```

ENABOL reads these traces through `TestbenchData`:

```python
from enabol import TestbenchData

tb = TestbenchData.from_dir(hls_model.config.get_output_dir())
tb.plot_training(window_size=30)
```

`TestbenchData.from_dir(...)` accepts the hls4ml output directory, `tb_data`, or `tb_data/training`. The returned object keeps top-level traces such as `loss`, `alpha`, and the columns from `controller.dat` at `tb.frame`, parsed display metadata at `tb.metadata`, and per-trace metadata at `tb.metadata_by_trace`.

Per-layer parameter traces are loaded into `tb.layers` when `load_weights` is enabled. For example, `tb.layers["dense0"].weights` holds the raw weight evolution and `tb.layers["dense0"].stats` holds scalar summaries such as `weights.mean`, `weights.std`, and `weights.norm_l2`. `tb.stats_frame` merges those layer statistics with names such as `dense0.weights.mean`, and `tb.scalar_frame` combines top-level traces plus layer statistics for plotting.

The generated metadata comments are emitted by the generated C++ CSIM testbench. Static values such as project, backend, controller, optimizer, loss, and learning rate are injected by the hls4ml writer when it writes `*_test.cpp`. Runtime values such as `User`, `Host`, and `Date` are read by that C++ testbench during CSIM from the process environment and local clock.

Controller IDs are normalized before trainable writer dispatch. For the currently wired no-controller path, spellings such as `none`, `ctrl-none`, `ctrl_none`, and `CTRL-NONE` resolve to the same `CTRL-NONE` behavior. This keeps ENABOL-facing names and generated hls4ml config IDs from fighting over capitalization or separator conventions.

## Bridge Change Log

| Task | Date | Files | Summary |
|---|---|---|---|
| [HLS4ML-012](/docs/status/tasks?query=HLS4ML-012) | 2026-05-22 | `enabol/compile.py`, `enabol/__init__.py`, `tests/test_compile.py` | Added `compile(...)`, `build_hls_config(...)`, trainability resolution, `PrecisionDict` mapping into ordinary and trainable hls4ml precision fields, dataset testbench export, internal toolchain context use for HLS build, and focused config tests. |
| [HLS4ML-012](/docs/status/tasks?query=HLS4ML-012) | 2026-05-22 | `enabol/compile.py`, `tests/test_compile.py` | Added default trainable precision emission, expanded semantic precision aliases, and changed `trainable=True` to apply only to parameterized layers such as Dense. |
| [HLS4ML-012](/docs/status/tasks?query=HLS4ML-012) | 2026-05-22 | `enabol/precision.py`, `enabol/compile.py`, `tests/test_precision.py` | Moved hls4ml precision translation out of `compile.py`; `compile.py` now calls `apply_hls_precision_config(...)` from `precision.py`. |
| [HLS4ML-020](/docs/status/tasks?query=HLS4ML-020) | 2026-05-23 | `hls4ml/writer/vivado_writer.py`, `hls4ml/model/graph.py`, `enabol/compile.py`, `test/pytest/test_trainable_config.py`, `tests/test_compile.py` | Replaced literal trainable buffer dimensions with generated config constants, moved static learning rate into the trainable config struct, and added the generated trainable epoch/shuffle/loss-log CSIM testbench. |
| [HLS4ML-020](/docs/status/tasks?query=HLS4ML-020) | 2026-05-23 | `enabol/compile.py` | Fixed the bridge build path so `build=True` writes the hls4ml project before Vitis runs, even when dataset staging already created the output directory. |
| [HLS4ML-037](/docs/status/tasks?query=HLS4ML-037) | 2026-05-23 | `hls4ml/writer/vivado_writer.py`, `hls4ml/model/graph.py`, `enabol/compile.py` | Replaced verbose prediction printing with compact progress logs and added metadata-bearing `.dat` traces under `tb_data/training/` for loss and alpha. |
| [ENB-024](/docs/status/tasks?query=ENB-024) | 2026-05-23 | `enabol/testbench.py`, `enabol/history.py`, `tests/test_history.py` | Added ENABOL-side testbench trace loading and plotting through `TestbenchData.from_dir(...)` and `TestbenchData.plot_training(...)`, while keeping `FitHistory` scoped to software training. |
| [HLS4ML-037](/docs/status/tasks?query=HLS4ML-037) | 2026-05-23 | `hls4ml/writer/vivado_writer.py`, `test/pytest/test_trainable_config.py` | Added per-layer epoch-level parameter traces, fixed testbench weight visibility through `extern`, and normalized accepted no-controller spellings before writer dispatch. |
| [ENB-024](/docs/status/tasks?query=ENB-024) | 2026-05-23 | `enabol/testbench.py`, `tests/test_history.py` | Added recursive nested-trace loading, sparse-trace plotting support, table-style `TestbenchData.__repr__`, and per-trace metadata storage. |
| [ENB-025](/docs/status/tasks?query=ENB-025) | 2026-05-23 | `enabol/testbench.py`, `tests/test_history.py` | Split top-level traces from raw per-layer parameter traces, added `load_weights`, and exposed per-layer parameter summary statistics through `tb.stats_frame` and `tb.scalar_frame`. |
| [HLS4ML-038](/docs/status/tasks?query=HLS4ML-038) | 2026-05-23 | `hls4ml/writer/vivado_writer.py`, `hls4ml/templates/vivado/trainable/controllers/global_throttle.h`, `enabol/testbench.py`, `tests/test_history.py` | Added explicit controller metric outputs and `tb_data/training/controller.dat`, then documented and tested ENABOL-side loading as top-level scalar traces. |

The first validation target should be:

```text
Dense(1 -> 1), no bias optional variant, MSE or half-MSE, SGD, CTRL-NONE
```

Then add:

```text
Dense(1 -> 1), bias, CTRL-GT-ORDER-0
Dense -> activation -> Dense
```

## Exit Criteria

Phase 5 is done when current ENABOL can generate a trainable hls4ml project and compare CSIM outputs against a saved software trace without manual patching of generated firmware.
