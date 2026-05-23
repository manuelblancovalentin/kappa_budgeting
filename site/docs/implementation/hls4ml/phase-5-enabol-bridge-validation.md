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
- writes `tb_data/trainable_loss.log` with `epoch`, `step`, `sample`, `global_step`, `loss`, and `alpha`
- prints progress every `Model.Training.LogEvery` global train steps

The current logging is intentionally testbench-side. The firmware kernels stay clean except for the optional `HLS4ML_TRAINABLE_TRACE` hooks already fenced out of synthesis.

## Bridge Change Log

| Task | Date | Files | Summary |
|---|---|---|---|
| [HLS4ML-012](/docs/status/tasks?query=HLS4ML-012) | 2026-05-22 | `enabol/compile.py`, `enabol/__init__.py`, `tests/test_compile.py` | Added `compile(...)`, `build_hls_config(...)`, trainability resolution, `PrecisionDict` mapping into ordinary and trainable hls4ml precision fields, dataset testbench export, internal toolchain context use for HLS build, and focused config tests. |
| [HLS4ML-012](/docs/status/tasks?query=HLS4ML-012) | 2026-05-22 | `enabol/compile.py`, `tests/test_compile.py` | Added default trainable precision emission, expanded semantic precision aliases, and changed `trainable=True` to apply only to parameterized layers such as Dense. |
| [HLS4ML-012](/docs/status/tasks?query=HLS4ML-012) | 2026-05-22 | `enabol/precision.py`, `enabol/compile.py`, `tests/test_precision.py` | Moved hls4ml precision translation out of `compile.py`; `compile.py` now calls `apply_hls_precision_config(...)` from `precision.py`. |
| [HLS4ML-020](/docs/status/tasks?query=HLS4ML-020) | 2026-05-23 | `hls4ml/writer/vivado_writer.py`, `hls4ml/model/graph.py`, `enabol/compile.py`, `test/pytest/test_trainable_config.py`, `tests/test_compile.py` | Replaced literal trainable buffer dimensions with generated config constants, moved static learning rate into the trainable config struct, and added the generated trainable epoch/shuffle/loss-log CSIM testbench. |

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
