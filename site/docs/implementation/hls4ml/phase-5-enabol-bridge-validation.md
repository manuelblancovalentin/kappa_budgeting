---
title: "Phase 5: ENABOL Bridge and Validation"
sidebar_label: "Phase 5: Bridge + Validation"
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
    batch_size=1,
    loss="half_mse",
    controller="ctrl-gt-order-0",
    write=True,
    compile_cpp=False,
    build=False,
)
```

`toolchain="auto"` is applied only when `build=True`. This lets Mac notebooks generate and inspect the hls4ml project without failing on Kona-only paths. On Kona, setting `build=True` activates the default 2024.1 toolchain profile before hls4ml calls `vitis-run`.

## Bridge Change Log

| Task | Date | Files | Summary |
|---|---|---|---|
| [HLS4ML-012](/docs/status/tasks?query=HLS4ML-012) | 2026-05-22 | `enabol/compile.py`, `enabol/__init__.py`, `tests/test_compile.py` | Added `compile(...)`, `build_hls_config(...)`, trainability resolution, `PrecisionDict` mapping into ordinary and trainable hls4ml precision fields, dataset testbench export, internal toolchain context use for HLS build, and focused config tests. |

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
