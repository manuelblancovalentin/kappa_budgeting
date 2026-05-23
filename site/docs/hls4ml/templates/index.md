---
title: "Trainable Vivado Templates"
sidebar_label: "Templates"
status:
  - inprogress
tags:
  - hls4ml
  - trainable
  - templates
last_modified: 2026-05-22
author: mbvalentin
---
# Trainable Vivado Templates
<PageMeta />
---

<TBox type="summary" title="What this section covers">

This section documents the static C++ headers under `hls4ml/templates/vivado/trainable`. These are the synthesizable building blocks for the first ENABOL/hls4ml-trainable path: loss endpoint, Dense backpass, raw SGD update proposal, and a no-throttle controller baseline.

</TBox>

## Directory Layout

```text
hls4ml/templates/vivado/trainable
├── backprop
│   └── nnet_dense_backprop.h
├── common
│   └── trainable_trace.h
├── controllers
│   └── global_throttle.h
├── losses
│   └── mse.h
└── optimizers
    └── sgd.h
```

## Pages

| Area | Header | Purpose |
|---|---|---|
| [Common trace helpers](./common-trace.md) | `common/trainable_trace.h` | Macro-gated trainable tracing through hls4ml's native trace storage. |
| [Dense backprop](./backprop-dense.md) | `backprop/nnet_dense_backprop.h` | Computes `dL/dx`, accumulates batch gradients, and emits averaged parameter gradients. |
| [Squared-error losses](./losses-mse.md) | `losses/mse.h` | Implements `mse` and `half_mse` from the same scalar/gradient core. |
| [SGD optimizer](./optimizers-sgd.md) | `optimizers/sgd.h` | Proposes raw `-learning_rate * gradient` updates without mutating parameters. |
| [Global throttle controllers](./controllers-global-throttle.md) | `controllers/global_throttle.h` | Provides `CTRL-NONE` and the shared alpha-scaled Dense update application helper. |

## Design Rule

The headers are deliberately static and dumb. They should consume `CONFIG_T` typedefs, dimensions, constants, and trace names emitted by hls4ml writer/config passes. They should not inspect Python-side dictionaries, infer layer names, or reproduce writer logic.

The generated firmware will decide which headers to include and which config structs to emit when `HLSConfig.Model.Training.Trainable=True`.

