---
title: "Phase 1: Trainable Configuration"
sidebar_label: "Phase 1: Config"
status:
  - inprogress
tags:
  - hls4ml
  - implementation
  - training
last_modified: 2026-05-22
author: mbvalentin
---
# Phase 1: Trainable Configuration
<PageMeta />
---

<TBox type="summary" title="Goal">

Add a first-class trainable schema to hls4ml configuration so later passes can ask explicit questions instead of rediscovering training intent from layer names or generated code.

</TBox>

## Linked Tasks

- [HLS4ML-001](/docs/status/tasks?query=HLS4ML-001): define the schema and accessors.
- [HLS4ML-003](/docs/status/tasks?query=HLS4ML-003): connect config precision fields to layer attributes.
- [HLS4ML-012](/docs/status/tasks?query=HLS4ML-012): emit this schema from ENABOL.

## Proposed Schema Boundary

The model-level block should live under `Model.Training` in the hls4ml config. It should contain:

| Field group | Purpose |
|---|---|
| `Trainable` | Enables training code generation. |
| `Loss` | Declares loss kind, endpoint mapping, ground-truth ports, and loss scale convention. |
| `Optimizer` | Declares optimizer kind, learning-rate policy, batch-size policy, and update accumulation behavior. |
| `Controller` | Declares global throttling controller kind, alpha rails, state precisions, curvature metrics, and optional safety-budget coupling. |
| `Precision` | Declares trainable-wide default types for losses, gradients, updates, accumulators, controller metrics, and optimizer state. |

Layer-level trainable settings should still live at layer scope. That is where per-layer precision overrides and trainability belong.

## Current Decisions

Learning rate should be configurable as static first, with a path to dynamic input later. We will need learning rate during precision planning, so the config must always declare the intended policy.

Batch size should be explicit in config for the same reason. Even if the first build uses one sample at a time, gradient accumulation and fixed-point ranges depend on this value.

Safety budget fields should exist, but the first correctness target can set them inactive. This keeps the κ-budgeting connection represented without blocking the global throttling path.

Precision fields should include at least weight, bias, result, gradient input, gradient output, weight gradient, bias gradient, raw update, applied update, gradient accumulator, loss, loss gradient, controller metric, alpha, and optimizer state. Some may alias in early configs, but they should be named separately because they answer different range questions.

## Accessors To Add

`HLSConfig` should gain narrow helper methods instead of callers reading nested dictionaries directly:

```text
is_trainable()
get_training_config()
get_loss_config()
get_optimizer_config()
get_controller_config()
get_trainable_precision_config()
get_layer_trainable_config(layer_name)
```

These methods should normalize defaults and make missing/disabled trainable mode easy for existing inference-only paths.

## Exit Criteria

Phase 1 is done when hls4ml can load a trainable config, expose stable accessor methods, and fail early on unsupported or ambiguous training settings without generating C++.
