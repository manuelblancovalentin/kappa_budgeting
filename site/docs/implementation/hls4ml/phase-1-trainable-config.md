---
title: "Phase 1: Trainable Configuration"
sidebar_label: "Phase 1: Config"
status:
  - complete
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

## Explicit Schema

`HLS4ML-001` defines the first normalized schema as:

```yaml
HLSConfig:
  Model:
    Precision: ap_fixed<16,6>
    ReuseFactor: 1
    Training:
      Trainable: true
      BatchSize: 1
      Loss:
        Kind: half_mse
        Output: output
      Optimizer:
        Kind: sgd
        LearningRate: 0.01
        LearningRateInput: null
      Controller:
        Kind: none
        SafetyBudget:
          Enabled: false
      Precision:
        loss: ap_fixed<24,10>
        loss_grad: ap_fixed<18,8>
        grad_in: ap_fixed<18,8>
        grad_out: ap_fixed<18,8>
        weight_grad: ap_fixed<18,8>
        bias_grad: ap_fixed<18,8>
        gradient_accum: ap_fixed<24,10>
        raw_update: ap_fixed<18,8>
        update: ap_fixed<18,8>
        optimizer_state: ap_fixed<18,8>
        controller_metric: ap_fixed<24,10>
        alpha: ap_fixed<16,4>
  LayerName:
    dense:
      Training:
        Trainable: true
        Precision:
          grad_in: ap_fixed<18,8>
```

The model-level `Training` block is optional. If it is absent, hls4ml behaves as inference-only. A boolean shorthand is also accepted:

```yaml
HLSConfig:
  Model:
    Training: true
```

That shorthand means `Trainable: true` with default loss, optimizer, controller, batch-size, and precision sections.

## Accessor Contract

`HLS4ML-001` added the following accessors in `hls4ml/model/graph.py`:

| Accessor | Returns | Purpose |
|---|---|---|
| `get_training_config()` | A copy of normalized `Model.Training`. | The single entry point for model-level training settings. |
| `is_trainable()` | Boolean. | Fast check for whether trainable code generation is enabled. |
| `get_loss_config()` | A copy of `Model.Training.Loss`. | Loss endpoint passes use this instead of reading nested dicts. |
| `get_optimizer_config()` | A copy of `Model.Training.Optimizer`. | Optimizer/update passes use this for SGD and learning-rate policy. |
| `get_controller_config()` | A copy of `Model.Training.Controller`. | Controller passes use this for global throttle and safety-budget settings. |
| `get_trainable_precision_config()` | A copy of `Model.Training.Precision`. | Precision passes use this for trainable-wide defaults. |
| `get_trainable_precision_fields()` | Tuple of recognized trainable precision field names. | ENABOL and validation code can agree on field names. |
| `get_layer_trainable_precision_config(layer)` | Merged copy of model-level trainable precision plus layer overrides. | Precision resolution uses this before creating layer attributes. |
| `get_trainable_precision(layer, var)` | Converted precision object and generated typedef name. | Layer initialization uses this to build `NamedType` attributes. |
| `get_layer_trainable_config(layer)` | A copy of layer-level `Training` settings with default `Trainable` filled from model state. | Layer passes use this to resolve per-layer trainability and overrides. |

All accessors return copies for dictionary sections. This prevents a caller from mutating global configuration accidentally while preparing graph attributes.

## Current Decisions

Learning rate should be configurable as static first, with a path to dynamic input later. We will need learning rate during precision planning, so the config must always declare the intended policy.

Batch size should be explicit in config for the same reason. Even if the first build uses one sample at a time, gradient accumulation and fixed-point ranges depend on this value.

Safety budget fields should exist, but the first correctness target can set them inactive. This keeps the κ-budgeting connection represented without blocking the global throttling path.

Precision fields should include at least weight, bias, result, gradient input, gradient output, weight gradient, bias gradient, raw update, applied update, gradient accumulator, loss, loss gradient, controller metric, alpha, and optimizer state. Some may alias in early configs, but they should be named separately because they answer different range questions.

## Accessors To Add

The first accessor set is now implemented:

```text
is_trainable()
get_training_config()
get_loss_config()
get_optimizer_config()
get_controller_config()
get_trainable_precision_config()
get_trainable_precision_fields()
get_layer_trainable_precision_config(layer)
get_trainable_precision(layer, var)
get_layer_trainable_config(layer)
```

These methods should normalize defaults and make missing/disabled trainable mode easy for existing inference-only paths.

## Layer Attribute Resolution

`HLS4ML-003` connects the schema to the graph. During `Layer` initialization, hls4ml now calls `_set_trainable_attributes()` after ordinary layer config has been applied and before `initialize()`.

That method does three things:

| Step | Behavior |
|---|---|
| Resolve trainability | Calls `HLSConfig.get_layer_trainable_config(self)` and writes a boolean `trainable` attribute on the layer. |
| Skip disabled layers | If `trainable` is false, no gradient/update/controller types are attached. |
| Attach trainable types | For each configured recognized trainable precision field, calls `get_trainable_precision(self, field)` and writes a `NamedType` attribute named `<field>_t`. |

For example, this config:

```yaml
HLSConfig:
  Model:
    Training:
      Trainable: true
      Precision:
        grad_in: ap_fixed<18,8>
        raw_update: ap_fixed<20,10>
        alpha: ap_fixed<16,4>
  LayerName:
    dense:
      Training:
        Precision:
          grad_in: ap_fixed<12,4>
```

creates these attributes on the `dense` layer:

```text
trainable = True
grad_in_t = NamedType("dense_grad_in_t", ap_fixed<12,4>)
raw_update_t = NamedType("dense_raw_update_t", ap_fixed<20,10>)
alpha_t = NamedType("dense_alpha_t", ap_fixed<16,4>)
```

This is the first concrete handoff from config into hls4ml graph metadata. Later backend template passes should consume these attributes directly instead of inferring gradient/update types from `parameters.h`.

## Change Log

| Task | Date | Files | Summary |
|---|---|---|---|
| [HLS4ML-001](/docs/status/tasks?query=HLS4ML-001) | 2026-05-22 | `hls4ml/model/graph.py`, `test/pytest/test_trainable_config.py` | Added normalized `Model.Training` defaults, training/loss/optimizer/controller/precision accessors, layer-level trainable config accessor, and unit tests for defaults, overrides, copy behavior, and boolean shorthand. |
| [HLS4ML-003](/docs/status/tasks?query=HLS4ML-003) | 2026-05-22 | `hls4ml/model/graph.py`, `hls4ml/model/layers.py`, `test/pytest/test_trainable_config.py` | Added merged trainable precision resolution, converted trainable precision lookup, and `Layer._set_trainable_attributes()` so configured fields become graph `NamedType` attributes like `grad_in_t`, `raw_update_t`, and `alpha_t`. |
| [HLS4ML-012](/docs/status/tasks?query=HLS4ML-012) | 2026-05-22 | `enabol/compile.py`, `enabol/precision.py`, `tests/test_compile.py`, `tests/test_precision.py` | Added ENABOL-side `compile(...)`, hls4ml config generation, `Model.Training` emission, semantic precision translation, default trainable precision fill, dataset testbench staging, and write-only hls4ml project generation. |

## Exit Criteria

Phase 1 is done. hls4ml can load a trainable config, expose stable accessor methods, attach configured trainable precision as layer attributes, and ENABOL can emit that schema into a generated hls4ml project.
