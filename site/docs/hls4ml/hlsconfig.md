---
id: hlsconfig
title: "HLSConfig"
sidebar_label: "HLSConfig"
status:
  - preliminary
  - inprogress
tags:
  - hls4ml
  - implementation
  - training
last_modified: 2026-05-21
author: mbvalentin
---
# HLSConfig
<PageMeta />
---

<TBox type="summary" title="Role">

`HLSConfig` is the parsed configuration object carried by `ModelGraph`. It is the right place to make trainability explicit before any backend pass or writer sees the model.

</TBox>

## What hls4ml Does Today

`HLSConfig` lives in `hls4ml/model/graph.py`. It wraps the user config dictionary and normalizes the subset hls4ml needs repeatedly:

| Method or field | Meaning |
|---|---|
| `self.backend` | Backend object selected from `Backend`, defaulting to Vivado. |
| `get_config_value()` | Reads top-level keys such as `ProjectName`, `OutputDir`, `IOType`, and `Backend`. |
| `get_layer_config_value()` | Looks up a setting by priority: `LayerName`, then `LayerType`, then `Model`. |
| `get_layer_config()` | Merges type-level and name-level layer configuration. |
| `get_precision()` | Resolves precision for a layer variable, falling back through layer-name, layer-type, model variable, and model default. |
| `get_reuse_factor()` | Resolves reuse factor by layer name, layer type, then model. |
| `_parse_hls_config()` | Converts the raw `HLSConfig` dictionary into internal lookup tables for precision, reuse, target cycles, strategy, convolution implementation, compression, pipeline settings, flows, and optimizers. |

The important current shape is:

```yaml
HLSConfig:
  Model:
    Precision: ap_fixed<16,6>
    ReuseFactor: 1
    Strategy: Latency
  LayerType:
    Dense:
      Precision:
        weight: ap_fixed<16,6>
  LayerName:
    dense_0:
      Precision:
        result: ap_fixed<18,8>
```

Layer objects later call `model.config.get_layer_config(self)`. Any matching config keys are converted to snake case and copied into the layer attribute dictionary. Precision keys ending in `_t` can become hls4ml `NamedType` objects.

## Why This Matters For Training

The old trainable fork repeatedly asked the raw config dictionary questions like:

```python
model.config.get_config_value("HLSConfig").get("Model", {}).get("Trainable", False)
```

That works, but it spreads policy across the writer. It also forces later code to infer missing structure from C++ strings: which layer is trainable, what the gradient type is, where `weight_t` and `bias_t` live, which loss feeds which output, and how optimizer state should be typed.

Training needs stronger configuration boundaries:

| Configuration | Scope |
|---|---|
| `Model.Trainable` | Enables trainable code generation globally. |
| `Model.Training.Losses` | Declares one loss per output, with broadcast rules only during parsing. |
| `Model.Training.BatchSize` | Defines accumulation/update cadence and batch log2 constants. |
| `Model.Training.Optimizer` | Defines raw update rule and optimizer state. |
| `Model.Training.Controller` | Defines global throttle controller and state. |
| `LayerName.*.Trainable` | Defines whether a specific layer participates in the backward/update chain. |
| `LayerName.*.Precision` | Defines forward, gradient, accumulator, update, and optimizer-state types. |

## Proposed Schema

Use a nested training block instead of placing all training fields directly in `Model`:

```yaml
HLSConfig:
  Model:
    Precision: ap_fixed<16,6>
    ReuseFactor: 1
    Strategy: Latency
    Trainable: true
    Training:
      BatchSize: 32
      Losses:
        - name: mse
          output: dense_out
      Optimizer:
        name: sgd
        learning_rate: 0.01
      Controller:
        name: global_throttle_order_0
        chi: 1.0
        alpha_min: 0.0
        alpha_max: 1.0
        eps: 1.0e-8
  LayerName:
    dense:
      Trainable: true
      Precision:
        weight: ap_fixed<16,6>
        bias: ap_fixed<16,6>
        result: ap_fixed<16,6>
        grad_in: ap_fixed<16,6>
        grad_out: ap_fixed<16,6>
        update: ap_fixed<16,6>
        gradient_accum: ap_fixed<24,12>
```

For backward compatibility during development, the parser can accept old aliases such as `Model.Losses`, `Model.BatchSize`, and `Model.LearningRate`, then normalize them into `Model.Training`.

## Recommended hls4ml Changes

Add parsed accessors to `HLSConfig`:

| Accessor | Purpose |
|---|---|
| `is_trainable()` | Returns global trainability. |
| `get_training_config()` | Returns normalized training config. |
| `get_loss_config()` | Returns normalized output loss endpoints. |
| `get_optimizer_config()` | Returns normalized optimizer parameters. |
| `get_controller_config()` | Returns normalized global throttle settings. |
| `is_layer_trainable(layer)` | Resolves layer trainability. |
| `get_training_precision(layer, field)` | Resolves trainable-specific precision fields. |

These accessors should not emit C++. They should only normalize and validate configuration.

## Validation Rules

Trainable config should fail early when:

- `Trainable: true` but no losses are defined.
- The number of losses does not match outputs after broadcast normalization.
- A trainable layer has weights but no update/gradient precision policy.
- A requested loss has no Vivado loss template.
- A controller requires state precision but no default can be resolved.
- `IOType` is unsupported by the first training implementation.

Early validation keeps writer code simple and makes failures readable.

