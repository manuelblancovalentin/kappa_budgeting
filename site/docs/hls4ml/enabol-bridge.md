---
id: enabol-bridge
title: "ENABOL Bridge"
sidebar_label: "ENABOL Bridge"
status:
  - preliminary
  - inprogress
tags:
  - hls4ml
  - implementation
  - precision
  - training
last_modified: 2026-05-21
author: mbvalentin
---
# ENABOL Bridge
<PageMeta />
---

<TBox type="summary" title="Role">

ENABOL should produce a clean hls4ml configuration. It should not patch generated firmware after hls4ml emits it.

</TBox>

## Old Bridge Behavior

The deprecated `old_enabol/compile.py` did the job of connecting ENABOL to hls4ml:

1. validate backend, IO type, strategy, reuse factor, optimizer, and batch size;
2. call `hls4ml.utils.config_from_keras_model()`;
3. set `IOType`, `Backend`, `XilinxPart`, strategy, and reuse factor;
4. resolve layer trainability from bool/list/int input;
5. write `Model.Trainable` and per-layer `Trainable`;
6. create optimizer config and learning rate;
7. write batch size, epochs, trace settings, and kappa precision;
8. call ENABOL precision assignment;
9. call `hls4ml.converters.convert_from_keras_model()`;
10. compile and optionally write dataset files.

That structure is still useful. The part to avoid is relying on hls4ml writer hacks downstream.

## Current ENABOL Inputs

Current ENABOL has better internal concepts:

| ENABOL object | Relevant information |
|---|---|
| `BaseModel` / model wrapper | Keras model, loss, optimizer metadata. |
| `PrecisionDict` | Semantic precision fields: input, loss, weight, bias, activation, gradient, update, accumulator. |
| `InstrumentedTrainer` | Software reference loop for gradients, updates, metrics, and controller behavior. |
| `Controller` classes | Global throttle controller formulas and state. |
| `Dataset` classes | Testbench inputs and expected outputs. |

The bridge should map these to hls4ml config.

## Proposed Bridge Output

ENABOL should produce:

```yaml
Backend: Vivado
IOType: io_parallel
HLSConfig:
  Model:
    Precision: ...
    ReuseFactor: ...
    Strategy: ...
    Trainable: true
    Training:
      Losses:
        - name: half_mse
          output: output
      Optimizer:
        name: sgd
        learning_rate: 0.01
      Controller:
        name: global_throttle_order_0
        chi: ...
        alpha_min: ...
        alpha_max: ...
        eps: ...
      BatchSize: 32
  LayerName:
    dense:
      Trainable: true
      Precision:
        weight: ...
        bias: ...
        result: ...
        grad_in: ...
        grad_out: ...
        update: ...
        gradient_accum: ...
```

This config should be enough for hls4ml to generate trainable firmware without ENABOL inspecting C++.

## Trainability Policy

Keep the convenient ENABOL input styles:

| Input | Meaning |
|---|---|
| `trainable=True` | Train all supported weight layers. |
| `trainable=False` | Generate inference-only firmware. |
| `trainable=["dense_1", "dense_2"]` | Train only named layers. |
| `trainable=2` | Train last two eligible weight layers. |

Normalize this before passing config to hls4ml. hls4ml should receive explicit per-layer `Trainable` booleans.

## Precision Policy

ENABOL should remain semantic:

```python
precision.dtype("dense", "gradient")
precision.dtype("dense", "update")
precision.dtype("loss", "value")
```

The bridge should translate those fields to hls4ml trainable precision names. Missing fields should follow documented defaults, not silent guessing.

## Recommended Page Split

Keep this hls4ml section as the design/reference map. Later, when implementation starts, add a second implementation-log subsection such as:

```text
hls4ml/
  index.md
  hlsconfig.md
  ...
  implementation-plan.md
  change-log.md
```

That keeps stable architecture notes separate from the messier sequence of commits, experiments, and broken builds.

