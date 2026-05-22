---
title: "Phase 2: Graph and Flow Integration"
sidebar_label: "🚧 Phase 2: Graph + Flow"
status:
  - inprogress
tags:
  - hls4ml
  - implementation
  - architecture
last_modified: 2026-05-22
author: mbvalentin
---
# Phase 2: Graph and Flow Integration
<PageMeta />
---

<TBox type="summary" title="Goal">

Represent training as hls4ml graph metadata and backend flow work, so the writer receives resolved training artifacts instead of deriving them.

</TBox>

## Linked Tasks

- [HLS4ML-002](/docs/status/tasks?query=HLS4ML-002): trainable flow and validation passes.
- [HLS4ML-003](/docs/status/tasks?query=HLS4ML-003): layer attributes and precision resolution.
- [HLS4ML-006](/docs/status/tasks?query=HLS4ML-006): loss endpoint metadata.

## Flow Shape

The trainable flow should run only when `Model.Training.Trainable` is enabled. It should be registered through the backend flow system rather than invoked manually from the writer.

The flow should perform these jobs:

| Pass | Job |
|---|---|
| trainable config validation | Reject unsupported layer types, missing loss endpoints, invalid controller names, or ambiguous output mappings. |
| endpoint resolution | Attach ground-truth, loss, loss-gradient seed, and effective output metadata. |
| reverse traversal | Produce a deterministic backward layer order for supported sequential graphs. |
| precision resolution | Attach trainable typedef names to each layer. |
| template resolution | Attach `backward_config_cpp`, `backward_function_cpp`, loss snippets, and controller snippets. |

## Implemented So Far

| Task | Commit | Files | What changed |
|---|---|---|---|
| [HLS4ML-002](/docs/status/tasks?query=HLS4ML-002) | `48a25d86` | `hls4ml/backends/vivado/vivado_backend.py`, `test/pytest/test_trainable_config.py` | Registered `vivado:trainable` as a backend flow that runs after `vivado:apply_templates` and before the writer-facing `vivado:ip` completion point. |
| [HLS4ML-002](/docs/status/tasks?query=HLS4ML-002) | `084968db` | `hls4ml/backends/vivado/passes/trainable.py`, `hls4ml/backends/vivado/vivado_backend.py`, `test/pytest/test_trainable_config.py` | Added `vivado:validate_trainable_config`. It is a no-op for inference models and, for trainable models, currently requires positive `BatchSize`, `half_mse`, `sgd`, a supported controller, one output, at least one trainable layer, Dense-only trainable layers, and the required trainable precision attributes. |
| [HLS4ML-002](/docs/status/tasks?query=HLS4ML-002) | `68165be0` | `hls4ml/backends/vivado/passes/trainable.py`, `hls4ml/backends/vivado/vivado_backend.py`, `test/pytest/test_trainable_config.py` | Added `vivado:resolve_trainable_backward_order`. It resolves a single sequential path from model output to model input, rejects branching graphs, and attaches `trainable_forward_path`, `trainable_forward_order`, `trainable_backward_order`, and `trainable_output_layer` to the `ModelGraph`. |
| [HLS4ML-006](/docs/status/tasks?query=HLS4ML-006) | `56f9ff00` | `hls4ml/backends/vivado/passes/trainable.py`, `hls4ml/backends/vivado/vivado_backend.py`, `test/pytest/test_trainable_config.py` | Added `vivado:resolve_trainable_loss_endpoints`. It attaches graph-level metadata for the one-output `half_mse` path: ground-truth input name, scalar loss name/type, loss-gradient seed name/type, loss input tensor/type/shape, output layer, and gradient scale. |

The current resolved metadata is intentionally graph-level metadata, not writer logic. The writer should later consume `model.trainable_backward_order` instead of deriving the backward pass from layer order or generated C++ strings.

Endpoint metadata has started under [HLS4ML-006](/docs/status/tasks?query=HLS4ML-006). The remaining `HLS4ML-006` work is now codegen-facing: loss config structs, loss helper kernels, ground-truth top-level IO, scalar loss outputs, and actual `dL/dy` buffer emission. That is the point where we need to review and slim down the old template/header code before porting it.

## Initial Graph Support

The first supported graph shape should be intentionally narrow:

```text
Input -> Dense -> OutputLoss
```

Then:

```text
Input -> Dense -> Activation -> Dense -> OutputLoss
```

Branches, merges, recurrent layers, convolution, and BatchNorm should be explicit future work, not accidentally half-supported.

## Layer Metadata

Trainable passes should attach explicit layer attributes for:

- trainable flag
- backward function code
- backward config code
- gradient input/output typedefs
- weight and bias gradient typedefs
- raw update and applied update typedefs
- accumulator typedefs
- optimizer-state typedefs
- references to forward weights, biases, and cached activations

## Exit Criteria

Phase 2 is done when a trainable Dense graph reaches the writer with all trainable endpoint and layer metadata already resolved.
