---
title: "Phase 2: Graph and Flow Integration"
sidebar_label: "🚧 Phase 2: Graph + Flow"
status:
  - planned
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
