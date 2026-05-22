---
title: "Phase 3: Templates and Kernels"
sidebar_label: "🚧 Phase 3: Templates"
status:
  - planned
tags:
  - hls4ml
  - implementation
  - template
last_modified: 2026-05-22
author: mbvalentin
---
# Phase 3: Templates and Kernels
<PageMeta />
---

<TBox type="summary" title="Goal">

Port the useful deprecated trainable kernels into the new hls4ml backend structure, starting with Dense, loss endpoints, and raw update storage.

</TBox>

## Linked Tasks

- [HLS4ML-004](/docs/status/tasks?query=HLS4ML-004): static autograd and loss headers.
- [HLS4ML-005](/docs/status/tasks?query=HLS4ML-005): Dense backpass templates.
- [HLS4ML-006](/docs/status/tasks?query=HLS4ML-006): loss endpoint templates.
- [HLS4ML-007](/docs/status/tasks?query=HLS4ML-007): two-phase raw update storage.

## First Kernels

The first C++ path should include:

```text
forward Dense
loss endpoint
Dense backpass
raw SGD update proposal
apply update with scalar alpha
```

This gives us a correctness baseline before adding controller complexity.

## Template Policy

Templates should consume config structs and layer attributes. They should not inspect arbitrary strings emitted by earlier templates.

Deprecated headers can be used as source material, but the new implementation should remove assumptions that only existed for the old ENABOL bridge.

## Loss Scaling Decision

We need to decide whether the first endpoint uses `mse` or `half_mse`. The important requirement is not the name; it is matching the software trace scale exactly:

```text
half_mse: dL/dy = y_hat - y
mse:      dL/dy = 2 * (y_hat - y)
```

The selected convention must be recorded in config and test traces.

## Exit Criteria

Phase 3 is done when generated C++ contains Dense backpass, loss, and update kernels that compile in CSIM and match ENABOL software gradients for the one-Dense case.
