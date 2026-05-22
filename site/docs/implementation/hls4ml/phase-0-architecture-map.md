---
title: "Phase 0: Architecture Map"
sidebar_label: "Phase 0: Architecture"
status:
  - complete
tags:
  - hls4ml
  - implementation
  - training
last_modified: 2026-05-22
author: mbvalentin
---
# Phase 0: Architecture Map
<PageMeta />
---

<TBox type="summary" title="Goal">

Freeze the first implementation strategy before code changes: keep trainable support hls4ml-native, centered on config, graph/flow passes, layer attributes, backend templates, and narrow writer hooks.

</TBox>

## Linked Tasks

- [HLS4ML-001](/docs/status/tasks?query=HLS4ML-001): trainable `HLSConfig` schema.
- [HLS4ML-002](/docs/status/tasks?query=HLS4ML-002): trainable `ModelGraph` flow and validation.
- [HLS4ML-011](/docs/status/tasks?query=HLS4ML-011): writer hooks.
- [HLS4ML-012](/docs/status/tasks?query=HLS4ML-012): ENABOL bridge.

## Decision

The implementation should not be a large Vivado-writer patch that reconstructs training behavior from generated C++ strings. The trainable path should be represented before writing:

```text
ENABOL config export
  -> HLSConfig Model.Training
  -> ModelGraph + trainable flow passes
  -> layer attributes and endpoint attributes
  -> backend templates
  -> writer emits already-resolved trainable code
```

This keeps the upstream patch smaller and makes the generated firmware easier to reason about.

## Relevant hls4ml Objects

`HLSConfig` is the normalized configuration access layer. It should own trainable schema accessors and validation-facing helpers.

`ModelGraph` is the compiled graph that carries layers, variables, attributes, and applied flows. It should remain the central object, not be replaced by a separate training graph.

Flows and optimizer passes are the right place to validate trainable mode, resolve loss endpoints, compute backward traversal order, and attach template-ready attributes.

Backend templates are the right place to generate Dense backpass, loss, update, and controller C++ snippets.

The Vivado writer should only add top-level IO, static trainable assets, and precomputed snippets. It should not infer weight, gradient, or loss semantics.

## Exit Criteria

Phase 0 is done when the roadmap, reference pages, and task stream agree on the same architecture. The remaining work should be executable as phase-specific hls4ml changes rather than open-ended reverse engineering.
