---
title: "Phase 4: Controllers and Synchronization"
sidebar_label: "🚧 Phase 4: Controllers"
status:
  - planned
tags:
  - hls4ml
  - controller
  - global-throttle
last_modified: 2026-05-22
author: mbvalentin
---
# Phase 4: Controllers and Synchronization
<PageMeta />
---

<TBox type="summary" title="Goal">

Add global throttling as a scalar synchronization mechanism over raw update proposals, with optional κ safety bounds represented as constraints rather than per-layer direction changes.

</TBox>

## Linked Tasks

- [HLS4ML-008](/docs/status/tasks?query=HLS4ML-008): controller kernels.
- [HLS4ML-009](/docs/status/tasks?query=HLS4ML-009): global reductions and alpha synchronization.
- [HLS4ML-010](/docs/status/tasks?query=HLS4ML-010): delayed-alpha schedule exploration.

## Controller Order

Implement controllers in this order:

```text
CTRL-NONE
CTRL-GT-ORDER-0
CTRL-GT-ORDER-1
CTRL-GT-ORDER-2
CTRL-GT-ORDER-2-QA
```

`CTRL-NONE` is a required debugging path, not an optional feature.

## κ Coupling

κ budgeting should be represented as a safety envelope when enabled. Global throttling remains the direction-preserving learning-speed controller.

The combined scalar should have the shape:

```text
alpha = min(alpha_curvature, alpha_kappa)
```

where `alpha_kappa` is the largest scalar that keeps the proposed global update inside the κ-safe set. This preserves the raw update direction instead of applying unrelated per-layer scaling.

## Synchronization Point

The simple reference implementation should use a two-phase schedule:

```text
1. forward + loss + backpass
2. compute raw updates and global metrics
3. compute alpha
4. apply alpha-scaled updates
```

The delayed-alpha schedule can be tested later against this baseline. It may be useful, but it should not define the first correctness path.

## Exit Criteria

Phase 4 is done when the one-Dense CSIM can report loss, gradient metrics, alpha, and updated weights for `CTRL-NONE` and `CTRL-GT-ORDER-0`.
