---
id: optimizer-controller
title: "Optimizer and Controller"
sidebar_label: "Optimizer + Controller"
status:
  - preliminary
  - inprogress
tags:
  - hls4ml
  - implementation
  - global-throttle
  - training
last_modified: 2026-05-21
author: mbvalentin
---
# Optimizer and Controller
<PageMeta />
---

<TBox type="summary" title="Role">

The optimizer proposes raw updates. The controller applies ENABOL's global throttle. These should be modeled separately so global throttling does not become hidden row/column projection inside dense or convolution kernels.

</TBox>

## Old Fork Behavior

The deprecated backpass headers accumulated gradients and updated weights inside layer backpass kernels. They also included row/column kappa throttling and optimizer helpers. This was practical but mixed several concepts:

```text
backward gradient propagation
gradient accumulation
optimizer state
row/column projection
weight update
logging
```

For ENABOL's current formulation, the clean separation is:

```text
backpass -> raw gradients/updates -> global controller -> applied updates
```

## ENABOL Global Throttle

Current ENABOL software training uses a controller that computes one scalar `alpha` for the full parameter update. Conceptually:

```math
\Delta_{\mathrm{raw}} = -\eta G
```

```math
\Delta_{\mathrm{actual}} = \alpha \Delta_{\mathrm{raw}}
```

This preserves update direction before fixed-point effects.

Hardware code should expose that same structure. Dense and convolution backpasses can still own local gradient accumulation, but global throttle state and alpha computation should be model-level.

## Proposed Hardware Phases

For a first sequential implementation:

```text
1. Forward pass.
2. Loss endpoint seeds dL/dy.
3. Reverse layer pass computes gradients.
4. Raw optimizer step is formed.
5. Global controller computes alpha.
6. Updates are scaled by alpha and applied.
```

There are two implementation variants:

| Variant | Description | Tradeoff |
|---|---|---|
| Layer-local update with shared alpha | Each layer computes raw update and immediately applies `alpha`. | Simpler memory layout, but alpha must be known before application. |
| Two-phase update | Layers compute/store raw updates; model-level controller computes alpha; second pass applies updates. | Cleaner global semantics, more storage. |

For global throttling that depends on global norms, the two-phase update is conceptually cleaner. For the first hardware prototype, we may start with layer-local metrics reduced into global accumulators, then apply updates after alpha is computed.

## Controller State

The controller config should define:

| Field | Meaning |
|---|---|
| `name` | `none`, `global_throttle_order_0`, `global_throttle_order_1`, `global_throttle_order_2`, or `global_throttle_order_2_qa`. |
| `chi` | Stability margin. |
| `alpha_min` | Minimum allowed throttle. |
| `alpha_max` | Maximum allowed throttle. |
| `eps` | Numerical guard. |
| `use_ema_max` | Whether control uses max of instantaneous and EMA curvature. |
| `ema_decay` | Curvature EMA decay. |
| `k_alpha` | State update gain for order-1/order-2 controllers. |
| `beta` | Velocity damping for order-2 controllers. |
| `state_t` | Precision for controller state. |

Generated C++ should include a controller config struct and static state variables.

## Optimizer Config

Start with SGD. Adam can follow after the gradient/controller path is validated.

```yaml
Optimizer:
  name: sgd
  learning_rate: 0.01
```

The deprecated fork had Adam helpers, but they bring extra state, square roots, bias correction, and fixed-point questions. Upstreamable work should land SGD first, then extend.

## Metrics Needed By Global Throttle

The controller needs global reductions:

| Metric | Source |
|---|---|
| `grad_norm` | Current flattened gradient. |
| `raw_update_norm` | Optimizer-proposed update before throttle. |
| `theta_delta_norm` | Parameter motion from previous step. |
| `grad_delta_norm` | Gradient change from previous step. |
| `curvature_proxy` | `grad_delta_norm / (theta_delta_norm + eps)`. |
| `curvature_ema` | Online filtered curvature proxy. |

In software these are easy because tensors are flattened. Hardware needs streaming or accumulated reductions across layer-local arrays.

## Recommended Implementation Boundary

Backpass kernels should expose enough information to compute model-level reductions:

```text
layer gradient norm contribution
layer raw update norm contribution
layer theta delta norm contribution
layer grad delta norm contribution
```

The controller combines those contributions into one alpha. That keeps the global throttle global.

