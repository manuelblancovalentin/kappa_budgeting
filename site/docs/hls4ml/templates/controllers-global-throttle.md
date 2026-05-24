---
title: "Global Throttle Controller Template"
sidebar_label: "controllers/global_throttle.h"
status:
  - inprogress
tags:
  - hls4ml
  - trainable
  - controller
  - global-throttle
last_modified: 2026-05-22
author: mbvalentin
source: "hls4ml/templates/vivado/trainable/controllers/global_throttle.h"
---
# Global Throttle Controller Template
<PageMeta />
---

`global_throttle.h` starts with the no-controller baseline and the shared update application primitive.

## `global_throttle_none`

```cpp
nnet::global_throttle_none<CONFIG_T>(alpha);
```

This emits:

```cpp
alpha[0] = 1;
```

It preserves the same hardware data path as the future global throttling controllers, but disables throttling. This is useful for isolating bugs in loss, backprop, SGD, and parameter application before adding curvature logic.

## `apply_dense_update`

```cpp
nnet::apply_dense_update<CONFIG_T>(weights, biases, weight_update, bias_update, alpha);
```

This applies the alpha-scaled raw updates:

```text
weights += alpha * weight_update
biases  += alpha * bias_update
```

The function converts through `update_t` before assigning back to `weight_t` or `bias_t`, so update precision can be controlled separately from stored parameter precision.

## Three-Phase Architecture

All non-NONE controllers use a three-phase `batch_end` block, emitted by the Vivado writer:

### Phase 1 — Curvature accumulation (per layer)

```cpp
nnet::curvature_sensor_order0<CONFIG_T>(weights, biases, weight_grad, bias_grad,
                                        &dtheta_sq_contrib, &dgrad_sq_contrib,
                                        reset_accumulators);
```

Maintains static `prev_weights`/`prev_biases`/`prev_weight_grad`/`prev_bias_grad` storage for the layer. Computes the squared L2 norm of the differences and outputs them as scalar references. The writer accumulates these into `global_dtheta_sq` / `global_dgrad_sq` across layers. The `has_prev` flag resets on `reset_numerator = true`.

### Phase 2 — Controller law (global, once)

```cpp
nnet::global_throttle_order0_law<CONFIG_T>(global_dtheta_sq, global_dgrad_sq, alpha, reset);
nnet::global_throttle_order1_law<CONFIG_T>(global_dtheta_sq, global_dgrad_sq, alpha, reset);
```

Takes the globally accumulated squared norms, computes `C = ||ΔG|| / (||Δθ|| + ε)` via double-precision sqrt, then applies the order-specific law. GT-0 emits the algebraic safe gain; GT-1 maintains a static `alpha_state`.

### Phase 3 — SGD + alpha-scaled apply (per layer)

```cpp
nnet::sgd<CONFIG_T>(...);
nnet::apply_dense_update<CONFIG_T>(weights, biases, weight_update, bias_update, alpha);
```

The same alpha is broadcast to all layers within the same `batch_end` block.

## Trace Diagnostics

When `!defined(__SYNTHESIS__) && defined(HLS4ML_TRAINABLE_TRACE)`, the law kernels log curvature, ||dθ||, ||dG||, alpha_state (GT-1) in addition to alpha.

## Naming Note

`global_throttle` is the code name for now. In prose, "global update throttle" or "global step throttle" is slightly more explicit, but the implementation name is short and matches the control signal we care about: one global `alpha`.

