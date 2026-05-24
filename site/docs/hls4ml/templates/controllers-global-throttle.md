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
nnet::apply_dense_update<CONFIG_T>(weights, biases, weight_update, bias_update, alpha, actual_update_norm_sq);
```

This applies the alpha-scaled raw updates:

```text
weights += alpha * weight_update
biases  += alpha * bias_update
```

The function converts through `update_t` before assigning back to `weight_t` or `bias_t`, so update precision can be controlled separately from stored parameter precision.
It also returns the actual squared movement after fixed-point assignment so CSIM can compare raw, controlled, and quantized parameter motion.

## Three-Phase Architecture

All non-NONE controllers use a three-phase `batch_end` block, emitted by the Vivado writer:

### Phase 1 — SGD proposal and raw-update sensing (per layer)

```cpp
nnet::sgd<CONFIG_T>(weight_grad, bias_grad, weight_update, bias_update, learning_rate);
nnet::raw_update_sensor_order0<CONFIG_T>(weight_update, bias_update, weight_grad, bias_grad,
                                         raw_update_norm_sq_contrib, dgrad_norm_sq_contrib,
                                         reset_accumulators);
```

Maintains static previous-gradient storage for the layer. The controller uses the raw optimizer proposal as its denominator geometry, not the already-throttled actual parameter movement. For SGD, `weight_update` and `bias_update` already include the learning rate because `sgd.h` computes `-learning_rate * gradient`. The writer accumulates per-layer raw update and gradient-difference norms into `global_raw_update_norm_sq` / `global_dgrad_norm_sq` across layers.

### Phase 2 — Controller law (global, once)

```cpp
nnet::global_throttle_order0_law<CONFIG_T>(global_raw_update_norm_sq, global_dgrad_norm_sq, alpha, ..., reset);
nnet::global_throttle_order1_law<CONFIG_T>(global_raw_update_norm_sq, global_dgrad_norm_sq, alpha, ..., reset);
```

Takes the globally accumulated squared norms and searches a binary-fraction alpha candidate table with no division, reciprocal, or sqrt:

```text
alpha^2 * eta^2 * ||Delta G||^2 <= chi^2 * (||Delta theta_raw||^2 + epsilon^2)
```

If no candidate satisfies the inequality, the minimum nonzero candidate is used and `controller_feasible` is logged as `0`.

### Phase 3 — Alpha-scaled apply (per layer)

```cpp
nnet::apply_dense_update<CONFIG_T>(weights, biases, weight_update, bias_update, alpha, actual_update_norm_sq);
```

The same alpha is broadcast to all layers within the same `batch_end` block.

## Trace Diagnostics

When `!defined(__SYNTHESIS__) && defined(HLS4ML_TRAINABLE_TRACE)`, the law kernels log controller internals in addition to alpha. The generated CSIM testbench also writes `controller.dat` with raw, controlled, and actual update norms, gradient-difference norm, stability inequality terms, alpha state, alpha code, alpha floor, and feasibility.

## Naming Note

`global_throttle` is the code name for now. In prose, "global update throttle" or "global step throttle" is slightly more explicit, but the implementation name is short and matches the control signal we care about: one global `alpha`.
