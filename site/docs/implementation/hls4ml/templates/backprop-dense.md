---
title: "Dense Backprop Template"
sidebar_label: "backprop/nnet_dense_backprop.h"
status:
  - inprogress
tags:
  - hls4ml
  - trainable
  - backprop
  - dense
last_modified: 2026-05-22
author: mbvalentin
source: "hls4ml/templates/vivado/trainable/backprop/nnet_dense_backprop.h"
---
# Dense Backprop Template
<PageMeta />
---

`nnet_dense_backprop.h` implements the first supported trainable layer backpass: a Dense layer in the sequential one-output path.

## Kernel Role

The kernel computes three things:

| Output | Meaning |
|---|---|
| `grad_out` | Input-side gradient, `dL/dx_i = sum_j dL/dy_j * W_ij`. |
| `weight_grad_accum`, `bias_grad_accum` | Running batch sums for `dL/dW` and `dL/db`. |
| `weight_grad`, `bias_grad` | Batch-averaged gradients emitted when `batch_end=True`. |

The kernel does not update weights. That separation is intentional: backprop computes gradients, SGD proposes raw updates, and the controller later applies a global `alpha`.

## Batch Averaging

The config struct must provide:

```cpp
static const unsigned batch_size_log2 = ...;
```

ENABOL rounds the requested batch size up to a power of two before emitting hls4ml config. The C++ kernel then averages accumulated gradients with a shift:

```cpp
avg >>= batch_shift;
```

This avoids synthesizing a divider in the minimal path.

## Trace Points

The file contains trace hooks for:

- forward input activation,
- incoming loss/activation gradient,
- outgoing previous-layer gradient,
- weight/bias accumulators,
- final weight/bias gradients at `batch_end`.

These hooks are inactive unless `HLS4ML_TRAINABLE_TRACE` is defined by generated firmware.

