---
title: "MSE Loss Templates"
sidebar_label: "losses/mse.h"
status:
  - inprogress
tags:
  - hls4ml
  - trainable
  - loss
last_modified: 2026-05-22
author: mbvalentin
source: "hls4ml/templates/vivado/trainable/losses/mse.h"
---
# MSE Loss Templates
<PageMeta />
---

`mse.h` implements the first trainable loss endpoint. It exposes two functions:

```cpp
nnet::mse<CONFIG_T>(prediction, ground_truth, loss, loss_grad);
nnet::half_mse<CONFIG_T>(prediction, ground_truth, loss, loss_grad);
```

Both functions use the same `mse_core` implementation with compile-time numerator/denominator factors.

## Scaling

| Function | Scalar loss | Gradient seed |
|---|---|---|
| `mse` | `sum((y_hat - y)^2)` | `2 * (y_hat - y)` |
| `half_mse` | `0.5 * sum((y_hat - y)^2)` | `y_hat - y` |

For the first hardware/software trace comparison, `half_mse` is the cleaner default because its output gradient is exactly the prediction error.

## Config Contract

The generated config struct must provide:

- `n_out`,
- `data_in_t`,
- `ground_truth_t`,
- `loss_t`,
- `grad_out_t`.

When trainable tracing is enabled, the writer must also emit trace names for prediction, ground truth, loss, and loss gradient.

