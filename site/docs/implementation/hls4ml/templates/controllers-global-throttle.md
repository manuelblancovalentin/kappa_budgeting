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

## Naming Note

`global_throttle` is the code name for now. In prose, "global update throttle" or "global step throttle" is slightly more explicit, but the implementation name is short and matches the control signal we care about: one global `alpha`.

