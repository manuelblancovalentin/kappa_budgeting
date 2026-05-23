---
title: "SGD Optimizer Template"
sidebar_label: "optimizers/sgd.h"
status:
  - inprogress
tags:
  - hls4ml
  - trainable
  - optimizer
  - sgd
last_modified: 2026-05-22
author: mbvalentin
source: "hls4ml/templates/vivado/trainable/optimizers/sgd.h"
---
# SGD Optimizer Template
<PageMeta />
---

`sgd.h` implements the minimal optimizer stage:

```cpp
raw_update = -learning_rate * gradient
```

It does not update parameters directly.

## Why Raw Updates

Global throttling needs to inspect or bound the update vector before it is applied. For that reason, the optimizer emits raw update proposals:

- `weight_update`,
- `bias_update`.

The controller/update stage later applies:

```cpp
parameter += alpha * raw_update
```

For `CTRL-NONE`, `alpha=1`, so this behaves like plain SGD.

## Config Contract

The generated config struct must provide:

- `n_in`, `n_out`,
- `weight_grad_t`, `bias_grad_t`,
- `raw_update_t`,
- `learning_rate_t`.

The learning rate may eventually be either a compile-time constant or a runtime input. The current template accepts it as an argument so the top-level writer can support both policies.

