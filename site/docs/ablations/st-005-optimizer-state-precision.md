---
title: "ST-005: Optimizer-State Precision"
sidebar_label: "ST-005 Optimizer State"
status:
  - preliminary
tags:
  - ablation-test
  - optimizer
  - adam
last_modified: 2026-05-18
author: mbvalentin
---
# ST-005: Accumulator / Optimizer-State Precision
<PageMeta />
---

<TBox type="summary" title="Purpose">

This stress test applies rail and floor analysis to training internals that are not part of the forward model but still determine the update.

</TBox>

## Mechanism

For Adam, vulnerable objects include:

```math
m_t,
\qquad
v_t,
\qquad
\sqrt{v_t}+\epsilon,
\qquad
\frac{m_t}{\sqrt{v_t}+\epsilon}.
```

For batch training, vulnerable objects include:

```math
\sum_{b=1}^{B}G_b,
\qquad
\frac{1}{B}\sum_{b=1}^{B}G_b.
```

## How This Differs From Rail/Floor Tests

The failure mechanism is still saturation, underflow, or quantization error. The difference is scope: the vulnerable tensors are optimizer state and accumulation registers rather than only activations, weights, gradients, and updates.

## Main Plots

- Adam `m` norm,
- Adam `v` norm,
- denominator statistics,
- raw step norm,
- quantized step norm,
- accumulator rail pressure,
- final update norm.

## Priority

This test should come after the simpler fixed-point path is clear.
