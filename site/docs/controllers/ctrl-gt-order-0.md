---
title: "CTRL-GT-ORDER-0"
sidebar_label: "CTRL-GT-ORDER-0"
status:
  - valid
tags:
  - controller
  - global-throttle
last_modified: 2026-05-18
author: mbvalentin
---
# CTRL-GT-ORDER-0
<PageMeta />
---

<TBox type="summary" title="Purpose">

`CTRL-GT-ORDER-0` is the algebraic safe-gain global throttle. It computes $\alpha_t$ directly from the current stability margin estimate, with no controller state.

</TBox>

## Definition

The controller tries to enforce:

```math
\alpha_t\eta C_t^{\mathrm{ctrl}}
\leq
\chi.
```

It sets:

```math
\boxed{
\alpha_t
=
\min
\left(
1,
\frac{\chi}
{\eta(C_t^{\mathrm{ctrl}}+\epsilon)}
\right).
}
```

Then:

```math
\Delta\theta_t^{\mathrm{ctrl}}
=
\alpha_t
\Delta\theta_t^{\mathrm{raw}}.
```

## Inputs

| Input | Meaning |
|---|---|
| $\Delta\theta_t^{\mathrm{raw}}$ | raw optimizer update |
| $\eta$ | base learning rate |
| $C_t^{\mathrm{ctrl}}$ | curvature / sensitivity proxy |
| $\chi$ | target stability margin |
| $\epsilon$ | numerical guard |

## Outputs

| Output | Meaning |
|---|---|
| $\alpha_t$ | global throttle scalar |
| $\eta_t^{\mathrm{eff}}=\alpha_t\eta$ | effective learning rate |
| $\Delta\theta_t^{\mathrm{ctrl}}$ | globally scaled update |

## State

None.

## Expected Behavior

This controller reacts immediately. If $C_t^{\mathrm{ctrl}}$ spikes, $\alpha_t$ drops in the same step.

## Known Failure Modes

- can oscillate if $C_t^{\mathrm{ctrl}}$ is noisy,
- can drive updates below the fixed-point quantum,
- does not reason about the lower useful-update bound.

## hls4ml Implementation

The hls4ml firmware kernel lives in `templates/vivado/trainable/controllers/global_throttle.h` as
`nnet::global_throttle_order0<CONFIG_T>(weights, biases, weight_grad, bias_grad, alpha, reset_numerator)`.

### Internal curvature sensor

The kernel maintains static persistent storage for the previous parameter vector
(`prev_weights`, `prev_biases`) and previous gradient vector
(`prev_weight_grad`, `prev_bias_grad`). On the first call or after
`reset_numerator = true` it stores the current values and emits $\alpha = 1$.
On subsequent calls it computes:

```math
||\Delta\theta||^2 = \sum_{\text{all params}} (\theta_t - \theta_{t-1})^2
\qquad
||\Delta G||^2 = \sum_{\text{all params}} (G_t - G_{t-1})^2
```

then:

```math
C = \frac{||\Delta G||}{||\Delta\theta|| + \epsilon}
\qquad
\alpha = \text{clip}\!\left(\frac{\chi}{\eta C + \epsilon},\ \alpha_{\min},\ \alpha_{\max}\right)
```

### Controller parameters in config struct

The synthesized `trainable_configN` struct receives four new
`static constexpr double` fields from the YAML `Controller` block:

| Field | YAML key | Default |
|---|---|---|
| `controller_chi` | `Chi` | `1.5` |
| `controller_epsilon` | `Epsilon` | `1e-12` |
| `controller_alpha_min` | `AlphaMin` | `0.0` |
| `controller_alpha_max` | `AlphaMax` | `1.0` |

These are emitted by `vivado_writer.py:_make_trainable_dense_config`.

### Call-chain emission

In `_make_trainable_call_chain`, if the normalized controller kind is
`ctrl_gt_order_0`, the generated `batch_end` block emits:

```cpp
// (1) Compute curvature and safe alpha from accumulated gradient
nnet::global_throttle_order0<config>(w, b, w_grad, b_grad, alpha, reset_accumulators);

// (2) Produce raw update direction
nnet::sgd<config>(w_grad, b_grad, w_update, b_update, lr);

// (3) Apply alpha-scaled update
nnet::apply_dense_update<config>(w, b, w_update, b_update, alpha);
```

### Square-root for CSIM vs synthesis

For C simulation (`!defined(__SYNTHESIS__)`) the kernel uses
`std::sqrt(double(...))`. For synthesis, `nnet::sqrt(...)` is the
placeholder until a fixed-point-safe sqrt is validated.

## Implementation Status

The CTRL-NONE CSIM (CSIM-001) validated the full Dense loss-backprop-SGD-apply
pipeline with $\alpha = 1$. CTRL-GT-ORDER-0 has been implemented and wired
(HLS4ML-021) but CSIM validation with GT-0 active is pending (planned as
CSIM-002 / ENB-017).

This is the controller used in the first global-throttle sanity experiments (EXP-000A, EXP-000B).
