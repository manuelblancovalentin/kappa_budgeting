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

### Architecture

GT-0 is implemented as two separate kernels in the three-phase `batch_end` block:

**Phase 1 — Curvature sensor** (`curvature_sensor_order0`, shared with GT-1):

```
nnet::curvature_sensor_order0<CONFIG>(weights, biases, wgrad, bgrad,
                                       &dtheta_sq, &dgrad_sq, reset);
```

Maintains static `prev_weights`/`prev_biases`/`prev_weight_grad`/`prev_bias_grad`.
On the first call or after `reset_numerator = true`, stores current values
and emits zero squared-norm contributions. On subsequent calls:

```math
||\Delta\theta||^2 = \sum (θ_t - θ_{t-1})^2
\qquad
||\Delta G||^2 = \sum (G_t - G_{t-1})^2
```

**Phase 2 — Law** (`global_throttle_order0_law`, called once with global sums):

```math
C = \frac{||\Delta G||}{||\Delta\theta|| + \epsilon}
\qquad
\alpha = \text{clip}\!\left(\frac{\chi}{\eta C + \epsilon},\ \alpha_{\min},\ \alpha_{\max}\right)
```

Sqrt is computed through `double` for CSIM (`std::sqrt`).

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

The three-phase `batch_end` block (all global-throttle controllers):

```cpp
if (batch_end) {
    // Phase 1 — Accumulate per-layer squared norms
    metric_t global_dtheta_sq = 0, global_dgrad_sq = 0;

    // (per Dense layer in backward order)
    {
        metric_t __dt, __dg;
        nnet::curvature_sensor_order0<config>(w, b, wgrad, bgrad,
                                              __dt, __dg, reset_accumulators);
        global_dtheta_sq += __dt;
        global_dgrad_sq += __dg;
    }

    // Phase 2 — Global controller law
    nnet::global_throttle_order0_law<config>(global_dtheta_sq, global_dgrad_sq,
                                              alpha, reset_accumulators);

    // Phase 3 — SGD + apply per layer
    nnet::sgd<config>(wgrad, bgrad, w_update, b_update, lr);
    nnet::apply_dense_update<config>(w, b, w_update, b_update, alpha);
}
```

For CTRL-NONE, phases 1 and 2 are replaced by a single `global_throttle_none` call (α=1).

For CTRL-GT-ORDER-1, phase 2 uses `global_throttle_order1_law` instead.

## Implementation Status

The CTRL-NONE CSIM (CSIM-001) validated the full Dense loss-backprop-SGD-apply
pipeline with $\alpha = 1$. CTRL-GT-ORDER-0 has been implemented and wired
(HLS4ML-021) but CSIM validation with GT-0 active is pending (planned as
CSIM-002 / ENB-017).

This is the controller used in the first global-throttle sanity experiments (EXP-000A, EXP-000B).
