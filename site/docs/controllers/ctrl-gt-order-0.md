---
title: "CTRL-GT-ORDER-0"
sidebar_label: "CTRL-GT-ORDER-0"
status:
  - valid
tags:
  - controller
  - global-throttle
last_modified: 2026-05-23
author: mbvalentin
---
# CTRL-GT-ORDER-0
<PageMeta />
---

<TBox type="summary" title="Purpose">

`CTRL-GT-ORDER-0` is the algebraic safe-gain global throttle. It computes $\alpha_t$ directly from the current stability margin estimate, with no controller state.  The firmware implementation uses a **division-free candidate search** over a table of binary-fraction alpha values to avoid `ap_fixed` division crashes in CSIM.

</TBox>

## Definition

The controller enforces:

```math
\alpha_t\eta C_t^{\mathrm{ctrl}}
\leq
\chi.
```

In squared-norm form (no division, no sqrt):

```math
\alpha_t^2 \cdot \eta^2 \cdot \|\Delta G\|^2
\;\leq\;
\chi^2 \cdot \bigl(\|\Delta\theta\|^2 + \varepsilon^2\bigr)
```

A precomputed table of binary-fraction candidates is searched in descending order.  The first (largest) $\alpha$ satisfying the inequality is selected; if none satisfy, $\alpha = \alpha_{\min}$:

```
table = {1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.1875, 0.125, 0.0625, 0.03125}
for each α_cand in table:
    if α_cand² · η² · ||ΔG||²  ≤  χ² · (||Δθ||² + ε²):
        α = α_cand (break)
    else:
        α = α_min (fallback)
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

This controller reacts immediately. If $C_t^{\mathrm{ctrl}}$ spikes, $\alpha_t$ drops in the same step.  The binary-fraction candidate set produces alpha values representable as shift-add combinations.

## Known Failure Modes

- can oscillate if $C_t^{\mathrm{ctrl}}$ is noisy,
- can drive updates below the fixed-point quantum,
- coarse candidate spacing (11 values between 1.0 and 0.03125) may produce visible alpha steps;
  mitigated by adding more candidates at the cost of iteration cycles.

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
\|\Delta\theta\|^2 = \sum (θ_t - θ_{t-1})^2
\qquad
\|\Delta G\|^2 = \sum (G_t - G_{t-1})^2
```

**Phase 2 — Law** (`global_throttle_order0_law`, called once with global sums):

The inequality $\alpha^2 \eta^2 \|\Delta G\|^2 \leq \chi^2 (\|\Delta\theta\|^2 + \varepsilon^2)$
is evaluated directly on the squared norms — no division, no sqrt, no reciprocal.
An 11-entry candidate table (descending) is searched via a fully unrolled loop.
The `controller_alpha_min` field acts as fallback when no candidate satisfies
the constraint.

**Phase 3 — SGD + apply** — shared across all controllers.

### Controller parameters in config struct

The synthesized `trainable_configN` struct receives four
`static constexpr double` fields from the YAML `Controller` block:

| Field | YAML key | Default |
|---|---|---|
| `controller_chi` | `Chi` | `1.5` |
| `controller_epsilon` | `Epsilon` | `1e-12` |
| `controller_alpha_min` | `AlphaMin` | `0.0` |
| `controller_alpha_max` | `AlphaMax` | `1.0` |

These are emitted by `vivado_writer.py:_make_trainable_dense_config`.

### Call-chain emission

The three-phase `batch_end` block:

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

    // Phase 2 — Global controller law (division-free candidate search)
    nnet::global_throttle_order0_law<config>(global_dtheta_sq, global_dgrad_sq,
                                              alpha, reset_accumulators);

    // Phase 3 — SGD + apply per layer
    nnet::sgd<config>(wgrad, bgrad, w_update, b_update, lr);
    nnet::apply_dense_update<config>(w, b, w_update, b_update, alpha);
}
```

## Implementation Status

The original division-based GT-0 law (`α = χ / (η·C + ε)`) crashed CSIM at step 1
because `ap_fixed` division in the two `/` lines caused unrecoverable errors.
The law was replaced with a division-free inequality candidate search over
binary-fraction alpha values.  The fix is in `global_throttle_order0_law` in
`global_throttle.h` (HLS4ML-021).  CSIM validation with GT-0 active is pending
(ENB-017).
