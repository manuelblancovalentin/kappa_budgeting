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
    if α_cand² · η² · ||ΔG||²  ≤  χ² · (||Δθ_raw||² + ε²):
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

**Phase 1 — Raw-update controller sensor** (`raw_update_sensor_order0`, shared with GT-1):

```
nnet::raw_update_sensor_order0<CONFIG>(w_update, b_update, wgrad, bgrad,
                                       &raw_update_norm_sq, &dgrad_norm_sq, reset);
```

Maintains static `prev_weight_grad`/`prev_bias_grad`. The controller no longer
uses actual/throttled parameter movement as its denominator. It uses the raw SGD
proposal before alpha:

```math
\|\Delta\theta_{\text{raw}}\|^2 = \sum (\Delta\theta_{\text{raw}})^2
\qquad
\|\Delta G\|^2 = \sum (G_t - G_{t-1})^2
```

For SGD, `sgd.h` computes `Delta theta_raw = -learning_rate * gradient`, so the
raw update norm already includes $\eta$.

**Phase 2 — Law** (`global_throttle_order0_law`, called once with global sums):

The inequality $\alpha^2 \eta^2 \|\Delta G\|^2 \leq \chi^2 (\|\Delta\theta_{\text{raw}}\|^2 + \varepsilon^2)$
is evaluated directly on the squared norms — no division, no sqrt, no reciprocal.
An 11-entry candidate table (descending) is searched via a fully unrolled loop.
If no candidate satisfies the constraint, the law uses a nonzero minimum
candidate (`0.03125`, or a higher configured `AlphaMin`) and logs
`controller_feasible = 0`.

**Phase 3 — Apply** — shared across all controllers. The writer logs actual
fixed-point movement after assignment separately as telemetry.

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
    // Phase 1 — SGD proposal and raw-update norm reduction
    metric_t global_raw_update_norm_sq = 0, global_dgrad_norm_sq = 0;

    // (per Dense layer in backward order)
    nnet::sgd<config>(wgrad, bgrad, w_update, b_update, lr);
    {
        metric_t __raw, __dg;
        nnet::raw_update_sensor_order0<config>(w_update, b_update, wgrad, bgrad,
                                               __raw, __dg, reset_accumulators);
        global_raw_update_norm_sq += __raw;
        global_dgrad_norm_sq += __dg;
    }

    // Phase 2 — Global controller law (division-free candidate search)
    nnet::global_throttle_order0_law<config>(global_raw_update_norm_sq, global_dgrad_norm_sq,
                                              alpha, reset_accumulators);

    // Phase 3 — alpha-scaled apply per layer
    metric_t actual_update_norm_sq;
    nnet::apply_dense_update<config>(w, b, w_update, b_update, alpha, actual_update_norm_sq);
}
```

## Implementation Status

The original division-based GT-0 law (`α = χ / (η·C + ε)`) crashed CSIM at step 1
because `ap_fixed` division in the two `/` lines caused unrecoverable errors.
The law was replaced with a division-free inequality candidate search over
binary-fraction alpha values. The raw-update geometry fix is tracked in
HLS4ML-039.
