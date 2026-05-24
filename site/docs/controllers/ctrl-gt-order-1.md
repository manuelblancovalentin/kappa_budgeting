---
title: "CTRL-GT-ORDER-1"
sidebar_label: "CTRL-GT-ORDER-1"
status:
  - inprogress
tags:
  - controller
  - global-throttle
last_modified: 2026-05-23
author: mbvalentin
---
# CTRL-GT-ORDER-1
<PageMeta />
---

<TBox type="summary" title="Purpose">

`CTRL-GT-ORDER-1` treats $\alpha_t$ as a first-order dynamic state. It smooths the instantaneous jumps of `CTRL-GT-ORDER-0`.  The firmware uses the same division-free inequality search as GT-0, replacing the division-based margin with an attractor toward the largest feasible candidate.

</TBox>

## Definition

The controller searches the candidate table to find the largest $\alpha$ satisfying the stability constraint:

```math
\alpha^2 \cdot \eta^2 \cdot \|\Delta G\|^2
\;\leq\;
\chi^2 \cdot \bigl(\|\Delta\theta\|^2 + \varepsilon^2\bigr)
```

Denote this value $\alpha^{\text{feasible}}_t$.  The first-order update toward it:

```math
\boxed{
\alpha_{t+1}
=
\alpha_t
+
k_\alpha
\left(
\alpha^{\text{feasible}}_t - \alpha_t
\right).
}
```

Then clamp:

```math
\alpha_{t+1}
\leftarrow
\operatorname{clip}(\alpha_{t+1},\alpha_{\min},\alpha_{\max}).
```

This replaces the continuous margin $m_t = \chi - \eta\alpha_t C$ with the gap
$\alpha^{\text{feasible}}_t - \alpha_t$, avoiding the division in $C = \|\Delta G\| / (\|\Delta\theta\| + \varepsilon)$.

## Inputs

| Input | Meaning |
|---|---|
| $\Delta\theta_t^{\mathrm{raw}}$ | raw optimizer update |
| $\eta$ | base learning rate |
| $\|\Delta\theta\|^2$ | squared parameter delta |
| $\|\Delta G\|^2$ | squared gradient delta |
| $\chi$ | target stability margin |
| $k_\alpha$ | controller adaptation gain |
| $\varepsilon$ | numerical guard |

## Outputs

| Output | Meaning |
|---|---|
| $\alpha_t$ | global throttle scalar |
| $\eta_t^{\mathrm{eff}}$ | effective learning rate |
| $\Delta\theta_t^{\mathrm{ctrl}}$ | globally scaled update |

## State

| State | Meaning |
|---|---|
| $\alpha_t$ | controller gain state |

## Expected Behavior

This controller reacts more smoothly than GT-0, at the cost of slower response to sudden spikes.  The attractor toward $\alpha^{\text{feasible}}$ acts as a first-order low-pass on the GT-0 decision, with $k_\alpha$ controlling the time constant.

## hls4ml Implementation

Kernel: `nnet::global_throttle_order1_law<CONFIG_T>(raw_update_norm_sq, dgrad_norm_sq, alpha, ..., reset_numerator)`.

Shares the `raw_update_sensor_order0` (Phase 1) with GT-0. The law uses raw optimizer update geometry, not actual/throttled parameter movement. For SGD, `raw_update_norm_sq` already includes the learning rate because the raw update is `-eta * gradient`. Maintains a static `alpha_state` variable. On `reset_numerator = true`, `alpha_state` reinitializes to 1.

Law (division-free, on squared norms):
```math
\alpha_{\text{feasible}} = \max\{\alpha_{\text{cand}} \,|\, \alpha^2 \eta^2 \|\Delta G\|^2 \leq \chi^2 (\|\Delta\theta_{\text{raw}}\|^2 + \varepsilon^2)\}
\qquad
\alpha \leftarrow \operatorname{clip}\bigl(\alpha + k_\alpha(\alpha_{\text{feasible}} - \alpha),\; \alpha_{\min},\; \alpha_{\max}\bigr)
```

Config field `controller_k_alpha` (default `0.1`, key `KAlpha` in YAML).

The candidate table does not use zero as a normal candidate. If no candidate satisfies the inequality, the law uses the nonzero minimum candidate and logs `controller_feasible = 0`.

Trace logging provides raw, controlled, and actual update squared norms, ||ΔG||², the squared stability terms, α_feasible, α_state, α_code, α_min, and feasibility.

## Implementation Status

Implemented in hls4ml-trainable (`HLS4ML-022`, inprogress). The original division-based GT-1 law was replaced with the division-free inequality-search approach, matching GT-0's fix. HLS4ML-039 replaced actual-update denominator geometry with raw-update control geometry to avoid alpha self-locking.
