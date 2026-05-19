---
title: "Ablation Tests"
sidebar_label: "Index"
status:
  - preliminary
  - inprogress
tags:
  - ablation-registry
  - global-throttle
last_modified: 2026-05-18
author: mbvalentin
---
# Ablation Tests
<PageMeta />
---

<TBox type="summary" title="Purpose">

This section defines the stress-test families used to evaluate fixed-point online learning. Each experiment should instantiate one family with a concrete dataset, model, precision map, optimizer, controller, and stress configuration.

</TBox>

The goal is not to make training fail by choosing arbitrary bad hyperparameters. The goal is to create controlled situations where:

```math
\text{float training is stable}
\quad
\text{but}
\quad
\text{hardware-like fixed-point training becomes unsafe}.
```

Then we test whether the global throttle can preserve useful learning by adapting:

```math
\eta_t^{\mathrm{eff}}=\alpha_t\eta,
\qquad
0<\alpha_t\leq 1.
```

## Experiment Trace

Experiment pages should reference ablation tests by stable ID:

```yaml
dataset: DS-...
model: MDL-...
precision: PREC-...
stress_test: ST-...
optimizer: ...
controller: CTRL-...
metrics: [...]
```

## Stress-Test Table

| Stress ID | Family | Main question | Priority |
|---|---|---|---|
| [`ST-000`](./st-000-high-learning-rate-sanity.md) | High learning-rate sanity | Can the controller stabilize a deliberately unstable loop? | <Badge status="priority-low" /> |
| [`ST-001`](./st-001-distribution-shift.md) | Distribution / operating-point shift | Does a fixed $\eta$ become unsafe after deployment drift? | <Badge status="priority-high" /> |
| [`ST-002`](./st-002-transient-spike.md) | Transient spike / corrupted batch | Can one abnormal batch corrupt the learning trajectory? | <Badge status="priority-medium" /> |
| [`ST-003`](./st-003-rail-saturation.md) | Rail saturation / overflow | Do fixed-point tensors hit representable rails? | <Badge status="priority-high" /> |
| [`ST-004`](./st-004-update-dead-zone.md) | Quantization floor / update dead-zone | Does learning die because updates are too small to represent? | <Badge status="priority-high" /> |
| [`ST-005`](./st-005-optimizer-state-precision.md) | Accumulator / optimizer-state precision | Do Adam state or gradient accumulators fail even when weights look safe? | <Badge status="priority-medium" /> |
| [`ST-006`](./st-006-naive-safety-mechanisms.md) | Naive safety mechanisms | Do clipping methods stabilize numerics while distorting learning? | <Badge status="priority-medium" /> |
| [`ST-007`](./st-007-kappa-projection-mismatch.md) | Legacy $\kappa$ projection mismatch | Does old row/column projection preserve rails by distorting the update trajectory? | <Badge status="priority-low" /> |

## Quantization And Throttling Are Coupled

The precision choice defines the representable range and resolution. The throttle controls the training trajectory inside that representable envelope.

For fixed-point updates, stable and useful learning requires:

```math
\boxed{
\frac{q_\Delta}{\eta\|G_t\|+\epsilon}
\lesssim
\alpha_t
\leq
\frac{\chi}{\eta C_t^{\mathrm{ctrl}}+\epsilon}.
}
```

The left side is a usefulness constraint: if $\alpha_t$ is too small, the update falls below the fixed-point quantum $q_\Delta$ and learning silently dies. The right side is a stability constraint: if $\alpha_t$ is too large, the update may overshoot or saturate.

<TBox type="warning" title="Throttle is not a substitute for valid precision">

The global throttle cannot make an impossible fixed-point design possible. Good dtype selection and input capping are still part of a robust hardware design. The controller should be evaluated inside a plausible representational envelope, not as a replacement for one.

</TBox>

## Shared Metrics

Every ablation should log enough information to identify the failure mechanism.

| Metric | Why it matters |
|---|---|
| loss | task-level stability |
| output error | task-level recovery |
| $\|\theta_t\|$ | weight growth |
| $\|G_t\|$ | gradient growth |
| $\|\Delta\theta_t\|$ | update magnitude |
| $\alpha_t$ | controller action |
| $\eta_t^{\mathrm{eff}}=\alpha_t\eta$ | effective learning rate |
| $C_t^{\mathrm{ctrl}}$ | online curvature / sensitivity proxy |
| $\alpha_t\eta C_t^{\mathrm{ctrl}}$ | effective stability margin |
| rail pressure | saturation risk |
| update underflow rate | dead-zone risk |
| update cosine / phase | offline geometry diagnostic |

For tiny models, also log:

```math
\lambda_{\max}(H_t),
\qquad
\rho(I-\eta H_t),
\qquad
\rho(I-\alpha_t\eta H_t).
```

<TBox type="summary" title="Diagnostic caution">

Update cosine and phase are useful offline diagnostics, especially when comparing against clipping or $\kappa$ projection. They should not be assumed to be practical hardware controller signals.

</TBox>

## Success And Failure Criteria

A throttled run is successful only if it remains stable and keeps learning:

```math
\text{finite loss}
\quad
\land
\quad
\text{bounded rail pressure}
\quad
\land
\quad
\text{nonzero useful updates}
\quad
\land
\quad
\text{improving error}.
```

<TBox type="warning" title="Do not confuse survival with learning">

A controller that prevents divergence only by making every update vanish has not solved online training. It has converted instability into frozen adaptation.

</TBox>

## Recommended First Wave

| Order | Stress test | Reason |
|---|---|---|
| 1 | [`ST-001`](./st-001-distribution-shift.md) | matches the deployment story |
| 2 | [`ST-003`](./st-003-rail-saturation.md) | directly hardware-relevant |
| 3 | [`ST-004`](./st-004-update-dead-zone.md) | defines when throttle cannot help |
| 4 | [`ST-002`](./st-002-transient-spike.md) | tests runtime shock absorption |
| 5 | [`ST-006`](./st-006-naive-safety-mechanisms.md) | compares against simple fixes |

`ST-005` and `ST-007` should come after the basic fixed-point path is clear.
