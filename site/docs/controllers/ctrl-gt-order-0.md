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

## Implementation Status

This is the controller used in the first global-throttle sanity experiments.
