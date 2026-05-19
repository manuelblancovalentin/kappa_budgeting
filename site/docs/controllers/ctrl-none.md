---
title: "CTRL-NONE"
sidebar_label: "CTRL-NONE"
status:
  - valid
tags:
  - controller
  - baseline
last_modified: 2026-05-18
author: mbvalentin
---
# CTRL-NONE
<PageMeta />
---

<TBox type="summary" title="Purpose">

`CTRL-NONE` is the baseline controller. It applies the raw optimizer update without any closed-loop throttling or safety intervention.

</TBox>

## Definition

The optimizer proposes:

```math
\Delta\theta_t^{\mathrm{raw}}
=
\mathrm{Optimizer}(G_t,\mathrm{state}_t).
```

The controller returns:

```math
\Delta\theta_t^{\mathrm{ctrl}}
=
\Delta\theta_t^{\mathrm{raw}}.
```

Equivalently:

```math
\alpha_t=1.
```

## Inputs

| Input | Meaning |
|---|---|
| $\Delta\theta_t^{\mathrm{raw}}$ | raw optimizer update |
| diagnostics | logged only, not used |

## Outputs

| Output | Meaning |
|---|---|
| $\Delta\theta_t^{\mathrm{ctrl}}$ | unchanged update |
| $\alpha_t$ | always 1 |

## Use

Use `CTRL-NONE` in every stress test as the uncontrolled baseline.

## Expected Failure Modes

Under fixed-point stress, this baseline may:

- diverge under high curvature,
- hit activation, gradient, update, or weight rails,
- corrupt optimizer state,
- continue with unstable effective learning rate.
