---
title: "CTRL-GT-ORDER-1"
sidebar_label: "CTRL-GT-ORDER-1"
status:
  - planned
tags:
  - controller
  - global-throttle
last_modified: 2026-05-18
author: mbvalentin
---
# CTRL-GT-ORDER-1
<PageMeta />
---

<TBox type="summary" title="Purpose">

`CTRL-GT-ORDER-1` treats $\alpha_t$ as a first-order dynamic state. It should smooth the instantaneous jumps of `CTRL-GT-ORDER-0`.

</TBox>

## Definition

Define the stability margin:

```math
m_t
=
\eta\alpha_t C_t^{\mathrm{ctrl}}.
```

The margin error is:

```math
e_t
=
\chi-m_t.
```

The controller update is:

```math
\boxed{
\alpha_{t+1}
=
\alpha_t
+
k_\alpha
\left(
\chi-\eta\alpha_t C_t^{\mathrm{ctrl}}
\right).
}
```

Then clamp:

```math
\alpha_{t+1}
\leftarrow
\operatorname{clip}(\alpha_{t+1},\alpha_{\min},1).
```

## Inputs

| Input | Meaning |
|---|---|
| $\Delta\theta_t^{\mathrm{raw}}$ | raw optimizer update |
| $\eta$ | base learning rate |
| $C_t^{\mathrm{ctrl}}$ | curvature / sensitivity proxy |
| $\chi$ | target stability margin |
| $k_\alpha$ | controller adaptation gain |
| $\alpha_{\min}$ | optional floor |

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

## Stability Note

For locally constant $C_t^{\mathrm{ctrl}}=C$, stable $\alpha$ adaptation requires:

```math
0<k_\alpha\eta C<2.
```

For non-oscillatory adaptation:

```math
0<k_\alpha\eta C<1.
```

## Expected Behavior

This controller should react more smoothly than `CTRL-GT-ORDER-0`, at the cost of slower response to sudden spikes.

## Implementation Status

Planned.
