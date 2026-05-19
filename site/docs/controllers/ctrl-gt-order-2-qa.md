---
title: "CTRL-GT-ORDER-2-QA"
sidebar_label: "CTRL-GT-ORDER-2-QA"
status:
  - planned
tags:
  - controller
  - global-throttle
  - quantization
last_modified: 2026-05-18
author: mbvalentin
---
# CTRL-GT-ORDER-2-QA
<PageMeta />
---

<TBox type="summary" title="Purpose">

`CTRL-GT-ORDER-2-QA` is the quantization-aware second-order global throttle. It tries to keep $\alpha_t$ inside the interval where learning is both stable and representable.

</TBox>

## Feasible Alpha Interval

Stability imposes an upper bound:

```math
\alpha_{\max,t}
=
\frac{\chi}
{\eta C_t^{\mathrm{ctrl}}+\epsilon}.
```

Fixed-point usefulness imposes a lower bound:

```math
\alpha_{\min,t}
=
\frac{q_\Delta}
{\eta\|G_t\|_2+\epsilon}.
```

Useful stable fixed-point learning requires:

```math
\boxed{
\alpha_{\min,t}
\leq
\alpha_t
\leq
\alpha_{\max,t}.
}
```

## Controller Dynamics

The nominal second-order update is:

```math
\begin{aligned}
v_{\alpha,t+1}
&=
\beta v_{\alpha,t}
+
k_\alpha
\left(
\chi-\eta\alpha_t C_t^{\mathrm{ctrl}}
\right), \\
\tilde{\alpha}_{t+1}
&=
\alpha_t+v_{\alpha,t+1}.
\end{aligned}
```

Then project into the feasible interval:

```math
\boxed{
\alpha_{t+1}
=
\operatorname{clip}
\left(
\tilde{\alpha}_{t+1},
\alpha_{\min,t},
\min(1,\alpha_{\max,t})
\right).
}
```

## Inputs

| Input | Meaning |
|---|---|
| $\Delta\theta_t^{\mathrm{raw}}$ | raw optimizer update |
| $\eta$ | base learning rate |
| $C_t^{\mathrm{ctrl}}$ | curvature / sensitivity proxy |
| $\chi$ | target stability margin |
| $\|G_t\|_2$ | gradient norm |
| $q_\Delta$ | update quantum |
| $k_\alpha,\beta$ | second-order controller parameters |

## State

| State | Meaning |
|---|---|
| $\alpha_t$ | throttle scalar |
| $v_{\alpha,t}$ | throttle velocity |

## Failure Detection

If:

```math
\alpha_{\min,t}
>
\alpha_{\max,t},
```

then the selected fixed-point update format cannot simultaneously provide stable and useful learning at that step.

<TBox type="warning" title="Controller limitation">

When the feasible interval is empty, the controller should report infeasibility. The solution is not another alpha value; the precision map, learning rate, update format, or data scaling must change.

</TBox>

## Implementation Status

Planned. This is the likely long-term controller for hardware-realistic fixed-point experiments.
