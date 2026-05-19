---
title: "CTRL-GT-ORDER-2"
sidebar_label: "CTRL-GT-ORDER-2"
status:
  - planned
tags:
  - controller
  - global-throttle
last_modified: 2026-05-18
author: mbvalentin
---
# CTRL-GT-ORDER-2
<PageMeta />
---

<TBox type="summary" title="Purpose">

`CTRL-GT-ORDER-2` gives the throttle its own velocity. It is a damped second-order controller for smoother alpha adaptation.

</TBox>

## Continuous-Time Form

```math
\ddot{\alpha}
+
k_d\dot{\alpha}
+
k_p
\left(
\eta\alpha C^{\mathrm{ctrl}}-\chi
\right)
=
0.
```

For locally constant $C^{\mathrm{ctrl}}$, the equilibrium is:

```math
\alpha^\star
=
\frac{\chi}
{\eta C^{\mathrm{ctrl}}}.
```

## Discrete-Time Form

A practical discrete controller is:

```math
\boxed{
\begin{aligned}
v_{\alpha,t+1}
&=
\beta v_{\alpha,t}
+
k_\alpha
\left(
\chi-\eta\alpha_t C_t^{\mathrm{ctrl}}
\right), \\
\alpha_{t+1}
&=
\alpha_t+v_{\alpha,t+1}.
\end{aligned}
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
| $k_\alpha$ | proportional gain on margin error |
| $\beta$ | velocity retention / damping parameter |
| $\alpha_{\min}$ | optional floor |

## State

| State | Meaning |
|---|---|
| $\alpha_t$ | throttle scalar |
| $v_{\alpha,t}$ | throttle velocity |

## Expected Behavior

This controller should reduce ringing and abrupt alpha changes when tuned well. It has more knobs than the order-0 and order-1 controllers, so it should be tested after simpler controllers are understood.

## Known Failure Modes

- underdamped oscillation if damping is too low,
- slow response if damping is too high,
- still unaware of fixed-point update floors unless quantization-aware logic is added.

## Implementation Status

Planned.
