---
title: "ST-003: Rail Saturation"
sidebar_label: "ST-003 Rail Saturation"
status:
  - preliminary
tags:
  - ablation-test
  - fixed-point
  - saturation
last_modified: 2026-05-18
author: mbvalentin
---
# ST-003: Rail Saturation / Overflow
<PageMeta />
---

<TBox type="summary" title="Purpose">

This is the core fixed-point hardware failure family. It tests what happens when activations, gradients, weights, updates, or accumulators approach the representable rails.

</TBox>

## Mechanism

Fixed-point quantization clips values to the representable range:

```math
Q(z)
=
\operatorname{clip}
\left(
\operatorname{round}(z),
z_{\min},
z_{\max}
\right).
```

Objects to test separately:

| Object | Failure symptom |
|---|---|
| activations | forward information clips |
| gradients | backpropagated learning signal clips |
| weights | stored model state clips |
| updates | parameter increments clip |
| accumulators | batch sums clip before averaging |

## Rail Pressure

Log rail pressure per object:

```math
r_z(t)
=
\frac{
\#\{|z_i(t)| \geq \rho z_{\max}\}
}{
\#\{z_i(t)\}
}.
```

where $\rho$ is a near-rail threshold such as $0.95$.

## Controller Hypothesis

The throttle should reduce update pressure before the system repeatedly hits rails:

```math
\alpha_t\downarrow
\Rightarrow
\|\Delta\theta_t\|\downarrow
\Rightarrow
r_z(t)\downarrow.
```

<TBox type="warning" title="Information loss">

If the input itself is irreversibly clipped before the model sees it, the throttle cannot recover the lost information. A wider input dtype or input cap may still be required.

</TBox>
