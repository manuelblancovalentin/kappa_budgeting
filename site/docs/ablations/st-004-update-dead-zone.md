---
title: "ST-004: Update Dead-Zone"
sidebar_label: "ST-004 Update Dead-Zone"
status:
  - preliminary
tags:
  - ablation-test
  - quantization
  - underflow
last_modified: 2026-05-18
author: mbvalentin
---
# ST-004: Quantization Floor / Update Dead-Zone
<PageMeta />
---

<TBox type="summary" title="Purpose">

This stress test targets the opposite of explosion. Learning can be numerically stable but useless because the update is too small to be represented.

</TBox>

## Mechanism

The dead-zone condition is:

```math
\alpha_t\eta\|G_t\|
\lesssim
q_\Delta.
```

Expected symptom:

```math
\Delta\theta_t^{\mathrm{float}} \neq 0
\qquad
\text{but}
\qquad
Q_\Delta(\Delta\theta_t)=0.
```

## Why This Matters

A conservative throttle can accidentally kill learning. A useful controller must avoid both sides:

```math
\text{too large}
\Rightarrow
\text{unstable}
```

```math
\text{too small}
\Rightarrow
\text{dead update}.
```

## Main Plots

- update norm,
- update underflow rate,
- $\alpha_t$,
- effective update quantum threshold,
- loss plateau behavior.

<TBox type="warning" title="Controller limitation">

If the stable-useful interval is empty, no scalar throttle can save the selected precision. The dtype, learning rate, or update format must change.

</TBox>
