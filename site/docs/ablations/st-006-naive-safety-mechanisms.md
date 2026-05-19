---
title: "ST-006: Naive Safety Mechanisms"
sidebar_label: "ST-006 Naive Safety"
status:
  - preliminary
tags:
  - ablation-test
  - baselines
  - clipping
last_modified: 2026-05-18
author: mbvalentin
---
# ST-006: Naive Safety Mechanisms
<PageMeta />
---

<TBox type="summary" title="Purpose">

This family compares global throttling against simple stabilizers from the literature and from practical engineering.

</TBox>

## Candidate Baselines

| Baseline | What it does | Main risk |
|---|---|---|
| input clipping | caps incoming samples | destroys input information |
| activation clipping | caps layer outputs | changes model function |
| gradient clipping | caps gradient magnitude | may distort descent direction |
| update clipping | caps parameter increments | changes optimizer dynamics |
| weight clipping | caps stored parameters | projects onto a box, can harm convergence |

## Geometry Risk

Elementwise clipping does not generally preserve direction:

```math
\operatorname{clip}(\Delta\theta_t)
\not\parallel
\Delta\theta_t.
```

Global norm clipping is less destructive:

```math
G_t^{\mathrm{clip}}
=
G_t
\min
\left(
1,
\frac{\tau}{\|G_t\|}
\right),
```

but it still acts directly on the gradient rather than on the closed-loop stability margin.

## Questions

- Can simple clipping prevent numerical failure?
- Does it preserve useful learning?
- Does it distort the actual update direction?
- Does global throttling recover better because it scales the full update uniformly?

## Main Plots

- loss,
- rail pressure,
- update norm,
- update cosine / phase,
- number of clipped elements,
- comparison against global throttle.
