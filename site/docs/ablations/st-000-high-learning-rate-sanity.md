---
title: "ST-000: High Learning-Rate Sanity"
sidebar_label: "ST-000 High LR"
status:
  - preliminary
tags:
  - ablation-test
  - sanity
  - global-throttle
last_modified: 2026-05-18
author: mbvalentin
---
# ST-000: High Learning-Rate Sanity
<PageMeta />
---

<TBox type="summary" title="Purpose">

This is the minimal controllability sanity check. We intentionally choose a learning rate that is too large and verify that the global throttle can reduce the effective step size.

</TBox>

## Mechanism

The instability is induced by violating the local stability margin:

```math
\eta C_t > 2.
```

or, when the true Hessian is available:

```math
\eta\lambda_{\max}(H_t)>2.
```

## Expected Behavior

| Variant | Expected result |
|---|---|
| float, no throttle | diverges or oscillates |
| float, global throttle | stabilizes by reducing $\eta_t^{\mathrm{eff}}=\alpha_t\eta$ |
| fixed-point, no throttle | diverges faster or hits rails |
| fixed-point, global throttle | stabilizes if the quantization interval is feasible |

## Main Plots

- loss vs step,
- $\alpha_t$ vs step,
- $\eta_t^{\mathrm{eff}}$ vs step,
- $C_t^{\mathrm{ctrl}}$ vs step,
- $\alpha_t\eta C_t^{\mathrm{ctrl}}$ vs step.

<TBox type="warning" title="Interpretation">

This test proves that the controller can reduce an unsafe learning rate. It does not prove the deployment claim by itself. It should be treated as a sanity check.

</TBox>
