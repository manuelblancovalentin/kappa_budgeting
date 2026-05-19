---
title: "ST-001: Distribution Shift"
sidebar_label: "ST-001 Distribution Shift"
status:
  - preliminary
tags:
  - ablation-test
  - distribution-shift
  - global-throttle
last_modified: 2026-05-18
author: mbvalentin
---
# ST-001: Distribution / Operating-Point Shift
<PageMeta />
---

<TBox type="summary" title="Purpose">

This is the main deployment-realistic stress family. The model is stable under a nominal condition, then the deployed input distribution changes and the same learning rate becomes unsafe.

</TBox>

## Mechanism

Examples:

```math
x' = \gamma x,
\qquad
x' = \gamma x+\beta,
\qquad
x'_i = 0 \quad \text{for a failed sensor column}.
```

The design rule is:

```math
\eta C_{\mathrm{nominal}} < \chi
\qquad
\text{but}
\qquad
\eta C_{\mathrm{shifted}} > \chi.
```

The learning rate is valid before deployment shift and becomes unsafe only after the operating condition changes.

## Variants

| Variant | Purpose |
|---|---|
| no input cap, no throttle | expose the raw failure |
| input cap, no throttle | test whether capping alone is enough |
| no input cap, throttle | isolate learning-loop stabilization |
| input cap, throttle | test practical combined protection |

## Controller Hypothesis

The global throttle should respond to the increased closed-loop sensitivity:

```math
C_t^{\mathrm{ctrl}}\uparrow
\Rightarrow
\alpha_t\downarrow
\Rightarrow
\eta_t^{\mathrm{eff}}\downarrow.
```

## Main Plots

- loss and output error,
- input scale/drift schedule,
- $C_t^{\mathrm{ctrl}}$,
- $\alpha_t$ and $\eta_t^{\mathrm{eff}}$,
- rail pressure if fixed-point is enabled,
- stability margin $\alpha_t\eta C_t^{\mathrm{ctrl}}$.

<TBox type="warning" title="Input capping critique">

Input clipping can protect the forward signal, but it does not necessarily stabilize the closed-loop weight update. Valid experiments should compare cap-only, throttle-only, and cap-plus-throttle variants.

</TBox>
