---
title: "ST-002: Transient Spike"
sidebar_label: "ST-002 Transient Spike"
status:
  - preliminary
tags:
  - ablation-test
  - transient
  - corruption
last_modified: 2026-05-18
author: mbvalentin
---
# ST-002: Transient Spike / Corrupted Batch
<PageMeta />
---

<TBox type="summary" title="Purpose">

This stress test injects a short-lived disturbance instead of a persistent distribution shift. It models sensor glitches, corrupted labels, rare outliers, bad batches, or soft-error-like events.

</TBox>

## Mechanism

Example input spike:

```math
x_t \leftarrow \gamma_{\mathrm{spike}}x_t
\quad
\text{for one batch}.
```

Example label corruption:

```math
y_t \leftarrow y_t+\epsilon_{\mathrm{spike}}
\quad
\text{for one batch}.
```

Expected failure chain:

```math
\text{abnormal batch}
\Rightarrow
\text{large } G_t
\Rightarrow
\text{large } \Delta\theta_t
\Rightarrow
\text{rail hit or optimizer-state contamination}.
```

## Controller Hypothesis

The throttle should absorb the transient:

```math
C_t^{\mathrm{ctrl}}\uparrow
\Rightarrow
\alpha_t\downarrow
\Rightarrow
\|\Delta\theta_t\|\downarrow.
```

Then, after the disturbance passes, $\alpha_t$ should recover toward its nominal value.

## Main Plots

- loss around the spike window,
- gradient norm around the spike,
- update norm,
- $\alpha_t$ recovery,
- rail pressure,
- Adam state norms if Adam is used.
