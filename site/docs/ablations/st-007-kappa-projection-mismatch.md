---
title: "ST-007: κ Projection Mismatch"
sidebar_label: "ST-007 κ Projection"
status:
  - preliminary
tags:
  - ablation-test
  - kappa-budgeting
  - legacy
last_modified: 2026-05-18
author: mbvalentin
---
# ST-007: Legacy κ Projection Mismatch
<PageMeta />
---

<TBox type="summary" title="Purpose">

This is not a natural external failure mode. It is a comparison against the original <ENABOL /> stabilizer.

</TBox>

## Mechanism

The old row/column $\kappa$ mechanisms constrain representational gain and rail pressure, but the applied update may become:

```math
\Delta\theta_t^{\mathrm{actual}}
=
\Pi_\kappa
\left(
\theta_t-\eta G_t
\right)
-\theta_t.
```

In general:

```math
\Delta\theta_t^{\mathrm{actual}}
\not\parallel
-\eta G_t.
```

The global throttle instead applies:

```math
\Delta\theta_t^{\mathrm{ctrl}}
=
\alpha_t
\Delta\theta_t^{\mathrm{raw}},
```

so the intended update direction is preserved before fixed-point quantization effects.

## Questions

- Does row/column projection keep rails safe?
- Does it rotate or distort the learning trajectory?
- Is the loss stable but learning degraded?
- Does global throttle recover stability with less geometry distortion?

## Main Plots

- loss,
- rail pressure,
- row/column scale events,
- update cosine / phase,
- per-layer update norm,
- global throttle comparison.

## Priority

This belongs late in the ablation ladder. It explains why the project moved away from aggressive row/column projection, but it should not be presented as the main way standard fixed-point training fails.
