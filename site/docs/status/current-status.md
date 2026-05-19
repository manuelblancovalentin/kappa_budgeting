---
sidebar_label: "📸 Current Snapshot"
sidebar_position: 2
status:
  - preliminary
  - inprogress
tags:
  - status
  - experiment
  - global-throttle
last_modified: 2026-05-15
author: mbvalentin
---
# 📸 Current Snapshot
<PageMeta />

---

<TBox type="summary" title="Snapshot">
<ENABOL /> ablation work has pivoted from hard row/column $\kappa$ projection toward a closed-loop stability controller. The current algorithm keeps static $\kappa$-style constraints as loose representational rails, but stabilizes online learning with one global throttle $\alpha_t$ that scales the full parameter update. This preserves the optimizer direction while reducing effective learning rate when curvature, saturation, or quantization pressure increases.
</TBox>


## Current formulation

At each online step, the raw update is:

```math
\Delta\theta_t^{\mathrm{raw}}=-\eta G_t.
```

The controlled update is:

```math
\Delta\theta_t^{\mathrm{ctrl}}=\alpha_t\Delta\theta_t^{\mathrm{raw}},
\qquad 0 < \alpha_t \le 1.
```

The controller estimates local closed-loop sensitivity with:

```math
C_t =
\frac{\lVert G_t-G_{t-1}\rVert}
{\lVert \theta_t-\theta_{t-1}\rVert + \varepsilon}.
```

Then $\alpha_t$ is selected so the effective update is damped during high-curvature or numerically fragile regimes. The key design rule is that $\alpha_t$ is global across all layers.

## What is currently proven

- <Badge status="valid" /> The floating-point one-layer sanity test shows that a global throttle can prevent divergence after input-gain drift.
- <Badge status="valid" /> The fake-fixed-point path works for staged quantization experiments: weights, updates, activations, rails, saturation counters, and update distortion plots.
- <Badge status="preliminary" /> Quantized one-layer tests support the controller idea, but tight update precision can still distort late-stage learning when gradients become small.
- <Badge status="preliminary" /> The phase/cosine diagnostics are useful as software observability tools, but should not be treated as direct hardware controller inputs yet.

## Tested systems

| Item | Current value |
|---|---|
| Dataset | [`DS-AFFINE-LINEAR-000`](../datasets/affine-linear-000.md) |
| Model | [`MDL-DENSE1-LINEAR-NOBIAS-000`](../models/dense1-linear-nobias-000.md) |
| Float experiment | [`EXP-000A`](../experiments/exp-000a-global-throttle-float-lin1.md) |
| Quantized experiment | [`EXP-000B`](../experiments/exp-000b-global-throttle-qfx-lin1.md) |
| Workspace | `workspace/ablations/exp_000_global_throttle_sanity/` |

## What currently works

- Custom training loop (loss, norms, curvature proxy, throttle, metrics...).
- Global throttle controller with a simple curvature proxy.
- Single-layer affine dataset with input gain drift.
- Fake quantization of weights, updates, activations, and rails.

## Main limitations

<TBox type="warning" title="Known limitations">

- Only the one-layer linear model has been tested end-to-end.
- Current quantization tests are software fake-fixed-point, not firmware-equivalent HLS traces.
- The global throttle law is still heuristic; we have not analytically tuned $\chi$, rail tightness, or precision maps.
- Row/column $\kappa$ projection is intentionally not the priority right now because it may rotate the update direction.

</TBox>

## Next actions

<TBox type="todo" title="Near-term TODOs">

- [ ] Add the clean curvature-only `EXP-001A` run with Hessian-selected learning rates.
- [ ] Add the rail/saturation-focused `EXP-001B` run with tighter fixed-point formats.
- [ ] Move to a two-layer linear model to expose inter-layer coupling without activation nonlinearities.
- [ ] Add a first precision-map registry once multiple quantization configurations are reused.
- [ ] Clone latest `hls4ml` and modify it to start synthesizing the current controller algorithm.

</TBox>

## Things not to repeat

- [x] ~~Do not lead with legacy row/column $\kappa$ projection as the main ablation path. It is useful as a later comparison, but it is not the cleanest first controller story.~~
- [x] ~~Do not mix curvature instability and rail/saturation instability in the first diagnostic experiment. The plots become hard to interpret.~~
- [x] ~~Do not treat update cosine as a hardware-available controller signal. It is a software diagnostic for update distortion.~~
