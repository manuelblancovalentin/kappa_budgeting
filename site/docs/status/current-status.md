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
last_modified: 2026-05-21
author: mbvalentin
---
# 📸 Current Snapshot
<PageMeta />

---

<TBox type="summary" title="Snapshot">
<ENABOL /> now has a cleaner formulation that connects the ICCAD $\kappa$-budgeting story to the global throttle work. $\kappa$-budgeting defines the fixed-point safety envelope: the admissible region where weights, activations, gradients, and updates remain bounded. Global throttling defines the trajectory controller inside that envelope: one scalar $\alpha_t$ scales the full parameter update so learning slows down when curvature or fixed-point pressure makes the current step too aggressive.
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

Then $\alpha_t$ is selected so the effective update is damped during high-curvature or numerically fragile regimes. The key design rule is that $\alpha_t$ is global across all trainable parameters.

## How κ and global throttle fit together

The project should no longer frame global throttling as replacing $\kappa$-budgeting. The combined view is:

```math
\boxed{\text{$\kappa$-budgeting defines the admissible fixed-point region.}}
```

```math
\boxed{\text{global throttle controls the learning speed inside that region.}}
```

Let $\mathcal{S}_{\kappa}$ be the set of parameters satisfying the selected $\kappa$ budgets. A direction-preserving controller can compute two global limits:

```math
\alpha_{\mathrm{curv},t}
=
\frac{\chi}
{\eta(\widehat{C}_t+\varepsilon)}
```

and

```math
\alpha_{\kappa,t}
=
\max \left\{
\alpha \in [0,1]:
\theta_t+\alpha\Delta\theta_t^{\mathrm{raw}}
\in
\mathcal{S}_{\kappa}
\right\}.
```

The applied scalar is:

```math
\alpha_t
=
\operatorname{clip}
\left(
\min(\alpha_{\mathrm{curv},t},\alpha_{\kappa,t}),
\alpha_{\min},
\alpha_{\max}
\right).
```

This keeps one global update direction:

```math
\theta_{t+1}
=
\theta_t+\alpha_t\Delta\theta_t^{\mathrm{raw}}.
```

The distinction is useful:

| Limiting signal | Meaning |
|---|---|
| $\alpha_{\kappa,t}<\alpha_{\mathrm{curv},t}$ | The proposed update would hit a fixed-point gain or rail budget first. $\kappa$ is limiting the step. |
| $\alpha_{\mathrm{curv},t}<\alpha_{\kappa,t}$ | The update fits the $\kappa$ envelope, but the local learning dynamics are too stiff. Curvature is limiting the step. |
| both near 1 | The update is inside the envelope and dynamically safe enough to apply normally. |

This reconciles the submitted $\kappa$-budgeting formulation with the new global-throttle direction-preservation result. The old row/column projection path remains useful for reproduction and comparison, but the preferred direction-preserving formulation is a $\kappa$-aware global alpha.

## What is currently proven

- <Badge status="valid" /> The floating-point one-layer sanity test shows that a global throttle can prevent divergence after input-gain drift.
- <Badge status="valid" /> The fake-fixed-point path works for staged quantization experiments: weights, updates, activations, rails, saturation counters, and update distortion plots.
- <Badge status="valid" /> The update-direction argument is clear: multiplying the flattened update by one scalar preserves the optimizer direction before fixed-point rounding and saturation.
- <Badge status="valid" /> The CSIM CTRL-NONE pipeline (half_mse → dense_backpass → sgd → global_throttle_none → apply_dense_update) compiles and trains a 1-Dense linear model with 0 errors (`CSIM-001`).
- <Badge status="preliminary" /> CTRL-GT-ORDER-0 is implemented in hls4ml as `global_throttle_order0` with internal curvature sensing; CSIM validation is pending (`CSIM-002`, `ENB-017`).
- <Badge status="preliminary" /> The $\kappa$-aware global-alpha formulation reconciles $\kappa$ safety rails with global throttling, but has not yet been implemented or validated in hls4ml.
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
- Global throttle controller with a simple curvature proxy (software: GT-0/1/2/2-QA).
- Single-layer affine dataset with input gain drift.
- Fake quantization of weights, updates, activations, and rails.
- hls4ml firmware chain for Dense: half_mse loss, dense_backpass, sgd optimizer, CTRL-NONE (α=1) controller, apply_dense_update.
- hls4ml CTRL-GT-ORDER-0 C++ kernel with curvature sensor + algebraic safe-alpha law (pending CSIM validation).
- CSIM-001: first compilation validation (1-Dense, batch-size-8, half_mse, sgd, CTRL-NONE, 0 errors).

## Main limitations

<TBox type="warning" title="Known limitations">

- Only the one-layer linear model has been tested end-to-end.
- Current quantization tests are software fake-fixed-point, not firmware-equivalent HLS traces.
- The global throttle law is still heuristic; we have not analytically tuned $\chi$, rail tightness, or precision maps.
- Legacy row/column $\kappa$ projection may rotate the update direction when it rescales different rows, columns, or layers differently.
- The $\kappa$-aware global alpha is still a formulation target, not a validated hardware implementation.

</TBox>

## Next actions

<TBox type="todo" title="Near-term TODOs">

- [x] Start the hls4ml implementation with the direction-preserving trainable path: `HLS4ML-001` through `HLS4ML-006` (**done**).
- [x] Keep the first correctness target narrow: one Dense layer, half-MSE loss, SGD, `CTRL-NONE`, `CTRL-GT-ORDER-0`.
- [x] Implement pure global throttle (CTRL-GT-ORDER-0) as the first non-trivial controller in hls4ml.
- [ ] Add the trainable config schema so `SafetyBudget` and `Controller` can be selected independently later.
- [ ] Preserve legacy row/column $\kappa$ projection as a selectable reproduction/comparison mode, not as the default learning controller.
- [ ] Run one-layer Dense CSIM comparison with CTRL-GT-ORDER-0 active (`CSIM-002` / `ENB-017`).
- [ ] Add $\kappa$-aware global alpha as the reconciliation mode.

</TBox>

## Things not to repeat

- [x] ~~Do not describe global throttle as abandoning $\kappa$-budgeting. The better framing is envelope plus trajectory controller.~~
- [x] ~~Do not lead with legacy row/column $\kappa$ projection as the main implementation path. It is useful as a reproduction/comparison mode, but it should not be the default controller.~~
- [x] ~~Do not mix curvature instability and rail/saturation instability in the first diagnostic experiment. The plots become hard to interpret.~~
- [x] ~~Do not treat update cosine as a hardware-available controller signal. It is a software diagnostic for update distortion.~~
