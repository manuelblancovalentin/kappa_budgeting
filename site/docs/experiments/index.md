# Experiments Index
<StatusBadges statuses="valid;inprogress" />
---

The first month of experiments focuses on small synthetic systems that expose when fixed-point online learning becomes unstable and whether dynamic closed-loop throttling prevents divergence while preserving update geometry.


| ID | Name | Status | Dataset ID | Model ID | Workspace `workspace/ablations` | Main Question |
|---|---|:---:|---|---|---|---|
| 000AB | [One-layer global throttle sanity check](./000-global-throttle-sanity.md) | <Badge status="valid" /> | [`DS-AFFINE-LINEAR-000`](../datasets/affine-linear-000.md) | [`MDL-DENSE1-LINEAR-NOBIAS-000`](../models/dense1-linear-nobias-000.md) | [`.../000_global_throttle_sanity`](https://github.com/manuelblancovalentin/kappa_budgeting/blob/102be154f09eef96536ea3dc6ca5ec7a16756979/workspace/ablations/000_global_throttle_sanity) | Can a global controller throttle the total learning rate and prevent one-layer divergence? |
| 000C | [One-layer global throttle with quantization](./000-global-throttle-sanity-quantization.md) | <Badge status="preliminary" /> | [`DS-AFFINE-LINEAR-000`](../datasets/affine-linear-000.md) | [`MDL-DENSE1-LINEAR-NOBIAS-000`](../models/dense1-linear-nobias-000.md) | `.../000_global_throttle_sanity` | Does throttling still help when the one-layer loop includes fake-fixed-point weights, updates, activations, and rails? |
| 001A | One-layer no-bias float scale drift | <Badge status="planned" /> | `DS-AFFINE-LINEAR-000` | `MDL-DENSE1-LINEAR-NOBIAS-000` | [notes.md](https://github.com/manuelblancovalentin/kappa_budgeting/blob/main/workspace/ablations/001a_one_layer_no_bias_float_scale_drift/notes.md) | Does the controller enforce the known Hessian stability boundary under pure scale drift? |
| 001B | One-layer fake fixed-point rails | <Badge status="planned" /> | `DS-AFFINE-LINEAR-000` | `MDL-DENSE1-LINEAR-NOBIAS-000` | TBD | Does throttling still help when instability is caused by quantization and saturation? |
| 002A | Two-layer linear network | <Badge status="planned" /> | TBD | TBD | TBD | Can global throttling stabilize inter-layer coupling without activation nonlinearities? |
| 002B | Two-layer ReLU teacher/student | <Badge status="planned" /> | TBD | TBD | [notes.md](https://github.com/manuelblancovalentin/kappa_budgeting/blob/main/workspace/ablations/002_two_layer_relu/notes.md) | Can global throttling stabilize coupled nonlinear layer dynamics without rotating the update? |

## Shared Hypothesis

There exists a region of drift parameters `(alpha, beta)` where ordinary fixed-point online training fails because some combination of activations, gradients, updates, or weights saturates. A useful dynamic controller should keep training bounded in this region while preserving the global descent direction.

## Drift Model

The starting drift model is affine input drift:

```text
x_drift = alpha x + beta
```

Two versions should be tested:

```text
x_drift = alpha x + beta
x_drift = clip(alpha x + beta, x_min, x_max)
```

The unclipped version isolates range expansion. The clipped version models sensor rails and information loss.

## Common Metrics

Every experiment should report:

| Metric | Why it matters |
|---|---|
| Loss curve | Shows recovery, divergence, or training death. |
| Saturation count | Direct evidence of fixed-point failure. |
| Activation range | Shows forward rail pressure. |
| Gradient range | Shows backward rail pressure. |
| Weight norm | Shows parameter growth. |
| Gradient norm | Shows update-field magnitude. |
| Update norm | Shows effective step size. |
| Curvature proxy `C(t)` | Estimates local closed-loop gain. |
| Global throttle `alpha(t)` | Shows controller intervention. |
| Update cosine | Measures whether budgeting preserves descent direction. |
| Hessian metrics | Validates whether `C(t)` tracks true curvature in toy models. |

## Near-Term Roadmap

| Step | Goal | Adds |
|---|---|---|
| 000 | Sanity-check instrumentation and controller | One layer, no bias, float, exploratory drift. |
| 000B | Add the first fake-fixed-point path | Quantized weights, updates, activations, saturation counters, and update underflow checks. |
| 001A | Clean curvature-only stability test | Pure scale drift, Hessian-selected learning rates. |
| 001B | Numerical hardware-style failure | Fake fixed-point quantization, rails, saturation counters. |
| 001C | More realistic affine drift | Bias and scale+shift drift. |
| 002A | Coupled linear dynamics | Two Dense layers, no activation. |
| 002B | Nonlinear hidden representation | Two Dense layers with ReLU. |
| 003 | ENABOL comparison | Loose kappa rails, legacy row/column projection if useful. |

## Initial Variant Matrix

| Variant | Purpose |
|---|---|
| Floating reference | Establish expected behavior without fixed-point limits. |
| Fixed-point baseline | Find drift regimes where online learning fails. |
| Dynamic global throttle | Test closed-loop stabilization with a single shared update scalar. |
| Loose kappa + throttle | Test static safety rails plus dynamic update control. |
| Global static kappa scale | Test representational gain control without row/layer direction changes. |
| Legacy row/column projection | Optional comparison only; do not rebuild first. |

## Documentation Rule

Each experiment workspace must contain:

- `config.yaml`: settings, seed, dataset ID, model ID, precision map, drift grid, and enabled controller mechanisms.
- `notes.md`: hypothesis, procedure, status, interpretation, and links to results.
- `notebooks/analysis.ipynb`: exploratory run and plots.
- `results/`: exported logs, CSV files, figures, or summaries.

Each experiment documentation page should start with an `Experiment Trace` table that names the dataset ID, model ID, training method, precision mode, and drift/corruption model. Do not duplicate full dataset or model derivations unless the experiment intentionally specializes them.

When an exploratory notebook becomes stable, add `run.py` for reproducible batch sweeps.
