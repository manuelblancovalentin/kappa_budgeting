# Experiments

The first month of experiments focuses on small synthetic systems that expose when fixed-point online learning becomes unstable and whether dynamic closed-loop throttling prevents divergence while preserving update geometry.

## Experiment Table

| ID | Name | Status | Workspace | Main Question |
|---|---|---|---|---|
| 001 | Affine single dense | Planned | [notes.md](https://github.com/manuelblancovalentin/kappa_budgeting/blob/main/workspace/ablations/001_affine_single_dense/notes.md) | Can global throttling stabilize a known linear system after input drift? |
| 002 | Two-layer ReLU teacher/student | Planned | [notes.md](https://github.com/manuelblancovalentin/kappa_budgeting/blob/main/workspace/ablations/002_two_layer_relu/notes.md) | Can global throttling stabilize coupled layer dynamics without rotating the update? |

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

- `config.yaml`: settings, seed, precision, drift grid, and enabled controller mechanisms.
- `notes.md`: hypothesis, procedure, status, interpretation, and links to results.
- `notebooks/analysis.ipynb`: exploratory run and plots.
- `results/`: exported logs, CSV files, figures, or summaries.

When an exploratory notebook becomes stable, add `run.py` for reproducible batch sweeps.
