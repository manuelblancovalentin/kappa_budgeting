# Experiments

The first month of experiments focuses on small synthetic systems that expose when fixed-point online learning becomes unstable and which kappa-budgeting mechanisms prevent it.

## Experiment Table

| ID | Name | Status | Workspace | Main Question |
|---|---|---|---|---|
| 001 | Affine single dense | Planned | [notes.md](https://github.com/manuelblancovalentin/kappa_budgeting/blob/main/workspace/ablations/001_affine_single_dense/notes.md) | Can budgeting stabilize a known linear system after input drift? |
| 002 | Two-layer ReLU teacher/student | Planned | [notes.md](https://github.com/manuelblancovalentin/kappa_budgeting/blob/main/workspace/ablations/002_two_layer_relu/notes.md) | How do budgets and throttling behave across layers with an intermediate activation? |

## Shared Hypothesis

There exists a region of drift parameters `(alpha, beta)` where ordinary fixed-point online training fails because some combination of activations, gradients, updates, or weights saturates. A useful kappa-budgeting method should keep training bounded in this region while preserving enough update direction to continue learning.

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
| Row L1 norms | Measures forward sensitivity and row-budget pressure. |
| Column L1 norms | Measures backward sensitivity and column-budget pressure. |
| Throttle shifts | Shows where update control intervenes. |
| Projection shifts | Shows where post-update projection intervenes. |
| Update cosine | Measures whether budgeting preserves descent direction. |

## Initial Variant Matrix

| Variant | Purpose |
|---|---|
| Floating reference | Establish expected behavior without fixed-point limits. |
| Fixed-point baseline | Find drift regimes where online learning fails. |
| Throttle only | Test whether bounded updates are enough. |
| Column scaling only | Test whether backward sensitivity control is enough. |
| Exact row projection | Test the mathematical projection without shift coarseness. |
| Power-of-two row projection | Test the current hardware-friendly approximation. |
| Full local budgeting | Combine row, column, and throttling locally. |
| Global uniform scaling | Test whether preserving parameter direction helps. |

## Documentation Rule

Each experiment workspace must contain:

- `config.yaml`: settings, seed, precision, drift grid, and enabled budgeting mechanisms.
- `notes.md`: hypothesis, procedure, status, interpretation, and links to results.
- `notebooks/analysis.ipynb`: exploratory run and plots.
- `results/`: exported logs, CSV files, figures, or summaries.

When an exploratory notebook becomes stable, add `run.py` for reproducible batch sweeps.
