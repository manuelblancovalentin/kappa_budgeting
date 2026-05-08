# Experiment 002: Two-Layer ReLU Teacher/Student

Status: Planned  
Owner: Manuel Blanco-Valentin  
Workspace: `workspace/ablations/002_two_layer_relu`

## Goal

Introduce one hidden activation and one inter-layer gradient path while keeping the system synthetic and inspectable. This experiment is meant to reveal whether a global closed-loop throttle can stabilize coupled layer dynamics without rotating the optimizer update.

The main controller under test is:

```text
Delta theta(t) = alpha(t) (-eta G(t))
```

where the same scalar `alpha(t)` is applied to all layers.

## Model

Teacher:

```text
y = A2 relu(A1 x + c1) + c2
```

Student:

```text
z1 = W1 x + b1
a1 = relu(z1)
y_hat = W2 a1 + b2
L = mean((y_hat - y)^2)
```

Drift:

```text
x_drift = alpha x + beta
```

## Diagram

```mermaid
flowchart LR
    X["X"] -->|x| L1["Dense 1<br/>W1, b1"]
    L1 -->|z1 = W1 x + b1| R1["ReLU"]
    R1 -->|a1 = relu(z1)| L2["Dense 2<br/>W2, b2"]
    L2 -->|y_hat = W2 a1 + b2| LOSS["MSE Loss<br/>L(y_hat, y)"]

    LOSS -->|g_y| B2["Backpass Dense 2<br/>g_W2, g_b2, g_a1"]
    B2 -->|g_a1| BR["Backpass ReLU<br/>g_z1 = g_a1 1[z1 > 0]"]
    BR -->|g_z1| B1["Backpass Dense 1<br/>g_W1, g_b1, g_x"]

    B2 -->|update W2, b2| L2
    B1 -->|update W1, b1| L1
```

## Hypothesis

The two-layer model should expose closed-loop coupling that the single-layer model cannot. If unthrottled fixed-point training fails after drift, dynamic global throttling should reduce effective step size while preserving the global descent direction:

```text
cos(Delta_actual, -G) ~= 1
```

Legacy row/column projection remains an optional comparison. The immediate priority is not to rebuild per-row or per-column kappa machinery, because the new hypothesis is about global closed-loop update control.

## Planned Procedure

1. Generate a fixed teacher network and dataset.
2. Train the student around `x in [0, 1]`.
3. Simulate fixed-point online training without drift.
4. Sweep `(alpha, beta)` to find a baseline failure region.
5. Compare dynamic global throttle, loose kappa plus throttle, and global static kappa scaling in the failure region.
6. Compare per-layer norms while keeping `alpha(t)` global.
7. Optionally compare legacy row/column projection if already available.

## Required Results

| Metric | Description |
|---|---|
| Global throttle | `alpha(t)` over time. |
| Per-layer weight norms | Growth or collapse in `dense_1` and `dense_2`. |
| Per-layer gradient norms | Coupled update pressure in `dense_1` and `dense_2`. |
| Per-layer update norms | Effective update size after global throttling. |
| Hidden activation range | Whether `z1` or `a1` saturates or dies. |
| Gradient range | Whether `g_z1` grows or vanishes. |
| Update cosine | Direction preservation per layer and globally. |
| Curvature proxy | Global `C(t)` and optional per-layer `C_l(t)`. |
| Recovery loss | Whether online training adapts after drift. |

## Related Documentation

- Site overview: `site/docs/background/overview.md`
- Architecture: `site/docs/implementation/architecture.md`
- Experiment index: `site/docs/experiments/index.md`
