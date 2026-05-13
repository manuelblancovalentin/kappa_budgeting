# Experiment 1A: One-Layer No-Bias Float Scale Drift

Status: Planned  
Owner: Manuel Blanco-Valentin  
Workspace: `workspace/ablations/001a_one_layer_no_bias_float_scale_drift`

## Purpose

Validate the closed-loop stability story in the simplest setting where the Hessian and stability boundary are analytically interpretable.

This experiment is intentionally not fixed-point yet. It tests whether the dynamic global throttle behaves correctly when instability comes only from curvature and learning-rate mismatch after input scale drift.

## Setup

Teacher:

```math
y = Ax
```

Student:

```math
\hat{y} = Wx
```

Loss:

```math
L = \frac{1}{N}\lVert \hat{y} - y \rVert_2^2
```

Scale drift:

```math
x' = \gamma x
```

For this first test, do not include bias and do not include shift drift. The goal is to keep the curvature story clean:

```math
\lambda_{\max}(H_{\mathrm{drift}}) \approx \gamma^2 \lambda_{\max}(H_{\mathrm{nom}})
```

## Main Hypothesis

After drift, a learning rate that was stable under nominal input can become unstable when:

```math
\eta \lambda_{\max}(H_{\mathrm{drift}}) > 2
```

The dynamic global throttle should reduce the effective learning rate:

```math
\eta_{\mathrm{eff}}(t) = \alpha(t)\eta
```

so that:

```math
\eta_{\mathrm{eff}}(t)\lambda_{\max}(H(t)) < 2
```

## Planned Comparisons

| Variant | Purpose |
|---|---|
| Float baseline, stable LR | Sanity check the model and data. |
| Float baseline, unstable LR | Confirm the known Hessian stability boundary. |
| Float + dynamic global throttle | Test whether `alpha(t)` stabilizes the unstable LR. |

## Required Plots

- Loss vs step.
- Output error vs step.
- Gradient norm vs step.
- Raw and actual update norm vs step.
- Curvature proxy `C(t)` vs true `lambda_max(H)`.
- `eta * lambda_max(H)` and `alpha(t) * eta * lambda_max(H)` with a horizontal line at `2`.
- Effective learning rate `eta_eff(t)` and `2 / lambda_max(H(t))`.
- Update cosine.

## Implementation Notes

Use a Keras model with a custom training loop. Avoid `model.fit` so the loop can flatten parameters, compute diagnostics, and apply the global throttle manually.

The first trainer only needs SGD. Adam and fixed-point emulation come later.

## Open Questions

- What values of `gamma` create a clean before/after stability transition?
- Should the initial model be pretrained to convergence before drift, or should online training begin from a partially trained state?
- What controller `beta` gives readable intervention without collapsing `alpha(t)` to `alpha_min`?
