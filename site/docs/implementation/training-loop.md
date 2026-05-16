---
status:
  - preliminary
tags:
  - implementation
  - training
  - global-throttle
last_modified: 2026-05-15
source: "enabol/nn.py"
source_url: "https://github.com/manuelblancovalentin/kappa_budgeting/blob/main/enabol/nn.py"
---
# Custom Training Loop
<PageMeta />
---

<TBox type="summary" title="What this page covers">
This page summarizes the custom online-training loop used by the ablation harness. The implementation currently lives in `BaseModel.train_instrumented(...)`, while this page keeps the controller logic and logged diagnostics readable at a glance.
</TBox>

The first target is Experiment 1A: a one-layer, no-bias, floating-point linear regression test with scale drift.

## Design Goals

- Use Keras models for the trainable layers.
- Avoid `model.fit`.
- Own the update step explicitly.
- Flatten trainable parameters into one global vector `theta`.
- Flatten gradients into one global vector `G`.
- Compute the dynamic global throttle `alpha(t)`.
- Apply the full update vector with one shared scalar throttle.
- Log every diagnostic needed to explain closed-loop stability.

## Minimal Loop Shape

```text
for step in steps:
    x_batch, y_batch = sample_batch()

    with GradientTape:
        y_hat = model(x_batch)
        loss = mse(y_batch, y_hat)

    grads = tape.gradient(loss, model.trainable_variables)

    theta = flatten(model.trainable_variables)
    G = flatten(grads)

    delta_raw = -eta * G
    alpha = controller(theta, G, theta_prev, G_prev)
    delta_actual = alpha * delta_raw

    apply_flat_update(model.trainable_variables, delta_actual)
    log_step(...)
```

## First Model

```math
y = Ax
```

```math
\hat{y} = Wx
```

No bias is included in the first test. Bias is added only after the pure scale-drift Hessian story is validated.

## First Controller

```math
C(t) =
\frac{
  \lVert G(t) - G(t-1) \rVert
}{
  \lVert \theta(t) - \theta(t-1) \rVert + \varepsilon
}
```

```math
S(t) = \operatorname{EMA}(C(t))
```

```math
\alpha(t) =
\operatorname{clamp}
\left(
  \frac{1}{1 + \beta S(t)},
  \alpha_{\min},
  1
\right)
```

## Diagnostics

The first implementation should log:

- loss,
- output error,
- `||theta||`,
- `||G||`,
- raw update norm,
- actual update norm,
- curvature proxy `C(t)`,
- EMA signal `S(t)`,
- `alpha(t)`,
- effective learning rate `alpha(t) * eta`,
- true Hessian metrics where feasible,
- stability margin `eta * lambda_max(H)`,
- throttled stability margin `alpha(t) * eta * lambda_max(H)`,
- update cosine.

## Notes

Keep this page updated with code snippets once the harness stabilizes.
