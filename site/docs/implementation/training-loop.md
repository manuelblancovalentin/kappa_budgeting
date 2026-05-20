---
status:
  - preliminary
tags:
  - implementation
  - training
  - global-throttle
last_modified: 2026-05-19
author: mbvalentin
source: "enabol/nn/training.py"
source_url: "https://github.com/manuelblancovalentin/kappa_budgeting/blob/master/enabol/nn/training.py"
---
# Custom Training Loop
<PageMeta />
---

<TBox type="summary" title="What this page covers">
This page summarizes the custom online-training loop used by the ablation harness. The public entry point is `BaseModel.train_instrumented(...)`; the implementation is orchestrated by `enabol.nn.training.InstrumentedTrainer`, with controller logic in `enabol.nn.controller`.
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

<div className="pseudo">
  <div className="pseudo-title">Algorithm: MinimalInstrumentedTrainingLoop</div>
  <div className="pseudo-code">

1. <span className="pseudo-kw">for</span> each online step $t$ <span className="pseudo-kw">do</span>
2. <span className="pseudo-indent-1">$x_b, y_b \leftarrow \operatorname{sample\_batch}()$</span>
3. <span className="pseudo-indent-1"><span className="pseudo-kw">with</span> `GradientTape` <span className="pseudo-kw">do</span></span>
4. <span className="pseudo-indent-2">$\hat{y}_b \leftarrow f(x_b; \theta_t)$</span>
5. <span className="pseudo-indent-2">$L_t \leftarrow \operatorname{loss}(y_b, \hat{y}_b)$</span>
6. <span className="pseudo-indent-1"><span className="pseudo-kw">end with</span></span>
7. <span className="pseudo-indent-1">$G_t \leftarrow \nabla_{\theta} L_t$</span>
8. <span className="pseudo-indent-1">$\theta_t \leftarrow \operatorname{flatten}(\text{trainable variables})$</span>
9. <span className="pseudo-indent-1">$\Delta_{\mathrm{raw}} \leftarrow -\eta G_t$</span>
10. <span className="pseudo-indent-1">$\alpha_t \leftarrow \operatorname{controller}(\theta_t, G_t, \theta_{t-1}, G_{t-1})$</span>
11. <span className="pseudo-indent-1">$\Delta_{\mathrm{actual}} \leftarrow \alpha_t \Delta_{\mathrm{raw}}$</span>
12. <span className="pseudo-indent-1">$\theta_{t+1} \leftarrow \theta_t + \Delta_{\mathrm{actual}}$</span>
13. <span className="pseudo-indent-1">$\operatorname{log\_step}(L_t, \theta_t, G_t, \alpha_t, \ldots)$</span>
14. <span className="pseudo-kw">end for</span>

  </div>
  <div className="pseudo-caption">This is the conceptual loop; `BaseModel.train_instrumented(...)` adds quantization hooks and diagnostics.</div>
</div>

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
