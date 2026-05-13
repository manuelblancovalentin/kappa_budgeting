---
id: global-throttle-sanity-quantization
title: "Experiment 000C: Global Throttle With Quantization"
sidebar_label: "000C: Throttle + Quantization"
---

# Experiment 000C: Global Throttle With Quantization

<p className="status-line">
  Status:
  <span className="status-badge status-badge--preliminary">Preliminary</span>
  <span className="status-badge status-badge--planned">Planned</span>
</p>
Workspace: `workspace/ablations/000_global_throttle_sanity/`  
Starting point: [Experiment 000](./000-global-throttle-sanity.md)

## Purpose

Experiment 000 showed that a dynamic global throttle can stabilize a one-layer floating-point online learning loop after input-gain drift. Experiment 000C adds the first hardware-style numerical effects:

- finite precision weights,
- finite precision updates,
- optional finite precision activations,
- clipping or wrapping at fixed-point rails,
- saturation and underflow diagnostics.

The goal is not yet to reproduce the full ENABOL firmware path. The goal is narrower:

> Can the same global throttle stabilize a one-layer online learner when the update path is fake-fixed-point and the failure mode is numerical saturation, update underflow, or both?

## Base System

We keep the same one-layer no-bias teacher/student system:

```math
x \sim \mathcal{U}([0,1]^{d_{\mathrm{in}}})
```

```math
y = Ax
```

```math
\hat{y} = Wx
```

with loss:

```math
\mathcal{L}
=
\frac{1}{2N}
\left\|
\hat{Y}-Y
\right\|_F^2.
```

The nominal online update is:

```math
\Delta W_{\mathrm{raw}}(t)
=
-\eta G_W(t).
```

The controlled floating-point update is:

```math
\Delta W_{\mathrm{ctrl}}(t)
=
-\alpha_t \eta G_W(t).
```

Experiment 000C replaces parts of this update with fake-fixed-point operators.

## Fixed-Point Operator

A signed fixed-point type is described as:

```text
ap_fixed<WL, IWL, QMODE, OMODE>
```

where:

- `WL` is the total word length,
- `IWL` is the integer word length including the sign bit,
- `F = WL - IWL` is the number of fractional bits,
- `q = 2^{-F}` is the quantization step,
- `QMODE` controls rounding,
- `OMODE` controls overflow behavior.

For a signed fixed-point type, the real-valued rail interval is approximately:

```math
x_{\min}
=
-2^{IWL-1}
```

```math
x_{\max}
=
2^{IWL-1}-2^{-F}.
```

The fake-fixed-point quantizer is:

```math
Q_T(x)
=
q \cdot
\operatorname{overflow}_{T}
\left(
\operatorname{round}_{T}
\left(
\frac{x}{q}
\right)
\right).
```

For saturation mode:

```math
\operatorname{overflow}_{T}(z)
=
\operatorname{clip}
\left(
z,
-2^{WL-1},
2^{WL-1}-1
\right).
```

For wrap mode, the integer value wraps modulo:

```math
2^{WL}.
```

The first ablation should use saturation mode first:

```text
OMODE = AP_SAT
```

because saturation is easier to interpret than two's-complement wraparound.

## Quantized Training Equations

The first fake-fixed-point update path should be explicit and staged. A useful general form is:

```math
W_q(t)
=
Q_W(W(t)).
```

```math
\hat{y}_q(t)
=
Q_Y
\left(
Q_{\mathrm{acc}}
\left(
Q_X(x_t) W_q(t)^T
\right)
\right).
```

The gradient is computed from the quantized forward path:

```math
G_q(t)
=
\nabla_W
\mathcal{L}
\left(
\hat{y}_q(t),
y_t
\right).
```

The global throttle is computed as in Experiment 000:

```math
C_t
=
\frac{
\left\|G_q(t)-G_q(t-1)\right\|_2
}{
\left\|W(t)-W(t-1)\right\|_2+\varepsilon
}.
```

```math
\alpha_t
=
\min
\left(
1,
\frac{\chi}{\eta C_t^{\mathrm{ctrl}}+\varepsilon}
\right).
```

Then the quantized update is:

```math
\Delta W_q(t)
=
Q_{\Delta}
\left(
-\alpha_t\eta G_q(t)
\right).
```

and the stored weight becomes:

```math
W(t+1)
=
Q_W
\left(
W(t)+\Delta W_q(t)
\right).
```

This is the main software model of the hardware update path.

## Useful Learning Interval

Quantization introduces a lower bound on the useful update size. If the update quantum is:

```math
q_{\Delta}
=
2^{-F_{\Delta}},
```

then a rough useful-update condition is:

```math
\alpha_t \eta \left\|G_q(t)\right\|_2
\gtrsim
q_{\Delta}.
```

Stability still gives an upper bound:

```math
\alpha_t \eta C_t^{\mathrm{ctrl}}
\le
\chi.
```

So useful stable fixed-point learning requires a nonempty interval:

```math
\boxed{
\frac{
q_{\Delta}
}{
\eta\left\|G_q(t)\right\|_2+\varepsilon
}
\lesssim
\alpha_t
\le
\frac{
\chi
}{
\eta C_t^{\mathrm{ctrl}}+\varepsilon
}.
}
```

This is one of the main quantities to log in Experiment 000C.

## Rail Statistics

For each quantized tensor `z`, log:

```math
r_{\mathrm{sat}}(z)
=
\frac{
\#\{i: z_i \le z_{\min} \;\lor\; z_i \ge z_{\max}\}
}{
\#\{i: z_i\}
}.
```

Also log near-rail pressure:

```math
r_{\mathrm{near}}(z)
=
\frac{
\#\{i: |z_i| \ge \rho z_{\max}\}
}{
\#\{i: z_i\}
},
\qquad
0<\rho<1.
```

The first value of `rho` should be:

```math
\rho = 0.95.
```

Track these for:

- inputs,
- activations,
- weights,
- gradients,
- raw updates,
- applied quantized updates,
- outputs.

## Implementation Plan

The first implementation should not be a custom Keras layer. Use quantizer hooks in the custom training loop first.

The reason is diagnostic control. We need to turn quantization on and off independently for each tensor family:

| Hook | Tensor | First purpose |
|---|---|---|
| `input_quantizer` | `x` | Model sensor/input precision. |
| `weight_quantizer` | `W` | Model stored parameter precision. |
| `activation_quantizer` | `y_hat` or intermediate activations | Model forward rails. |
| `gradient_quantizer` | `G` | Model backward-path precision. |
| `update_quantizer` | `alpha * eta * G` | Model optimizer/update precision. |
| `accumulator_quantizer` | dot-product accumulator | Model MAC accumulation rails. |

The training loop should expose a configuration object:

```python
quant_config = QuantizationConfig(
    enabled=True,
    input_dtype="ap_fixed<12,4,AP_RND,AP_SAT>",
    weight_dtype="ap_fixed<12,4,AP_RND,AP_SAT>",
    activation_dtype="ap_fixed<12,5,AP_RND,AP_SAT>",
    gradient_dtype="ap_fixed<16,6,AP_RND,AP_SAT>",
    update_dtype="ap_fixed<16,4,AP_RND,AP_SAT>",
    accumulator_dtype="ap_fixed<24,10,AP_RND,AP_SAT>",
)
```

The minimum training-loop shape is:

```text
for step in online_steps:
    x_q = Qx(x)
    W_q = Qw(W)

    with GradientTape:
        y_hat = model_forward(x_q, W_q)
        y_hat_q = Qy(y_hat)
        loss = mse(y, y_hat_q)

    G = gradient(loss, W)
    G_q = Qg(G)

    alpha = global_throttle(W, G_q, W_prev, G_prev)
    delta_raw = -alpha * eta * G_q
    delta_q = Qdelta(delta_raw)

    W_next = Qw(W + delta_q)
    assign(W_next)

    log(loss, alpha, rails, norms, update_cosine)
```

Later, after the quantization semantics stabilize, we can move some of this into reusable layers or model wrappers.

## Experiment Matrix

### 000C.0: Float Reproduction

Repeat Experiment 000 with quantization disabled.

Expected result placeholder:

| Metric | Expected |
|---|---|
| Loss | Matches Experiment 000. |
| `alpha_t` | Drops after drift. |
| Saturation | Exactly zero. |
| Update cosine | Near 1. |

Result figure placeholder:

```text
TODO: results/000C_0_float_reproduction.png
```

### 000C.1: Weight Quantization Only

Enable:

```text
Q_W
```

Disable:

```text
Q_X, Q_Y, Q_G, Q_Delta, Q_acc
```

Purpose: isolate whether stored weight precision alone prevents convergence.

Expected result placeholder:

| Metric | Expected |
|---|---|
| Loss | Converges to a quantization floor. |
| Weight error | Stops near the nearest representable `W`. |
| Saturation | Low unless rails are too tight. |
| `alpha_t` | Similar to float unless quantization creates sharp jumps. |

### 000C.2: Update Quantization Only

Enable:

```text
Q_{\Delta}
```

Purpose: identify update underflow and dead learning.

Expected result placeholder:

| Metric | Expected |
|---|---|
| Loss | May plateau when updates underflow. |
| Applied update norm | Can collapse to zero. |
| Useful interval | Lower bound may exceed upper bound. |
| `alpha_t` | Too-small `alpha_t` may stabilize but also kill learning. |

### 000C.3: Weights Plus Updates

Enable:

```text
Q_W,\quad Q_{\Delta}
```

Purpose: model the minimum realistic parameter/update path.

Expected result placeholder:

| Metric | Expected |
|---|---|
| Loss | Stable if precision is sufficient. |
| Saturation | Low for wide rails. |
| Update cosine | Near 1 before clipping; lower if update quantization is coarse. |

### 000C.4: Full Fake-Fixed-Point Path With Wide Rails

Enable:

```text
Q_X,\quad Q_W,\quad Q_Y,\quad Q_G,\quad Q_{\Delta},\quad Q_{\mathrm{acc}}
```

Use wide rails first.

Purpose: confirm that quantization noise alone does not break the controller.

Expected result placeholder:

| Metric | Expected |
|---|---|
| Loss | Stable with a quantization floor. |
| Saturation | Near zero. |
| `alpha_t` | Similar to float, possibly noisier. |

### 000C.5: Full Fake-Fixed-Point Path With Tight Rails

Use intentionally tight rails to create saturation.

Purpose: test whether the global throttle prevents divergence when the numerical path is near hardware limits.

Expected result placeholder:

| Metric | Expected |
|---|---|
| Loss without throttle | Divergence, plateau, or rail-locking. |
| Loss with throttle | More stable, but may not recover if information is clipped away. |
| Saturation | Nonzero and correlated with instability. |
| Useful interval | May become empty in extreme cases. |

## Plots To Produce

Each run should produce:

1. Loss and RMSE versus step.
2. Weight error versus step.
3. Weight, gradient, and update norms.
4. Curvature proxy and EMA.
5. `alpha_t` and effective learning rate.
6. Raw and throttled stability margins.
7. Saturation fraction by tensor.
8. Near-rail fraction by tensor.
9. Applied update norm and update underflow fraction.
10. Update cosine between intended and actual applied update.
11. Useful lower/upper bounds for `alpha_t`.

## Interpretation Rules

If the throttled run stays stable but reaches a nonzero error floor, that is acceptable. It means quantization is limiting accuracy but not destabilizing the loop.

If the throttled run is stable but the applied update norm becomes zero, that is not success. It means the controller avoided divergence by killing learning.

If activation or input rails clip heavily, recovery may be impossible because the target information has been destroyed before the optimizer sees it.

If update cosine drops far below 1, the quantization or clipping path is changing the descent direction. That is the same diagnostic we eventually want for legacy row/column kappa projection.

## Results

### Summary

```text
TODO: Add run summary after the first notebook is complete.
```

### Figures

```text
TODO: Add exported plots from workspace/ablations/000_b_global_throttle_sanity_quantization/results/.
```

### Conclusions

```text
TODO: State whether global throttling stabilizes the fake-fixed-point one-layer learner, and identify whether the limiting failure mode is saturation, update underflow, or both.
```
