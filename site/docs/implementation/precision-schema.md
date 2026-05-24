---
sidebar_label: "🧭 Precision Schema"
status:
  - valid
  - inprogress
tags:
  - implementation
  - precision
  - hls4ml
  - trainable
last_modified: 2026-05-23
author: mbvalentin
source: "enabol/precision.py"
source_url: "https://github.com/manuelblancovalentin/kappa_budgeting/blob/master/enabol/precision.py"
---
# 🧭 Trainable Precision Schema
<PageMeta />
---

<TBox type="summary" title="What this page covers">
This page records the precision relationships needed by ENABOL and hls4ml-trainable. It is intentionally incomplete. Unknown relationships are marked with `?` so we can fill them in as controller, optimizer, and layer rules become clear.
</TBox>

## Purpose

`PrecisionDict` is the user-facing description of hardware data types. It should stay compact in notebooks, but the compiler eventually needs a richer trainable schema:

```text
input / activation / weight / bias / loss
gradient / accumulator / update / optimizer state / controller metric
```

The goal is not to force users to specify every internal type forever. The goal is to identify which type can be inferred from which base type, and which type must be explicitly supplied until ENABOL has a reliable inference rule.

## Current Compile Behavior

Today, ENABOL accepts compact semantic fields such as:

```python
PrecisionDict({
    "input": {"value": wide_activation},
    "dense0": {
        "weight": wide_weight,
        "activation": wide_activation,
        "gradient": wide_gradient,
        "update": wide_update,
        "accumulator": wide_accumulator,
    },
    "loss": {"value": wide_loss},
})
```

The bridge expands aliases:

| User field | Current hls4ml-trainable expansion |
|---|---|
| `gradient` | `grad_in`, `grad_out`, `weight_grad`, `bias_grad`, `loss_grad` |
| `update` | `raw_update`, `update`, `optimizer_state` |
| `accumulator` | `gradient_accum`, `controller_metric` |
| `loss.value` | `loss` |

That last alias is useful for bring-up but is not mathematically correct as a long-term policy. `controller_metric` has different range and fractional-bit requirements from `gradient_accum`.

## Tensor Families

For a Dense layer, the trainable path currently uses these tensor families:

| Family | Symbol | Meaning | Current source | Inference status |
|---|---:|---|---|---|
| Input / activation | `x` | Layer input activation. | `input.value` or layer `activation` | ? |
| Output activation | `y` | Layer output activation / forward result. | layer `activation` / `result` | ? |
| Weight | `W` | Stored trainable weight. | layer `weight` | Base candidate |
| Bias | `b` | Stored trainable bias. | layer `bias`, or hls4ml default | ? |
| Loss value | `L` | Scalar loss. | `loss.value` | ? |
| Loss seed | `dL/dy` | Loss endpoint gradient seed. | `gradient` / `loss_grad` | ? |
| Backprop input gradient | `dL/dx` | Gradient sent to previous layer. | `gradient` / `grad_out` | ? |
| Weight gradient | `G_W` | Averaged `dL/dW` at batch end. | `gradient` / `weight_grad` | ? |
| Bias gradient | `G_b` | Averaged `dL/db` at batch end. | `gradient` / `bias_grad` | ? |
| Gradient accumulator | `sum(G)` | Batch accumulator before shift average. | `accumulator` / `gradient_accum` | ? |
| Raw optimizer update | `Delta theta_raw` | Optimizer-proposed update before alpha. | `update` / `raw_update` | ? |
| Applied update | `Delta theta` | Alpha-scaled update before cast into stored parameters. | `update` | ? |
| Optimizer state | `s_opt` | Momentum/Adam/etc. state, currently SGD placeholder. | `update` / `optimizer_state` | ? |
| Controller alpha | `alpha` | Global scalar throttle. | `alpha` or default | ? |
| Controller metric | `m_ctrl` | Squared norms and inequality terms used by controller laws. | `controller_metric`, currently aliased from `accumulator` if omitted | Known partial rule below |

## Known Rule: Parameter Precision To Actual Update Telemetry

The controller metric stores squared geometry. For GT-0 and GT-1, the controller law now uses raw optimizer update geometry:

```text
raw_update_norm_sq    = ||Delta theta_raw||^2
dgrad_norm_sq         = ||G_t - G_{t-1}||^2
stability_lhs_raw     = learning_rate^2 * dgrad_norm_sq
stability_lhs_ctrl    = alpha^2 * stability_lhs_raw
stability_rhs         = chi^2 * (raw_update_norm_sq + epsilon^2)
actual_update_norm_sq = ||theta_after - theta_before||^2
```

For SGD, `raw_update_norm_sq` already contains the learning rate because the raw update is `-learning_rate * gradient`. The separate `actual_update_norm_sq` telemetry measures the movement after alpha and fixed-point assignment. These squared quantities mean `controller_metric` cannot be copied blindly from `gradient_accum`, and it usually needs more fractional bits than the stored weight.

Let a stored parameter type have:

```text
WL_theta = total word length
IWL_theta = integer word length
F_theta = WL_theta - IWL_theta
q_theta = 2^-F_theta
```

The smallest nonzero one-parameter movement is approximately `q_theta`, so the smallest one-term squared movement is:

```text
q_theta^2 = 2^(-2 * F_theta)
```

Therefore, if we want actual movement telemetry to resolve one stored-parameter LSB, `controller_metric` must have enough fractional bits to represent at least that squared movement:

```text
F_controller_metric >= 2 * F_theta + guard_bits
```

For bring-up, use a few guard bits because:

- several products are evaluated in the controller law,
- `learning_rate^2 * dgrad_sq` can be smaller than `dgrad_sq`,
- `epsilon^2` may be tiny,
- small models have few terms in the norm sum, so summation may not rescue underflow.

Example:

```text
weight_t = ap_fixed<16,6>
F_theta = 10
q_theta^2 = 2^-20
```

A controller metric with only 14 fractional bits cannot represent one squared weight LSB:

```text
ap_fixed<28,14>  -> F = 14 -> LSB = 2^-14
2^-20 < 2^-14   -> dtheta_sq can quantize to zero
```

For this case, a bring-up type like the following is more appropriate:

```text
controller_metric: ap_fixed<48,16,AP_RND,AP_SAT>
```

This gives 32 fractional bits and enough integer range for the current small Dense tests.

## Relationship Table

This table is the starting point for the final precision inference pass.

| Target type | Depends on | Rule |
|---|---|---|
| `activation` / `result` | input range, weight range, fan-in, activation function | ? |
| `accum` | input precision, weight precision, fan-in | ? |
| `loss` | prediction precision, target precision, loss kind, output dimension | ? |
| `loss_grad` | prediction precision, target precision, loss kind | ? |
| `weight_grad` | input precision, loss/next gradient precision, batch size | ? |
| `bias_grad` | loss/next gradient precision, batch size | ? |
| `gradient_accum` | unaveraged gradient precision, batch size | ? |
| `raw_update` | optimizer kind, learning rate, gradient precision, optimizer state | ? |
| `update` | raw update precision, alpha precision, stored parameter precision | ? |
| `optimizer_state` | optimizer kind, raw update precision, gradient precision | ? |
| `alpha` | controller kind, controller rails, smoothing rule | ? |
| `controller_metric` | stored parameter precision, gradient precision, learning rate, controller law | `F_controller_metric >= 2 * F_theta + guard_bits` for actual update telemetry; raw-update controller law also depends on update/gradient precision |

## Notebook Guidance For Now

Until ENABOL implements automatic trainable precision inference, specify `controller_metric` explicitly when using global throttling:

```python
wide_controller_metric = dtypes.ap_fixed(WL=48, IWL=16, QMODE="AP_RND", OMODE="AP_SAT")

precision_dict = PrecisionDict({
    "input": {"value": wide_activation},
    "dense0": {
        "weight": wide_weight,
        "activation": wide_activation,
        "gradient": wide_gradient,
        "update": wide_update,
        "accumulator": wide_accumulator,
        "controller_metric": wide_controller_metric,
    },
    "loss": {"value": wide_loss},
})
```

If `controller_metric` is omitted today, ENABOL may still alias it from `accumulator`. That is acceptable only for smoke tests where the controller trace is not being interpreted numerically.

## Implementation Direction

The target behavior is:

1. User provides a compact `PrecisionDict`.
2. ENABOL expands semantic fields into hls4ml-trainable fields.
3. ENABOL applies explicit user overrides first.
4. ENABOL infers missing internal fields from known rules.
5. ENABOL warns when a fallback alias is used for a field with known non-equivalent requirements.

This belongs to [ENB-023](/docs/status/tasks?query=ENB-023): infer hls4ml trainable precisions from ENABOL model, dataset, optimizer, and controller.

## Change Log

| Task | Date | Files | Summary |
|---|---|---|---|
| [ENB-028](/docs/status/tasks?query=ENB-028) | 2026-05-23 | `site/docs/implementation/precision-schema.md` | Added the first trainable precision relationship document and recorded the stored-parameter movement telemetry to `controller_metric` fractional-bit rule. |
