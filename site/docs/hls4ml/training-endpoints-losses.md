---
id: training-endpoints-losses
title: "Training Endpoints and Losses"
sidebar_label: "Loss Endpoints"
status:
  - preliminary
  - inprogress
tags:
  - hls4ml
  - implementation
  - training
last_modified: 2026-05-21
author: mbvalentin
---
# Training Endpoints and Losses
<PageMeta />
---

<TBox type="summary" title="Role">

Loss computation is the bridge between the forward graph and the backward chain. It should be represented as explicit endpoint metadata, not hand-written inside the writer.

</TBox>

## What The Old Fork Did

The deprecated writer added, for each model output:

1. a ground-truth top-level input;
2. a scalar loss output;
3. a gradient buffer seeded by the loss;
4. a loss config struct;
5. a loss function call;
6. optional rewiring to logits for cross-entropy losses;
7. a skip marker for the final activation backpass when loss consumed logits.

This is exactly the behavior we need, but it should be produced by a trainable pass.

## Endpoint Metadata

Create a normalized endpoint record for each output:

| Field | Meaning |
|---|---|
| `output_name` | Nominal model output variable. |
| `ground_truth_name` | Generated top-level ground truth variable. |
| `loss_name` | Requested loss, normalized to lowercase canonical name. |
| `effective_loss_name` | Loss implementation actually called, such as `binary_crossentropy_from_logits`. |
| `loss_input_name` | Tensor fed into the loss. Usually the output, sometimes logits. |
| `loss_scalar_name` | Generated scalar loss variable. |
| `loss_scalar_type` | Generated loss scalar typedef. |
| `loss_gradient_name` | Generated output gradient buffer. |
| `loss_gradient_type` | Generated gradient typedef. |
| `skip_backward_layer` | Optional layer name to skip during backpass, such as final sigmoid/softmax. |

This endpoint record becomes the source for top-level IO, loss config structs, and the first `grads_in` value for reverse traversal.

## Current Implementation

`HLS4ML-006` has started with metadata only. Commit `56f9ff00` added `vivado:resolve_trainable_loss_endpoints` to the `vivado:trainable` flow after backward-order resolution.

For the first supported graph shape, the pass attaches:

| Field | Current behavior |
|---|---|
| `output_name` | The single model output, or `Loss.Output` if explicitly provided and valid. |
| `output_layer` | The graph layer resolved by `vivado:resolve_trainable_backward_order`. |
| `ground_truth_name` | `Loss.GroundTruthName`, otherwise `{output_name}_truth`. |
| `loss_name` | Canonical normalized loss kind. Currently only `half_mse`. |
| `effective_loss_name` | Same as `loss_name` for `half_mse`; from-logits rewiring is future work. |
| `loss_input_name` | The tensor consumed by the loss. Currently the nominal output. |
| `loss_input_type` | The output tensor type name already known to the graph. |
| `loss_input_shape` / `loss_input_size` | Shape and flattened size of the loss input tensor. |
| `loss_scalar_name` / `loss_scalar_type` | `Loss.LossScalarName` or `loss0`, with `loss0_t`. |
| `loss_gradient_name` / `loss_gradient_type` | `Loss.LossGradientName` or `{output_name}_loss_grad`, with `{output_name}_loss_grad_t`. |
| `loss_gradient_scale` | `1.0` for `half_mse`, meaning `dL/dy = y_hat - y`. |
| `skip_backward_layer` | `None` for now. |

This is deliberately not template generation yet. The next step crosses into codegen: loss config structs, loss helper kernels, ground-truth top-level IO, scalar loss outputs, and actual `dL/dy` buffer emission.

## From-Logits Handling

For cross-entropy losses, the numerically stable path is often:

```text
logits -> cross_entropy_from_logits
```

instead of:

```text
logits -> sigmoid/softmax -> cross_entropy
```

The trainable endpoint pass should inspect the final output-producing layer:

| Requested loss | Final activation | Effective behavior |
|---|---|---|
| `binary_crossentropy` | `sigmoid` | Feed previous tensor to `binary_crossentropy_from_logits`; skip sigmoid backpass. |
| `categorical_crossentropy` | `softmax` | Feed previous tensor to `categorical_crossentropy_from_logits`; skip softmax backpass. |
| `mse` | any | Feed nominal output. |

This should be represented in metadata before C++ generation.

## Loss Config Generation

Each endpoint should generate a config struct:

```cpp
struct mse_config0 {
    static const unsigned n_out = N;
    static const unsigned loss_index = 0;
    using data_in_t = ...;
    using loss_t = ...;
    using grad_out_t = ...;
};
```

Loss precision should come from:

1. explicit `Model.Training.LossPrecision`;
2. output/effective loss input type plus sign/widening rules;
3. model default precision.

Do not duplicate type aliases for nominal and effective loss inputs manually in the writer. The endpoint pass should generate compatibility aliases when needed.

## Loss Kernels

Start with the small set needed by ENABOL:

| Loss | First support |
|---|---|
| `mse` or `half_mse` | required |
| `mae` | optional |
| `binary_crossentropy` | useful |
| `categorical_crossentropy` | useful |

The current ENABOL software loop uses `half_mse` as a common training loss. Hardware support should be explicit about scaling. If C++ implements full MSE but Python compares half MSE, loss traces and gradients will disagree by a constant factor.

## Multi-Output Policy

Normalize loss lists early:

- one loss and one output: use it directly;
- one loss and many outputs: broadcast if explicitly allowed;
- many losses: require length equal to number of outputs.

Then freeze the endpoint list. Later passes should not re-broadcast or infer output counts.
