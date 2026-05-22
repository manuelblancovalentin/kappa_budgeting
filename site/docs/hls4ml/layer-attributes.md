---
id: layer-attributes
title: "Layer Attributes"
sidebar_label: "Layer Attributes"
status:
  - preliminary
  - inprogress
tags:
  - hls4ml
  - implementation
  - precision
  - training
last_modified: 2026-05-21
author: mbvalentin
---
# Layer Attributes
<PageMeta />
---

<TBox type="summary" title="Role">

hls4ml layers are attribute containers. This is the cleanest place to store resolved trainable facts: gradient types, update types, trainability, backward calls, and required headers.

</TBox>

## What hls4ml Does Today

Every hls4ml layer derives from `Layer` in `hls4ml/model/layers.py`. During initialization:

1. The layer stores raw converter attributes.
2. It creates mappings for weights, variables, types, and code.
3. It gets a unique `index`.
4. It resolves layer config from `HLSConfig`.
5. It copies matching config keys into the layer attribute dictionary.
6. It calls `initialize()`.
7. It validates required attributes.

The attribute system is defined in `hls4ml/model/attributes.py`.

| Attribute class | Meaning |
|---|---|
| `Attribute` | Basic expected attribute. |
| `ConfigurableAttribute` | User-overridable field. |
| `TypeAttribute` | Stores a named precision type. |
| `WeightAttribute` | Stores a weight variable. |
| `CodeAttribute` | Stores generated source code. |
| `WeightMapping` | View of all weight variables. |
| `VariableMapping` | View of all tensor variables. |
| `TypeMapping` | View of all named types. |
| `CodeMapping` | View of generated source blocks. |

Forward code generation already uses this model. A template pass stores:

```text
layer["config_cpp"]
layer["function_cpp"]
layer["include_header"]
```

The writer later emits these strings.

## What The Old Trainable Writer Had To Infer

The deprecated writer inferred or patched several things that should be attributes:

| Old behavior | Better owner |
|---|---|
| Parse `config_cpp` to find the struct name. | `layer.backward_config_name` or `layer.config_name`. |
| Look for `weight_t` and `bias_t` in generated C++. | Layer `weight_t` and `bias_t` attributes. |
| Add `grad_in_t` and `grad_out_t` by string injection. | `TypeAttribute` or generated trainable type attributes. |
| Synthesize reshape/flatten configs if missing. | Reshape/flatten trainable template pass. |
| Add `layer_name` inside C++ config structs. | Template pass field. |
| Add kappa/log2 enums. | Trainable/controller config attributes. |
| Decide pass-through behavior for unsupported layers. | Backward template pass. |

## Proposed Trainable Attributes

At minimum, trainable layers should receive:

| Attribute | Meaning |
|---|---|
| `trainable` | Whether this layer updates weights and allows gradient flow through it. |
| `backward_supported` | Whether a true backpass implementation exists. |
| `backward_passthrough` | Whether unsupported behavior is an intentional pass-through. |
| `backward_config_cpp` | C++ config additions or complete backward config struct. |
| `backward_function_cpp` | C++ call for the backpass. |
| `backward_include_header` | Headers needed by the backpass. |
| `grad_input_name` | Name of incoming gradient tensor. |
| `grad_output_name` | Name of outgoing gradient tensor. |
| `forward_input_name` | Forward input tensor used by the backpass. |
| `forward_output_name` | Forward output tensor used by the backpass. |
| `requires_gradient_cast` | Whether incoming gradient needs a cast buffer. |

Trainable type fields should include:

| Type field | Meaning |
|---|---|
| `data_in_t` | Type of forward input seen by the backpass. |
| `grad_in_t` | Type of `dL/dy`. |
| `grad_out_t` | Type of `dL/dx`. |
| `gW_t` | Per-sample weight-gradient product. |
| `gb_t` | Per-sample bias-gradient product. |
| `dW_accum_t` | Batch accumulation type for weight gradients. |
| `db_accum_t` | Batch accumulation type for bias gradients. |
| `update_W_t` | Type used for weight update deltas. |
| `update_b_t` | Type used for bias update deltas. |
| `optimizer_state_t` | Default optimizer state type when layer-specific names are not needed. |

## Precision Naming Policy

Use semantic fields in ENABOL and normalize to hls4ml names during config conversion:

| ENABOL field | hls4ml trainable field |
|---|---|
| `weight` | `weight_t` |
| `bias` | `bias_t` |
| `activation` | `result_t` |
| `accumulator` | `accum_t` |
| `gradient` | `grad_in_t` and/or `grad_out_t` |
| `update` | `update_W_t`, `update_b_t` |
| `gradient_accum` | `dW_accum_t`, `db_accum_t` |

Avoid forcing ENABOL users to know every C++ typedef name. ENABOL should expose semantic fields; hls4ml should resolve the C++ names.

## Layer-Specific Notes

Dense and convolution layers need real backpass templates and update handling. Activation, reshape, flatten, pooling, upsampling, and padding layers can start with backpass templates that compute gradient transport. Unsupported layers should fail during validation unless explicitly marked as pass-through.

Batch normalization needs special handling because hls4ml forward config uses scale/bias naming rather than weight/bias naming. That should be handled in a batchnorm trainable template, not by regex editing generated C++.

