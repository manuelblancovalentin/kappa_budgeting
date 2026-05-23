---
title: "Trainable Trace Helper"
sidebar_label: "common/trainable_trace.h"
status:
  - inprogress
tags:
  - hls4ml
  - trainable
  - trace
last_modified: 2026-05-22
author: mbvalentin
source: "hls4ml/templates/vivado/trainable/common/trainable_trace.h"
---
# Trainable Trace Helper
<PageMeta />
---

`trainable_trace.h` defines one macro:

```cpp
HLS4ML_TRAINABLE_TRACE_ARRAY(name, data, size)
```

The macro is a no-op unless both conditions are true:

- the code is not being synthesized, and
- generated firmware defines `HLS4ML_TRAINABLE_TRACE`.

When active, the macro calls hls4ml's existing `nnet::save_layer_output(...)`. This means trainable logs should use the same bridge/testbench trace machinery as normal forward-layer traces.

## Why A Macro

The trainable headers are static assets, but trace names are model-specific. The macro lets each kernel write trace hooks such as:

```cpp
HLS4ML_TRAINABLE_TRACE_ARRAY(CONFIG_T::trace_weight_grad_name, weight_grad, n_weights);
```

If trainable tracing is disabled, `CONFIG_T::trace_weight_grad_name` is discarded by the preprocessor and the config struct does not need to define it. If trainable tracing is enabled, the writer must emit those names and allocate matching trace buffers.

## Writer Implication

The writer should define `HLS4ML_TRAINABLE_TRACE` only when the model is trainable and trace collection is requested. Otherwise these calls should compile away.

