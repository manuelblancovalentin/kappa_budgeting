---
id: models-index
title: "Models"
sidebar_label: "Index"
status:
  - valid
  - inprogress
tags:
  - model-registry
last_modified: 2026-05-15
---
# Models
<PageMeta />
---

This section is the registry of neural-network models used by ENABOL ablation experiments. Each model gets a stable ID so an experiment can be described as a composition of:

```text
Dataset + Model + Training method + Precision map + Drift/corruption
```

Use model IDs in experiment trace blocks:

```text
Model: MDL-DENSE1-LINEAR-NOBIAS-000
```

## Model Table

| Model ID | Name | Layers | Activation | Bias | Trainable Params | Status | Notes |
|---|---|---:|---|---|---:|---|---|
| [`MDL-DENSE1-LINEAR-NOBIAS-000`](./dense1-linear-nobias-000.md) | One Dense linear student, no bias | 1 Dense | None | No | 8 | Active | Used by EXP-000A and EXP-000B. |

## ID Convention

Use:

```text
MDL-<ARCH>-<DETAIL>-<NUMBER>
```

Examples:

```text
MDL-DENSE1-LINEAR-NOBIAS-000
MDL-DENSE2-RELU-BIAS-001
MDL-CNN-SMALL-010
```

Keep IDs stable once used in experiment documentation. If the architecture, activation, bias behavior, or layer naming changes in a way that affects results, create a new model ID.

## Model Record Checklist

Each model page should define:

- ID and short name,
- layer sequence,
- stable layer names,
- forward equations,
- trainable parameters,
- expected input/output shapes,
- whether bias, activation, or normalization is enabled,
- code path used to instantiate it,
- caveats that affect interpretation.
