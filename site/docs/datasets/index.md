---
id: datasets-index
title: "Datasets"
sidebar_label: "Index"
status:
  - valid
  - inprogress
tags:
  - dataset-registry
last_modified: 2026-05-15
---
# Datasets
<PageMeta />
---

This section is the registry of datasets used by ENABOL ablation experiments. Each dataset gets a stable ID so experiment pages can reference the input distribution by name instead of re-explaining the full construction.

Use dataset IDs in experiment trace blocks:

```text
Dataset: DS-AFFINE-LINEAR-000
```

## Dataset Table

| Dataset ID | Name | Type | Input |  Status | Used by |
|---|---|---|---|---|---|
| [`DS-AFFINE-LINEAR-000`](./affine-linear-000.md) | Affine linear, no bias | Synthetic regression | $x \sim \mathcal{U}([0,1]^4)$ | <Badge status="valid" /> | EXP-000A and EXP-000B. |

## ID Convention

Use:

```text
DS-<FAMILY>-<TASK>-<NUMBER>
```

Examples:

```text
DS-AFFINE-LINEAR-000
DS-AFFINE-BIAS-001
DS-SENSOR-DRIFT-010
```

Keep IDs stable once they are used in an experiment page. If the data generation changes in a way that affects interpretation, create a new dataset ID.

## Dataset Record Checklist

Each dataset page should define:

- ID and short name,
- generation equations,
- dimensions,
- train/test or online sampling procedure,
- drift/corruption model,
- default random seed if relevant,
- links to plots or notebooks,
- caveats that affect interpretation.
