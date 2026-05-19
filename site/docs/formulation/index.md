---
id: index
title: "Docs"
sidebar_label: "🏷️ Docs"
status:
  - valid
  - inprogress
tags:
  - project
last_modified: 2026-05-18
author: mbvalentin
---
# 🏷️ Docs
<PageMeta />
---

<TBox type="summary" title="How to use this section">

Use this page as the technical documentation entry point. The sidebar is organized around the artifacts that define an experiment: formulation, implementation, datasets, models, ablation tests, controllers, and experiment records.

</TBox>

## Main Sections

| Section | Purpose |
|---|---|
| [Formulation](overview.md) | Mathematical framing for fixed-point online learning and global throttling. |
| [Implementation](../implementation/code-map.md) | Code-facing reference for the Python modules and training loop. |
| [Datasets](../datasets/) | Stable dataset IDs used in experiment traces. |
| [Models](../models/) | Stable model IDs used in experiment traces. |
| [Ablation Tests](../ablations/) | Stress-test families used to make fixed-point training fail in controlled ways. |
| [Controllers](../controllers/) | Stable controller IDs used to modify or throttle optimizer updates. |
| [Experiments](../experiments/) | Actual experiment reports, plots, notebooks, and conclusions. |

## Experiment Trace Pattern

Most technical pages should eventually support experiment traces of the form:

```yaml
dataset: DS-...
model: MDL-...
precision: PREC-...
stress_test: ST-...
optimizer: ...
controller: CTRL-...
metrics: [...]
```
