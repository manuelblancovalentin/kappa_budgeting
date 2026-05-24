---
title: "Controllers"
sidebar_label: "Index"
status:
  - preliminary
  - inprogress
tags:
  - controller-registry
  - global-throttle
last_modified: 2026-05-18
author: mbvalentin
---
# 🕹️ Controllers
<PageMeta />
---

<TBox type="summary" title="Purpose">

This section is the registry of controller policies used by <ENABOL /> experiments. Controllers modify the update magnitude, feasibility, or safety behavior after the optimizer proposes a raw update.

</TBox>


## Optimizer vs Controller

The optimizer computes a raw update direction:

```math
\Delta\theta_t^{\mathrm{raw}}
=
\mathrm{Optimizer}(G_t, \mathrm{state}_t).
```

The controller modifies that update before it is applied:

```math
\Delta\theta_t^{\mathrm{ctrl}}
=
\mathrm{Controller}
\left(
\Delta\theta_t^{\mathrm{raw}},
\mathrm{diagnostics}_t
\right).
```

For global-throttle controllers:

```math
\Delta\theta_t^{\mathrm{ctrl}}
=
\alpha_t
\Delta\theta_t^{\mathrm{raw}},
\qquad
0<\alpha_t\leq 1.
```

This preserves the raw optimizer direction before fixed-point quantization effects.

## Controller Table

| Controller ID | Name | Dynamics | Quantization-aware | Direction-preserving | Status |
|---|---|---|---|---|---|---|
| [`CTRL-NONE`](./ctrl-none.md) | No controller | none | no | yes | <Badge status="valid" /> |
| [`CTRL-GT-ORDER-0`](./ctrl-gt-order-0.md) | Algebraic global throttle | order 0 | no | yes | <Badge status="valid" /> |
| [`CTRL-GT-ORDER-1`](./ctrl-gt-order-1.md) | First-order global throttle | order 1 | no | yes | <Badge status="planned" /> |
| [`CTRL-GT-ORDER-2`](./ctrl-gt-order-2.md) | Second-order damped global throttle | order 2 | no | yes | <Badge status="planned" /> |
| [`CTRL-GT-ORDER-2-QA`](./ctrl-gt-order-2-qa.md) | Quantization-aware second-order global throttle | order 2 | yes | yes | <Badge status="planned" /> |

## ID Convention

Use:

```text
CTRL-<FAMILY>-<DYNAMICS>[-<FEATURE>]
```

Examples:

```text
CTRL-NONE
CTRL-GT-ORDER-0
CTRL-BINARY-THROTTLE
CTRL-GT-ORDER-1
CTRL-GT-ORDER-2
CTRL-GT-ORDER-2-QA
```

`ORDER-0` means **zero controller state**, not zeroth-order gradient-free optimization.

## Controller Record Checklist

Each controller page should define:

- ID and short name,
- inputs,
- outputs,
- state variables,
- update equations,
- required diagnostics,
- expected behavior,
- known failure modes,
- implementation status.
