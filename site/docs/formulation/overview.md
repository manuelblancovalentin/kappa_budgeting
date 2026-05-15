---
status:
  - preliminary
tags:
  - todo
  - formulation
  - overview
last_modified: 2026-05-15
---
# 🔎 Overview
<PageMeta />
---

This project studies how to make fixed-point online learning stable when a deployed model must keep training after its input distribution changes.

[The <ENABOL /> paper](../../../iccad2026-paper1079.pdf) frames the stability problem as a bounded-gain control problem. A trainable fixed-point network should not let activations, gradients, weights, or optimizer updates exceed the rails implied by the chosen `ap_fixed` formats. The proposed mechanism is kappa budgeting: assign per-layer induced-norm budgets and enforce them during training.
