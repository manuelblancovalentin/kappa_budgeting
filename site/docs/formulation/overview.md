---
status:
  - preliminary
tags:
  - todo
  - formulation
  - overview
last_modified: 2026-05-15
---


# 🔎 Background
<PageMeta />
---

This project studies how to make fixed-point online learning stable when a deployed model must keep training after its input distribution changes.

[The <ENABOL /> paper](../../../iccad2026-paper1079.pdf) frames the stability problem as a bounded-gain control problem. A trainable fixed-point network should not let activations, gradients, weights, or optimizer updates exceed the rails implied by the chosen `ap_fixed` formats. The proposed mechanism is kappa budgeting: assign per-layer induced-norm budgets and enforce them during training.


# Loss minimization and the gradient

Let's start with the basics. 

<SeqDiagram
  maxWidth="760px"
  items={[
    {label: String.raw`x\in\mathbb{R}^{d_\mathrm{in}}`, kind: 'plain'},
    {
      label: String.raw`f_\theta`,
      sublabel: String.raw`f_\theta:\mathbb{R}^{d_\mathrm{in}}\to\mathbb{R}^{d_\mathrm{out}}`,
      kind: 'box',
    },
    {label: String.raw`\hat{y}\in\mathbb{R}^{d_\mathrm{out}}`, kind: 'plain'},
  ]}
  arrows={[{label: String.raw`x`}, {label: String.raw`\hat{y}=f_\theta(x)`}]}
/>

```math
f:\mathbb{R}^{d_\mathrm{in}}\to\mathbb{R}^{d_\mathrm{out}} (the model itself)
* x\in\mathbb{R}^{d_\mathrm{in}}
```

<Figure
  src="/img/formulation/3D.png"
  alt="3D loss landscape arbitrary"
  maxWidth="70%"
  label="Figure 1"
  caption="Illustrative loss landscape for the closed-loop training stability discussion."
/>
