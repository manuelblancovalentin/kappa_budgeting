# Background: Kappa Budgeting Ablations

This project studies how to make fixed-point online learning stable when a deployed model must keep training after its input distribution changes.

The ENABOL paper frames the stability problem as a bounded-gain control problem. A trainable fixed-point network should not let activations, gradients, weights, or optimizer updates exceed the rails implied by the chosen `ap_fixed` formats. The proposed mechanism is kappa budgeting: assign per-layer induced-norm budgets and enforce them during training.

For an affine layer,

```text
y_l = W_l x_l + b_l
```

the forward sensitivity is bounded by the row-sum norm:

```text
||y_l||_inf <= ||W_l||_inf ||x_l||_inf + ||b_l||_inf
```

and the backward sensitivity is bounded by the column-sum norm:

```text
||g_{x_l}||_inf <= ||W_l||_1 ||g_{y_l}||_inf
```

The intended safety condition is therefore:

```text
||W_l||_inf <= kappa_row_l
||W_l||_1   <= kappa_col_l
||b_l||_inf <= beta_l
```

with enough activation, gradient, accumulator, and update precision to keep every intermediate value away from saturation.

## Current Concern

The paper implementation used a hardware-friendly approximation: when a layer exceeded a row or column budget, it used power-of-two right shifts instead of exact division. This is cheap in HLS, but it can be too coarse for learning.

If a row only slightly exceeds its budget, a one-bit correction still divides that row by two. Applied independently across rows and layers, this does more than reduce update magnitude: it changes the relative geometry of the parameter vector. In a multilayer network, that can alter the effective loss landscape seen by the optimizer.

There are three practical concerns to isolate:

1. **Projection coarseness:** exact projection may be stable while power-of-two projection is disruptive.
2. **Layer-local direction change:** projecting one row or one layer changes parameter direction, not just global scale.
3. **Optimizer-state mismatch:** Adam state can become inconsistent if weights are projected but `m` and `v` are not transformed consistently.

## Goal

The immediate goal is to build small, controlled ablation tests where ordinary fixed-point online learning fails under input drift, then test which kappa-budgeting variants prevent saturation or divergence without destroying learning.

The first tests should be simple enough that every failure mode can be inspected directly:

- row and column norms,
- activations and gradients,
- saturation counts,
- throttle shifts,
- projection shifts,
- update direction change,
- recovery loss after drift.

The long-term goal is to turn these observations into a valid budgeting methodology for setting layer precision, `kappa_row`, `kappa_col`, and update controls before moving back to realistic hls4ml models.
