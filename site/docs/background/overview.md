# Background: Closed-Loop Stability Ablations

This project studies how to make fixed-point online learning stable when a deployed model must keep training after its input distribution changes.

The ENABOL paper frames the stability problem as a bounded-gain control problem. A trainable fixed-point network should not let activations, gradients, weights, or optimizer updates exceed the rails implied by the chosen `ap_fixed` formats. The proposed mechanism is kappa budgeting: assign per-layer induced-norm budgets and enforce them during training.

For an affine layer,

```math
y_\ell = W_\ell x_\ell + b_\ell
```

the forward sensitivity is bounded by the row-sum norm:

```math
\lVert y_\ell \rVert_{_\infty}
\le
\lVert W_\ell \rVert_{_\infty} \lVert x_\ell \rVert_{_\infty}
+ \lVert b_\ell \rVert_{_\infty}
```

and the backward sensitivity is bounded by the column-sum norm:

```math
\lVert g_{x_\ell} \rVert_{_\infty}
\le
\lVert W_\ell \rVert_{_1} \lVert g_{y_\ell} \rVert_{_\infty}
```

The intended safety condition is therefore:

```math
\lVert W_\ell \rVert_{_\infty} \le \kappa_{\mathrm{row},\ell}
```

```math
\lVert W_\ell \rVert_{_1} \le \kappa_{\mathrm{col},\ell}
```

```math
\lVert b_\ell \rVert_{_\infty} \le \beta_\ell
```

with enough activation, gradient, accumulator, and update precision to keep every intermediate value away from saturation.

## New Framing

Static kappa budgets control the representational gain of a frozen network. Online training is different because the weights themselves are the state of a dynamical system:

```math
\theta(t+1) = \theta(t) - \eta G(\theta(t))
```

where $\theta$ is the flattened vector of all trainable weights and biases. Locally, the learning dynamics are approximated by:

```math
\delta \theta(t+1) = (I - \eta H)\delta \theta(t)
```

where $H$ is the Hessian of the loss with respect to trainable parameters. A local stability condition is:

```math
\rho(I - \eta H) < 1
```

This means a network can satisfy static gain rails such as $\lVert W_\ell \rVert \le \kappa_\ell$ and still train unstably if the closed-loop update field has high curvature or if fixed-point saturation corrupts the update.

The new ablation direction is therefore:

```text
Static kappa budgeting controls representational gain.
Dynamic closed-loop throttling controls online learning stability.
```

Kappa remains useful as a loose safety rail, but the primary mechanism to test is a global scalar update throttle.

## Dynamic Global Throttle

The proposed update is:

```math
\Delta \theta_{\mathrm{raw}}(t) = -\eta G(\theta(t))
```

```math
\Delta \theta(t) = \alpha(t)\Delta \theta_{\mathrm{raw}}(t)
```

```math
\theta(t+1) = \theta(t) + \Delta \theta(t)
```

with:

```math
0 < \alpha(t) \le 1
```

The scalar $\alpha(t)$ is shared across all layers. This is important: a global scalar slows down learning time without rotating the update direction in parameter space.

The first controller should use a cheap online curvature proxy:

```math
C(t) =
\frac{
  \lVert G(t) - G(t-1) \rVert
}{
  \lVert \theta(t) - \theta(t-1) \rVert + \varepsilon
}
```

```math
S(t) = \langle C(t) \rangle_\mathrm{EMA}
```

```math
\alpha(t) =
\operatorname{clamp}
\left(
  \frac{1}{1 + \beta S(t)},
  \alpha_{\min},
  1
\right)
```

For tiny models, the experiments should also compute the true Hessian and compare:

```math
\lVert H(t) \rVert
```

```math
\lambda_{\max}(H)
```

```math
\rho(I - \eta H)
```

```math
\rho(I - \alpha(t)\eta H)
```

against the proxy $C(t)$.

## Legacy Kappa Projection Concern

The paper implementation used a hardware-friendly approximation: when a layer exceeded a row or column budget, it used power-of-two right shifts instead of exact division. This is cheap in HLS, but it can be too coarse for learning.

If a row only slightly exceeds its budget, a one-bit correction still divides that row by two. Applied independently across rows and layers, this does more than reduce update magnitude: it changes the relative geometry of the parameter vector. In a multilayer network, that can alter the effective loss landscape seen by the optimizer.

There are three practical concerns to isolate:

1. **Projection coarseness:** exact projection may be stable while power-of-two projection is disruptive.
2. **Layer-local direction change:** projecting one row or one layer changes parameter direction, not just global scale.
3. **Optimizer-state mismatch:** Adam state can become inconsistent if weights are projected but `m` and `v` are not transformed consistently.

These legacy row/column mechanisms should be treated as comparison baselines only when they are already available or cheap to stub. The immediate implementation should not spend time rebuilding full row/column kappa projection if the main experiment is dynamic global throttling.

## Goal

The immediate goal is to build small, controlled ablation tests where ordinary fixed-point online learning fails under input drift, then test whether dynamic global throttling prevents divergence or saturation while preserving the descent direction.

The first tests should be simple enough that every failure mode can be inspected directly:

- loss and output error,
- global and per-layer weight norms,
- global and per-layer gradient norms,
- update norms,
- curvature proxy $C(t)$,
- true Hessian metrics where feasible,
- global throttle $\alpha(t)$,
- activations and gradients,
- saturation counts,
- update direction change,
- recovery loss after drift.

The long-term goal is to turn these observations into a valid methodology for combining loose static kappa rails, precision selection, and dynamic closed-loop update control before moving back to realistic hls4ml models.
