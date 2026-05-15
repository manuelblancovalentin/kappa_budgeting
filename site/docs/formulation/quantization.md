---
status:
  - preliminary
  - inprogress
tags:
  - todo
  - formulation
  - quantization
last_modified: 2026-05-15
---
# 🧮 Fixed-point quantization
<PageMeta />
---

Eventually, the real hardware update must be quantized to the fixed-point format. This can be modeled as:

```math
\boxed{
\theta_{t+1}
=
Q_\Theta
\left[
\theta_t
-
Q_\Delta(\alpha_t\eta G_t)
\right].
}
```

Where:

* $Q_\Theta$ quantizes/clips weights,
* $Q_\Delta$ quantizes/clips updates,
* $\alpha_t$ is the global throttle.

This introduces quantization error:
```math
\xi_t
=
\theta_{t+1}
-
(\theta_t-\alpha_t\eta G_t).
```

## Stability analysis (Lyapunov) under quantization
The Lyapunov/descent condition becomes roughly:

```math
\mathcal{L}_{t+1}-\mathcal{L}_t
\lesssim
-\alpha_t\eta
\left(
1-\frac{\alpha_t\eta L_t}{2}
\right)
\|G_t\|^2
+
\text{quantization error terms}.
```

This tells us two things. First, stability requires an upper bound:

```math
\alpha_t
\leq
\frac{\chi}{\eta C_t^{\text{ctrl}}}.
```

Second, fixed-point usefulness requires the update not to underflow. If update quantum is q_\Delta, then approximately:

```math
\alpha_t\eta\|G_t\|
\gtrsim
q_\Delta.
```

So:
```math
\alpha_t
\gtrsim
\frac{q_\Delta}{\eta\|G_t\|}.
```

#### Therefore, useful stable fixed-point learning requires a nonempty interval:

```math
\boxed{
\frac{q_\Delta}{\eta\|G_t\|+\epsilon}
\lesssim
\alpha_t
\leq
\frac{\chi}{\eta C_t^{\text{ctrl}}+\epsilon}.
}
```

<div class="summary-box">
    <strong>Key Insight:</strong> Fixed-point precision gives a minimum useful update size. Stability gives a maximum safe update size. Online learning is possible only when these bounds overlap.
</div>