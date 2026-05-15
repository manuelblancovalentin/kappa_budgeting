# 🗂️ How this fits into κ-budgeting
<StatusBadges statuses="todo;revise" />
---

$\kappa$-budgeting is still useful as a static safety rail $\|W_\ell\|\leq \kappa_\ell$, since it bounds:

* fixed-point dynamic range,
* layer gain,
* representational Lipschitzness,
* overflow risk,
* static hardware safety.

On the other hand, the throttle controls:
```math
\theta_{t+1}
=
\theta_t-\alpha_t\eta G_t.
```

It bounds:

* learning speed,
* closed-loop adaptation stability,
* overshoot under drift,
* optimizer stiffness.

So:

```math
\boxed{
\text{$\kappa$-budgeting controls the admissible region.}
}
```
```math
\boxed{
\text{global throttling controls the trajectory inside that region.}
}
```

A very clean combined formulation is:
```math
\theta_{t+1}^{\text{ctrl}}
=
\theta_t-\alpha_t\eta G_t.
```
Then:

```math
\theta_{t+1}
=
\Pi_{\kappa}
\left(
\theta_{t+1}^{\text{ctrl}}
\right).
```

But the key is that $\Pi_\kappa$ should be a loose guard rail, not the primary stabilizer applied aggressively every step.

<div class="summary-box">
    <strong>Key Insight:</strong> The throttle should prevent the system from hitting the $\kappa$ rails too violently. The $\kappa$ rails are still there as final protection. <b>So if we want, we can actually disable the $\kappa$ projection to spare hardware and computation and keep the global throttle as the main stabilizer.</b>
</div>