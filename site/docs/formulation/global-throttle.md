---
status:
  - preliminary
  - inprogress
tags:
  - todo
  - formulation
  - global-throttle
last_modified: 2026-05-15
author: mbvalentin
---
# 🎚️ The global throttle mechanism
<PageMeta />
---

## Motivation
Instead of projecting each row/column independently (which as we saw in the [kappa-budgeting section](./kappa-budgeting.md) introduces rotation in the $\theta$ plane), we introduce a single global scalar:

```math
0<\alpha_t\leq 1.
```

Then the SGD update becomes:

```math
\boxed{
\theta_{t+1}
=
\theta_t-\alpha_t\eta G_t
} \qquad \text{with} \quad 
G_t=\nabla_\theta\mathcal{L}(\theta_t).
```

This is important because the update remains parallel to the original gradient update.

Raw update:

```math
\Delta\theta_t^{\text{raw}}
=
-\eta G_t.
```

Controlled update:

```math
\Delta\theta_t^{\text{ctrl}}
=
-\alpha_t\eta G_t.
```

Therefore:

```math
\Delta\theta_t^{\text{ctrl}}
=
\alpha_t\Delta\theta_t^{\text{raw}}.
```

Now, to verify whether direction is preserved or not, we can use the cosine similarity:

```math
\cos
\left(\Delta\theta_t^{\text{ctrl}},
\Delta\theta_t^{\text{raw}}
\right)
=
\frac{\Delta\theta_t^{\text{ctrl}}\,\Delta\theta_t^{\text{raw}}}{
\|\Delta\theta_t^{\text{ctrl}}\|
\|\Delta\theta_t^{\text{raw}}\|
}.
```


As mentioned before, we want this to be $1$, which means the scaling used to ensure stability (on every layer) do not change the direction of the update. Now, if we replace the definition for our global throttle, we get:

```math
\cos
\left(\Delta\theta_t^{\text{ctrl}},
\Delta\theta_t^{\text{raw}}
\right)
=
\frac{\alpha_t\Delta\theta_t^{\text{raw}}\,\Delta\theta_t^{\text{raw}}}{
\|\alpha_t\Delta\theta_t^{\text{raw}}\|
\|\Delta\theta_t^{\text{raw}}\|
}
=
\frac{\alpha_t\|\Delta\theta_t^{\text{raw}}\|^2}{
\alpha_t\|\Delta\theta_t^{\text{raw}}\|
\|\Delta\theta_t^{\text{raw}}\|
}
=
1.
```

<TBox type="summary" title="Key insight">

That means global throttling preserves the direction of learning. It only changes the speed.

```math
\text{row/column $\kappa$ projection}
\Rightarrow
\text{may rotate the update}
```
```math
\text{global throttle}
\Rightarrow
\text{preserves the update direction}
```
</TBox>

## Estimation of local curvature/sensitivity

In [section overview](./overview.md#lyapunov-stability-analysis-on-sgd) we showed that convergence of SGD is guaranteed if the learning rate $\eta$ satisfies:

```math
\eta<\frac{2}{\lambda_{\max}(H)},
```

where $H$ is the Hessian of the loss. Therefore, to adapt $\alpha_t$ to ensure stability, we need an estimate of $\lambda_{\max}(H)$. However, as also introduced in [the overview section](./overview.md#why-stop-at-the-hessian), finding the actual hessian is computationally expensive and sometimes even not feasible. Therefore, we need a cheap online estimate of the local curvature/sensitivity of the loss landscape.

Let's recall the formulas for the gradient and the Hessian. 

```math
\begin{aligned}
G(\theta) &= \nabla_\theta\mathcal{L}(\theta) \\ 
H(\theta) &= \nabla_\theta G(\theta) &= \nabla_\theta^2\mathcal{L}(\theta)
\end{aligned}
```

Now, we can approximate the hessian using Taylor expansion around some operating point. For a small change in $\theta$, we have:

```math
\Delta\theta_t
=
\theta_t-\theta_{t-1},
```

Taylor expansion gives:

```math
G_t-G_{t-1}
\approx
H_t(\theta_t-\theta_{t-1}).
```

So:
```math
\Delta G_t
\approx
H_t\Delta\theta_t.
```

Taking norms:

```math
\|\Delta G_t\|
\approx
\|H_t\Delta\theta_t\|.
```

Divide by:
```math
\|\Delta\theta_t\|.
```

Then:
```math
\frac{
\|\Delta G_t\|
}{
\|\Delta\theta_t\|
}
\approx
\frac{
\|H_t\Delta\theta_t\|
}{
\|\Delta\theta_t\|
}.
```

This is an observed directional curvature estimate. In other words, this is a first-order approximation of the curvature along the direction the optimizer just moved. 

So we define:

```math
\boxed{
\widehat{C}_t
=
\frac{
\|G_t-G_{t-1}\|_2
}{
\|\theta_t-\theta_{t-1}\|_2+\epsilon
}.
}
```

### How does this relate to $\lambda_{\max}(H_t)$?

For a quadratic loss, we have:

```math
\mathcal{L}(\theta)
=
\frac{1}{2}\theta^\top H \theta,
```
then:
```math
G(\theta)
=
H\theta,
```
and:
```math
\Delta G_t
=
H\Delta\theta_t.
```


Then:
```math
\frac{\|\Delta G_t\|
}{\|\Delta\theta_t\|
}
=
\frac{\|H\Delta\theta_t\|
}{\|\Delta\theta_t\|
}
\leq
\|H\|_2
=
\lambda_{\max}(H).
```

<TBox type="summary" title="Key insight">
  So as we can see, for a quadratic loss, this estimator is an upper bound on the true Hessian spectral norm. For general losses, it is a local directional curvature estimate that can be used as a proxy for the local Lipschitzness of the gradient field.

  > **If we ensure that our estimated curvature $\widehat{C}_t$ is below some threshold, we can be SURE that the $\lambda_{\max}(H_t)$ is also below that threshold, which in turn ensures stability of the update.**
</TBox>


<TBox type="warning" title="A caveat">
  The previous estimate is not perfect. It is very good for convex problems and things like a single-dense layer architecture, but it can (**and will**) be inaccurate for non-convex problems and more complex architectures. However, it is a simple and computationally cheap way to get a sense of the local curvature/sensitivity, which is what we need to adapt the learning rate and prevent divergence.

  The global throttle mechanism preserves the geometry of the original gradient update, but it adaptively reduces the learning rate when the gradient field becomes stiff/sharp. This allows the model to keep learning without diverging, even under distribution shift.
</TBox>


## What happens for nonlinear / nonconvex networks?

For deeper networks, ReLU networks, CNNs, cross-entropy, etc., the loss is not globally quadratic.

For a general network:

```math
\hat{y}=f(x;\theta),
```

with loss:
```math
\mathcal{L}(\theta),
```
we use a local Taylor expansion:
```math
\mathcal{L}(\theta+\delta)
\approx
\mathcal{L}(\theta)
+
G(\theta)^\top\delta
+
\frac{1}{2}\delta^\top H(\theta)\delta.
```

The gradient descent map is:
```math
\theta_{t+1}
=
\theta_t-\eta G(\theta_t).
```
Linearizing perturbations:

```math
\delta_{t+1}
\approx
(I-\eta H_t)\delta_t.
```

So the same stability idea applies locally.

However:

* $H_t$ changes over time,
* $H_t$ may be indefinite,
* ReLU networks are piecewise smooth,
* mini-batch gradients include sampling noise,
* quantization introduces non-smooth perturbations.

Therefore, for general networks we should not claim $\widehat{C}_t=\lambda_{\max}(H_t)$ exactly. Instead, we claim: $\widehat{C}_t$ is an online estimate of local update-field sensitivity:
```math
\widehat{C}_t
\approx
\frac{
\|H_t\Delta\theta_t\|
}{
\|\Delta\theta_t\|
}.
```
Then, for a general loss, we want to keep:
```math
\alpha_t\eta L_t<\chi,
```
where $L_t$ is a local estimate of gradient-field Lipschitzness and $chi$ is some stability margin (normally $\chi \leq 2$). Then, for instance, if the controller uses $L_t\approx C_t^{\text{ctrl}}$:

```math
\boxed{
\alpha_t\eta C_t^{\text{ctrl}}\leq\chi.
}
```

<TBox type="summary" title="Key insight">
Even for nonlinear, non-convex networks, this global throttle mechanism can still be used as a stabilizer by using the local curvature/sensitivity estimate to adapt the learning rate and prevent divergence.
</TBox>


## Dynamical system controller design
As we just saw, the task of our controller is to guarantee, at any time, that the effective learning rate $\alpha_t\eta$ is below the stability threshold $\chi/C_t^{\text{ctrl}}$. Because training is a stochastic noisy process, we want to ensure that the system can react quickly to sudden spikes in curvature/sensitivity. In other words: we want an adaptive controller. Thus, we have different options for this (depending on how fancy we want to get). In order of complexity:

| Complexity Order | Controller type | Description | Equation |
|---|---|---|---|
| 0 | Proportional (P) | $\alpha$ response is proportional to the error | $\alpha_t = k_p e_t$ |
| 1 | Proportional-Integral (PI) | $\alpha$ response is proportional to the error and its integral | $\alpha_t = k_p e_t + k_i \sum_{i=0}^t e_i$ |
| 2 | Proportional-Integral-Derivative (PID) | $\alpha$ response is proportional to the error, its integral, and its derivative | $\alpha_t = k_p e_t + k_i \sum_{i=0}^t e_i + k_d (e_t - e_{t-1})$ |

Where $e_t$ is the error signal at time $t$, defined as:
```math
e_t
=
\alpha_t\eta C_t^{\text{ctrl}}-\chi
``` 

<TBox type="summary" title="Key insight">
  The global throttle can be imagined as a dynamical controller that tries to keep the system effective learning rate at the stability threshold $\chi$. The actual dynamical response of the controller can be simple but fast (P), slower and more stable (PI), or even predictive (PID).
</TBox>

### Zero-th order Proportional controller 
The simplest controller is a proportional controller, which sets $\alpha_t$ proportional to the error signal $e_t$. For instance:

```math
\alpha_t
=
\frac{\chi}{\eta C_t^{\text{ctrl}}}
```

The issue with this controller is that it can react too aggressively to spikes in curvature/sensitivity, which can lead to oscillations and instability. However, it is very simple and computationally cheap.

### First-order Proportional-Integral controller
A more stable controller is a proportional-integral controller, which sets $\alpha_t$ proportional to the error signal $e_t$ and its integral. For instance:
```math
\tau_\alpha\dot{\alpha}_t = \chi - \alpha_t\eta C_t^{\text{ctrl}}.
```

**But how to choose $\tau_\alpha$?**

### Second-order PID controller
An even more sophisticated controller is a proportional-integral-derivative controller, which sets $\alpha_t$ proportional to the error signal $e_t$, its integral, and its derivative. For instance:
```math
\ddot{\alpha}_t + k_d \dot{\alpha}_t + k_p (\alpha_t\eta C_t^{\text{ctrl}} - \chi) = 0.
```

**But how to choose $k_d$, and $k_p$?**