---
status:
  - preliminary
  - inprogress
tags:
  - todo
  - formulation
  - global-throttle
last_modified: 2026-05-18
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

The global throttle can be interpreted as a closed-loop controller on the effective learning rate. The controlled SGD update is

```math
\theta_{t+1}
=
\theta_t-\alpha_t\eta G_t,
```

where $G_t=\nabla_\theta \mathcal{L}(\theta_t)$. The role of the controller is to keep the effective stability margin

```math
m_t
=
\eta\alpha_t C_t^{\mathrm{ctrl}}
```

near a chosen target $\chi<2$. We define the margin error

```math
e_t
=
m_t-\chi
=
\eta\alpha_t C_t^{\mathrm{ctrl}}-\chi.
```

If $e_t>0$, the update is too aggressive and the controller should reduce $\alpha_t$. If $\alpha_t<0$, the update is conservative and the controller may increase $\alpha_t$.

### Case 0: Algebraic safe-gain throttle

The simplest controller directly sets $\alpha_t$ to the estimated safe value:

```math
\alpha_t
=
\min\left(
1,
\frac{\chi}
{\eta(C_t^{\mathrm{ctrl}}+\epsilon)}
\right).
```

This is the controller used in the first global-throttle sanity experiment. It is the instantaneous solution of

```math
\eta\alpha_t C_t^{\mathrm{ctrl}}=\chi.
```

Therefore, it can be interpreted as the zero-time-constant limit of a dynamical controller.

### Case 1: First-order gain controller

Instead of setting $\alpha_t$ instantaneously, we can treat it as a state variable. In continuous time:

```math
\dot{\alpha}
=
k_\alpha(\chi-\eta\alpha C^{\mathrm{ctrl}}).
```

If $\eta\alpha C^{\mathrm{ctrl}}>\chi$, then $\dot{\alpha}<0$, so the controller brakes. If $\eta\alpha C^{\mathrm{ctrl}}<\chi$, then $\dot{\alpha}>0$, so the controller accelerates.

For locally constant $C^{\mathrm{ctrl}}$, the equilibrium is

```math
\alpha^\star
=
\frac{\chi}
{\eta C^{\mathrm{ctrl}}}.
```

A discrete-time implementation is

```math
\alpha_{t+1}
=
\alpha_t
+
k_\alpha
\left(
\chi-\eta\alpha_t C_t^{\mathrm{ctrl}}
\right).
```

To choose $k_\alpha$, assume $C_t^{\mathrm{ctrl}}=C$ is locally constant. Then the error $u_t=\alpha_t-\alpha^\star$ evolves as

```math
u_{t+1}
=
(1-k_\alpha\eta C)u_t.
```

Thus, stable $\alpha$-adaptation requires

```math
0<k_\alpha\eta C<2.
```

For non-oscillatory adaptation, choose

```math
0<k_\alpha\eta C<1.
```

Using an estimated maximum curvature $C_{\max}$, a conservative choice is

```math
k_\alpha
=
\frac{c_k}{\eta C_{\max}},
\qquad
0<c_k<1.
``` 

<TBox type="summary" title="Key insight">
  A first-order controller can adapt the global throttle $\alpha_t$ over time to keep the effective learning rate near the stability margin. The adaptation speed can be tuned via $k_\alpha$. A first choice can be:
  ```math
  k_\alpha
  =
  \frac{0.5}{\eta C_{\max}} \quad \leadsto \quad \dot{\alpha}=k_\alpha(\chi-\eta\alpha C^{\mathrm{ctrl}}).
  ```
</TBox>

### Case 2: Second-order damped gain controller

A second-order controller gives $\alpha_t$ its own velocity:

```math
\ddot{\alpha}
+
k_d\dot{\alpha}
+
k_p
\left(
\eta\alpha C^{\mathrm{ctrl}}-\chi
\right)
=
0.
```

For locally constant $C^{\mathrm{ctrl}}$, the equilibrium is again

```math
\alpha^\star
=
\frac{\chi}
{\eta C^{\mathrm{ctrl}}}.
```

Let

```math
u=\alpha-\alpha^\star.
```

Then

```math
\eta\alpha C^{\mathrm{ctrl}}-\chi
=
\eta C^{\mathrm{ctrl}}u,
```

and the local controller dynamics become

```math
\ddot{u}
+
k_d\dot{u}
+
k_p\eta C^{\mathrm{ctrl}}u
=
0.
```

This is a damped second-order system with natural frequency

```math
\omega_n
=
\sqrt{k_p\eta C^{\mathrm{ctrl}}}
```

and damping ratio

```math
\zeta
=
\frac{k_d}
{2\sqrt{k_p\eta C^{\mathrm{ctrl}}}}.
```

Critical damping occurs when

```math
k_d
=
2\sqrt{k_p\eta C^{\mathrm{ctrl}}}.
```

In practice, choose a reference curvature $C_{\mathrm{ref}}$, a desired gain-settling time $T_\alpha$, and set

```math
\omega_n=\frac{4}{T_\alpha},
```

```math
k_p
=
\frac{\omega_n^2}
{\eta C_{\mathrm{ref}}},
```

```math
k_d=2\omega_n.
```

If one wants conservative non-ringing behavior over all expected curvatures $C\leq C_{\max}$, choose

```math
k_d
\geq
2\sqrt{k_p\eta C_{\max}}. 
```

A hardware-friendly discrete version is

```math
\begin{aligned}
v_{\alpha,t+1}
&=
\beta v_{\alpha,t}
+
k_\alpha
\left(
\chi-\eta\alpha_t C_t^{\mathrm{ctrl}}
\right) \\[8pt]
\alpha_{t+1}
&=
\alpha_t+v_{\alpha,t+1} 
\end{aligned}
```

which is basically a momentum update on $\alpha$ with a proportional term on the margin error. 

In fact, a good choice could be to make this system be critically-damped at the reference curvature $C_{\mathrm{ref}}$:

```math
k_\alpha
=
\frac{16}{T_\alpha^2\eta C_{\mathrm{ref}}}, \qquad
\beta
=
2\sqrt{k_\alpha\eta C_{\mathrm{ref}}}-1.
```

<TBox type="warning" title="But what is the best choice for $C_{\mathrm{ref}}$?">
  ???
</TBox>

### Case 3: Quantization-aware gain feasibility

In fixed-point training, reducing $\alpha_t$ too much can make the update vanish after quantization. If $q_\Delta$ is the update quantum, a useful update requires approximately

```math
\eta\alpha_t\|G_t\|_2
\gtrsim
q_\Delta.
```

Therefore,

```math
\alpha_t
\gtrsim
\frac{q_\Delta}
{\eta\|G_t\|_2+\epsilon}.
```

At the same time, stability requires

```math
\alpha_t
\lesssim
\frac{\chi}
{\eta C_t^{\mathrm{ctrl}}+\epsilon}.
```

Define

```math
\alpha_{\min,t}
=
\frac{q_\Delta}
{\eta\|G_t\|_2+\epsilon}
```

and

```math
\alpha_{\max,t}
=
\frac{\chi}
{\eta C_t^{\mathrm{ctrl}}+\epsilon}.
```

Useful stable fixed-point learning requires

```math
\alpha_{\min,t}
\leq
\alpha_{\max,t}.
```

If this interval collapses, the current fixed-point format cannot simultaneously provide stable and useful updates.