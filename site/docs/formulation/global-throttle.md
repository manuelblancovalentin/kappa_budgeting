# 🎚️ The global throttle mechanism

Instead of projecting each row/column independently, we introduce a single global scalar:

```math
0<\alpha_t\leq 1.
```

The update becomes:

```math
\boxed{
\theta_{t+1}
=
\theta_t-\alpha_t\eta G_t
}
```

where:

```math
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

So:

```math
\cos
\left(
\Delta\theta_t^{\text{ctrl}},
\Delta\theta_t^{\text{raw}}
\right)
=
1.
```

That means global throttling preserves the direction of learning. It only changes the speed.

This is the key contrast:
```math
\text{row/column κ projection}
\Rightarrow
\text{may rotate the update}
```
```math
\text{global throttle}
\Rightarrow
\text{preserves the update direction}
```


## Estimation of local curvature/sensitivity
Finding the actual hessian is computationally expensive and sometimes even not feasible. Thus, the controller chooses $\alpha_t$ using an online curvature estimate.

The gradient is:

```math
G(\theta)=\nabla_\theta\mathcal{L}(\theta).
```

The Hessian is the derivative of the gradient:

```math
H(\theta)
=
\frac{\partial G}{\partial\theta}
=
\nabla_\theta^2\mathcal{L}(\theta).
```

For a small change:

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

This is an observed directional curvature estimate.

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

This does not always equal:
```math
\lambda_{\max}(H_t).
```
More precisely, it estimates the curvature along the direction the optimizer just moved.


## Smoothing and control

To avoid reacting to noise, we can smooth the curvature estimate with an exponential moving average:
```math
S_t
=
(1-\rho)S_{t-1}
+
\rho \widehat{C}_t.
```
And use:

```math
C_t^{\text{ctrl}}
=
\max(\widehat{C}_t,S_t).
```

Finally:

```math
\boxed{
\alpha_t
=
\min
\left(
1,
\frac{\chi}{\eta(C_t^{\text{ctrl}}+\epsilon)}
\right)
}
```

where:

* $\chi$ is the desired stability margin,
* $\eta$ is the base learning rate,
* $C_t^{\text{ctrl}}$ is the curvature/sensitivity estimate,
* $\epsilon$ avoids division by zero.

The controller attempts to enforce:

```math
\boxed{
\alpha_t\eta C_t^{\text{ctrl}}
\leq
\chi.
}
```

## Caveat and summary
The previous estimate is not perfect. It is very good for convex problems and things like a single-dense layer architecture, but it can be inaccurate for non-convex problems and more complex architectures. However, it is a simple and computationally cheap way to get a sense of the local curvature/sensitivity, which is what we need to adapt the learning rate and prevent divergence.

<div class="summary-box">
    <strong>Key insight:</strong> The global throttle mechanism preserves the geometry of the original gradient update, but it adaptively reduces the learning rate when the gradient field becomes stiff/sharp. This allows the model to keep learning without diverging, even under distribution shift.
</div>


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

Therefore, for general networks we should not claim:
```math
\widehat{C}_t=\lambda_{\max}(H_t)
```
exactly.

Instead, we claim:
```math
\widehat{C}_t
```
is an online estimate of local update-field sensitivity:
```math
\widehat{C}_t
\approx
\frac{
\|H_t\Delta\theta_t\|
}{
\|\Delta\theta_t\|
}.
```
This is a directional curvature estimate.

For a general loss, we want to keep:
```math
\alpha_t\eta L_t<\chi,
```
where $L_t$ is a local estimate of gradient-field Lipschitzness.

The controller uses:

```math
L_t\approx C_t^{\text{ctrl}}.
```

So:

```math
\boxed{
\alpha_t\eta C_t^{\text{ctrl}}\leq\chi.
}
```

<div class="summary-box">
    <strong>Key insight:</strong> For the linear case, this matches the true Hessian stability condition. For general networks, it becomes a local adaptive control rule.
</div>


## Implementation

Implementation-wise, this throttle can be implemented layer by layer, but the final controller is global. For each layer $\ell$, we have parameters $\theta_{\ell,t}$ and gradients $G_{\ell,t}$.

We can compute local contributions:
```math
A_{\ell,t}
=
\|G_{\ell,t}-G_{\ell,t-1}\|_2^2,
```
```math
B_{\ell,t}
=
\|\theta_{\ell,t}-\theta_{\ell,t-1}\|_2^2.
```

Then aggregate globally:
```math
A_t
=
\sum_{\ell=1}^L A_{\ell,t},
```
```math
B_t
=
\sum_{\ell=1}^L B_{\ell,t}.
```

Then:

```math
\boxed{
\widehat{C}_t
=
\sqrt{
\frac{
\sum_{\ell=1}^L
\|G_{\ell,t}-G_{\ell,t-1}\|_2^2
}{
\sum_{\ell=1}^L
\|\theta_{\ell,t}-\theta_{\ell,t-1}\|_2^2
+\epsilon
}
}.
}
```

This is equivalent to flattening all layers into one vector:

```math
\theta_t=
\operatorname{vec}(\theta_{1,t},\dots,\theta_{L,t}),
```
```math
G_t=
\operatorname{vec}(G_{1,t},\dots,G_{L,t}),
```
and computing:
```math
\widehat{C}_t
=
\frac{
\|G_t-G_{t-1}\|_2
}{
\|\theta_t-\theta_{t-1}\|_2+\epsilon
}.
```

Hardware-wise, this can be implemented as streaming reductions:

```math
\text{accum\_dG2}
\leftarrow
\text{accum\_dG2}
+
(G_{\ell,t}-G_{\ell,t-1})^2,
```
```math
\text{accum\_dtheta2}
\leftarrow
\text{accum\_dtheta2}
+
(\theta_{\ell,t}-\theta_{\ell,t-1})^2.
```

At the end, compute one global scalar: $\widehat{C}_t$. Then broadcast one global: $\alpha_t$

So we do not multiply per-layer gains at the top level for this controller. The controller is based on the global sensitivity of the optimizer trajectory.

Layerwise versions are possible later:
```math
\widehat{C}_{\ell,t}
=
\frac{
\|G_{\ell,t}-G_{\ell,t-1}\|
}{
\|\theta_{\ell,t}-\theta_{\ell,t-1}\|+\epsilon
}.
```

But the first version should be global because global scaling preserves update geometry.


<div class="summary-box">
    <strong>Key insight:</strong> The estimator can be accumulated layer by layer, but the control action is a single global scalar. That lets us keep hardware implementation simple and avoids layerwise distortion of the descent direction.
</div>