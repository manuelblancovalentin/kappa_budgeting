# ❌ Why κ row/col failed
<StatusBadges statuses="revise;preliminary" />
---

In our initial experiments, we found that we had to disable $\kappa$-row projection (```RowScale{}``` block) to get any meaningful learning. This is because the original $\kappa$ mechanism was trying to enforce local gain constraints on each layer, row, or column. A simplified version is:
```math
W_\ell \leftarrow \Pi_{\kappa_\ell}(W_\ell),
```

where:

* $W_\ell$ is the weight matrix of layer $\ell$,
* $\kappa_\ell$ is the allowed gain/norm budget for that layer,
* $\Pi_{\kappa_\ell}$ is some projection/rescaling operator that pushes $W_\ell$ back into the allowed range.

For row scaling, this looks roughly like:

```math
W_{\ell,i:}
\leftarrow
s_{\ell,i} W_{\ell,i:},
```

where $W_{\ell,i:}$ is row $i$ of layer $\ell$, and $s_{\ell,i}$ is a row-dependent scale factor.

For column scaling:

```math
W_{\ell,:j}
\leftarrow
s_{\ell,j} W_{\ell,:j}.
```

The problem is that online learning is not just about whether the weights are inside a norm box. It is about the direction in parameter space.

If the unconstrained gradient update is:

```math
\theta_{t+1}^{\text{raw}}
=
\theta_t-\eta G_t,
```

where:

* $\theta_t$ is the flattened vector of all trainable parameters,
* $G_t=\nabla_\theta \mathcal{L}(\theta_t)$ is the gradient,
* $\eta$ is the learning rate,

then the ideal update direction is:

```math
\Delta\theta_t^{\text{raw}}
=
-\eta G_t.
```

But with row/column $\kappa$ projection, the actual update becomes:

```math
\theta_{t+1}^{\text{actual}}
=
\Pi_\kappa(\theta_t-\eta G_t),
```

so the actual applied update is:

```math
\Delta\theta_t^{\text{actual}}
=
\Pi_\kappa(\theta_t-\eta G_t)-\theta_t.
```

This is generally not parallel to the gradient direction.

The diagnostic is the cosine similarity:

```math
\cos_t
=
\frac{
\left\langle
\Delta\theta_t^{\text{actual}},
\Delta\theta_t^{\text{raw}}
\right\rangle
}{
\|\Delta\theta_t^{\text{actual}}\|_2
\|\Delta\theta_t^{\text{raw}}\|_2+\epsilon
}.
```

If:

```math
\begin{cases}
\text{projection preserved the intended learning direction} & \text{if } \cos_t\approx 1 \\
\text{projection made the update nearly orthogonal to the gradient} & \text{if } \cos_t\approx 0 \\
\text{projection is actively fighting the descent direction} & \text{if } \cos_t<0
\end{cases}
```


## Summary box
<div class="summary-box">
    <strong>Key insight:</strong> The row/column $\kappa$ projection was enforcing local numerical constraints, but the projection operator was not geometry-preserving. It could rotate the effective update in parameter space. For inference-time gain control that may be acceptable, but during online training it can interfere with the optimizer’s descent direction.
</div>


