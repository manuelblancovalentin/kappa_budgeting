# 🌀 1-Lipschitz and stability
<StatusBadges statuses="revise;preliminary" />
---

## 1. The original idea of $\kappa$-per layer
The original $\kappa$-budgeting idea is about controlling the network gain. For a network:
```math
f(x;\theta),
```
we want a Lipschitz bound:
```math
\|f(x_1;\theta)-f(x_2;\theta)\|
\leq
L
\|x_1-x_2\|.
```
Here:

* $x_1,x_2$ are two inputs,
* $f(x;\theta)$ is the network output,
* $L$ is the Lipschitz constant.

For a purely linear network:
```math
f(x)=W_LW_{L-1}\cdots W_1x,
```
the exact input-output gain is:
```math
K
=
\left\|
W_LW_{L-1}\cdots W_1
\right\|.
```

Using submultiplicativity of matrix norms:

```math
\left\|
W_LW_{L-1}\cdots W_1
\right\|
\leq
\prod_{\ell=1}^L
\|W_\ell\|.
```

So if we enforce:
```math
\|W_\ell\|\leq \kappa_\ell,
```
then:
```math
K
\leq
\prod_{\ell=1}^L \kappa_\ell.
```

That is the static $\kappa$-budgeting idea:
```math
\boxed{
\prod_{\ell=1}^{L}\kappa_\ell
\leq
L_{\max}.
}
```

<div class="summary-box">
    <strong>Key insight:</strong> The original $\kappa$-budgeting idea was about controlling the static Lipschitz gain of the network. This is important for inference-time robustness and numerical stability.
</div>

## 2. The problem of online training
But online training introduces a different map: not only
```math
x\mapsto f(x;\theta),
```
but also:
```math
\theta_t
\mapsto
\theta_{t+1}.
```
The learning rule is itself a dynamical system:
```math
\theta_{t+1}
=
\theta_t-\eta\nabla_\theta\mathcal{L}(\theta_t).
```
So the stability of online learning is governed by the sensitivity of the gradient field:
```math
G(\theta)
=
\nabla_\theta \mathcal{L}(\theta).
```
A natural smoothness/Lipschitz condition for the gradient is:
```math
\|G(\theta_a)-G(\theta_b)\|
\leq
L_G
\|\theta_a-\theta_b\|.
```
Here $L_G$ is the Lipschitz constant of the gradient field. In smooth optimization, this is related to the Hessian norm:
```math
L_G
\approx
\|H(\theta)\|_2,
```
where:
```math
H(\theta)=\nabla_\theta^2\mathcal{L}(\theta).
```
So there are two different Lipschitz ideas:

### a. Static network Lipschitzness
```math
\|f(x_1;\theta)-f(x_2;\theta)\|
\leq
L_f
\|x_1-x_2\|.
```
This is about input-output gain.

### b. Dynamic optimizer Lipschitzness

```math
\|\nabla\mathcal{L}(\theta_a)-\nabla\mathcal{L}(\theta_b)\|
\leq
L_G
\|\theta_a-\theta_b\|.
```

This is about how violently the gradient changes when the weights move.

$\kappa$-budgeting mostly targets the first. The new throttle targets the second.

## Summary box
<div class="summary-box">
    <strong>Key insight:</strong> We were controlling the static Lipschitz gain of the network, but the instability during online training is governed by the Lipschitzness of the gradient/update field. The new controller targets that closed-loop sensitivity.
</div>