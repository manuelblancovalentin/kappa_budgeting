---
status:
  - preliminary
tags:
  - todo
  - formulation
  - overview
last_modified: 2026-05-15
author: mbvalentin
---


# 🔎 Background
<PageMeta />
---

This project studies how to make fixed-point online learning stable when a deployed model must keep training after its input distribution changes.

[The <ENABOL /> paper](../../../iccad2026-paper1079.pdf) frames the stability problem as a bounded-gain control problem. A trainable fixed-point network should not let activations, gradients, weights, or optimizer updates exceed the rails implied by the chosen `ap_fixed` formats. The proposed mechanism is kappa budgeting: assign per-layer induced-norm budgets and enforce them during training.


## The very basics of neural networks

Let's start with the basics. We have some input data $x\in\R^{d_\mathrm{in}}$ that defines some predictive features; and some output data $y\in\R^{d_\mathrm{out}}$ that defines some observations; and we expect these two spaces to be related by some underlying function $f^\ast:\R^{d_\mathrm{in}}\to\R^{d_\mathrm{out}}$ that we want to learn. 

Then, we define a parametric model $f_\theta:\R^{d_\mathrm{in}}\to\R^{d_\mathrm{out}}$ that we hope can approximate $f^\ast$ well for some choice of parameters $\theta$. This is our neural network. 

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

The question now is **how do we tune $\theta$ to make $f_\theta$ a good approximation of $f^\ast$?** The standard answer is to define a loss function $\mathcal{L}:\R^{d_\mathrm{out}}\times\R^{d_\mathrm{out}}\to\R$ that measures how well the model's predictions match the true outputs (according to some criterion of wellness), and then minimize the expected loss over the data distribution:

```math
\theta^\ast = \arg\min_\theta \mathbb{E}_{(x,y)\sim\mathcal{D}}[\mathcal{L}(f_\theta(x), y)].
```

In practice, we usually don't have access to the true data distribution $\mathcal{D}$, so we work with a finite dataset $\{(x_i,y_i)\}_{i=1}^N$ and minimize the empirical loss:

```math
\theta^\ast = \arg\min_\theta \frac{1}{N}\sum_{i=1}^N \mathcal{L}(f_\theta(x_i), y_i).
```

## The loss landscape
Now, the loss function $\mathcal{L}$ and the model $f_\theta$ together define a landscape over the parameter space $\theta$. What on earth does this mean? Imagine a very *simple* problem in which we only have two parameters $\theta=(\theta_1,\theta_2)$, and the loss is a function of these two parameters. Because both $f_\theta$ and $\mathcal{L}$ are deterministic, we know that for any combination of input value $x$, output value $y$ and parameter values $\theta$, the final loss will be a single static number. 

This is important. Training, then, is just a process of `sampling` points in this loss landscape. We do not necessarily have access to the full landscape, nor we store it in memory, because doing so would be impossible for even moderately sized models (curse of dimensionality). So instead in regular stochastic training what we do is follow some local path through the landscape. But note that the important matter here is that the landscape itself is fixed. It does not change as we train. Parameters $\theta$ *DO* change, but the loss landscape itself is a static object that we are navigating through (as long as the loss function and model are deterministic, which is the case for our experiments).

<FigureRef target="fig-loss-landscape">Figure 1</FigureRef> shows an illustration of this. The 3D surface in this plot represents the loss landscape. The horizontal axes are the two parameters $\theta_1$ and $\theta_2$, and the vertical axis is the loss value $\mathcal{L}$. The landscape has hills, valleys, and plateaus. The goal of training is to find a path through this landscape that leads us to a low-loss region, ideally a global minimum.

<Figure
  id="fig-loss-landscape"
  src="/img/formulation/3D.png"
  alt="3D loss landscape arbitrary"
  maxWidth="70%"
  label="Figure 1"
  caption="Illustrative loss landscape for the closed-loop training stability discussion."
/>

## The gradient and curvature of the loss landscape
Since we are defining the loss as a function of fitness, we normally call `training` the process of following a certain path through the landscape such that we optimize this function (e.g., minimize the loss -as a definition of error- for a typical classification or regression problem). But what does it mean to follow a path through the landscape? How do we know which direction to go?

The most common answer is to use the gradient of the loss with respect to the parameters $\nabla_\theta \mathcal{L}(\theta)$. This gradient is basically a vector field that tells us: 

> If I change the parameters $\theta$ by a small infinitesimal amount $\delta$ in some particular direction, how does the loss change?

Imagine that you are blindfolded and stopped at some point in a hilly landscape. This particular point where you are is defined by the current *state* of your model, i.e., the current parameters $\theta$. Now, you want to find your way, say, downhill. What would you do? You would probably try to *feel* or *probe* the ground around you with your feet to see which direction is going down. This, my friend, is precisely what obtaining the gradient around that point means. The gradient is the local directionality of the loss landscape. 

Mathematically, this gradient is defined as:
```math
G(\theta) = \nabla_\theta \mathcal{L}(\theta) = \left(\frac{\partial \mathcal{L}}{\partial \theta_1}, \frac{\partial \mathcal{L}}{\partial \theta_2}, \ldots, \frac{\partial \mathcal{L}}{\partial \theta_d}\right).
```

## Gradient descent and the update rule
Once we have the gradient, we can use it to define an update rule for our parameters. The most basic one is the vanilla gradient descent update (also known Stochastic Gradient Descent or SGD when applied to mini-batches of data):
```math
\theta_{t+1} = \theta_t - \eta G(\theta_t),
```
where $\eta$ is the learning rate, a hyperparameter that controls how big of a step we take in the direction of the negative gradient. What this rule is telling us is that the change in our parameters at step $t$ (mathematically, $\Delta\theta_{t} = \theta_{t+1} - \theta_t$) is proportional to the negative of the gradient at step $t$. 

In other words, in continuous time, this would be represented by something like:
```math
\dot{\theta}(t) = -\eta G_\theta(t),
```
which is a differential equation that describes how the parameters evolve over time under the influence of the gradient. Now, this is one of the easiest and most intuitive ODEs to solve, because it's simple and first order. 

Given that $\eta$ is a positive constant, if we analyze its stability we see that this system will converge to a local minimum of the loss as long as $\eta$ is not too large. Why? Let's apply Lyapunov stability analysis to see why. 

## Lyapunov stability analysis on SGD
We want to find a Lyapunov function $\mathcal{V}(\theta)$ that can help us analyze the stability of the system. A natural choice for this function is the loss itself, $\mathcal{V}(\theta) = \mathcal{L}(\theta)$. Now, we need to compute the time derivative of this Lyapunov function along the trajectories of the system. Using the chain rule (yes, the standard chain rule of calculus they teach you at school), we have:
```math
\dot{\mathcal{V}}(\theta) = \nabla_\theta \mathcal{L}(\theta)^\top \dot{\theta} = G(\theta)^\top (-\eta G(\theta)) = -\eta \|G(\theta)\|^2.
``` 

Since $\|G(\theta)\|^2$ is always non-negative, we have that $\dot{\mathcal{V}}(\theta) \leq 0$ for all $\theta$. This means that the Lyapunov function $\mathcal{V}(\theta)$ is non-increasing along the trajectories of the system. Therefore, the system is stable in the sense of Lyapunov. Moreover, if the loss function $\mathcal{L}(\theta)$ is radially unbounded (i.e., $\mathcal{L}(\theta) \to \infty$ as $\|\theta\| \to \infty$), then we can conclude that the system is globally asymptotically stable, meaning that it will converge to a local minimum of the loss from any initial condition.

But what about the maximum $\eta$ value that we mentioned earlier? To find this, we can look at the curvature of the loss landscape, which is given by the Hessian matrix $H(\theta) = \nabla^2_\theta \mathcal{L}(\theta)$. The eigenvalues of this Hessian matrix tell us about the curvature in different directions. If the largest eigenvalue of the Hessian is $\lambda_{\max}$, then we need to ensure that our learning rate $\eta$ is less than $2/\lambda_{\max}$ to guarantee convergence. This is because if $\eta$ is too large, we might overshoot and diverge instead of converging to a minimum.

```math
\boxed{
\text{For convergence, we need: } \eta < \frac{2}{\lambda_{\max}(H(\theta))}.
}
```

**Slow down. Where did this condition come from?** Let's derive it step by step and very carefully. We will use another basic tool for this: Taylor expansion. First, recall the formula for Taylor series:
```math
f(x) = \sum_{n=0}^\infty \frac{f^{(n)}(a)}{n!}(x-a)^n \quad \leadsto \quad f(x) \approx f(a) + f'(a)(x-a) + \frac{1}{2}f''(a)(x-a)^2 + \ldots
```
where $f^{(n)}(a)$ is the $n$-th derivative of $f$ evaluated at $a$. Now, for any given point $\theta$, let's approximate this loss function around $a$ up to second order:
```math
\mathcal{L}(\theta) 
\approx 
\underbrace{\mathcal{L}(a)}_{\text{0th order}} + 
\underbrace{\nabla_\theta \mathcal{L}(a)^\top (\theta - a)}_{\text{1st order}} + 
\underbrace{\frac{1}{2}(\theta - a)^\top \nabla^2_\theta \mathcal{L}(a) (\theta - a)}_{\text{2nd order}} + \ldots
```

But now, let's note something important:
* $\nabla_\theta \mathcal{L}$ is precisely the gradient $G(\theta)$ introduced before, and
* $\nabla^2_\theta \mathcal{L}$ is precisely the Hessian $H(\theta)$, and

So we can rewrite the Taylor expansion as:
```math
\mathcal{L}(\theta)
\approx
\mathcal{L}(a) +
G(a)^\top (\theta - a) +
\frac{1}{2}(\theta - a)^\top H(a) (\theta - a) + \ldots
```

Now, let's analyze the stability of this system around some small perturbation $\delta$ around a local minimum $\theta^\ast$. This means that we will set $a = \theta^\ast$ and $\theta = \theta^\ast + \delta$. Substituting this into the Taylor expansion, we get:
```math
\mathcal{L}(\theta^\ast + \delta)
\approx
\mathcal{L}(\theta^\ast) +
G(\theta^\ast)^\top \delta +
\frac{1}{2}\delta^\top H(\theta^\ast) \delta + \ldots
```

Now, if we apply the gradient descent update rule, we have:
```math
\theta_{t+1} = \theta_t - \eta G(\theta_t).
```

Let's define $\delta_t = \theta_t - \theta^\ast$, where $\theta^\ast$ is a local minimum of the loss. Then, we can write:
```math
\delta_{t+1} = \theta_{t+1} - \theta^\ast = \theta_t - \eta G(\theta_t) - \theta^\ast = \delta_t - \eta G(\theta_t).
```

Now, we can use the Taylor expansion of the loss around $\theta^\ast$ to express $G(\theta_t)$ in terms of $\delta_t$:
```math
G(\theta_t) = \nabla_\theta \mathcal{L}(\theta_t) \approx \nabla_\theta \mathcal{L}(\theta^\ast) + H(\theta^\ast) \delta_t = H(\theta^\ast) \delta_t,
``` 

since $\nabla_\theta \mathcal{L}(\theta^\ast) = 0$ at a local minimum. Substituting this back into the update for $\delta_{t+1}$, we get:
```math
\delta_{t+1} = \delta_t - \eta H(\theta^\ast) \delta_t = (I - \eta H(\theta^\ast)) \delta_t.
```

Now, for the system to be stable, we need the spectral radius of the matrix $(I - \eta H(\theta^\ast))$ to be less than 1. The spectral radius is the largest absolute value of the eigenvalues of this matrix. The eigenvalues of $(I - \eta H(\theta^\ast))$ are given by $1 - \eta \lambda_i$, where $\lambda_i$ are the eigenvalues of $H(\theta^\ast)$. Therefore, we need:
```math
|1 - \eta \lambda_i| < 1 \quad \forall i.
``` 
This leads to the condition:

```math
-1 < 1 - \eta \lambda_i < 1 \quad \forall i,
```
sum $+1$ to both sides, and this simplifies to:

```math
0 < \eta \lambda_i < 2 \quad \forall i.
```

Since $\lambda_{\max}$ is the largest eigenvalue, we need to ensure that $\eta \lambda_{\max} < 2$, which gives us the final condition for convergence:
```math
\boxed{
\eta < \frac{2}{\lambda_{\max}(H(\theta^\ast))}.
}
```

## Why stop at the Hessian?
When analyzing stability we introduced an approximation around some point, via taylor expansion, but we stopped at second order (Hessian). But why not go further? **We could, in principle**. The reason why we don't is ... boring: *the hessian is already too computationally expensive for us to compute, let alone higher-order derivatives*.

In fact, we never even use Hessians in practice for our neural networks. Even modest-sized neural networks have millions of parameters, which means that computing the full Hessian matrix would require storing and manipulating a matrix with trillions of entries, which is completely infeasible. Instead, we often use approximations to the Hessian, such as the diagonal approximation used in algorithms like Adam, or we use techniques like stochastic estimation of the curvature.

## The issue of online learning
Training a neural network on chip has many issues, but in this case we are going to focus on two:
1. Quantization and fixed-point representation turns our problem into a non-trivial control problem because quantization introduces errors in the loss-landscape;
2. The data distribution can change over time, which means that the loss landscape itself can change as we train.

Both of these problems, as you can see, are issues that, in the end, affect in the same way: they can change the training process in a way that can make the system unstable. The task of <ENABOL /> is to design a controller that can keep the training process stable under these conditions.


