---
status:
  - preliminary
tags:
  - revise
  - formulation
  - stability
last_modified: 2026-05-15
author: mbvalentin
---
# 🌀 1-Lipschitz and stability
<PageMeta />
---

## Lipschitzness
Let us start with a definition. A function $f:\mathbb{R}^n\to\mathbb{R}^m$ is 1-Lipschitz if:
```math
\|f(x)-f(y)\|
\leq
\|x-y\|
```
for all $x,y\in\mathbb{R}^n$. 

**What does this all mean?**. I know. It may look like a random math definition, but it has a very intuitive meaning. In plain english, this means that if you change the input a little bit, the output will also change only a little bit. In other words, the function is not "stretching" the input too much. 

From a systems point of view, this is the definition of a sort of overall **gain**. In fact, we can define the Lipschitzness as the maximum gain of the function:
```math
L
=\sup_{x\neq y}
\frac{\|f(x)-f(y)\|}{\|x-y\|}.
```

In other words, this constant is how much the input gets amplified or attenuated by the function mapping.

**Why do we care about this property?**. You might be wondering that. In order to answer that we need to change gears from general discrete updates of training neural networks, to dynamical systems. 


## The gradient feedbackloop and network stability
Recall the standard training loop using SGD in continuous time:
```math
\frac{d\theta}{dt}
=
-\eta \nabla_\theta \mathcal{L}(\theta).
```

Now let's imagine a simple 2-dense layer network like this:


<SeqDiagram
  maxWidth="800px"
  items={[
    {label: String.raw`x`, kind: 'plain'},
    {
      label: String.raw`\theta_1: W_1,b_1`,
      kind: 'box',
    },
    {label: String.raw`y_1`, kind: 'plain'},
    {
      label: String.raw`\theta_2: W_2,b_2`,
      kind: 'box',
    },
    {label: String.raw`\hat{y}`, kind: 'plain'},
  ]}
/>

Let's explicitly write the function and the intermediate activations:
```math
\begin{cases}
y_1 &= W_1x+b_1, \\
\hat{y} &= W_2y_1+b_2.
\end{cases}
```

The loss is:
```math
\mathcal{L}(\theta_1,\theta_2)
=
\ell(\hat{y},y_{\text{true}}).
```

The gradient with respect to each parameter is:
```math
\theta_2: \begin{cases}
\nabla_{W_2}\mathcal{L}
&=
\frac{\partial \ell}{\partial \hat{y}}
\frac{\partial \hat{y}}{\partial W_2}
=
\frac{\partial \ell}{\partial \hat{y}}
\frac{\partial (W_2y_1+b_2)}{\partial W_2} 
= 
\frac{\partial \ell}{\partial \hat{y}} y_1 \\[8pt]
\nabla_{b_2}\mathcal{L}
&=
\frac{\partial \ell}{\partial \hat{y}}
\frac{\partial \hat{y}}{\partial b_2}
=
\frac{\partial \ell}{\partial \hat{y}}
\frac{\partial (W_2y_1+b_2)}{\partial b_2} 
=
\frac{\partial \ell}{\partial \hat{y}} \\[8pt]
\end{cases}
\qquad 
\theta_1: \begin{cases}
\nabla_{W_1}\mathcal{L}
&=
\frac{\partial \ell}{\partial \hat{y}}
\frac{\partial \hat{y}}{\partial y_1}
\frac{\partial y_1}{\partial W_1}
=
\frac{\partial \ell}{\partial \hat{y}}
\frac{\partial (W_2y_1+b_2)}{\partial y_1}
\frac{\partial (W_1x+b_1)}{\partial W_1} 
=
\frac{\partial \ell}{\partial \hat{y}} W_2 x \\[8pt]
\nabla_{b_1}\mathcal{L}
&=
\frac{\partial \ell}{\partial \hat{y}}
\frac{\partial \hat{y}}{\partial y_1}
\frac{\partial y_1}{\partial b_1}
=
\frac{\partial \ell}{\partial \hat{y}}
\frac{\partial (W_2y_1+b_2)}{\partial y_1}
\frac{\partial (W_1x+b_1)}{\partial b_1} 
=
\frac{\partial \ell}{\partial \hat{y}} W_2
\end{cases}
```

But recall that $y_1$ is:
```math
y_1=W_1x+b_1.
```

But now note what happens when we write the update rule for each parameter, in terms of these gradients:
```math
\begin{cases}
\dot{W_2} = -\eta \frac{\partial \ell}{\partial \hat{y}} (W_1x + b_1) \\
\dot{b_2} = -\eta \frac{\partial \ell}{\partial \hat{y}} \\
\dot{W_1} = -\eta \frac{\partial \ell}{\partial \hat{y}} W_2 x \\
\dot{b_1} = -\eta \frac{\partial \ell}{\partial \hat{y}} W_2
\end{cases}
```

This system has the form:
```math
\dot{\theta} = A(\theta)\theta + b(\theta) + c(\theta)x,
```
where the matrix $A(\theta)$ depends on the parameters themselves. This is a **nonlinear dynamical system**(non-linear because the $\frac{\partial\ell}{\partial\hat{y}}$ term will most likely -depending on the specific loss function- depend on $y_2$, which depends on $\theta$), and its stability depends on the eigenvalues of the Jacobian of the update rule. In particular, the gain of the second layer $W_2$ directly affects the stability of the updates of the first layer $\theta_1$. If $W_2$ has a large norm, then the updates of $\theta_1$ can become very large, which can lead to instability in the training process.

This makes it clear that the update of $\theta_1$ depends on the gain of the second layer, which on its turn depends on the gain of the first layer. If the gain of the second layer is too high, then the update of $\theta_1$ can become unstable, even if the loss landscape is smooth. In other words, the training update itself creates two feedback loops, as captured by the following diagram:

<FeedbackLoopDiagram
  id="fig-gradient-feedback-loop"
  maxWidth="920px"
/>

<TBox type="summary" title="Key insight">
  The stability of the training process is not only governed by the smoothness of the loss landscape, but also by the gain of the network itself. The gain of the network can amplify the updates in a way that can lead to instability, even if the loss landscape is smooth. This is why controlling the Lipschitzness of the network is important for stable training.
</TBox>



## So what would the ideal Lipschitzness be?
The ideal Lipschitzness for stability is 1. If the network is 1-Lipschitz, then the gain of the network is at most 1, which means that the updates will not be amplified too much, and the training process will be stable. If the network is not 1-Lipschitz, then the gain can be greater than 1, which can lead to instability in the training process. Therefore, controlling the Lipschitzness of the network is crucial for ensuring stable training.