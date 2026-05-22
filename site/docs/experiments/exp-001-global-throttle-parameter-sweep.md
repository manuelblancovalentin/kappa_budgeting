---
id: exp-001-global-throttle-parameter-sweep
title: "EXP-001: Global Throttle Parameter Sweep"
sidebar_label: "EXP-001: GLTHR PARAM SWEEP"
status:
  - valid
tags:
  - experiment
  - global-throttle
  - quantization
  - qfx
  - lin1
last_modified: 2026-05-15
author: mbvalentin
workspace: "workspace/ablations/exp_001_lin1_stability_phase_diagram/"
notebook: "${WORKSPACE}/notebooks/exp_001_global_throttle_parameter_sweep.ipynb"
notebook_url: "https://github.com/manuelblancovalentin/kappa_budgeting/blob/master/workspace/ablations/exp_001_lin1_stability_phase_diagram/notebooks/exp_001_global_throttle_parameter_sweep.ipynb"
---
# EXP-001: Global Throttle Parameter Sweep

<PageMeta />

---

## Experiment Trace

| Field | Value |
|---|---|
| Dataset | [`DS-AFFINE-LINEAR-000`](../datasets/affine-linear-000.md) |
| Model | [`MDL-DENSE1-LINEAR-NOBIAS-000`](../models/dense1-linear-nobias-000.md) |
| Optimizer | SGD-style online loop |
| Controller | <ControllerBadge controller="none" /> <ControllerBadge controller="gt-order-0" />  <ControllerBadge controller="gt-order-1" /> |
| Precision mode | `ap_fixed<16,8, AP_RND, AP_SAT>` |
| Ablation focus | [`ST-000`](../ablations/st-000-high-learning-rate-sanity.md)|
| Drift | Parameter sweep $\gamma \in [1.0, 4.0]$ |
| Learning rate range | $\eta \in [0.01, 2.0]$ |

---

## Summary and background

<TBox type="summary" title="Purpose of the experiment">
  This experiment aims to investigate how the stability of the system depends on the learning rate $\eta$ and the drift parameter $\gamma$ under different global throttle controllers. By sweeping over a range of $\eta$ and $\gamma$ values, we can visualize the stability regions and understand the impact of the global throttle on the training dynamics. As we introduced in the [Formulation Overview section](../formulation/overview#lyapunov-stability-analysis-on-sgd), the condition of convergence depends on:

  ```math
  \eta > \frac{2}{\lambda_{\max}(H_\text{drift})} \approx \frac{2}{\gamma^2\lambda_{\max}(H_\text{nom})}
  ```
  where we estimate the $\lambda_{\max}(H(\theta))$ with the curvature proxy $\widehat{C}$, and thus:
  
  ```math
  \eta \gtrsim \frac{\chi}{\widehat{C}}\frac{1}{\gamma^2}
  ```
  
  Because this $\lambda_{\max}(H(\theta))$ depends on $\gamma$ (the drift parameter), we expect to see a stability region in the $(\eta, \gamma)$ space, for which the system diverges (final loss is bigger than some threshold) when no controller is used; but that it does converge (or at least that the divergence area is smaller) when the global throttle is used ($\alpha$ is used):

  ```math 
  \alpha\eta \gtrsim \frac{\chi}{\widehat{C}}\frac{1}{\gamma^2}
  ```

  
</TBox>



## Setup and code
Here we will briefly describe the setup of the experiment, and then we will jump to the results. For more details on the code, please check the [notebook](../../../workspace/ablations/exp_001_lin1_stability_phase_diagram/notebooks/exp_001_global_throttle_parameter_sweep.ipynb) where the code is fully documented and explained.

First, we define the global parameters for the simulation (sweep, thresholds, etc):
```python
N = 1000
SEED = 42
MAX_STEPS = 1000
TARGET_LOSS = None
PATIENCE = 0
BLOWUP_LOSS = 1e2
controller_enabled = True
controller_name = [None, "gt-order-0", "gt-order-1", "gt-order-2"][2 if controller_enabled else 0]

# Gamma space 
ETAS = np.logspace(-2, 0.3, 41)         # 0.01 to about 2
GAMMAS = np.geomspace(1.0, 4.0, 41)    # 1 to 16
CHI = 1.0
PRECISIONS = [[16, 6]] # [[WL, IWL]]
```

The creation of the dataset and the model is exactly as shown in [EXP-000A](exp-000a-global-throttle-float-lin1.md) and [EXP-000B](exp-000b-global-throttle-qfx-lin1.md), so we skip the details here. Because this test does require a few utilities, we'll add them here explicitly. 

### Reinitialization utility
We need to make sure that **all tests start from the same initial conditions** (same initial weights, same data distribution, etc) to make sure that the results are comparable. For that, we create a utility to reset the model weights to their initial state before every run of the simulation.
```python
# Utility to reset the model weights to their initial state (we want every point to start from the same initial condition)
initial_weights = model.model.get_weights()

def reset_model(model):
    model.model.set_weights(initial_weights)
```

### Make precision utility
In the results shown here we do not sweep over different quantization/precisions, but doing so would be straightforward by just adding more precisions to the `PRECISIONS` variable and then looping over them in the main loop. For that reason, we create a utility to convert the model weights to a given precision, so that we can easily apply it in case we want to add quantization sweeps in the future.

<TBox type="warning" title="Note on the precision stack">
  Note that in this case we assume some relations between the precisions of different tensors (e.g., the gradient WL is 4 bits wider than the weight WL, etc). This is, by no means, the optimal way to do it, and it should be investigated further in the future to answer the question **what is the optimal precision stack for any layer, given the precision of the weights?**. For now, we just assume some relations that seem reasonable and that are consistent with what we have seen in previous experiments.
</TBox>

```python
from enabol import dtypes, PrecisionDict

def make_precision_dict(BASE_WL, BASE_IWL, QMODE="AP_RND", OMODE="AP_SAT"):
    wide_weight = dtypes.ap_fixed(WL=BASE_WL, IWL=BASE_IWL, QMODE=QMODE, OMODE=OMODE)
    wide_activation = dtypes.ap_fixed(WL=BASE_WL, IWL=BASE_IWL+2, QMODE=QMODE, OMODE=OMODE)
    wide_gradient = dtypes.ap_fixed(WL=BASE_WL+4, IWL=BASE_IWL+2, QMODE=QMODE, OMODE=OMODE)
    wide_update = dtypes.ap_fixed(WL=BASE_WL+4, IWL=BASE_IWL, QMODE=QMODE, OMODE=OMODE)
    wide_accumulator = dtypes.ap_fixed(WL=BASE_WL+12, IWL=BASE_IWL+8, QMODE=QMODE, OMODE=OMODE)
    wide_loss = dtypes.ap_fixed(WL=2*BASE_WL, IWL=BASE_IWL+10, QMODE=QMODE, OMODE=OMODE)
    return PrecisionDict({
                "input": {"value": wide_activation},
                "dense0": {
                    "weight": wide_weight,
                    "activation": wide_activation,
                    "gradient": wide_gradient,
                    "update": wide_update,
                    "accumulator": wide_accumulator,
                },
                "loss": {"value": wide_loss},
            })
```

### Summarizing the results
Inside the loop we will get some results from the metrics and we'll have to summarize them in a way that can be useful later. For that, we create a small utility function:

```python
def safe_max(h, key):
    if key not in h:
        return np.nan
    x = np.asarray(h[key], dtype=float)
    if x.size == 0:
        return np.nan
    return float(np.nanmax(x))


def safe_min(h, key):
    if key not in h:
        return np.nan
    x = np.asarray(h[key], dtype=float)
    if x.size == 0:
        return np.nan
    return float(np.nanmin(x))


def safe_final(h, key):
    if key not in h:
        return np.nan
    x = np.asarray(h[key], dtype=float)
    if x.size == 0:
        return np.nan
    return float(x[-1])

def summarize_run(h, eta, gamma, precision_dict, controller_name):
    loss = h["loss"]
    finite = np.all(np.isfinite(loss))
    diverged = (not finite) or bool(np.any(h["diverged"] > 0)) or np.nanmax(loss) > BLOWUP_LOSS

    converged = False
    converged_step = np.nan
    if not diverged:
        if TARGET_LOSS is not None:
            below = loss < TARGET_LOSS
            for idx in range(len(loss) - PATIENCE):
                if np.all(below[idx:idx + PATIENCE]):
                    converged = True
                    converged_step = idx
                    break
        else:
            converged = True
            converged_step = len(loss)

    return {
        "eta": eta,
        "gamma": gamma,
        "controller": controller_name or "none",
        "precision": str(precision_dict["dense0"]["weight"]),
        "diverged": diverged,
        "converged": converged,
        "converged_step": converged_step,
        "final_loss": float(loss[-1]),
        "log_final_loss": float(np.log10(loss[-1] + 1e-12)) if not diverged else np.nan,
        "final_alpha": float(h["alpha"][-1]),
        "min_alpha": float(np.nanmin(h["alpha"])),
        "min_update_cosine": float(np.nanmin(h["update_cosine"])),
        "max_grad_sat": float(np.nanmax(h["gradient_saturation_fraction_max"])),
        "max_curvature_proxy": safe_max(h, "curvature_proxy"),
        "max_curvature_for_control": safe_max(h, "curvature_for_control"),
        "max_hessian_lambda": safe_max(h, "hessian_lambda_max"),
        "max_hessian_spectral_norm": safe_max(h, "hessian_spectral_norm"),
        "max_margin_lambda_raw": safe_max(h, "stability_margin_lambda_raw"),
        "max_margin_lambda_ctrl": safe_max(h, "stability_margin_lambda_ctrl"),
        "max_margin_norm_raw": safe_max(h, "stability_margin_norm_raw"),
        "max_margin_norm_ctrl": safe_max(h, "stability_margin_norm_ctrl"),
        "max_spectral_radius_raw": safe_max(h, "spectral_radius_raw"),
        "max_spectral_radius_ctrl": safe_max(h, "spectral_radius_ctrl"),
        "final_margin_lambda_raw": safe_final(h, "stability_margin_lambda_raw"),
        "final_margin_lambda_ctrl": safe_final(h, "stability_margin_lambda_ctrl"),
        "min_controller_feasible": safe_min(h, "controller_feasible"),
        "min_alpha_min_bound": safe_min(h, "alpha_min_bound"),
        "max_alpha_min_bound": safe_max(h, "alpha_min_bound"),
        "min_alpha_max_bound": safe_min(h, "alpha_max_bound"),
        "max_alpha_max_bound": safe_max(h, "alpha_max_bound"),
    }
```

### Sweep loop
The main loop that performs the sweeps, trains, and summarizes the results is the following:

```python
results = []

total_runs = len(PRECISIONS) * len(ETAS) * len(GAMMAS)
for i, (WL, IWL) in enumerate(PRECISIONS):
    precision_dict = make_precision_dict(WL, IWL)
    
    for j, GAMMA in enumerate(GAMMAS):

        # Make drifted dataset
        Xd = GAMMA * X

        # Metrics
        drift_metrics = MetricsConfig(
            profiles=("core", "geometry", "stability", "teacher", "quantization", "controller_bounds"),
            reference_A=dataset.reference_weight_matrix / GAMMA,
        )


        for k, ETA in enumerate(ETAS):
            # Print
            iteration = i * len(GAMMAS) * len(ETAS) + j * len(ETAS) + k
            print(f"[{(iteration+1)/total_runs:.2%}] Running with ETA={ETA}, GAMMA={GAMMA}, {precision_dict['dense0']['weight']}")

            # Reset model weights
            reset_model(model)
            
            # We need a new controller for every run, since it has states (not for 0th or None, but it's good practice anyways)
            controller = enabol.Controller.from_str(controller_name, chi=CHI)

            # Train
            h_nom = model.train_instrumented(
                Xd,
                Y,
                epochs=MAX_STEPS,
                stop_loss=None,       # auto means lower bound according to quantization.
                diverge_loss=BLOWUP_LOSS,   # auto means upper bound according to quantization.
                stop_patience=1, 
                stop_min_steps=0,
                batch_size=len(Xd),
                learning_rate=ETA,
                controller=controller,
                metrics=drift_metrics,
                precision_dict=precision_dict,
                verbose = 'progressbar'
            )

            # Append result
            results.append(summarize_run(h_nom, ETA, GAMMA, precision_dict, controller_name))

import pandas as pd
df = pd.DataFrame(results)
FILE = f"../results/parameter_sweep_controller_{controller_name}.csv"
df.to_csv(FILE, index=False)
```

### Plotting the results
Finally, we can plot the results using the following utility function:

```python

# Now we need to form the (eta,gamma) grid that's the same for all metrics.
import numpy as np
import matplotlib.pyplot as plt

def _parse_float(value: str) -> float:
    if value == "" or value is None:
        return np.nan
    return float(value)

def get_mesh(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    etas = np.array(sorted(df['eta'].unique()), dtype=float)
    gammas = np.array(sorted(df['gamma'].unique()), dtype=float)
    return etas, gammas

def pivot_grid(
    df: pd.DataFrame,
    etas: np.ndarray,
    gammas: np.ndarray,
    *,
    value: str,
    black_diverged: bool = True,
) -> np.ndarray:
    grid = np.full((len(etas), len(gammas)), np.nan, dtype=float)

    eta_index = {eta: i for i, eta in enumerate(etas)}
    gamma_index = {gamma: j for j, gamma in enumerate(gammas)}
    for _, row in df.iterrows():
        eta = float(row["eta"])
        gamma = float(row["gamma"])
        if black_diverged and row.get("diverged"):
            continue
        grid[eta_index[eta], gamma_index[gamma]] = _parse_float(row.get(value, ""))
    return grid

def plot_phase(
    etas: np.ndarray,
    gammas: np.ndarray,
    *,
    grids: dict[str, np.ndarray],
    title: str | None = None,
    cmaps: dict[str, str] = {"log_final_loss": "viridis"},
):

    NPLOTS = len(grids)
    NCOLS = np.sqrt(NPLOTS).round().astype(int)
    NROWS = np.ceil((NPLOTS / NCOLS)).astype(int)

    fig, axs = plt.subplots(nrows=NROWS, ncols=NCOLS, figsize=(7*NCOLS, 5*NROWS), 
                            sharex=True, sharey=True)

    axs = np.array(axs).flatten() if NPLOTS > 1 else np.array([axs])

    if len(etas) >= 2 and len(gammas) >= 2:
        for ax, (value, grid) in zip(axs, grids.items()):
            im = ax.imshow(
                np.ma.masked_invalid(grid),
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                cmap=cmaps.get(value, "viridis"),
                extent=[gammas.min(), gammas.max(), etas.min(), etas.max()],
            )

            ax.set_xlabel("γ")
            ax.set_ylabel("η")
            ax.set_yscale("log")
            ax.set_xscale("log")
            ax.set_title(title or value)
            fig.colorbar(im, ax=ax, label=value)
    else:
        raise ValueError("Need at least 2 unique values for eta and gamma to plot the phase diagram.")

    fig.tight_layout()
    plt.show()
```

which is called with the following code after loading the results from the CSV:

```python
import pandas as pd

# Reload (so I can re-run this code without re-running the whole sweep)
FILE = f"../results/parameter_sweep_controller_{controller_name}.csv"
#FILE = "../results/parameter_sweep_controller_gt0.csv"
#FILE = "../results/parameter_sweep_controller_gt-order-1.csv"
df = pd.read_csv(FILE)

from IPython.display import display
display(df)

etas, gammas = get_mesh(df)
grids = {key: pivot_grid(df, etas, gammas, value=key) 
         for key in ["log_final_loss", "final_margin_lambda_raw", "min_alpha", "min_update_cosine", "max_spectral_radius_raw", "max_curvature_proxy"]}
plot_phase(etas, gammas, grids=grids, cmaps={"log_final_loss": "viridis", "final_margin_lambda_raw": "cividis", "min_alpha": "bwr", "min_update_cosine": "plasma", "max_grad_sat": "inferno", "max_spectral_radius_raw": "Spectral", "max_curvature_proxy": "magma_r"})

# Compute the area of the divergence region (where log_final_loss is NaN)
divergence_mask = np.isnan(grids["log_final_loss"])
divergence_area = np.sum(divergence_mask) * (np.diff(gammas).mean() * np.diff(etas).mean())
total_area = (gammas.max() - gammas.min()) * (etas.max() - etas.min())
divergence_fraction = divergence_area / total_area
print(f"Estimated area of divergence region fraction: {divergence_fraction:.4%}")
```

---

## Results
The final results are shown below. The results mainly consist of a set of plots for different metrics (final loss, stability margin, curvature proxy, etc) in the $(\eta, \gamma)$ space, for different controllers. The main metric of interest is the final loss (or log final loss), which we use to determine whether the system diverged or not. The different tests are:
- No controller used:  <ControllerBadge controller="none" />
- Global throttle of order 0: <ControllerBadge controller="gt-order-0" />
- Global throttle of order 1: <ControllerBadge controller="gt-order-1" />

### No controller
In this case no controller was used, which means that the result is just the plain SGD algorithm without any stability control. As mentioned at the top, we would expect a large divergence region for any condition over $\eta \gtrsim \frac{\chi}{\widehat{C}}\frac{1}{\gamma^2}$. <FigureRef target="fig-phase-diagram-controller-none"> Figure 3</FigureRef> shows the phase diagram for three metrics: `log of the final loss value`, `minimum update cosine similarity`, and `minimum alpha value`. For each plot, the dashed line represents precisely this theoretical boundary of divergence (note plots are in log-log scale, so the boundary is a straight line with slope -2).

The most outstanding elements shown in this plot are:

  - The divergence area (estimated to be around $28.5\%$ of the total shown area) is quite large and it is almost exactly located to the NE of the theoretical boundary, which is consistent. The only reason why the boundary is not sharper is hypothesized to be that the simulation did not run for long enough.



<Figure 
  id="fig-phase-diagram-controller-none"
  src="https://github.com/manuelblancovalentin/kappa_budgeting/blob/2b52c4b9f9bc80c9769877f5623ba1e5b0973656/workspace/ablations/exp_001_lin1_stability_phase_diagram/results/controller_none_result.png?raw=true" 
  alt="Phase diagram for the no controller"
  maxWidth="100%"
  label="Figure 3"
  caption="Phase diagram for the case where no global throttle controller is used. The divergence region (where log_final_loss is NaN) is quite large, especially for higher values of gamma and eta, which is consistent with our theoretical analysis." 
/>


### Global throttle order 0

In this case, we activate the simplest throttle controller, which is of order $0$. This means that at any time $t$, the controller tries to apply a deterministic $\alpha_t$ value computed according to:

```math
\alpha_t = \min\left(1, \frac{\chi}{\widehat{C}_t + \epsilon}\right)
```

where $\epsilon$ is just a small value to avoid division by zero. The expectation is that this control mechanism will reduce the divergence area. The results are shown in <FigureRef target="fig-phase-diagram-controller-gt-order-0"> Figure 4</FigureRef>. The most outstanding elements are:

  - The divergence area is significantly reduced (estimated to be around $12.2\%$ of the total shown area), which is consistent with our expectations and shows that the global throttle controller is effective in improving stability.
  - <font color="red">Note that in reality, the divergence area shown in this plot does not necessarily correlate with true network divergence</font>, but rather with the fact that the final loss is above the `BLOWUP_LOSS` threshold. What this means is that, as long as those regions have an equivalent $\alpha$ value of $0$, the network will remain stable, since the updates will be effectively stopped.

<Figure 
  id="fig-phase-diagram-controller-gt-order-0"
  src="https://github.com/manuelblancovalentin/kappa_budgeting/blob/c99abc221a6b804eefcedd70b2bf6421280bd826/workspace/ablations/exp_001_lin1_stability_phase_diagram/results/controller_gt0_result.png?raw=true" 
  alt="Phase diagram for the global throttle order 0"
  maxWidth="100%"
  label="Figure 4"
  caption="Phase diagram for the case where the global throttle controller of order 0 is used. The divergence region (where log_final_loss is NaN) is significantly reduced compared to the no controller case." 
/>

### Global throttle order 1

In this final case, we activate the global throttle controller of order 1. This means that at any time $t$, the $\alpha$ coefficient *reacts* to the curvature via the following dynamics:

```math
\begin{aligned}
\dot{\alpha}_t &= k_\alpha\left( \chi - \eta\alpha_t\widehat{C}_t \right) \\
\alpha_t &\leftarrow \min(1, \alpha_t)
\end{aligned}
```

Because the initial alpha is $\alpha(0)=0$, this means that this system is inherently much more stable than the previous one, since it starts with a very conservative throttle and only after some iterations of stable curvature it starts to increase the throttle and allow for more aggressive updates. The results are shown in <FigureRef target="fig-phase-diagram-controller-gt-order-1"> Figure 5</FigureRef>. The most outstanding elements are:

  - The divergence area is further reduced (estimated to be around $1.4\%$ of the total shown area), which is consistent with our expectations and shows that the global throttle controller of order 1 is even more effective in improving stability compared to the order 0 controller.
  - Again, we want to highlight that the divergence area shown in this plot does not necessarily correlate with true network divergence, but rather with the fact that the final loss is above the `BLOWUP_LOSS` threshold. As long as those regions have an equivalent $\alpha$ value of $0$, the network will remain stable, since the updates will be effectively stopped.

<Figure 
  id="fig-phase-diagram-controller-gt-order-1"
  src="https://github.com/manuelblancovalentin/kappa_budgeting/blob/c99abc221a6b804eefcedd70b2bf6421280bd826/workspace/ablations/exp_001_lin1_stability_phase_diagram/results/controller_gt1_result.png?raw=true" 
  alt="Phase diagram for the global throttle order 1"
  maxWidth="100%"
  label="Figure 5"
  caption="Phase diagram for the case where the global throttle controller of order 1 is used. The divergence region (where log_final_loss is NaN) is significantly reduced compared to the no controller case." 
/>

--- 

## Conclusions



<TBox type="success" title="Summary">
  The results of this experiment show a clear improvement in stability when using the global throttle controllers, especially the order 1 controller. The divergence area is significantly reduced compared to the no controller case, which is consistent with our theoretical analysis. 


  <div style={{"align": "center", "width": "100%"}}>
    <table style={{"marginLeft": "auto", "marginRight": "auto", "border-color": "#000000", "border-width": "1px", "border-style": "solid", "width": "fit-content"}}>
      <thead>
        <tr>
          <th style={{"background-color": "#e3dede"}}>Controller</th>
          <th style={{"background-color": "#e3dede"}}>Divergence area fraction</th>
        </tr>
      </thead>
      <tbody>
        <tr style={{"textAlign": "center"}}>
          <td style={{"background-color": "#faf8f8"}}><ControllerBadge controller="none" /></td>
          <td style={{"background-color": "#faf8f8"}}>28.5%</td>
        </tr>
        <tr style={{"textAlign": "center"}}>
          <td style={{"background-color": "#faf8f8"}}><ControllerBadge controller="gt-order-0" /></td>
          <td style={{"background-color": "#faf8f8"}}>12.2%</td>
        </tr>
        <tr style={{"textAlign": "center"}}>
          <td style={{"background-color": "#faf8f8"}}><ControllerBadge controller="gt-order-1" /></td>
          <td style={{"background-color": "#faf8f8"}}>1.4%</td>
        </tr>
      </tbody>
    </table>
  </div>

  **The main conclusion is that the global throttle controllers are effective in improving stability and guaranteeing convergence in a much larger region of the hyperparameter space, at least for the ablation test of drift and learning rate.**

</TBox>
