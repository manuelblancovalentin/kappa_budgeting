# Experiment Template

Copy this structure when documenting an ablation experiment.

## Experiment title

Date: YYYY-MM-DD  
Owner: NAME  
Status: Planned / Running / Done / Failed / Deprecated

## Goal

What question are we trying to answer?

## Hypothesis

What do we expect to happen?

## Setup

| Field | Value |
|---|---|
| Branch | `branch-name` |
| Commit | `commit-hash` |
| Dataset | Dataset |
| Architecture | Architecture |
| Precision | Precision |
| Drift | `(alpha, beta)` range |
| Budgeting variant | Variant |
| Optimizer | Optimizer |
| Environment | Environment |

## Model Math

Add the forward equations and loss.

```text
z1 = W1 x + b1
a1 = relu(z1)
y_hat = W2 a1 + b2
L = ...
```

## Diagram

Use Mermaid for the forward and backward paths.

```mermaid
flowchart LR
    X["X"] -->|x| L1["Layer 1<br/>W1, b1"]
    L1 -->|a1| LOSS["Loss"]
    LOSS -->|g1| B1["Backpass Layer 1"]
    B1 -->|update| L1
```

## Procedure

1. Step 1
2. Step 2
3. Step 3

## Results

Add tables, plots, screenshots, or links to raw logs.

| Metric | Baseline | Budgeted |
|---|---:|---:|
| Final loss | TBD | TBD |
| Saturation events | TBD | TBD |
| Max throttle shift | TBD | TBD |
| Mean update cosine | TBD | TBD |

## Interpretation

What do the results mean?

## Failure modes / caveats

- Caveat 1
- Caveat 2

## Next steps

- [ ] Next action
- [ ] Next action

## Related links

- Issue:
- Pull request:
- Lab log:
- Notebook:
- Data:
