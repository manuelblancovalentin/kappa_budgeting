# Code Map

This repository is the new implementation and documentation home for the kappa-budgeting ablations. Related legacy/generated code currently lives outside this repository and should be treated as reference material.

## Main Repository

| Path | Purpose |
|---|---|
| `kappa/` | Future Python package for reusable kappa-budgeting simulation code. |
| `workspace/ablations/` | Reproducible experiment workspaces, notebooks, configs, notes, and results. |
| `site/docs/` | Docusaurus documentation for methodology, experiment logs, decisions, and handoff. |
| `site/blog/` | Short lab-log entries for chronological updates. |
| `.github/workflows/deploy-docs.yml` | GitHub Pages deployment for the Docusaurus site. |

## Reference Code Outside This Repo

| Path | Purpose |
|---|---|
| `/Users/mbvalentin/enabol/budgeting.py` | Current rail-driven kappa budget estimator used for the paper workflow. |
| `/Users/mbvalentin/firmware/autograd/nnet_dense_backprop.h` | Generated dense backprop implementation with active column scaling and throttling, and disabled dense `RowScale`. |
| `/Users/mbvalentin/firmware/autograd/optimizers.h` | Throttled SGD/Adam update implementations. |
| `/Users/mbvalentin/firmware/parameters.h` | Generated layer configs, precision aliases, and `kappa_row_log2` / `kappa_col_log2` values. |
| `/Users/mbvalentin/firmware/defines.h` | Generated fixed-point type definitions and accumulator precision. |

## Workspace Convention

Each ablation gets its own folder:

```text
workspace/
  ablations/
    001_affine_single_dense/
      config.yaml
      notes.md
      notebooks/
        analysis.ipynb
      results/
        .gitkeep
    002_two_layer_relu/
      config.yaml
      notes.md
      notebooks/
        analysis.ipynb
      results/
        .gitkeep
```

The recommended workflow is notebook-first, script-second:

- Use `notebooks/analysis.ipynb` while exploring, because plots, tensor dumps, and commentary are part of the reasoning.
- Keep experiment settings in `config.yaml`, even when the run is launched from a notebook.
- Store exported plots, CSV logs, and summaries in `results/`.
- Add a `run.py` later only after a notebook workflow becomes stable enough to rerun automatically.

This keeps exploration easy while preserving enough structure for handoff and reproducibility.

## Planned Python Package Shape

The eventual `kappa/` package should separate concerns:

```text
kappa/
  fixed_point.py       # quantization, rails, saturation counters
  datasets.py          # synthetic affine and teacher-network datasets
  models.py            # small dense networks and forward/backward helpers
  budgeting.py         # kappa allocation and projection policies
  training.py          # custom online training loops
  logging.py           # tensor statistics and run artifacts
  metrics.py           # saturation, norms, update cosine, recovery metrics
```

Keep the first implementation simple. It is better to have one complete, inspectable ablation than a broad framework too early.
