---
status:
  - preliminary
tags:
  - onboarding
last_modified: 2026-05-15
author: mbvalentin
---
# ⚙️ Setup
<PageMeta />
---

This page describes the baseline setup for working on <ENABOL /> locally, including
notebooks under `workspace/`.

## Requirements

- Python 3.11 or newer.
- `pip` from the same Python environment you plan to use.
- (<b>Optional</b>) Node.js 20 or newer, only if you want to run the documentation site.
- (<b>Optional</b>) `nvm`, recommended for matching the documentation site's Node version.

## Clone the repository

```bash
git clone https://github.com/manuelblancovalentin/ENABOL
cd ENABOL
```

## Python Environment Setup

Create and activate a Python environment for the project:

> #### Using [Conda](https://docs.conda.io/en/latest/) or [Mamba](https://mamba.readthedocs.io/en/latest/) (Recommended):
>```bash
>conda create -n enabol python=3.11
>conda activate enabol
>```

> #### Using [venv](https://docs.python.org/3/library/venv.html):
>```bash
>python -m venv enabol-env
>source enabol-env/bin/activate  # On Windows: enabol-env\Scripts\activate
>```


Install the development requirements from the repository root:

```bash
python -m pip install -r requirements-dev.txt
```

The current `requirements-dev.txt` includes:

```text
-e .
jupyter
ipykernel
pytest
```

The `-e .` entry installs the local `enabol` package in editable mode. This
means notebooks, scripts, and tests can import <ENABOL /> without manually editing
`sys.path`:

```python
import enabol
```

Editable mode also means local changes under `enabol/` are picked up without
reinstalling the package.

## Register the Notebook Kernel

If you use Jupyter notebooks, register the environment as a kernel:

```bash
python -m ipykernel install --user --name enabol --display-name "Python (enabol)"
```

Then select `Python (enabol)` in notebooks under `workspace/`.

## Verify the Setup
From the repository root:

```bash
python -c "import enabol; print(enabol.__file__)"
python -m pytest
```

Which should print a message like:
<Terminal title="import check">
[INFO] - ENABOL imported successfully! Version: 0.1.0, URL: https://manuelblancovalentin.github.io/ENABOL/
</Terminal>

## Build the documentation site

<TBox type="warning" title="Documentation is optional for test-only users">
If you are only supposed to run experiments or tests, you probably do not need to build the documentation site. This section is mainly for active documentation/site developers.
</TBox>

The documentation site lives under `site/`, so run Node commands from that
directory:

```bash
cd site
nvm use
npm ci
npm run start
```

## Common setup problems

| Problem | Likely cause | Solution |
|---|---|---|
| `ModuleNotFoundError: No module named 'enabol'` | The active environment does not have the local package installed. | Activate the project environment and rerun `python -m pip install -r requirements-dev.txt` from the repository root. |
| `pip install -e .` reports multiple top-level packages | Setuptools is trying to auto-discover packages in the flat repository layout. | Keep the package discovery rule in `pyproject.toml` restricted to `enabol*`. |
| Notebook imports work in one notebook but not another | The notebook is using a different kernel/environment. | Select the `Python (enabol)` kernel or register it with `python -m ipykernel install --user --name enabol --display-name "Python (enabol)"`. |
| `nvm use` fails in `site/` | Node Version Manager is not installed or not loaded in the shell. | Install/load `nvm`, or manually use Node.js 20 or newer. |
