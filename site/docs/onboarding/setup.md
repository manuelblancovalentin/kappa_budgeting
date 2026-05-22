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
-e ./hls4ml-trainable
jupyter
ipykernel
pytest
```

The `-e .` entry installs the local `enabol` package in editable mode, while the `-e ./hls4ml-trainable` entry installs the local `hls4ml` package in editable mode. This means notebooks, scripts, and tests can import <ENABOL /> without manually editing
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




### Setting up hls4ml trainable
1. Fork the [hls4ml repository](https://github.com/fastmachinelearning/hls4ml) directly on your Github.
2. Clone that forked repo to your local machine (inside the ENABOL folder!) -- it might take a bit.
> ```bash
> git clone git@github.com:<YOUR_USERNAME>/hls4ml.git hls4ml-trainable
> ```
3. Move onto that directory and add the official repo as upstream.
> ```bash
> cd hls4ml-trainable
> git remote add upstream https://github.com/fastmachinelearning/hls4ml.git
> git remote -v
> ```

You should see something like:
<Terminal title="git remote -v" >
  origin    git@github.com:YOUR_USERNAME/hls4ml.git (fetch)
origin    git@github.com:YOUR_USERNAME/hls4ml.git (push)
upstream  https://github.com/fastmachinelearning/hls4ml.git (fetch)
upstream  https://github.com/fastmachinelearning/hls4ml.git (push)
</Terminal>

4. Now fetch the latest upstream changes:
> ```bash
> git fetch upstream
> ```

5. Create your development branch from official main:
> ```bash
> git checkout -b hls4ml-trainable upstream/main
> ```

6. Push that branch to your fork:
> ```bash
> git push -u origin hls4ml-trainable
> ```

7. Now you can finally install your local hls4ml-trainable branch in editable mode like (<font color="red">MAKE SURE YOU HAVE THE RIGHT ENVIRONMENT ACTIVATED!</font>):
> ```bash
> python -m pip install -e ./hls4ml-trainable
> ```

8. Finally verify that both imports work:
> ```bash
> python - <<'PY'
> import enabol
> import hls4ml
> print("enabol:", enabol.__file__)
> print("hls4ml:", hls4ml.__file__)
> PY
> ```

> You should see something like:
<Terminal title="import check" >
[INFO] - ENABOL imported successfully! Version: 0.1.0, URL: https://manuelblancovalentin.github.io/ENABOL/
enabol: /Users/mbvalentin/scripts/ENABOL/enabol/__init__.py
hls4ml: /Users/mbvalentin/scripts/ENABOL/hls4ml-trainable/hls4ml/__init__.py
</Terminal>


## Server Toolchain Check

On `kona-ubuntu`, ENABOL uses toolchain profiles so notebooks do not have to source Vivado/Vitis shell files manually. The default profile is:

```text
kona-vitis-2024_1
```

Verify the profile machinery from the repository root:

```bash
python - <<'PY'
from enabol.toolchain import list_toolchain_profiles, toolchain_environment

print(list_toolchain_profiles())
with toolchain_environment("auto", backend="Vitis") as profile:
    print(profile)
PY
```

The 2024.1 profile should be the default because CSIM with 2023.2 is not reliable on Ubuntu 24.04 without a container. Before running the first hls4ml build on `kona-ubuntu`, also check:

```bash
which vitis-run
which vitis_hls
vitis_hls -version
```

The older compatibility profile `kona-vivado-2023_2` is still documented because `/usr/local/bin/vivado_hls` exists on `kona-ubuntu`, but it should not be the default compile path.




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
