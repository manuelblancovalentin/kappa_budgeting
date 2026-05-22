---
title: "Toolchain Profiles"
sidebar_label: "Toolchain"
status:
  - inprogress
tags:
  - implementation
  - hls4ml
last_modified: 2026-05-22
author: mbvalentin
---
# Toolchain Profiles
<PageMeta />
---

<TBox type="summary" title="Purpose">

ENABOL needs to call hls4ml on machines with multiple Vivado/Vitis installs. hls4ml expects tools such as `vivado_hls` or `vitis-run` to already be discoverable on `PATH`, so ENABOL owns a small profile layer that applies machine-specific environment changes before compilation.

</TBox>

## Decision

Toolchain setup belongs inside `enabol.compile`, not inside notebooks. A notebook should request a profile name, and the compile path should apply it with a scoped context manager:

```python
with toolchain_environment(toolchain, backend=backend):
    # hls4ml conversion, compile, and build happen here
```

The user should not have to source shell files manually for ordinary ENABOL compilation runs.

## Files

| File | Role |
|---|---|
| `enabol/toolchain.py` | Loads profiles, validates host/platform/backend/commands, applies env edits, and restores the previous environment. |
| `enabol/config/toolchains.example.toml` | Tracked example profile file. It documents the lab server profiles but should not be the only config source. |
| `enabol/config/toolchains.local.toml` | Optional repo-local machine config. This should stay untracked. |
| `~/.config/enabol/toolchains.toml` | Preferred user/server config location. |

`ENABOL_TOOLCHAIN_CONFIG` can point to any other config path.

## Why TOML

The old ENABOL bridge used YAML. The new implementation uses TOML because Python has a standard TOML reader on Python 3.11+ and a tiny `tomli` fallback for older Python versions. This avoids adding a YAML parser dependency just for machine-local environment profiles.

The structure is still the same idea as the old config: named profiles with explicit environment changes.

## Profile Shape

```toml
default_profile = "kona-vitis-2024_1"

[profiles.kona-vitis-2024_1]
backend = "Vitis"
platforms = ["linux"]
hosts = ["kona-ubuntu"]
required_commands = ["vitis-run"]

[profiles.kona-vitis-2024_1.env.PATH]
prepend = [
  "/usr/local/bin",
  "/usr/bin",
  "/mnt/raid5/fpga/cad/xilinx/Vitis_HLS/2024.1/bin",
  "/mnt/raid5/fpga/cad/xilinx/Vitis/2024.1/bin",
  "/mnt/raid5/fpga/cad/xilinx/Vivado/2024.1/bin",
]

[profiles.kona-vitis-2024_1.env.LD_LIBRARY_PATH]
prepend = [
  "/usr/lib/x86_64-linux-gnu",
  "/usr/lib32",
  "/mnt/raid5/fpga/cad/xilinx/Vitis/2024.1/lib/lnx64.o",
  "/mnt/raid5/fpga/cad/xilinx/Vivado/2024.1/lib/lnx64.o",
]

[profiles.kona-vitis-2024_1.env.CXXFLAGS]
set = "-D_GLIBCXX_USE_CXX11_ABI=0"
```

Each environment variable supports:

| Operation | Meaning |
|---|---|
| `set` | Replace the variable while the profile is active. |
| `prepend` | Add values before the existing variable. |
| `append` | Add values after the existing variable. |

Values are de-duplicated and the previous environment is restored after the context exits.

## Validation

`toolchain_environment()` validates:

- profile exists
- current hostname matches `hosts`, unless `*` is allowed
- current platform matches `platforms`
- requested hls4ml backend matches the profile backend
- configured path entries exist
- required commands are discoverable after applying the profile

This is intentionally stricter than a notebook cell that mutates `PATH`: it should fail early with a precise error if the server layout changes.

## Default Profile

`toolchain_environment("auto", backend=...)` resolves the profile in this order:

1. `ENABOL_TOOLCHAIN_PROFILE`, if set.
2. `default_profile` from the toolchain config.

The example config currently sets:

```toml
default_profile = "kona-vitis-2024_1"
```

This reflects the lab decision to move to the latest hls4ml-supported 2024.1 toolchain because CSIM in 2023.2 is not reliable on Ubuntu 24.04 without a container.

## Current Server Profiles

The first example profiles target:

| Profile | Host | Backend | Tool version |
|---|---|---|---|
| `kona-vivado-2023_2` | `kona-ubuntu` | `Vivado` | `/usr/local/bin/vivado_hls`, reporting Vitis HLS 2023.2 |
| `kona-vitis-2024_1` | `kona-ubuntu` | `Vitis` | `/mnt/raid5/fpga/cad/xilinx/Vitis_HLS/2024.1/bin/vitis_hls` plus Vitis/Vivado 2024.1 paths |

The Vivado profile currently requires `vivado_hls`, because that is what hls4ml's Vivado backend calls. On `kona-ubuntu`, `/usr/local/bin/vivado_hls` exists and reports Vitis HLS 2023.2. This profile is retained for compatibility and for debugging older generated projects, but it should not be the default path on Ubuntu 24.04.

The default profile is now `kona-vitis-2024_1`. The Vitis HLS 2024.1 install exposes `vitis_hls`, while the current hls4ml Vitis backend checks for `vitis-run`. Before running the first server build, verify `which vitis-run` on `kona-ubuntu`. If `vitis-run` is unavailable, we should either add the path that provides it or decide explicitly whether the bridge should call the Vivado backend with a 2024.1-compatible HLS command.

## Linked Task

- [ENB-022](/docs/status/tasks?query=ENB-022): add scoped ENABOL toolchain profiles for hls4ml compilation.
