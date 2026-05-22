"""Toolchain profile handling for hls4ml compilation.

The hls4ml Vivado/Vitis backends expect compiler commands such as
``vivado_hls`` or ``vitis-run`` to be available on ``PATH``. ENABOL keeps that
machine-specific setup in explicit profiles so notebooks can ask for a profile
name without sourcing shell startup files.
"""

from __future__ import annotations

import contextlib
import os
import platform
import shutil
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised only on Python < 3.11
    import tomli as tomllib # type: ignore[import]


DEFAULT_TOOLCHAIN_CONFIG_PATHS = (
    Path('enabol/config/toolchains.local.toml'),
    Path.home() / '.config/enabol/toolchains.toml',
    Path(__file__).resolve().parent / 'config/toolchains.example.toml',
)


class ToolchainError(RuntimeError):
    """Raised when a requested hardware toolchain profile cannot be applied."""


@dataclass(frozen=True)
class ToolchainProfile:
    """Resolved ENABOL toolchain profile."""

    name: str
    backend: str | None
    env: dict[str, dict[str, Any]]
    required_commands: tuple[str, ...]
    hosts: tuple[str, ...]
    platforms: tuple[str, ...]


def get_hostname() -> str:
    """Return the short hostname used for profile host guards."""

    return socket.gethostname().split('.')[0]


def get_platform_name() -> str:
    """Return the lowercase platform name used for profile platform guards."""

    return sys.platform.lower()


def find_toolchain_config(config_path: str | os.PathLike[str] | None = None) -> Path:
    """Find the toolchain config file.

    Search order:
      1. explicit ``config_path`` argument
      2. ``ENABOL_TOOLCHAIN_CONFIG``
      3. ``enabol/config/toolchains.local.toml`` in the current repo
      4. ``~/.config/enabol/toolchains.toml``
      5. packaged ``toolchains.example.toml``
    """

    candidates: list[Path] = []
    if config_path is not None:
        candidates.append(Path(config_path).expanduser())

    env_path = os.environ.get('ENABOL_TOOLCHAIN_CONFIG')
    if env_path:
        candidates.append(Path(env_path).expanduser())

    candidates.extend(DEFAULT_TOOLCHAIN_CONFIG_PATHS)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise ToolchainError('No ENABOL toolchain config found.')


def load_toolchain_config(config_path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Load the TOML toolchain config."""

    resolved_path = find_toolchain_config(config_path)
    with resolved_path.open('rb') as config_file:
        return tomllib.load(config_file)


def list_toolchain_profiles(config_path: str | os.PathLike[str] | None = None) -> tuple[str, ...]:
    """List available toolchain profile names."""

    config = load_toolchain_config(config_path)
    return tuple(sorted(config.get('profiles', {}).keys()))


def get_toolchain_profile(
    profile: str,
    config_path: str | os.PathLike[str] | None = None,
) -> ToolchainProfile:
    """Return a normalized profile from the toolchain config."""

    config = load_toolchain_config(config_path)
    profiles = config.get('profiles', {})
    if profile not in profiles:
        available = ', '.join(sorted(profiles)) or '<none>'
        raise ToolchainError(f'Unknown ENABOL toolchain profile "{profile}". Available profiles: {available}.')

    profile_config = profiles[profile]
    return ToolchainProfile(
        name=profile,
        backend=profile_config.get('backend'),
        env=profile_config.get('env', {}),
        required_commands=tuple(profile_config.get('required_commands', ())),
        hosts=tuple(profile_config.get('hosts', ())),
        platforms=tuple(profile_config.get('platforms', ())),
    )


def _matches_platform(platform_pattern: str, current_platform: str) -> bool:
    if platform_pattern == current_platform:
        return True
    if platform_pattern == 'linux' and current_platform.startswith('linux'):
        return True
    if platform_pattern == 'darwin' and current_platform == 'darwin':
        return True
    return platform_pattern == '*'


def validate_toolchain_profile(
    profile: ToolchainProfile,
    backend: str | None = None,
    require_commands: bool = True,
) -> None:
    """Validate host, platform, backend, paths, and required commands."""

    hostname = get_hostname()
    if profile.hosts and hostname not in profile.hosts and '*' not in profile.hosts:
        raise ToolchainError(
            f'Toolchain profile "{profile.name}" is restricted to hosts {profile.hosts}, '
            f'but current host is "{hostname}".'
        )

    current_platform = get_platform_name()
    if profile.platforms and not any(_matches_platform(p, current_platform) for p in profile.platforms):
        raise ToolchainError(
            f'Toolchain profile "{profile.name}" is restricted to platforms {profile.platforms}, '
            f'but current platform is "{current_platform}".'
        )

    if backend is not None and profile.backend is not None and profile.backend.lower() != backend.lower():
        raise ToolchainError(
            f'Toolchain profile "{profile.name}" targets backend "{profile.backend}", '
            f'but compile requested "{backend}".'
        )

    for env_ops in profile.env.values():
        for op_name in ('prepend', 'append'):
            for path_value in env_ops.get(op_name, ()):
                if not Path(path_value).exists():
                    raise ToolchainError(
                        f'Toolchain profile "{profile.name}" references missing path "{path_value}".'
                    )

    if require_commands:
        for command in profile.required_commands:
            if shutil.which(command) is None:
                raise ToolchainError(
                    f'Toolchain profile "{profile.name}" requires command "{command}", '
                    'but it was not found after applying the profile environment.'
                )


def apply_env_operations(env: dict[str, str], operations: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Return a copy of ``env`` with profile environment operations applied."""

    updated_env = env.copy()
    path_separator = os.pathsep

    for var_name, env_ops in operations.items():
        value = updated_env.get(var_name, '')

        if 'set' in env_ops:
            value = str(env_ops['set'])

        prepend_values = [str(v) for v in env_ops.get('prepend', ())]
        append_values = [str(v) for v in env_ops.get('append', ())]

        values: list[str] = []
        values.extend(prepend_values)
        if value:
            values.extend(value.split(path_separator))
        values.extend(append_values)

        if values:
            deduped_values = []
            for item in values:
                if item and item not in deduped_values:
                    deduped_values.append(item)
            updated_env[var_name] = path_separator.join(deduped_values)

    return updated_env


@contextlib.contextmanager
def toolchain_environment(
    profile: str | None,
    backend: str | None = None,
    config_path: str | os.PathLike[str] | None = None,
    require_commands: bool = True,
) -> Iterator[ToolchainProfile | None]:
    """Temporarily apply an ENABOL toolchain profile to ``os.environ``.

    Passing ``profile=None`` leaves the environment unchanged and yields
    ``None``. This lets ``enabol.compile`` make toolchain setup optional while
    still keeping the same control flow.
    """

    if profile is None:
        yield None
        return

    resolved_profile = get_toolchain_profile(profile, config_path)
    original_env = os.environ.copy()

    try:
        updated_env = apply_env_operations(original_env, resolved_profile.env)
        os.environ.clear()
        os.environ.update(updated_env)
        validate_toolchain_profile(resolved_profile, backend=backend, require_commands=require_commands)
        yield resolved_profile
    finally:
        os.environ.clear()
        os.environ.update(original_env)


def describe_toolchain_profile(profile: ToolchainProfile) -> str:
    """Return a short human-readable profile summary."""

    command_text = ', '.join(profile.required_commands) if profile.required_commands else '<none>'
    host_text = ', '.join(profile.hosts) if profile.hosts else '<any>'
    platform_text = ', '.join(profile.platforms) if profile.platforms else '<any>'
    return (
        f'{profile.name}: backend={profile.backend or "<any>"}, hosts={host_text}, '
        f'platforms={platform_text}, required_commands={command_text}'
    )
