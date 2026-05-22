import os
import sys

import pytest   # type: ignore[import]

from enabol.toolchain import (
    ToolchainError,
    apply_env_operations,
    get_toolchain_profile,
    list_toolchain_profiles,
    toolchain_environment,
)


def write_config(path, profile_name='local-test', command='python'):
    path.write_text(
        f"""
[profiles.{profile_name}]
backend = "Vitis"
platforms = ["{sys.platform}"]
hosts = ["*"]
required_commands = ["{command}"]

[profiles.{profile_name}.env.PATH]
prepend = ["{path.parent}"]

[profiles.{profile_name}.env.TEST_TOOLCHAIN_FLAG]
set = "enabled"
""".strip()
    )


def test_list_toolchain_profiles(tmp_path):
    config_path = tmp_path / 'toolchains.toml'
    write_config(config_path)

    assert list_toolchain_profiles(config_path) == ('local-test',)


def test_get_toolchain_profile(tmp_path):
    config_path = tmp_path / 'toolchains.toml'
    write_config(config_path)

    profile = get_toolchain_profile('local-test', config_path)

    assert profile.name == 'local-test'
    assert profile.backend == 'Vitis'
    assert profile.hosts == ('*',)
    assert profile.required_commands == ('python',)


def test_apply_env_operations_deduplicates_path_entries(tmp_path):
    env = {'PATH': os.pathsep.join([str(tmp_path), '/usr/bin'])}
    updated = apply_env_operations(env, {'PATH': {'prepend': [str(tmp_path), '/bin']}})

    assert updated['PATH'].split(os.pathsep) == [str(tmp_path), '/bin', '/usr/bin']


def test_toolchain_environment_applies_and_restores_env(tmp_path, monkeypatch):
    config_path = tmp_path / 'toolchains.toml'
    write_config(config_path)
    monkeypatch.setenv('TEST_TOOLCHAIN_FLAG', 'original')

    with toolchain_environment('local-test', backend='Vitis', config_path=config_path) as profile:
        assert profile.name == 'local-test' # type: ignore[union-attr]
        assert os.environ['TEST_TOOLCHAIN_FLAG'] == 'enabled'

    assert os.environ['TEST_TOOLCHAIN_FLAG'] == 'original'


def test_toolchain_environment_rejects_backend_mismatch(tmp_path):
    config_path = tmp_path / 'toolchains.toml'
    write_config(config_path)

    with pytest.raises(ToolchainError, match='targets backend'):
        with toolchain_environment('local-test', backend='Vivado', config_path=config_path):
            pass
