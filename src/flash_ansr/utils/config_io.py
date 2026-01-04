"""Configuration file helpers."""
import copy
import os
from typing import Any, Callable, Generator, cast

import yaml

from flash_ansr.utils.paths import get_path, normalize_path_preserve_leading_dot, substitute_root_path


def apply_on_nested(structure: list | dict, func: Callable[[Any], Any]) -> list | dict:
    """Apply ``func`` recursively across nested ``structure`` values."""
    if isinstance(structure, list):
        for i, value in enumerate(structure):
            if isinstance(value, dict):
                structure[i] = apply_on_nested(value, func)
            else:
                structure[i] = func(value)
        return structure

    if isinstance(structure, dict):
        for key, value in structure.items():
            if isinstance(value, dict):
                structure[key] = apply_on_nested(value, func)
            else:
                structure[key] = func(value)
        return structure

    return structure


def load_config(config: dict[str, Any] | str, resolve_paths: bool = True) -> dict[str, Any]:
    """Load a YAML config (optionally resolving nested relative paths)."""
    if isinstance(config, str):
        config_path = substitute_root_path(config)
        config_base_path = os.path.dirname(config_path)

        if not os.path.exists(config_path):
            raise FileNotFoundError(f'Config file {config_path} not found.')
        if os.path.isfile(config_path):
            with open(config_path, 'r') as config_file:
                config_ = yaml.safe_load(config_file)
        else:
            raise ValueError(f'Config file {config_path} is not a valid file.')

        def resolve_path(value: Any) -> Any:
            if (
                isinstance(value, str)
                and (value.endswith('.yaml') or value.endswith('.json'))
                and value.startswith('.')
            ):
                return normalize_path_preserve_leading_dot(os.path.join(config_base_path, value))
            return value

        if resolve_paths:
            config_ = apply_on_nested(config_, resolve_path)
    else:
        config_ = config

    return config_


def unfold_config(config: dict[str, Any], max_depth: int = 3) -> dict[str, Any]:
    """Recursively load nested config references up to ``max_depth``."""

    def try_load_config(value: Any) -> Any:
        if isinstance(value, str) and value.endswith('.yaml'):
            return load_config(get_path(value))
        return value

    for _ in range(max_depth):
        config = cast(dict[str, Any], apply_on_nested(config, try_load_config))
    return config


def save_config(
    config: dict[str, Any],
    directory: str,
    filename: str,
    reference: str = 'relative',
    recursive: bool = True,
    resolve_paths: bool = False,
) -> None:
    """Persist ``config`` to ``directory``/``filename`` handling nested includes."""
    config_ = copy.deepcopy(config)

    def save_config_relative_func(value: Any) -> Any:
        relative_path = value
        if isinstance(value, str) and value.endswith('.yaml'):
            if not value.startswith('.'):
                relative_path = normalize_path_preserve_leading_dot(os.path.join('.', os.path.basename(value)))
            save_config(
                load_config(value, resolve_paths=resolve_paths),
                directory,
                os.path.basename(relative_path),
                reference=reference,
                recursive=recursive,
                resolve_paths=resolve_paths,
            )
        return relative_path

    def save_config_project_func(value: Any) -> Any:
        relative_path = value
        if isinstance(value, str) and value.endswith('.yaml'):
            if not value.startswith('.'):
                relative_path = normalize_path_preserve_leading_dot(value.replace(get_path(), '{{ROOT}}'))
            save_config(
                load_config(value, resolve_paths=resolve_paths),
                directory,
                os.path.basename(relative_path),
                reference=reference,
                recursive=recursive,
                resolve_paths=resolve_paths,
            )
        return relative_path

    def save_config_absolute_func(value: Any) -> Any:
        relative_path = value
        if isinstance(value, str) and value.endswith('.yaml'):
            if not value.startswith('.'):
                relative_path = normalize_path_preserve_leading_dot(os.path.abspath(substitute_root_path(value)))
            save_config(
                load_config(value, resolve_paths=resolve_paths),
                directory,
                os.path.basename(relative_path),
                reference=reference,
                recursive=recursive,
                resolve_paths=resolve_paths,
            )
        return relative_path

    if recursive:
        match reference:
            case 'relative':
                config_with_corrected_paths = apply_on_nested(config_, save_config_relative_func)
            case 'project':
                config_with_corrected_paths = apply_on_nested(config_, save_config_project_func)
            case 'absolute':
                config_with_corrected_paths = apply_on_nested(config_, save_config_absolute_func)
            case _:
                raise ValueError(f'Invalid reference type: {reference}')
    else:
        config_with_corrected_paths = config_

    with open(get_path(directory, filename=filename, create=True), 'w') as config_file:
        yaml.dump(config_with_corrected_paths, config_file, sort_keys=False)


def traverse_dict(dict_: dict[str, Any]) -> Generator[tuple[str, Any], None, None]:
    """Yield ``(key, value)`` pairs for leaves in a nested dictionary."""
    for key, value in dict_.items():
        if isinstance(value, dict):
            yield from traverse_dict(value)
        else:
            yield key, value
