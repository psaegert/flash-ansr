import copy
import os
from typing import Any, Generator

import yaml


def get_path(*args: str, filename: str | None = None, create: bool = False) -> str:
    '''
    Get the path to a file or directory.

    Parameters
    ----------
    args : str
        The path to the file or directory, starting from the root of the project.
    filename : str, optional
        The filename to append to the path, by default None.
    create : bool, optional
        Whether to create the directory if it does not exist, by default False.

    Returns
    -------
    str
        The path to the file or directory.
    '''
    if any(not isinstance(arg, str) for arg in args):
        raise TypeError("All arguments must be strings.")

    path = os.path.join(os.path.dirname(__file__), '..', '..', *args, filename or '')

    if create:
        if filename is not None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        else:
            os.makedirs(path, exist_ok=True)

    return os.path.abspath(path)


def substitute_root_path(path: str) -> str:
    '''
    Replace {{ROOT}} with the root path of the project given by get_path().

    Parameters
    ----------
    path : str
        The path to replace

    Returns
    -------
    new_path : str
        The new path with the root path replaced
    '''
    return path.replace(r"{{ROOT}}", get_path())


def load_config(config: dict[str, Any] | str, resolve_paths: bool = True) -> dict[str, Any]:
    '''
    Load a configuration file.

    Parameters
    ----------
    config : dict or str
        The configuration dictionary or path to the configuration file.
    resolve_paths : bool, optional
        Whether to resolve relative paths in the configuration file, by default True.

    Returns
    -------
    dict
        The configuration dictionary.
    '''
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

        if resolve_paths:
            for key, value in traverse_dict(config_):
                if isinstance(value, str) and value.endswith('.yaml') and value.startswith('.'):
                    # Convert the relative path to an absolute path
                    config_[key] = os.path.join(config_base_path, value)
    else:
        config_ = config

    return config_


def save_config(config: dict[str, Any], directory: str, filename: str, reference: str = 'relative', recursive: bool = True, resolve_paths: bool = False) -> None:
    '''
    Save a configuration dictionary to a YAML file.

    Parameters
    ----------
    config : dict
        The configuration dictionary to save.
    directory : str
        The directory to save the configuration file to.
    filename : str
        The name of the configuration file.
    reference : str, optional
        Determines the reference base path. One of
        - 'relative': relative to the specified directory
        - 'project': relative to the project root
        - 'absolute': absolute paths
    recursive : bool, optional
        Save any referenced configs too
    '''
    config_ = copy.deepcopy(config)

    if recursive:
        if reference == 'relative':
            for key, value in traverse_dict(config_):
                if isinstance(value, str) and value.endswith('.yaml'):
                    # First, convert project paths to absolute paths
                    config_[key] = os.path.join('.', os.path.basename(config_[key]))
                    save_config(load_config(value, resolve_paths=resolve_paths), directory, os.path.basename(config_[key]), reference=reference, recursive=recursive, resolve_paths=resolve_paths)

        elif reference == 'project':
            for key, value in traverse_dict(config_):
                if isinstance(value, str) and value.endswith('.yaml'):
                    config_[key] = value.replace(get_path(), '{{ROOT}}')
                    save_config(load_config(config_[key], resolve_paths=resolve_paths), directory, os.path.basename(config_[key]), reference=reference, recursive=recursive, resolve_paths=resolve_paths)

        elif reference == 'absolute':
            for key, value in traverse_dict(config_):
                if isinstance(value, str) and value.endswith('.yaml'):
                    config_[key] = os.path.abspath(substitute_root_path(value))
                    save_config(load_config(config_[key], resolve_paths=resolve_paths), directory, os.path.basename(config_[key]), reference=reference, recursive=recursive, resolve_paths=resolve_paths)

    with open(get_path(directory, filename=filename, create=True), 'w') as config_file:
        yaml.dump(config_, config_file, sort_keys=False)


def traverse_dict(dict_: dict[str, Any]) -> Generator[tuple[str, Any], None, None]:
    '''
    Traverse a dictionary recursively.

    Parameters
    ----------
    d : dict
        The dictionary to traverse.

    Yields
    ------
    tuple
        A tuple containing the key and value of the current dictionary item.
    '''
    for key, value in dict_.items():
        if isinstance(value, dict):
            yield from traverse_dict(value)
        else:
            yield key, value
