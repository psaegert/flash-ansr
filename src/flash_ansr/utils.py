import copy
import os
from typing import Any, Generator, Callable, Literal, Iterator, Mapping

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

        def resolve_path(value: Any) -> str:
            if isinstance(value, str) and value.endswith('.yaml') and value.startswith('.'):
                return os.path.join(config_base_path, value)
            return value

        if resolve_paths:
            config_ = apply_on_nested(config_, resolve_path)

    else:
        config_ = config

    return config_


def apply_on_nested(structure: list | dict, func: Callable) -> list | dict:
    '''
    Apply a function to all values in a nested dictionary.

    Parameters
    ----------
    d : list or dict
        The dictionary to apply the function to.
    func : Callable
        The function to apply to the dictionary values.

    Returns
    -------
    dict
        The dictionary with the function applied to all values.
    '''
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
    # '''
    config_ = copy.deepcopy(config)

    def save_config_relative_func(value: Any) -> Any:
        if isinstance(value, str) and value.endswith('.yaml'):
            relative_path = value
            if not value.startswith('.'):
                relative_path = os.path.join('.', os.path.basename(value))
            save_config(load_config(value, resolve_paths=resolve_paths), directory, os.path.basename(relative_path), reference=reference, recursive=recursive, resolve_paths=resolve_paths)
        return value

    def save_config_project_func(value: Any) -> Any:
        if isinstance(value, str) and value.endswith('.yaml'):
            relative_path = value
            if not value.startswith('.'):
                relative_path = value.replace(get_path(), '{{ROOT}}')
            save_config(load_config(value, resolve_paths=resolve_paths), directory, os.path.basename(relative_path), reference=reference, recursive=recursive, resolve_paths=resolve_paths)
        return value

    def save_config_absolute_func(value: Any) -> Any:
        if isinstance(value, str) and value.endswith('.yaml'):
            relative_path = value
            if not value.startswith('.'):
                relative_path = os.path.abspath(substitute_root_path(value))
            save_config(load_config(value, resolve_paths=resolve_paths), directory, os.path.basename(relative_path), reference=reference, recursive=recursive, resolve_paths=resolve_paths)
        return value

    if recursive:
        match reference:
            case 'relative':
                apply_on_nested(config_, save_config_relative_func)
            case 'project':
                apply_on_nested(config_, save_config_project_func)
            case 'absolute':
                apply_on_nested(config_, save_config_absolute_func)
            case _:
                raise ValueError(f'Invalid reference type: {reference}')

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


class GenerationConfig(Mapping[str, Any]):
    '''
    A class to store generation configuration.

    Parameters
    ----------
    method : str, optional, one of 'beam_search' or 'softmax_sampling'
        The generation method to use, by default 'beam_search'.
    **kwargs : Any
        Additional configuration parameters.

    Attributes
    ----------
    method : str
        The generation method to use.
    config : dict
        The configuration dictionary.
    '''
    def __init__(self, method: Literal['beam_search', 'softmax_sampling'] = 'beam_search', **kwargs: Any) -> None:
        self.defaults = {
            'beam_search': {
                'beam_width': 32,
                'max_len': 32,
                'mini_batch_size': 128,
                'equivalence_pruning': True
            },
            'softmax_sampling': {
                'choices': 32,
                'top_k': 0,
                'top_p': 1,
                'max_len': 32,
                'mini_batch_size': 128,
                'temperature': 1,
                'valid_only': True,
                'simplify': True,
                'unique': True
            }
        }

        if method not in self.defaults:
            raise ValueError(f'Invalid generation method: {method}')

        self.method = method

        self.config = dict(**kwargs)

        # Set defaults if not provided
        if method in self.defaults:
            for key, value in self.defaults[method].items():
                if key not in self.config:
                    self.config[key] = value

        for key, value in self.config.items():
            setattr(self, key, value)

    def __getitem__(self, key: str) -> Any:
        return self.config[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.config[key] = value
        setattr(self, key, value)

    def __delitem__(self, key: str) -> None:
        del self.config[key]
        delattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(self.config)

    def __len__(self) -> int:
        return len(self.config)

    # When printed, show the config as a dictionary
    def __repr__(self) -> str:
        return str(self.config)

    def __str__(self) -> str:
        return str(self.config)
