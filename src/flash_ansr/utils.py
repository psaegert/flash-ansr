import copy
import os
from typing import Any, Generator, Callable, Literal, Iterator, Mapping

import yaml
import numpy as np
import torch
from torch import nn


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

    path = normalize_path_preserve_leading_dot(os.path.join(os.path.dirname(__file__), '..', '..', *args, filename or ''))

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
            if isinstance(value, str) and (value.endswith('.yaml') or value.endswith('.json')) and value.startswith('.'):  # HACK: Find a way to check if a string is a path
                return normalize_path_preserve_leading_dot(os.path.join(config_base_path, value))
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


def unfold_config(config: dict[str, Any], max_depth: int = 3) -> dict[str, Any]:
    '''
    Recursively load configuration files referenced in a configuration dictionary.

    Parameters
    ----------
    config : dict
        The configuration dictionary to unfold.
    max_depth : int, optional
        The maximum depth to unfold, by default 3.

    Returns
    -------
    dict
        The unfolded configuration dictionary.
    '''
    def try_load_config(x: Any) -> Any:
        if isinstance(x, str) and x.endswith(".yaml"):
            return load_config(get_path(x))
        return x

    for _ in range(max_depth):
        config = apply_on_nested(config, try_load_config)  # type: ignore
    return config


def normalize_path_preserve_leading_dot(path: str) -> str:
    """
    Normalizes a path to remove redundant parts like '..' and '.',
    while preserving a leading './' if it was in the original path.

    Parameters
    ----------
    path : str
        The path to normalize.

    Returns
    -------
    str
        The normalized path with leading './' preserved if applicable.
    """
    # Check if the original path started with './' (or '.\' on Windows)
    starts_with_dot_sep = path.startswith(f'.{os.sep}')

    # Normalize the path to resolve '..' and '.'
    normalized_path = os.path.normpath(path)

    # If the original path had a leading './' and the normalized result
    # is a simple relative path (i.e., not '.', '..', or absolute),
    # then prepend the './' back.
    if (starts_with_dot_sep and not os.path.isabs(normalized_path) and not normalized_path.startswith('..') and normalized_path != '.'):
        return f'.{os.sep}{normalized_path}'

    return normalized_path


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
        relative_path = value
        if isinstance(value, str) and value.endswith('.yaml'):
            if not value.startswith('.'):
                relative_path = normalize_path_preserve_leading_dot(os.path.join('.', os.path.basename(value)))
            save_config(load_config(value, resolve_paths=resolve_paths), directory, os.path.basename(relative_path), reference=reference, recursive=recursive, resolve_paths=resolve_paths)
        return relative_path

    def save_config_project_func(value: Any) -> Any:
        relative_path = value
        if isinstance(value, str) and value.endswith('.yaml'):
            if not value.startswith('.'):
                relative_path = normalize_path_preserve_leading_dot(value.replace(get_path(), '{{ROOT}}'))
            save_config(load_config(value, resolve_paths=resolve_paths), directory, os.path.basename(relative_path), reference=reference, recursive=recursive, resolve_paths=resolve_paths)
        return relative_path

    def save_config_absolute_func(value: Any) -> Any:
        relative_path = value
        if isinstance(value, str) and value.endswith('.yaml'):
            if not value.startswith('.'):
                relative_path = normalize_path_preserve_leading_dot(os.path.abspath(substitute_root_path(value)))
            save_config(load_config(value, resolve_paths=resolve_paths), directory, os.path.basename(relative_path), reference=reference, recursive=recursive, resolve_paths=resolve_paths)
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

    with open(get_path(directory, filename=filename, create=True), 'w') as config_file:
        yaml.dump(config_with_corrected_paths, config_file, sort_keys=False)


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

    Notes
    -----
    Each generation method ships with sensible defaults plus the knobs listed
    below. Pass keyword arguments to ``GenerationConfig`` to override any of
    them.

    - ``beam_search`` keeps the highest scoring partial programs at every
        decoding step.
            - ``beam_width`` (32): number of beams tracked during search.
            - ``max_len`` (32): maximum decoded token length before truncating.
            - ``mini_batch_size`` (128): batch size used when scoring beams in
                vectorised forward passes.
            - ``equivalence_pruning`` (True): if enabled, simplified expressions
                that canonicalise to duplicates are pruned from the beam.

    - ``softmax_sampling`` draws multiple candidates via stochastic decoding.
            - ``choices`` (32): how many independent samples to draw.
            - ``top_k`` (0): optional top-k cutoff applied before sampling (0
                disables the filter).
            - ``top_p`` (1): nucleus sampling threshold; values < 1 restrict the
                cumulative probability mass considered.
            - ``max_len`` (32): hard stop on sampled sequence length.
            - ``mini_batch_size`` (128): batch size for batched sampling steps.
            - ``temperature`` (1): scales logits prior to sampling; <1 sharpens,
                >1 smooths the distribution.
            - ``valid_only`` (True): reject samples that cannot be parsed into
                valid expressions.
            - ``simplify`` (True): optionally simplify sampled expressions before
                returning them.
            - ``unique`` (True): ensure each returned expression is unique after
                simplification.

    - ``mcts`` runs Monte Carlo Tree Search on the transformer policy.
            - ``beam_width`` (16): number of completions returned after the search
                concludes.
            - ``simulations`` (256): total simulation rollouts executed from the
                root.
            - ``uct_c`` (1.4): exploration constant in the UCT score balancing
                exploitation and exploration.
            - ``expansion_top_k`` (32): how many highest-probability children to
                expand per node.
            - ``max_depth`` (64): maximum token depth before the search or rollout
                terminates a path.
            - ``rollout_max_len`` (None): optional cap for rollout length; falls
                back to ``max_depth`` when ``None``.
            - ``rollout_policy`` ('sample'): strategy for selecting rollout tokens
                ('sample' or 'greedy').
            - ``temperature`` (1.0): sampling temperature applied when the rollout
                policy is ``'sample'``.
            - ``dirichlet_alpha`` (None): concentration for optional Dirichlet noise
                injected at the root to encourage exploration.
            - ``dirichlet_epsilon`` (0.25): mix ratio between model priors and
                Dirichlet noise at the root.
            - ``invalid_penalty`` (1e6): penalty subtracted when a rollout finishes
                without a valid terminal token.
            - ``min_visits_before_expansion`` (1): number of visits required before
                expanding a node's children.
            - ``reward_transform`` (None): optional callable applied to rewards
                before backpropagation.
            - ``completion_sort`` ('reward'): criterion used to rank collected
                completions (``'reward'`` or ``'log_prob'``).
    '''
    def __init__(self, method: Literal['beam_search', 'softmax_sampling', 'mcts'] = 'beam_search', **kwargs: Any) -> None:
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
            },
            'mcts': {
                'beam_width': 16,
                'simulations': 256,
                'uct_c': 1.4,
                'expansion_top_k': 32,
                'max_depth': 64,
                'rollout_max_len': None,
                'rollout_policy': 'sample',
                'temperature': 1.0,
                'dirichlet_alpha': None,
                'dirichlet_epsilon': 0.25,
                'invalid_penalty': 1e6,
                'min_visits_before_expansion': 1,
                'reward_transform': None,
                'completion_sort': 'reward'
            }
        }

        if method not in self.defaults:
            raise ValueError(f'Invalid generation method: {method}')

        self.method = method

        self.config = {**kwargs}

        # Set defaults if not provided
        if method in self.defaults:
            method_defaults = self.defaults[method]
            if not isinstance(method_defaults, dict):
                raise TypeError(f"Defaults for method '{method}' must be a mapping")
            for key, value in method_defaults.items():
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


def pad_input_set(X: np.ndarray | torch.Tensor, length: int) -> np.ndarray | torch.Tensor:
    pad_length = length - X.shape[-1]
    if pad_length > 0:
        # Pad the x_tensor with zeros to match the expected maximum input dimension of the set transformer
        if isinstance(X, torch.Tensor):
            X = nn.functional.pad(X, (0, pad_length, 0, 0), value=0)
        elif isinstance(X, np.ndarray):
            X = np.pad(X, ((0, 0), (0, pad_length)), mode='constant', constant_values=0)  # type: ignore

    return X
