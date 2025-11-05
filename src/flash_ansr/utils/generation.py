"""Generation configuration helpers."""
from typing import Any, Iterator, Literal, Mapping


class GenerationConfig(Mapping[str, Any]):
    """Structured container for generation hyperparameters."""

    def __init__(self, method: Literal['beam_search', 'softmax_sampling', 'mcts'] = 'beam_search', **kwargs: Any) -> None:
        self.defaults = {
            'beam_search': {
                'beam_width': 32,
                'max_len': 32,
                'mini_batch_size': 128,
                'equivalence_pruning': True,
                'limit_expansions': True,
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
                'unique': True,
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
                'completion_sort': 'reward',
            },
        }

        if method not in self.defaults:
            raise ValueError(f'Invalid generation method: {method}')

        self.method = method
        self.config = {**kwargs}

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

    def __repr__(self) -> str:
        return str(self.config)

    def __str__(self) -> str:
        return str(self.config)
