"""Generation configuration helpers with method-specific signatures."""
from typing import Any, Callable, Iterator, Literal, Mapping, overload


class GenerationConfigBase(Mapping[str, Any]):
    """Common interface implemented by all generation configuration objects."""

    __slots__ = ('method',)
    method: Literal['beam_search', 'softmax_sampling', 'mcts', 'prior_sampling']

    def to_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments appropriate for the configured method."""
        raise NotImplementedError

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_kwargs())

    def __len__(self) -> int:
        return len(self.to_kwargs())

    def __getitem__(self, key: str) -> Any:
        return self.to_kwargs()[key]

    def as_dict(self) -> dict[str, Any]:
        """Alias for :meth:`to_kwargs` mirroring the Mapping protocol."""
        return self.to_kwargs()

    def __repr__(self) -> str:
        params = ', '.join(f"{key}={value!r}" for key, value in self.to_kwargs().items())
        return f"{self.__class__.__name__}({params})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.to_kwargs() == other.to_kwargs()


class BeamSearchConfig(GenerationConfigBase):
    """Configuration for beam-search based generation."""

    __slots__ = (
        'beam_width',
        'max_len',
        'batch_size',
        'unique',
        'limit_expansions',
    )

    method: Literal['beam_search']
    beam_width: int
    max_len: int
    batch_size: int
    unique: bool
    limit_expansions: bool

    def __init__(
        self,
        *,
        beam_width: int = 32,
        max_len: int = 32,
        batch_size: int = 128,
        unique: bool = True,
        limit_expansions: bool = True,
    ) -> None:
        self.method = 'beam_search'
        self.beam_width = beam_width
        self.max_len = max_len
        self.batch_size = batch_size
        self.unique = unique
        self.limit_expansions = limit_expansions

    def to_kwargs(self) -> dict[str, Any]:
        return {
            'beam_width': self.beam_width,
            'max_len': self.max_len,
            'batch_size': self.batch_size,
            'unique': self.unique,
            'limit_expansions': self.limit_expansions,
        }


class SoftmaxSamplingConfig(GenerationConfigBase):
    """Configuration for softmax sampling generation."""

    __slots__ = (
        'choices',
        'top_k',
        'top_p',
        'max_len',
        'batch_size',
        'temperature',
        'valid_only',
        'simplify',
        'unique',
    )

    method: Literal['softmax_sampling']
    choices: int
    top_k: int
    top_p: float
    max_len: int
    batch_size: int
    temperature: float
    valid_only: bool
    simplify: bool
    unique: bool

    def __init__(
        self,
        *,
        choices: int = 32,
        top_k: int = 0,
        top_p: float = 1.0,
        max_len: int = 64,
        batch_size: int = 128,
        temperature: float = 1.0,
        valid_only: bool = True,
        simplify: bool = True,
        unique: bool = True,
    ) -> None:
        self.method = 'softmax_sampling'
        self.choices = choices
        self.top_k = top_k
        self.top_p = top_p
        self.max_len = max_len
        self.batch_size = batch_size
        self.temperature = temperature
        self.valid_only = valid_only
        self.simplify = simplify
        self.unique = unique

    def to_kwargs(self) -> dict[str, Any]:
        return {
            'choices': self.choices,
            'top_k': self.top_k,
            'top_p': self.top_p,
            'max_len': self.max_len,
            'batch_size': self.batch_size,
            'temperature': self.temperature,
            'valid_only': self.valid_only,
            'simplify': self.simplify,
            'unique': self.unique,
        }


class MCTSGenerationConfig(GenerationConfigBase):
    """Configuration for Monte Carlo tree search generation."""

    __slots__ = (
        'beam_width',
        'simulations',
        'uct_c',
        'expansion_top_k',
        'max_depth',
        'rollout_max_len',
        'rollout_policy',
        'temperature',
        'dirichlet_alpha',
        'dirichlet_epsilon',
        'invalid_penalty',
        'min_visits_before_expansion',
        'reward_transform',
        'completion_sort',
    )

    method: Literal['mcts']
    beam_width: int
    simulations: int
    uct_c: float
    expansion_top_k: int
    max_depth: int
    rollout_max_len: int | None
    rollout_policy: str
    temperature: float
    dirichlet_alpha: float | None
    dirichlet_epsilon: float
    invalid_penalty: float
    min_visits_before_expansion: int
    reward_transform: Callable[[float], float] | None
    completion_sort: str

    def __init__(
        self,
        *,
        beam_width: int = 16,
        simulations: int = 256,
        uct_c: float = 1.4,
        expansion_top_k: int = 32,
        max_depth: int = 64,
        rollout_max_len: int | None = None,
        rollout_policy: str = 'sample',
        temperature: float = 1.0,
        dirichlet_alpha: float | None = None,
        dirichlet_epsilon: float = 0.25,
        invalid_penalty: float = 1e6,
        min_visits_before_expansion: int = 1,
        reward_transform: Callable[[float], float] | None = None,
        completion_sort: str = 'reward',
    ) -> None:
        self.method = 'mcts'
        self.beam_width = beam_width
        self.simulations = simulations
        self.uct_c = uct_c
        self.expansion_top_k = expansion_top_k
        self.max_depth = max_depth
        self.rollout_max_len = rollout_max_len
        self.rollout_policy = rollout_policy
        self.temperature = temperature
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.invalid_penalty = invalid_penalty
        self.min_visits_before_expansion = min_visits_before_expansion
        self.reward_transform = reward_transform
        self.completion_sort = completion_sort

    def to_kwargs(self) -> dict[str, Any]:
        return {
            'beam_width': self.beam_width,
            'simulations': self.simulations,
            'uct_c': self.uct_c,
            'expansion_top_k': self.expansion_top_k,
            'max_depth': self.max_depth,
            'rollout_max_len': self.rollout_max_len,
            'rollout_policy': self.rollout_policy,
            'temperature': self.temperature,
            'dirichlet_alpha': self.dirichlet_alpha,
            'dirichlet_epsilon': self.dirichlet_epsilon,
            'invalid_penalty': self.invalid_penalty,
            'min_visits_before_expansion': self.min_visits_before_expansion,
            'reward_transform': self.reward_transform,
            'completion_sort': self.completion_sort,
        }


GenerationConfig = BeamSearchConfig | SoftmaxSamplingConfig | MCTSGenerationConfig


@overload
def create_generation_config(*, method: Literal['beam_search'] = 'beam_search', **kwargs: Any) -> BeamSearchConfig:
    ...


@overload
def create_generation_config(*, method: Literal['softmax_sampling'], **kwargs: Any) -> SoftmaxSamplingConfig:
    ...


@overload
def create_generation_config(*, method: Literal['mcts'], **kwargs: Any) -> MCTSGenerationConfig:
    ...


def create_generation_config(*, method: Literal['beam_search', 'softmax_sampling', 'mcts'] = 'beam_search', **kwargs: Any) -> GenerationConfig:
    """Factory that builds the method-specific generation configuration."""
    method_normalized = method.lower()
    if method_normalized == 'beam_search':
        return BeamSearchConfig(**kwargs)
    if method_normalized == 'softmax_sampling':
        return SoftmaxSamplingConfig(**kwargs)
    if method_normalized == 'mcts':
        return MCTSGenerationConfig(**kwargs)
    raise ValueError(f"Invalid generation method: {method}")
