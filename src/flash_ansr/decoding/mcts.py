import math
from operator import attrgetter, itemgetter, methodcaller
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import torch
from tqdm import tqdm


@dataclass
class MCTSConfig:
    """Configuration parameters controlling Monte Carlo Tree Search decoding."""

    simulations: int = 256
    """Number of simulation rollouts executed from the root node."""

    uct_c: float = 1.4
    """Exploration constant used in the Upper Confidence bounds for Trees (UCT) score."""

    expansion_top_k: int = 32
    """How many children to instantiate per expansion step (top-k by policy log-prob)."""

    max_depth: int = 64
    """Maximum tree depth (in tokens) before forcing rollout termination."""

    rollout_max_len: Optional[int] = None
    """Optional cap on rollout length; defaults to ``max_depth`` when ``None``."""

    rollout_policy: str = "sample"
    """Rollout strategy, either ``'sample'`` or ``'greedy'``."""

    temperature: float = 1.0
    """Sampling temperature used during rollouts when ``rollout_policy == 'sample'``."""

    dirichlet_alpha: Optional[float] = None
    """If set, inject Dirichlet noise at the root with concentration ``alpha``."""

    dirichlet_epsilon: float = 0.25
    """Mixing factor between model prior and Dirichlet noise at the root."""

    invalid_penalty: float = 1e6
    """Penalty applied when a rollout ends without reaching a terminal token."""

    min_visits_before_expansion: int = 1
    """Minimum visit count required before expanding a node."""

    reward_transform: Optional[Callable[[float], float]] = None
    """Optional transform applied to rewards before backpropagation."""

    def __post_init__(self) -> None:
        if self.simulations <= 0:
            raise ValueError("simulations must be positive")
        if self.expansion_top_k <= 0:
            raise ValueError("expansion_top_k must be positive")
        if self.max_depth <= 0:
            raise ValueError("max_depth must be positive")
        if self.rollout_policy not in {"sample", "greedy"}:
            raise ValueError("rollout_policy must be either 'sample' or 'greedy'")
        if self.rollout_max_len is not None and self.rollout_max_len <= 0:
            raise ValueError("rollout_max_len must be positive when provided")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.dirichlet_epsilon < 0 or self.dirichlet_epsilon > 1:
            raise ValueError("dirichlet_epsilon must be in [0, 1]")


@dataclass
class PolicyStep:
    """Container for the policy model outputs used during expansion/rollout."""

    log_probs: torch.Tensor
    """Log-probabilities over the full vocabulary (1D tensor)."""

    child_states: Optional[Dict[int, Any]] = None
    """Optional per-token decoder state to attach to expanded children."""


PolicyFn = Callable[[Tuple[int, ...], Optional[Any]], PolicyStep]
"""Callable returning next-token log-probabilities (and optional child states)."""


@dataclass(frozen=True)
class ValueEstimate:
    reward: float
    info: Optional[Mapping[str, Any]] = None


ValueFnResult = Union[float, ValueEstimate, Tuple[float, Mapping[str, Any]]]


ValueFn = Callable[[Tuple[int, ...]], ValueFnResult]
"""Callable scoring a completed sequence; higher is better."""


TerminalFn = Callable[[Tuple[int, ...]], bool]
"""Callable determining whether a sequence represents a terminal program."""


@dataclass
class MCTSNode:
    """Single node in the Monte Carlo search tree."""

    tokens: Tuple[int, ...]
    prior: float
    parent: Optional["MCTSNode"] = None
    depth: int = 0
    decoder_state: Optional[Any] = None
    log_prob: float = 0.0

    visits: int = 0
    value_sum: float = 0.0
    best_value: float = float("-inf")
    expanded: bool = False
    terminal: bool = False
    children: Dict[int, "MCTSNode"] = field(default_factory=dict)

    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0

    def uct_score(self, parent_visits: int, exploration: float) -> float:
        exploitation = self.mean_value()
        exploration_term = exploration * self.prior * math.sqrt(parent_visits) / (1 + self.visits)
        return exploitation + exploration_term


class MonteCarloTreeSearch:
    """Generic Monte Carlo Tree Search tailored for sequence decoding."""

    def __init__(
        self,
        policy_fn: PolicyFn,
        value_fn: ValueFn,
        terminal_fn: TerminalFn,
        config: MCTSConfig,
        eos_token_id: int,
        pad_token_id: Optional[int] = None,
        invalid_sequence_fn: Optional[Callable[[Tuple[int, ...]], bool]] = None,
    ) -> None:
        self.policy_fn = policy_fn
        self.value_fn = value_fn
        self.terminal_fn = terminal_fn
        self.invalid_sequence_fn = invalid_sequence_fn
        self.config = config
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

        rollout_cap = config.rollout_max_len or config.max_depth
        self.rollout_cap = rollout_cap

        # Accumulates completed sequences encountered during simulations.
        self._completions: list[tuple[Tuple[int, ...], float, float]] = []
        self._completion_info: list[dict[str, Any]] = []

        self.root: Optional[MCTSNode] = None
        self._dirichlet_applied = False
        self._completion_info.clear()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(
        self,
        initial_tokens: Sequence[int],
        initial_state: Optional[Any] = None,
        *,
        progress: bool = False,
        progress_desc: Optional[str] = None,
    ) -> MCTSNode:
        """Run MCTS starting from ``initial_tokens`` and return the populated root node."""
        root_tokens = tuple(initial_tokens)
        self.root = MCTSNode(tokens=root_tokens, prior=1.0, parent=None, depth=len(root_tokens), decoder_state=initial_state)
        self.root.terminal = self.terminal_fn(root_tokens)
        self._dirichlet_applied = False
        self._completions.clear()
        self._completion_info.clear()

        pbar = tqdm(
            total=self.config.simulations,
            desc=progress_desc or "MCTS decode",
            dynamic_ncols=True,
            disable=not progress,
            smoothing=0.0
        ) if progress else None

        try:
            for _ in range(self.config.simulations):
                node, path = self._select()

                if node.terminal:
                    reward = self._evaluate_terminal(node)
                    self._backpropagate(path, reward)
                    self._update_progress_bar(pbar)
                    continue

                if node.visits < self.config.min_visits_before_expansion:
                    reward = self._rollout_from(node)
                    self._backpropagate(path, reward)
                    self._update_progress_bar(pbar)
                    continue

                if not node.expanded:
                    expanded = self._expand(node)
                    if not expanded:
                        reward = self._rollout_from(node)
                        self._backpropagate(path, reward)
                        self._update_progress_bar(pbar)
                        continue

                    node = self._pick_child_for_simulation(node)
                    path.append(node)

                reward = self._rollout_from(node)
                self._backpropagate(path, reward)
                self._update_progress_bar(pbar)
        finally:
            if pbar is not None:
                pbar.close()

        if self.root is None:
            raise RuntimeError("MCTS did not initialize a root node")

        return self.root

    # ------------------------------------------------------------------
    # Selection & Expansion
    # ------------------------------------------------------------------
    def _select(self) -> Tuple[MCTSNode, list[MCTSNode]]:
        if self.root is None:
            raise RuntimeError("MCTS root not initialized")

        node = self.root
        path = [node]

        while node.expanded and not node.terminal:
            node = self._select_child(node)
            path.append(node)

        return node, path

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        exploration = self.config.uct_c
        parent_visits = max(1, node.visits)

        best_child: Optional[MCTSNode] = None
        best_score = float("-inf")

        for child in node.children.values():
            score = child.uct_score(parent_visits, exploration)
            if score > best_score:
                best_score = score
                best_child = child

        if best_child is None:
            raise RuntimeError("Expanded node has no children during selection")

        return best_child

    def _expand(self, node: MCTSNode) -> bool:
        policy_step = self.policy_fn(node.tokens, node.decoder_state)
        log_probs = policy_step.log_probs.detach()

        if log_probs.ndim != 1:
            raise ValueError("policy_fn must return a 1D tensor of log probabilities")

        if log_probs.numel() == 0:
            return False

        top_k = min(self.config.expansion_top_k, log_probs.numel())
        values, indices = torch.topk(log_probs, k=top_k)

        child_states = policy_step.child_states or {}

        created = 0
        for log_prob, token_id_tensor in zip(values, indices):
            token_id = int(token_id_tensor.item())

            if self.pad_token_id is not None and token_id == self.pad_token_id:
                continue

            child_tokens = node.tokens + (token_id,)

            if self.invalid_sequence_fn and self.invalid_sequence_fn(child_tokens):
                continue

            prior = float(torch.exp(log_prob).item())
            child_state = child_states.get(token_id) if token_id in child_states else None

            child_node = MCTSNode(
                tokens=child_tokens,
                prior=prior,
                parent=node,
                depth=node.depth + 1,
                decoder_state=child_state,
                log_prob=node.log_prob + float(log_prob.item()),
            )
            child_node.terminal = self._is_terminal(child_node)
            node.children[token_id] = child_node
            created += 1

        if created == 0:
            return False

        node.expanded = True

        if node is self.root and self.config.dirichlet_alpha is not None and not self._dirichlet_applied:
            self._apply_dirichlet_noise(node)
            self._dirichlet_applied = True

        return True

    def _pick_child_for_simulation(self, node: MCTSNode) -> MCTSNode:
        unexplored = [child for child in node.children.values() if child.visits == 0]
        if unexplored:
            return unexplored[0]
        return self._select_child(node)

    def _apply_dirichlet_noise(self, node: MCTSNode) -> None:
        if not node.children:
            return

        alpha = self.config.dirichlet_alpha
        if alpha is None:
            return

        noise = torch.distributions.dirichlet.Dirichlet(torch.full((len(node.children),), alpha)).sample()
        for (child, eta) in zip(node.children.values(), noise):
            child.prior = (1 - self.config.dirichlet_epsilon) * child.prior + self.config.dirichlet_epsilon * float(eta.item())

    # ------------------------------------------------------------------
    # Simulation / Rollout
    # ------------------------------------------------------------------
    def _rollout_from(self, node: MCTSNode) -> float:
        tokens = list(node.tokens)
        state = node.decoder_state
        depth = node.depth
        log_prob = node.log_prob

        while depth < self.rollout_cap:
            if self._is_terminal_tokens(tokens):
                break

            policy_step = self.policy_fn(tuple(tokens), state)
            log_probs = policy_step.log_probs.detach()

            if log_probs.numel() == 0:
                break

            if self.config.rollout_policy == "greedy":
                next_token_id = int(torch.argmax(log_probs).item())
            else:
                probs = torch.softmax(log_probs / self.config.temperature, dim=0)
                next_token_id = int(torch.multinomial(probs, num_samples=1).item())

            if self.pad_token_id is not None and next_token_id == self.pad_token_id:
                continue

            candidate_tokens = tokens + [next_token_id]
            if self.invalid_sequence_fn and self.invalid_sequence_fn(tuple(candidate_tokens)):
                continue

            tokens.append(next_token_id)
            log_prob += float(log_probs[next_token_id].item())
            state = policy_step.child_states.get(next_token_id) if policy_step.child_states else None
            depth += 1

            if next_token_id == self.eos_token_id:
                break

        if not self._is_terminal_tokens(tokens):
            return -self.config.invalid_penalty

        reward, info = self._call_value_fn(tuple(tokens))
        if self.config.reward_transform is not None:
            reward = self.config.reward_transform(reward)

        self._register_completion(tuple(tokens), reward, log_prob, info)
        return reward

    # ------------------------------------------------------------------
    # Evaluation & Backpropagation
    # ------------------------------------------------------------------
    def _evaluate_terminal(self, node: MCTSNode) -> float:
        if not self._is_terminal_tokens(node.tokens):
            return -self.config.invalid_penalty

        reward, info = self._call_value_fn(node.tokens)
        if self.config.reward_transform is not None:
            reward = self.config.reward_transform(reward)
        self._register_completion(node.tokens, reward, node.log_prob, info)
        return reward

    def _backpropagate(self, path: Iterable[MCTSNode], reward: float) -> None:
        for node in path:
            node.visits += 1
            node.value_sum += reward
            node.best_value = max(node.best_value, reward)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _is_terminal(self, node: MCTSNode) -> bool:
        return self._is_terminal_tokens(node.tokens)

    def _is_terminal_tokens(self, tokens: Sequence[int]) -> bool:
        if not tokens:
            return False
        if tokens[-1] == self.eos_token_id:
            return True
        if len(tokens) >= self.config.max_depth:
            return True
        return self.terminal_fn(tuple(tokens))

    # Convenience accessors ------------------------------------------------
    def _child_ranking_key(self, by: str) -> Callable[[MCTSNode], float]:
        if by == "visits":
            return attrgetter("visits")
        if by == "value":
            return methodcaller("mean_value")
        if by == "best":
            return attrgetter("best_value")
        raise ValueError("Unsupported selection key")

    def _completion_sort_key(self, by: str) -> Callable[[tuple[Tuple[int, ...], float, float]], float]:
        if by == "reward":
            return itemgetter(1)
        if by == "log_prob":
            return itemgetter(2)
        raise ValueError("Unsupported sorting key for completions")

    def best_child(self, root: Optional[MCTSNode] = None, by: str = "visits") -> MCTSNode:
        """Return the best child of ``root`` (default: visits)."""
        root = root or self.root
        if root is None:
            raise RuntimeError("MCTS has not been executed yet")
        if not root.children:
            raise ValueError("Root node has no children")

        key_fn = self._child_ranking_key(by)

        return max(root.children.values(), key=key_fn)

    def get_top_completions(self, limit: Optional[int] = None, by: str = "reward") -> list[tuple[Tuple[int, ...], float, float]]:
        """Return completed sequences collected during search.

        Parameters
        ----------
        limit : int, optional
            Maximum number of sequences to return.
        by : {'reward', 'log_prob'}
            Sorting criterion for the completions.

        Returns
        -------
        list of tuples
            Each tuple contains ``(tokens, reward, log_prob)``.
        """
        key_fn = self._completion_sort_key(by)

        sorted_completions = sorted(self._completions, key=key_fn, reverse=True)

        if limit is not None:
            return sorted_completions[:limit]

        return sorted_completions

    # ------------------------------------------------------------------
    # Internal state helpers
    # ------------------------------------------------------------------
    def _register_completion(self, tokens: Tuple[int, ...], reward: float, log_prob: float, info: Optional[Mapping[str, Any]] = None) -> None:
        info_dict: dict[str, Any] = dict(info) if info is not None else {}

        if "length" not in info_dict:
            info_dict["length"] = len(tokens)

        if "log_fvu" not in info_dict and "fvu" in info_dict:
            fvu = info_dict.get("fvu")
            if isinstance(fvu, (int, float)) and fvu > 0:
                info_dict["log_fvu"] = math.log10(float(fvu))

        self._completions.append((tokens, reward, log_prob))
        self._completion_info.append(info_dict)

    def _best_completion_reward(self) -> float:
        if not self._completions:
            return float("-inf")
        return max(reward for _, reward, _ in self._completions)

    def _best_completion_entry(self) -> Optional[tuple[Tuple[int, ...], float, float, dict[str, Any]]]:
        if not self._completions:
            return None

        best_index = max(range(len(self._completions)), key=lambda idx: self._completions[idx][1])
        tokens, reward, log_prob = self._completions[best_index]
        info = self._completion_info[best_index]
        return tokens, reward, log_prob, info

    def _update_progress_bar(self, pbar: Optional[Any]) -> None:
        if pbar is None:
            return

        best_entry = self._best_completion_entry()

        if best_entry is None:
            postfix: Dict[str, Any] = {"log_fvu": "nan", "length": "nan"}
        else:
            tokens, _, _, info = best_entry
            raw_log_fvu = info.get("log_fvu")
            length = info.get("length", len(tokens))

            if isinstance(length, float):
                length_display: Any = f"{length:.1f}"
            else:
                length_display = int(length)

            if isinstance(raw_log_fvu, (int, float)) and math.isfinite(raw_log_fvu):
                log_fvu_display: Any = f"{float(raw_log_fvu):.3f}"
            else:
                log_fvu_display = "nan"

            postfix = {"log_fvu": log_fvu_display, "length": length_display}

        pbar.update(1)
        pbar.set_postfix(postfix, refresh=False)

    def ranked_children(self, root: Optional[MCTSNode] = None, by: str = "visits") -> list[MCTSNode]:
        """Return all children of ``root`` ranked by the requested statistic."""
        root = root or self.root
        if root is None:
            raise RuntimeError("MCTS has not been executed yet")
        key_fn = self._child_ranking_key(by)
        return sorted(root.children.values(), key=key_fn, reverse=True)

    def _call_value_fn(self, tokens: Tuple[int, ...]) -> tuple[float, dict[str, Any]]:
        result = self.value_fn(tokens)

        if isinstance(result, ValueEstimate):
            reward = result.reward
            info = dict(result.info) if result.info is not None else {}
        elif isinstance(result, tuple) and len(result) == 2:
            reward, metadata = result
            if metadata is None:
                info = {}
            elif isinstance(metadata, Mapping):
                info = dict(metadata)
            else:
                info = dict(metadata)
        else:
            reward = result  # type: ignore[assignment]
            info = {}

        reward = float(reward)
        return reward, info
