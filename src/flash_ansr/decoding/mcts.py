import math
from operator import attrgetter, itemgetter, methodcaller
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Hashable, Iterable, Mapping, Optional, Sequence, Tuple, Union

import torch
from tqdm import tqdm


@dataclass
class MCTSConfig:
    """Configuration for value-guided best-first (max-backup PUCT) tree-search decoding.

    The search backs up a bounded fit-quality value ``q in [0, 1]`` derived from ``log10(fvu)`` (NOT the raw
    unbounded ``-score``), so the PUCT exploration term is commensurate with exploitation. See
    the 2026-07 decoder-rewrite notes for the audit and reconciled fix spec.
    """

    simulations: int = 256
    """Fallback rollout budget when ``max_rollouts`` is unset. Demoted from 'the budget' to a safety cap."""

    max_rollouts: Optional[int] = None
    """Hard cap on total rollouts (safety bound). Defaults to ``simulations`` when ``None``."""

    refine_budget: Optional[int] = None
    """Primary budget: stop once this many DISTINCT valid canonical completions have been registered
    (== distinct refiner calls, the axis comparable to iid ``c``). ``None`` -> run ``max_rollouts`` rollouts."""

    batch_width: int = 1
    """Leaf-parallel batch width K. ``1`` = the serial loop (byte-identical to the pre-batching search).
    ``K > 1`` collects K leaves per iteration via visit-only virtual loss and advances their expansions/rollouts
    with BATCHED policy forwards (amortizing the batch-1 forward overhead), requiring ``batched_policy_fn`` /
    ``batched_rollout_fn`` to be provided. ``rollout_policy`` must be ``'sample'`` when ``K > 1`` (batched
    divergence relies on stochastic rollouts)."""

    async_search: bool = False
    """Opt-in gate for the OVERLAPPED (generation/refine-concurrent) event loop (``_run_async``). ``False`` keeps
    the proven synchronous ``_run_batched`` (or serial ``run``). When ``True`` AND ``inflight > 1`` AND the
    batched primitives + async refine hooks are present, refines are dispatched as non-blocking Futures so the GPU
    forwards the next micro-batch of leaves WHILE the fork pool refines earlier ones -- porting the softmax path's
    saturated-pool + overlap engine into the tree. Added in the 2026-07 async-overlap rework."""

    inflight: int = 128
    """Async-only: max outstanding un-refined leaves (Futures live at once); the continuous analog of the
    per-batch barrier count K. ``1`` == synchronous (no overlap). Keep ``>= refiner_workers`` (start 2-4x) to
    saturate the pool; do NOT max it -- large ``inflight`` runs selection on stale virtual-loss-only paths
    (toward iid). Swept against the recovery-parity gate."""

    gpu_batch: Optional[int] = None
    """Async-only: micro-batch width for ONE batched policy/rollout forward per produce slice (forward
    amortization; the K-sweep 1.62x came from batched forwards over the overhead-bound ~3.3ms-flat forward, so a
    one-leaf-at-a-time producer would give back part of it). Decoupled from ``inflight``. ``None`` -> defaults to
    ``batch_width`` (or 256)."""

    uct_c: float = 1.4
    """Exploration constant. Meaningful now that the backed-up value is normalized to [0, 1]."""

    expansion_top_k: int = 32
    """How many children to instantiate per expansion step (top-k by policy log-prob)."""

    max_depth: int = 64
    """Maximum tree depth (in tokens); nodes at/below this depth are not expanded."""

    rollout_max_len: Optional[int] = None
    """Optional cap on rollout length; defaults to ``max_depth`` when ``None``."""

    rollout_policy: str = "sample"
    """Rollout strategy, either ``'sample'`` or ``'greedy'``."""

    temperature: float = 1.0
    """Sampling temperature used during rollouts when ``rollout_policy == 'sample'``."""

    rollout_resample_retries: int = 8
    """Max mask-and-resample attempts per rollout step when a pad/invalid token is drawn (kills the
    non-advancing-``continue`` infinite loop under greedy rollout)."""

    dirichlet_alpha: Optional[float] = None
    """If set, inject Dirichlet noise at the root with concentration ``alpha``."""

    dirichlet_epsilon: float = 0.25
    """Mixing factor between model prior and Dirichlet noise at the root."""

    backup: str = "max"
    """Exploitation statistic: ``'max'`` (best_value, the SR best-of-search estimator) or ``'mean'`` (ablation)."""

    fpu_reduction: float = 0.0
    """First-Play-Urgency reduction: an unvisited child's exploitation = parent_exploitation - fpu_reduction."""

    renormalize_prior: bool = True
    """Renormalize the policy prior over the retained top-k children so it sums to 1 (PUCT correctness)."""

    reward_log_fvu_hi: float = 0.0
    """log10(fvu) mapped to q=0 (fvu=1.0, 'explains nothing')."""

    reward_log_fvu_lo: float = -8.0
    """log10(fvu) mapped to q=1 (fvu=1e-8, ~float32 recovery). Must be < ``reward_log_fvu_hi``."""

    value_objective: str = "score"
    """What the tree value optimizes: ``'score'`` (the FULL parsimony selection score
    ``log10(fvu) + length/const/likelihood penalties`` = the SAME objective the winner is chosen by, DEFAULT) or
    ``'fvu'`` (fit quality only, penalty-independent). ``'score'`` aligns search with selection."""

    invalid_penalty: float = 1.0
    """Magnitude of the RETURNED raw reward for invalids (kept finite/orderable). NEVER backed up
    (invalids back up the q=0 floor); demoted from 1e6 which poisoned the mean-value backup."""

    min_visits_before_expansion: int = 1
    """Minimum visit count required before expanding a node."""

    reward_transform: Optional[Callable[[float], float]] = None
    """Optional override mapping raw reward -> q. When set it REPLACES the log_fvu normalization; its output
    is clamped to [0, 1]. Default ``None`` uses the internal log_fvu map."""

    def __post_init__(self) -> None:
        if self.simulations <= 0:
            raise ValueError("simulations must be positive")
        if self.max_rollouts is not None and self.max_rollouts <= 0:
            raise ValueError("max_rollouts must be positive when provided")
        if self.refine_budget is not None and self.refine_budget <= 0:
            raise ValueError("refine_budget must be positive when provided")
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
        if self.rollout_resample_retries < 0:
            raise ValueError("rollout_resample_retries must be non-negative")
        if self.dirichlet_epsilon < 0 or self.dirichlet_epsilon > 1:
            raise ValueError("dirichlet_epsilon must be in [0, 1]")
        if self.backup not in {"max", "mean"}:
            raise ValueError("backup must be either 'max' or 'mean'")
        if self.value_objective not in {"score", "fvu"}:
            raise ValueError("value_objective must be either 'score' or 'fvu'")
        if self.batch_width < 1:
            raise ValueError("batch_width must be >= 1")
        if self.batch_width > 1 and self.rollout_policy != "sample":
            raise ValueError("batch_width > 1 requires rollout_policy='sample' (batched divergence needs stochastic rollouts)")
        if self.inflight < 1:
            raise ValueError("inflight must be >= 1")
        if self.gpu_batch is not None and self.gpu_batch < 1:
            raise ValueError("gpu_batch must be >= 1 when provided")
        if self.async_search and self.inflight > 1 and self.rollout_policy != "sample":
            raise ValueError("async_search with inflight > 1 requires rollout_policy='sample' (batched divergence needs stochastic rollouts)")
        if self.fpu_reduction < 0:
            raise ValueError("fpu_reduction must be non-negative")
        if self.reward_log_fvu_hi <= self.reward_log_fvu_lo:
            raise ValueError("reward_log_fvu_hi must be greater than reward_log_fvu_lo")


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


CanonicalizeFn = Callable[[Tuple[int, ...]], Hashable]
"""Callable mapping a completion's tokens to its canonical dedup key (e.g. simplify+constantify).
Must match the downstream refiner/selector key so 'distinct' means distinct to the refiner."""


@dataclass
class AsyncRefineHooks:
    """Injected refine primitives for the OVERLAPPED async loop (``_run_async``), built in ``generation/mcts.py``
    closing over the fork pool + ``refiner_cache``. They SPLIT the synchronous ``batched_value_fn`` into
    submit/commit halves so the loop can overlap the GPU produce phase with pool refinement, while preserving the
    identical canonical key, cache schema, and (reward, info) contract.
    """

    canon: Callable[[Tuple[int, ...]], Tuple[Hashable, Any]]
    """tokens -> (canonical_key, expr). ``expr is None`` marks an unparseable/invalid completion. The key MUST
    equal ``canonicalize_fn(tokens)`` (both via ``canonicalize_beam``) so dedup/harvest stay consistent."""

    cached: Callable[[Hashable, Any], Optional[Tuple[float, Mapping[str, Any]]]]
    """(key, expr) -> (reward, info) if ``key`` is already in ``refiner_cache``, else ``None`` (a fresh key)."""

    submit: Callable[[Any], Any]
    """expr -> Future (fresh-entropy seed minted internally). ``future.result()`` yields the worker outcome
    ``(result_dict | None, warning)``."""

    serial: Callable[[Any], Any]
    """expr -> result_dict | None. In-process refine (fresh seed) for the pool-broken degrade (no Future)."""

    commit: Callable[[Hashable, Any, Any], Tuple[float, Mapping[str, Any]]]
    """(key, expr, result_dict) -> (reward, info). Populates ``refiner_cache[key]`` iff the fit is valid (schema
    byte-identical to ``batched_value_fn``); returns the invalid floor otherwise. Called ONCE per distinct key."""


@dataclass
class MCTSNode:
    """Single node in the Monte Carlo search tree.

    ``value_sum`` / ``best_value`` accumulate the NORMALIZED value ``q in [0, 1]`` (not raw reward), so both
    ``mean_value()`` and ``best_value`` are on the same bounded scale as the PUCT exploration term.
    """

    tokens: Tuple[int, ...]
    prior: float
    parent: Optional["MCTSNode"] = None
    depth: int = 0
    decoder_state: Optional[Any] = None
    log_prob: float = 0.0

    visits: int = 0
    value_sum: float = 0.0
    best_value: float = 0.0  # finite floor (q=0), not -inf: an unvisited/all-invalid subtree reads as q=0
    vloss: int = 0           # transient leaf-parallel virtual-loss reservations (visit-channel only); sum-zero per batch
    expanded: bool = False
    terminal: bool = False
    children: Dict[int, "MCTSNode"] = field(default_factory=dict)

    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0

    def exploitation(self, backup: str) -> float:
        """The exploitation statistic used in the PUCT score (already in [0, 1])."""
        if backup == "max":
            return self.best_value
        return self.mean_value()


class MonteCarloTreeSearch:
    """Value-guided best-first (max-backup PUCT) tree search for sequence decoding."""

    def __init__(
        self,
        policy_fn: PolicyFn,
        value_fn: ValueFn,
        terminal_fn: TerminalFn,
        config: MCTSConfig,
        eos_token_id: int,
        pad_token_id: Optional[int] = None,
        invalid_sequence_fn: Optional[Callable[[Tuple[int, ...]], bool]] = None,
        canonicalize_fn: Optional[CanonicalizeFn] = None,
        batched_policy_fn: Optional[Callable[[list[Tuple[int, ...]]], list[torch.Tensor]]] = None,
        batched_rollout_fn: Optional[Callable[[list[Tuple[int, ...]]], list[tuple[Optional[Tuple[int, ...]], bool, float]]]] = None,
        batched_value_fn: Optional[Callable[[list[Tuple[int, ...]]], list[tuple[float, Mapping[str, Any]]]]] = None,
        async_hooks: Optional[AsyncRefineHooks] = None,
    ) -> None:
        self.policy_fn = policy_fn
        self.value_fn = value_fn
        self.terminal_fn = terminal_fn
        self.invalid_sequence_fn = invalid_sequence_fn
        self.canonicalize_fn = canonicalize_fn
        # Leaf-parallel batched primitives (provided by mcts_decode); used only when config.batch_width > 1.
        # batched_policy_fn(list_of_token_tuples) -> list of 1D next-token log-prob tensors (batched expansion).
        # batched_rollout_fn(list_of_leaf_tokens) -> list of (final_tokens|None, terminal, rollout_log_prob).
        self.batched_policy_fn = batched_policy_fn
        self.batched_rollout_fn = batched_rollout_fn
        # batched_value_fn(list_of_completion_tokens) -> list of (reward, info): values a whole batch's
        # completions at once, refining the distinct new ones IN PARALLEL (fork pool). None -> serial value_fn.
        self.batched_value_fn = batched_value_fn
        # async_hooks: submit/commit/canon/cached/serial primitives for the OVERLAPPED loop (_run_async); when
        # present with config.async_search + the batched primitives, refines are non-blocking Futures.
        self.async_hooks = async_hooks
        self.config = config
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

        self.rollout_cap = config.rollout_max_len or config.max_depth

        # Completions deduplicated by canonical key -> (tokens, raw_reward, log_prob) keeping the best raw_reward.
        self._completions: dict[Hashable, tuple[Tuple[int, ...], float, float]] = {}
        self._completion_info: dict[Hashable, dict[str, Any]] = {}
        self._n_distinct_valid = 0

        self.root: Optional[MCTSNode] = None
        self._dirichlet_applied = False

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
        """Run the search from ``initial_tokens`` and return the populated root node.

        Stops at ``refine_budget`` distinct valid canonical completions (when set), else after
        ``max_rollouts`` (default ``simulations``) rollouts.
        """
        if (self.config.async_search and self.async_hooks is not None
                and self.batched_policy_fn is not None and self.batched_rollout_fn is not None):
            return self._run_async(initial_tokens, initial_state, progress=progress, progress_desc=progress_desc)

        if self.config.batch_width > 1 and self.batched_policy_fn is not None and self.batched_rollout_fn is not None:
            return self._run_batched(initial_tokens, initial_state, progress=progress, progress_desc=progress_desc)

        root_tokens = tuple(initial_tokens)
        self.root = MCTSNode(tokens=root_tokens, prior=1.0, parent=None, depth=len(root_tokens), decoder_state=initial_state)
        self.root.terminal = self._is_terminal_tokens(root_tokens)
        self._dirichlet_applied = False
        self._completions.clear()
        self._completion_info.clear()
        self._n_distinct_valid = 0

        max_rollouts = self.config.max_rollouts or self.config.simulations
        refine_budget = self.config.refine_budget

        pbar = tqdm(
            total=refine_budget if refine_budget is not None else max_rollouts,
            desc=progress_desc or "MCTS decode",
            dynamic_ncols=True,
            disable=not progress,
            smoothing=0.0,
        ) if progress else None

        try:
            rollout = 0
            while rollout < max_rollouts and (refine_budget is None or self._n_distinct_valid < refine_budget):
                node, path = self._select()

                if node.terminal:
                    q = self._evaluate_terminal(node)
                elif node.visits < self.config.min_visits_before_expansion:
                    q = self._rollout_from(node)
                elif not node.expanded:
                    if self._expand(node):
                        node = self._pick_child_for_simulation(node)
                        path.append(node)
                    q = self._rollout_from(node)
                else:
                    q = self._rollout_from(node)

                self._backpropagate(path, q)
                self._update_progress_bar(pbar, refine_budget)
                rollout += 1
        finally:
            if pbar is not None:
                pbar.close()

        if self.root is None:
            raise RuntimeError("MCTS did not initialize a root node")

        return self.root

    def _run_batched(
        self,
        initial_tokens: Sequence[int],
        initial_state: Optional[Any] = None,
        *,
        progress: bool = False,
        progress_desc: Optional[str] = None,
    ) -> MCTSNode:
        """Leaf-parallel variant of :meth:`run` (config.batch_width K > 1).

        Per iteration: (1) select K leaves, each reserving visit-only virtual loss on its path so the K
        selections diverge; (2) BATCH the expansion policy forwards; (3) pick a simulation node per leaf;
        (4) BATCH the rollout forwards, then score/register the completions through the SAME serial tail
        (value_fn -> _register_completion -> _q_value); (5) undo the reservations and backpropagate the real q.
        Everything except the batching of GPU forwards is byte-identical to the serial loop, and the
        virtual-loss reservations are sum-zero per batch (see the try/finally).
        """
        root_tokens = tuple(initial_tokens)
        self.root = MCTSNode(tokens=root_tokens, prior=1.0, parent=None, depth=len(root_tokens), decoder_state=initial_state)
        self.root.terminal = self._is_terminal_tokens(root_tokens)
        self._dirichlet_applied = False
        self._completions.clear()
        self._completion_info.clear()
        self._n_distinct_valid = 0

        assert self.batched_policy_fn is not None and self.batched_rollout_fn is not None
        K = self.config.batch_width
        max_rollouts = self.config.max_rollouts or self.config.simulations
        refine_budget = self.config.refine_budget
        min_visits = self.config.min_visits_before_expansion

        pbar = tqdm(
            total=refine_budget if refine_budget is not None else max_rollouts,
            desc=progress_desc or "MCTS decode",
            dynamic_ncols=True,
            disable=not progress,
            smoothing=0.0,
        ) if progress else None

        rollout = 0
        try:
            while rollout < max_rollouts and (refine_budget is None or self._n_distinct_valid < refine_budget):
                # Clamp the final batch so we overshoot the distinct-valid budget by at most K-1.
                k_eff = K if refine_budget is None else max(1, min(K, refine_budget - self._n_distinct_valid))

                collected: list[tuple[MCTSNode, list[MCTSNode]]] = []
                touched: dict[int, MCTSNode] = {}   # id(node) -> node (MCTSNode is an unhashable dataclass)
                qs: list[float] = [0.0] * k_eff
                try:
                    # --- Phase 1: select k_eff leaves with visit-only virtual-loss reservations ---
                    for _ in range(k_eff):
                        node, path = self._select()
                        collected.append((node, path))
                        for n in path:
                            n.vloss += 1
                            touched[id(n)] = n

                    # --- Phase 2: BATCHED expansion (dedup identical leaves; forward once each) ---
                    expand_nodes: list[MCTSNode] = []
                    seen_expand: set[int] = set()
                    for node, _path in collected:
                        if (not node.terminal and node.visits >= min_visits and not node.expanded
                                and id(node) not in seen_expand):
                            seen_expand.add(id(node))
                            expand_nodes.append(node)
                    if expand_nodes:
                        logits_list = self.batched_policy_fn([n.tokens for n in expand_nodes])
                        for n, lp in zip(expand_nodes, logits_list):
                            self._expand(n, log_probs=lp)

                    # --- Phase 3: pick a simulation-start node per collected leaf ---
                    sim_nodes: list[MCTSNode] = []
                    for node, path in collected:
                        if node.terminal or node.visits < min_visits:
                            sim = node
                        elif node.expanded and node.children:
                            sim = self._pick_child_for_simulation(node)
                            path.append(sim)
                        else:
                            sim = node  # expansion failed -> roll out from the leaf as-is
                        sim_nodes.append(sim)

                    # --- Phase 4: gather this batch's terminal completions (terminal sims + terminal rollouts),
                    # then VALUE them all at once (parallel refine via batched_value_fn if available). The
                    # rollout log-prob is suffix-only -> prepend the sim node's prefix log-prob to match serial. ---
                    to_value: list[tuple[int, Tuple[int, ...], float]] = []   # (sim index, tokens, full_log_prob)
                    for i, s in enumerate(sim_nodes):
                        if s.terminal:
                            to_value.append((i, s.tokens, s.log_prob))
                    roll_idx = [i for i, s in enumerate(sim_nodes) if not s.terminal]
                    if roll_idx:
                        results = self.batched_rollout_fn([sim_nodes[i].tokens for i in roll_idx])
                        for j, i in enumerate(roll_idx):
                            toks, is_terminal, lp = results[j]
                            if toks is None or not is_terminal or not self._is_terminal_tokens(toks):
                                qs[i] = 0.0  # non-terminal / truncated -> invalid floor, unregistered
                            else:
                                to_value.append((i, tuple(toks), sim_nodes[i].log_prob + float(lp)))

                    if to_value:
                        tokens_list = [t for _, t, _ in to_value]
                        if self.batched_value_fn is not None:
                            vals = self.batched_value_fn(tokens_list)   # one PARALLEL refine batch
                        else:
                            vals = [self._call_value_fn(t) for t in tokens_list]
                        for (i, toks, full_log_prob), (raw, info) in zip(to_value, vals):
                            self._register_completion(toks, raw, float(full_log_prob), dict(info))
                            qs[i] = self._q_value(raw, dict(info))
                finally:
                    # Undo every reservation BEFORE real backprop -- sum-zero, leak-proof even on exception.
                    for n in touched.values():
                        n.vloss = 0

                # --- Phase 5: real backprop per path (finish the batch, then the while re-checks budget) ---
                for (node, path), q in zip(collected, qs):
                    self._backpropagate(path, q)
                    rollout += 1
                    self._update_progress_bar(pbar, refine_budget)
        finally:
            if pbar is not None:
                pbar.close()

        if self.root is None:
            raise RuntimeError("MCTS did not initialize a root node")

        return self.root

    def _run_async(
        self,
        initial_tokens: Sequence[int],
        initial_state: Optional[Any] = None,
        *,
        progress: bool = False,
        progress_desc: Optional[str] = None,
    ) -> MCTSNode:
        """OVERLAPPED variant of :meth:`run` (config.async_search): generation and refinement run CONCURRENTLY.

        A single-thread event loop (no consumer thread, no locks -- the tree is mutated only here) that keeps the
        GPU producing the next micro-batch of ``gpu_batch`` leaves WHILE the fork pool refines up to ``inflight``
        earlier ones as non-blocking Futures. This collapses the sync loop's ~30 per-batch refiner barriers to
        ~one drain, matching the softmax path's saturated pool. Correctness spine:

        * per-leaf virtual loss: reserved (+=1) at select, released (-=1) at that leaf's reap -- a COUNTER, NEVER
          reset-to-0 (the sync idiom), because a node carries reservations from several concurrent in-flight leaves;
        * each distinct canonical key refined AT MOST ONCE: ``submitted_keys`` gates submission, a cache hit
          short-circuits, and a 2nd in-flight leaf on the same key PARKS in ``pending[key]`` (backprops on reap);
        * budget driven off the existing ``_n_distinct_valid`` counter with a ``room`` clamp that subtracts the
          in-flight-key reserve, so ``len(refiner_cache) == _n_distinct_valid == refine_budget`` post-drain;
        * max backup is order-independent, so out-of-order reaps yield identical node stats. At ``inflight=1``,
          ``gpu_batch=1`` the loop is byte-identical to the serial search (the correctness anchor).

        On :class:`BrokenProcessPool` (MCTS is not ``_overlap_mode``) it degrades to serial in-process refine for
        the rest of the tree (``hooks.serial``); it never re-forks mid-tree (the loop is a live-CUDA producer).
        """
        import concurrent.futures
        from concurrent.futures.process import BrokenProcessPool

        root_tokens = tuple(initial_tokens)
        self.root = MCTSNode(tokens=root_tokens, prior=1.0, parent=None, depth=len(root_tokens), decoder_state=initial_state)
        self.root.terminal = self._is_terminal_tokens(root_tokens)
        self._dirichlet_applied = False
        self._completions.clear()
        self._completion_info.clear()
        self._n_distinct_valid = 0

        assert self.batched_policy_fn is not None and self.batched_rollout_fn is not None
        hooks = self.async_hooks
        assert hooks is not None

        cfg = self.config
        inflight = max(1, cfg.inflight)
        gpu_batch = max(1, cfg.gpu_batch if cfg.gpu_batch is not None else (cfg.batch_width if cfg.batch_width > 1 else 256))
        max_rollouts = cfg.max_rollouts or cfg.simulations
        B = cfg.refine_budget if cfg.refine_budget is not None else max_rollouts
        min_visits = cfg.min_visits_before_expansion

        # All state below is touched ONLY by this (main) thread -> lock-free.
        inflight_jobs: dict[Any, tuple[Hashable, Any, tuple[list[MCTSNode], Tuple[int, ...], float, list[MCTSNode]]]] = {}
        submitted_keys: set[Hashable] = set()
        pending: dict[Hashable, list[tuple[list[MCTSNode], Tuple[int, ...], float, list[MCTSNode]]]] = {}
        st = {"n_keys": 0, "broken": False}   # n_inflight_keys (budget reserve); pool-broken degrade flag

        def _release(vnodes: list[MCTSNode]) -> None:
            for n in vnodes:
                n.vloss -= 1

        def _finish_key(key: Hashable, expr: Any, res: Any,
                        leaves: list[tuple[list[MCTSNode], Tuple[int, ...], float, list[MCTSNode]]]) -> None:
            # ONE commit per distinct key (populates refiner_cache iff valid) -> the primary + every parked
            # duplicate leaf gets its real register + backprop + vloss release.
            reward, info = hooks.commit(key, expr, res)
            for (path, toks, lp, vnodes) in leaves:
                self._register_completion(toks, reward, float(lp), dict(info))
                self._backpropagate(path, self._q_value(reward, dict(info)))
                _release(vnodes)

        def _reap(fut: Any) -> None:
            key, expr, primary = inflight_jobs.pop(fut)
            submitted_keys.discard(key)
            st["n_keys"] -= 1
            try:
                outcome = fut.result()
                res = outcome[0] if isinstance(outcome, tuple) else outcome
            except BrokenProcessPool:
                st["broken"] = True
                res = hooks.serial(expr)   # degrade: refine this key in-process; never re-fork mid-tree
            _finish_key(key, expr, res, [primary] + pending.pop(key, []))

        pbar = tqdm(
            total=B, desc=progress_desc or "MCTS decode (async)", dynamic_ncols=True,
            disable=not progress, smoothing=0.0,
        ) if progress else None

        n_produced = 0
        try:
            while self._n_distinct_valid < B and n_produced < max_rollouts:
                # 1) REAP all completed futures (main-thread tree mutation; no callback thread touches the tree)
                for fut in [f for f in inflight_jobs if f.done()]:
                    _reap(fut)
                self._update_progress_bar(pbar, B)
                if self._n_distinct_valid >= B:
                    break

                # 2) BACKPRESSURE: pool full OR budget reserve full -> block on >=1 completion, then loop to reap
                room = min(gpu_batch, inflight - len(inflight_jobs),
                           B - self._n_distinct_valid - st["n_keys"], max_rollouts - n_produced)
                if room <= 0:
                    if not inflight_jobs:
                        break   # nothing in flight, reserve can't yield a new valid key -> done
                    concurrent.futures.wait(list(inflight_jobs), return_when=concurrent.futures.FIRST_COMPLETED)
                    continue

                # 3) PRODUCE a micro-batch of `room` fresh leaves under per-leaf virtual loss (batched forwards)
                leaves_meta: list[tuple[MCTSNode, list[MCTSNode], list[MCTSNode]]] = []
                for _ in range(room):
                    node, path = self._select()
                    vnodes = list(path)
                    for n in vnodes:
                        n.vloss += 1        # reserve BEFORE the next select so this slice's selections diverge
                    leaves_meta.append((node, path, vnodes))
                n_produced += room

                expand_nodes: list[MCTSNode] = []
                seen_expand: set[int] = set()
                for node, _p, _v in leaves_meta:
                    if (not node.terminal and node.visits >= min_visits and not node.expanded and id(node) not in seen_expand):
                        seen_expand.add(id(node))
                        expand_nodes.append(node)
                if expand_nodes:
                    logits_list = self.batched_policy_fn([n.tokens for n in expand_nodes])
                    for n, lp in zip(expand_nodes, logits_list):
                        self._expand(n, log_probs=lp)

                sim_nodes: list[MCTSNode] = []
                for node, path, _v in leaves_meta:
                    if node.terminal or node.visits < min_visits:
                        sim = node
                    elif node.expanded and node.children:
                        sim = self._pick_child_for_simulation(node)
                        path.append(sim)
                    else:
                        sim = node
                    sim_nodes.append(sim)

                # gather completions: terminal sims + BATCHED rollout of non-terminal sims (suffix lp + prefix lp)
                completions: list[Optional[tuple[Tuple[int, ...], float]]] = [None] * len(leaves_meta)
                for i, s in enumerate(sim_nodes):
                    if s.terminal:
                        completions[i] = (s.tokens, s.log_prob)
                roll_idx = [i for i, s in enumerate(sim_nodes) if not s.terminal]
                if roll_idx:
                    results = self.batched_rollout_fn([sim_nodes[i].tokens for i in roll_idx])
                    for j, i in enumerate(roll_idx):
                        toks, is_terminal, lp = results[j]
                        if toks is None or not is_terminal or not self._is_terminal_tokens(toks):
                            completions[i] = None   # non-terminal / truncated -> invalid floor, unregistered
                        else:
                            completions[i] = (tuple(toks), sim_nodes[i].log_prob + float(lp))

                # 4) branch each completion: non-terminal floor / cache hit / park in-flight dup / submit new key
                for i, (node, path, vnodes) in enumerate(leaves_meta):
                    comp = completions[i]
                    if comp is None:
                        self._backpropagate(path, 0.0)
                        _release(vnodes)
                        continue
                    toks, full_lp = comp
                    key, expr = hooks.canon(toks)
                    if expr is None:
                        self._backpropagate(path, 0.0)   # unparseable -> invalid floor, no budget
                        _release(vnodes)
                        continue
                    cached = hooks.cached(key, expr)
                    if cached is not None:               # HIT: backprop now, no Future, no budget
                        reward, info = cached
                        self._register_completion(toks, reward, float(full_lp), dict(info))
                        self._backpropagate(path, self._q_value(reward, dict(info)))
                        _release(vnodes)
                        continue
                    if key in submitted_keys:            # distinct key already in flight -> park (no 2nd refine)
                        pending.setdefault(key, []).append((path, toks, full_lp, vnodes))
                        continue                          # vnodes released when the key reaps
                    # NEW distinct key -> reserve a budget slot + refine
                    submitted_keys.add(key)
                    st["n_keys"] += 1
                    primary = (path, toks, full_lp, vnodes)
                    if not st["broken"]:
                        try:
                            inflight_jobs[hooks.submit(expr)] = (key, expr, primary)
                            continue
                        except Exception:
                            st["broken"] = True   # pool closed/failed at submit -> degrade to serial below
                    # pool degraded -> refine in-process now (no overlap possible anyway), commit + release
                    _finish_key(key, expr, hooks.serial(expr), [primary] + pending.pop(key, []))
                    submitted_keys.discard(key)
                    st["n_keys"] -= 1

            # 5) DRAIN: stop producing, reap every outstanding Future so refiner_cache is complete
            for fut in concurrent.futures.as_completed(list(inflight_jobs)):
                _reap(fut)
                self._update_progress_bar(pbar, B)
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
        backup = self.config.backup
        # Effective visits fold in transient virtual-loss reservations (vloss), so an in-flight leaf-parallel
        # batch diverges via the EXPLORATION channel (the only channel that works under max backup, where a
        # low virtual value is inert). vloss == 0 in the serial loop -> byte-identical to before.
        parent_visits = max(1, node.visits + node.vloss)
        sqrt_parent = math.sqrt(parent_visits)

        # First-Play-Urgency: an unvisited child's exploitation defaults to the parent's exploitation
        # (minus a reduction), clamped to [0, 1] -- scale-correct now that values are normalized.
        fpu = min(1.0, max(0.0, node.exploitation(backup) - self.config.fpu_reduction))

        best_child: Optional[MCTSNode] = None
        best_score = float("-inf")

        for child in node.children.values():
            exploit = child.exploitation(backup) if child.visits > 0 else fpu
            explore = exploration * child.prior * sqrt_parent / (1 + child.visits + child.vloss)
            score = exploit + explore
            if score > best_score:
                best_score = score
                best_child = child

        if best_child is None:
            raise RuntimeError("Expanded node has no children during selection")

        return best_child

    def _expand(self, node: MCTSNode, log_probs: Optional[torch.Tensor] = None) -> bool:
        if node.depth >= self.config.max_depth:
            return False

        # ``log_probs`` may be supplied by a BATCHED policy forward (leaf-parallel path); otherwise fall back
        # to the per-node serial policy_fn. The rest of the expansion body is identical either way.
        child_states: Dict[int, Any] = {}
        if log_probs is None:
            policy_step = self.policy_fn(node.tokens, node.decoder_state)
            log_probs = policy_step.log_probs.detach()
            child_states = policy_step.child_states or {}
        else:
            log_probs = log_probs.detach()

        if log_probs.ndim != 1:
            raise ValueError("policy_fn must return a 1D tensor of log probabilities")

        if log_probs.numel() == 0:
            return False

        top_k = min(self.config.expansion_top_k, log_probs.numel())
        values, indices = torch.topk(log_probs, k=top_k)

        created: list[MCTSNode] = []
        for log_prob, token_id_tensor in zip(values, indices):
            token_id = int(token_id_tensor.item())

            if not math.isfinite(float(log_prob.item())):
                continue  # a non-finite logit (degenerate forward) would poison the prior -> skip the token

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
            child_node.terminal = self._is_terminal_tokens(child_node.tokens)
            node.children[token_id] = child_node
            created.append(child_node)

        if not created:
            return False

        if self.config.renormalize_prior:
            total_prior = sum(c.prior for c in created)
            if total_prior > 0:
                for c in created:
                    c.prior = c.prior / total_prior

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
        alpha = self.config.dirichlet_alpha
        if alpha is None or not node.children:
            return

        noise = torch.distributions.dirichlet.Dirichlet(torch.full((len(node.children),), alpha)).sample()
        for (child, eta) in zip(node.children.values(), noise):
            child.prior = (1 - self.config.dirichlet_epsilon) * child.prior + self.config.dirichlet_epsilon * float(eta.item())

    # ------------------------------------------------------------------
    # Simulation / Rollout
    # ------------------------------------------------------------------
    def _rollout_from(self, node: MCTSNode) -> float:
        """Roll out to a terminal sequence and return the normalized value q in [0, 1].

        A non-terminal outcome (length cap or no legal token) returns the q=0 floor and is not registered.
        """
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

            next_token_id = self._sample_rollout_token(tokens, log_probs)
            if next_token_id is None:
                break  # every candidate was pad/invalid -> stop (non-terminal -> floor)

            tokens.append(next_token_id)
            log_prob += float(log_probs[next_token_id].item())
            state = policy_step.child_states.get(next_token_id) if policy_step.child_states else None
            depth += 1

            if next_token_id == self.eos_token_id:
                break

        if not self._is_terminal_tokens(tokens):
            return 0.0  # non-terminal / truncated -> invalid floor, not scored, not registered

        raw_reward, info = self._call_value_fn(tuple(tokens))
        self._register_completion(tuple(tokens), raw_reward, log_prob, info)
        return self._q_value(raw_reward, info)

    def _sample_rollout_token(self, tokens: list[int], log_probs: torch.Tensor) -> Optional[int]:
        """Pick the next rollout token, masking pad/invalid tokens and resampling (bounded).

        Returns ``None`` when no legal token remains -- fixes the non-advancing ``continue`` that could spin
        forever under greedy rollout.
        """
        masked = log_probs.clone()
        greedy = self.config.rollout_policy == "greedy"

        for _ in range(self.config.rollout_resample_retries + 1):
            if not torch.isfinite(masked).any():
                return None

            if greedy:
                cand = int(torch.argmax(masked).item())
            else:
                probs = torch.softmax(masked / self.config.temperature, dim=0)
                if not torch.isfinite(probs).all() or float(probs.sum().item()) <= 0.0:
                    return None
                cand = int(torch.multinomial(probs, num_samples=1).item())

            if self.pad_token_id is not None and cand == self.pad_token_id:
                masked[cand] = float("-inf")
                continue

            if self.invalid_sequence_fn and self.invalid_sequence_fn(tuple(tokens + [cand])):
                masked[cand] = float("-inf")
                continue

            return cand

        return None

    # ------------------------------------------------------------------
    # Evaluation & Backpropagation
    # ------------------------------------------------------------------
    def _evaluate_terminal(self, node: MCTSNode) -> float:
        if not self._is_terminal_tokens(node.tokens):
            return 0.0

        raw_reward, info = self._call_value_fn(node.tokens)
        self._register_completion(node.tokens, raw_reward, node.log_prob, info)
        return self._q_value(raw_reward, info)

    def _backpropagate(self, path: Iterable[MCTSNode], q: float) -> None:
        for node in path:
            node.visits += 1
            node.value_sum += q
            if q > node.best_value:
                node.best_value = q

    def _q_value(self, raw_reward: float, info: Mapping[str, Any]) -> float:
        """Map a completion's raw reward to a bounded value q in [0, 1] for backup.

        Primary path (``value_objective``): a linear clip mapping the objective to [0, 1] -- the full penalized
        selection score (``'score'``, default) or ``log10(fvu)`` alone (``'fvu'``). Fallbacks: an explicit
        ``reward_transform`` override, an invalid floor for NaN log_fvu, and a sigmoid squash of the raw reward
        when no ``log_fvu`` is available (generic / non-SR value functions).
        """
        if self.config.reward_transform is not None:
            return min(1.0, max(0.0, float(self.config.reward_transform(raw_reward))))

        lf = info.get("log_fvu") if info else None
        if lf is None:
            # generic bounded fallback (non-SR value fns): overflow-safe sigmoid, floor on non-finite input
            r = float(raw_reward)
            if not math.isfinite(r):
                return 0.0
            if r >= 0.0:
                return 1.0 / (1.0 + math.exp(-r))
            e = math.exp(r)
            return e / (1.0 + e)
        if not math.isfinite(float(lf)):
            return 0.0  # invalid floor

        hi, lo = self.config.reward_log_fvu_hi, self.config.reward_log_fvu_lo
        if self.config.value_objective == "fvu":
            # fit quality only: anchor on log10(fvu)
            q = (hi - float(lf)) / (hi - lo)
        else:
            # full penalized selection score: score = -raw_reward = log10(fvu) + length/const/likelihood
            # penalties. Anchor on the score (same hi/lo scale), so the search optimizes the SELECTION objective.
            q = (hi + float(raw_reward)) / (hi - lo)
        return min(1.0, max(0.0, q))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _is_terminal_tokens(self, tokens: Sequence[int]) -> bool:
        # NOTE: a length-truncated non-eos sequence is NOT terminal-valid (it must not be scored as a program).
        if not tokens:
            return False
        if tokens[-1] == self.eos_token_id:
            return True
        return self.terminal_fn(tuple(tokens))

    @staticmethod
    def _is_valid_completion(info: Mapping[str, Any]) -> bool:
        """A completion is harvest-valid unless it is explicitly flagged invalid (NaN ``log_fvu``)."""
        lf = info.get("log_fvu") if info else None
        if lf is not None and not math.isfinite(float(lf)):
            return False
        return True

    def _child_ranking_key(self, by: str) -> Callable[[MCTSNode], float]:
        if by == "visits":
            return attrgetter("visits")
        if by == "value":
            return methodcaller("mean_value")
        if by == "best":
            return attrgetter("best_value")
        raise ValueError("Unsupported selection key")

    def best_child(self, root: Optional[MCTSNode] = None, by: str = "visits") -> MCTSNode:
        """Return the best child of ``root`` by ``'visits'``, ``'value'`` (mean), or ``'best'`` (max)."""
        root = root or self.root
        if root is None:
            raise RuntimeError("MCTS has not been executed yet")
        if not root.children:
            raise ValueError("Root node has no children")

        return max(root.children.values(), key=self._child_ranking_key(by))

    def get_top_completions(self, limit: Optional[int] = None, by: str = "reward") -> list[tuple[Tuple[int, ...], float, float]]:
        """Return valid completions, deduplicated by canonical key BEFORE the limit cut.

        Because completions are deduplicated by their canonical (simplify+constantify) key at registration,
        the top ``limit`` entries are ``limit`` DISTINCT candidates as seen by the refiner -- fixing the
        duplicate-dominated harvest that starved the refiner.

        Parameters
        ----------
        limit : int, optional
            Maximum number of distinct completions to return.
        by : {'reward', 'log_prob'}
            Sorting criterion. Each tuple is ``(tokens, reward, log_prob)`` with ``reward`` the RAW ``-score``.
        """
        if by == "reward":
            sort_index = 1
        elif by == "log_prob":
            sort_index = 2
        else:
            raise ValueError("Unsupported sorting key for completions")

        entries = [
            entry for key, entry in self._completions.items()
            if self._is_valid_completion(self._completion_info.get(key, {}))
        ]
        sorted_completions = sorted(entries, key=itemgetter(sort_index), reverse=True)

        if limit is not None:
            return sorted_completions[:limit]
        return sorted_completions

    def ranked_children(self, root: Optional[MCTSNode] = None, by: str = "visits") -> list[MCTSNode]:
        """Return all children of ``root`` ranked by the requested statistic."""
        root = root or self.root
        if root is None:
            raise RuntimeError("MCTS has not been executed yet")
        return sorted(root.children.values(), key=self._child_ranking_key(by), reverse=True)

    # ------------------------------------------------------------------
    # Internal state helpers
    # ------------------------------------------------------------------
    def _canonical_key(self, tokens: Tuple[int, ...]) -> Hashable:
        if self.canonicalize_fn is None:
            return tokens
        try:
            return self.canonicalize_fn(tokens)
        except Exception:
            return tokens

    def _register_completion(self, tokens: Tuple[int, ...], reward: float, log_prob: float, info: Optional[Mapping[str, Any]] = None) -> None:
        info_dict: dict[str, Any] = dict(info) if info is not None else {}

        if "length" not in info_dict:
            info_dict["length"] = len(tokens)
        if "log_fvu" not in info_dict and "fvu" in info_dict:
            fvu = info_dict.get("fvu")
            if isinstance(fvu, (int, float)) and fvu > 0:
                info_dict["log_fvu"] = math.log10(float(fvu))

        key = self._canonical_key(tokens)
        is_valid = self._is_valid_completion(info_dict)

        existing = self._completions.get(key)
        if existing is None:
            self._completions[key] = (tokens, reward, log_prob)
            self._completion_info[key] = info_dict
            if is_valid:
                self._n_distinct_valid += 1
            return

        # Choose the representative per canonical class with VALIDITY as the primary key and raw reward only
        # as a within-validity tiebreak. The invalid floor (-invalid_penalty) sits INSIDE the valid raw-reward
        # range (a valid poor fit scores below it), so a plain reward comparison would let an invalid entry
        # evict / suppress a valid one -- corrupting the harvest (B4) and the distinct-valid budget count (B5).
        was_valid = self._is_valid_completion(self._completion_info[key])
        if is_valid and not was_valid:
            replace = True                      # a valid completion always displaces an invalid representative
        elif was_valid and not is_valid:
            replace = False                     # an invalid completion never displaces a valid representative
        else:
            replace = reward > existing[1]       # same validity class: keep the best raw reward

        if replace:
            self._completions[key] = (tokens, reward, log_prob)
            self._completion_info[key] = info_dict
        if is_valid and not was_valid:
            self._n_distinct_valid += 1          # monotonic: a key is counted once it first gains a valid rep

    def _best_completion_entry(self) -> Optional[tuple[Tuple[int, ...], float, float, dict[str, Any]]]:
        best_key = None
        best_reward = float("-inf")
        for key, (_tokens, reward, _lp) in self._completions.items():
            if reward > best_reward:
                best_reward = reward
                best_key = key
        if best_key is None:
            return None
        tokens, reward, log_prob = self._completions[best_key]
        return tokens, reward, log_prob, self._completion_info[best_key]

    def _update_progress_bar(self, pbar: Optional[Any], refine_budget: Optional[int]) -> None:
        if pbar is None:
            return

        best_entry = self._best_completion_entry()
        if best_entry is None:
            postfix: Dict[str, Any] = {"log_fvu": "nan", "distinct": self._n_distinct_valid}
        else:
            _tokens, _reward, _lp, info = best_entry
            raw_log_fvu = info.get("log_fvu")
            if isinstance(raw_log_fvu, (int, float)) and math.isfinite(raw_log_fvu):
                log_fvu_display: Any = f"{float(raw_log_fvu):.3f}"
            else:
                log_fvu_display = "nan"
            postfix = {"log_fvu": log_fvu_display, "distinct": self._n_distinct_valid}

        if refine_budget is not None:
            # budget-driven: reflect distinct-valid progress toward the target
            pbar.n = min(self._n_distinct_valid, refine_budget)
            pbar.set_postfix(postfix, refresh=False)
            pbar.refresh()
        else:
            pbar.update(1)
            pbar.set_postfix(postfix, refresh=False)

    def _call_value_fn(self, tokens: Tuple[int, ...]) -> tuple[float, dict[str, Any]]:
        result = self.value_fn(tokens)

        if isinstance(result, ValueEstimate):
            reward = result.reward
            info = dict(result.info) if result.info is not None else {}
        elif isinstance(result, tuple) and len(result) == 2:
            reward, metadata = result
            info = dict(metadata) if isinstance(metadata, Mapping) else {}
        else:
            reward = result  # type: ignore[assignment]
            info = {}

        return float(reward), info


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _std(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 3:
        return float("nan")
    mx, my = _mean(xs), _mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")


def tree_stats(root: MCTSNode, uct_c: float = 1.4) -> Dict[str, Any]:
    """Balance diagnostics for a completed search tree (the root returned by ``run()``). O(nodes), cheap.

    Healthy signatures: the root explores >1 child (``root_n_visited`` > 1) but concentrates rather than
    spreading uniformly (``root_top1_visit_frac`` well above 1/n_children); visits track value
    (``visit_value_corr`` > 0 -- the search spends where the value is); and the exploration term is not swamped
    by exploitation (``root_explore_mean`` comparable to ``root_exploit_spread`` -- the original greedy-collapse
    bug was exploration << exploitation). A tree that visits only ONE child, or whose visits are uncorrelated
    with value, is degenerate (effectively greedy / iid-like).
    """
    nodes: list[MCTSNode] = []
    stack = [root]
    while stack:
        n = stack.pop()
        nodes.append(n)
        stack.extend(n.children.values())

    max_depth = max((n.depth for n in nodes), default=root.depth) - root.depth
    expanded = [n for n in nodes if n.children]
    mean_branching = _mean([len(n.children) for n in expanded]) if expanded else 0.0

    rc = list(root.children.values())
    rc_visits = [c.visits for c in rc]
    tot = sum(rc_visits)
    n_children = len(rc)
    n_visited = sum(1 for v in rc_visits if v > 0)
    top1 = (max(rc_visits) / tot) if tot > 0 else float("nan")
    if tot > 0 and n_children > 1:
        ps = [v / tot for v in rc_visits if v > 0]
        ent = -sum(p * math.log(p) for p in ps) / math.log(n_children)   # normalized [0,1]
    else:
        ent = float("nan")

    vv = [(n.visits, n.best_value) for n in nodes if n is not root and n.visits > 0]
    corr = _pearson([a for a, _ in vv], [b for _, b in vv])

    sqrt_pv = math.sqrt(max(1, root.visits))
    exploit_spread = _std([c.best_value for c in rc])
    explore_mean = _mean([uct_c * c.prior * sqrt_pv / (1 + c.visits) for c in rc])

    return {
        "n_nodes": len(nodes),
        "max_depth": max_depth,
        "mean_branching": round(mean_branching, 2),
        "root_n_children": n_children,
        "root_n_visited": n_visited,
        "root_top1_visit_frac": round(top1, 3) if top1 == top1 else top1,
        "root_visit_entropy": round(ent, 3) if ent == ent else ent,
        "visit_value_corr": round(corr, 3) if corr == corr else corr,
        "root_exploit_spread": round(exploit_spread, 4),
        "root_explore_mean": round(explore_mean, 4),
    }
