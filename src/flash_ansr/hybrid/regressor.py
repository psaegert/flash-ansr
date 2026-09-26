"""The hybrid: Flash-ANSR seeds PySR; PySR's hall of fame joins Flash-ANSR's candidates; Flash-ANSR's ranking picks.

Two ways to use it. ``HybridRegressor.fit`` is the estimator: one problem in, a :class:`HybridFitResult` out (the
extended candidate pool in Flash-ANSR's ranking order), by work: Flash-ANSR draws ``draws`` candidates and PySR runs
``niterations`` iterations, by default the r* = 0.5 pairing (:data:`R_STAR_LADDER`: the two stages take equal time on
the reference machine). ``HybridRegressor.solve`` is the evaluation path a harness such as srbf drives: the same method
as a record, by work (the knob mode) or by the clock (the clock mode below), with a per-problem generation cache that
lets one pass serve every rung of a ladder or every cell of a sweep.

The clock mode: both stages by the clock.

One budget T per problem is split by a ratio r: Flash-ANSR gets (1 - r) T, PySR gets r T, both by
the clock. Flash-ANSR generates in chunks until its wall time reaches its share (the candidate count
is what the clock allowed); PySR runs its own ``timeout_in_seconds`` = its share minus the running
means of its fixed cost and of the pricing of the candidates it adds, so every problem lands at T
within a landing tolerance. The achieved wall time of every stage is recorded next to the target.

r = 0: Flash-ANSR alone, its rank-0 answer. r = 1: PySR alone, cold. In between: the top-K refined
Flash-ANSR candidates enter PySR as initial ``guesses``. PySR adds candidates: its whole hall of fame
joins the Flash-ANSR candidate pool, priced the way Flash-ANSR prices its own (fit, MDL, the
ranking's score), and Flash-ANSR's sorting picks the prediction from the extended pool.

The snapshot design: iid draws compose, so ONE chunked generation pass per problem serves every
ratio. The pass generates until the wall clock reaches the largest share, snapshotting the merged
pool's top-K (and the seeds) when it passes each ratio's share (1 - r) T -- exactly what a run with
that much time would have produced. Snapshots are cached on disk per problem (``snapshot_dir``,
keyed by ``problem_id``), so the cells for the other ratios reuse them. A snapshot is only valid for
the exact (X, y) it was generated on: it records a fingerprint of the data it saw, and a mismatch
regenerates (and warns) instead of returning another instance's predictions.

The knob mode (``draws`` and ``niterations`` set): the method is defined by its WORK, not by a clock.
Flash-ANSR draws exactly ``draws`` candidates, PySR runs exactly ``niterations`` iterations from the
seeds (its timeout is a safety cap only), and the achieved times are recorded. The same pair runs
identically on any machine; the reference machine turns the pair into a time. ``rungs`` lists every
(draws, niterations) pair of a ladder: one generation pass per problem serves every rung (iid draws
compose), snapshotting the pool at each draws count in one increment per rung.
"""
from __future__ import annotations

import copy
import math
import pickle
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from symbolic_data.token_ops import apply_variable_mapping, normalize_expression

from flash_ansr.hybrid.bridge import data_fingerprint, evaluate_prefix, literal_value, prefix_to_julia
from flash_ansr.hybrid.candidates import pick_prediction, pysr_candidates, refine_settings, score_key
from flash_ansr.inference import RESULT_FORMAT_VERSION, Candidate, CandidateLedger, FitResult
from flash_ansr.scoring import RankingConfig
from flash_ansr.hybrid.clock import ClockState
from flash_ansr.hybrid.pysr_model import PySRSettings, run_pysr, warmup

__all__ = ["HybridConfig", "HybridRegressor"]


#: The r* = 0.5 pairing (owner ruling 2026-09-23): each Flash-ANSR draw count with the PySR iteration count whose
#: times on the reference machine are nearest in log space, so both stages take half of the hybrid's time. Measured
#: for Flash-ANSR T8-20M; re-checked 2026-09-26 against the measured times, every pair is the nearest (512 draws sits
#: between 16 and 32 iterations, equally near both).
R_STAR_LADDER: tuple[tuple[int, int], ...] = ((512, 16), (1024, 64), (2048, 256), (4096, 512), (8192, 1024))
#: Flash-ANSR's own default budget, and so the hybrid's.
DEFAULT_DRAWS = 1024


def r_star_iterations(draws: int) -> int:
    """PySR's iterations for ``draws`` Flash-ANSR draws at r* = 0.5: the ladder's pair, or the pair of the ladder rung
    nearest in log draws (the ladder's ends outside it)."""
    d = max(1, int(draws))
    return min(R_STAR_LADDER, key=lambda pair: (abs(math.log(pair[0]) - math.log(d)), pair[0]))[1]


@dataclass
class HybridConfig:
    """The hybrid's knobs. Without any, it runs by work at the r* = 0.5 default: ``DEFAULT_DRAWS`` draws and their
    paired iterations, with the whole :data:`R_STAR_LADDER` as its rungs; ``draws`` alone takes its r* = 0.5 pair.

    ``budget_s``: T, the time budget per problem (this cell). ``ratio``: r, PySR's share of it (this
    cell). ``ratios`` / ``budgets``: every r and every T of the sweep, so one generation pass per
    problem snapshots at each share (1 - r) T (``ratio`` and ``budget_s`` are always included): an
    r-sweep at a fixed T lists ``ratios``, a T-curve at a fixed r lists ``budgets``. ``k_seeds``: the top-K Flash-ANSR candidates offered to PySR as
    guesses and the size of the stored pool. ``max_seed_complexity``: a token-count cap on the seeds.
    ``landing_tolerance``: a fraction of T the generation may stop short of its share.
    ``pysr_overhead_s`` / ``pricing_reserve_s``: the seeds of the running means of PySR's fixed cost
    and of the added candidates' pricing that PySR's share pays besides the search.
    ``first_chunk`` / ``min_chunk``: the generation chunk sizes; ``pysr_niterations_ceiling``: PySR's
    iteration cap (the clock stops it first); ``min_search_s``: below this search time PySR is not
    run. ``emission``: the emission format Flash-ANSR generates in. ``pysr``: PySR's own knobs."""

    budget_s: float | None = None
    ratio: float | None = None
    ratios: Sequence[float] | None = None
    budgets: Sequence[float] | None = None
    draws: int | None = None
    niterations: int | None = None
    rungs: Sequence[Sequence[int]] | None = None
    pysr_timeout_cap_s: float = 3600.0
    k_seeds: int = 100
    max_seed_complexity: int | None = None
    landing_tolerance: float = 0.01
    pysr_overhead_s: float = 4.0
    pricing_reserve_s: float = 0.2
    first_chunk: int = 64
    min_chunk: int = 16
    pysr_niterations_ceiling: int = 1_000_000
    min_search_s: float = 0.5
    emission: str = "fittable"
    pysr: PySRSettings = field(default_factory=PySRSettings)

    def __post_init__(self) -> None:
        if all(v is None for v in (self.draws, self.niterations, self.budget_s, self.ratio)):
            # the default: the method by its work at r* = 0.5, every rung of the ladder served
            self.draws = DEFAULT_DRAWS
            self.rungs = list(self.rungs or R_STAR_LADDER)
        if self.draws is not None and self.niterations is None:
            self.niterations = r_star_iterations(int(self.draws))
        if self.draws is not None or self.niterations is not None:
            # the knob mode: both stages by their work; the clock fields stay unused
            if self.draws is None or self.niterations is None:
                raise ValueError("the knob mode sets draws (and optionally niterations; r* = 0.5 pairs them otherwise)")
            self.draws, self.niterations = int(self.draws), int(self.niterations)
            if self.draws < 0 or self.niterations < 0:
                raise ValueError("draws and niterations must not be negative")
            pairs = {(int(d), int(i)) for d, i in (self.rungs or [])} | {(self.draws, self.niterations)}
            if any(d < 0 or i < 0 for d, i in pairs):
                raise ValueError("every rung is a (draws, niterations) pair of non-negative counts")
            self.rungs = sorted(pairs)
            self.pysr_timeout_cap_s = float(self.pysr_timeout_cap_s)
            if self.pysr_timeout_cap_s <= 0:
                raise ValueError("pysr_timeout_cap_s must be positive")
            self.ratios, self.budgets = [], []
            self.k_seeds = int(self.k_seeds)
            self.max_seed_complexity = None if self.max_seed_complexity is None else int(self.max_seed_complexity)
            if not isinstance(self.pysr, PySRSettings):
                self.pysr = PySRSettings.from_mapping(self.pysr)
            return
        if self.budget_s is None or self.ratio is None:
            raise ValueError("the clock mode needs budget_s and ratio (the knob mode: draws and niterations)")
        self.rungs = None
        self.budget_s = float(self.budget_s)
        self.ratio = float(self.ratio)
        if self.budget_s <= 0:
            raise ValueError("budget_s must be positive")
        if not 0.0 <= self.ratio <= 1.0:
            raise ValueError("ratio must lie in [0, 1]")
        if not 0.0 < self.landing_tolerance < 0.5:
            raise ValueError("landing_tolerance is a fraction of the budget in (0, 0.5)")
        ratios = {float(r) for r in (self.ratios or [])} | {self.ratio}
        if any(not 0.0 <= r <= 1.0 for r in ratios):
            raise ValueError("every ratio must lie in [0, 1]")
        self.ratios = sorted(ratios)
        budgets = {float(t) for t in (self.budgets or [])} | {self.budget_s}
        if any(t <= 0 for t in budgets):
            raise ValueError("every budget must be positive")
        self.budgets = sorted(budgets)
        self.k_seeds = int(self.k_seeds)
        self.max_seed_complexity = None if self.max_seed_complexity is None else int(self.max_seed_complexity)
        if not isinstance(self.pysr, PySRSettings):
            self.pysr = PySRSettings.from_mapping(self.pysr)

    @property
    def mode(self) -> str:
        """``knobs`` (draws and niterations) or ``clock`` (a budget and a ratio)."""
        return "knobs" if self.draws is not None else "clock"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "HybridConfig":
        value = dict(value)
        pysr = value.pop("pysr", None)
        known = {f for f in cls.__dataclass_fields__ if f != "pysr"}
        unknown = sorted(set(value) - known)
        if unknown:
            raise ValueError(f"unknown hybrid option(s): {', '.join(unknown)}")
        return cls(pysr=PySRSettings.from_mapping(pysr), **value)


class HybridRegressor:
    """Flash-ANSR (a loaded ``FlashANSR``) seeding PySR (in-process); see the module docstring. ``config`` defaults to
    the r* = 0.5 knobs. For :meth:`solve`: ``snapshot_dir`` caches the generation pass per problem (None: no cache);
    ``clock_state_path`` carries the clock's running costs across cells (default: ``clock_state.json`` beside
    ``snapshot_dir``; None: not persisted)."""

    def __init__(self, model: Any, config: HybridConfig | None = None, *, snapshot_dir: str | Path | None = None,
                 clock_state_path: str | Path | None = None) -> None:
        self.model = model
        config = config if config is not None else HybridConfig()
        self.config = config
        self.result_: HybridFitResult | None = None
        self._prepared = False
        self.snapshot_dir = Path(snapshot_dir) if snapshot_dir is not None else None
        if clock_state_path is None and self.snapshot_dir is not None:
            clock_state_path = self.snapshot_dir.parent / "clock_state.json"
        self.clock = ClockState(clock_state_path, config.pysr_overhead_s, config.pricing_reserve_s)

    # -- the model's side ------------------------------------------------------------------------
    @property
    def engine(self) -> Any:
        return self.model.simplipy_engine

    def ranking_config(self) -> dict[str, Any] | None:
        """The model's resolved candidate ranking (flash-ansr's ``RankingConfig.as_dict()``)."""
        ranking = getattr(self.model, "ranking", None)
        as_dict = getattr(ranking, "as_dict", None)
        return dict(as_dict()) if callable(as_dict) else None

    def prepare(self) -> None:
        """Once before the first problem: the snapshot directory, PySR's Julia warmup (when PySR has
        a share and the warmup is on)."""
        self._prepared = True
        if self.snapshot_dir is not None:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        # The emission format is the decoder's sampling policy (flash-ansr 0.17): set once, here.
        generation = getattr(self.model, "generation_config", None)
        if generation is not None and hasattr(generation, "emission"):
            generation.emission = self.config.emission
        cfg = self.config
        searches = int(cfg.niterations or 0) > 0 if cfg.mode == "knobs" else self.pysr_seconds(float(cfg.ratio or 0.0)) > 0
        if searches and cfg.pysr.warmup:
            warmup(cfg.pysr)

    # -- the budget split, by the clock ---------------------------------------------------------
    def generation_seconds(self, ratio: float, budget: float | None = None) -> float:
        """Flash-ANSR's share of the budget: (1 - r) T of wall time (T: the configured budget)."""
        return round((1.0 - ratio) * (float(self.config.budget_s or 0.0) if budget is None else float(budget)), 6)

    def pysr_seconds(self, ratio: float, budget: float | None = None) -> float:
        """PySR's share of the budget: r T of wall time (search + its fixed cost + the added
        candidates' pricing)."""
        return round(ratio * (float(self.config.budget_s or 0.0) if budget is None else float(budget)), 6)

    def time_targets(self) -> list[float]:
        """The generation wall-time targets of every (ratio, budget) cell of the sweep, ascending:
        one chunked generation pass per problem snapshots at each of them."""
        return sorted({self.generation_seconds(r, t) for r in (self.config.ratios or []) for t in (self.config.budgets or [])} - {0.0})

    def pysr_search_seconds(self, ratio: float, budget: float | None = None) -> float:
        """PySR's own clock (``timeout_in_seconds``): its share minus the running means of its fixed
        cost and of the added candidates' pricing; the search stops between iterations, so the
        achieved share lands within one iteration of r T."""
        return max(0.0, self.pysr_seconds(ratio, budget) - self.clock.pysr_overhead.value - self.clock.pricing_reserve.value)

    # -- snapshots -------------------------------------------------------------------------------
    def draw_targets(self) -> list[int]:
        """The knob mode's generation targets: every draws count of the ladder, ascending, without 0."""
        return sorted({int(d) for d, _ in (self.config.rungs or [])} - {0})

    def _snapshot_path(self, problem_id: int | None) -> Path | None:
        if self.snapshot_dir is None or problem_id is None:
            return None
        stem = f"problem_{int(problem_id):06d}"
        return self.snapshot_dir / (f"{stem}.draws.pkl" if self.config.mode == "knobs" else f"{stem}.pkl")

    def _snapshot(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray | None, *, problem_id: int | None,
                  variables: Sequence[str], complexity: int | float | None, up_to: int | None = None) -> dict[Any, dict[str, Any]]:
        """The generation pass for this problem, from the cache when one holds it. ``up_to`` (the knob mode): the
        rung being fitted; without a cache the pass stops there -- a unit of one rung is one generation call of
        exactly its draws -- while a cached pass serves the whole ladder."""
        path = self._snapshot_path(problem_id)
        fingerprint = data_fingerprint(X, y, X_val)
        if path is not None and path.exists():
            with path.open("rb") as fh:
                stored = pickle.load(fh)
            if isinstance(stored, Mapping) and "targets" in stored:
                if stored.get("fingerprint") != fingerprint:
                    warnings.warn(
                        f"hybrid snapshot {path.name} was generated on different data than this problem "
                        "(the data source re-drew it); regenerating. Every cell must read the same frozen data.")
                elif set(self._targets()) <= set(stored["targets"]):
                    return stored["targets"]
        if self.config.mode == "knobs":
            targets = self._generate_draw_snapshots(X, y, X_val, variables=variables, complexity=complexity,
                                                    up_to=None if path is not None else up_to)
        else:
            targets = self._generate_snapshots(X, y, X_val, variables=variables, complexity=complexity)  # type: ignore[assignment]
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".pkl.tmp")
            with tmp.open("wb") as fh:
                pickle.dump({"fingerprint": fingerprint, "targets": targets}, fh)
            tmp.replace(path)
        return targets

    def _targets(self) -> list[Any]:
        if self.config.mode == "knobs":
            return list(self.draw_targets())
        return list(self.time_targets())

    def _pass(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray | None, *, variables: Sequence[str],
              complexity: int | float | None) -> "_GenerationPass":
        return _GenerationPass(self, X, y, X_val, variables=variables, complexity=complexity)

    def _generate_snapshots(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray | None, *, variables: Sequence[str],
                            complexity: int | float | None) -> dict[float, dict[str, Any]]:
        """The clock mode's pass: generate until the wall clock reaches each share."""
        cfg = self.config
        run = self._pass(X, y, X_val, variables=variables, complexity=complexity)
        tolerance = cfg.landing_tolerance * min(cfg.budgets or [0.0])   # the tightest cell's tolerance serves every target
        snap: dict[float, dict[str, Any]] = {}
        for target in self.time_targets():
            # generate until the wall clock reaches the target: a big chunk aimed at 97 % of the
            # remaining seconds from the rate measured on THIS problem so far, then landing chunks;
            # a chunk's time is spent whether or not it yields candidates (a failed chunk counts)
            first_call = run.calls
            while run.cum_wall < target - tolerance:
                remaining = target - run.cum_wall
                if run.drawn == 0 or run.cum_wall <= 0:
                    n = cfg.first_chunk
                else:
                    rate = run.drawn / run.cum_wall
                    n = max(cfg.min_chunk, int(remaining * rate * (0.97 if remaining > 3 * tolerance else 1.0)))
                if run.calls - first_call >= 32:
                    warnings.warn(f"hybrid generation: 32 chunks did not reach the {target:.1f} s target ({run.cum_wall:.1f} s); snapshotting as is")
                    break
                run.chunk(n)
            snap[target] = {"seconds": target, **run.snapshot()}
        return snap

    def _generate_draw_snapshots(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray | None, *, variables: Sequence[str],
                                 complexity: int | float | None, up_to: int | None = None) -> dict[int, dict[str, Any]]:
        """The knob mode's pass: one increment per rung, the pool snapshotted at every draws count up to
        ``up_to`` (None: the whole ladder). A ladder of one rung is one generation call of exactly that
        many draws."""
        run = self._pass(X, y, X_val, variables=variables, complexity=complexity)
        snap: dict[int, dict[str, Any]] = {}
        for target in self.draw_targets():
            if up_to is not None and target > up_to:
                break
            if target > run.drawn:
                run.chunk(target - run.drawn)
            snap[target] = {"draws": target, **run.snapshot()}
        return snap

    @staticmethod
    def _candidate_dict(cand: Any, *, result: Any = None, X: Any = None, X_val: Any = None) -> dict[str, Any]:
        """A candidate as the pool stores it; the curves are evaluated through ``result`` (flash-ansr
        0.17: a candidate carries no predictions) when one is given, else left None (``_seeds``
        evaluates lazily)."""
        y_pred = y_pred_val = None
        if result is not None:
            try:
                y_pred = np.asarray(result.predict(X, rank=cand.rank), dtype=float).reshape(-1)
                if X_val is not None:
                    y_pred_val = np.asarray(result.predict(X_val, rank=cand.rank), dtype=float).reshape(-1)
            except Exception:  # noqa: BLE001 - an unevaluable candidate keeps its record, without curves
                y_pred = y_pred_val = None
        return {
            "raw_beam": [int(t) for t in cand.raw_beam],
            "expression": list(cand.expression),
            "expression_prefix": list(cand.expression_prefix),
            "expression_infix": str(cand.expression_infix),
            "skeleton_prefix": list(cand.skeleton_prefix),
            "constants": list(cand.constants) if cand.constants is not None else None,
            "score": cand.score, "fvu": cand.fvu, "mdl": cand.mdl, "n_nodes": cand.n_nodes,
            "log_prob": cand.log_prob, "pareto_rank": cand.pareto_rank, "spelling": getattr(cand, "spelling", None),
            "y_pred": y_pred,
            "y_pred_val": y_pred_val,
        }

    def _seeds(self, ranked: Sequence[Mapping[str, Any]], arity: Mapping[str, int], variables: Sequence[str],
               X: np.ndarray) -> list[str]:
        """Top-K candidates as Julia-syntax guesses: finite over the support, inside PySR's
        vocabulary, under the complexity cap when one is set. Duplicates collapse."""
        out: list[str] = []
        seen: set[str] = set()
        names = [f"x{i + 1}" for i in range(X.shape[1])]
        for cand in ranked:
            if len(out) >= self.config.k_seeds:
                break
            prefix = cand.get("expression_prefix") or []
            if not prefix or any(t == "<constant>" for t in prefix):
                continue
            if self.config.max_seed_complexity is not None and len(prefix) > self.config.max_seed_complexity:
                continue
            y_pred = cand.get("y_pred")
            if y_pred is None:
                try:
                    (y_pred,) = evaluate_prefix(self.engine, list(prefix), names, X)
                except Exception:  # noqa: BLE001 - an unevaluable candidate is not a seed
                    continue
            if not np.all(np.isfinite(np.asarray(y_pred, dtype=float))):
                continue
            text = prefix_to_julia(prefix, arity, variables)
            if text is None or text in seen:
                continue
            seen.add(text)
            out.append(text)
        return out

    # -- the fit ---------------------------------------------------------------------------------
    @staticmethod
    def _variables(variables: Sequence[str] | None, n: int) -> list[str]:
        names = [str(v) for v in (variables or [])]
        return names if len(names) == n else [f"x{i + 1}" for i in range(n)]

    def solve(self, X: np.ndarray, y: np.ndarray, *, X_val: np.ndarray | None = None, variables: Sequence[str] | None = None,
              problem_id: int | None = None, complexity: int | float | None = None, ratio: float | None = None,
              budget: float | None = None, draws: int | None = None, niterations: int | None = None) -> dict[str, Any]:
        """One problem at the configured ratio and budget (``ratio`` / ``budget`` override them; each
        must be one of the sweep's, so the cached generation pass covers it) -- or, in the knob mode,
        at the configured (draws, niterations) rung (``draws`` / ``niterations`` override it; the pair
        must be one of ``rungs``).

        ``X`` (n, d), ``y`` (n,): the support the candidates are fitted on; ``X_val``: rows to predict
        for; ``variables``: the names of X's columns (PySR's and the reader's; ``x1..xd`` when None
        or of another length); ``problem_id``: the snapshot cache key; ``complexity``: a complexity
        prompt for Flash-ANSR (None: none). Returns a record: the ``hybrid_*`` accounting (targets and
        achieved times of both stages), ``fit_time`` (the achieved total), ``predicted_*`` (rank 0 of
        the extended pool: its infix, prefix, skeleton, constants, score, MDL, source), ``y_pred`` /
        ``y_pred_val`` as (n, 1) columns, PySR's own pick under ``pysr_expression`` and its hall of
        fame under ``equations``, ``hybrid_candidates`` (the ranked pool) and ``prediction_success`` /
        ``error``."""
        cfg = self.config
        if cfg.mode == "knobs":
            return self._fit_knobs(X, y, X_val=X_val, variables=variables, problem_id=problem_id, complexity=complexity,
                                   draws=draws, niterations=niterations)
        if draws is not None or niterations is not None:
            raise ValueError("draws / niterations select a rung of the knob mode; this configuration runs by the clock")
        r = float(cfg.ratio or 0.0) if ratio is None else float(ratio)
        if r not in (cfg.ratios or []):
            raise ValueError(f"ratio {r} is not one of the configured ratios {cfg.ratios}")
        T = float(cfg.budget_s or 0.0) if budget is None else float(budget)
        if T not in (cfg.budgets or []):
            raise ValueError(f"budget {T} is not one of the configured budgets {cfg.budgets}")
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        Xv = np.asarray(X_val, dtype=float) if X_val is not None and np.size(X_val) else np.empty((0, X.shape[1]))
        names = self._variables(variables, X.shape[1])
        gen_seconds = self.generation_seconds(r, T)
        gp_seconds = self.pysr_seconds(r, T)
        values: dict[str, Any] = {
            "hybrid_mode": "clock", "hybrid_ratio": r, "hybrid_budget_s": T, "hybrid_k_seeds": cfg.k_seeds,
            "hybrid_target_generation_s": gen_seconds, "hybrid_target_gp_s": gp_seconds,
            "hybrid_choices": 0, "hybrid_niterations": 0, "prediction_success": False, "error": None,
        }
        seeds: list[str] = []
        gen_wall = 0.0
        best: Mapping[str, Any] | None = None
        flash_candidates: list[Mapping[str, Any]] = []
        if gen_seconds > 0:
            try:
                snap = self._snapshot(X, y, Xv, problem_id=problem_id, variables=names, complexity=complexity)[gen_seconds]
            except Exception as exc:  # noqa: BLE001
                values["error"] = f"hybrid generation failed: {exc}"
                return values
            gen_wall = float(snap["cum_wall"])
            values["hybrid_choices"] = int(snap.get("choices") or 0)     # achieved: the draws the clock allowed
            values["hybrid_generation_calls"] = int(snap.get("calls") or 0)
            seeds = list(snap["seeds"])[: cfg.k_seeds]
            best = snap["best"]
            # the Flash-ANSR candidates PySR's join: the snapshot's top-K pool (rank 0 carries its
            # predictions); a snapshot from before the pool was stored holds rank 0 alone, which the
            # same sorting puts first either way
            flash_candidates = list(snap.get("candidates") or ([best] if best is not None else []))
            values.update(hybrid_generation_s=gen_wall, hybrid_pool_size=snap["pool_size"],
                          hybrid_generation_time=snap["cum_generation"], hybrid_refinement_time=snap["cum_refinement"],
                          hybrid_n_seeds=len(seeds), generation_time=snap["cum_generation"], refinement_time=snap["cum_refinement"])
            if best is not None:
                values["hybrid_seed_best_expression"] = best.get("expression_infix")

        if gp_seconds <= 0:
            # Flash-ANSR alone: the merged pool's rank 0 at this budget
            values["fit_time"] = gen_wall
            if best is None:
                values["error"] = "hybrid: no Flash-ANSR candidate at this budget"
                return values
            self._flash_answer(values, best, n_fit=int(y.shape[0]))
            return values

        # the GP stage, seeded (or cold at r = 1), on PySR's own clock: its share minus the running
        # means of its fixed cost and of the added candidates' pricing that follows the search
        search_s = self.pysr_search_seconds(r, T)
        values["hybrid_gp_timeout_s"] = search_s
        values["hybrid_pysr_overhead_estimate_s"] = self.clock.pysr_overhead.value
        values["hybrid_pricing_reserve_s"] = self.clock.pricing_reserve.value
        if search_s < cfg.min_search_s:
            # the share is eaten by the fixed costs: no search, the Flash-ANSR candidates rank alone
            values.update(hybrid_gp_s=0.0, gp_fit_time=0.0, fit_time=gen_wall, equations=[], pysr_expression=None,
                          pysr_error=f"hybrid: PySR's share ({gp_seconds:.1f} s) leaves no search time")
            self._pick(values, flash_candidates, X, y, Xv, names)
            if not values.get("prediction_success"):
                values["error"] = values["pysr_error"]
            return values
        out = run_pysr(X, y, names, timeout_in_seconds=search_s, niterations=cfg.pysr_niterations_ceiling,
                       guesses=seeds, X_val=Xv if Xv.shape[0] else None, settings=cfg.pysr)
        gp_wall = float(out["wall_s"])
        if not out.get("error"):
            # A FAILED search is not an observation of PySR's fixed cost: a stalled worker returns after
            # the protocol's timeout, and one such row (7,208 s over its clock, 2026-09-12) in the
            # sweep-wide running mean (2.8 s -> 21.8 s) zeroed the search clock of every later problem
            # at r <= 0.2. Only a search that ran to its clock says what the dispatch and the return cost.
            self.clock.pysr_overhead.observe(gp_wall - search_s)
        values.update(hybrid_gp_s=gp_wall, gp_fit_time=gp_wall, fit_time=gen_wall + gp_wall,
                      hybrid_niterations=int(out.get("niterations_used") or 0), equations=list(out.get("equations") or []),
                      n_guesses=int(out.get("n_guesses") or 0), niterations_used=int(out.get("niterations_used") or 0))
        self._pysr_answer(values, out, X, y, Xv, names)
        self._pick(values, flash_candidates, X, y, Xv, names)
        self.clock.pricing_reserve.observe(float(values.get("hybrid_ranking_s") or 0.0))
        self.clock.save()
        return values

    def _fit_knobs(self, X: np.ndarray, y: np.ndarray, *, X_val: np.ndarray | None, variables: Sequence[str] | None,
                   problem_id: int | None, complexity: int | float | None, draws: int | None, niterations: int | None) -> dict[str, Any]:
        """One problem at a (draws, niterations) rung: Flash-ANSR draws exactly ``draws`` candidates (from
        the cached pass when the ladder has several rungs), PySR runs exactly ``niterations`` iterations
        from the seeds under a safety cap, the hall of fame joins the pool and the ranking picks."""
        cfg = self.config
        D = int(cfg.draws or 0) if draws is None else int(draws)
        iters = int(cfg.niterations or 0) if niterations is None else int(niterations)
        if (D, iters) not in (cfg.rungs or []):
            raise ValueError(f"rung ({D}, {iters}) is not one of the configured rungs {cfg.rungs}")
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        Xv = np.asarray(X_val, dtype=float) if X_val is not None and np.size(X_val) else np.empty((0, X.shape[1]))
        names = self._variables(variables, X.shape[1])
        values: dict[str, Any] = {
            "hybrid_mode": "knobs", "hybrid_draws": D, "hybrid_niterations": iters, "hybrid_k_seeds": cfg.k_seeds,
            "hybrid_choices": D, "prediction_success": False, "error": None,
        }
        seeds: list[str] = []
        gen_wall = 0.0
        best: Mapping[str, Any] | None = None
        flash_candidates: list[Mapping[str, Any]] = []
        if D > 0:
            try:
                snap = self._snapshot(X, y, Xv, problem_id=problem_id, variables=names, complexity=complexity, up_to=D)[D]
            except Exception as exc:  # noqa: BLE001
                values["error"] = f"hybrid generation failed: {exc}"
                return values
            gen_wall = float(snap["cum_wall"])
            values["hybrid_generation_calls"] = int(snap.get("calls") or 0)
            seeds = list(snap["seeds"])[: cfg.k_seeds]
            best = snap["best"]
            flash_candidates = list(snap.get("candidates") or ([best] if best is not None else []))
            values.update(hybrid_generation_s=gen_wall, hybrid_pool_size=snap["pool_size"],
                          hybrid_generation_time=snap["cum_generation"], hybrid_refinement_time=snap["cum_refinement"],
                          hybrid_n_seeds=len(seeds), generation_time=snap["cum_generation"], refinement_time=snap["cum_refinement"])
            if best is not None:
                values["hybrid_seed_best_expression"] = best.get("expression_infix")
        if iters <= 0:
            values["fit_time"] = gen_wall
            if best is None:
                values["error"] = "hybrid: no Flash-ANSR candidate at this rung"
                return values
            self._flash_answer(values, best, n_fit=int(y.shape[0]))
            return values
        values["hybrid_gp_timeout_s"] = cfg.pysr_timeout_cap_s
        out = run_pysr(X, y, names, timeout_in_seconds=cfg.pysr_timeout_cap_s, niterations=iters,
                       guesses=seeds, X_val=Xv if Xv.shape[0] else None, settings=cfg.pysr)
        gp_wall = float(out["wall_s"])
        values.update(hybrid_gp_s=gp_wall, gp_fit_time=gp_wall, fit_time=gen_wall + gp_wall,
                      equations=list(out.get("equations") or []), n_guesses=int(out.get("n_guesses") or 0),
                      niterations_used=int(out.get("niterations_used") or 0))
        self._pysr_answer(values, out, X, y, Xv, names)
        self._pick(values, flash_candidates, X, y, Xv, names)
        return values

    @staticmethod
    def _flash_answer(values: dict[str, Any], best: Mapping[str, Any], *, n_fit: int) -> None:
        values["predicted_source"] = "flash-ansr"
        values["predicted_hof_index"] = -1
        values["prediction_success"] = True
        values["predicted_expression"] = best["expression_infix"]
        values["predicted_expression_prefix"] = normalize_expression(list(best["expression_prefix"]))
        values["predicted_skeleton_prefix"] = list(best["skeleton_prefix"])
        values["predicted_constants"] = best["constants"]
        values["predicted_score"] = best["score"]
        values["predicted_log_prob"] = best["log_prob"]
        values["predicted_mdl"] = best["mdl"]
        values["predicted_n_nodes"] = best["n_nodes"]
        values["predicted_pareto_rank"] = best["pareto_rank"]
        y_pred = best["y_pred"]
        values["y_pred"] = np.asarray(y_pred, dtype=float).reshape(-1, 1) if y_pred is not None else np.full((n_fit, 1), np.nan)
        y_pred_val = best["y_pred_val"]
        values["y_pred_val"] = np.asarray(y_pred_val, dtype=float).reshape(-1, 1) if y_pred_val is not None else np.empty((0, 1))

    def _pysr_answer(self, values: dict[str, Any], out: Mapping[str, Any], X: np.ndarray, y: np.ndarray, Xv: np.ndarray,
                     names: Sequence[str]) -> None:
        """PySR's own pick as the record's answer (what stands when nothing can be priced): its infix
        in the fit's variable names, its prefix in the engine grammar, its curves."""
        expression = out.get("expression")
        values["prediction_success"] = expression is not None
        values["error"] = out.get("error")
        values["predicted_expression"] = expression
        values["predicted_expression_prefix"] = None
        if expression is not None:
            rename = {str(v): f"x{i + 1}" for i, v in enumerate(names)}
            try:
                raw = [rename.get(str(t), str(t)) for t in self.engine.read_infix(str(expression))]
                values["predicted_expression_prefix"] = list(normalize_expression(raw) or [])
            except Exception:  # noqa: BLE001 - PySR's spelling the engine cannot read: the infix stands
                pass
        y_pred = out.get("y_pred")
        values["y_pred"] = (np.asarray(y_pred, dtype=float).reshape(-1, 1) if y_pred is not None and np.size(y_pred)
                            else np.full((int(y.shape[0]), 1), np.nan))
        y_pred_val = out.get("y_pred_val")
        values["y_pred_val"] = (np.asarray(y_pred_val, dtype=float).reshape(-1, 1)
                                if y_pred_val is not None and np.size(y_pred_val) else np.empty((0, 1)))

    def _pick(self, values: dict[str, Any], flash: Sequence[Mapping[str, Any]], X: np.ndarray, y: np.ndarray, Xv: np.ndarray,
              names: Sequence[str]) -> None:
        """PySR's hall of fame joins the Flash-ANSR candidates and Flash-ANSR's sorting picks the
        prediction; the pricing time of the added candidates is part of the recorded fit time."""
        ranking = self.ranking_config()
        if not ranking:
            values["predicted_source"] = "pysr"
            values["hybrid_ranking_error"] = "no ranking config to price the candidates with"
            return
        t0 = time.time()
        # The two-part code (flash-ansr >= 0.18's default ranking) weighs each bit by the support size, so
        # the weights come from weights_for(n): n = the rows with a finite target, exactly as flash-ansr counts
        # the n its own candidates were ranked with. effective_weights raises under that ranking.
        n_points = int(np.isfinite(np.asarray(y, dtype=float)).reshape(len(y), -1).all(axis=1).sum())
        pick_prediction(values, flash=flash, equations=values.get("equations"), engine=self.engine,
                        weights=RankingConfig.from_dict(dict(ranking)).weights_for(n_points),
                        x_support=X, y_fit=y, x_val=Xv if Xv.shape[0] else None, variables=list(names),
                        refine=refine_settings(self.model))
        values["hybrid_ranking_s"] = time.time() - t0
        values["fit_time"] = float(values.get("fit_time") or 0.0) + values["hybrid_ranking_s"]

    # -- the estimator: one problem in, a HybridFitResult out ----------------------------------------------------
    def fit(self, X: Any, y: Any, variable_names: list[str] | dict[str, str] | Literal['auto'] | None = 'auto', *,
            draws: int | None = None, niterations: int | None = None, complexity: int | float | None = None,
            seed: int | None = None) -> "HybridFitResult":
        """Symbolic regression on ``(X, y)`` by work: Flash-ANSR draws ``draws`` candidates and fits them, its best
        ``k_seeds`` seed PySR's populations, PySR runs ``niterations`` iterations, its hall of fame is priced the way
        Flash-ANSR prices its own candidates and joins them, and Flash-ANSR's ranking orders the extended pool.

        ``draws`` / ``niterations`` default to the configuration's pair (the r* = 0.5 default: 1024 draws, 64
        iterations); ``draws`` alone takes its r* = 0.5 pair (:func:`r_star_iterations`); ``niterations=0`` is
        Flash-ANSR alone. ``variable_names`` / ``complexity`` / ``seed`` are Flash-ANSR's (``FlashANSR.fit``). Returns
        the :class:`HybridFitResult` and keeps it as ``self.result_``, which :meth:`predict` and
        :meth:`get_expression` read."""
        cfg = self.config
        if draws is not None:
            D = int(draws)
            iters = int(niterations) if niterations is not None else r_star_iterations(D)
        elif cfg.mode == "knobs":
            D = int(cfg.draws or 0)
            iters = int(niterations) if niterations is not None else int(cfg.niterations or 0)
        else:   # a clock configuration serves solve(); fit runs by work, at the default pair
            D = DEFAULT_DRAWS
            iters = int(niterations) if niterations is not None else r_star_iterations(D)
        if D < 1 or iters < 0:
            raise ValueError("fit needs at least one draw and a non-negative iteration count")
        if not self._prepared:
            self.prepare()
        Xn = np.asarray(X.values if hasattr(X, "values") else X, dtype=float)
        Xn = Xn.reshape(-1, 1) if Xn.ndim == 1 else Xn
        yn = np.asarray(y.values if hasattr(y, "values") else y, dtype=float).reshape(-1)
        result = self.model.fit(X, y, variable_names=variable_names, draws=D, complexity=complexity, seed=seed)
        names = [f"x{i + 1}" for i in range(Xn.shape[1])]   # PySR's names: always valid identifiers
        seeds: list[str] = []
        if iters > 0 and cfg.k_seeds > 0:
            top = [self._candidate_dict(c, result=result, X=Xn) for c in result.candidates[: cfg.k_seeds]]
            seeds = self._seeds(top, dict(self.engine.operator_arity), names, Xn)
        out: Mapping[str, Any] = {"equations": [], "wall_s": 0.0, "error": None}
        rows: list[dict[str, Any]] = []
        pricing = 0.0
        if iters > 0:
            out = run_pysr(Xn, yn, names, timeout_in_seconds=cfg.pysr_timeout_cap_s, niterations=iters, guesses=seeds, settings=cfg.pysr)
            ranking = self.ranking_config()
            if ranking and out.get("equations"):
                n_points = result.n_points if result.n_points is not None else int(np.isfinite(yn).sum())
                t0 = time.time()
                rows = pysr_candidates(out.get("equations"), engine=self.engine,
                                       weights=RankingConfig.from_dict(dict(ranking)).weights_for(int(n_points)),
                                       x_support=Xn, y_fit=yn, x_val=None, variables=names, refine=refine_settings(self.model))
                pricing = time.time() - t0
        tagged = [(HybridCandidate(**c.__dict__, source="flash-ansr"), ("flash-ansr", i)) for i, c in enumerate(result.candidates)]
        tagged += [(self._pysr_candidate(row, variable_mapping=result.variable_mapping), ("pysr", j)) for j, row in enumerate(rows)]
        tagged.sort(key=lambda pair: _candidate_order(pair[0]))
        candidates = [c for c, _ in tagged]
        for k, c in enumerate(candidates):
            c.rank = k   # the candidates are this call's own copies
        position = {tag: k for k, (_, tag) in enumerate(tagged)}
        ledger = copy.deepcopy(result.ledger)   # its links into the candidate list follow the new order
        for i, old in enumerate(ledger.result_index):
            if old >= 0:
                ledger.result_index[i] = ledger.rank[i] = position[("flash-ansr", old)]
        self.result_ = HybridFitResult(
            candidates=candidates, ledger=ledger, generation_time=result.generation_time, refinement_time=result.refinement_time,
            ranking=result.ranking, n_variables=result.n_variables, variable_mapping=dict(result.variable_mapping),
            draws=D, n_points=result.n_points, engine=result.engine, niterations=iters,
            pysr_time=float(out.get("wall_s") or 0.0) + pricing, n_seeds=len(seeds),
            pysr_equations=[dict(e) for e in (out.get("equations") or [])], pysr_error=out.get("error"))
        return self.result_

    def _pysr_candidate(self, row: Mapping[str, Any], *, variable_mapping: Mapping[str, str] | None) -> "HybridCandidate":
        """A priced PySR row as a candidate of the result: its non-integer literals are its constants (the slots
        ``get_expression(precision=)`` rounds), integers such as exponents stay spelled."""
        realized = [str(t) for t in row["expression_prefix"]]
        expression, slots, constants = [], [], []
        for i, token in enumerate(realized):
            value = literal_value(token)
            if value is not None and token not in ("np.pi", "np.e", "pi", "e", "E") and not float(value).is_integer():
                expression.append("<constant>")
                slots.append(i)
                constants.append(float(value))
            else:
                expression.append(token)
        shown = list(apply_variable_mapping(realized, dict(variable_mapping))) if variable_mapping else realized
        return HybridCandidate(
            raw_beam=[], expression=expression, slots=slots, expression_prefix=realized,
            expression_infix=str(self.engine.prefix_to_infix(shown)), skeleton_prefix=list(row.get("skeleton_prefix") or []),
            constants=constants, constants_emitted=None, log_prob=float("nan"), score=float(row["score"]), fvu=float(row["fvu"]),
            n_nodes=len(realized), mu=None, mdl=row.get("mdl"), constant_count=len(slots), pruned_variant=False,
            pareto_rank=-1, rank=-1, spelling=row.get("spelling"), source="pysr", hof_index=int(row.get("hof_index", -1)))

    def _result(self) -> "HybridFitResult":
        if self.result_ is None:
            raise ValueError("call fit() first")
        return self.result_

    def predict(self, X: Any, rank: int = 0) -> np.ndarray:
        """Evaluate the fitted candidate at ``rank`` (0 = the prediction) on ``X`` -> ``(n_points, 1)``."""
        return self._result().predict(X, rank=rank)

    def get_expression(self, rank: int = 0, **kwargs: Any) -> list[str] | str:
        """The candidate at ``rank`` with its constants substituted (see ``FitResult.get_expression``)."""
        return self._result().get_expression(rank, **kwargs)


def _candidate_order(c: Candidate) -> tuple[float, tuple[str, ...]]:
    """Flash-ANSR's ranking order (``score_key``): the score, a non-finite one last, ties on the expression tokens."""
    try:
        value = float(c.score)
    except (TypeError, ValueError):
        value = math.inf
    return (value if math.isfinite(value) else math.inf, tuple(map(str, c.expression_prefix)))


@dataclass
class HybridCandidate(Candidate):
    """A candidate of the extended pool: one of Flash-ANSR's own, or one of PySR's hall of fame priced and scored the
    way Flash-ANSR prices its own (``raw_beam`` empty, ``log_prob`` nan)."""

    source: str = "flash-ansr"          # "flash-ansr" | "pysr"
    hof_index: int = -1                 # the row of PySR's hall of fame; -1 for a Flash-ANSR candidate


@dataclass
class HybridFitResult(FitResult):
    """A ``FitResult`` whose ``candidates`` are the extended pool -- Flash-ANSR's and PySR's, in Flash-ANSR's ranking
    order -- and what the PySR stage did. ``ledger`` is Flash-ANSR's own generation pool."""

    niterations: int | None = None      # PySR's iterations
    pysr_time: float = 0.0              # PySR's wall time, the pricing of its candidates included
    n_seeds: int = 0                    # the Flash-ANSR candidates offered to PySR as initial guesses
    pysr_equations: list[dict[str, Any]] = field(default_factory=list)   # PySR's hall of fame, as it reported it
    pysr_error: str | None = None       # a failed PySR stage: Flash-ANSR's candidates ranked alone

    def save(self, path: str | Path) -> None:
        """Persist the result as plain data (no engine, no model objects), the PySR stage included."""
        payload = {
            'format_version': RESULT_FORMAT_VERSION, 'hybrid': True,
            'candidates': [dict(c.__dict__) for c in self.candidates], 'ledger': self.ledger.__dict__,
            'generation_time': self.generation_time, 'refinement_time': self.refinement_time,
            'ranking': self.ranking.as_dict(), 'n_variables': self.n_variables, 'variable_mapping': dict(self.variable_mapping),
            'draws': self.draws, 'n_points': self.n_points, 'niterations': self.niterations, 'pysr_time': self.pysr_time,
            'n_seeds': self.n_seeds, 'pysr_equations': list(self.pysr_equations), 'pysr_error': self.pysr_error,
        }
        with open(path, 'wb') as fh:
            pickle.dump(payload, fh)

    @classmethod
    def load(cls, path: str | Path, *, engine: Any = None) -> "HybridFitResult":
        """Read a saved hybrid result; ``engine`` (a simplipy engine) enables ``predict`` / rendering."""
        with open(path, 'rb') as fh:
            payload = pickle.load(fh)
        if int(payload.get('format_version', 0)) != RESULT_FORMAT_VERSION or not payload.get('hybrid'):
            raise ValueError("not a hybrid FitResult of the supported format")
        return cls(
            candidates=[HybridCandidate(**c) for c in payload['candidates']], ledger=CandidateLedger(**payload['ledger']),
            generation_time=float(payload['generation_time']), refinement_time=float(payload['refinement_time']),
            ranking=RankingConfig.from_dict(payload['ranking']), n_variables=int(payload['n_variables']),
            variable_mapping=dict(payload.get('variable_mapping') or {}), draws=payload.get('draws'),
            n_points=payload.get('n_points'), engine=engine, niterations=payload.get('niterations'),
            pysr_time=float(payload.get('pysr_time') or 0.0), n_seeds=int(payload.get('n_seeds') or 0),
            pysr_equations=list(payload.get('pysr_equations') or []), pysr_error=payload.get('pysr_error'))


class _GenerationPass:
    """One problem's generation pass: chunks of draws merged into one pool (the best entry per
    distinct beam), with the achieved wall, generation and refinement time; ``snapshot()`` is the
    pool's state as a cell reads it (its top-K, the seeds, rank 0 with its curves)."""

    def __init__(self, owner: HybridRegressor, X: np.ndarray, y: np.ndarray, X_val: np.ndarray | None, *,
                 variables: Sequence[str], complexity: int | float | None) -> None:
        self.owner, self.X, self.y, self.variables, self.complexity = owner, X, y, list(variables), complexity
        self.x_val = X_val if X_val is not None and np.asarray(X_val).shape[0] > 0 else None
        self.numpy_errors = getattr(owner.model, "numpy_errors", None)
        self.arity = dict(getattr(owner.model.simplipy_engine, "operator_arity", {}) or {})
        self.pool: dict[tuple[Any, ...], dict[str, Any]] = {}
        self.cum_wall = self.cum_gen = self.cum_ref = 0.0
        self.drawn = 0
        self.calls = 0

    def chunk(self, n: int) -> None:
        """One generation call of ``n`` draws; its time is spent whether or not it yields candidates."""
        model, cfg = self.owner.model, self.owner.config
        t0 = time.time()
        try:
            if self.numpy_errors is not None:
                with np.errstate(all=self.numpy_errors):
                    result = model.fit(self.X, self.y, variable_names=self.variables, draws=int(n), complexity=self.complexity)
            else:
                result = model.fit(self.X, self.y, variable_names=self.variables, draws=int(n), complexity=self.complexity)
        except Exception as exc:  # noqa: BLE001 - a failed chunk leaves the pool as it was; its time is spent
            warnings.warn(f"hybrid generation chunk of {n} failed: {exc}")
            result = None
        self.cum_wall += time.time() - t0
        self.drawn += int(n)
        self.calls += 1
        if result is not None:
            self.cum_gen += float(getattr(result, "generation_time", 0.0) or 0.0)
            self.cum_ref += float(getattr(result, "refinement_time", 0.0) or 0.0)
            n_curves = max(1, cfg.k_seeds)   # the seeds and the answer need their curves; the rest stay lazy
            for cand in result.candidates:
                key = tuple(int(t) for t in cand.raw_beam) + ((cand.spelling,) if getattr(cand, "spelling", None) else ())
                entry = self.owner._candidate_dict(cand, result=result if cand.rank < n_curves else None, X=self.X, X_val=self.x_val)
                old = self.pool.get(key)
                if old is None or score_key(entry) < score_key(old):
                    self.pool[key] = entry

    def snapshot(self) -> dict[str, Any]:
        cfg = self.owner.config
        ranked = sorted(self.pool.values(), key=score_key)
        seeds = self.owner._seeds(ranked, self.arity, self.variables, self.X)
        return {
            "choices": self.drawn, "cum_wall": self.cum_wall, "cum_generation": self.cum_gen,
            "cum_refinement": self.cum_ref, "calls": self.calls,
            "pool_size": len(self.pool), "best": ranked[0] if ranked else None, "seeds": seeds,
            # the top-K of the ranked pool without their prediction curves: the candidates PySR's join
            "candidates": [ranked[0]] + [{k: v for k, v in c.items() if k not in ("y_pred", "y_pred_val")}
                                         for c in ranked[1:max(1, cfg.k_seeds)]] if ranked else [],
        }
