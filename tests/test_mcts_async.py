"""Correctness anchor for the OVERLAPPED async MCTS loop (``MonteCarloTreeSearch._run_async``).

These tests use DETERMINISTIC mocks (fixed policy, greedy rollout, value keyed by canonical key, real Futures via
a ThreadPoolExecutor) so a failure means a real loop bug, not model/refiner noise. They assert the invariants the
async loop must preserve vs the proven synchronous ``_run_batched``:

* IDENTITY: at inflight=1, gpu_batch=1 the async loop is byte-identical to _run_batched(batch_width=1) -- same
  refiner_cache, same _completions, same per-node {visits, value_sum, best_value}.
* BUDGET EXACTNESS: len(refiner_cache) == _n_distinct_valid == refine_budget across (inflight, gpu_batch).
* VLOSS SUM-ZERO: every node.vloss == 0 after the run (per-leaf reserve/release, no reset-to-0 clobber).
* ONE-REFINE-PER-KEY: each valid canonical key is committed exactly once (submitted_keys + pending coalescing).
* ORDER-INVARIANCE: shuffling reap order (via variable mock refine latency) yields identical node stats.
"""
from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional, Tuple

import torch

from flash_ansr.decoding.mcts import AsyncRefineHooks, MCTSConfig, MonteCarloTreeSearch, PolicyStep


EOS = 0
VOCAB = 6            # tokens 0..5; 0 == eos
MAX_TOK = 5


def _logits(tokens: Tuple[int, ...]) -> torch.Tensor:
    """Deterministic next-token log-probs: a fixed, token-dependent (but non-degenerate) distribution."""
    base = torch.tensor([0.5, 1.0, 0.8, 0.6, 0.4, 0.2])          # favors eos + low tokens
    tilt = torch.tensor([(1 + (sum(tokens) + v) % 3) * 0.3 for v in range(VOCAB)])
    return torch.log_softmax(base + tilt, dim=-1)


def _terminal(tokens: Tuple[int, ...]) -> bool:
    return bool(tokens) and tokens[-1] == EOS


def _canonical_key(tokens: Tuple[int, ...]) -> Tuple[int, ...]:
    """Collapse several token sequences onto one key (exercise dedup + park-coalescing): drop eos, sort,
    and bucket the length so distinct rollouts can share a canonical key."""
    body = tuple(sorted(t for t in tokens if t != EOS))
    return body


def _key_reward(key: Tuple[int, ...], invalids: bool) -> tuple[float, dict[str, Any]]:
    """Deterministic value for a canonical key: reward = -score, info carries log_fvu. Optionally mark a
    subset invalid (log_fvu nan) to exercise the invalid path + budget accounting."""
    if invalids and (sum(key) % 4 == 0) and len(key) > 0:
        return (-1.0, {"length": len(key), "log_fvu": float("nan")})     # invalid floor
    log_fvu = -(1.0 + (sum(key) % 5))          # deterministic, finite
    reward = -(-log_fvu)                        # score == -log_fvu (no penalties in the mock) -> reward == log_fvu
    return (reward, {"fvu": 10.0 ** log_fvu, "log_fvu": float(log_fvu), "length": len(key)})


def _make_batched_primitives():
    def batched_policy_fn(token_lists: list[Tuple[int, ...]]) -> list[torch.Tensor]:
        return [_logits(t) for t in token_lists]

    def batched_rollout_fn(token_lists: list[Tuple[int, ...]]) -> list[tuple[Optional[Tuple[int, ...]], bool, float]]:
        out = []
        for t in token_lists:
            toks = list(t)
            lp = 0.0
            depth = len(toks)
            while depth < 16 and not _terminal(tuple(toks)):
                lps = _logits(tuple(toks))
                nxt = int(torch.argmax(lps).item())          # GREEDY -> deterministic
                lp += float(lps[nxt].item())
                toks.append(nxt)
                depth += 1
                if nxt == EOS:
                    break
            out.append((tuple(toks), _terminal(tuple(toks)), lp))
        return out

    def policy_fn(tokens: Tuple[int, ...], _state: Any) -> PolicyStep:
        return PolicyStep(log_probs=_logits(tokens))

    return policy_fn, batched_policy_fn, batched_rollout_fn


def _make_async_hooks(refiner_cache: dict, executor: ThreadPoolExecutor, invalids: bool,
                      commit_log: list, latency: bool = False):
    """Async hooks over a real ThreadPoolExecutor (real Futures: done()/result()/wait). Deterministic refine
    keyed by canonical key; optional variable latency to shuffle completion order."""
    def canon(tokens: Tuple[int, ...]) -> tuple[Any, Any]:
        if not _terminal(tokens):
            return (tokens, None)
        return (_canonical_key(tokens), _canonical_key(tokens))    # expr == key (mock)

    def cached(key: Any, expr: Any) -> Optional[tuple[float, dict[str, Any]]]:
        c = refiner_cache.get(key)
        if c is None:
            return None
        return (c["reward"], {"fvu": c["fvu"], "log_fvu": c["log_fvu"], "length": len(expr)})

    def _refine(expr: Any) -> dict:
        if latency:
            time.sleep(0.001 * (1 + (sum(expr) % 5)))       # variable -> out-of-order completion
        reward, info = _key_reward(expr, invalids)
        return {"reward": reward, "info": info, "expr": expr}

    def submit(expr: Any) -> Any:
        return executor.submit(_refine, expr)               # real Future

    def serial(expr: Any) -> dict:
        return _refine(expr)

    def commit(key: Any, expr: Any, res: Any) -> tuple[float, dict[str, Any]]:
        reward, info = res["reward"], dict(res["info"])
        commit_log.append((key, math.isfinite(float(info.get("log_fvu", float("nan"))))))
        if math.isfinite(float(info.get("log_fvu", float("nan")))):
            refiner_cache[key] = {"reward": reward, "fvu": info["fvu"], "log_fvu": info["log_fvu"]}
        return (reward, info)

    return AsyncRefineHooks(canon=canon, cached=cached, submit=submit, serial=serial, commit=commit)


def _make_batched_value_fn(refiner_cache: dict, invalids: bool):
    """Sync equivalent of the async hooks for _run_batched: value a batch by canonical key, populate the SAME
    refiner_cache with the SAME schema."""
    def batched_value_fn(token_lists: list[Tuple[int, ...]]) -> list[tuple[float, dict[str, Any]]]:
        out = []
        for t in token_lists:
            key = _canonical_key(t)
            reward, info = _key_reward(key, invalids)
            if math.isfinite(float(info.get("log_fvu", float("nan")))) and key not in refiner_cache:
                refiner_cache[key] = {"reward": reward, "fvu": info["fvu"], "log_fvu": info["log_fvu"]}
            out.append((reward, dict(info)))
        return out
    return batched_value_fn


def _cfg(**kw) -> MCTSConfig:
    # rollout_policy='sample' only to satisfy the batched-divergence guard; the MOCK batched_rollout_fn is
    # deterministic (greedy argmax) regardless, so the tree stays reproducible. The async loop uses gpu_batch,
    # not batch_width, for its micro-batch, so batch_width stays 1 here.
    base = dict(refine_budget=24, max_rollouts=4000, expansion_top_k=5, max_depth=12,
                rollout_policy="sample", backup="max", uct_c=1.4, batch_width=1)
    base.update(kw)
    return MCTSConfig(**base)


def _all_nodes(root):
    nodes, stack = [], [root]
    while stack:
        n = stack.pop()
        nodes.append(n)
        stack.extend(n.children.values())
    return nodes


def _run_async_tree(cfg, invalids=False, latency=False):
    refiner_cache: dict = {}
    commit_log: list = []
    policy_fn, bpf, brf = _make_batched_primitives()
    with ThreadPoolExecutor(max_workers=4) as ex:
        hooks = _make_async_hooks(refiner_cache, ex, invalids, commit_log, latency=latency)
        mcts = MonteCarloTreeSearch(policy_fn=policy_fn, value_fn=lambda t: (0.0, {}), terminal_fn=_terminal,
                                    config=cfg, eos_token_id=EOS, pad_token_id=None,
                                    canonicalize_fn=_canonical_key, batched_policy_fn=bpf,
                                    batched_rollout_fn=brf, async_hooks=hooks)
        root = mcts.run([1], progress=False)
    return mcts, root, refiner_cache, commit_log


def _run_batched_tree(cfg, invalids=False):
    refiner_cache: dict = {}
    policy_fn, bpf, brf = _make_batched_primitives()
    bvf = _make_batched_value_fn(refiner_cache, invalids)
    mcts = MonteCarloTreeSearch(policy_fn=policy_fn, value_fn=lambda t: (0.0, {}), terminal_fn=_terminal,
                                config=cfg, eos_token_id=EOS, pad_token_id=None,
                                canonicalize_fn=_canonical_key, batched_policy_fn=bpf,
                                batched_rollout_fn=brf, batched_value_fn=bvf)
    root = mcts._run_batched([1], progress=False)      # K=1 -> 1 leaf per iter (batched primitives, serial order)
    return mcts, root, refiner_cache


def test_async_identity_to_sync_at_inflight1():
    """The tightest anchor: async(inflight=1, gpu_batch=1) == sync _run_batched(batch_width=1)."""
    ca = _cfg(async_search=True, inflight=1, gpu_batch=1, batch_width=1)
    cb = _cfg(batch_width=1)
    amcts, aroot, acache, _ = _run_async_tree(ca)
    bmcts, broot, bcache = _run_batched_tree(cb)

    assert amcts._n_distinct_valid == bmcts._n_distinct_valid, "distinct-valid count diverged"
    assert set(acache) == set(bcache), "refiner_cache key sets diverged"
    for k in acache:
        assert abs(acache[k]["reward"] - bcache[k]["reward"]) < 1e-12
    assert set(amcts._completions) == set(bmcts._completions), "completion key sets diverged"
    for k in amcts._completions:
        at, ar, alp = amcts._completions[k]
        bt, br, blp = bmcts._completions[k]
        assert at == bt and abs(ar - br) < 1e-12 and abs(alp - blp) < 1e-9
    # whole-tree per-node stats identical (walk both in lockstep by token path)
    a_by_tok = {n.tokens: n for n in _all_nodes(aroot)}
    b_by_tok = {n.tokens: n for n in _all_nodes(broot)}
    assert set(a_by_tok) == set(b_by_tok), "tree node sets diverged"
    for tok, an in a_by_tok.items():
        bn = b_by_tok[tok]
        assert an.visits == bn.visits, f"visits diverged at {tok}: {an.visits} vs {bn.visits}"
        assert abs(an.value_sum - bn.value_sum) < 1e-9
        assert abs(an.best_value - bn.best_value) < 1e-12


def test_async_budget_exactness_and_vloss_zero():
    """len(refiner_cache) == _n_distinct_valid == refine_budget, and all vloss == 0, across (inflight, gpu_batch)."""
    for inflight in (1, 4, 16):
        for gpu_batch in (1, 8):
            cfg = _cfg(async_search=True, inflight=inflight, gpu_batch=gpu_batch, batch_width=8)
            mcts, root, cache, _ = _run_async_tree(cfg)
            assert len(cache) == cfg.refine_budget, f"budget: {len(cache)} != {cfg.refine_budget} (inflight={inflight}, gpu_batch={gpu_batch})"
            assert mcts._n_distinct_valid == cfg.refine_budget, f"distinct-valid {mcts._n_distinct_valid} != budget"
            leaks = [n.tokens for n in _all_nodes(root) if n.vloss != 0]
            assert not leaks, f"vloss leak on {len(leaks)} nodes (inflight={inflight}, gpu_batch={gpu_batch})"


def test_async_budget_exactness_with_invalids():
    """Invalid keys consume no budget; len(refiner_cache) still reaches refine_budget (valid keys only)."""
    cfg = _cfg(async_search=True, inflight=8, gpu_batch=4, batch_width=8, refine_budget=16)
    mcts, root, cache, _ = _run_async_tree(cfg, invalids=True)
    assert len(cache) == cfg.refine_budget
    assert all(math.isfinite(v["log_fvu"]) for v in cache.values())
    assert all(n.vloss == 0 for n in _all_nodes(root))


def test_async_one_refine_per_valid_key():
    """Each VALID canonical key is committed-as-valid exactly once (submitted_keys + pending coalescing)."""
    cfg = _cfg(async_search=True, inflight=16, gpu_batch=8, batch_width=8, refine_budget=20)
    _mcts, _root, cache, commit_log = _run_async_tree(cfg, invalids=True)
    valid_commits = [k for k, ok in commit_log if ok]
    assert len(valid_commits) == len(set(valid_commits)), "a valid key was refined more than once"
    assert set(valid_commits) == set(cache), "valid commits and refiner_cache keys must coincide"


def test_async_backup_order_invariance():
    """Variable mock refine latency shuffles reap order; node stats must be identical to the no-latency run."""
    cfg = _cfg(async_search=True, inflight=16, gpu_batch=8, batch_width=8, refine_budget=20)
    m0, r0, c0, _ = _run_async_tree(cfg, latency=False)
    m1, r1, c1, _ = _run_async_tree(cfg, latency=True)
    assert set(c0) == set(c1), "out-of-order reaps changed which keys were refined"
    n0 = {n.tokens: n for n in _all_nodes(r0)}
    n1 = {n.tokens: n for n in _all_nodes(r1)}
    assert set(n0) == set(n1), "out-of-order reaps changed the tree shape"
    for tok, a in n0.items():
        b = n1[tok]
        assert a.visits == b.visits and abs(a.value_sum - b.value_sum) < 1e-9 and abs(a.best_value - b.best_value) < 1e-12
