"""flash_ansr.datagen: shard round trip, K samplers, grouped cross-attention, the packed sampler against teacher
forcing, the overlapped sampler against the serial one, and the fitting stage against the deployed FlashANSR.fit on
the same raw draws."""
import itertools
import os
import zlib

import numpy as np
import pytest
import torch

from flash_ansr.datagen import problems as PR
from flash_ansr.datagen import shards

MODEL = os.path.expanduser("~/models/flash-ansr-v25.0-T8-3M")
needs_model = pytest.mark.skipif(not os.path.isdir(MODEL), reason="T8-3M bundle not available")


def test_k_samplers():
    rng = np.random.default_rng(0)
    ks = [PR.k_sampler("log2uniform:1:512")(rng) for _ in range(2000)]
    assert set(ks) <= {2 ** j for j in range(10)} and len(set(ks)) == 10
    assert all(1 <= PR.k_sampler("loguniform:1:512")(rng) <= 512 for _ in range(200))
    assert PR.k_sampler("fixed:7")(rng) == 7
    with pytest.raises(ValueError):
        PR.k_sampler("log2uniform:3:512")


def test_raw_shard_round_trip(tmp_path):
    rng = np.random.default_rng(1)
    probs = [PR.Problem(problem_id=f"p{i}", x=rng.normal(size=(5 + i, 2 + i)), y_noisy=rng.normal(size=5 + i),
                        y=rng.normal(size=5 + i), x_val=rng.normal(size=(4, 2 + i)), y_val=rng.normal(size=4),
                        skeleton=["x1"], noise={"level": 0.1 * i}, k=1 + i) for i in range(3)]
    draws = [([[1, 15, 9, 3 + j, 2] for j in range(p.k)], [-1.5 * j for j in range(p.k)]) for p in probs]
    path = str(tmp_path / "raw.npz")
    shards.write_raw_shard(path, probs, draws)
    recs = list(shards.iter_raw_shard(path))
    for r, p, d in zip(recs, probs, draws):
        assert r["problem_id"] == p.problem_id and r["k"] == p.k
        np.testing.assert_array_equal(r["x"], p.x)
        np.testing.assert_array_equal(r["x_val"], p.x_val)
        np.testing.assert_array_equal(r["y_noisy"], p.y_noisy)
        assert r["tokens"] == d[0]
        np.testing.assert_allclose(r["log_prob"], d[1])


def test_grouped_cross_attention_matches_expanded():
    from flash_ansr.model.decoders.components import Attention
    torch.manual_seed(0)
    att = Attention(dim=32, n_heads=4).eval()
    P, R, S = 3, 5, 7
    query = torch.randn(P * R, 1, 32)
    memory = torch.randn(P, S, 32)
    with torch.no_grad():
        grouped = att.forward_static_cross(query, memory, [None])
        per_row = att.forward_static_cross(query, memory.repeat_interleave(R, 0), [None])
    torch.testing.assert_close(grouped, per_row, rtol=1e-5, atol=1e-6)


@needs_model
def test_packed_sampler_matches_teacher_forcing():
    from flash_ansr import FlashANSR
    from flash_ansr.datagen import sample as S
    w = FlashANSR.load(directory=MODEL)
    m = w.flash_ansr_model.eval()
    pp = w._prepare_prompt_prefix(emission="fittable", complexity=None)
    probs = list(itertools.islice(PR.problem_stream(MODEL, seed=3, k_spec="fixed:8"), 3))
    with torch.no_grad():
        mem = torch.cat([m._create_memory(S.data_tensor(w, p.x, p.y_noisy, "cpu")) for p in probs], 0)
        g = torch.Generator()
        g.manual_seed(0)
        seqs, sc = S.decode_pack(m, mem, 8, pp, 128, generator=g)
        err = S.verify(m, mem, seqs, sc, 8, pp, n_check=8)
    assert seqs.shape == (24, 128)
    assert err < 1e-3


@needs_model
def test_fit_stage_matches_deployed_fit(tmp_path):
    from flash_ansr import FlashANSR, SoftmaxSamplingConfig
    from flash_ansr.datagen import fit as FIT
    from flash_ansr.datagen import sample as S
    torch.manual_seed(0)
    probs = list(itertools.islice(PR.problem_stream(MODEL, seed=5, k_spec="fixed:24"), 2))
    w = FlashANSR.load(directory=MODEL, compute={"device": "cpu", "workers": 0})
    m = w.flash_ansr_model.eval()
    pp = w._prepare_prompt_prefix(emission="fittable", complexity=None)
    eos, pad = int(m.tokenizer["<eos>"]), int(m.tokenizer["<pad>"])
    draws = []
    with torch.no_grad():
        for p in probs:
            mem = m._create_memory(S.data_tensor(w, p.x, p.y_noisy, "cpu"))
            seqs, sc = S.decode_pack(m, mem, p.k, pp, 128)
            draws.append(([S.trim(r, eos, pad) for r in seqs.tolist()], sc.tolist()))
    path = str(tmp_path / "raw.npz")
    shards.write_raw_shard(path, probs, draws)
    for rec in shards.iter_raw_shard(path):
        ours = FIT.fit_problem(rec, w=w, prompt_prefix=pp)
        post = m._postprocess_sampled
        orig = m.sample_top_kp
        w.generation_config = SoftmaxSamplingConfig(batch_size=1024, max_len=128)
        m.sample_top_kp = lambda *a, **k: ((rec["tokens"], rec["log_prob"]) if k.get("return_raw") else
                                           post(rec["tokens"], rec["log_prob"], simplify=True, unique=True, valid_only=True))
        try:
            res = w.fit(rec["x"], rec["y_noisy"], seed=zlib.crc32(rec["problem_id"].encode()), draws=rec["k"])
        finally:
            m.sample_top_kp = orig
        L = res.ledger
        fitted = [i for i in range(len(L)) if L.result_index[i] is not None and L.result_index[i] >= 0]
        assert len(fitted) == ours["n_fitted"]
        kept = sorted(round(L.fvu[i], 9) for i in fitted
                      if FIT._fresh_fvu(res.predict(rec["x_val"], rank=L.result_index[i]), np.asarray(rec["y_val"])) < 1)
        assert kept == sorted(round(c["fvu"], 9) for c in ours["cand"])


@needs_model
def test_draw_map_traces_every_draw_to_its_family():
    """The per-draw map of the fitting stage: post-processing with the trace's simplify_map is byte-identical to the
    plain call; every valid draw lands in the family whose beam is its dedup class's first draw; draws of one raw
    expression share their action and family; a draw has a kept family iff some row of its draw (fit, ladder variant
    or typed-thaw re-fit, all sharing its log-prob) is kept. 12 problems at K = 48 reliably include typed-thaw rows."""
    from flash_ansr import FlashANSR
    from flash_ansr.datagen import fit as FIT
    from flash_ansr.datagen import sample as S
    torch.manual_seed(0)
    probs = list(itertools.islice(PR.problem_stream(MODEL, seed=7, k_spec="fixed:48"), 12))
    w = FlashANSR.load(directory=MODEL, compute={"device": "cpu", "workers": 0})
    m = w.flash_ansr_model.eval()
    pp = w._prepare_prompt_prefix(emission="fittable", complexity=None)
    eos, pad = int(m.tokenizer["<eos>"]), int(m.tokenizer["<pad>"])
    tok = m.tokenizer
    for p in probs:
        with torch.no_grad():
            mem = m._create_memory(S.data_tensor(w, p.x, p.y_noisy, "cpu"))
            seqs, sc = S.decode_pack(m, mem, p.k, pp, 128)
        rec = dict(problem_id=p.problem_id, k=p.k, x=p.x, y_noisy=p.y_noisy, x_val=p.x_val, y_val=p.y_val,
                   tokens=[S.trim(r, eos, pad) for r in seqs.tolist()], log_prob=sc.tolist())
        trace, smap = FIT.trace_draws(m, rec["tokens"])
        plain = m._postprocess_sampled(rec["tokens"], rec["log_prob"], simplify=True, unique=True, valid_only=True)
        mapped = m._postprocess_sampled(rec["tokens"], rec["log_prob"], simplify=True, unique=True, valid_only=True,
                                        simplify_map=smap)
        assert plain == mapped
        ours = FIT.fit_problem(rec, w=w, prompt_prefix=pp)
        st, act, fam = ours["draw_status"], ours["draw_act"], ours["draw_fam"]
        assert len(st) == len(act) == len(fam) == p.k
        assert not (st == FIT.NOT_IN_LEDGER).any()
        assert ((st == FIT.DRAW_OK).sum() == sum(t[0] == FIT.DRAW_OK for t in trace))
        body = lambda s: tuple(tok.extract_expression_from_beam(list(s))[0])  # noqa: E731
        root_body = {c["fam"]: body(c["tokens"]) for c in ours["cand"]}
        for f in root_body:                                             # a family is one draw: one beam, one log-prob
            assert len({body(c["tokens"]) for c in ours["cand"] if c["fam"] == f}) == 1
            assert len({c["log_prob"] for c in ours["cand"] if c["fam"] == f}) == 1
        first_body: dict = {}
        for d, (s, raw, key, b) in enumerate(trace):
            if s != FIT.DRAW_OK:
                assert fam[d] == -1
                continue
            first_body.setdefault(key, b)
            if fam[d] >= 0:
                assert root_body[fam[d]] == first_body[key]
        kept_bodies = set(root_body.values())
        for d, (s, raw, key, b) in enumerate(trace):
            if s == FIT.DRAW_OK:
                assert (fam[d] >= 0) == (first_body[key] in kept_bodies)
        for a in set(act[act >= 0].tolist()):
            assert len(set(fam[act == a].tolist())) == 1
            assert len({trace[d][1] for d in np.flatnonzero(act == a)}) == 1


def test_pool_front_definition():
    """Members are distinct raw expressions; dominance is family-wise; the constant model (8 bits, 0) removes an
    action it dominates entirely; an action without rows is never a member."""
    from flash_ansr.datagen import targets as TG
    # rows (bits, y): family 0 = {(10, -3)}, family 1 = {(20, -5), (12, -2)}, family 2 = {(30, -4)}, family 3 = {(9, 0.5)}
    cand_fam = np.array([0, 1, 1, 2, 3])
    b = np.array([10.0, 20.0, 12.0, 30.0, 9.0])
    y = np.array([-3.0, -5.0, -2.0, -4.0, 0.5])
    # draws: actions 0 and 1 are two raw forms of family 0; action 2 -> family 1; 3 -> family 2; 4 -> family 3;
    # action 5 has no fitted family; -1 = invalid draw
    draw_act = np.array([0, 1, 1, 2, 3, 4, 5, -1])
    draw_fam = np.array([0, 0, 0, 1, 2, 3, -1, -1])
    acts, afam, front = TG.pool_front(draw_act, draw_fam, cand_fam, b, y)
    assert acts.tolist() == [0, 1, 2, 3, 4, 5]
    # family 2 (30, -4) is dominated by family 1's (20, -5); family 3 (9, 0.5) is dominated by the constant model;
    # family 1 is not dominated: its row (20, -5) is dominated by nobody
    assert front.tolist() == [True, True, True, False, False, False]
    assert TG.y_floor(0.0) == -7.0 and TG.y_floor(0.1) == pytest.approx(-4.0) and TG.y_floor(1e-6) == -9.0


@needs_model
def test_target_shards_end_to_end(tmp_path):
    """sample (one pool of 24 draws per problem) -> fit -> targets under two K specs: every problem is recorded, its K
    is deterministic and within the pool, front weights sum to 1 where there is a front, every target is one of the
    problem's first K raw draws, and the full-pool spec gives the front of all draws."""
    from flash_ansr import FlashANSR
    from flash_ansr.datagen import fit as FIT
    from flash_ansr.datagen import sample as S
    from flash_ansr.datagen import targets as TG
    torch.manual_seed(0)
    raw_dir, fit_dir = str(tmp_path / "raw"), str(tmp_path / "fit")
    S.run(MODEL, raw_dir, n_problems=4, seed=11, k_spec="fixed:24", shard_problems=4, device="cpu", amp="fp32")
    w = FlashANSR.load(directory=MODEL, compute={"device": "cpu", "workers": 0})
    FIT._W.update(w=w, pp=w._prepare_prompt_prefix(emission="fittable", complexity=None))
    (raw_path,) = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir)]
    os.makedirs(fit_dir)
    fit_path = os.path.join(fit_dir, os.path.basename(raw_path))
    FIT._fit_shard((raw_path, fit_path))
    fz = shards.load(fit_path)
    assert "x" not in fz and "x_val" not in fz                         # the problem data stay in the raw shard
    raw = shards.load(raw_path)
    for spec in ("uniform:1:24", "fixed:24"):
        tgt_dir = str(tmp_path / spec.replace(":", "_"))
        os.makedirs(tgt_dir)
        tgt_path = os.path.join(tgt_dir, os.path.basename(raw_path))
        _, n_in, n_out, _ = TG._targets_shard((raw_path, fit_path, tgt_path, spec))
        t = shards.load(tgt_path)
        assert n_in == 4 and len(t["problem_id"]) == 4 and str(t["k_spec"]) == spec and "x" not in t
        for p, pid in enumerate(t["problem_id"]):
            k = int(t["k"][p])
            assert k == TG.assign_k(str(pid), spec) and 1 <= k <= 24
            draws = {tuple(raw["tok"][raw["tok_off"][j]:raw["tok_off"][j + 1]].tolist())
                     for j in range(raw["draw_off"][p], raw["draw_off"][p] + k)}
            for name in ("front", "ctrl"):
                s0, s1 = int(t[f"{name}_off"][p]), int(t[f"{name}_off"][p + 1])
                assert (s1 > s0) == (t["n_front"][p] > 0)
                if s1 > s0:
                    assert t[f"{name}_w"][s0:s1].sum() == pytest.approx(1.0)
                for j in range(s0, s1):
                    assert tuple(t[f"{name}_tok"][t[f"{name}_tok_off"][j]:t[f"{name}_tok_off"][j + 1]].tolist()) in draws
        if spec == "fixed:24":
            assert (t["k"] == 24).all() and n_out >= 1


def test_assign_k_is_deterministic_and_spec_specific():
    from flash_ansr.datagen import targets as TG
    ks = [TG.assign_k(f"p{i}", "uniform:1:512") for i in range(2000)]
    assert ks == [TG.assign_k(f"p{i}", "uniform:1:512") for i in range(2000)]
    assert min(ks) >= 1 and max(ks) <= 512 and 200 < np.mean(ks) < 310
    assert ks[:50] != [TG.assign_k(f"p{i}", "loguniform:1:512") for i in range(50)]
    assert all(TG.assign_k(f"p{i}", "fixed:512") == 512 for i in range(10))


@needs_model
def test_chunked_packing_any_k(tmp_path):
    """Chunked packing (groups of R rows, problems of any K mixed in one decode): every problem gets exactly K draws,
    and every draw's score equals its teacher-forced log-likelihood under ITS problem's memory."""
    from flash_ansr import FlashANSR
    from flash_ansr.datagen import sample as S
    torch.manual_seed(0)
    S.run(MODEL, str(tmp_path), n_problems=6, seed=21, k_spec="uniform:1:40", shard_problems=6, rows_target=64,
          chunk_rows=8, device="cpu", amp="fp32")
    (path,) = [os.path.join(tmp_path, f) for f in os.listdir(tmp_path)]
    recs = list(shards.iter_raw_shard(path))
    probs = list(itertools.islice(PR.problem_stream(MODEL, seed=21, k_spec="uniform:1:40"), 6))
    assert [r["k"] for r in recs] == [p.k for p in probs]
    assert len({r["k"] for r in recs}) > 1 and any(r["k"] % 8 for r in recs)
    w = FlashANSR.load(directory=MODEL)
    m = w.flash_ansr_model.eval()
    pp = w._prepare_prompt_prefix(emission="fittable", complexity=None)
    tokens0, num0 = m._resolve_generation_prefix(prompt_prefix=pp, initial_tokens=None, input_num=None)
    with torch.no_grad():
        for r, p in zip(recs, probs):
            assert len(r["tokens"]) == len(r["log_prob"]) == r["k"]
            mem = m._create_memory(S.data_tensor(w, p.x, p.y_noisy, "cpu"))
            tf = S._teacher_forced(m, mem, r["tokens"], len(tokens0), num0)
            np.testing.assert_allclose(tf, r["log_prob"], atol=1e-3)


@needs_model
def test_parallel_fit_equals_serial(tmp_path):
    """fit.run (problem-level pool, shards written as they complete) writes the same fitted shards as the serial
    in-process path, and returns instead of hanging (an exception in the loop must not block Pool.terminate)."""
    from flash_ansr import FlashANSR
    from flash_ansr.datagen import fit as FIT
    from flash_ansr.datagen import sample as S
    torch.manual_seed(0)
    raw = str(tmp_path / "raw")
    S.run(MODEL, raw, n_problems=4, seed=31, k_spec="fixed:8", shard_problems=2, device="cpu", amp="fp32")
    FIT.run(raw, str(tmp_path / "par"), MODEL, workers=2, n_restarts=2, shards_in_flight=1)
    w = FlashANSR.load(directory=MODEL, refine={"n_restarts": 2}, compute={"device": "cpu", "workers": 0})
    FIT._W.update(w=w, pp=w._prepare_prompt_prefix(emission="fittable", complexity=None))
    os.makedirs(tmp_path / "ser")
    for f in sorted(os.listdir(raw)):
        FIT._fit_shard((os.path.join(raw, f), str(tmp_path / "ser" / f)))
        a, b = shards.load(str(tmp_path / "par" / f)), shards.load(str(tmp_path / "ser" / f))
        assert set(a) == set(b)
        for k in a:
            if k in ("t_refine", "t_post"):
                continue
            np.testing.assert_array_equal(a[k], b[k], err_msg=k)


@needs_model
def test_overlapped_sampler_equals_serial(tmp_path):
    """The overlapped sampler (problems from a spawned producer process, shards written by a background thread)
    writes byte-for-byte the same raw shards as the serial path; a resumed run skips the shards on disk."""
    from flash_ansr.datagen import sample as S
    kw = dict(n_problems=5, seed=41, k_spec="uniform:1:24", shard_problems=2, rows_target=32, chunk_rows=8,
              device="cpu", amp="fp32")
    S.run(MODEL, str(tmp_path / "ser"), overlap=False, **kw)
    S.run(MODEL, str(tmp_path / "ovl"), overlap=True, **kw)
    names = sorted(os.listdir(tmp_path / "ser"))
    assert len(names) == 3 and names == sorted(os.listdir(tmp_path / "ovl"))
    for f in names:
        a, b = shards.load(str(tmp_path / "ser" / f)), shards.load(str(tmp_path / "ovl" / f))
        assert set(a) == set(b)
        for k in a:
            np.testing.assert_array_equal(a[k], b[k], err_msg=f"{f}: {k}")
    # resume: delete the middle shard; a rerun rewrites exactly it, the others are skipped and stay untouched
    os.remove(tmp_path / "ovl" / names[1])
    t_other = os.path.getmtime(tmp_path / "ovl" / names[0])
    S.run(MODEL, str(tmp_path / "ovl"), overlap=True, **kw)
    assert os.path.getmtime(tmp_path / "ovl" / names[0]) == t_other
    a, b = shards.load(str(tmp_path / "ser" / names[1])), shards.load(str(tmp_path / "ovl" / names[1]))
    for k in a:
        np.testing.assert_array_equal(a[k], b[k], err_msg=f"resumed {names[1]}: {k}")


def test_negatives_dominance_layers_and_V():
    """Hand-made pool: family-wise dominance, V = (-inf bits, 0) over FVU >= 1-only actions, Kahn layers = repeated
    peeling, and the pair rules (y+ dominates y-'s best row; a draw only V dominates is dropped)."""
    from flash_ansr.datagen import negatives as NG
    from flash_ansr.datagen.fit import DRAW_OK, INVALID
    # rows: (family, bits, fvu_val)
    rows = [(0, 5.0, 10 ** -2.0), (1, 10.0, 10 ** -3.0), (2, 12.0, 10 ** -1.0), (2, 9.0, 10 ** -0.5),
            (3, 7.0, 10 ** 0.2), (4, 3.0, 10 ** 0.5)]
    cand_fam = np.array([r[0] for r in rows]); cand_mdl = np.array([r[1] * 1000 for r in rows])
    cand_fvu = np.array([r[2] for r in rows]); cand_score = np.zeros(len(rows))
    # draws: actions 0..4 (family = action), plus an invalid draw and a repeat of action 3
    draw_act = np.array([0, 1, 2, 3, 4, -1, 3]); draw_fam = np.array([0, 1, 2, 3, 4, -1, 3])
    draw_status = np.array([DRAW_OK] * 5 + [INVALID, DRAW_OK]); draw_gt = np.zeros(7, bool)
    usable = np.ones(len(rows), bool)
    acts, R = NG.action_rows(draw_status, draw_act, draw_fam, cand_fam, cand_fvu, usable)
    b = cand_mdl / 1000.0; y = np.log10(cand_fvu)
    FD, M, vdom = NG.dominance(R, b, y)
    assert list(vdom) == [False, False, False, True, True]
    # family 2's rows (12, -1) and (9, -0.5): (10, -3) covers only the first (it is longer than 9 bits), so family 1
    # does not dominate family 2; (5, -2) covers both, so family 0 does
    assert not FD[1, 2] and FD[0, 2]
    L = NG.layers(FD, vdom)
    # naive peeling for comparison, V in layer 1
    rem, naive, lay = set(range(5)), {}, 1
    while rem:
        top = {x for x in rem if not any(FD[z, x] for z in rem) and not (lay == 1 and vdom[x])}
        for x in top:
            naive[x] = lay
        rem -= top
        lay += 1
    assert [naive[i] for i in range(5)] == list(L) == [1, 1, 2, 2, 2]
    tok = {d: (1, 9, 100 + d, 2) for d in range(7)}
    out = NG.problem_negatives("p", 7, "fixed:7", lambda d: tok[d], draw_status, draw_act, draw_fam, draw_gt, cand_fam,
                               cand_mdl, cand_fvu, cand_score, 0.0, pairs=8)
    assert out["n_front_k"] == 2 and out["n_invalid"] == 1
    assert out["n_drop_vonly"] == 1                    # action 4 (3 bits, FVU > 1): only V dominates it
    negs = sorted(p["neg"][2] - 100 for p in out["pairs"])
    assert negs == [2, 3, 6]                           # draws of actions 2 and 3 (twice); never 4, never the members
    for p in out["pairs"]:
        assert p["pos"][2] - 100 in (0, 1)
        assert p["pos_dom_b"] <= p["neg_b"] and p["pos_dom_y"] <= p["neg_y"]
        assert (p["pos_dom_b"] < p["neg_b"]) or (p["pos_dom_y"] < p["neg_y"])
        if p["neg"][2] - 100 in (3, 6):
            assert p["neg_kind"] == 1 and p["pos"][2] - 100 == 0   # only (5 bits) is no longer than 7 bits


@needs_model
def test_keep_all_rows_refit_and_negatives(tmp_path):
    """fit with keep_all_rows stores the same FVU < 1 rows as the default fit (plus FVU >= 1 ones), the front
    targets built from either shard are identical, and the negatives built from it satisfy the pair invariants and
    agree with the front targets' front."""
    from flash_ansr import FlashANSR
    from flash_ansr.datagen import fit as FIT
    from flash_ansr.datagen import negatives as NG
    from flash_ansr.datagen import sample as S
    from flash_ansr.datagen import targets as TG
    torch.manual_seed(0)
    raw = str(tmp_path / "raw")
    S.run(MODEL, raw, n_problems=4, seed=51, k_spec="fixed:24", shard_problems=4, device="cpu", amp="fp32")
    w = FlashANSR.load(directory=MODEL, refine={"n_restarts": 2}, compute={"device": "cpu", "workers": 0})
    FIT._W.update(w=w, pp=w._prepare_prompt_prefix(emission="fittable", complexity=None))
    (name,) = os.listdir(raw)
    for tag, keep in (("def", False), ("all", True)):
        os.makedirs(tmp_path / tag)
        FIT._W["keep_all"] = keep
        FIT._fit_shard((os.path.join(raw, name), str(tmp_path / tag / name)))
    FIT._W["keep_all"] = False
    a, b = shards.load(str(tmp_path / "def" / name)), shards.load(str(tmp_path / "all" / name))
    assert bool(b["rows_all"]) and not bool(a["rows_all"])
    assert (b["n_rows"] >= b["n_kept"]).all() and (a["n_kept"] == b["n_kept"]).all()
    lt1 = b["fvu_val"] < 1.0
    key = lambda z, m: sorted(zip(z["log_prob_draw"][m].round(5), z["mdl"][m], z["fvu_val"][m]))  # noqa: E731
    assert key(a, np.ones(len(a["mdl"]), bool)) == key(b, lt1)
    assert (b["fvu_val"] >= 1.0).any(), "expected some FVU >= 1 rows in the 3M pools"
    for tag in ("def", "all"):
        TG._targets_shard((os.path.join(raw, name), str(tmp_path / tag / name), str(tmp_path / f"t_{tag}.npz"), "fixed:24"))
    ta, tb = shards.load(str(tmp_path / "t_def.npz")), shards.load(str(tmp_path / "t_all.npz"))
    for k in ("n_front", "front_off", "front_tok", "front_tok_off", "front_w", "ctrl_tok", "n_kept_actions"):
        np.testing.assert_array_equal(ta[k], tb[k], err_msg=k)
    NG._negatives_shard((os.path.join(raw, name), str(tmp_path / "all" / name), str(tmp_path / "n.npz"), "fixed:24", 8))
    ng = shards.load(str(tmp_path / "n.npz"))
    np.testing.assert_array_equal(ng["n_front_k"], tb["n_front"])
    assert ng["n_pairs"].sum() > 0
    rw = shards.load(os.path.join(raw, name))
    for p in range(len(ng["problem_id"])):
        d0, d1 = int(rw["draw_off"][p]), int(rw["draw_off"][p + 1])
        drawn = {tuple(rw["tok"][rw["tok_off"][d]:rw["tok_off"][d + 1]].tolist()) for d in range(d0, d1)}
        for i in range(int(ng["pair_off"][p]), int(ng["pair_off"][p + 1])):
            pos = tuple(ng["pos_tok"][ng["pos_tok_off"][i]:ng["pos_tok_off"][i + 1]].tolist())
            neg = tuple(ng["neg_tok"][ng["neg_tok_off"][i]:ng["neg_tok_off"][i + 1]].tolist())
            assert pos in drawn and neg in drawn and pos != neg
            assert ng["pos_dom_b"][i] <= ng["neg_b"][i] + 1e-6 and ng["pos_dom_y"][i] <= ng["neg_y"][i] + 1e-6
            assert ng["layer"][i] >= 2
