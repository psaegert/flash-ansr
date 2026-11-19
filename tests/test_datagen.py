"""flash_ansr.datagen: shard round trip, K samplers, grouped cross-attention, the packed sampler against teacher
forcing, and the fitting stage against the deployed FlashANSR.fit on the same raw draws."""
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
    """sample -> fit -> targets on a few problems: front weights sum to 1 per problem, every target sequence is one of
    the problem's raw draws, and the control has as many members as the front (or all kept actions)."""
    from flash_ansr import FlashANSR
    from flash_ansr.datagen import fit as FIT
    from flash_ansr.datagen import sample as S
    from flash_ansr.datagen import targets as TG
    raw_dir, fit_dir, tgt_dir = (str(tmp_path / d) for d in ("raw", "fit", "tgt"))
    S.run(MODEL, raw_dir, n_problems=4, seed=11, k_spec="fixed:24", shard_problems=4, device="cpu", amp="fp32")
    w = FlashANSR.load(directory=MODEL, compute={"device": "cpu", "workers": 0})
    FIT._W.update(w=w, pp=w._prepare_prompt_prefix(emission="fittable", complexity=None))
    (raw_path,) = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir)]
    fit_path = os.path.join(fit_dir, os.path.basename(raw_path))
    os.makedirs(fit_dir)
    os.makedirs(tgt_dir)
    FIT._fit_shard((raw_path, fit_path))
    tgt_path = os.path.join(tgt_dir, os.path.basename(raw_path))
    _, n_in, n_out, _ = TG._targets_shard((raw_path, fit_path, tgt_path))
    assert n_in == 4 and 1 <= n_out <= 4
    raw, t = shards.load(raw_path), shards.load(tgt_path)
    pid_raw = list(raw["problem_id"])
    for i, pid in enumerate(t["problem_id"]):
        p = pid_raw.index(pid)
        draws = {tuple(raw["tok"][raw["tok_off"][j]:raw["tok_off"][j + 1]].tolist())
                 for j in range(raw["draw_off"][p], raw["draw_off"][p + 1])}
        for name in ("front", "ctrl"):
            s0, s1 = int(t[f"{name}_off"][i]), int(t[f"{name}_off"][i + 1])
            assert s1 > s0
            assert t[f"{name}_w"][s0:s1].sum() == pytest.approx(1.0)
            for j in range(s0, s1):
                assert tuple(t[f"{name}_tok"][t[f"{name}_tok_off"][j]:t[f"{name}_tok_off"][j + 1]].tolist()) in draws
        np.testing.assert_array_equal(shards.unpack(t["y_noisy"], t["y_off"], i), shards.unpack(raw["y_noisy"], raw["y_off"], p))
        assert t["n_front"][i] >= 1 and t["n_kept_actions"][i] >= 1


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
