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
