"""Tests for the save-all candidate store + ledger join (STANDARD_EVAL.md item 5)."""
import json
import types
import warnings as _warnings

import numpy as np
import pytest

from flash_ansr.eval.candidate_store import (
    CandidateStoreReader,
    CandidateStoreWriter,
    FIT_FAILED,
    FIT_OK,
    INVALID,
    build_candidate_ledger,
)
from flash_ansr.eval.model_adapters import FlashANSRAdapter


def test_build_candidate_ledger_join_classifies_every_candidate():
    # gen pool: A,B,C,D (B,D NOT refined). results: A,C fitted + pruned variant E (not in gen pool).
    A, B, C, D, E = [1, 2, 3], [4, 5], [6, 7, 8], [9], [2, 2, 2]
    raw_beams = [A, B, C, D]
    log_probs = [-1.0, -2.0, -3.0, -4.0]
    results = [
        {"raw_beam": A, "fvu": 0.10, "log_prob": -1.0, "fits": [(np.array([1.5, 2.5]), None, 0.10)]},
        {"raw_beam": C, "fvu": 0.50, "log_prob": -3.0, "fits": [(np.array([9.0]), None, 0.50)]},
        {"raw_beam": E, "fvu": 0.05, "log_prob": -9.0, "fits": [(np.array([]), None, 0.05)]},  # pruned variant
    ]
    valid_map = {tuple(B): True, tuple(D): False}  # B is a valid skeleton that failed to fit; D is invalid
    ledger = build_candidate_ledger(
        raw_beams, log_probs, results,
        decode_expr=lambda rb: list(rb),            # identity stand-in
        is_valid=lambda toks: valid_map[tuple(toks)],
    )

    # 5 candidates total: the 4 gen + the 1 pruned variant, gen-order first
    assert ledger["token_lists"] == [A, B, C, D, E]
    np.testing.assert_array_equal(ledger["fit_status"], [FIT_OK, FIT_FAILED, FIT_OK, INVALID, FIT_OK])
    np.testing.assert_array_equal(ledger["valid"], [1, 1, 1, 0, 1])
    # fvu: A,C,E from results; B,D NaN
    fvu = np.array(ledger["fvu"], dtype=np.float64)
    np.testing.assert_allclose(fvu[[0, 2, 4]], [0.10, 0.50, 0.05])
    assert np.isnan(fvu[1]) and np.isnan(fvu[3])
    assert ledger["constants"][0] == [1.5, 2.5] and ledger["constants"][3] == []  # fitted vs not


def test_ledger_round_trips_through_writer_reader(tmp_path):
    A, B = [1, 2, 3], [4, 5]
    ledger = build_candidate_ledger(
        [A, B], [-1.0, -2.0],
        [{"raw_beam": A, "fvu": 0.2, "log_prob": -1.0, "fits": [(np.array([7.0]), None, 0.2)]}],
        decode_expr=lambda rb: list(rb), is_valid=lambda toks: True,
    )
    w = CandidateStoreWriter(tmp_path, vocab_size=83)
    w.write_problem(0, **ledger)
    w.close()
    block = next(iter(CandidateStoreReader(tmp_path)))
    np.testing.assert_array_equal(CandidateStoreReader.candidate_tokens(block, 0), np.array(A, np.uint8))
    np.testing.assert_array_equal(block["fit_status"], [FIT_OK, FIT_FAILED])
    assert np.isnan(block["fvu"][1])


def test_writer_resume_rebuilds_index_and_skips_done(tmp_path):
    w1 = CandidateStoreWriter(tmp_path, vocab_size=83)
    w1.write_problem(0, [[1, 2]], [0.1], [-1.0])
    w1.write_problem(1, [[3, 4]], [0.2], [-2.0])
    w1.close()

    # fresh writer over the same dir (a resume): index rebuilt from disk, done problems skippable
    w2 = CandidateStoreWriter(tmp_path, vocab_size=83)
    assert w2.has_problem(0) and w2.has_problem(1) and not w2.has_problem(2)
    w2.write_problem(2, [[5, 6]], [0.3], [-3.0])
    man = w2.close()
    assert man["n_problems"] == 3                      # manifest covers ALL, not just the resume slice
    assert {p["problem_id"] for p in man["problems"]} == {0, 1, 2}


def test_reader_globs_authoritatively_over_stale_manifest(tmp_path):
    w = CandidateStoreWriter(tmp_path, vocab_size=83)
    for i in range(3):
        w.write_problem(i, [[i, i + 1]], [0.1 * i], [-1.0 * i])
    w.close()
    # corrupt the manifest to claim only problem 0 -- the reader must still find all three on disk
    (tmp_path / "manifest.json").write_text(json.dumps({"problems": [{"problem_id": 0}]}))
    assert CandidateStoreReader(tmp_path).problem_ids() == [0, 1, 2]
    assert len(list(CandidateStoreReader(tmp_path))) == 3


# --- adapter capture (mock model; no GPU) -------------------------------------------------
class _Tok:
    # __len__ must live on the type (dunder lookup skips instances), so a class not SimpleNamespace
    def extract_expression_from_beam(self, beam):
        return (list(beam), [], [])

    def decode(self, ids, special_tokens=None):
        return [str(i) for i in ids]

    def __len__(self):
        return 83


def _mock_model():
    engine = types.SimpleNamespace(is_valid=lambda toks: toks != ["3", "4"])  # [3,4] is "invalid"
    return types.SimpleNamespace(tokenizer=_Tok(), simplipy_engine=engine)


def test_adapter_capture_writes_ledger_keyed_by_row_index(tmp_path):
    adapter = FlashANSRAdapter(_mock_model(), candidate_store_dir=str(tmp_path))
    gen_state = types.SimpleNamespace(raw_beams=[[1, 2], [3, 4]], log_probs=[-1.0, -2.0])
    fit_result = types.SimpleNamespace(
        results=[{"raw_beam": [1, 2], "fvu": 0.1, "log_prob": -1.0, "fits": [(np.array([2.0]), None, 0.1)]}]
    )
    adapter._capture_candidate_ledger({"eval_row_index": 5}, gen_state, fit_result)

    assert (tmp_path / "problem_000005.npz").exists()
    block = next(iter(CandidateStoreReader(tmp_path)))
    np.testing.assert_array_equal(block["fit_status"], [FIT_OK, INVALID])  # [1,2] fitted; [3,4] invalid
    np.testing.assert_array_equal(block["valid"], [1, 0])

    # resume: a second capture of the same row is a no-op (file already present)
    adapter._capture_candidate_ledger({"eval_row_index": 5}, gen_state, fit_result)
    assert CandidateStoreReader(tmp_path).problem_ids() == [5]


def test_adapter_capture_skips_without_row_index(tmp_path):
    adapter = FlashANSRAdapter(_mock_model(), candidate_store_dir=str(tmp_path))
    gen_state = types.SimpleNamespace(raw_beams=[[1, 2]], log_probs=[-1.0])
    fit_result = types.SimpleNamespace(results=[])
    with pytest.warns(RuntimeWarning, match="eval_row_index"):
        adapter._capture_candidate_ledger({}, gen_state, fit_result)
    assert not list(tmp_path.glob("problem_*.npz"))


def test_adapter_capture_is_best_effort_on_error(tmp_path):
    adapter = FlashANSRAdapter(_mock_model(), candidate_store_dir=str(tmp_path))
    broken = types.SimpleNamespace(raw_beams=None, log_probs=None)  # will blow up inside the join
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        adapter._capture_candidate_ledger({"eval_row_index": 0}, broken, types.SimpleNamespace(results=[]))
    # no crash; nothing written
    assert not list(tmp_path.glob("problem_*.npz"))
