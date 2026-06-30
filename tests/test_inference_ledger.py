"""Unit tests for the candidate-ledger join (no model needed)."""
import numpy as np

from flash_ansr.inference import (
    FIT_FAILED,
    FIT_OK,
    INVALID,
    CandidateLedger,
    _best_constants,
    build_candidate_ledger,
)


def test_build_candidate_ledger_classifies_and_rides_pruned():
    raw_beams = [[1, 2], [3, 4], [5, 6], [7, 8]]      # the generation pool
    log_probs = [-0.1, -0.2, -0.3, -0.4]
    results = [                                        # refined survivors (keyed by raw_beam)
        {"raw_beam": [1, 2], "fvu": 0.01, "log_prob": -0.1, "fits": [(np.array([1.5, 2.5]), None, 0.01)]},
        {"raw_beam": [9, 9], "fvu": 0.05, "log_prob": -0.9, "fits": [(np.array([3.0]), None, 0.05)]},  # pruned (not in pool)
    ]
    valid_set = {(3, 4), (7, 8)}                       # [5,6] is the only invalid not-fitted beam
    led = build_candidate_ledger(
        raw_beams, log_probs, results,
        decode_expr=lambda ids: list(ids),
        is_valid=lambda toks: tuple(toks) in valid_set,
    )
    assert isinstance(led, CandidateLedger) and len(led) == 5   # 4 gen + 1 pruned ride-along
    by_tok = {tuple(t): i for i, t in enumerate(led.token_lists)}

    i = by_tok[(1, 2)]                                 # fitted gen candidate -> FIT_OK
    assert led.fit_status[i] == FIT_OK and led.valid[i] == 1
    assert led.fvu[i] == 0.01 and led.constants[i] == [1.5, 2.5]

    for k in [(3, 4), (7, 8)]:                          # valid, not fitted -> FIT_FAILED
        j = by_tok[k]
        assert led.fit_status[j] == FIT_FAILED and led.valid[j] == 1
        assert np.isnan(led.fvu[j]) and led.constants[j] == []

    j = by_tok[(5, 6)]                                  # invalid, not fitted -> INVALID
    assert led.fit_status[j] == INVALID and led.valid[j] == 0 and np.isnan(led.fvu[j])

    j = by_tok[(9, 9)]                                  # pruned variant (not in pool) -> FIT_OK ride-along
    assert led.fit_status[j] == FIT_OK and led.fvu[j] == 0.05 and led.constants[j] == [3.0]


def test_best_constants_picks_lowest_loss():
    r = {"fits": [(np.array([1.0]), None, 0.5), (np.array([2.0]), None, 0.1), (np.array([3.0]), None, 0.9)]}
    assert _best_constants(r) == [2.0]
    assert _best_constants({"fits": []}) == []
