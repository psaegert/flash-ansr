"""Pure-function regression tests for the refinement-determinism fixes.

These validate EXACTLY the two things the reproducibility work changed, with no GPU and no
curve_fit (whose convergence-boundary noise is inherent and not what we changed):

  1. ``_candidate_refine_seed`` assigns an INTRINSIC seed (a function of the candidate's tokens,
     not its position in the job list), so the same candidate always gets the same p0-noise seed
     regardless of ordering / completion order / a dropped neighbour. This is what makes a
     re-scheduled (overlapped) refinement reproduce the serial result candidate-for-candidate.
  2. The ``compile_results`` sort gains an expression-tuple tie-break, making the final ranking a
     TOTAL order: exact-score ties resolve deterministically instead of keeping the (parallel,
     non-deterministic) completion order.
"""
import random

import numpy as np

from flash_ansr.flash_ansr import _candidate_refine_seed


def test_candidate_refine_seed_is_intrinsic_and_position_independent():
    toks_a = ['*', '<constant>', 'x1', 'x2']
    toks_b = ['+', 'x3', 'pow2', 'x4']

    # Deterministic: same tokens + same refine_seed -> same seed, every time.
    assert _candidate_refine_seed(42, toks_a) == _candidate_refine_seed(42, toks_a)

    # Intrinsic to the tokens: distinct candidates get distinct seeds (w.h.p.).
    assert _candidate_refine_seed(42, toks_a) != _candidate_refine_seed(42, toks_b)

    # Depends on the problem seed: different refine_seed -> different per-candidate seed.
    assert _candidate_refine_seed(42, toks_a) != _candidate_refine_seed(43, toks_a)

    # Position-independent: a candidate's seed does not depend on where it sits in the job list,
    # so dropping/adding a neighbour (CUDA multinomial is not bit-reproducible) cannot cascade-shift
    # other candidates' seeds the way a positional SeedSequence().spawn(len(jobs)) would.
    cands = [toks_a, toks_b, ['sin', 'x1'], ['/', 'x2', 'x3'], ['<constant>']]
    seeds_forward = {tuple(c): _candidate_refine_seed(7, c) for c in cands}
    seeds_reversed = {tuple(c): _candidate_refine_seed(7, c) for c in reversed(cands)}
    seeds_dropped = {tuple(c): _candidate_refine_seed(7, c) for c in cands if c is not toks_b}
    assert seeds_forward == seeds_reversed
    for c in cands:
        if c is not toks_b:
            assert seeds_dropped[tuple(c)] == seeds_forward[tuple(c)]

    # raw_beam token ids (ints) are hashable as a key too (the deployed key is the raw_beam).
    assert _candidate_refine_seed(1, [12, 5, 99, 3]) == _candidate_refine_seed(1, [12, 5, 99, 3])
    assert _candidate_refine_seed(1, [12, 5, 99, 3]) != _candidate_refine_seed(1, [12, 5, 99, 4])


def test_candidate_refine_seed_none_is_fresh_entropy():
    # Legacy behaviour: refine_seed=None -> fresh OS entropy (not reproducible).
    seeds = {_candidate_refine_seed(None, ['x1']) for _ in range(8)}
    assert len(seeds) > 1


def _compile_results_sort_key(x):
    # Mirror of the sort key in FlashANSR.compile_results (flash_ansr.py): (score|inf, isnan, expr).
    score = x['score']
    return (
        score if not np.isnan(score) else float('inf'),
        np.isnan(score),
        tuple(map(str, x.get('expression', []))),
    )


def test_compile_results_tiebreak_is_total_order():
    # Distinct expressions, with EXACT score ties (incl. NaN) that the old 2-key sort left in
    # insertion (parallel-completion) order. The expression tie-break makes the order total.
    base = [
        {'score': 1.0, 'expression': ['a']},
        {'score': 1.0, 'expression': ['b']},
        {'score': 0.5, 'expression': ['c']},
        {'score': float('nan'), 'expression': ['d']},
        {'score': 1.0, 'expression': ['a', 'a']},
        {'score': float('nan'), 'expression': ['e']},
        {'score': 0.5, 'expression': ['c', 'x']},
    ]
    reference = [r['expression'] for r in sorted(base, key=_compile_results_sort_key)]
    rng = random.Random(0)
    for _ in range(50):
        shuffled = base[:]
        rng.shuffle(shuffled)
        out = [r['expression'] for r in sorted(shuffled, key=_compile_results_sort_key)]
        assert out == reference, "sort must be independent of input (completion) order"

    # Best (lowest finite score) first; all NaN-score entries last.
    assert reference[0] == ['c']
    assert reference[-1] in (['d'], ['e'])
