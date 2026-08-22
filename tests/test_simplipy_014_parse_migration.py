"""simplipy 0.14 removed ``SimpliPyEngine.parse``. Its replacement per the 0.14
migration table is ``read_infix`` -- the SAME raw reader under the name that states
its contract ("renamed from ``parse``", simplipy ruling 2026-08-18), with the same
``mask_numbers=`` flag routed to the same internal masking. Both call sites are
MIGRATED as a rename (``convert_data._process_expression`` and the
``flash_ansr_model`` sympy post-processing site), and the byte-identity proof below
runs over the same frozen corpora that once recorded the sites as blocked.

THE HISTORY THIS FILE CARRIES. An earlier revision (branch
``compat/simplipy-0.14-parse``) measured ``engine.mask(to_prefix(s))`` as the
candidate and recorded both sites BLOCKED on three axes -- policy (the float()-probe
predicate is neither ``mask_all`` nor ``mask_fittable``), collect (``engine.mask``
always collects and reshapes), and canonicalisation (``to_prefix`` folds, re-spells
and refuses ``sqrt``). Those measurements were sound; the candidate was wrong. The
distinction tests that survive the 0.14 alias removals are kept below; the full
blocked-era record lives in that branch's history.

The removed method was a pure delegation to the Rust raw reader
(``self._core.parse``); ``read_infix`` delegates to the same reader. The oracle
tests compare the two spellings in one process via ``_core.parse`` -- its subject IS
the removed behaviour; production code must not use it.
"""
import os

import pytest
import yaml
from simplipy import SimpliPyEngine, masking

from flash_ansr.utils.skeleton import simplify_and_mask
from flash_ansr import get_path


FASTSRB = get_path('data', 'ansr-data', 'test_set', 'fastsrb', 'expressions.yaml')

# Infix spellings the FastSRB `prepared` column cannot exercise (it spells pi as the
# decimal 3.1415926535897), each discriminating one policy axis.
DISCRIMINATING = [
    'pi * x1 + 2',
    'e ** x1',
    'pi * e * x1',
    'x1 ** 3 + 2 * x1',
    'x1 ** 2 / 3',
    '2.5 * x1 + 0.5',
    '1e-05 * x1',
    '-2 * x1 - 3',
    'log(2 * x1) + exp(0.5 * x1)',
    'pi * x1 ** 2 + 2.5 * x2',
]

# Real `simplified_infix` values at the flash_ansr_model.py post-processing site,
# captured from the site's own pipeline: beam tokens -> prefix_to_infix(power='**') ->
# sympy.simplify -> Abs->abs. Frozen so the test needs neither sympy nor a trained model.
SYMPY_OUTPUTS = [
    '2/(x1*log(2))', 'x1 + x2*sin(x1)', 'cos(2/x0)', '0.5**(0.5/x1**2)',
    '33.1154519586923', '-0.500000000000000', '-4/(x1 - x2)', 'tanh(exp(sin(3)))',
    '3.50000000000000', 'atan(sin(x1))',
    'sin(3*x1)**0.5', '1/x0', 'x0/2 + x1/2', '-x0 + x1', 'log(x2**(-3))',
    '3*x2/atan(2) + 1/x1',
]


@pytest.fixture(scope='module')
def engine() -> SimpliPyEngine:
    return SimpliPyEngine.load('acj-4-3', install=True)


@pytest.fixture(scope='module')
def fastsrb_corpus() -> list[tuple[str, str]]:
    """The real convert_data corpus: FastSRB `prepared` expressions, with the
    FastSRBParser's own ``^`` -> ``**`` normalisation applied."""
    if not os.path.exists(FASTSRB):
        pytest.skip('FastSRB benchmark fixture not downloaded')
    with open(FASTSRB) as file:
        document = yaml.safe_load(file)
    return [(eq_id, entry['prepared'].replace('^', '**'))
            for eq_id, entry in document.items()
            if isinstance(entry.get('prepared'), str) and entry['prepared'].strip()]


def old_parse(engine: SimpliPyEngine, expression: str, mask_numbers: bool = False) -> list[str]:
    """The removed ``engine.parse``, verbatim: it delegated to this reader and nothing else."""
    return engine._core.parse(expression, True, mask_numbers)


# --------------------------------------------------------------------------------------
# THE MIGRATION, PROVEN: read_infix IS the removed reader, both mask flags
# --------------------------------------------------------------------------------------

def test_read_infix_is_the_removed_reader_corpus_wide(engine: SimpliPyEngine, fastsrb_corpus: list[tuple[str, str]]) -> None:
    """Byte identity against the raw-reader oracle, under BOTH mask flags, over the
    real FastSRB corpus, the discriminating spellings, and every frozen SymPy output --
    the whole input language of both migrated sites. All three blocked-era axes were
    properties of the ``mask(to_prefix(...))`` candidate; the rename has none of them."""
    corpus = ([expression for _, expression in fastsrb_corpus]
              + DISCRIMINATING + SYMPY_OUTPUTS)
    assert len(corpus) > 100
    for expression in corpus:
        assert engine.read_infix(expression) == old_parse(engine, expression), expression
        assert (engine.read_infix(expression, mask_numbers=True)
                == old_parse(engine, expression, mask_numbers=True)), expression


def test_convert_data_attrition_semantics_are_unchanged(engine: SimpliPyEngine) -> None:
    """The site's designed attrition survives: an out-of-vocabulary name (``sqrt``,
    26 of the 120 FastSRB entries) passes through the reader as a leaf, fails
    ``is_valid``, and is COUNTED AND SKIPPED -- never an abort, which is what
    ``to_prefix``'s refusal would have forced at ``skip_unparseable=False``."""
    passed_through = engine.read_infix('sqrt(v1 * v2 / v3)')
    assert passed_through == ['sqrt', '/', '*', 'v1', 'v2', 'v3']
    assert engine.is_valid(passed_through) is False


def test_convert_data_artifact_is_byte_identical(engine: SimpliPyEngine) -> None:
    """The blocked-era acceptance criterion, now green: the imported skeleton --
    masked parse through the site's own simplify_and_mask -- is unchanged on the
    specimen expressions whose degree-of-freedom counts the wrong candidate moved."""
    cases = [
        ('1 / (4 * 3.1415926535897 * 8.854e-12 * 2.99792458e8 ** 2) * 2 * v1 / v2', 3),
        ('6.67430e-11 * v1 * v2 * (1 / v3 - 1 / v4)', 3),
        ('1 / (1 / v1 + v2 / v3)', 2),
    ]
    for expression, n_free in cases:
        old_artifact = simplify_and_mask(engine, old_parse(engine, expression, mask_numbers=True))
        new_artifact = simplify_and_mask(engine, engine.read_infix(expression, mask_numbers=True))
        assert new_artifact == old_artifact, expression
        assert new_artifact.count('<constant>') == n_free, (expression, new_artifact)


# --------------------------------------------------------------------------------------
# THE DISTINCTION, KEPT: what mask_numbers=True is, and is not
# --------------------------------------------------------------------------------------

def test_mask_numbers_true_is_not_mask_fittable(engine: SimpliPyEngine) -> None:
    """``fittable`` KEEPS what ``mask_numbers=True`` masks: the structural exponent and
    the root index. The reader flag and the training-data policy are different masks,
    which is why the sites keep the flag rather than adopting a policy."""
    raw = engine.read_infix('x1 ** 3 + 2 * x1')
    assert engine.read_infix('x1 ** 3 + 2 * x1', mask_numbers=True) == [
        '+', 'pow', 'x1', '<constant>', '*', '<constant>', 'x1']
    assert masking.mask(list(raw), engine, masking.mask_fittable, collect=False) == [
        '+', 'pow', 'x1', '3', '*', '<constant>', 'x1']


def test_engine_mask_collects_and_reshapes(engine: SimpliPyEngine) -> None:
    """``engine.mask`` has no ``collect=False`` escape, and the collect stage re-orders
    and re-shapes -- one of the three axes that disqualified ``mask(to_prefix(...))``."""
    expression = 'v1 / (2 * (1 + v2))'
    assert engine.read_infix(expression, mask_numbers=True) == [
        '/', 'v1', '*', '<constant>', '+', '<constant>', 'v2']
    assert engine.mask(engine.read_infix(expression), policy='all') == [
        '/', 'v1', '*', '<constant>', '+', 'v2', '<constant>']
