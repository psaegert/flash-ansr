"""Guards that make a numeric-width mismatch loud instead of silent.

Written ahead of the f64 + byte-token migration. `Tensor.view(dtype)` does not convert, it
REINTERPRETS and resizes the last dimension -- 18 float32 values viewed as int64 and expanded
to 64 bits each is 576, exactly the width a 32-bit pre-encoder produces. Without these guards
a half-migrated stack yields correctly-shaped tensors of scrambled bits and raises nothing.
"""
import pytest
import torch

from flash_ansr.model.pre_encoder import IEEE75432PreEncoder, float32_to_ieee754_bits
from flash_ansr.utils.ieee754 import IEEE754_N_NIBBLES


def test_bit_decomposition_refuses_a_non_float32_tensor():
    with pytest.raises(TypeError, match="float32"):
        float32_to_ieee754_bits(torch.zeros(4, 3, dtype=torch.float64))


def test_bit_decomposition_accepts_float32():
    """The codec returns RAW bits {0,1}; the +-1 mapping is the pre-encoder's forward."""
    out = float32_to_ieee754_bits(torch.tensor([[1.0, -2.0]], dtype=torch.float32))
    assert out.shape == (1, 2, 32)
    assert set(out.unique().tolist()) <= {0, 1}
    # -2.0 is 0xc0000000: sign 1, then 0100 0000 ...
    assert out[0, 1, 0].item() == 1 and out[0, 1, 1].item() == 1
    mapped = IEEE75432PreEncoder(input_size=2)(torch.tensor([[1.0, -2.0]], dtype=torch.float32))
    assert set(mapped.unique().tolist()) <= {-1.0, 1.0}


def test_the_scrambling_shape_coincidence_is_real():
    """The reason the assert exists: the wrong dtype yields the RIGHT width."""
    x64 = torch.zeros(2, 18, dtype=torch.float64)
    reinterpreted = x64.view(torch.int32)          # no error, no conversion
    assert reinterpreted.shape == (2, 36)          # 18 f64 -> 36 i32, silently
    # and 36 * 32 bits would be 1152, while 18 * 64 is also 1152: the widths collide.
    assert 36 * 32 == 18 * 64


def test_pre_encoder_reports_its_own_width():
    enc = IEEE75432PreEncoder(input_size=18)
    assert enc.encoding_size == 32 and enc.output_size == 18 * 32


def test_tasks_uses_the_shared_span_constant():
    """tasks.py carried its own NIBBLES_PER_SPAN = 8 ten lines after importing the real one."""
    import flash_ansr.tasks as tasks
    assert not hasattr(tasks, "NIBBLES_PER_SPAN")
    assert tasks.IEEE754_N_NIBBLES == IEEE754_N_NIBBLES


def test_is_constant_token_uses_a_set_not_a_scan():
    from flash_ansr import scoring
    assert isinstance(scoring._NIBBLE_TOKEN_SET, frozenset)
    from flash_ansr.utils.ieee754 import NIBBLE_TOKENS
    assert scoring._NIBBLE_TOKEN_SET == frozenset(NIBBLE_TOKENS)


# ---------------------------------------------------------------------------
# The end-to-end guard: one dtype on every numeric surface the model reads.
#
# The per-site asserts above catch a mismatch once it reaches the pre-encoder. This one
# catches it at the seam where it is INTRODUCED -- collate is the last place a batch can
# still be normalized, and a single site left behind there is the failure the migration's
# own plan flags as most likely to hide.
# ---------------------------------------------------------------------------

class _DummyCatalog:
    simplipy_engine = None
    variables = ["x1", "x2"]


class _DummySource:
    """The collate-only seam: the dataset never streams here, so __init__'s reads suffice."""
    config = {"catalog": {"type": "lample_charton"},
              "sampling": {"n_support": "prior", "n_validation": 0}}
    max_n_support = 4
    catalog = _DummyCatalog()
    noise_spec = None


def test_collate_normalizes_every_numeric_surface():
    """Whatever width the worker produced, what leaves collate is ONE dtype.

    Deliberately fed float32 -- the pre-migration width -- because a batch that arrives
    already correct proves nothing about the cast sites.
    """
    from flash_ansr.data import FlashANSRDataset
    from flash_ansr.model.tokenizer import Tokenizer
    from flash_ansr.utils.ieee754 import IEEE754_SPECIAL_TOKENS
    from flash_ansr.utils.numeric import NUMERIC_DTYPE

    tokenizer = Tokenizer(
        vocab=["x1", "x2"],
        special_tokens=["<pad>", "<bos>", "<eos>", "<expression>", "</expression>",
                        *IEEE754_SPECIAL_TOKENS, "<float>"],
    )
    with FlashANSRDataset(source=_DummySource(), tokenizer=tokenizer, padding="zero") as dataset:
        batch = {
            "input_ids": [[tokenizer["<bos>"], tokenizer["x1"]],
                          [tokenizer["<bos>"], tokenizer["x2"]]],
            "x_tensors": [torch.zeros((2, 2), dtype=torch.float32),
                          torch.ones((2, 2), dtype=torch.float32)],
            "y_tensors": [torch.ones((2, 1), dtype=torch.float32),
                          torch.zeros((2, 1), dtype=torch.float32)],
            "constants": [[0.1, 0.2], [0.3]],
            "input_num": [[float("nan"), 1.0], [float("nan"), 2.0]],
            "residual": torch.zeros((2, 2), dtype=torch.float32),
        }
        collated = dataset.collate(batch, device="cpu")

    for key in ("x_tensors", "y_tensors", "input_num", "residual"):
        assert collated[key].dtype is NUMERIC_DTYPE, f"{key} left collate as {collated[key].dtype}"
    for tensor in collated["constants"]:
        assert tensor.dtype is NUMERIC_DTYPE, f"constants left collate as {tensor.dtype}"
    # Masks are boolean data and are NOT part of the numeric width (see S10).
    assert collated["data_attn_mask"].dtype is torch.bool


def test_the_shared_dtype_and_the_configured_width_are_the_same_number():
    """`NUMERIC_DTYPE` and `pre_encoder_bits` are set in different files by different
    kinds of edit; nothing but this ties them together."""
    import torch as _torch

    from flash_ansr.model.flash_ansr_model import FlashANSRModel
    from flash_ansr.utils.numeric import NUMERIC_DTYPE
    from flash_ansr.utils.paths import get_path

    model = FlashANSRModel.from_config(get_path("configs", "test", "model.yaml"))
    assert _torch.finfo(NUMERIC_DTYPE).bits == model.pre_encoder_bits
    assert model.pre_encoder.encoding_size == model.pre_encoder_numeric_tokens.encoding_size
