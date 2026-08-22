"""Tests for mask-aware query/residual-stream SetNorms (mask_query_norms), the input_num NaN
guard (sanitize_input_num), and the encoder padding-mask dtype coercion.

Legacy behavior (both flags False, the default) is load-bearing: existing checkpoints were
trained with padding-diluted set statistics and with the NaN-bit-pattern numeric embedding at
non-constant positions, so the defaults must keep reproducing it exactly.
"""
import pytest
import torch

from flash_ansr import FlashANSRModel, get_path
from flash_ansr.model.encoders.set_transformer import MAB, SetTransformer
from flash_ansr.model.pre_encoder import IEEE75432PreEncoder


def _make_set_transformer(mask_query_norms: bool, seed: int = 7) -> SetTransformer:
    torch.manual_seed(seed)
    return SetTransformer(
        input_dim=24, output_dim=None, model_dim=32, n_heads=4, n_isab=2, n_sab=1,
        n_inducing_points=8, n_seeds=4, ffn_hidden_dim=64, dropout=0.0,
        attn_norm="rms_set", ffn_norm="rms_set", output_norm="rms_set",
        mask_query_norms=mask_query_norms,
    ).eval()


def _encode(model: SetTransformer, x_valid: torch.Tensor, total_len: int) -> torch.Tensor:
    n = x_valid.shape[1]
    x = torch.zeros(1, total_len, x_valid.shape[-1])
    x[:, :n] = x_valid
    mask = torch.zeros(1, total_len, dtype=torch.bool)
    mask[:, :n] = True
    with torch.no_grad():
        return model(x, attn_mask=mask)


class TestMaskQueryNorms:
    def test_default_is_legacy(self):
        mab = MAB(dim_q=16, dim_kv=16, dim=16, n_heads=2)
        assert mab.mask_query_norms is False
        st = _make_set_transformer(mask_query_norms=False)
        assert all(isab.mab_self.mask_query_norms is False for isab in st.isabs)

    def test_flag_reaches_mab_self_only(self):
        st = _make_set_transformer(mask_query_norms=True)
        for isab in st.isabs:
            assert isab.mab_self.mask_query_norms is True
            # mab_cross queries are the dense inducing points; no query mask needed there.
            assert isab.mab_cross.mask_query_norms is False

    def test_flag_adds_no_parameters(self):
        legacy = _make_set_transformer(mask_query_norms=False)
        fixed = _make_set_transformer(mask_query_norms=True)
        assert set(legacy.state_dict().keys()) == set(fixed.state_dict().keys())
        # Old checkpoints load into flag-enabled models.
        fixed.load_state_dict(legacy.state_dict())

    def test_padding_invariance_with_flag(self):
        model = _make_set_transformer(mask_query_norms=True)
        x_valid = torch.randn(1, 5, 24, generator=torch.Generator().manual_seed(1))
        tight = _encode(model, x_valid, 5)
        padded = _encode(model, x_valid, 64)
        rel = (tight - padded).abs().max() / tight.abs().max()
        assert rel < 1e-4, f"flag-on encoding must not depend on padding length (rel diff {rel:.2e})"

    def test_legacy_is_padding_dependent(self):
        # Documents the legacy semantics: set statistics include zero-padding, so the same
        # sample encodes differently at different padding lengths. Existing checkpoints were
        # trained with this behavior; if this test ever fails, the legacy default changed.
        model = _make_set_transformer(mask_query_norms=False)
        x_valid = torch.randn(1, 5, 24, generator=torch.Generator().manual_seed(1))
        tight = _encode(model, x_valid, 5)
        padded = _encode(model, x_valid, 64)
        rel = (tight - padded).abs().max() / tight.abs().max()
        assert rel > 1e-2

    @pytest.mark.parametrize("norm_position", ["pre", "post"])
    def test_mab_padded_query_rows_stay_zero(self, norm_position):
        torch.manual_seed(3)
        mab = MAB(
            dim_q=16, dim_kv=16, dim=16, n_heads=2, ffn_hidden_dim=32, dropout=0.0,
            attn_norm="rms_set", ffn_norm="rms_set", norm_position=norm_position,
            mask_query_norms=True,
        ).eval()
        query = torch.randn(2, 10, 16)
        query_mask = torch.zeros(2, 10, dtype=torch.bool)
        query_mask[:, :4] = True
        query = query * query_mask.unsqueeze(-1)
        kv = torch.randn(2, 6, 16)
        with torch.no_grad():
            out = mab(query, kv, query_mask=query_mask)
        if norm_position == "pre":
            # Pre-norm: padded rows enter as zero and every sub-layer output is re-masked,
            # so the residual stream must stay exactly zero on padding.
            assert torch.equal(out[~query_mask], torch.zeros_like(out[~query_mask]))
        else:
            # Post-norm normalizes the (zero) residual rows; they need not be exactly zero,
            # but they must not influence the valid rows (covered by the invariance test).
            assert out.shape == query.shape


class TestMaskDtypeCoercion:
    def test_float_mask_warns_and_matches_bool(self):
        model = _make_set_transformer(mask_query_norms=False)
        x_valid = torch.randn(1, 5, 24, generator=torch.Generator().manual_seed(2))
        out_bool = _encode(model, x_valid, 16)
        x = torch.zeros(1, 16, 24)
        x[:, :5] = x_valid
        float_mask = torch.zeros(1, 16)
        float_mask[:, :5] = 1.0
        with pytest.warns(UserWarning, match="non-bool attn_mask"):
            with torch.no_grad():
                out_float = model(x, attn_mask=float_mask)
        # A float 0/1 mask would silently become an ADDITIVE attention bias in SDPA
        # (masking nothing); after coercion it must behave exactly like the bool mask.
        assert torch.equal(out_bool, out_float)


class TestSanitizeInputNum:
    def test_bits_of_nan_are_finite(self):
        # The premise of the fix: the legacy guard checked isnan on the bit encodings,
        # which are ±1 for ANY input (including NaN), so it could never fire.
        bits = IEEE75432PreEncoder(1)(torch.tensor([[[float("nan")]]]))
        assert not torch.isnan(bits).any()
        assert bits.abs().eq(1).all()

    def test_sanitize_zeroes_only_nan_positions(self):
        model = FlashANSRModel.from_config(get_path("configs", "test", "model.yaml"))
        model.eval()
        # from_config defaults: both flags stay legacy-False for configs without the new keys.
        assert model.sanitize_input_num is False
        assert all(isab.mab_self.mask_query_norms is False for isab in model.encoder.isabs)

        x = torch.rand(2, 10, 11)
        input_tokens = torch.randint(
            low=len(model.tokenizer.special_tokens), high=len(model.tokenizer), size=(2, 7)
        )
        input_num = torch.full((2, 7, 1), float("nan"))
        input_num[0, 2, 0] = 1.5

        with torch.no_grad():
            logits_legacy = model(input_tokens, x, input_num=input_num)
            model.sanitize_input_num = True
            logits_sane = model(input_tokens, x, input_num=input_num)

        # The guard changes the numeric contribution at NaN positions, so logits must differ...
        assert not torch.equal(logits_legacy, logits_sane)

        # ...but with no NaNs present, sanitization must be a no-op.
        input_num_full = torch.full((2, 7, 1), 2.0)
        with torch.no_grad():
            model.sanitize_input_num = False
            a = model(input_tokens, x, input_num=input_num_full)
            model.sanitize_input_num = True
            b = model(input_tokens, x, input_num=input_num_full)
        assert torch.equal(a, b)
