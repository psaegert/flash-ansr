"""Tests for the pre/post-norm decoder switch and the LayerNorm encoder ablation."""
import pytest
import torch

from flash_ansr import FlashANSRModel, get_path
from flash_ansr.model.decoders import TransformerDecoder
from flash_ansr.model.decoders.components import TransformerDecoderBlock
from flash_ansr.model.encoders.set_transformer import MAB, SetTransformer


def _make_block(norm_position: str = "pre") -> TransformerDecoderBlock:
    return TransformerDecoderBlock(
        dim=32,
        n_heads=4,
        ffn_hidden_dim=64,
        dropout=0.0,
        use_rope_self_attn=False,
        norm_position=norm_position,
    )


class TestDecoderBlockNormPosition:
    def test_default_is_pre_norm(self):
        block = TransformerDecoderBlock(dim=16, n_heads=2)
        assert block.norm_position == "pre"

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            TransformerDecoderBlock(dim=16, n_heads=2, norm_position="middle")

    @pytest.mark.parametrize("norm_position", ["pre", "post"])
    def test_forward_shapes(self, norm_position):
        block = _make_block(norm_position)
        x = torch.randn(2, 7, 32)
        memory = torch.randn(2, 5, 32)
        # Provide RoPE buffers shaped as the block expects (cos, sin) of shape (1,1,T,head_dim)
        head_dim = 32 // 4
        rope = (
            torch.zeros(1, 1, 7, head_dim),
            torch.zeros(1, 1, 7, head_dim),
        )
        out = block(x, memory, rope_emb=rope)
        assert out.shape == x.shape

    def test_pre_and_post_produce_different_outputs(self):
        torch.manual_seed(0)
        pre = _make_block("pre")
        torch.manual_seed(0)
        post = _make_block("post")
        x = torch.randn(2, 7, 32)
        memory = torch.randn(2, 5, 32)
        head_dim = 32 // 4
        rope = (torch.zeros(1, 1, 7, head_dim), torch.zeros(1, 1, 7, head_dim))
        out_pre = pre(x, memory, rope_emb=rope)
        out_post = post(x, memory, rope_emb=rope)
        assert not torch.allclose(out_pre, out_post)


class TestTransformerDecoderNormPosition:
    @pytest.mark.parametrize("norm_position", ["pre", "post"])
    def test_decoder_forward(self, norm_position):
        dec = TransformerDecoder(
            vocab_size=50,
            input_dim=32,
            model_dim=32,
            n_layers=2,
            n_heads=4,
            max_seq_len=64,
            ffn_hidden_dim=64,
            dropout=0.0,
            block_norm_position=norm_position,
        )
        tokens = torch.randint(0, 50, (2, 7))
        memory = torch.randn(2, 5, 32)
        out = dec(tokens, memory)
        assert out.shape == (2, 7, 32)
        for layer in dec.layers:
            assert layer.norm_position == norm_position


class TestFlashANSRModelAblations:
    """End-to-end forward passes through FlashANSRModel for each ablation."""

    def _forward(self, model: FlashANSRModel) -> torch.Tensor:
        batch, set_size, n_vars = 4, 6, 11
        x = torch.randn(batch, set_size, n_vars)
        seq_len = 9
        tokens = torch.randint(
            len(model.tokenizer.special_tokens), len(model.tokenizer), (batch, seq_len)
        )
        return model.forward(tokens, x)

    def test_default_loads_pre_norm_and_32bit(self):
        model = FlashANSRModel.from_config(get_path('configs', 'test', 'model.yaml'))
        assert model.pre_encoder.encoding_size == 32
        assert model.pre_encoder_bits == 32
        for layer in model.decoder.layers:
            assert layer.norm_position == "pre"

    def test_b1_post_norm_decoder(self):
        from flash_ansr.utils.config_io import load_config
        cfg = load_config(get_path('configs', 'test', 'model.yaml'))
        cfg['decoder_block_norm_position'] = 'post'

        from simplipy import SimpliPyEngine
        from flash_ansr.model.tokenizer import Tokenizer
        engine = SimpliPyEngine.load(cfg["simplipy_engine"], install=True)
        tokenizer = Tokenizer.from_config(get_path('configs', 'test', 'tokenizer.yaml'))

        model = FlashANSRModel(
            simplipy_engine=engine,
            tokenizer=tokenizer,
            pre_encoder_noise_scale=cfg['pre_encoder_noise_scale'],
            encoder_max_n_variables=cfg['encoder_max_n_variables'],
            encoder_dim=cfg['encoder_dim'],
            encoder_n_heads=cfg['encoder_n_heads'],
            encoder_n_isab=cfg['encoder_n_isab'],
            encoder_n_sab=cfg['encoder_n_sab'],
            encoder_n_inducing_points=cfg['encoder_n_inducing_points'],
            encoder_n_seeds=cfg['encoder_n_seeds'],
            encoder_ffn_hidden_dim=cfg['encoder_ffn_hidden_dim'],
            encoder_dropout=cfg['encoder_dropout'],
            encoder_attn_norm=cfg['encoder_attn_norm'],
            encoder_ffn_norm=cfg['encoder_ffn_norm'],
            encoder_output_norm=cfg['encoder_output_norm'],
            decoder_input_dim=cfg['decoder_input_dim'],
            decoder_model_dim=cfg['decoder_model_dim'],
            decoder_n_layers=cfg['decoder_n_layers'],
            decoder_n_heads=cfg['decoder_n_heads'],
            decoder_max_seq_len=cfg['decoder_max_seq_len'],
            decoder_ffn_hidden_dim=cfg['decoder_ffn_hidden_dim'],
            decoder_dropout=cfg['decoder_dropout'],
            decoder_block_self_attn_norm=cfg['decoder_block_self_attn_norm'],
            decoder_block_cross_attn_norm=cfg['decoder_block_cross_attn_norm'],
            decoder_block_ffn_norm=cfg['decoder_block_ffn_norm'],
            decoder_cross_attn_kv_norm=cfg['decoder_cross_attn_kv_norm'],
            decoder_output_norm=cfg['decoder_output_norm'],
            decoder_use_rope_self_attn=cfg['decoder_use_rope_self_attn'],
            decoder_use_rope_cross_attn=cfg['decoder_use_rope_cross_attn'],
            decoder_block_norm_position='post',
            encoder_block_norm_position='post',
            pre_encoder_bits=32,
            use_checkpointing=cfg['use_checkpointing'],
        )
        for layer in model.decoder.layers:
            assert layer.norm_position == "post"
        # Encoder side: every MAB inside ISAB / SAB / PMA must report post-norm.
        for isab in model.encoder.isabs:
            assert isab.mab_cross.norm_position == "post"
            assert isab.mab_self.norm_position == "post"
        for sab in model.encoder.sabs:
            assert sab.mab.norm_position == "post"
        assert model.encoder.pma.mab.norm_position == "post"
        logits = self._forward(model)
        assert logits.shape[-1] == len(model.tokenizer)
        assert torch.isfinite(logits).all()

    def test_b2_pre_encoder_16bit(self):
        from flash_ansr.utils.config_io import load_config
        cfg = load_config(get_path('configs', 'test', 'model.yaml'))

        from simplipy import SimpliPyEngine
        from flash_ansr.model.tokenizer import Tokenizer
        engine = SimpliPyEngine.load(cfg["simplipy_engine"], install=True)
        tokenizer = Tokenizer.from_config(get_path('configs', 'test', 'tokenizer.yaml'))

        kwargs = {k: cfg[k] for k in cfg if k != 'simplipy_engine' and k != 'tokenizer'}
        model = FlashANSRModel(
            simplipy_engine=engine,
            tokenizer=tokenizer,
            pre_encoder_bits=16,
            **kwargs,
        )
        assert model.pre_encoder.encoding_size == 16
        assert model.pre_encoder_numeric_tokens.encoding_size == 16
        logits = self._forward(model)
        assert logits.shape[-1] == len(model.tokenizer)
        assert torch.isfinite(logits).all()

    def test_b4_layernorm_encoder(self):
        from flash_ansr.utils.config_io import load_config
        cfg = load_config(get_path('configs', 'test', 'model.yaml'))

        from simplipy import SimpliPyEngine
        from flash_ansr.model.tokenizer import Tokenizer
        engine = SimpliPyEngine.load(cfg["simplipy_engine"], install=True)
        tokenizer = Tokenizer.from_config(get_path('configs', 'test', 'tokenizer.yaml'))

        kwargs = {k: cfg[k] for k in cfg if k not in ('simplipy_engine', 'tokenizer',
                                                      'encoder_attn_norm', 'encoder_ffn_norm',
                                                      'encoder_output_norm')}
        model = FlashANSRModel(
            simplipy_engine=engine,
            tokenizer=tokenizer,
            encoder_attn_norm='layer',
            encoder_ffn_norm='layer',
            encoder_output_norm='layer',
            **kwargs,
        )
        logits = self._forward(model)
        assert logits.shape[-1] == len(model.tokenizer)
        assert torch.isfinite(logits).all()


class TestEncoderBlockNormPosition:
    """Direct tests for the Set Transformer pre/post-norm switch."""

    def test_mab_default_is_pre(self):
        mab = MAB(dim_q=16, dim_kv=16, dim=16, n_heads=4, attn_norm='layer', ffn_norm='layer')
        assert mab.norm_position == "pre"

    def test_mab_invalid_raises(self):
        with pytest.raises(ValueError):
            MAB(dim_q=16, dim_kv=16, dim=16, n_heads=4, norm_position='middle')

    @pytest.mark.parametrize("norm_position", ["pre", "post"])
    def test_mab_self_attention_forward(self, norm_position):
        mab = MAB(dim_q=16, dim_kv=16, dim=16, n_heads=4, attn_norm='layer', ffn_norm='layer',
                  is_self_attention=True, norm_position=norm_position)
        x = torch.randn(2, 5, 16)
        out = mab(x, x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize("norm_position", ["pre", "post"])
    def test_mab_cross_attention_forward(self, norm_position):
        # query_is_projected=True so dim_q == dim_out (matches ISAB.mab_cross/PMA.mab usage).
        mab = MAB(dim_q=16, dim_kv=8, dim=16, n_heads=4, attn_norm='layer', ffn_norm='layer',
                  is_self_attention=False, query_is_projected=True, norm_position=norm_position)
        q = torch.randn(2, 3, 16)
        kv = torch.randn(2, 7, 8)
        out = mab(q, kv)
        assert out.shape == q.shape
        assert torch.isfinite(out).all()

    def test_mab_pre_and_post_differ(self):
        torch.manual_seed(0)
        pre = MAB(dim_q=16, dim_kv=16, dim=16, n_heads=4, attn_norm='layer', ffn_norm='layer',
                  is_self_attention=True, norm_position='pre')
        torch.manual_seed(0)
        post = MAB(dim_q=16, dim_kv=16, dim=16, n_heads=4, attn_norm='layer', ffn_norm='layer',
                   is_self_attention=True, norm_position='post')
        x = torch.randn(2, 5, 16)
        assert not torch.allclose(pre(x, x), post(x, x))

    @pytest.mark.parametrize("norm_position", ["pre", "post"])
    def test_set_transformer_forward(self, norm_position):
        enc = SetTransformer(
            input_dim=4, output_dim=16, model_dim=16, n_heads=4,
            n_isab=2, n_sab=1, n_inducing_points=8, n_seeds=4,
            ffn_hidden_dim=32, dropout=0.0,
            attn_norm='layer', ffn_norm='layer', output_norm='layer',
            norm_position=norm_position,
        )
        x = torch.randn(2, 6, 4)
        attn_mask = torch.ones(2, 6)
        out = enc(x, attn_mask=attn_mask)
        assert out.shape == (2, 4, 16)
        assert torch.isfinite(out).all()
        for isab in enc.isabs:
            assert isab.mab_cross.norm_position == norm_position
            assert isab.mab_self.norm_position == norm_position
        assert enc.pma.mab.norm_position == norm_position
        for sab in enc.sabs:
            assert sab.mab.norm_position == norm_position

    @pytest.mark.parametrize("norm_position", ["pre", "post"])
    def test_set_transformer_with_rms_set_norm(self, norm_position):
        # Mirrors the production encoder norm choice (rms_set) so we exercise the
        # SetNormBase masked-norm code path under both pre- and post-norm.
        enc = SetTransformer(
            input_dim=4, output_dim=16, model_dim=16, n_heads=4,
            n_isab=2, n_sab=1, n_inducing_points=8, n_seeds=4,
            ffn_hidden_dim=32, dropout=0.0,
            attn_norm='rms_set', ffn_norm='rms_set', output_norm='rms_set',
            norm_position=norm_position,
        )
        x = torch.randn(2, 6, 4)
        attn_mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0]], dtype=torch.float32)
        out = enc(x, attn_mask=attn_mask)
        assert out.shape == (2, 4, 16)
        assert torch.isfinite(out).all()
