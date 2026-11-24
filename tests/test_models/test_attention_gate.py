"""Gated attention (Qiu et al. 2025, arXiv:2505.06708) in the decoder: off by default, so checkpoints without
it load unchanged; on, every decode path (full, cached incremental, static buffers) computes the same thing."""
import pytest
import torch

from flash_ansr.model.decoders.components import Attention
from flash_ansr.model.decoders.static_kv import StaticKVCache
from flash_ansr.model.decoders.transformer import TransformerDecoder

B, M, VOCAB, DIM, HEADS, LAYERS, FFN, MAX_LEN, INPUT_DIM = 2, 5, 37, 32, 4, 2, 64, 64, 16


def _decoder(gate_self: bool, gate_cross: bool) -> TransformerDecoder:
    torch.manual_seed(0)
    return TransformerDecoder(
        vocab_size=VOCAB, input_dim=INPUT_DIM, model_dim=DIM, n_layers=LAYERS, n_heads=HEADS, max_seq_len=MAX_LEN,
        ffn_hidden_dim=FFN, dropout=0.0, use_rope_self_attn=True, gate_self_attn=gate_self, gate_cross_attn=gate_cross,
    ).eval()


def test_off_by_default_adds_no_parameters():
    assert Attention(dim=DIM, n_heads=HEADS).w_g is None
    keys = set(_decoder(False, False).state_dict())
    assert not any(".w_g." in k for k in keys)
    gated = set(_decoder(True, True).state_dict())
    assert gated - keys == {f"layers.{i}.{part}.w_g.{w}" for i in range(LAYERS)
                            for part in ("self_attention", "cross_attention") for w in ("weight", "bias")}


@pytest.mark.parametrize("gate_self,gate_cross", [(True, False), (False, True), (True, True)])
def test_cached_decoding_matches_the_full_forward(gate_self, gate_cross):
    dec = _decoder(gate_self, gate_cross)
    torch.manual_seed(1)
    memory = torch.randn(B, M, INPUT_DIM)
    tokens = torch.randint(0, VOCAB, (B, 9))
    with torch.no_grad():
        full = dec.forward(tokens, memory)
        h, cache = dec.forward(tokens[:, :4], memory, use_cache=True)
        steps = [h]
        for t in range(4, tokens.shape[1]):
            h, cache = dec.forward(tokens[:, t:t + 1], memory, past_key_values=cache, use_cache=True)
            steps.append(h)
    torch.testing.assert_close(torch.cat(steps, dim=1), full, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("gate_self,gate_cross", [(False, False), (True, True)])
def test_static_decoding_matches_dynamic_decoding(gate_self, gate_cross):
    dec = _decoder(gate_self, gate_cross)
    torch.manual_seed(2)
    memory = torch.randn(B, M, INPUT_DIM)
    prefix = torch.randint(0, VOCAB, (B, 3))
    steps = torch.randint(0, VOCAB, (8, B, 1))
    worst = 0.0
    with torch.no_grad():
        h, cache = dec.forward(prefix, memory, use_cache=True)
        static = StaticKVCache(n_layers=LAYERS, batch=B, n_heads=HEADS, head_dim=DIM // HEADS, max_len=MAX_LEN,
                               device=h.device, dtype=h.dtype)
        static.seed_from_dynamic(cache)
        for t, tok in enumerate(steps):
            h_dyn, cache = dec.forward(tok, memory, past_key_values=cache, use_cache=True)
            h_stat = dec.forward_static(tok, memory, None, static, position=prefix.shape[1] + t)
            worst = max(worst, (h_dyn - h_stat).abs().max().item())
    assert worst < 1e-5


def test_a_closed_gate_silences_the_head():
    attn = Attention(dim=DIM, n_heads=HEADS, use_gate=True).eval()
    with torch.no_grad():
        attn.w_g.weight.zero_()
        attn.w_g.bias.fill_(-1e4)
        x = torch.randn(B, 6, DIM)
        cos, sin = torch.ones(6, DIM // HEADS), torch.zeros(6, DIM // HEADS)
        out = attn(x, x, (cos, sin), is_causal=True)
    torch.testing.assert_close(out, attn.w_o.bias.expand_as(out), rtol=0, atol=1e-6)
