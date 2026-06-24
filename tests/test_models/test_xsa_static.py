"""Bit-identity gate for XSA under static decode (the verification that authorizes opening the
capability gate). Mirrors gate_static_kv_logits.py at the decoder level with a TINY random model
(bit-identity is weight-independent): a dynamic prefill seeds both paths, then identical tokens are
fed to ``forward`` (dynamic, cat-grow incremental) and ``forward_static`` (position-indexed buffer),
comparing the per-step hidden state at atol=1e-5.

The ``use_xsa=False`` arm validates the harness itself (the proven non-XSA static path must match);
the ``use_xsa=True`` arm is the actual check that the XSA own-value orthogonalization in
``forward_static_self`` is bit-identical to the dynamic path.
"""
import unittest

import torch

from flash_ansr.model.decoders.transformer import TransformerDecoder
from flash_ansr.model.decoders.static_kv import StaticKVCache

ATOL = 1e-5


def _max_static_vs_dynamic_diff(use_xsa: bool) -> float:
    torch.manual_seed(0)
    B, M, T_prefix, T_steps = 2, 5, 3, 10
    vocab, dim, heads, layers, ffn, max_len, input_dim = 37, 32, 4, 2, 64, 64, 16
    dec = TransformerDecoder(
        vocab_size=vocab, input_dim=input_dim, model_dim=dim, n_layers=layers, n_heads=heads,
        max_seq_len=max_len, ffn_hidden_dim=ffn, dropout=0.0,
        use_rope_self_attn=True, use_xsa_self_attn=use_xsa,
    ).eval()
    enc_mem = torch.randn(B, M, input_dim)
    prefix = torch.randint(0, vocab, (B, T_prefix))
    toks = torch.randint(0, vocab, (T_steps, B, 1))   # identical tokens fed to BOTH paths each step
    worst = 0.0
    with torch.no_grad():
        h_pre, cache = dec.forward(prefix, enc_mem, use_cache=True)
        sc = StaticKVCache(n_layers=layers, batch=B, n_heads=heads, head_dim=dim // heads,
                           max_len=max_len, device=h_pre.device, dtype=h_pre.dtype)
        sc.seed_from_dynamic(cache)
        for t in range(T_steps):
            pos = T_prefix + t
            tok = toks[t]
            h_dyn, cache = dec.forward(tok, enc_mem, past_key_values=cache, use_cache=True)
            h_stat = dec.forward_static(tok, enc_mem, None, sc, position=pos)
            worst = max(worst, (h_dyn - h_stat).abs().max().item())
    return worst


class TestXSAStaticBitIdentity(unittest.TestCase):
    def test_harness_valid_non_xsa(self) -> None:
        # Sanity: the proven non-XSA static path must match dynamic -- validates the test harness itself.
        self.assertLess(_max_static_vs_dynamic_diff(use_xsa=False), ATOL)

    def test_xsa_static_matches_dynamic(self) -> None:
        # The actual check: own_v orthogonalization in forward_static_self == the dynamic path.
        self.assertLess(_max_static_vs_dynamic_diff(use_xsa=True), ATOL)


if __name__ == "__main__":
    unittest.main()
