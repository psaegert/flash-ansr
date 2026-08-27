"""Cross-attention must not rely on the K/V batch dim broadcasting against Q.

Decoding runs `choices` rows against ONE encoder memory, so cross-attention K/V arrive with
batch 1. Letting that broadcast inside ``scaled_dot_product_attention`` is outside the
documented contract -- the signature pairs (N, ..., H, L, E) with (N, ..., H, S, E), the same
N -- and backends disagree on it. These tests assert the shapes reaching the op, which is
device-independent and so runs anywhere CI does.
"""
import unittest
from unittest.mock import patch

import torch

from flash_ansr.utils.numeric import NUMERIC_DTYPE
import torch.nn.functional as F

from flash_ansr import FlashANSRModel, get_path


class TestCrossAttentionBatchDims(unittest.TestCase):
    """Every scaled_dot_product_attention call must receive matched batch dims."""

    def setUp(self) -> None:
        self.nsr = FlashANSRModel.from_config(get_path('configs', 'test', 'model.yaml'))
        self.nsr.eval()
        self.real_sdpa = F.scaled_dot_product_attention

    def _record_sdpa_batches(self, run) -> list[tuple[int, int, int]]:
        """Run `run()` with scaled_dot_product_attention instrumented; return the (q, k, v)
        batch dims of every call made."""
        seen: list[tuple[int, int, int]] = []

        def spy(query, key, value, *args, **kwargs):
            seen.append((query.shape[0], key.shape[0], value.shape[0]))
            return self.real_sdpa(query, key, value, *args, **kwargs)

        with patch.object(F, 'scaled_dot_product_attention', spy):
            run()
        return seen

    def test_batch_dims_match_with_broadcast_memory(self):
        """A batch-1 memory against multi-row tokens must not leave k/v batch at 1."""
        n_rows = 4
        memory = self.nsr._create_memory(torch.rand(1, 13, 11, dtype=NUMERIC_DTYPE))
        self.assertEqual(memory.shape[0], 1, "fixture should produce a batch-1 memory")
        tokens = torch.randint(1, 10, (n_rows, 3))

        with torch.no_grad():
            seen = self._record_sdpa_batches(
                lambda: self.nsr.forward(tokens, None, memory=memory, use_cache=False))

        self.assertTrue(seen, "no attention calls were recorded")
        mismatched = [s for s in seen if not (s[0] == s[1] == s[2])]
        self.assertEqual(
            mismatched, [],
            f"scaled_dot_product_attention received mismatched batch dims (q, k, v): {mismatched}")

    def test_batch_dims_match_during_cached_decode(self):
        """The same holds on the incremental path, where cross-attention K/V come from cache."""
        n_rows = 4
        memory = self.nsr._create_memory(torch.rand(1, 13, 11, dtype=NUMERIC_DTYPE))
        tokens = torch.randint(1, 10, (n_rows, 4))

        def run():
            past = None
            for t in range(tokens.shape[1]):
                _, past = self.nsr.forward(tokens[:, t:t + 1], None, memory=memory,
                                           past_key_values=past, use_cache=True)

        with torch.no_grad():
            seen = self._record_sdpa_batches(run)

        self.assertTrue(seen, "no attention calls were recorded")
        mismatched = [s for s in seen if not (s[0] == s[1] == s[2])]
        self.assertEqual(
            mismatched, [],
            f"scaled_dot_product_attention received mismatched batch dims (q, k, v): {mismatched}")

    def test_cached_kv_still_carries_query_batch(self):
        """The hoist must not change what use_cache returns: K/V still come back at Q's batch."""
        n_rows = 3
        memory = self.nsr._create_memory(torch.rand(1, 13, 11, dtype=NUMERIC_DTYPE))
        tokens = torch.randint(1, 10, (n_rows, 3))

        with torch.no_grad():
            _, past = self.nsr.forward(tokens, None, memory=memory, use_cache=True)

        for layer, entry in enumerate(past):
            for tensor in entry:
                if isinstance(tensor, torch.Tensor) and tensor.ndim == 4:
                    self.assertEqual(tensor.shape[0], n_rows,
                                     f"layer {layer} cached K/V has batch {tensor.shape[0]}, "
                                     f"expected {n_rows}")

    def test_static_cross_caches_a_view_and_matches_a_contiguous_reference(self):
        """forward_static_cross satisfies the batch contract without copying the memory."""
        n_rows = 8
        memory = self.nsr._create_memory(torch.rand(1, 13, 11, dtype=NUMERIC_DTYPE))
        attn = self.nsr.decoder.layers[0].cross_attention
        dim = attn.n_heads * attn.head_dim
        q = torch.randn(n_rows, 1, dim)

        holder = [None]
        with torch.no_grad():
            first = attn.forward_static_cross(q, memory, holder)
            second = attn.forward_static_cross(q, memory, holder)   # reuses the cached pair

        k, v = holder[0]
        self.assertEqual(k.shape[0], n_rows)
        self.assertEqual(k.stride()[0], 0, "cached K was materialized instead of expanded")
        self.assertEqual(v.stride()[0], 0, "cached V was materialized instead of expanded")
        self.assertTrue(torch.equal(first, second), "cache reuse changed the result")

        # ... and the view must give exactly what a materialized copy would.
        with torch.no_grad():
            ref = attn.forward_static_cross(q, memory, [(k.contiguous(), v.contiguous())])
        self.assertTrue(torch.allclose(first, ref, atol=1e-6))


if __name__ == '__main__':
    unittest.main()
