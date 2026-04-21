"""Tests for optional KV-cache in the decoder and generation methods."""
import unittest

import torch

from flash_ansr import FlashANSRModel, get_path


class TestKVCacheForward(unittest.TestCase):
    """Verify that incremental (cached) forward passes match full recomputation."""

    def setUp(self) -> None:
        self.nsr = FlashANSRModel.from_config(get_path('configs', 'test', 'model.yaml'))
        self.nsr.eval()

    def test_forward_use_cache_false_returns_tensor(self):
        """use_cache=False returns a plain Tensor, not a tuple."""
        x = torch.rand(1, 13, 11)
        tokens = torch.tensor([[1, 5, 7]])
        result = self.nsr.forward(tokens, x, use_cache=False)
        self.assertIsInstance(result, torch.Tensor)

    def test_forward_use_cache_true_returns_tuple(self):
        """use_cache=True returns (logits, past_key_values)."""
        x = torch.rand(1, 13, 11)
        tokens = torch.tensor([[1, 5, 7]])
        result = self.nsr.forward(tokens, x, use_cache=True)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        logits, past = result
        self.assertIsInstance(logits, torch.Tensor)
        self.assertIsInstance(past, list)

    def test_prefill_matches_full_forward(self):
        """Logits from a prefill step (use_cache=True, no past) match use_cache=False."""
        x = torch.rand(1, 13, 11)
        tokens = torch.tensor([[1, 5, 7, 9]])
        logits_full = self.nsr.forward(tokens, x, use_cache=False)
        logits_cached, _ = self.nsr.forward(tokens, x, use_cache=True)
        self.assertTrue(torch.allclose(logits_full, logits_cached, atol=1e-5))

    def test_incremental_matches_full(self):
        """Logits from prefill(2 tokens) + incremental(1 token) match full(3 tokens)."""
        x = torch.rand(1, 13, 11)
        memory = self.nsr._create_memory(x)
        tokens = torch.tensor([[1, 5, 7]])

        logits_full = self.nsr.forward(tokens, None, memory=memory, use_cache=False)

        logits_pf, past = self.nsr.forward(tokens[:, :2], None, memory=memory, use_cache=True)
        logits_inc, _ = self.nsr.forward(tokens[:, 2:], None, memory=memory, past_key_values=past, use_cache=True)

        self.assertTrue(
            torch.allclose(logits_full[:, -1], logits_inc[:, -1], atol=1e-5),
            f"Max diff: {(logits_full[:, -1] - logits_inc[:, -1]).abs().max().item()}"
        )

    def test_multi_step_incremental(self):
        """Multiple single-token incremental steps match full recomputation."""
        x = torch.rand(1, 13, 11)
        memory = self.nsr._create_memory(x)
        tokens = torch.tensor([[1, 5, 7, 9, 11]])

        logits_full = self.nsr.forward(tokens, None, memory=memory, use_cache=False)

        # Prefill with first 2 tokens
        _, past = self.nsr.forward(tokens[:, :2], None, memory=memory, use_cache=True)

        # Incrementally add tokens 2, 3, 4
        for i in range(2, 5):
            logits_inc, past = self.nsr.forward(tokens[:, i:i + 1], None, memory=memory, past_key_values=past, use_cache=True)

        self.assertTrue(
            torch.allclose(logits_full[:, -1], logits_inc[:, -1], atol=1e-5),
            f"Max diff: {(logits_full[:, -1] - logits_inc[:, -1]).abs().max().item()}"
        )

    def test_cache_shapes(self):
        """Verify KV-cache tensor shapes match expectations."""
        x = torch.rand(2, 13, 11)
        tokens = torch.tensor([[1, 5, 7], [1, 8, 9]])
        _, past = self.nsr.forward(tokens, x, use_cache=True)

        n_layers = self.nsr.decoder.layers.__len__()
        n_heads = 4
        head_dim = 128 // n_heads
        seq_len = 3
        batch_size = 2

        self.assertEqual(len(past), n_layers)
        for layer_cache in past:
            # ((sa_k, sa_v), (ca_k, ca_v))
            sa_cache, ca_cache = layer_cache
            sa_k, sa_v = sa_cache
            ca_k, ca_v = ca_cache

            # Self-attention cache: batch × heads × seq_len × head_dim
            self.assertEqual(sa_k.shape, (batch_size, n_heads, seq_len, head_dim))
            self.assertEqual(sa_v.shape, (batch_size, n_heads, seq_len, head_dim))

            # Cross-attention cache: batch × heads × encoder_seq_len × head_dim
            self.assertEqual(ca_k.shape[0], batch_size)
            self.assertEqual(ca_k.shape[1], n_heads)
            self.assertEqual(ca_k.shape[3], head_dim)

    def test_cache_grows_on_incremental_step(self):
        """Self-attention cache seq dim grows by 1 each step; cross-attention stays fixed."""
        x = torch.rand(1, 13, 11)
        memory = self.nsr._create_memory(x)
        tokens = torch.tensor([[1, 5, 7]])

        _, past = self.nsr.forward(tokens[:, :2], None, memory=memory, use_cache=True)
        sa_seq_before = past[0][0][0].shape[2]
        ca_seq_before = past[0][1][0].shape[2]

        _, past2 = self.nsr.forward(tokens[:, 2:], None, memory=memory, past_key_values=past, use_cache=True)
        sa_seq_after = past2[0][0][0].shape[2]
        ca_seq_after = past2[0][1][0].shape[2]

        self.assertEqual(sa_seq_after, sa_seq_before + 1, "Self-attn cache should grow by 1")
        self.assertEqual(ca_seq_after, ca_seq_before, "Cross-attn cache should stay fixed")

    def test_numeric_embeddings_with_cache(self):
        """Incremental step with numeric embeddings matches full forward."""
        x = torch.rand(1, 13, 11)
        memory = self.nsr._create_memory(x)
        tokens = torch.tensor([[1, 5, 7]])
        input_num = torch.tensor([[[0.5], [float('nan')], [float('nan')]]])

        logits_full = self.nsr.forward(tokens, None, input_num=input_num, memory=memory, use_cache=False)

        logits_pf, past = self.nsr.forward(tokens[:, :2], None, input_num=input_num[:, :2], memory=memory, use_cache=True)
        logits_inc, _ = self.nsr.forward(tokens[:, 2:], None, input_num=input_num[:, 2:], memory=memory, past_key_values=past, use_cache=True)

        self.assertTrue(
            torch.allclose(logits_full[:, -1], logits_inc[:, -1], atol=1e-5),
            f"Max diff: {(logits_full[:, -1] - logits_inc[:, -1]).abs().max().item()}"
        )


class TestKVCacheBeamSearch(unittest.TestCase):
    """Verify beam search produces identical results with and without KV-cache."""

    def setUp(self) -> None:
        self.nsr = FlashANSRModel.from_config(get_path('configs', 'test', 'model.yaml'))
        self.nsr.eval()
        self.x = torch.rand(13, 11)

    def test_beam_search_sequences_match(self):
        """Beam search with use_cache=True produces the same sequences."""
        torch.manual_seed(0)
        beams_no, scores_no, _ = self.nsr.beam_search(self.x, beam_width=4, max_len=10, use_cache=False)
        torch.manual_seed(0)
        beams_c, scores_c, _ = self.nsr.beam_search(self.x, beam_width=4, max_len=10, use_cache=True)

        self.assertEqual(beams_no, beams_c)

    def test_beam_search_scores_match(self):
        """Beam search with use_cache=True produces the same log-prob scores."""
        torch.manual_seed(0)
        _, scores_no, _ = self.nsr.beam_search(self.x, beam_width=4, max_len=10, use_cache=False)
        torch.manual_seed(0)
        _, scores_c, _ = self.nsr.beam_search(self.x, beam_width=4, max_len=10, use_cache=True)

        for s_no, s_c in zip(scores_no, scores_c):
            self.assertAlmostEqual(s_no, s_c, places=3)

    def test_beam_search_wider_beam(self):
        """Equivalence holds with a wider beam that forces more reshuffling."""
        torch.manual_seed(42)
        beams_no, scores_no, _ = self.nsr.beam_search(self.x, beam_width=8, max_len=12, use_cache=False)
        torch.manual_seed(42)
        beams_c, scores_c, _ = self.nsr.beam_search(self.x, beam_width=8, max_len=12, use_cache=True)

        self.assertEqual(beams_no, beams_c)
        for s_no, s_c in zip(scores_no, scores_c):
            self.assertAlmostEqual(s_no, s_c, places=3)

    def test_beam_search_non_unique(self):
        """Equivalence holds with unique=False (no simplification dedup)."""
        torch.manual_seed(7)
        beams_no, scores_no, _ = self.nsr.beam_search(self.x, beam_width=4, max_len=10, unique=False, use_cache=False)
        torch.manual_seed(7)
        beams_c, scores_c, _ = self.nsr.beam_search(self.x, beam_width=4, max_len=10, unique=False, use_cache=True)

        self.assertEqual(beams_no, beams_c)


class TestKVCacheSampling(unittest.TestCase):
    """Verify sampling produces identical results with and without KV-cache."""

    def setUp(self) -> None:
        self.nsr = FlashANSRModel.from_config(get_path('configs', 'test', 'model.yaml'))
        self.nsr.eval()
        self.x = torch.rand(13, 11)

    def test_sampling_sequences_match(self):
        """Sampling with use_cache=True produces the same sequences."""
        torch.manual_seed(0)
        beams_no, scores_no, _ = self.nsr.sample_top_kp(
            self.x, choices=8, max_len=10, use_cache=False,
            unique=False, valid_only=False, simplify=False,
        )
        torch.manual_seed(0)
        beams_c, scores_c, _ = self.nsr.sample_top_kp(
            self.x, choices=8, max_len=10, use_cache=True,
            unique=False, valid_only=False, simplify=False,
        )

        self.assertEqual(beams_no, beams_c)

    def test_sampling_scores_match(self):
        """Sampling with use_cache=True produces the same log-prob scores."""
        torch.manual_seed(0)
        _, scores_no, _ = self.nsr.sample_top_kp(
            self.x, choices=8, max_len=10, use_cache=False,
            unique=False, valid_only=False, simplify=False,
        )
        torch.manual_seed(0)
        _, scores_c, _ = self.nsr.sample_top_kp(
            self.x, choices=8, max_len=10, use_cache=True,
            unique=False, valid_only=False, simplify=False,
        )

        for s_no, s_c in zip(scores_no, scores_c):
            self.assertAlmostEqual(s_no, s_c, places=3)

    def test_sampling_with_temperature(self):
        """Equivalence holds with temperature scaling."""
        torch.manual_seed(99)
        beams_no, _, _ = self.nsr.sample_top_kp(
            self.x, choices=6, max_len=10, temperature=0.5, use_cache=False,
            unique=False, valid_only=False, simplify=False,
        )
        torch.manual_seed(99)
        beams_c, _, _ = self.nsr.sample_top_kp(
            self.x, choices=6, max_len=10, temperature=0.5, use_cache=True,
            unique=False, valid_only=False, simplify=False,
        )

        self.assertEqual(beams_no, beams_c)

    def test_sampling_with_top_k(self):
        """Equivalence holds with top-k filtering."""
        torch.manual_seed(11)
        beams_no, _, _ = self.nsr.sample_top_kp(
            self.x, choices=6, max_len=10, top_k=5, use_cache=False,
            unique=False, valid_only=False, simplify=False,
        )
        torch.manual_seed(11)
        beams_c, _, _ = self.nsr.sample_top_kp(
            self.x, choices=6, max_len=10, top_k=5, use_cache=True,
            unique=False, valid_only=False, simplify=False,
        )

        self.assertEqual(beams_no, beams_c)

    def test_sampling_mini_batch(self):
        """Equivalence holds when choices exceed batch_size (forces mini-batching)."""
        torch.manual_seed(3)
        beams_no, scores_no, _ = self.nsr.sample_top_kp(
            self.x, choices=10, max_len=8, batch_size=4, use_cache=False,
            unique=False, valid_only=False, simplify=False,
        )
        torch.manual_seed(3)
        beams_c, scores_c, _ = self.nsr.sample_top_kp(
            self.x, choices=10, max_len=8, batch_size=4, use_cache=True,
            unique=False, valid_only=False, simplify=False,
        )

        self.assertEqual(beams_no, beams_c)
        for s_no, s_c in zip(scores_no, scores_c):
            self.assertAlmostEqual(s_no, s_c, places=3)
