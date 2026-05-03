import unittest
import shutil
import tempfile

import torch

from flash_ansr.model.decoders import TransformerDecoder
from flash_ansr import FlashANSRModel, get_path, SetTransformer


class TestFlashANSRTransformer(unittest.TestCase):
    def setUp(self) -> None:
        self.save_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.save_dir)

    def test_nsr_forward(self):
        nsr = FlashANSRModel.from_config(get_path('configs', 'test', 'model.yaml'))

        batch_size = 257
        sequence_length = 17

        x = torch.rand(batch_size, 10, 11)
        input_tokens = torch.randint(low=len(nsr.tokenizer.special_tokens), high=len(nsr.tokenizer), size=(batch_size, sequence_length))

        random_padding_beginnings = torch.randint(0, sequence_length, (batch_size,))

        for i in range(batch_size):
            input_tokens[i, random_padding_beginnings[i]:] = nsr.tokenizer['<pad>']

        print(input_tokens.shape, x.shape)

        logits = nsr.forward(input_tokens, x)
        assert logits.shape == (batch_size, sequence_length, len(nsr.tokenizer))

    def test_nsr_beam_search(self):
        nsr = FlashANSRModel.from_config(get_path('configs', 'test', 'model.yaml'))

        x = torch.rand(13, 11)

        beams, scores, _ = nsr.beam_search(x, beam_width=4, max_len=10)

        assert len(beams) == 4
        assert len(scores) == 4

    def test_beam_search_eos_not_dropped(self):
        """Regression test: EOS candidates ranked below the beam_width-th non-EOS
        must still be registered as completed sequences.

        With beam_width=1 the single active-beam slot fills on the very first
        non-EOS candidate.  Under the old (buggy) code any EOS at rank ≥ 1 was
        silently dropped, leaving no completed sequence at all.  With the fix the
        returned sequence must be EOS-terminated.
        """
        nsr = FlashANSRModel.from_config(get_path('configs', 'test', 'model.yaml'))
        eos_id = nsr.tokenizer['<eos>']

        for seed in range(5):
            torch.manual_seed(seed)
            x = torch.rand(13, 11)
            beams, scores, _ = nsr.beam_search(
                x,
                beam_width=1,
                max_len=20,
                unique=False,
                limit_expansions=False,
            )
            assert len(beams) >= 1, f"seed={seed}: no beams returned"
            assert beams[0][-1] == eos_id, (
                f"seed={seed}: top beam is not EOS-terminated — got token {beams[0][-1]}"
            )

    def test_beam_search_active_beams_not_mixed_with_completed(self):
        """Regression test: active (max-len) beams must not displace EOS-terminated
        sequences in the output.

        Under the old code, active beams were always appended to combined_sequences
        and then sorted by score.  Because active beams carry no EOS log-probability
        penalty their cumulative log-probs are higher, so they ranked above completed
        sequences and were returned — producing beams without </expression> that
        crashed the downstream refiner.

        The fix gates the active-beam fallback: active beams are only included when
        the completed pool has fewer than beam_width entries.
        """
        nsr = FlashANSRModel.from_config(get_path('configs', 'test', 'model.yaml'))

        for seed in range(5):
            torch.manual_seed(seed)
            x = torch.rand(13, 11)
            beams, _, completed = nsr.beam_search(
                x,
                beam_width=4,
                max_len=20,
                unique=False,
                limit_expansions=False,
            )
            n_completed = sum(completed)

            # If the completed pool filled the beam, no active beams should appear.
            if n_completed >= 4:
                assert all(completed), (
                    f"seed={seed}: active beams mixed into output despite "
                    f"{n_completed} completed sequences being available"
                )

    def test_beam_search_truncated_beams_are_parseable(self):
        """Regression test: all returned beams must be parseable even when max_len
        forces truncation before an EOS token is emitted.

        With a very small max_len the beam search loop ends with active (non-EOS)
        sequences that lack </expression>.  The fixed fallback appends </expression>
        to any such sequence before returning it, so extract_expression_from_beam
        must never raise ValueError regardless of max_len.
        """
        nsr = FlashANSRModel.from_config(get_path('configs', 'test', 'model.yaml'))

        for seed in range(5):
            torch.manual_seed(seed)
            x = torch.rand(13, 11)
            beams, _, _ = nsr.beam_search(
                x,
                beam_width=4,
                max_len=5,
                unique=False,
                limit_expansions=False,
            )
            for i, beam in enumerate(beams):
                try:
                    nsr.tokenizer.extract_expression_from_beam(beam)
                except ValueError as exc:
                    raise AssertionError(
                        f"seed={seed}, beam {i}: extract_expression_from_beam raised ValueError — {exc}"
                    ) from exc

    def test_nsr_sample_top_kp(self):
        nsr = FlashANSRModel.from_config(get_path('configs', 'test', 'model.yaml'))

        x = torch.rand(13, 11)

        try:
            beams, scores, _ = nsr.sample_top_kp(x, choices=4, max_len=10)
        except ValueError:
            beams, scores = [], []

        assert len(beams) <= 4
        assert len(scores) <= 4

    def test_nsr_from_config(self):
        nsr = FlashANSRModel.from_config(get_path('configs', 'test', filename='model.yaml'))

        assert isinstance(nsr, FlashANSRModel)
        assert isinstance(nsr.encoder, SetTransformer)
        assert isinstance(nsr.decoder, TransformerDecoder)

        x = torch.rand(256, 10, 11)
        input_tokens = torch.randint(5, 10, (256, 17))

        random_padding_beginnings = torch.randint(0, 17, (256,))

        for i in range(32):
            input_tokens[i, random_padding_beginnings[i]:] = 0

        logits = nsr.forward(input_tokens, x)

        assert logits.shape == (256, 17, len(nsr.tokenizer))

    def test_save_load_relative(self):
        # Create from config
        nsr_config_path = get_path('configs', 'test', 'model.yaml')
        nsr = FlashANSRModel.from_config(nsr_config_path)

        # Save
        saved_model_path = self.save_dir
        nsr.save(saved_model_path, config=nsr_config_path, reference='relative')

        # Re-load
        nsr_reload_config, nsr_reload = FlashANSRModel.load(saved_model_path)

        for param_nsr, param_nsr_reload in zip(nsr.parameters(), nsr_reload.parameters()):
            assert param_nsr.data.eq(param_nsr_reload.data).all()

    def test_save_load_absolute(self):
        # Create from config
        nsr_config_path = get_path('configs', 'test', 'model.yaml')
        nsr = FlashANSRModel.from_config(nsr_config_path)

        # Save
        saved_model_path = self.save_dir
        nsr.save(saved_model_path, config=nsr_config_path, reference='absolute')

        # Re-load
        nsr_reload_config, nsr_reload = FlashANSRModel.load(saved_model_path)

        for param_nsr, param_nsr_reload in zip(nsr.parameters(), nsr_reload.parameters()):
            assert param_nsr.data.eq(param_nsr_reload.data).all()

    def test_save_load_project(self):
        # Create from config
        nsr_config_path = get_path('configs', 'test', 'model.yaml')
        nsr = FlashANSRModel.from_config(nsr_config_path)

        # Save
        saved_model_path = self.save_dir
        nsr.save(saved_model_path, config=nsr_config_path, reference='project')

        # Re-load
        nsr_reload_config, nsr_reload = FlashANSRModel.load(saved_model_path)

        for param_nsr, param_nsr_reload in zip(nsr.parameters(), nsr_reload.parameters()):
            assert param_nsr.data.eq(param_nsr_reload.data).all()

    def test_masking(self):
        nsr = FlashANSRModel.from_config(get_path('configs', 'test', 'model.yaml'))
        nsr.eval()

        B = 7
        S = 13
        x = torch.rand(B, 10, 11)
        input_tokens = torch.randint(5, 10, (B, S))

        random_padding_beginnings = torch.randint(5, S, (B,))

        for i in range(B):
            input_tokens[i, random_padding_beginnings[i]:] = 0

        modified_input = input_tokens.clone()
        modified_input[:, 3] = 3

        output = nsr.forward(input_tokens, x)
        modified_output = nsr.forward(modified_input, x)

        assert torch.allclose(output[:, :3], modified_output[:, :3])
        assert not torch.allclose(output[:, 3:], modified_output[:, 3:])
