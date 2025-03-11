import unittest
import shutil
import tempfile

import torch

from flash_ansr import FlashANSRTransformer, ExpressionSpace, get_path, SetTransformer


class TestFlashANSRTransformer(unittest.TestCase):
    def setUp(self) -> None:
        self.save_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.save_dir)

    def test_nsr_forward(self):
        nsr = FlashANSRTransformer(
            expression_space=ExpressionSpace.from_config(get_path('configs', 'test', 'expression_space.yaml')),
            encoder_max_n_variables=1024,
            encoder='SetTransformer',
            encoder_kwargs={'n_seeds': 10})

        batch_size = 257
        sequence_length = 17

        x = torch.rand(batch_size, 10, 1024)
        input_tokens = torch.randint(low=len(nsr.expression_space.tokenizer.special_tokens), high=len(nsr.expression_space.tokenizer), size=(batch_size, sequence_length))

        random_padding_beginnings = torch.randint(0, sequence_length, (batch_size,))

        for i in range(32):
            input_tokens[i, random_padding_beginnings[i]:] = nsr.expression_space.tokenizer['<pad>']

        logits, _ = nsr.forward(input_tokens, x, numeric_head=True)
        assert logits.shape == (batch_size, sequence_length, len(nsr.expression_space.tokenizer))

    def test_nsr_beam_search(self):
        nsr = FlashANSRTransformer(
            expression_space=ExpressionSpace.from_config(get_path('configs', 'test', 'expression_space.yaml')),
            encoder_max_n_variables=6,
            encoder='SetTransformer',
            encoder_kwargs={'n_seeds': 10})

        x = torch.rand(13, 6)

        beams, scores, _ = nsr.beam_search(x, beam_width=4, max_len=10)

        assert len(beams) == 4
        assert len(scores) == 4

    def test_nsr_sample_top_kp(self):
        nsr = FlashANSRTransformer(
            expression_space=ExpressionSpace.from_config(get_path('configs', 'test', 'expression_space.yaml')),
            encoder_max_n_variables=6,
            encoder='SetTransformer',
            encoder_kwargs={'n_seeds': 10})

        x = torch.rand(13, 6)

        beams, scores, _ = nsr.sample_top_kp(x, choices=4, max_len=10)

        assert len(beams) <= 4
        assert len(scores) <= 4

    def test_nsr_from_config(self):
        nsr = FlashANSRTransformer.from_config(get_path('configs', 'test', filename='nsr.yaml'))

        assert isinstance(nsr, FlashANSRTransformer)
        assert isinstance(nsr.encoder, SetTransformer)
        assert isinstance(nsr.decoder, torch.nn.TransformerDecoder)

        assert nsr.expression_space.variables == ['x1', 'x2', 'x3']

        x = torch.rand(256, 10, 4)
        input_tokens = torch.randint(5, 10, (256, 17))

        random_padding_beginnings = torch.randint(0, 17, (256,))

        for i in range(32):
            input_tokens[i, random_padding_beginnings[i]:] = 0

        logits, _ = nsr.forward(input_tokens, x)

        assert logits.shape == (256, 17, 33)

    def test_save_load_relative(self):
        # Create from config
        nsr_config_path = get_path('configs', 'test', 'nsr.yaml')
        nsr = FlashANSRTransformer.from_config(nsr_config_path)

        # Save
        saved_model_path = self.save_dir
        nsr.save(saved_model_path, config=nsr_config_path, reference='relative')

        # Re-load
        nsr_reload_config, nsr_reload = FlashANSRTransformer.load(saved_model_path)

        for param_nsr, param_nsr_reload in zip(nsr.parameters(), nsr_reload.parameters()):
            assert param_nsr.data.eq(param_nsr_reload.data).all()

    def test_save_load_absolute(self):
        # Create from config
        nsr_config_path = get_path('configs', 'test', 'nsr.yaml')
        nsr = FlashANSRTransformer.from_config(nsr_config_path)

        # Save
        saved_model_path = self.save_dir
        nsr.save(saved_model_path, config=nsr_config_path, reference='absolute')

        # Re-load
        nsr_reload_config, nsr_reload = FlashANSRTransformer.load(saved_model_path)

        for param_nsr, param_nsr_reload in zip(nsr.parameters(), nsr_reload.parameters()):
            assert param_nsr.data.eq(param_nsr_reload.data).all()

    def test_save_load_project(self):
        # Create from config
        nsr_config_path = get_path('configs', 'test', 'nsr.yaml')
        nsr = FlashANSRTransformer.from_config(nsr_config_path)

        # Save
        saved_model_path = self.save_dir
        nsr.save(saved_model_path, config=nsr_config_path, reference='project')

        # Re-load
        nsr_reload_config, nsr_reload = FlashANSRTransformer.load(saved_model_path)

        for param_nsr, param_nsr_reload in zip(nsr.parameters(), nsr_reload.parameters()):
            assert param_nsr.data.eq(param_nsr_reload.data).all()

    def test_masking(self):
        nsr = FlashANSRTransformer.from_config(get_path('configs', 'test', 'nsr.yaml'))
        nsr.eval()

        B = 7
        S = 13
        x = torch.rand(B, 10, 4)
        input_tokens = torch.randint(5, 10, (B, S))

        random_padding_beginnings = torch.randint(5, S, (B,))

        for i in range(B):
            input_tokens[i, random_padding_beginnings[i]:] = 0

        modified_input = input_tokens.clone()
        modified_input[:, 3] = 3

        output, _ = nsr.forward(input_tokens, x)
        modified_output, _ = nsr.forward(modified_input, x, numeric_head=False)

        assert torch.allclose(output[:, :3], modified_output[:, :3])
        assert not torch.allclose(output[:, 3:], modified_output[:, 3:])
