import unittest
import shutil
import tempfile

import torch

from simplipy import SimpliPyEngine

from flash_ansr.model.transformer import Tokenizer, TransformerDecoder
from flash_ansr import FlashANSRModel, get_path, SetTransformer


class TestFlashANSRTransformer(unittest.TestCase):
    def setUp(self) -> None:
        self.save_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.save_dir)

    def test_nsr_forward(self):
        nsr = FlashANSRModel(
            simplipy_engine=SimpliPyEngine.from_config(get_path('configs', 'test', 'simplipy_engine.yaml')),
            tokenizer=Tokenizer.from_config(get_path('configs', 'test', 'tokenizer.yaml')),
            encoder_max_n_variables=1024,
            encoder_n_seeds=10)

        batch_size = 257
        sequence_length = 17

        x = torch.rand(batch_size, 10, 1024)
        input_tokens = torch.randint(low=len(nsr.tokenizer.special_tokens), high=len(nsr.tokenizer), size=(batch_size, sequence_length))

        random_padding_beginnings = torch.randint(0, sequence_length, (batch_size,))

        for i in range(32):
            input_tokens[i, random_padding_beginnings[i]:] = nsr.tokenizer['<pad>']

        logits = nsr.forward(input_tokens, x)
        assert logits.shape == (batch_size, sequence_length, len(nsr.tokenizer))

    def test_nsr_beam_search(self):
        nsr = FlashANSRModel(
            simplipy_engine=SimpliPyEngine.from_config(get_path('configs', 'test', 'simplipy_engine.yaml')),
            tokenizer=Tokenizer.from_config(get_path('configs', 'test', 'tokenizer.yaml')),
            encoder_max_n_variables=6,
            encoder_n_seeds=10)

        x = torch.rand(13, 6)

        beams, scores, _ = nsr.beam_search(x, beam_width=4, max_len=10)

        assert len(beams) == 4
        assert len(scores) == 4

    def test_nsr_sample_top_kp(self):
        nsr = FlashANSRModel(
            simplipy_engine=SimpliPyEngine.from_config(get_path('configs', 'test', 'simplipy_engine.yaml')),
            tokenizer=Tokenizer.from_config(get_path('configs', 'test', 'tokenizer.yaml')),
            encoder_max_n_variables=6,
            encoder_n_seeds=10)

        x = torch.rand(13, 6)

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

        assert logits.shape == (256, 17, 50)

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
