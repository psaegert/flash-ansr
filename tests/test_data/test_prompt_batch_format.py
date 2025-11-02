import copy

import torch

from flash_ansr.data import FlashANSRDataset
from flash_ansr.model.tokenizer import Tokenizer


class _DummySkeletonPool:
    simplipy_engine = None
    n_support_prior_config = {'kwargs': {'max_value': 4}}


def test_collate_preserves_prompt_metadata_alignment() -> None:
    tokenizer = Tokenizer(
        vocab=['x1', 'x2', 'x3'],
        special_tokens=['<pad>', '<bos>', '<eos>', '<prompt>', '</prompt>', '<expression>'],
    )
    dataset = FlashANSRDataset(skeleton_pool=_DummySkeletonPool(), tokenizer=tokenizer, padding='zero')

    raw_batch = {
        'input_ids': [
            [tokenizer['<bos>'], tokenizer['x1'], tokenizer['x2']],
            [tokenizer['<bos>'], tokenizer['x3']],
        ],
        'x_tensors': [
            torch.zeros((2, 2), dtype=torch.float32),
            torch.ones((2, 2), dtype=torch.float32),
        ],
        'y_tensors': [
            torch.ones((2, 2), dtype=torch.float32),
            torch.zeros((2, 2), dtype=torch.float32),
        ],
        'constants': [
            [0.1, 0.2],
            [0.3],
        ],
        'input_num': [
            [float('nan'), 1.0, float('nan')],
            [float('nan'), float('nan')],
        ],
        'prompt_mask': [
            [False, True, True],
            [False, True],
        ],
        'prompt_metadata': [
            {'allowed_terms': [['+', 'x1']], 'include_terms': [['x1']], 'exclude_terms': []},
            {'allowed_terms': [], 'include_terms': [], 'exclude_terms': []},
        ],
    }

    batch = copy.deepcopy(raw_batch)
    collated = dataset.collate(batch, device='cpu')

    assert tuple(collated['input_ids'].shape) == (2, 3)
    assert tuple(collated['prompt_mask'].shape) == (2, 3)
    assert collated['labels'].shape == (2, 2)
    assert not collated['prompt_mask'][0, 0].item()
    assert collated['prompt_mask'][0, 2].item()
    assert not collated['prompt_mask'][1, 2].item()  # padding introduced for shorter sample

    assert collated['labels'][0].tolist() == collated['input_ids'][0, 1:].tolist()
    assert collated['labels'][1].tolist() == collated['input_ids'][1, 1:].tolist()

    assert len(collated['prompt_metadata']) == 2
    assert collated['prompt_metadata'][0]['allowed_terms'] == raw_batch['prompt_metadata'][0]['allowed_terms']
    assert collated['prompt_metadata'][1]['include_terms'] == []
