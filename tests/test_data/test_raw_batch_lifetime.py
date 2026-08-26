"""The raw batches ``iterate`` yields are views into the worker ring, valid until refill.

Anything that outlives its batch must COPY. The paired constant-span eval is the one
consumer that does (`Trainer._detach_raw_batch`); before it did, holding batch 0 and
reading it after the loop was a hard SIGSEGV, not an exception -- so this file's job is
to keep that copy honest. A regression here takes the whole pytest process down.
"""
import torch

from flash_ansr import get_path
from flash_ansr.data import FlashANSRDataset
from flash_ansr.train.train import _detach_raw_batch

FIELDS = ("skeleton", "constants", "x_tensors", "y_tensors", "data_attn_mask")


def test_detached_raw_batch_outlives_the_worker_ring() -> None:
    dataset = FlashANSRDataset.from_config(get_path("configs", "test", "dataset_val.yaml"))
    try:
        kept = None
        expected_sum = None
        for index, batch in enumerate(dataset.iterate(size=12, batch_size=4, num_workers=1,
                                                      max_seq_len=64)):
            if kept is None:
                kept = _detach_raw_batch(batch)
                expected_sum = float(batch["x_tensors"].sum())
            if index >= 2:
                break
    finally:
        dataset.shutdown()

    assert kept is not None
    # Reading AFTER the loop and after shutdown: the copy owns its memory.
    assert float(kept["x_tensors"].sum()) == expected_sum
    for field in FIELDS:
        assert field in kept, field


def test_detach_copies_rather_than_aliases() -> None:
    source = {
        "x_tensors": torch.ones((2, 3)),
        "constants": [torch.tensor([1.5]), torch.tensor([])],
        "skeleton": [["+", "<constant>", "x1"]],
        "expression": ["ignored"],          # not a field the paired eval reads
    }
    kept = _detach_raw_batch(source)

    source["x_tensors"].fill_(9.0)
    source["constants"][0].fill_(9.0)
    source["skeleton"][0].append("mutated")

    assert float(kept["x_tensors"].sum()) == 6.0
    assert float(kept["constants"][0].item()) == 1.5
    assert kept["skeleton"][0] == ["+", "<constant>", "x1"]
    assert "expression" not in kept, "only the fields the paired eval reads are kept"
