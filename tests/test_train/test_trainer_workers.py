"""The validation pool's worker count is its own knob (kept alive next to the training pool)."""
from flash_ansr.train.train import Trainer


def _bare(num_workers, validate_num_workers):
    trainer = object.__new__(Trainer)
    trainer.num_workers = num_workers
    trainer.validate_num_workers = validate_num_workers
    return trainer


def test_validate_pool_defaults_to_the_training_pool():
    assert _bare(18, None)._effective_validate_num_workers() == 18


def test_validate_pool_override_wins():
    assert _bare(32, 6)._effective_validate_num_workers() == 6
