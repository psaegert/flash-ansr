"""Worker pools are sized by the cores this process may run on (a SLURM allocation, ``taskset``), not by the
machine's core count: a pool of ``os.cpu_count()`` workers oversubscribes a job that was given fewer cores."""
import os

from flash_ansr.estimator_config import ComputeConfig
from flash_ansr.utils import available_cores


def test_the_process_count_is_used_where_python_has_it(monkeypatch):
    monkeypatch.setattr(os, "process_cpu_count", lambda: 3, raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    assert available_cores() == 3


def test_the_affinity_mask_is_used_before_python_3_13(monkeypatch):
    monkeypatch.delattr(os, "process_cpu_count", raising=False)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 5}, raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    assert available_cores() == 2


def test_the_machine_count_is_the_last_resort_and_at_least_one(monkeypatch):
    monkeypatch.delattr(os, "process_cpu_count", raising=False)
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 7)
    assert available_cores() == 7
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    assert available_cores() == 1


def test_the_refiner_pool_defaults_to_the_cores_the_process_was_given(monkeypatch):
    monkeypatch.setattr(os, "process_cpu_count", lambda: 4, raising=False)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1, 2, 3}, raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    assert ComputeConfig().resolved_workers == 4
    assert ComputeConfig(workers=9).resolved_workers == 9 and ComputeConfig(workers=0).resolved_workers == 0
