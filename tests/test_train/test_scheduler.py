from flash_ansr.train.scheduler import BatchSizeSchedulerFactory, LinearIncreaseBatchSizeScheduler


def test_batch_size_scheduler_factory():
    scheduler = BatchSizeSchedulerFactory.get_scheduler('LinearIncrease', min_batch_size=1, max_batch_size=10, total_steps=100, gradient_accumulation_steps=1)

    assert isinstance(scheduler, LinearIncreaseBatchSizeScheduler)


def test_linear_increase_batch_size_scheduler():
    scheduler = LinearIncreaseBatchSizeScheduler(low=1, high=10, total_steps=100)

    assert scheduler.steps == 0
    assert scheduler.batch_size == 10

    scheduler.step(1)

    assert scheduler.steps == 1
    assert scheduler.batch_size == 1
