from typing import Any

import torch
from torch.optim.lr_scheduler import LRScheduler


class LRSchedulerFactory():
    @staticmethod
    def get_scheduler(name: str, optimizer: Any, *args: Any, **kwargs: Any) -> LRScheduler:
        if name == 'WarmupLinearAnnealing':
            # Increase the learning rate linearly during the warmup phase and decrease it linearly during the annealing phase
            # https://www.desmos.com/calculator/xqgt1s1x4t

            min_lr = float(kwargs['min_lr'])
            max_lr = float(kwargs['max_lr'])
            warmup_steps = int(kwargs['warmup_steps'])
            total_steps = int(kwargs['total_steps'])

            return torch.optim.lr_scheduler.LambdaLR(
                optimizer=optimizer,
                lr_lambda=lambda step: min_lr + (max_lr - min_lr) * step / warmup_steps if step < warmup_steps else max_lr - (max_lr - min_lr) * (step - warmup_steps) / (total_steps - warmup_steps))

        if name == 'Warmup':
            # Increase the learning rate linearly during the warmup phase
            min_lr = float(kwargs['min_lr'])
            max_lr = float(kwargs['max_lr'])
            warmup_steps = int(kwargs['warmup_steps'])

            return torch.optim.lr_scheduler.LambdaLR(
                optimizer=optimizer,
                lr_lambda=lambda step: min_lr + (max_lr - min_lr) * step / warmup_steps if step < warmup_steps else max_lr)

        if name == 'Trapezoidal':
            # Three phases:
            # 1. Warmup the learning rate linearly from min_lr to max_lr in the first warmup_steps steps
            # 2. Keep the learning rate at max_lr until the end of the plateau_steps steps. If plateau_steps is negative, keep the learning rate at max_lr until total_steps + plateau_steps
            # 3. Anneal the learning rate linearly from max_lr to min_lr in the last (total_steps - plateau_steps) steps
            # https://www.desmos.com/calculator/qfmk2dnbew

            min_lr = float(kwargs['min_lr'])
            max_lr = float(kwargs['max_lr'])
            warmup_steps = int(kwargs['warmup_steps'])
            total_steps = int(kwargs['total_steps'])

            if 'plateau_steps' not in kwargs and 'annealing_steps' not in kwargs:
                raise ValueError("Either plateau_steps or annealing_steps must be provided.")

            if 'plateau_steps' in kwargs:
                plateau_steps = int(kwargs['plateau_steps'])
                annealing_steps = total_steps - plateau_steps - warmup_steps
            else:
                annealing_steps = int(kwargs['annealing_steps'])
                plateau_steps = total_steps - annealing_steps - warmup_steps

            def lr_schedule(step: int) -> float:
                # Phase 1
                if step < warmup_steps:
                    return min_lr + (max_lr - min_lr) * step / warmup_steps
                # Phase 2
                if step < warmup_steps + plateau_steps:
                    return max_lr
                # Phase 3
                if step < total_steps:
                    return max_lr - (max_lr - min_lr) * (step - warmup_steps - plateau_steps) / annealing_steps
                # Out of range
                return min_lr

            return torch.optim.lr_scheduler.LambdaLR(
                optimizer=optimizer,
                lr_lambda=lr_schedule)

        if hasattr(torch.optim.lr_scheduler, name):
            return getattr(torch.optim.lr_scheduler, name)(optimizer=optimizer, *args, **kwargs)

        raise NotImplementedError(f"Scheduler {name} not found in torch.optim.lr_scheduler or not supported.")
