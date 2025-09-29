from typing import Any

import torch
import torch_optimizer


class OptimizerFactory():
    @staticmethod
    def get_optimizer(name: str, *args: Any, **kwargs: Any) -> torch.optim.Optimizer:
        if hasattr(torch.optim, name):
            return getattr(torch.optim, name)(*args, **kwargs)
        if hasattr(torch_optimizer, name):
            return getattr(torch_optimizer, name)(*args, **kwargs)

        raise NotImplementedError(f"Optimizer {name} not found in torch.optim, torch_optimizer")


def pw_linear_schedule(step: int, points: list[tuple[int | float | str, int | float | str]]) -> float:
    """Schedules a value based on piecewise linear interpolation between given points."""
    parsed_points: list[tuple[int | float, float]] = [(int(x) if isinstance(x, str) and x.isdigit() else float(x), float(y)) for x, y in points]

    if step <= parsed_points[0][0]:
        return parsed_points[0][1]
    for i in range(1, len(parsed_points)):
        if step <= parsed_points[i][0]:
            x0, y0 = parsed_points[i - 1]
            x1, y1 = parsed_points[i]
            return y0 + (y1 - y0) * (step - x0) / (x1 - x0)
    return parsed_points[-1][1]
