"""Learning rate schedule helpers."""


def pw_linear_schedule(step: int, points: list[tuple[int | float | str, int | float | str]]) -> float:
    """Piecewise-linear interpolation across ``points`` evaluated at ``step``."""
    parsed_points: list[tuple[int | float, float]] = [
        (
            int(x) if isinstance(x, str) and x.isdigit() else float(x),
            float(y),
        )
        for x, y in points
    ]

    if step <= parsed_points[0][0]:
        return parsed_points[0][1]
    for i in range(1, len(parsed_points)):
        if step <= parsed_points[i][0]:
            x0, y0 = parsed_points[i - 1]
            x1, y1 = parsed_points[i]
            return y0 + (y1 - y0) * (step - x0) / (x1 - x0)
    return parsed_points[-1][1]
