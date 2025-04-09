import numpy as np


def inv(x: float) -> float:
    # numpy will handle the x = 0 case
    if isinstance(x, np.ndarray):
        return 1 / x

    # Manually handle scalar case
    if x == 0:
        return float('inf')

    # All safe
    return 1 / x


def div(x: float, y: float) -> float:
    # numpy will handle the x = 0 case
    if isinstance(y, np.ndarray):
        return x / y

    # Manually handle scalar case
    if y == 0:
        # When x is an iterable, multiply with infinity to let the sign determine the result
        if isinstance(x, np.ndarray):
            return x * float('inf')

        # When x is a scalar, return inf or -inf depending on the sign of x
        if not isinstance(x, complex):
            if x > 0:
                return float('inf')
            elif x < 0:
                return float('-inf')

        # Both x and y are zero.
        # Return NaN to indicate an undefined result
        return float('nan')

    # All safe
    return x / y


def mult2(x: float) -> float:
    return 2 * x


def mult3(x: float) -> float:
    return 3 * x


def mult4(x: float) -> float:
    return 4 * x


def mult5(x: float) -> float:
    return 5 * x


def div2(x: float) -> float:
    return x / 2


def div3(x: float) -> float:
    return x / 3


def div4(x: float) -> float:
    return x / 4


def div5(x: float) -> float:
    return x / 5
