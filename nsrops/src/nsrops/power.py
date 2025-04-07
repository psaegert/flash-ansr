import numpy as np


def pow2(x: float) -> float:
    return x ** 2


def pow3(x: float) -> float:
    return x ** 3


def pow4(x: float) -> float:
    return x ** 4


def pow5(x: float) -> float:
    return x ** 5


def pow1_2(x: float) -> float:
    return x ** 0.5


def pow1_3(x: float) -> float:
    if isinstance(x, np.ndarray):
        # Handle numpy arrays
        if x.dtype == np.complexfloating:
            # Handle complex numbers
            return np.cbrt(x)
        x = np.asarray(x)
        x = np.where(x < 0, -(-x) ** (1 / 3), x ** (1 / 3))
        return x

    if not isinstance(x, complex) and x < 0:
        # Discard imaginary component
        return - (-x) ** (1 / 3)
    else:
        return x ** (1 / 3)


def pow1_4(x: float) -> float:
    return x ** 0.25


def pow1_5(x: float) -> float:
    if isinstance(x, np.ndarray):
        # Handle numpy arrays
        if x.dtype == np.complexfloating:
            # Handle complex numbers
            return x ** (1 / 5)
        x = np.asarray(x)
        x = np.where(x < 0, -(-x) ** (1 / 5), x ** (1 / 5))
        return x

    if not isinstance(x, complex) and x < 0:
        # Discard imaginary component
        return - (-x) ** (1 / 5)
    else:
        return x ** (1 / 5)
