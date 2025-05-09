import numpy as np
import torch


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
        if np.iscomplexobj(x):
            # Handle complex numbers
            return np.cbrt(x)
        x = np.asarray(x)
        x = np.where(x < 0, -(-x) ** (1 / 3), x ** (1 / 3))
        return x

    if isinstance(x, torch.Tensor):
        # Handle torch tensors
        if x.dtype == torch.complex64 or x.dtype == torch.complex128:
            # Handle complex numbers
            return x ** (1 / 3)
        x = torch.where(x < 0, -(-x) ** (1 / 3), x ** (1 / 3))
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
        if np.iscomplexobj(x):
            # Handle complex numbers
            return x ** (1 / 5)
        x = np.asarray(x)
        x = np.where(x < 0, -(-x) ** (1 / 5), x ** (1 / 5))
        return x

    if isinstance(x, torch.Tensor):
        # Handle torch tensors
        if x.dtype == torch.complex64 or x.dtype == torch.complex128:
            # Handle complex numbers
            return x ** (1 / 5)
        x = torch.where(x < 0, -(-x) ** (1 / 5), x ** (1 / 5))
        return x

    if not isinstance(x, complex) and x < 0:
        # Discard imaginary component
        return - (-x) ** (1 / 5)
    else:
        return x ** (1 / 5)
