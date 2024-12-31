from scipy.special import jn, yn


def j2(x: float) -> float:
    return jn(2, x)


def y2(x: float) -> float:
    return yn(2, x)
