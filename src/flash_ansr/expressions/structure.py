"""Structural helpers for reasoning about expression trees."""


def generate_ubi_dist(
    max_n_operators: int,
    n_leaves: int,
    n_unary_operators: int,
    n_binary_operators: int,
) -> list[list[int]]:
    """Pre-compute the number of possible trees for operator/leaf counts."""
    D: list[list[int]] = []
    D.append([0] + ([n_leaves ** i for i in range(1, 2 * max_n_operators + 1)]))
    for n in range(1, 2 * max_n_operators + 1):
        s = [0]
        for e in range(1, 2 * max_n_operators - n + 1):
            s.append(
                n_leaves * s[e - 1]
                + n_unary_operators * D[n - 1][e]
                + n_binary_operators * D[n - 1][e + 1]
            )
        D.append(s)
    assert all(len(D[i]) >= len(D[i + 1]) for i in range(len(D) - 1))
    D = [[D[j][i] for j in range(len(D)) if i < len(D[j])] for i in range(max(len(x) for x in D))]
    return D
