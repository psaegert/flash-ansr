import numpy as np

from flash_ansr.utils.tensor_ops import mask_unused_variable_columns


def test_mask_unused_variable_columns_zeroes_unused_features():
    X = np.ones((4, 3), dtype=np.float32)
    arrays = [X]

    mask_unused_variable_columns(
        arrays,
        variables=["x1", "x2", "x3"],
        skeleton_tokens=["x1", "x3"],
        padding="zero",
    )

    np.testing.assert_array_equal(arrays[0][:, 0], np.ones(4))
    np.testing.assert_array_equal(arrays[0][:, 2], np.ones(4))
    assert np.all(arrays[0][:, 1] == 0)


def test_mask_unused_variable_columns_noop_for_random_padding():
    X = np.arange(6, dtype=np.float32).reshape(3, 2)
    arrays = [X.copy()]

    mask_unused_variable_columns(
        arrays,
        variables=["x1", "x2"],
        skeleton_tokens=["x2"],
        padding="random",
    )

    np.testing.assert_array_equal(arrays[0], X)
