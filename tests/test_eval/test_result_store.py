import pytest

from flash_ansr.eval.result_store import ResultStore


def test_append_backfills_missing_keys_and_defaults_new_fields() -> None:
    store = ResultStore()
    store.append({"a": 1})

    # Append a record that introduces a new key and omits an existing one
    store.append({"b": 2})

    snapshot = store.snapshot()
    assert snapshot["a"] == [1, None]
    assert snapshot["b"] == [None, 2]


def test_extend_handles_new_and_missing_fields() -> None:
    store = ResultStore({"a": [1]})

    # Extend with records that add "c" but omit "a" in some entries
    store.extend({"a": [2], "c": [4]})

    snapshot = store.snapshot()
    assert snapshot["a"] == [1, 2]
    assert snapshot["c"] == [None, 4]


@pytest.mark.parametrize(
    "records",
    [
        {"a": [5, 6]},
        {"b": [7, 8]},
    ],
)
def test_extend_preserves_lengths_with_existing_store(records) -> None:
    store = ResultStore({"a": [1], "b": [None]})
    store.extend(records)
    lengths = {len(values) for values in store.snapshot().values()}
    assert lengths == {1 + len(next(iter(records.values())))}
