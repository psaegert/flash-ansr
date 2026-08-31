"""The tagged canonicalization follows the catalog's configured target canon (task #92)."""
from flash_ansr.data.streaming import _tagged_canonical_mode


class _Cat:
    simplify_mode = "permissive"


class _Bare:
    pass


def test_configured_mode_is_resolved():
    assert _tagged_canonical_mode(_Cat()) == "permissive"


def test_catalogs_without_the_knob_keep_the_engine_default():
    assert _tagged_canonical_mode(_Bare()) is None


def test_tagged_canonical_passes_the_mode_through(monkeypatch):
    import symbolic_data.token_ops as to

    seen = {}

    class _Engine:
        def to_tagged(self, tokens):
            return list(tokens)

        def simplify(self, tokens, mode=None, **kw):
            seen["mode"] = mode
            return list(tokens)

    to.tagged_canonical(_Engine(), ["x1"], mode="permissive")
    assert seen["mode"] == "permissive"
    to.tagged_canonical(_Engine(), ["x1"])
    assert seen["mode"] is None
