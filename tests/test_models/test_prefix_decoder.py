"""The prefix decoder (v26, ``decoder_data_mode: prefix``): the encoder memory is a prefix of the
decoder's own sequence, ``tokens[0] <data> m_1..m_S </data> tokens[1:]``, bidirectional over the data
block, causal over the expression, no cross-attention. The tests pin the mask, the one-state-per-token
contract, causality over the tokens, bidirectionality inside the block, the null-memory routing, the
dynamic and static cache paths against the full forward, the optimizer roles, the config surface and
that the v25 default is untouched."""
import pytest
import torch

from flash_ansr import get_path
from flash_ansr.model.decoders.static_kv import StaticKVCache
from flash_ansr.model.decoders.transformer import TransformerDecoder
from flash_ansr.model.flash_ansr_model import FlashANSRModel
from flash_ansr.model.tokenizer import Tokenizer
from flash_ansr.utils.config_io import load_config
from flash_ansr.utils.numeric import NUMERIC_DTYPE

CROSS_KEYS = ("decoder_block_cross_attn_norm", "decoder_cross_attn_kv_norm", "decoder_use_rope_cross_attn")


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    return Tokenizer.from_config(get_path("configs", "test", "tokenizer.yaml"))


@pytest.fixture(scope="module")
def engine():  # type: ignore[no-untyped-def]
    from simplipy import SimpliPyEngine
    return SimpliPyEngine.load("base", install=True)


def _prefix_config() -> dict:
    cfg = load_config(get_path("configs", "test", "model.yaml"))
    cfg["decoder_data_mode"] = "prefix"
    for key in CROSS_KEYS:
        cfg.pop(key, None)
    return cfg


def _model(tokenizer, engine, **overrides):  # type: ignore[no-untyped-def]
    cfg = _prefix_config()
    kwargs = {k: v for k, v in cfg.items() if k not in ("simplipy_engine", "tokenizer")}
    kwargs.update(overrides)
    torch.manual_seed(3)
    model = FlashANSRModel(simplipy_engine=engine, tokenizer=tokenizer, **kwargs)
    model.eval()
    return model, kwargs


def _batch(tokenizer, kwargs, B=2, M=12, L=9):  # type: ignore[no-untyped-def]
    torch.manual_seed(0)
    data = torch.randn(B, M, kwargs["encoder_max_n_variables"], dtype=NUMERIC_DTYPE)
    tokens = torch.randint(0, len(tokenizer), (B, L))
    return tokens, data


class TestMask:
    def test_block_is_bidirectional_and_tokens_causal(self) -> None:
        dec = TransformerDecoder(vocab_size=11, input_dim=None, model_dim=16, n_layers=1, n_heads=2,
                                 max_seq_len=8, data_mode="prefix", memory_len=3, use_rope_self_attn=True)
        assert dec.prefix_len == 5   # <data> + 3 slots + </data>
        mask = dec.prefix_mask(9, torch.device("cpu"))
        block = dec.prefix_len + 1    # tokens[0] and the inserted block
        assert mask[:block, :block].all()                      # every block position sees every other
        assert not mask[:block, block:].any()                  # the block never sees a later token
        tail = mask[block:, :]
        idx = torch.arange(9)
        expected = idx[None, :] <= idx[block:, None]
        assert torch.equal(tail, expected)                    # tokens: causal over everything before

    def test_rope_table_covers_tokens_plus_prefix(self) -> None:
        dec = TransformerDecoder(vocab_size=11, input_dim=None, model_dim=16, n_layers=1, n_heads=2,
                                 max_seq_len=40, data_mode="prefix", memory_len=6, use_rope_self_attn=True)
        assert dec.rope.max_seq_len == 40 + 8 and dec.token_max_seq_len == 40

    def test_rejects_unknown_mode_and_missing_memory_len(self) -> None:
        with pytest.raises(ValueError):
            TransformerDecoder(vocab_size=11, input_dim=None, model_dim=16, n_layers=1, n_heads=2, data_mode="both")
        with pytest.raises(ValueError):
            TransformerDecoder(vocab_size=11, input_dim=None, model_dim=16, n_layers=1, n_heads=2, data_mode="prefix")


class TestForward:
    def test_one_logit_row_per_input_token(self, tokenizer, engine) -> None:  # type: ignore[no-untyped-def]
        model, kwargs = _model(tokenizer, engine)
        tokens, data = _batch(tokenizer, kwargs)
        with torch.no_grad():
            logits = model(tokens, data)
        assert logits.shape == (tokens.shape[0], tokens.shape[1], len(tokenizer))
        assert model.decoder.prefix_len == kwargs["encoder_n_seeds"] + 2
        assert all(layer.cross_attention is None for layer in model.decoder.layers)

    def test_tokens_are_causal(self, tokenizer, engine) -> None:  # type: ignore[no-untyped-def]
        model, kwargs = _model(tokenizer, engine)
        tokens, data = _batch(tokenizer, kwargs)
        other = tokens.clone()
        other[:, 6:] = (other[:, 6:] + 1) % len(tokenizer)
        with torch.no_grad():
            a = model(tokens, data)
            b = model(other, data)
        assert torch.allclose(a[:, :6], b[:, :6], atol=1e-6)
        assert not torch.allclose(a[:, 6:], b[:, 6:])

    def test_first_row_reads_the_data(self, tokenizer, engine) -> None:  # type: ignore[no-untyped-def]
        """The state standing in for tokens[0] is </data>'s: it has seen every memory slot."""
        model, kwargs = _model(tokenizer, engine)
        tokens, data = _batch(tokenizer, kwargs)
        with torch.no_grad():
            a = model(tokens, data)
            b = model(tokens, data * 3.0 + 1.0)
        assert not torch.allclose(a[:, 0], b[:, 0])

    def test_data_block_is_bidirectional(self, tokenizer, engine) -> None:  # type: ignore[no-untyped-def]
        """Perturbing the LAST memory slot changes the first slot's state after one block -- only a
        bidirectional block lets slot 1 read slot S."""
        model, kwargs = _model(tokenizer, engine)
        tokens, data = _batch(tokenizer, kwargs, B=1)
        captured: list[torch.Tensor] = []
        handle = model.decoder.layers[0].register_forward_hook(lambda m, i, o: captured.append(o.detach().clone()))
        with torch.no_grad():
            model(tokens, data)
            memory = model.memory.clone()
            memory[:, -1] += 1.0
            model(tokens, None, memory=memory)
        handle.remove()
        first_slot = 2   # tokens[0], <data>, then slot 1
        assert not torch.allclose(captured[0][:, first_slot], captured[1][:, first_slot])
        assert not torch.allclose(captured[0][:, 1], captured[1][:, 1])   # the <data> tag reads the slots too
        last_token = captured[0].shape[1] - 1                                # and the expression sees the data
        assert not torch.allclose(captured[0][:, last_token], captured[1][:, last_token])

    def test_null_memory_routing(self, tokenizer, engine) -> None:  # type: ignore[no-untyped-def]
        model, kwargs = _model(tokenizer, engine, optional_condition=True)
        tokens, data = _batch(tokenizer, kwargs)
        with torch.no_grad():
            conditioned = model(tokens, data, condition_mask=torch.tensor([True, True]))
            routed = model(tokens, data, condition_mask=torch.tensor([True, False]))
            null_only = model(tokens, None, memory=model.null_memory.expand(2, -1, -1))
        assert torch.allclose(conditioned[0], routed[0], atol=1e-6)
        assert torch.allclose(routed[1], null_only[1], atol=1e-6)
        assert not torch.allclose(conditioned[1], routed[1])

    def test_shared_memory_broadcasts_over_rows(self, tokenizer, engine) -> None:  # type: ignore[no-untyped-def]
        model, kwargs = _model(tokenizer, engine)
        tokens, data = _batch(tokenizer, kwargs, B=3)
        with torch.no_grad():
            memory = model._create_memory(data[:1])
            shared = model(tokens, None, memory=memory)
            each = torch.stack([model(tokens[i:i + 1], None, memory=memory)[0] for i in range(3)])
        assert torch.allclose(shared, each, atol=1e-6)


class TestCaches:
    def test_incremental_matches_full_forward(self, tokenizer, engine) -> None:  # type: ignore[no-untyped-def]
        model, kwargs = _model(tokenizer, engine)
        tokens, data = _batch(tokenizer, kwargs)
        with torch.no_grad():
            full = model(tokens, data)
            pre, cache = model(tokens[:, :4], data, use_cache=True)
            assert cache[0][0][0].shape[2] == 4 + model.decoder.prefix_len
            assert cache[0][1][0].shape[2] == 0      # no cross-attention: a zero-length pair
            rows = [pre[:, -1]]
            for t in range(4, tokens.shape[1]):
                step, cache = model(tokens[:, t:t + 1], None, memory=model.memory, past_key_values=cache, use_cache=True)
                rows.append(step[:, -1])
        assert torch.allclose(torch.stack(rows, 1), full[:, 3:], atol=1e-5)

    def test_static_matches_dynamic(self, tokenizer, engine) -> None:  # type: ignore[no-untyped-def]
        model, kwargs = _model(tokenizer, engine)
        tokens, data = _batch(tokenizer, kwargs)
        ok, reason = model.supports_static_decode()
        assert ok, reason
        block0 = model.decoder.layers[0]
        with torch.no_grad():
            full = model(tokens, data)
            pre, cache = model(tokens[:, :4], data, use_cache=True)
            static = StaticKVCache(n_layers=len(model.decoder.layers), batch=2, n_heads=block0.self_attention.n_heads,
                                   head_dim=block0.self_attention.head_dim, max_len=tokens.shape[1] + model.decoder.prefix_len,
                                   device=tokens.device, dtype=pre.dtype)
            static.seed_from_dynamic(cache)
            rows = [pre[:, -1]]
            for t in range(4, tokens.shape[1]):
                rows.append(model.forward_static(tokens[:, t:t + 1], None, model.memory, static, position=t)[:, -1])
        assert torch.allclose(torch.stack(rows, 1), full[:, 3:], atol=1e-5)

    def test_sampling_runs_on_both_paths(self, tokenizer, engine) -> None:  # type: ignore[no-untyped-def]
        model, kwargs = _model(tokenizer, engine)
        _, data = _batch(tokenizer, kwargs, B=1)
        torch.manual_seed(0)
        dyn = model.sample_top_kp(data, choices=4, top_k=1, max_len=12, valid_only=False, simplify=False, unique=False, use_cache=True, return_raw=True)
        torch.manual_seed(0)
        stat = model.sample_top_kp(data, choices=4, top_k=1, max_len=12, valid_only=False, simplify=False, unique=False, static_decode=True, return_raw=True)
        assert dyn[0] == stat[0]      # greedy is path-independent


class TestSurface:
    def test_parameter_roles_cover_the_prefix_parameters(self, tokenizer, engine) -> None:  # type: ignore[no-untyped-def]
        model, _ = _model(tokenizer, engine)
        roles = model.parameter_roles()
        assert roles["decoder.data_tags"] == "embedding"
        assert roles["decoder.data_norm.weight"] == "vector"
        assert not any("cross_attention" in name or "cross_attn" in name for name in roles)
        from flash_ansr.train.optimizers import adamuon_param_groups
        groups = adamuon_param_groups(model, weight_decay=0.1, adam_weight_decay=0.01)
        assert sum(len(g["params"]) for g in groups) == sum(1 for p in model.parameters() if p.requires_grad)

    def test_from_config_and_save_load_round_trip(self, tokenizer, engine, tmp_path) -> None:  # type: ignore[no-untyped-def]
        cfg = _prefix_config()
        model = FlashANSRModel.from_config(cfg)
        assert model.decoder_data_mode == "prefix" and model.decoder.data_mode == "prefix"
        model.save(str(tmp_path), config=cfg)
        loaded_cfg, loaded = FlashANSRModel.load(str(tmp_path))
        assert loaded.decoder.data_mode == "prefix"
        assert torch.equal(loaded.decoder.data_tags, model.decoder.data_tags)

    def test_cross_attention_keys_required_only_in_cross_mode(self) -> None:
        cfg = load_config(get_path("configs", "test", "model.yaml"))
        cfg.pop("decoder_cross_attn_kv_norm")
        with pytest.raises(KeyError):
            FlashANSRModel.from_config(cfg)

    def test_v25_default_is_cross_attention(self) -> None:
        cfg = load_config(get_path("configs", "test", "model.yaml"))
        model = FlashANSRModel.from_config(cfg)
        assert model.decoder_data_mode == "cross_attention"
        assert model.decoder.prefix_len == 0
        assert all(layer.cross_attention is not None for layer in model.decoder.layers)
        assert model.decoder.rope.max_seq_len == cfg["decoder_max_seq_len"]

    def test_rejects_unknown_mode(self) -> None:
        cfg = load_config(get_path("configs", "test", "model.yaml"))
        cfg["decoder_data_mode"] = "concat"
        with pytest.raises(ValueError):
            FlashANSRModel.from_config(cfg)


class TestSharedPrefill:
    """The sampler prefills the shared token prefix once per problem and broadcasts it over the
    candidate rows (`_shared_prefill` / `_expand_prefill`); both decode paths must produce the same
    greedy sequences and near-identical logits as the per-row prefill they replace."""

    @pytest.mark.parametrize("mode", ["prefix", "cross_attention"])
    def test_shared_prefill_matches_per_row(self, tokenizer, engine, mode) -> None:  # type: ignore[no-untyped-def]
        cfg = load_config(get_path("configs", "test", "model.yaml"))
        if mode == "prefix":
            cfg = _prefix_config()
        kwargs = {k: v for k, v in cfg.items() if k not in ("simplipy_engine", "tokenizer")}
        torch.manual_seed(3)
        model = FlashANSRModel(simplipy_engine=engine, tokenizer=tokenizer, **kwargs).eval()
        torch.manual_seed(0)
        data = torch.randn(1, 12, kwargs["encoder_max_n_variables"], dtype=NUMERIC_DTYPE)
        prefix = torch.tensor([[tokenizer["<bos>"], tokenizer["<expression>"]]] * 6)
        with torch.no_grad():
            memory = model._create_memory(data)
            shared = model._shared_prefill(prefix, None, memory)
            assert shared is not None
            logits_s, cache_s = model._expand_prefill(shared[0], shared[1], 6)
            logits_r, cache_r = model(prefix, None, memory=memory, use_cache=True)
        assert torch.allclose(logits_s, logits_r, atol=1e-5)
        for (ks, vs), _ in cache_s:
            assert ks.shape[0] == 6
        assert torch.allclose(cache_s[0][0][0], cache_r[0][0][0], atol=1e-5)
        # rows that differ, or a per-row memory, fall back to the per-row prefill
        other = prefix.clone(); other[3, 1] = tokenizer["<eos>"]
        assert model._shared_prefill(other, None, memory) is None
        assert model._shared_prefill(prefix, None, memory.expand(6, -1, -1)) is None

    def test_greedy_decode_unchanged_by_sharing(self, tokenizer, engine, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        model, kwargs = _model(tokenizer, engine)
        _, data = _batch(tokenizer, kwargs, B=1)
        run = lambda static: model.sample_top_kp(data, choices=6, top_k=1, max_len=14, valid_only=False, simplify=False, unique=False, static_decode=static, batch_size=4, return_raw=True)[0]
        torch.manual_seed(0); shared_static = run(True)
        torch.manual_seed(0); shared_dynamic = run(False)
        monkeypatch.setattr(FlashANSRModel, "_shared_prefill", lambda self, *a, **k: None)
        torch.manual_seed(0); per_row_static = run(True)
        torch.manual_seed(0); per_row_dynamic = run(False)
        assert shared_static == per_row_static == shared_dynamic == per_row_dynamic
