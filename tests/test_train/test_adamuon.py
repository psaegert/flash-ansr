"""AdaMuon (train.yaml ``optimizer: {name: AdaMuon}``): the paper's update for the hidden matrices,
AdamW for everything else, in one optimizer.

Checked here: the Newton-Schulz step lands the singular values in a band around 1; the Adam
branch is torch's AdamW to numerical precision; every Muon update has the paper's RMS of 0.2 so
the AdamW schedule carries over; the FlashANSRModel's parameters land in the right groups exactly
once; state round-trips through ``state_dict`` so a resumed run continues bit-for-bit; and the
trainer builds it from a config.
"""
import copy
import math

import pytest
import torch
from torch import nn

from flash_ansr import get_path
from flash_ansr.model.tokenizer import Tokenizer
from flash_ansr.train.optimizers import (
    ROLE_EMBEDDING, ROLE_HIDDEN_MATRIX, ROLE_OUTPUT_PROJECTION, ROLE_VECTOR,
    AdaMuon, adamuon_param_groups, build_optimizer, newton_schulz_orthogonalize, parameter_roles,
)
from flash_ansr.utils.config_io import load_config


class TestNewtonSchulz:
    def test_singular_values_land_near_one(self) -> None:
        torch.manual_seed(0)
        for shape in [(64, 96), (96, 64), (32, 32)]:
            out = newton_schulz_orthogonalize(torch.randn(*shape), steps=5)
            assert out.dtype == torch.bfloat16 and out.shape == shape
            s = torch.linalg.svdvals(out.float())
            assert s.min() > 0.5 and s.max() < 1.5, (shape, s.min(), s.max())

    def test_rejects_non_matrices(self) -> None:
        with pytest.raises(ValueError):
            newton_schulz_orthogonalize(torch.randn(8))


def _toy() -> nn.Sequential:
    torch.manual_seed(1)
    return nn.Sequential(nn.Linear(6, 8), nn.Tanh(), nn.Linear(8, 3))


def _loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return nn.functional.mse_loss(model(x), y)


class TestAdamBranch:
    def test_matches_torch_adamw(self) -> None:
        torch.manual_seed(2)
        x, y = torch.randn(16, 6), torch.randn(16, 3)
        a, b = _toy(), _toy()
        b.load_state_dict(a.state_dict())
        ref = torch.optim.AdamW(a.parameters(), lr=1e-2, weight_decay=0.05, betas=(0.9, 0.95), eps=1e-8)
        ours = AdaMuon([{"params": list(b.parameters()), "use_muon": False, "weight_decay": 0.05}],
                       lr=1e-2, betas=(0.9, 0.95), adam_eps=1e-8)
        for _ in range(5):
            for model, opt in ((a, ref), (b, ours)):
                opt.zero_grad()
                _loss(model, x, y).backward()
                opt.step()
        for pa, pb in zip(a.parameters(), b.parameters()):
            assert torch.allclose(pa, pb, atol=1e-7, rtol=1e-6)

    def test_adam_group_without_weight_decay_uses_the_adam_rate(self) -> None:
        w = nn.Parameter(torch.randn(4, 4))
        opt = AdaMuon([{"params": [w], "use_muon": False}], weight_decay=0.1, adam_weight_decay=0.01)
        assert opt.param_groups[0]["weight_decay"] == 0.01


class TestMuonBranch:
    def test_update_rms_is_the_papers_0_2(self) -> None:
        torch.manual_seed(3)
        w = nn.Parameter(torch.randn(48, 80))
        opt = AdaMuon([w], lr=1e-3, weight_decay=0.0)
        before = w.detach().clone()
        w.grad = torch.randn_like(w)
        opt.step()
        step = (w.detach() - before) / 1e-3
        rms = step.pow(2).mean().sqrt().item()
        assert abs(rms - 0.2) < 2e-3, rms

    def test_weight_decay_is_decoupled(self) -> None:
        w = nn.Parameter(torch.ones(8, 8))
        opt = AdaMuon([w], lr=0.1, weight_decay=0.5)
        w.grad = torch.zeros_like(w)
        # Zero gradient: sign(0)=0, Newton-Schulz of the zero matrix is zero, so only the decay acts.
        opt.step()
        assert torch.allclose(w.detach(), torch.full((8, 8), 1.0 - 0.1 * 0.5))

    def test_rejects_vectors_in_muon_groups(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            AdaMuon([nn.Parameter(torch.randn(5))])

    def test_fits_a_teacher_network(self) -> None:
        # A target the student architecture can represent exactly, so the loss has room to fall.
        torch.manual_seed(4)
        teacher = nn.Sequential(nn.Linear(6, 8), nn.Tanh(), nn.Linear(8, 3))
        x = torch.randn(256, 6)
        with torch.no_grad():
            y = teacher(x) * 3.0
        model = _toy()
        opt = AdaMuon(adamuon_param_groups(model, weight_decay=0.0, adam_weight_decay=0.0), lr=2e-2)
        first = _loss(model, x, y).item()
        for _ in range(300):
            opt.zero_grad()
            _loss(model, x, y).backward()
            opt.step()
        assert _loss(model, x, y).item() < 0.3 * first


class TestStateRoundTrip:
    def test_resume_continues_identically(self) -> None:
        torch.manual_seed(5)
        x, y = torch.randn(16, 6), torch.randn(16, 3)
        model = _toy()
        opt = AdaMuon(adamuon_param_groups(model, weight_decay=0.05, adam_weight_decay=0.01), lr=1e-2)
        for _ in range(3):
            opt.zero_grad()
            _loss(model, x, y).backward()
            opt.step()
        saved_model = copy.deepcopy(model.state_dict())
        saved_opt = copy.deepcopy(opt.state_dict())

        resumed = _toy()
        resumed.load_state_dict(saved_model)
        opt2 = AdaMuon(adamuon_param_groups(resumed, weight_decay=0.05, adam_weight_decay=0.01), lr=1e-2)
        opt2.load_state_dict(saved_opt)
        for _ in range(2):
            for m, o in ((model, opt), (resumed, opt2)):
                o.zero_grad()
                _loss(m, x, y).backward()
                o.step()
        for pa, pb in zip(model.parameters(), resumed.parameters()):
            assert torch.equal(pa, pb)


class TestParameterRoles:
    @pytest.fixture(scope="class")
    def model(self):  # type: ignore[no-untyped-def]
        from simplipy import SimpliPyEngine
        from flash_ansr.model.flash_ansr_model import FlashANSRModel
        cfg = load_config(get_path("configs", "test", "model.yaml"))
        kwargs = {k: v for k, v in cfg.items() if k not in ("simplipy_engine", "tokenizer")}
        tokenizer = Tokenizer.from_config(get_path("configs", "test", "tokenizer.yaml"))
        return FlashANSRModel(simplipy_engine=SimpliPyEngine.load("base", install=True), tokenizer=tokenizer, **kwargs)

    def test_every_parameter_has_one_role(self, model) -> None:  # type: ignore[no-untyped-def]
        roles = parameter_roles(model)
        names = [n for n, p in model.named_parameters() if p.requires_grad]
        assert sorted(roles) == sorted(names)
        params = dict(model.named_parameters())
        for name, role in roles.items():
            p = params[name]
            if role == ROLE_HIDDEN_MATRIX:
                assert p.ndim == 2, name
            if role == ROLE_VECTOR:
                assert p.numel() == p.shape[-1] or p.ndim == 1, name
        head_proj = f"next_token_head.{len(model.next_token_head) - 1}.weight"
        assert roles[head_proj] == ROLE_OUTPUT_PROJECTION
        assert roles["next_token_head.0.weight"] == ROLE_HIDDEN_MATRIX
        assert roles["numeric_embedding.weight"] == ROLE_EMBEDDING
        assert roles["decoder.tok_embeddings.weight"] == ROLE_EMBEDDING
        assert roles["decoder.layers.0.self_attention.w_q.weight"] == ROLE_HIDDEN_MATRIX
        assert roles["decoder.layers.0.self_attention.w_q.bias"] == ROLE_VECTOR
        assert roles["decoder.layers.0.self_attn_norm.weight"] == ROLE_VECTOR
        # The learned constant vectors and the broadcast-shaped norm gains, where the test model has them.
        for name, role in [("null_memory", ROLE_EMBEDDING), ("encoder.isabs.0.inducing_points", ROLE_EMBEDDING),
                           ("encoder.pma.seed_vectors", ROLE_EMBEDDING), ("encoder.isabs.0.mab_self.norm_q.gamma", ROLE_VECTOR)]:
            if name in roles:
                assert roles[name] == role, name
        assert any(params[n].ndim >= 3 and r == ROLE_EMBEDDING for n, r in roles.items()), "no learned constant vector found"

    def test_groups_partition_the_parameters(self, model) -> None:  # type: ignore[no-untyped-def]
        groups = adamuon_param_groups(model, weight_decay=0.1, adam_weight_decay=0.01)
        seen = [id(p) for g in groups for p in g["params"]]
        assert len(seen) == len(set(seen)) == sum(1 for p in model.parameters() if p.requires_grad)
        by_role = {g["role"]: g for g in groups}
        assert by_role["hidden_matrix"]["use_muon"] and by_role["hidden_matrix"]["weight_decay"] == 0.1
        assert not by_role["embedding_and_output"]["use_muon"] and by_role["embedding_and_output"]["weight_decay"] == 0.01
        assert not by_role["vector"]["use_muon"] and by_role["vector"]["weight_decay"] == 0.0
        # Most of the model's parameters are hidden matrices, the point of the optimizer.
        n_muon = sum(p.numel() for p in by_role["hidden_matrix"]["params"])
        n_total = sum(p.numel() for p in model.parameters())
        assert n_muon > 0.5 * n_total

    def test_build_optimizer_from_a_config_block(self, model) -> None:  # type: ignore[no-untyped-def]
        spec = {"name": "AdaMuon", "kwargs": {"lr": 1, "weight_decay": 0.1, "adam_weight_decay": 0.01,
                                              "betas": [0.9, 0.95], "momentum": 0.95, "nesterov": True, "ns_steps": 5}}
        opt = build_optimizer(spec, model)
        assert isinstance(opt, AdaMuon)
        assert all(g["lr"] == 1 for g in opt.param_groups)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda step: 1e-4 * min(1.0, step / 10))
        sched.step()
        assert all(math.isclose(g["lr"], 1e-5) for g in opt.param_groups)

    def test_generic_modules_get_the_fallback_roles(self) -> None:
        roles = parameter_roles(nn.Sequential(nn.Embedding(5, 4), nn.Linear(4, 3), nn.LayerNorm(3)))
        assert roles["0.weight"] == ROLE_EMBEDDING
        assert roles["1.weight"] == ROLE_HIDDEN_MATRIX
        assert roles["1.bias"] == ROLE_VECTOR and roles["2.weight"] == ROLE_VECTOR


class TestTrainerIntegration:
    """The T8 recipe end to end on the test config: AdaMuon + z-loss through Trainer.from_config,
    two optimizer steps, a validation pass, the logit-scale metrics logged."""

    def test_two_steps_with_adamuon_and_z_loss(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        import shutil
        from unittest import mock
        import yaml
        from flash_ansr.train import Trainer

        cfg_dir = tmp_path / "cfg"
        shutil.copytree(get_path("configs", "test"), cfg_dir)
        train_yaml = cfg_dir / "train.yaml"
        cfg = yaml.safe_load(train_yaml.read_text())
        cfg["optimizer"] = {"name": "AdaMuon", "kwargs": {"lr": 1, "weight_decay": 0.1, "adam_weight_decay": 0.01,
                                                          "betas": [0.9, 0.95], "momentum": 0.95, "nesterov": True, "ns_steps": 5}}
        cfg["z_loss_weight"] = 1.0e-4
        train_yaml.write_text(yaml.safe_dump(cfg))

        logged: list[dict] = []
        with mock.patch("wandb.init"), mock.patch("wandb.log", side_effect=lambda *a, **k: logged.append(a[0] if a else k)):
            trainer = Trainer.from_config(str(train_yaml))
            assert isinstance(trainer.optimizer, AdaMuon)
            assert trainer.z_loss_weight == pytest.approx(1e-4)
            trainer.run(project_name="neural-symbolic-regression-test", entity="psaegert", name="pytest-adamuon",
                        verbose=False, steps=2, device="cpu", preprocess=False, checkpoint_interval=None,
                        checkpoint_directory=None, wandb_mode="disabled", validate_size=4, validate_interval=1)
        keys = {k for entry in logged if isinstance(entry, dict) for k in entry}
        for key in ("train_z_loss", "train_log_z", "train_logit_absmean", "head_row_norm_max"):
            assert key in keys, (key, sorted(keys))
        # Every Muon matrix has been stepped: its second-moment buffer exists and is non-zero.
        muon_group = next(g for g in trainer.optimizer.param_groups if g["use_muon"])
        for p in muon_group["params"]:
            state = trainer.optimizer.state[p]
            assert "second_moment" in state and float(state["second_moment"].abs().sum()) > 0


class TestBatchedStepEquivalence:
    """The foreach/batched step is the per-matrix reference arithmetic, launched differently."""

    @staticmethod
    def _reference_muon_update(p: torch.Tensor, grad: torch.Tensor, state: dict, *, lr: float, weight_decay: float,
                               momentum: float, nesterov: bool, ns_steps: int, eps: float) -> None:
        # the original per-parameter loop (flash-ansr f6d918e), kept here as the oracle
        if not state:
            state["momentum_buffer"] = torch.zeros_like(p)
            state["second_moment"] = torch.zeros_like(p)
        buf, second = state["momentum_buffer"], state["second_moment"]
        buf.mul_(momentum).add_(grad)
        direction = grad.add(buf, alpha=momentum) if nesterov else buf
        ortho = newton_schulz_orthogonalize(torch.sign(direction), steps=ns_steps).to(p.dtype)
        second.mul_(momentum).addcmul_(ortho, ortho, value=1.0 - momentum)
        update = ortho / (second.sqrt() + eps)
        scale = 0.2 * math.sqrt(p.shape[0] * p.shape[1]) / (update.norm() + eps)
        if weight_decay != 0.0:
            p.mul_(1.0 - lr * weight_decay)
        p.add_(update, alpha=-lr * float(scale))

    def test_batched_newton_schulz_matches_the_single_matrix_iteration(self) -> None:
        from flash_ansr.train.optimizers import newton_schulz_orthogonalize_batched
        torch.manual_seed(11)
        for shape in ((8, 40, 96), (5, 96, 40), (3, 64, 64)):
            stack = torch.randn(*shape)
            batched = newton_schulz_orthogonalize_batched(stack)
            for i in range(shape[0]):
                single = newton_schulz_orthogonalize(stack[i])
                torch.testing.assert_close(batched[i].float(), single.float(), atol=2e-2, rtol=2e-2)

    def test_step_matches_the_per_matrix_reference(self) -> None:
        torch.manual_seed(5)
        shapes = [(48, 80), (48, 80), (96, 32), (16, 16)]
        ws_new = [nn.Parameter(torch.randn(*s)) for s in shapes]
        ws_ref = [nn.Parameter(w.detach().clone()) for w in ws_new]
        kw = dict(lr=3e-3, weight_decay=0.1, momentum=0.95, nesterov=True, ns_steps=5, eps=1e-8)
        opt = AdaMuon(ws_new, **kw)
        ref_state = [dict() for _ in ws_ref]
        for _ in range(4):
            grads = [torch.randn(*s) for s in shapes]
            for w, g in zip(ws_new, grads):
                w.grad = g.clone()
            opt.step()
            with torch.no_grad():
                for w, g, st in zip(ws_ref, grads, ref_state):
                    self._reference_muon_update(w, g, st, **kw)
        for w_new, w_ref in zip(ws_new, ws_ref):
            torch.testing.assert_close(w_new.detach(), w_ref.detach(), atol=1e-4, rtol=1e-4)
        for w, st in zip(ws_new, ref_state):
            torch.testing.assert_close(opt.state[w]["momentum_buffer"], st["momentum_buffer"], atol=1e-6, rtol=1e-6)

    def test_adam_group_step_matches_the_per_parameter_reference(self) -> None:
        torch.manual_seed(6)
        ps_new = [nn.Parameter(torch.randn(30, 12)), nn.Parameter(torch.randn(12)), nn.Parameter(torch.randn(7, 5))]
        ps_ref = [nn.Parameter(p.detach().clone()) for p in ps_new]
        opt = AdaMuon([{"params": ps_new, "use_muon": False}], lr=2e-3, adam_weight_decay=0.01)
        ref = torch.optim.AdamW(ps_ref, lr=2e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01, foreach=False)
        for _ in range(5):
            for a, b in zip(ps_new, ps_ref):
                g = torch.randn_like(a); a.grad = g.clone(); b.grad = g.clone()
            opt.step(); ref.step()
        for a, b in zip(ps_new, ps_ref):
            torch.testing.assert_close(a.detach(), b.detach(), atol=1e-6, rtol=1e-6)
