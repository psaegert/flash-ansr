"""Run provenance: pin every evaluation result to its code, inputs, and system.

`collect_provenance(config)` gathers:
  - CODE   : git commit + branch + dirty flag + hash of the uncommitted tracked diff
             (captures a frozen-but-dirty src exactly) + a fingerprint of the dirty/untracked
             file list.
  - INPUTS : the result-affecting files NOT covered by the commit (model weights/state_dict.pt,
             tokenizer, benchmark / dataset files) - sha256, streamed and cached by size+mtime.
  - SYSTEM : hostname, platform, GPU (name/driver/CUDA), CPU count.
  - ENV    : python / torch / numpy / scipy versions.

The dict is printed at run start and embedded into each result pickle under the reserved
`__meta__` key (a non-list value; `ResultStore.save` injects it, resume strips it, and
`filter_payload` passes non-list keys through untouched, so downstream readers are unaffected).
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from flash_ansr.utils.config_io import load_config
from flash_ansr.utils.paths import substitute_root_path

META_KEY = "__meta__"
_REPO = Path(__file__).resolve().parents[3]
_CACHE = _REPO / ".provenance_cache"


def _git(*args: str) -> str:
    root = _REPO if (_REPO / ".git").exists() else Path.cwd()
    try:
        return subprocess.run(["git", "-C", str(root), *args],
                              capture_output=True, text=True, timeout=15).stdout.strip()
    except Exception:
        return ""


def git_provenance() -> dict:
    porcelain = _git("status", "--porcelain")
    dirty = bool(porcelain)
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": dirty,
        "tracked_diff_sha": hashlib.sha256(_git("diff", "HEAD").encode()).hexdigest()[:16] if dirty else None,
        "worktree_fingerprint": hashlib.sha256(porcelain.encode()).hexdigest()[:16] if dirty else None,
        "n_dirty_or_untracked": len(porcelain.splitlines()) if porcelain else 0,
    }


def hash_file(path: str | Path) -> dict | None:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    st = p.stat()
    key = hashlib.sha256(f"{p.resolve()}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()
    side = _CACHE / f"{key}.json"
    if side.exists():
        try:
            return json.loads(side.read_text())
        except Exception:
            pass
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    rec = {"path": str(p), "size": st.st_size, "sha256": h.hexdigest()}
    try:
        _CACHE.mkdir(exist_ok=True)
        side.write_text(json.dumps(rec))
    except Exception:
        pass
    return rec


def _resolve_inputs(config_path: str, experiment: str | None) -> dict[str, str]:
    cfg = load_config(config_path)
    exps = cfg.get("experiments") if isinstance(cfg, dict) else None
    if exps:
        item = exps.get(experiment) if experiment else next(iter(exps.values()))
    else:
        item = cfg
    run = item.get("run", item) if isinstance(item, dict) else {}
    files: dict[str, str] = {}
    mp = run.get("model_adapter", {}).get("model_path")
    if mp:
        mpr = Path(substitute_root_path(mp))
        files["model/state_dict.pt"] = str(mpr / "state_dict.pt")
        for extra in ("config.yaml", "tokenizer.yaml", "tokenizer.json"):
            if (mpr / extra).exists():
                files[f"model/{extra}"] = str(mpr / extra)
    ds = run.get("data_source", {})
    if ds.get("benchmark_path"):
        files["benchmark"] = substitute_root_path(str(ds["benchmark_path"]))
    if ds.get("dataset"):
        d = ds["dataset"]
        files["dataset_cfg"] = substitute_root_path(str(d if isinstance(d, str) else (d.get("path") or d.get("config"))))
    if isinstance(ds.get("skeleton_list"), str):
        # the pinned eval-set file (e.g. standard-eval val100) -- hashing it records WHICH skeletons
        # ran, so "val" is provenance-verifiable even though the pool lives under gitignored data/.
        files["skeleton_list_pin"] = substitute_root_path(str(ds["skeleton_list"]))
    return files


def system_provenance() -> dict:
    info: dict[str, Any] = {"hostname": socket.gethostname(), "platform": platform.platform(),
                            "cpu_count": os.cpu_count()}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15).stdout.strip().splitlines()
        info["gpu"] = out[0].strip() if out else None
        info["n_gpu"] = len(out)
    except Exception:
        info["gpu"] = None
    return info


def env_provenance() -> dict:
    out = {}
    for mod in ("torch", "numpy", "scipy"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = "?"
    import sys
    out["python"] = sys.version.split()[0]
    try:
        import torch
        out["cuda"] = torch.version.cuda
    except Exception:
        out["cuda"] = None
    return out


def collect_provenance(config_path: str | None = None, experiment: str | None = None) -> dict:
    prov: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "experiment": experiment,
        "git": git_provenance(),
        "system": system_provenance(),
        "env": env_provenance(),
        "inputs": {},
    }
    if config_path:
        prov["config"] = config_path
        prov["config_sha"] = (hash_file(config_path) or {}).get("sha256")
        for name, path in _resolve_inputs(config_path, experiment).items():
            prov["inputs"][name] = hash_file(path)
    return prov


def format_provenance(prov: dict) -> str:
    g, s, e = prov.get("git", {}), prov.get("system", {}), prov.get("env", {})
    lines = ["-" * 78, f"PROVENANCE  {prov.get('timestamp', '')}"]
    git_state = (f"DIRTY tracked_diff={g.get('tracked_diff_sha')} wt_fp={g.get('worktree_fingerprint')} "
                 f"({g.get('n_dirty_or_untracked')} files)") if g.get("dirty") else "clean"
    lines.append(f"  git    : {str(g.get('commit'))[:12]} ({g.get('branch')}) {git_state}")
    lines.append(f"  system : {s.get('hostname')} | {s.get('gpu')} | {s.get('platform')} | {s.get('cpu_count')} cpu")
    lines.append(f"  env    : python {e.get('python')}  torch {e.get('torch')}  cuda {e.get('cuda')}  "
                 f"numpy {e.get('numpy')}  scipy {e.get('scipy')}")
    if prov.get("config"):
        lines.append(f"  config : {prov['config']}  sha={str(prov.get('config_sha'))[:16]}")
    for name, rec in prov.get("inputs", {}).items():
        if rec is None:
            lines.append(f"  input  : {name:<22} MISSING")
        else:
            lines.append(f"  input  : {name:<22} {rec['sha256'][:16]}  ({rec['size'] / 1e6:.1f} MB)")
    lines.append("-" * 78)
    return "\n".join(lines)


def emit_provenance(config_path: str | None = None, experiment: str | None = None) -> dict:
    prov = collect_provenance(config_path, experiment)
    print(format_provenance(prov), flush=True)
    return prov


__all__ = ["collect_provenance", "format_provenance", "emit_provenance", "META_KEY"]
