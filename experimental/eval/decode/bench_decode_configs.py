"""Wall-time of every sampling decode configuration.

Arms (the compaction ones need the grammar, so a grammar-only arm isolates its cost):

  A  no cache                       use_cache=False, static_decode=False
  B  dynamic cache                  use_cache=True,  static_decode=False
  C  static cache                   static_decode=True
  D  static + grammar               + constrain_ieee754
  E  static + grammar + compaction  + compact_ieee754

  dynamic + compaction is N/A by construction: compaction desynchronizes rows and the
  dynamic cat-grow path has no key-pad mask, so its rows cannot carry different lengths.
  That is the gate in sample_top_kp, not an omission here.
"""
import argparse
import json
import statistics
import time

import torch

from flash_ansr import get_path
from flash_ansr.model.flash_ansr_model import FlashANSRModel
from flash_ansr.model.tokenizer import Tokenizer
from flash_ansr.utils.config_io import load_config
from flash_ansr.utils.ieee754 import IEEE754_START_TOKEN
from flash_ansr.utils.numeric import NUMERIC_DTYPE

ARMS = {
    "A no-cache":                 dict(use_cache=False, static_decode=False),
    "B dynamic":                  dict(use_cache=True, static_decode=False),
    "C static":                   dict(static_decode=True),
    "D static+grammar":           dict(static_decode=True, constrain_ieee754=True),
    "E static+grammar+compact":   dict(static_decode=True, constrain_ieee754=True,
                                       compact_ieee754=True),
}


def build(device, span_bias, config="test"):
    tokenizer = Tokenizer.from_config(get_path("configs", config, "tokenizer.yaml"))
    from simplipy import SimpliPyEngine
    cfg = load_config(get_path("configs", config, "model.yaml"))
    kwargs = {k: v for k, v in cfg.items() if k not in ("simplipy_engine", "tokenizer")}
    torch.manual_seed(0x24C2)
    model = FlashANSRModel(simplipy_engine=SimpliPyEngine.load("base", install=True),
                           tokenizer=tokenizer, **kwargs).eval().to(device)
    if span_bias:
        with torch.no_grad():
            model.next_token_head[-1].bias[tokenizer[IEEE754_START_TOKEN]] += span_bias
    return model, tokenizer, kwargs


def time_arm(model, tokenizer, kwargs, *, choices, max_len, batch_size, device, repeats, n_vars):
    x = torch.rand(13, n_vars, dtype=NUMERIC_DTYPE, device=device)
    bos = [tokenizer["<bos>"]]
    times, tokens = [], 0
    for r in range(repeats + 1):                       # first pass is warm-up, discarded
        if device.type == "cuda":
            torch.cuda.synchronize()
        torch.manual_seed(11)
        t0 = time.perf_counter()
        raw, _ = model.sample_top_kp(x, choices=choices, max_len=max_len, batch_size=batch_size,
                                     return_raw=True, initial_tokens=bos, **kwargs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if r:
            times.append(dt)
            tokens = sum(len(s) for s in raw)
    return statistics.median(times), min(times), tokens


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--choices", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=96)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--span-bias", type=float, default=6.0,
                    help="bias the <ieee754> logit so spans actually occur; 0 disables")
    ap.add_argument("--config", default="test", help="configs/<name> to size the model from")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    model, tokenizer, mk = build(device, args.span_bias, args.config)
    n_vars = int(mk["encoder_max_n_variables"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"config={args.config}  params={n_params/1e6:.2f}M  n_vars={n_vars}")
    print(f"device={device}  choices={args.choices}  max_len={args.max_len}  "
          f"batch={args.batch_size}  repeats={args.repeats}  span_bias={args.span_bias}")
    print(f"{'arm':<28}{'median s':>10}{'best s':>10}{'tok/s':>12}{'vs B':>8}")
    rows, baseline = {}, None
    for name, kwargs in ARMS.items():
        try:
            median, best, tokens = time_arm(
                model, tokenizer, kwargs, choices=args.choices, max_len=args.max_len,
                batch_size=args.batch_size, device=device, repeats=args.repeats, n_vars=n_vars)
        except Exception as exc:                       # a refused pairing is a RESULT, not a crash
            print(f"{name:<28}{'N/A':>10}  {type(exc).__name__}: {str(exc)[:60]}")
            rows[name] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        if name.startswith("B "):
            baseline = median
        ratio = f"{baseline / median:.2f}x" if baseline else "-"
        print(f"{name:<28}{median:>10.3f}{best:>10.3f}{tokens / median:>12.0f}{ratio:>8}")
        rows[name] = {"median_s": median, "best_s": best, "tokens": tokens,
                      "tokens_per_s": tokens / median}
    print("\ndynamic+compaction        N/A  rows desynchronize; the dynamic cat-grow path "
          "has no key-pad mask")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"device": str(device), "config": args.config,
                       "params": n_params, "choices": args.choices, "max_len": args.max_len,
                       "batch_size": args.batch_size, "repeats": args.repeats,
                       "span_bias": args.span_bias, "arms": rows}, fh, indent=2)


if __name__ == "__main__":
    main()
