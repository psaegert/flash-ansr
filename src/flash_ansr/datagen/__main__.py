"""CLI of the split pipeline.

    python -m flash_ansr.datagen sample --model-dir M --out RAW --n-problems N --seed S [--k-spec fixed:512]
                                        [--max-len 128] [--rows 4096] [--chunk-rows 8] [--amp tf32|bf16|fp32]
                                        [--verify-every 0] [--serial]
    python -m flash_ansr.datagen fit --model-dir M --raw RAW --out FIT [--workers W] [--part i/n] [--restarts R]
    python -m flash_ansr.datagen targets --raw RAW --fit FIT --out TGT --k-spec SPEC [--workers W] [--part i/n]
"""
import argparse


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m flash_ansr.datagen")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample", help="stage 1 (GPU): raw draws, packed across problems")
    s.add_argument("--model-dir", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--n-problems", type=int, required=True)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--k-spec", default="fixed:512")
    s.add_argument("--prior-dir", default=None, help="bundle holding catalog_train.yaml / dataset_train.yaml "
                   "(default: --model-dir)")
    s.add_argument("--max-len", type=int, default=128)
    s.add_argument("--shard-problems", type=int, default=256)
    s.add_argument("--rows", type=int, default=4096, help="target rows per packed decode")
    s.add_argument("--chunk-rows", type=int, default=8, help="rows per group (0: equal-K packing, rows = K)")
    s.add_argument("--amp", default="tf32", choices=["fp32", "tf32", "bf16"])
    s.add_argument("--verify-every", type=int, default=0, help="verify the first pack of every n-th shard (0: never)")
    s.add_argument("--serial", action="store_true", help="no CPU/GPU overlap (same output; for checks)")
    f = sub.add_parser("fit", help="stage 2 (CPU): post-process, refit, ladder, rank, fresh FVU")
    f.add_argument("--model-dir", required=True)
    f.add_argument("--raw", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--workers", type=int, default=None)
    f.add_argument("--part", default="0/1")
    f.add_argument("--restarts", type=int, default=None)
    t = sub.add_parser("targets", help="stage 3 (CPU): front-member and control targets from fitted shards")
    t.add_argument("--raw", required=True)
    t.add_argument("--fit", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--k-spec", required=True, help="training pool size per problem, e.g. uniform:1:512, fixed:512")
    t.add_argument("--workers", type=int, default=None)
    t.add_argument("--part", default="0/1")
    a = ap.parse_args()
    if a.cmd == "sample":
        from flash_ansr.datagen import sample
        sample.run(a.model_dir, a.out, n_problems=a.n_problems, seed=a.seed, k_spec=a.k_spec, max_len=a.max_len,
                   shard_problems=a.shard_problems, rows_target=a.rows, chunk_rows=a.chunk_rows, amp=a.amp,
                   verify_every=a.verify_every, prior_dir=a.prior_dir, overlap=not a.serial)
    elif a.cmd == "fit":
        from flash_ansr.datagen import fit
        i, n = (int(v) for v in a.part.split("/"))
        fit.run(a.raw, a.out, a.model_dir, workers=a.workers, n_restarts=a.restarts, part=(i, n))
    else:
        from flash_ansr.datagen import targets
        i, n = (int(v) for v in a.part.split("/"))
        targets.run(a.raw, a.fit, a.out, k_spec=a.k_spec, workers=a.workers, part=(i, n))


if __name__ == "__main__":
    main()
