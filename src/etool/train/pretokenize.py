"""Fill the trainer's tokenisation cache on CPU so GPU jobs start training immediately.

    python -m etool.train.pretokenize --model <base path> --sft_files data/sft/v1/sft_train.jsonl --val_files data/sft/v1/sft_val.jsonl \
        --pair_files data/sft/v1/pairs_train.jsonl --cache_dir data/cache [--max_len 4096] [--val_n 64]
The cache key is (file sizes + mtimes, tag, max_len), identical to what the trainer computes, so pass the same
--max_len / --val_n / --max_per_class / --pair_classes that the training arm uses.
"""
from __future__ import annotations

import argparse
import time

from etool.templates import ChatFormatter, load_tokenizer
from etool.train.data import load_pairs, load_sft


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sft_files", nargs="*", default=[])
    ap.add_argument("--val_files", nargs="*", default=[])
    ap.add_argument("--pair_files", nargs="*", default=[])
    ap.add_argument("--pair_classes", nargs="*", default=None)
    ap.add_argument("--max_per_class", type=int, default=None)
    ap.add_argument("--limit_per_file", type=int, default=None)
    ap.add_argument("--val_n", type=int, default=64)
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    fmt = ChatFormatter(load_tokenizer(args.model), max_len=args.max_len)
    t0 = time.time()
    if args.sft_files:
        n = len(load_sft(args.sft_files, fmt, args.cache_dir, args.limit_per_file))
        print(f"sft: {n} examples cached ({time.time()-t0:.0f}s)", flush=True)
    if args.val_files:
        n = len(load_sft(args.val_files, fmt, args.cache_dir, args.val_n))
        print(f"val: {n} examples cached ({time.time()-t0:.0f}s)", flush=True)
    if args.pair_files:
        n = len(load_pairs(args.pair_files, fmt, args.cache_dir, set(args.pair_classes) if args.pair_classes else None, args.max_per_class, args.seed))
        print(f"pairs: {n} examples cached ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
