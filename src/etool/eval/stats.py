"""Paired per-task bootstrap comparison of two world-eval runs (same tasks, same modes).

    python -m etool.eval.stats --a outputs/world/base_test/results.jsonl --b outputs/world/sft_v1_test/results.jsonl \
        [--metrics success excess_calls excess_turns out_tokens n_violations] [--n_boot 2000]
Prints, per mode (prog/step) and metric: mean A, mean B, paired difference B−A with a 95 % bootstrap CI and the
fraction of bootstrap samples where B ≥ A (one-sided). Rows are paired on (task_id, refs). ``n_violation_types`` (distinct
violation types per task, the summary's definition) is derived on load; ``n_violations`` counts instances.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics


def load(path: str) -> dict[tuple[str, bool], dict]:
    out = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if "excess_calls" in r:
                    # the summary's violations_per_task counts distinct violation *types* per task; n_violations counts instances
                    r["n_violation_types"] = len(r.get("violation_types") or [])
                    out[(r["task_id"], bool(r["refs"]))] = r
    return out


def paired_bootstrap(xa: list[float], xb: list[float], n_boot: int, seed: int = 0) -> tuple[float, float, float, float]:
    rng = random.Random(seed)
    d = [b - a for a, b in zip(xa, xb)]
    n = len(d)
    boots = []
    for _ in range(n_boot):
        s = [d[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(s) / n)
    boots.sort()
    return sum(d) / n, boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1], sum(1 for b in boots if b >= 0) / n_boot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--metrics", nargs="*", default=["success", "excess_calls", "excess_turns", "out_tokens", "n_violations", "latency_ms"])
    ap.add_argument("--n_boot", type=int, default=2000)
    args = ap.parse_args()
    A, B = load(args.a), load(args.b)
    keys = sorted(set(A) & set(B))
    print(f"paired rows: {len(keys)} (A has {len(A)}, B has {len(B)})")
    for refs in (True, False):
        ks = [k for k in keys if k[1] == refs]
        if not ks:
            continue
        print(f"\n== mode {'prog' if refs else 'step'} (n={len(ks)}) ==")
        print(f"{'metric':14s} {'A':>9s} {'B':>9s} {'B-A':>9s} {'95% CI':>22s} {'P(B>=A)':>9s}")
        for m in args.metrics:
            xa = [float(A[k].get(m, 0) or 0) for k in ks]
            xb = [float(B[k].get(m, 0) or 0) for k in ks]
            diff, lo, hi, p = paired_bootstrap(xa, xb, args.n_boot)
            print(f"{m:14s} {statistics.mean(xa):9.4f} {statistics.mean(xb):9.4f} {diff:+9.4f} [{lo:+9.4f}, {hi:+9.4f}] {p:9.3f}")


if __name__ == "__main__":
    main()
