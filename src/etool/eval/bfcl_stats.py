"""Paired per-prompt bootstrap comparison of two BFCL runs.

    python -m etool.eval.bfcl_stats --a outputs/bfcl/s2_cont_sft --b outputs/bfcl/s2_sup_ul [--n_boot 2000]
BFCL writes one result line per prompt (result/<model>/<group>/BFCL_v4_<cat>_result.json) and, in the score files, a
summary line followed by one line per *failed* prompt; a prompt is correct iff it has a result and no failure entry.
Prints, per category and pooled per group, accuracy A, accuracy B, the paired difference with a 95 % bootstrap CI and
P(B >= A). Categories are the BFCL v4 names (simple_python, multiple, parallel, parallel_multiple, irrelevance,
live_*, multi_turn_*).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

from etool.eval.stats import paired_bootstrap

GROUP_OF = {"non_live": "non_live", "live": "live", "multi_turn": "multi_turn"}


def per_item(root: str) -> dict[str, dict[str, int]]:
    """{category: {prompt_id: 0/1}} for one run directory (outputs/bfcl/<tag>)."""
    out: dict[str, dict[str, int]] = {}
    for rf in glob.glob(os.path.join(root, "result", "*", "*", "BFCL_v4_*_result.json")):
        cat = re.search(r"BFCL_v4_(.+)_result\.json$", rf).group(1)
        ids = [json.loads(l)["id"] for l in open(rf) if l.strip()]
        sf = rf.replace(os.sep + "result" + os.sep, os.sep + "score" + os.sep).replace("_result.json", "_score.json")
        failed: set[str] = set()
        if os.path.exists(sf):
            lines = [l for l in open(sf) if l.strip()]
            failed = {json.loads(l)["id"] for l in lines[1:]}
        out[cat] = {i: int(i not in failed) for i in ids}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--n_boot", type=int, default=2000)
    args = ap.parse_args()
    A, B = per_item(args.a), per_item(args.b)
    cats = sorted(set(A) & set(B))
    print(f"{'category':24s} {'n':>5s} {'A':>7s} {'B':>7s} {'B-A':>7s} {'95% CI':>18s} {'P(B>=A)':>8s}")
    pooled: dict[str, tuple[list[float], list[float]]] = {}
    for c in cats:
        ids = sorted(set(A[c]) & set(B[c]))
        if not ids:
            continue
        xa, xb = [float(A[c][i]) for i in ids], [float(B[c][i]) for i in ids]
        d, lo, hi, p = paired_bootstrap(xa, xb, args.n_boot)
        print(f"{c:24s} {len(ids):5d} {100*sum(xa)/len(xa):7.1f} {100*sum(xb)/len(xb):7.1f} {100*d:+7.1f} [{100*lo:+7.1f}, {100*hi:+6.1f}] {p:8.3f}")
        grp = "multi_turn" if c.startswith("multi_turn") else ("live" if c.startswith("live") else "non_live")
        pooled.setdefault(grp, ([], []))
        pooled[grp][0].extend(xa)
        pooled[grp][1].extend(xb)
    print()
    for grp, (xa, xb) in pooled.items():
        d, lo, hi, p = paired_bootstrap(xa, xb, args.n_boot)
        print(f"{'pooled ' + grp:24s} {len(xa):5d} {100*sum(xa)/len(xa):7.1f} {100*sum(xb)/len(xb):7.1f} {100*d:+7.1f} [{100*lo:+7.1f}, {100*hi:+6.1f}] {p:8.3f}")


if __name__ == "__main__":
    main()
