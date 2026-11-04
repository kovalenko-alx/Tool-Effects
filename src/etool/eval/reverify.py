"""Re-score an existing world-eval run with the current verifier (replay-based, so no model needed).

    python -m etool.eval.reverify --run outputs/world/base_test [--out outputs/world/base_test/reverified]
Rewrites results.jsonl rows (keeping token/latency columns) and summary.json; useful after verifier fixes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from etool.eval.world_eval import DOMAINS, summarize
from etool.optimize import optimize
from etool.schema import Trajectory, read_jsonl
from etool.world.tasks import Task
from etool.world.verifier import verify


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no_opt", action="store_true")
    args = ap.parse_args()
    run = Path(args.run)
    out = Path(args.out) if args.out else run
    out.mkdir(parents=True, exist_ok=True)
    old = {}
    if (run / "results.jsonl").exists():
        for l in open(run / "results.jsonl"):
            if l.strip():
                r = json.loads(l)
                old[(r["task_id"], bool(r.get("refs")))] = r
    rows = []
    for traj in read_jsonl(run / "trajectories.jsonl", Trajectory):
        task = Task.from_dict(traj.task)
        dom = DOMAINS[task.domain]()
        v = verify(dom, task, traj)
        prev = old.get((task.id, traj.capability_refs), {})
        row = {"task_id": task.id, "kind": task.kind, "refs": traj.capability_refs, **v.as_row(),
               **{k: prev.get(k) for k in ("out_tokens", "prompt_tokens", "model_calls", "wall_s", "stop") if k in prev},
               "final_answer": v.details.get("final_answer", "")[:300]}
        if not args.no_opt:
            try:
                o = optimize(dom, traj)
                row["opt_accepted"] = o.accepted
                row["opt_calls"] = o.optimized.n_calls() if o.accepted else v.n_calls
                row["opt_turns"] = o.optimized.n_action_turns() if o.accepted else v.n_action_turns
            except Exception as e:
                row["opt_error"] = repr(e)[:200]
        rows.append(row)
    with (out / "results.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summary = {"reverified_from": str(run), "n_rows": len(rows)}
    for refs, name in ((True, "prog"), (False, "step")):
        rs = [r for r in rows if r["refs"] == refs]
        summary[name] = summarize(rs) if rs else None
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    for name in ("prog", "step"):
        m = summary.get(name) or {}
        print(name, {k: m.get(k) for k in ("n", "success", "goal_ok", "answer_ok", "excess_calls", "excess_turns", "violations_per_task")})


if __name__ == "__main__":
    main()
