"""NESTful (ibm-research/nestful): nested tool-call sequences as a one-turn program benchmark.

Gold format: a list of calls with ``label`` ("$var_1") and argument references like ``"$var_1.result$"``.
We map this onto our program IR: call k ↔ ``$k``, ``"$var_j.field$"`` ↔ ``"$j.field"``, and ask the model for the whole
sequence in a single assistant turn (capability flag ON). Scoring is structural (no execution):
  * seq_exact: predicted sequence equals gold after normalising references and argument order
  * name_f1: F1 over multiset of tool names
  * call_match: fraction of gold calls that appear in the prediction with identical (normalised) arguments
Usage: python -m etool.eval.nestful_run --base_url ... --model NAME --tokenizer PATH --out outputs/nestful/base [--limit 200]
Offline self-check: python -m etool.eval.nestful_run --selfcheck
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
from pathlib import Path
from typing import Any

from etool.schema import ToolCall, ToolSpec, Turn
from etool.templates import ChatFormatter, load_tokenizer
from etool.world.base import REF_CAPABILITY_TEXT

VAR_RE = re.compile(r"^\$var_(\d+)((?:\.[A-Za-z_][A-Za-z0-9_]*|\[\d+\])*)\$$")


def nestful_path() -> str:
    home = os.environ.get("HF_HOME", os.path.expanduser("~/EfficeintTool/hf"))
    return glob.glob(f"{home}/hub/datasets--ibm-research--nestful/snapshots/*/data/*.parquet")[0]


def convert_value(v: Any, label_to_idx: dict[str, int]) -> Any:
    if isinstance(v, str):
        m = VAR_RE.match(v.strip())
        if m:
            k = label_to_idx.get(f"$var_{m.group(1)}", int(m.group(1)))
            return f"${k}{m.group(2)}"
        return v
    if isinstance(v, list):
        return [convert_value(x, label_to_idx) for x in v]
    if isinstance(v, dict):
        return {a: convert_value(b, label_to_idx) for a, b in v.items()}
    return v


def convert_gold(output: list[dict[str, Any]]) -> list[ToolCall]:
    label_to_idx = {c.get("label"): i + 1 for i, c in enumerate(output) if c.get("label")}
    return [ToolCall(c["name"], convert_value(c.get("arguments", {}), label_to_idx), str(i + 1)) for i, c in enumerate(output)]


def convert_tools(tools: list[dict[str, Any]]) -> list[ToolSpec]:
    out = []
    for t in tools:
        params = t.get("parameters") or {}
        props = {}
        for k, v in params.items():
            if not isinstance(v, dict):
                continue
            ty = str(v.get("type", "string")).lower()
            jt = "number" if ("float" in ty or "number" in ty) else "integer" if "int" in ty else "boolean" if "bool" in ty else "array" if ("list" in ty or "array" in ty) else "object" if "dict" in ty else "string"
            props[k] = {"type": jt, "description": v.get("description", "")}
        desc = t.get("description", "")
        if t.get("output_parameters"):
            desc += " Returns: " + ", ".join(f"{k} ({v.get('description', '')})" for k, v in t["output_parameters"].items() if isinstance(v, dict))
        out.append(ToolSpec(t["name"], desc, {"type": "object", "properties": props, "required": list(props)}))
    return out


def _norm_args(a: Any) -> Any:
    if isinstance(a, dict):
        return {k: _norm_args(v) for k, v in sorted(a.items())}
    if isinstance(a, list):
        return [_norm_args(v) for v in a]
    if isinstance(a, float) and a.is_integer():
        return int(a)
    if isinstance(a, str):
        try:
            f = float(a)
            return int(f) if f.is_integer() else f
        except ValueError:
            return a.strip()
    return a


def score(pred: list[ToolCall], gold: list[ToolCall]) -> dict[str, float]:
    p = [(c.name, json.dumps(_norm_args(c.arguments), sort_keys=True)) for c in pred]
    g = [(c.name, json.dumps(_norm_args(c.arguments), sort_keys=True)) for c in gold]
    exact = float(p == g)
    pn, gn = collections.Counter(c.name for c in pred), collections.Counter(c.name for c in gold)
    inter = sum((pn & gn).values())
    prec = inter / max(1, sum(pn.values()))
    rec = inter / max(1, sum(gn.values()))
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    pc = collections.Counter(p)
    match = sum(min(pc[x], 1) for x in set(g) if x in pc) / max(1, len(g))
    return {"seq_exact": exact, "name_f1": round(f1, 4), "call_match": round(match, 4), "n_pred": len(pred), "n_gold": len(gold)}


SYSTEM = ("You solve the user's request by composing the provided tools. " + REF_CAPABILITY_TEXT +
          " Emit all calls needed, in one turn, in dependency order; use \"$N.field\" to pass an earlier result's field. Do not explain.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--base_url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model")
    ap.add_argument("--tokenizer")
    ap.add_argument("--out")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    import pandas as pd

    df = pd.read_parquet(nestful_path())
    rows = df.to_dict("records")
    if args.limit:
        rows = rows[: args.limit]
    if args.selfcheck:
        agg = collections.Counter()
        for r in rows[:300]:
            gold = convert_gold(json.loads(r["output"]))
            s = score(gold, gold)
            agg["exact"] += s["seq_exact"]
            agg["n"] += 1
            agg["has_ref"] += any(isinstance(v, str) and v.startswith("$") for c in gold for v in c.arguments.values())
            convert_tools(json.loads(r["tools"]))
        print("selfcheck:", dict(agg), "| example:", [(c.name, c.arguments) for c in convert_gold(json.loads(rows[0]["output"]))])
        return
    from openai import OpenAI
    import concurrent.futures as cf

    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=600)
    fmt = ChatFormatter(load_tokenizer(args.tokenizer), max_len=10**9)

    def one(r):
        tools = convert_tools(json.loads(r["tools"]))
        gold = convert_gold(json.loads(r["output"]))
        turns = [Turn("system", SYSTEM), Turn("user", r["input"])]
        prompt = fmt.render(turns, [t.to_openai() for t in tools], add_generation_prompt=True)
        resp = client.completions.create(model=args.model, prompt=prompt, max_tokens=args.max_tokens, temperature=0.0, stop=list(fmt.profile.stop))
        pred = fmt.parse(resp.choices[0].text).tool_calls
        return {"sample_id": r["sample_id"], **score(pred, gold), "out_tokens": resp.usage.completion_tokens, "pred": [(c.name, c.arguments) for c in pred]}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = []
    with cf.ThreadPoolExecutor(args.workers) as ex:
        for x in ex.map(one, rows):
            res.append(x)
    with (out / "results.jsonl").open("w") as f:
        for x in res:
            f.write(json.dumps(x) + "\n")
    summary = {k: round(sum(x[k] for x in res) / max(1, len(res)), 4) for k in ("seq_exact", "name_f1", "call_match", "out_tokens")}
    summary["n"] = len(res)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
