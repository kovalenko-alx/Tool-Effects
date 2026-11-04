"""Evaluate a served model on mock-world tasks.

Prompts are rendered with the *same* chat template as training (``ChatFormatter``) and sent to vLLM's raw
``/v1/completions`` endpoint, so train/serve formatting is identical and we parse ``<tool_call>`` blocks ourselves.
Each task runs as a conversation: model turn → (tool calls executed by the world executor | scripted user reply) …
until the model answers without a question or budgets are hit. Every trajectory is scored by the replay verifier.

    python -m etool.eval.world_eval --base_url http://127.0.0.1:8000/v1 --model NAME --tokenizer PATH \
        --tasks data/world/v1/test/tasks.jsonl --out outputs/world/base_test --refs on|off|both [--runtime_opt]
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import json
import statistics
import time
from pathlib import Path
from typing import Any, Optional

from etool.optimize import optimize
from etool.schema import ToolCall, Trajectory, Turn, dumps
from etool.templates import ChatFormatter, load_tokenizer
from etool.world.domains.workspace import Workspace
from etool.world.domains.commerce import Commerce
from etool.world.domains.workspace_hard import WorkspaceHard
from etool.world.executor import execute_turn
from etool.world.tasks import Task
from etool.world.user import ScriptedUser
from etool.world.verifier import verify

DOMAINS = {"workspace": Workspace, "workspace_hard": WorkspaceHard, "commerce": Commerce}


def _chat_step(client, model, api_messages, tools, max_tokens, temperature, extra_body):
    """One assistant step through /v1/chat/completions (server-side template + tool parser). Returns (turn, usage, assistant_msg)."""
    r = client.chat.completions.create(model=model, messages=api_messages, tools=tools or None, max_tokens=max_tokens, temperature=temperature,
                                       extra_body=extra_body or None)
    m = r.choices[0].message
    calls, api_ids = [], []
    for i, tc in enumerate(m.tool_calls or []):
        try:
            args = json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else (tc.function.arguments or {})
        except json.JSONDecodeError:
            args = {"_raw": tc.function.arguments}
        calls.append(ToolCall(tc.function.name, args if isinstance(args, dict) else {"_raw": args}, str(i + 1)))
        api_ids.append(tc.id)
    content = (m.content or "").strip()
    turn = Turn("assistant", content, calls, meta={"finish": r.choices[0].finish_reason, "api_ids": api_ids,
                                                  "reasoning": (getattr(m, "reasoning_content", None) or "")[:2000]})
    assistant_msg = {"role": "assistant", "content": m.content or ""}
    if m.tool_calls:
        assistant_msg["tool_calls"] = [{"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in m.tool_calls]
    return turn, r.usage, assistant_msg


def run_task(client, model: str, fmt: ChatFormatter, task: Task, refs: bool, max_turns: int, max_tokens: int, temperature: float,
             tag: str, api: str = "completions", extra_body: Optional[dict] = None, effect_hints: bool = False) -> dict[str, Any]:
    dom = DOMAINS[task.domain]()
    excluded = frozenset(task.tools_excluded)
    specs = [t for t in dom.specs() if t.name not in excluded]
    if effect_hints:  # v3b: the schema carries the declared effect (must match the training data of the evaluated adapter)
        specs = [t.with_effect_hint() for t in specs]
    tools = [t.to_openai() for t in specs]
    state = dom.new_state(task.seed)
    turns = [Turn("system", dom.system_prompt(refs)), Turn("user", task.utterance)]
    api_messages = [t.to_message() for t in turns]  # chat-API transcript (keeps the server's tool-call ids)
    user = ScriptedUser(task)
    out_tokens = prompt_tokens = 0
    n_model_calls = 0
    t0 = time.time()
    stop_reason = "answer"
    for _ in range(max_turns):
        if api == "chat":
            turn, usage, assistant_msg = _chat_step(client, model, api_messages, tools, max_tokens, temperature, extra_body)
            api_messages.append(assistant_msg)
        else:
            prompt = fmt.render(turns, tools, add_generation_prompt=True)
            r = client.completions.create(model=model, prompt=prompt, max_tokens=max_tokens, temperature=temperature, stop=list(fmt.profile.stop))
            usage = r.usage
            text = r.choices[0].text
            turn = fmt.parse(text)
            turn.meta = {"raw": text[:4000], "finish": r.choices[0].finish_reason}
        n_model_calls += 1
        out_tokens += usage.completion_tokens
        prompt_tokens += usage.prompt_tokens
        turns.append(turn)
        if turn.tool_calls:
            res = execute_turn(dom, state, turn, turn_idx=len(turns) - 1, capability_refs=refs, excluded=excluded)
            turns.extend(res.tool_turns)
            if api == "chat":
                ids = turn.meta.get("api_ids") or []
                for i, tt in enumerate(res.tool_turns):
                    api_messages.append({"role": "tool", "tool_call_id": ids[i] if i < len(ids) else tt.tool_call_id, "name": tt.name, "content": tt.content})
            continue
        reply = user.respond(turn.content)
        if reply is None:
            break
        turns.append(Turn("user", reply))
        api_messages.append({"role": "user", "content": reply})
    else:
        stop_reason = "max_turns"
    traj = Trajectory(id=f"{task.id}:{tag}:{'prog' if refs else 'step'}", source=f"model:{tag}", tools=specs, turns=turns, capability_refs=refs,
                      task=task.to_dict(), meta={"seed": task.seed, "kind": task.kind, "policy": tag})
    v = verify(dom, task, traj)
    row = {"task_id": task.id, "kind": task.kind, "refs": refs, **v.as_row(), "out_tokens": out_tokens, "prompt_tokens": prompt_tokens,
           "model_calls": n_model_calls, "wall_s": round(time.time() - t0, 2), "stop": stop_reason, "final_answer": v.details.get("final_answer", "")[:300]}
    try:
        o = optimize(dom, traj)
        row["opt_accepted"] = o.accepted
        row["opt_calls"] = o.optimized.n_calls() if o.accepted else v.n_calls
        row["opt_turns"] = o.optimized.n_action_turns() if o.accepted else v.n_action_turns
        row["opt_classes"] = o.classes if o.accepted else []
    except Exception as e:  # the optimizer must never break evaluation
        row["opt_error"] = repr(e)[:200]
    return {"row": row, "traj": traj}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(k):
        xs = [float(r[k]) for r in rows if isinstance(r.get(k), (int, float, bool))]
        return round(statistics.mean(xs), 4) if xs else None
    vio = collections.Counter()
    for r in rows:
        for t in r.get("violation_types", []):
            vio[t] += 1
    by_kind = {}
    for kind in sorted({r["kind"] for r in rows}):
        rs = [r for r in rows if r["kind"] == kind]
        by_kind[kind] = {"n": len(rs), "success": round(sum(r["success"] for r in rs) / len(rs), 4), "excess_calls": round(statistics.mean(r["excess_calls"] for r in rs), 3),
                         "excess_turns": round(statistics.mean(r["excess_turns"] for r in rs), 3), "out_tokens": round(statistics.mean(r["out_tokens"] for r in rs), 1)}
    return {"n": len(rows), "success": mean("success"), "goal_ok": mean("goal_ok"), "answer_ok": mean("answer_ok"), "excess_calls": mean("excess_calls"),
            "excess_turns": mean("excess_turns"), "n_calls": mean("n_calls"), "n_action_turns": mean("n_action_turns"), "latency_ms": mean("latency_ms"),
            "errors": mean("errors"), "unnecessary_calls": mean("unnecessary_calls"), "asked_ok": mean("asked_ok"), "out_tokens": mean("out_tokens"), "prompt_tokens": mean("prompt_tokens"),
            "model_calls": mean("model_calls"), "wall_s": mean("wall_s"), "collateral_rate": mean("collateral"),
            "violations_per_task": round(sum(vio.values()) / max(1, len(rows)), 4), "violation_counts": dict(vio),
            "opt_calls": mean("opt_calls"), "opt_turns": mean("opt_turns"), "max_turns_rate": round(sum(r["stop"] == "max_turns" for r in rows) / max(1, len(rows)), 4),
            "by_kind": by_kind}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", required=True, help="served model name")
    ap.add_argument("--tokenizer", default=None, help="tokenizer path (same template as training); required for --api completions")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="model")
    ap.add_argument("--refs", choices=["on", "off", "both"], default="both")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max_turns", type=int, default=12)
    ap.add_argument("--max_tokens", type=int, default=768)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--api", choices=["completions", "chat"], default="completions",
                    help="completions: render with --tokenizer's template and parse <tool_call> ourselves (train/serve identical); chat: server-side template + tool parser (any model family)")
    ap.add_argument("--effect_hints", action="store_true", help="annotate every tool description with its declared effect (v3b adapters)")
    ap.add_argument("--no_thinking", action="store_true", help="chat API: pass chat_template_kwargs.enable_thinking=false (hybrid Qwen3 / Qwen3.5)")
    args = ap.parse_args()
    from openai import OpenAI

    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=600, max_retries=2)
    fmt = ChatFormatter(load_tokenizer(args.tokenizer), max_len=10**9) if args.api == "completions" else None
    extra_body = {"chat_template_kwargs": {"enable_thinking": False}} if args.no_thinking else None
    tasks = []
    with open(args.tasks) as f:
        for line in f:
            if line.strip():
                tasks.append(Task.from_dict(json.loads(line)))
    if args.limit:
        tasks = tasks[: args.limit]
    modes = {"on": [True], "off": [False], "both": [True, False]}[args.refs]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(t, r) for r in modes for t in tasks]
    rows, trajs = [], []
    t0 = time.time()
    with cf.ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(run_task, client, args.model, fmt, t, r, args.max_turns, args.max_tokens, args.temperature, args.tag, args.api, extra_body,
                          args.effect_hints): (t, r) for t, r in jobs}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            t, r = futs[fut]
            try:
                res = fut.result()
            except Exception as e:
                rows.append({"task_id": t.id, "kind": t.kind, "refs": r, "success": False, "error": repr(e)[:300]})
                continue
            rows.append(res["row"])
            trajs.append(res["traj"])
            if i % 25 == 0:
                print(f"  {i}/{len(jobs)} done, {time.time()-t0:.0f}s", flush=True)
    with (out / "results.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with (out / "trajectories.jsonl").open("w") as f:
        for t in trajs:
            f.write(dumps(t) + "\n")
    summary = {"tag": args.tag, "model": args.model, "tasks": args.tasks, "n_tasks": len(tasks), "elapsed_s": round(time.time() - t0)}
    for r in modes:
        rs = [x for x in rows if x.get("refs") == r and "excess_calls" in x]
        summary["prog" if r else "step"] = summarize(rs) if rs else None
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "by_kind"}) for k, v in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
