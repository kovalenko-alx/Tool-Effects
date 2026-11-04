"""Rule-based perturbations of gold trajectories → ``Negative`` records (bad action at a gold prefix).

Each generator looks at one assistant turn of a gold trajectory and returns a *bad* replacement turn, keeping the
gold turn as the good action at the same prefix. Perturbations are deterministic given (trajectory, seed).
"""
from __future__ import annotations

import copy
import datetime as _dt
import json
import random
import re
from typing import Any, Callable, Optional

from etool.schema import Negative, ToolCall, ToolSpec, Trajectory, Turn
from etool.world.tasks import Task

Perturber = Callable[[Trajectory, int, random.Random, dict[str, ToolSpec], Optional[Task]], Optional[Turn]]
PERTURBERS: dict[str, Perturber] = {}

_ID_RE = re.compile(r"^[a-z]{1,2}\d{1,4}$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TIME_RE = re.compile(r"^\d{2}:\d{2}$")


def perturber(name: str):
    def deco(fn: Perturber) -> Perturber:
        PERTURBERS[name] = fn
        return fn
    return deco


def _first_action(traj: Trajectory) -> Optional[int]:
    idx = [i for i, t in enumerate(traj.turns) if t.is_action]
    return idx[0] if idx else None


def _seen_values(traj: Trajectory, upto: int) -> dict[str, list[Any]]:
    """Ids / emails / dates / strings seen in tool results before turn ``upto`` (to make plausible wrong values)."""
    pools: dict[str, list[Any]] = {"id": [], "email": [], "date": [], "time": [], "str": []}
    for t in traj.turns[:upto]:
        if t.role != "tool":
            continue
        try:
            obj = json.loads(t.content)
        except json.JSONDecodeError:
            continue
        stack = [obj]
        while stack:
            v = stack.pop()
            if isinstance(v, dict):
                stack.extend(v.values())
            elif isinstance(v, list):
                stack.extend(v)
            elif isinstance(v, str):
                if _ID_RE.match(v):
                    pools["id"].append(v)
                elif "@" in v:
                    pools["email"].append(v)
                elif _DATE_RE.match(v):
                    pools["date"].append(v)
                elif _TIME_RE.match(v):
                    pools["time"].append(v)
                elif 3 <= len(v) <= 40:
                    pools["str"].append(v)
    return {k: sorted(set(v)) for k, v in pools.items()}


def _shift_date(d: str, days: int) -> str:
    return (_dt.date.fromisoformat(d) + _dt.timedelta(days=days)).isoformat()


def _shift_time(t: str, hours: int) -> str:
    h, m = map(int, t.split(":"))
    return f"{(h + hours) % 24:02d}:{m:02d}"


def _perturb_value(key: str, v: Any, rng: random.Random, pools: dict[str, list[Any]]) -> Optional[Any]:
    if isinstance(v, str):
        if v.startswith("$"):
            return None
        if _DATE_RE.match(v):
            return _shift_date(v, rng.choice([-1, 1, 7]))
        if _TIME_RE.match(v):
            return _shift_time(v, rng.choice([-1, 1, 2]))
        if _ID_RE.match(v):
            alts = [x for x in pools["id"] if x != v and x[0] == v[0]]
            return rng.choice(alts) if alts else v[:-1] + str((int(v[1:]) + 1) % 10)
        if "@" in v:
            alts = [x for x in pools["email"] if x != v]
            return rng.choice(alts) if alts else "nobody@example.com"
        alts = [x for x in pools["str"] if x != v]
        return rng.choice(alts) if alts and rng.random() < 0.6 else v + " (draft)"
    if isinstance(v, bool):
        return not v
    if isinstance(v, int):
        return v + rng.choice([-1, 1, 10])
    if isinstance(v, list) and v:
        if len(v) > 1:
            return v[:-1]
        inner = _perturb_value(key, v[0], rng, pools)
        return [inner] if inner is not None else None
    return None


# ----------------------------------------------------------------------------- correctness
@perturber("wrong_tool")
def wrong_tool(traj, k, rng, specs, task):
    t = traj.turns[k]
    if not t.tool_calls:
        return None
    calls = copy.deepcopy(t.tool_calls)
    j = rng.randrange(len(calls))
    c = calls[j]
    cands = [s for s in specs.values() if s.name != c.name and s.name != specs.get(c.name, ToolSpec("", "", {})).batch_of]
    same_effect = [s for s in cands if specs.get(c.name) and s.effect == specs[c.name].effect] or cands
    other = rng.choice(same_effect)
    props = other.parameters.get("properties", {})
    args = {a: v for a, v in c.arguments.items() if a in props}
    calls[j] = ToolCall(other.name, args, c.id)
    return Turn("assistant", t.content, calls)


@perturber("hallucinated_tool")
def hallucinated_tool(traj, k, rng, specs, task):
    t = traj.turns[k]
    if not t.tool_calls:
        return None
    calls = copy.deepcopy(t.tool_calls)
    j = rng.randrange(len(calls))
    c = calls[j]
    fake = rng.choice([c.name + "_v2", "get_" + c.name.split("_", 1)[-1] + "_details", c.name.replace("list_", "fetch_").replace("get_", "lookup_"), "schedule_" + c.name.split("_")[-1]])
    if fake in specs:
        fake += "_ext"
    calls[j] = ToolCall(fake, c.arguments, c.id)
    return Turn("assistant", t.content, calls)


@perturber("missing_required_arg")
def missing_required_arg(traj, k, rng, specs, task):
    t = traj.turns[k]
    options = [(j, a) for j, c in enumerate(t.tool_calls) if c.name in specs for a in specs[c.name].required if a in c.arguments]
    if not options:
        return None
    j, a = rng.choice(options)
    calls = copy.deepcopy(t.tool_calls)
    calls[j].arguments = {x: v for x, v in calls[j].arguments.items() if x != a}
    return Turn("assistant", t.content, calls)


@perturber("wrong_arg_value")
def wrong_arg_value(traj, k, rng, specs, task):
    t = traj.turns[k]
    pools = _seen_values(traj, k)
    options = [(j, a) for j, c in enumerate(t.tool_calls) for a, v in c.arguments.items() if not (isinstance(v, str) and v.startswith("$"))]
    rng.shuffle(options)
    for j, a in options:
        calls = copy.deepcopy(t.tool_calls)
        nv = _perturb_value(a, calls[j].arguments[a], rng, pools)
        if nv is not None and nv != calls[j].arguments[a]:
            calls[j].arguments[a] = nv
            return Turn("assistant", t.content, calls)
    return None


@perturber("wrong_arg_type")
def wrong_arg_type(traj, k, rng, specs, task):
    t = traj.turns[k]
    options = [(j, a) for j, c in enumerate(t.tool_calls) for a, v in c.arguments.items() if not (isinstance(v, str) and v.startswith("$"))]
    if not options:
        return None
    j, a = rng.choice(options)
    calls = copy.deepcopy(t.tool_calls)
    v = calls[j].arguments[a]
    if isinstance(v, list):
        calls[j].arguments[a] = ", ".join(map(str, v))
    elif isinstance(v, bool):
        calls[j].arguments[a] = "yes" if v else "no"
    elif isinstance(v, int):
        calls[j].arguments[a] = str(v)
    elif isinstance(v, str):
        calls[j].arguments[a] = [v]
    else:
        return None
    return Turn("assistant", t.content, calls)


# ----------------------------------------------------------------------------- over / under-calling
@perturber("unnecessary_call")
def unnecessary_call(traj, k, rng, specs, task):
    t = traj.turns[k]
    if t.tool_calls or not task or not task.irrelevant:
        return None
    cands = [s for s in specs.values() if s.effect in ("read", "pure")]
    s = rng.choice(cands)
    args: dict[str, Any] = {}
    for a in s.required:
        p = s.parameters["properties"][a]
        args[a] = {"string": "today", "integer": 5, "boolean": True, "array": []}.get(p.get("type"), "x")
        if p.get("type") == "string" and "date" in a:
            args[a] = _dt.date(2026, 3, 2).isoformat()
    return Turn("assistant", "", [ToolCall(s.name, args, "1")])


@perturber("missing_call")
def missing_call(traj, k, rng, specs, task):
    """Answer a read question without looking anything up (fabricated answer)."""
    if k != _first_action(traj) or not task or task.irrelevant or any(p["type"] != "no_change" for p in task.goal):
        return None
    return Turn("assistant", rng.choice(["You have nothing scheduled then.", "No, there is nothing new from them.", "Your calendar is free and your inbox has no unread messages."]))


@perturber("premature_answer")
def premature_answer(traj, k, rng, specs, task):
    """Claim completion of a write task without doing it."""
    if k != _first_action(traj) or not task or task.irrelevant or all(p["type"] == "no_change" for p in task.goal):
        return None
    return Turn("assistant", rng.choice(["Done! I've taken care of that for you.", "All set, that's been completed.", "Sure, I've done it."]))


@perturber("guessed_missing_info")
def guessed_missing_info(traj, k, rng, specs, task):
    """Instead of asking for the missing slot, act with a guessed value."""
    if not task or not task.missing_info:
        return None
    t = traj.turns[k]
    if t.is_action or "?" not in t.content:
        return None
    nxt = next((x for x in traj.turns[k + 1:] if x.is_action), None)
    if nxt is None:
        return None
    calls = copy.deepcopy(nxt.tool_calls)
    answer = task.missing_info["answer"]
    for c in calls:
        for a, v in list(c.arguments.items()):
            if isinstance(v, str) and _TIME_RE.match(v) and v in answer:
                c.arguments[a] = _shift_time(v, rng.choice([1, 2, -1]))
            elif isinstance(v, str) and _TIME_RE.match(v):
                c.arguments[a] = _shift_time(v, rng.choice([1, 2, -1]))
    return Turn("assistant", "", calls)


@perturber("ref_without_capability")
def ref_without_capability(traj, k, rng, specs, task):
    """Use $-references although the capability line is absent (needs the program-mode sibling in meta)."""
    if traj.capability_refs or k != _first_action(traj):
        return None
    sibling = traj.meta.get("program_first_action")
    if not sibling:
        return None
    calls = [ToolCall(c["name"], c["arguments"], c.get("id")) for c in sibling]
    if not any(isinstance(v, str) and v.startswith("$") for c in calls for v in _flat(c.arguments)):
        return None
    return Turn("assistant", "", calls)


def _flat(v: Any):
    if isinstance(v, dict):
        for x in v.values():
            yield from _flat(x)
    elif isinstance(v, list):
        for x in v:
            yield from _flat(x)
    else:
        yield v


# ----------------------------------------------------------------------------- policy
@perturber("unconfirmed_destructive")
def unconfirmed_destructive(traj, k, rng, specs, task):
    """At the confirmation question, perform the destructive action right away."""
    if not task or not task.requires_confirmation:
        return None
    t = traj.turns[k]
    if t.is_action or "?" not in t.content:
        return None
    nxt = next((x for x in traj.turns[k + 1:] if x.is_action), None)
    if nxt is None or not any(specs.get(c.name) and specs[c.name].effect == "destructive" for c in nxt.tool_calls):
        return None
    return Turn("assistant", "", copy.deepcopy(nxt.tool_calls))


@perturber("duplicate_write")
def duplicate_write(traj, k, rng, specs, task):
    """Repeat the previous successful non-idempotent write instead of answering."""
    t = traj.turns[k]
    if t.is_action:
        return None
    prev = None
    for x in reversed(traj.turns[:k]):
        if x.is_action:
            prev = x
            break
    if prev is None:
        return None
    writes = [c for c in prev.tool_calls if specs.get(c.name) and specs[c.name].effect in ("write", "destructive") and specs[c.name].idempotent is False
              and not any(isinstance(v, str) and v.startswith("$") for v in _flat(c.arguments))]
    if not writes:
        return None
    return Turn("assistant", "", [copy.deepcopy(rng.choice(writes))])


@perturber("stale_state_write")
def stale_state_write(traj, k, rng, specs, task):
    """Modify/delete by a guessed id without reading first."""
    if k != _first_action(traj):
        return None
    later = [c for x in traj.turns[k:] if x.is_action for c in x.tool_calls if specs.get(c.name) and specs[c.name].effect in ("write", "destructive") and specs[c.name].reads]
    if not later:
        return None
    c = copy.deepcopy(later[0])
    for a, v in list(c.arguments.items()):
        if isinstance(v, str) and (v.startswith("$") or _ID_RE.match(v)):
            c.arguments[a] = rng.choice(["e1", "e2", "t1", "m1"])
        elif isinstance(v, list):
            c.arguments[a] = [x for x in v if not (isinstance(x, str) and x.startswith("$"))] or ["someone@example.com"]
    c.id = "1"
    return Turn("assistant", "", [c])


# ----------------------------------------------------------------------------- driver
def perturb_trajectory(traj: Trajectory, task: Optional[Task], *, seed: int = 0, classes: Optional[list[str]] = None,
                       max_per_class: int = 1) -> list[Negative]:
    specs = traj.tool_by_name()
    rng = random.Random(seed)
    out: list[Negative] = []
    for name in (classes or list(PERTURBERS)):
        fn = PERTURBERS[name]
        made = 0
        order = traj.assistant_indices()
        rng.shuffle(order)
        for k in sorted(order, key=lambda i: rng.random()):
            bad = fn(traj, k, rng, specs, task)
            if bad is None or bad == traj.turns[k]:
                continue
            out.append(Negative(traj.id, k, bad, traj.turns[k], name, "rule", {"kind": traj.meta.get("kind"), "refs": traj.capability_refs}))
            made += 1
            if made >= max_per_class:
                break
    return out
