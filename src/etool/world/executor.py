"""Execute one assistant turn (a tool program) against a domain state.

* builds the dependency DAG from ``$ref`` arguments, groups independent calls into parallel levels,
* resolves references, validates arguments, runs the tool, records an ``Op`` with resolved footprints,
* returns one ``tool`` turn per call (in call order) with JSON results or structured errors,
* simulated latency = sum over levels of the max latency inside the level.

The executor never *refuses* a policy-violating call (e.g. destructive without confirmation): mistakes must be
observable so the verifier can score them. It does refuse references when the capability flag is off.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from etool.refs import RefError, dependencies, has_refs, resolve
from etool.schema import ToolCall, Turn
from etool.world.base import Domain, Op, ToolError, WorldState, footprints_conflict, validate_args


@dataclass
class ExecResult:
    tool_turns: list[Turn]
    groups: list[list[str]]  # call ids per parallel level, in execution order
    latency_ms: float
    ops: list[Op]
    errors: dict[str, str] = field(default_factory=dict)  # call id -> error code

    @property
    def ok(self) -> bool:
        return not self.errors


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)


def assign_ids(calls: list[ToolCall]) -> list[ToolCall]:
    """Give calls their 1-based positional ids if missing (returns new objects)."""
    return [ToolCall(c.name, c.arguments, c.id or str(i + 1)) for i, c in enumerate(calls)]


def _footprint_edges(calls: list[ToolCall], domain: Optional[Domain]) -> dict[str, set[str]]:
    """Ordering edges from footprint conflicts (RAW / WAW / WAR) between calls of one turn, computed on the *unresolved*
    arguments (a ``$ref`` placeholder becomes a ``*`` wildcard, i.e. conservatively conflicting). Two reads never conflict."""
    out: dict[str, set[str]] = {c.id: set() for c in calls}
    if domain is None:
        return out
    fp = {}
    for c in calls:
        tool = domain.tools.get(c.name)
        if tool is None or not isinstance(c.arguments, dict):
            fp[c.id] = ([], [])
            continue
        fp[c.id] = tool.footprints({k: ("*" if has_refs({k: v}) else v) for k, v in c.arguments.items()})
    for i, b in enumerate(calls):
        rb, wb = fp[b.id]
        for a in calls[:i]:
            ra, wa = fp[a.id]
            if any(footprints_conflict(x, y) for x in wa for y in rb + wb) or any(footprints_conflict(x, y) for x in ra for y in wb):
                out[b.id].add(a.id)
    return out


def plan_levels(calls: list[ToolCall], domain: Optional[Domain] = None) -> tuple[list[list[str]], dict[str, str]]:
    """Kahn levels over the ``$ref`` DAG plus footprint-conflict edges (a write and a later read/write of the same resource
    in one turn run in order, not in parallel). Returns (levels, errors) where errors maps call id -> code for
    forward/self references (those calls are excluded from the levels)."""
    ids = [c.id for c in calls]
    pos = {cid: i for i, cid in enumerate(ids)}
    deps: dict[str, set[str]] = {}
    errors: dict[str, str] = {}
    fp_edges = _footprint_edges(calls, domain)
    for c in calls:
        d = {str(i) for i in dependencies(c.arguments)}
        bad = [x for x in d if x not in pos or pos[x] >= pos[c.id]]  # unknown, self or forward reference
        if bad:
            errors[c.id] = "invalid_ref"
            continue
        deps[c.id] = d | fp_edges.get(c.id, set())
    # calls depending (transitively) on an invalid call are also invalid
    changed = True
    while changed:
        changed = False
        for cid, d in list(deps.items()):
            if any(x in errors for x in d):
                errors[cid] = "invalid_ref"
                del deps[cid]
                changed = True
    levels: list[list[str]] = []
    done: set[str] = set()
    remaining = dict(deps)
    while remaining:
        ready = [cid for cid, d in remaining.items() if d <= done]
        if not ready:  # cannot happen for a DAG of backward refs, but keep it total
            for cid in remaining:
                errors[cid] = "invalid_ref"
            break
        ready.sort(key=lambda x: pos[x])
        levels.append(ready)
        done.update(ready)
        for cid in ready:
            del remaining[cid]
    return levels, errors


def execute_turn(domain: Domain, state: WorldState, turn: Turn, *, turn_idx: int, capability_refs: bool,
                 excluded: frozenset[str] = frozenset()) -> ExecResult:
    """``excluded``: tool names hidden from the prompt for this task; calling one is an ``unknown_tool`` error."""
    calls = assign_ids(turn.tool_calls)
    by_id = {c.id: c for c in calls}
    results: dict[int, Any] = {}
    observations: dict[str, Any] = {}
    errors: dict[str, str] = {}
    ops: list[Op] = []

    # 1. references only when advertised
    if not capability_refs:
        for c in calls:
            if has_refs(c.arguments):
                errors[c.id] = "refs_not_supported"
                observations[c.id] = ToolError(
                    "refs_not_supported", "References like \"$1\" are not supported here; call tools one step at a time."
                ).observation()
    live = [c for c in calls if c.id not in errors]

    # 2. DAG levels
    levels, ref_errors = plan_levels(live, domain)
    for cid, code in ref_errors.items():
        errors[cid] = code
        observations[cid] = ToolError("invalid_ref", "Reference to an unknown, later or failed call in this turn.").observation()

    # 3. execute level by level
    total_latency = 0.0
    for g, level in enumerate(levels):
        level_latency = 0.0
        for cid in level:
            c = by_id[cid]
            tool = None if c.name in excluded else domain.tools.get(c.name)
            lat = 0.0
            reads: list[str] = []
            writes: list[str] = []
            ok = False
            code: Optional[str] = None
            args: dict[str, Any] = c.arguments
            try:
                try:
                    args = resolve(c.arguments, results)
                except RefError as e:
                    raise ToolError("unresolved_ref", str(e))
                if tool is None:
                    raise ToolError("unknown_tool", f"No tool named '{c.name}'.")
                problems = validate_args(tool.spec, args)
                if problems:
                    raise ToolError("invalid_arguments", "; ".join(problems))
                reads, writes = tool.footprints(args)
                lat = tool.spec.latency_ms * state.rng.uniform(0.8, 1.2)
                out = tool.fn(state, args)
                results[int(cid)] = out
                observations[cid] = out
                ok = True
            except ToolError as e:
                code = e.code
                errors[cid] = code
                observations[cid] = e.observation()
                if tool is not None:
                    reads, writes = tool.footprints(args) if isinstance(args, dict) else ([], [])
                    if not ok:
                        writes = []  # failed calls do not write
            ops.append(Op(turn_idx, cid, c.name, tool.spec.effect if tool else None, tool.spec.idempotent if tool else None,
                          reads, writes if ok else [], ok, code, lat, args if isinstance(args, dict) else {}, group=g))
            level_latency = max(level_latency, lat)
        total_latency += level_latency

    # calls that were never scheduled (invalid refs) still get an op for bookkeeping
    scheduled = {cid for lvl in levels for cid in lvl}
    for c in calls:
        if c.id not in scheduled and not any(o.call_id == c.id for o in ops):
            ops.append(Op(turn_idx, c.id, c.name, None, None, [], [], False, errors.get(c.id), 0.0, c.arguments, group=-1))
    ops.sort(key=lambda o: int(o.call_id))
    state.ops.extend(ops)

    tool_turns = [
        Turn("tool", _dumps(observations[c.id]), tool_call_id=c.id, name=c.name,
             meta={"ok": c.id not in errors, "error": errors.get(c.id)})
        for c in calls
    ]
    return ExecResult(tool_turns, levels, total_latency, ops, errors)
