"""Compiler-style optimization of tool-call trajectories.

Pipeline: replay (resolved args, effects, footprints) → segment at conversational barriers (user / assistant-text
turns) → passes inside each segment → rebuild → replay the result and accept only if the final state is identical
and no new tool errors appear (legality) → diff with error classes for negative examples.

Passes (effectful calls are never removed, folded or batched away except by the batching pass that merges *identical
tool* writes into their declared batch variant; failed calls are never touched):
  cse            identical pure/read call repeated with no conflicting write in between          -> redundant_read
  info           pure/read call that adds no information not already known                      -> redundant_read
  dead           pure/read call whose new information is never used by a later call or message  -> dead_call
  fold           pure call whose result is derivable from its literal args (domain hook)         -> lookup_of_known_value
  batch          n calls of tool T with batch variant B and unused individual results            -> unbatched
  schedule       (program mode) merge calls into as few turns as data/footprint barriers allow   -> serialized_calls
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from etool.optimize.provenance import Dep, infer_arg_template, leaf_set, known_in_text, path_text
from etool.refs import dependencies
from etool.schema import ToolCall, ToolSpec, Trajectory, Turn
from etool.world.base import Domain, Op, footprints_conflict
from etool.world.verifier import NEGATION  # 'nothing found' phrasing: a final failed search is not dead
from etool.world.executor import execute_turn
from etool.world.rollout import replay


@dataclass
class Call:
    idx: int  # global index in the trajectory
    turn_idx: int
    call_id: str
    tool: str
    args: dict[str, Any]  # resolved literal args
    result: Any
    ok: bool
    spec: Optional[ToolSpec]
    reads: list[str]
    writes: list[str]
    segment: int
    removed: Optional[str] = None  # error class if removed
    template: Any = None  # args with Dep placeholders (kept calls only)
    deps: list[Dep] = field(default_factory=list)
    batched_into: Optional[int] = None
    refs: set[int] = field(default_factory=set)  # global indices of calls this call referenced explicitly with $n (same turn)

    @property
    def effect(self) -> Optional[str]:
        return self.spec.effect if self.spec else None

    @property
    def read_only(self) -> bool:
        return self.ok and self.effect in ("pure", "read")

    @property
    def effectful(self) -> bool:
        return self.effect in ("write", "destructive")


@dataclass
class Change:
    pass_name: str
    error_class: str
    turn_idx: int
    call_id: str
    tool: str
    detail: str = ""


@dataclass
class OptResult:
    original: Trajectory
    optimized: Trajectory
    changes: list[Change]
    accepted: bool
    reason: str = ""

    @property
    def classes(self) -> list[str]:
        return [c.error_class for c in self.changes]


# --------------------------------------------------------------------------- analysis
def _calls_from(domain: Domain, traj: Trajectory) -> tuple[list[Call], dict[str, Any]]:
    state = replay(domain, traj)
    ops: dict[tuple[int, str], Op] = {(o.turn_idx, o.call_id): o for o in state.ops}
    results: dict[tuple[int, str], Any] = {}
    for t in traj.turns:
        if t.role == "tool" and t.tool_call_id is not None:
            pass
    # map tool turns to the preceding action turn
    last_action = -1
    for i, t in enumerate(traj.turns):
        if t.is_action:
            last_action = i
        elif t.role == "tool" and last_action >= 0:
            try:
                results[(last_action, t.tool_call_id or "")] = json.loads(t.content)
            except json.JSONDecodeError:
                results[(last_action, t.tool_call_id or "")] = t.content
    calls: list[Call] = []
    seg = 0
    prev_was_action_block = False
    for i, t in enumerate(traj.turns):
        if t.role in ("user",) or (t.role == "assistant" and not t.is_action):
            if prev_was_action_block:
                seg += 1
            prev_was_action_block = False
            continue
        if t.is_action:
            prev_was_action_block = True
            first = len(calls)
            for j, c in enumerate(t.tool_calls):
                cid = c.id or str(j + 1)
                op = ops.get((i, cid))
                spec = domain.tools[c.name].spec if c.name in domain.tools else None
                refs = {first + k - 1 for k in dependencies(c.arguments) if 1 <= k <= j} if traj.capability_refs else set()
                calls.append(Call(len(calls), i, cid, c.name, (op.args if op else c.arguments), results.get((i, cid)),
                                  bool(op and op.ok), spec, list(op.reads) if op else [], list(op.writes) if op else [], seg, refs=refs))
    return calls, state.snapshot()


_EMPTY_WORDS = ("nothing", "no ", "none", "empty", "0 ", "zero", "free", "not find", "n't find", "no longer")


def _mentions_emptiness(text: str) -> bool:
    return any(w in text for w in _EMPTY_WORDS)


def _has_empty_collection(value: Any) -> bool:
    if isinstance(value, (list, dict)) and len(value) == 0:
        return True
    if isinstance(value, dict):
        return any(_has_empty_collection(v) for v in value.values())
    if isinstance(value, list):
        return any(_has_empty_collection(v) for v in value)
    return False


def _dicts(value: Any):
    if isinstance(value, dict):
        yield value
        for v in value.values():
            yield from _dicts(v)
    elif isinstance(value, list):
        for v in value:
            yield from _dicts(v)


def _record_contexts(result: Any, target: str) -> list[set[str]]:
    """For every dict record inside ``result`` that holds ``target`` as one of its *own* scalar values (or inside a list of
    scalars), the record's other leaves. A sender who wrote several emails yields one context per email record."""
    out = []
    for d in _dicts(result):
        own = set()
        for v in d.values():
            if isinstance(v, (str, int, float)) and not isinstance(v, bool):
                own |= leaf_set(v)
            elif isinstance(v, list) and all(not isinstance(x, (dict, list)) for x in v):
                own |= leaf_set(v)
        if target in own:
            out.append(leaf_set(d) - {target})
    return out


_WORD = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)*")  # quotes/apostrophes around a word are not part of it


def _user_keyed(x: str, user_texts: list[str]) -> bool:
    """Does record value ``x`` correspond to something the user said: the value occurs in the user's text, or a two-word
    phrase of the user's text occurs in the value (\"vendor contract\" in \"Re: vendor contract notes\")."""
    if len(x) < 2:
        return False
    if known_in_text(x, user_texts):
        return True
    xl = x.lower()
    for t in user_texts:
        w = [x for x in _WORD.findall(t.lower()) if len(x) >= 3]
        if any(f"{a} {b}" in xl for a, b in zip(w, w[1:])):
            return True
    return False


def _supplier(v: str, producers: list["Call"], user_texts: list[str]) -> Optional[int]:
    """Which earlier kept call supplies the literal ``v`` to a later argument. Several results may carry the same value
    (an address id also sits in every order record shipped there): prefer the earliest producer whose record holding ``v``
    also holds a value from the user's own words (a lookup by a user-given key, e.g. label "office" -> id "a2"); otherwise the
    earliest producer."""
    cands = [c for c in producers if c.result is not None and v in leaf_set(c.result)]
    strong = [c for c in cands if any(_user_keyed(x, user_texts) for ctx in _record_contexts(c.result, v) for x in ctx)]
    pick = strong[0] if strong else (cands[0] if cands else None)
    return pick.idx if pick else None


def _conflict(a: Call, b: Call) -> bool:
    """Is there a footprint conflict between earlier call a and later call b (RAW / WAW / WAR)?"""
    return any(footprints_conflict(x, y) for x in a.writes for y in b.reads + b.writes) or \
        any(footprints_conflict(x, y) for x in a.reads for y in b.writes)


def _texts(traj: Trajectory, upto_turn: Optional[int] = None) -> list[str]:
    return [t.content for i, t in enumerate(traj.turns) if t.role in ("user", "system") and (upto_turn is None or i < upto_turn)]


def _later_text(traj: Trajectory, after_turn: int) -> list[str]:
    return [t.content for i, t in enumerate(traj.turns) if i > after_turn and t.role == "assistant" and not t.is_action]


# --------------------------------------------------------------------------- passes
def pass_cse_info_dead(domain: Domain, traj: Trajectory, calls: list[Call], conservative: bool = True) -> list[Change]:
    changes: list[Change] = []
    texts = _texts(traj)
    known: set[str] = set()  # normalised leaves of kept results
    kept: list[Call] = []

    def unknown(values: set[str]) -> set[str]:
        return {v for v in values if v not in known and not known_in_text(v, texts)}

    for c in calls:
        if c.removed:
            continue
        if not c.read_only:
            kept.append(c)
            known |= leaf_set(c.result)
            continue
        res_leaves = leaf_set(c.result)
        if c.effect == "pure" and res_leaves and all(known_in_text(v, texts) for v in res_leaves):
            c.removed = "lookup_of_known_value"
            changes.append(Change("info", "lookup_of_known_value", c.turn_idx, c.call_id, c.tool, "the user already gave this value"))
            continue
        # cse: identical earlier kept call in the same segment. Intervening writes are the agent's own; their results
        # already describe the new state, so a re-read is redundant unless one of those conflicting writes failed.
        dup = None
        failed_write_between = False
        for k in reversed(kept):
            if k.segment != c.segment:
                break
            if k.tool == c.tool and k.args == c.args and k.read_only:
                dup = k
                break
            if k.effectful and not k.ok and _conflict(k, c):
                failed_write_between = True
        if dup is not None and not failed_write_between:
            c.removed = "redundant_read"
            changes.append(Change("cse", "redundant_read", c.turn_idx, c.call_id, c.tool, f"same as call {dup.call_id} in turn {dup.turn_idx}"))
            continue
        new_info = unknown(res_leaves)
        later_text = " ".join(_later_text(traj, c.turn_idx)).lower()
        if not new_info:
            if _has_empty_collection(c.result) and (conservative or _mentions_emptiness(later_text)):
                kept.append(c)  # "nothing there" is information, and it is (or may be) what the agent reports
                continue
            # every value was already known (or the emptiness is never reported)
            cls = "dead_call" if _has_empty_collection(c.result) else "redundant_read"
            c.removed = cls
            changes.append(Change("info", cls, c.turn_idx, c.call_id, c.tool, "adds no new information"))
            continue
        # dead: new info never used by a later call's args (beyond what the user said) or a later assistant message.
        # Argument use is attributed to one supplier per value (see _supplier), so a value that several results carry
        # counts for the result the agent actually looked it up in.
        later_args: set[str] = set()
        for d in calls[c.idx + 1:]:
            if d.removed:
                continue
            d_texts = _texts(traj, d.turn_idx)
            d_user = [t.content for t in traj.turns[: d.turn_idx] if t.role == "user"]
            d_producers = [k for k in calls[: d.idx] if not k.removed and k.result is not None]
            for v in leaf_set(d.args):
                if not known_in_text(v, d_texts) and _supplier(v, d_producers, d_user) == c.idx:
                    later_args.add(v)
        # a value that a *later* kept call also returns is attributed to that fresher source for text uses
        later_results: set[str] = set()
        for d in calls[c.idx + 1:]:
            if not d.removed and d.result is not None:
                later_results |= leaf_set(d.result)
        used = bool(later_args) or any(len(v) >= 2 and v in later_text and v not in later_results for v in new_info)
        # an explicit $ref from a kept later call is a declared data dependency: the producer is live even when the same value
        # happens to occur elsewhere (e.g. an address id that another order record also carries)
        used = used or any(c.idx in d.refs for d in calls[c.idx + 1:] if not d.removed)
        # Conservative dead-call rule: a read that brings *new identifiers* into play (a contact, an event id, an
        # address) may silently drive the agent's later decisions (what to filter, which item to pick) even if no value
        # is copied, so it is kept. Pure calls and detail look-ups on already-known entities are safe to drop when unused.
        new_ids = {v for v in new_info if domain.is_identifier(v)}
        later_calls = any(not d.removed for d in calls[c.idx + 1:])
        # A final read whose (non-empty) result the agent then reports as "not found" is a failed search: the absence of a
        # match is the information used, so it cannot be proven dead (test-only kinds missing_event / missing_sender).
        if not used and c.read_only and not later_calls and NEGATION.search(later_text):
            kept.append(c)
            known |= res_leaves
            continue
        if not used and (not conservative or c.effect == "pure" or not new_ids or not later_calls):
            c.removed = "dead_call"
            changes.append(Change("dead", "dead_call", c.turn_idx, c.call_id, c.tool, "result never used"))
            continue
        kept.append(c)
        known |= res_leaves
    return changes


def pass_fold(domain: Domain, calls: list[Call]) -> list[Change]:
    changes: list[Change] = []
    for c in calls:
        if c.removed or not c.read_only or c.effect != "pure":
            continue
        folded = domain.fold(c.tool, c.args)
        if folded is not None and folded == c.result:
            c.removed = "lookup_of_known_value"
            changes.append(Change("fold", "lookup_of_known_value", c.turn_idx, c.call_id, c.tool, "result derivable from the arguments"))
    return changes


def _batch_mapping(domain: Domain, item_spec: ToolSpec, batch_spec: ToolSpec) -> Optional[tuple[str, str]]:
    ip = set(item_spec.parameters.get("properties", {}))
    bp = set(batch_spec.parameters.get("properties", {}))
    only_item, only_batch = ip - bp, bp - ip
    if len(only_item) == 1 and len(only_batch) == 1:
        la = next(iter(only_batch))
        if batch_spec.parameters["properties"][la].get("type") == "array":
            return next(iter(only_item)), la
    return None


def pass_batch(domain: Domain, calls: list[Call]) -> tuple[list[Change], list[Call]]:
    """Merge runs of per-item calls into their batch variant. Returns changes and synthetic batch calls (appended)."""
    changes: list[Change] = []
    batch_specs = {t.spec.batch_of: t.spec for t in domain.tools.values() if t.spec.batch_of}
    new_calls: list[Call] = []
    by_seg_tool: dict[tuple[int, str], list[Call]] = {}
    for c in calls:
        if c.removed or not c.ok or c.tool not in batch_specs:
            continue
        by_seg_tool.setdefault((c.segment, c.tool), []).append(c)
    for (seg, tool), group in by_seg_tool.items():
        if len(group) < 2:
            continue
        item_spec, batch_spec = domain.tools[tool].spec, batch_specs[tool]
        mapping = _batch_mapping(domain, item_spec, batch_spec)
        if mapping is None:
            continue
        item_arg, list_arg = mapping
        rest = {k: v for k, v in group[0].args.items() if k != item_arg}
        if any({k: v for k, v in g.args.items() if k != item_arg} != rest for g in group):
            continue
        # individual results must not feed later calls; no conflicting op between first and last of the group
        first, last = group[0], group[-1]
        between = [c for c in calls if first.idx < c.idx < last.idx and not c.removed and c not in group]
        if any(_conflict(b, last) or _conflict(first, b) for b in between):
            continue
        later = [c for c in calls if c.idx > first.idx and not c.removed and c not in group]
        leaves_group = set().union(*[leaf_set(g.result) for g in group]) - set().union(*[leaf_set(g.args) for g in group])
        if any(leaf_set(l.args) & leaves_group for l in later):
            continue
        bc = Call(first.idx, first.turn_idx, first.call_id, batch_spec.name, {**rest, list_arg: [g.args[item_arg] for g in group]}, None, True,
                  batch_spec, [], [], seg)
        bc.template = dict(bc.args)
        for g in group:
            g.removed = "unbatched"
            g.batched_into = first.idx
            changes.append(Change("batch", "unbatched", g.turn_idx, g.call_id, g.tool, f"merged into {batch_spec.name}"))
        new_calls.append(bc)
    return changes, new_calls


# --------------------------------------------------------------------------- rebuild
def _schedule_segment(seg_calls: list[Call], program: bool) -> list[list[Call]]:
    """Assign kept calls of one segment to turns. Program mode: turn = barrier depth; stepwise: one per turn."""
    if not program:
        return [[c] for c in seg_calls]
    depth: dict[int, int] = {}
    for c in seg_calls:
        d = 0
        for e in seg_calls:
            if e.idx >= c.idx:
                break
            barrier = False
            for dep in c.deps:
                if dep.src == e.idx and not dep.in_program:
                    barrier = True  # data-dependent selection or composed text: must see e's result first
            if not barrier and (e.effectful or c.effectful) and _conflict(e, c) and not any(dp.src == e.idx for dp in c.deps):
                barrier = True  # conflicting effects without a data edge: keep them in separate turns
            if barrier:
                d = max(d, depth[e.idx] + 1)
            elif any(dp.src == e.idx for dp in c.deps):
                d = max(d, depth[e.idx])
        depth[c.idx] = d
    turns: dict[int, list[Call]] = {}
    for c in seg_calls:
        turns.setdefault(depth[c.idx], []).append(c)
    return [turns[k] for k in sorted(turns)]


def _materialize(template: Any, same_turn: dict[int, int], results: dict[int, Any]) -> Any:
    if isinstance(template, Dep):
        if template.src in same_turn:
            return f"${same_turn[template.src]}{path_text(template.path)}"
        from etool.refs import follow
        return follow(results[template.src], template.path)
    if isinstance(template, dict):
        return {k: _materialize(v, same_turn, results) for k, v in template.items()}
    if isinstance(template, list):
        return [_materialize(v, same_turn, results) for v in template]
    return template


def _rebuild(domain: Domain, traj: Trajectory, calls: list[Call], program: bool) -> tuple[Trajectory, int]:
    """Emit the optimized trajectory by executing the kept calls on a fresh state. Returns (trajectory, n_errors)."""
    kept = [c for c in calls if not c.removed]
    seg_of: dict[int, list[Call]] = {}
    for c in kept:
        seg_of.setdefault(c.segment, []).append(c)
    for seg in seg_of.values():
        seg.sort(key=lambda c: c.idx)
    state = domain.new_state(int(traj.meta.get("seed", traj.task["seed"] if traj.task else 0)))
    new_turns: list[Turn] = []
    results: dict[int, Any] = {}
    n_err = 0
    emitted_segments: set[int] = set()
    seg = 0
    prev_was_action_block = False

    def emit_segment(s: int) -> None:
        nonlocal n_err
        for group in _schedule_segment(seg_of.get(s, []), program):
            same_turn = {c.idx: i + 1 for i, c in enumerate(group)}
            tcs = [ToolCall(c.tool, _materialize(c.template if c.template is not None else c.args, same_turn, results), str(i + 1)) for i, c in enumerate(group)]
            turn = Turn("assistant", "", tcs)
            ti = len(new_turns)
            new_turns.append(turn)
            res = execute_turn(domain, state, turn, turn_idx=ti, capability_refs=program)
            for c, tt in zip(group, res.tool_turns):
                try:
                    results[c.idx] = json.loads(tt.content)
                except json.JSONDecodeError:
                    results[c.idx] = tt.content
                if tt.meta.get("error"):
                    n_err += 1
            new_turns.extend(res.tool_turns)
        emitted_segments.add(s)

    for t in traj.turns:
        if t.role == "user" or (t.role == "assistant" and not t.is_action):
            if prev_was_action_block:
                emit_segment(seg)
                seg += 1
            prev_was_action_block = False
            new_turns.append(t)
        elif t.is_action:
            prev_was_action_block = True
        # tool turns are regenerated
    if prev_was_action_block:
        emit_segment(seg)
    out = Trajectory(id=traj.id + ":opt", source=traj.source + ":opt", tools=traj.tools, turns=new_turns, capability_refs=program,
                     task=traj.task, gold_final_state=traj.gold_final_state, meta={**traj.meta, "optimized_from": traj.id})
    return out, n_err


def _infer_templates(traj: Trajectory, calls: list[Call]) -> None:
    """Earliest producer wins, across segments: a value first seen in an earlier segment is a literal here (no
    in-segment dependency), which keeps scheduling barriers honest."""
    kept = [c for c in calls if not c.removed]
    for c in kept:
        producers = [(k.idx, k.result) for k in kept if k.idx < c.idx and k.result is not None]
        texts = _texts(traj, c.turn_idx)
        c.template, c.deps = infer_arg_template(c.args, producers, texts)
        c.deps = [d for d in c.deps if calls_by_idx(calls, d.src).segment == c.segment] + \
                 [d for d in c.deps if calls_by_idx(calls, d.src).segment != c.segment]


def calls_by_idx(calls: list[Call], idx: int) -> Call:
    for c in calls:
        if c.idx == idx and not c.removed:
            return c
    return [c for c in calls if c.idx == idx][0]


# --------------------------------------------------------------------------- entry point
def optimize(domain: Domain, traj: Trajectory, *, program: Optional[bool] = None, conservative: bool = True) -> OptResult:
    """``conservative`` (default): never drop a read that introduces new identifiers while later tool calls exist, since
    such a result may drive the agent's decisions without any value being copied (use for teacher / model traces).
    Set False for planner-generated traces whose redundant calls are redundant by construction."""
    program = traj.capability_refs if program is None else program
    calls, final_orig = _calls_from(domain, traj)
    n_err_orig = sum(1 for c in calls if not c.ok)
    changes = pass_fold(domain, calls)
    for _ in range(8):  # fixpoint: a call that only fed a removed call becomes dead in the next round
        ch = pass_cse_info_dead(domain, traj, calls, conservative)
        changes += ch
        if not ch:
            break
    bch, new_calls = pass_batch(domain, calls)
    changes += bch
    calls = sorted(calls + new_calls, key=lambda c: (c.idx, c.tool))
    _infer_templates(traj, calls)
    # serialized calls: in program mode, count original action turns beyond what the schedule needs
    opt, n_err = _rebuild(domain, traj, calls, program)
    if program:
        orig_turns = traj.n_action_turns()
        if opt.n_action_turns() < orig_turns:
            changes.append(Change("schedule", "serialized_calls", -1, "", "", f"{orig_turns} action turns -> {opt.n_action_turns()}"))
    final_new = replay(domain, opt).snapshot()
    accepted = final_new == final_orig and n_err <= n_err_orig
    if not accepted:
        return OptResult(traj, traj, [], False, "state mismatch" if final_new != final_orig else "new tool errors")
    return OptResult(traj, opt, changes, True)
