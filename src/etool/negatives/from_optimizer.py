"""Negatives from the optimizer diff: at a prefix of the original trajectory, the redundant/serialized action is the
bad turn and the optimized continuation is the good turn."""
from __future__ import annotations

from typing import Optional

from etool.optimize.core import OptResult
from etool.schema import Negative, Trajectory, Turn


def _same_turn(a: Turn, b: Turn) -> bool:
    if a.role != b.role or a.content.strip() != b.content.strip():
        return False
    return [(c.name, c.arguments) for c in a.tool_calls] == [(c.name, c.arguments) for c in b.tool_calls]


def _first_divergence(orig: Trajectory, opt: Trajectory) -> Optional[tuple[int, int]]:
    i = j = 0
    while i < len(orig.turns) and j < len(opt.turns):
        a, b = orig.turns[i], opt.turns[j]
        if a.role == "assistant" and b.role == "assistant":
            if not _same_turn(a, b):
                return i, j
        elif not (a.role == b.role and (a.role != "tool" or a.content == b.content) and (a.role == "tool" or a.content == b.content)):
            return None  # prefixes diverged on a non-assistant turn: cannot pair
        i += 1
        j += 1
    return None


def negatives_from_optimizer(res: OptResult) -> list[Negative]:
    if not res.accepted or not res.changes:
        return []
    orig, opt = res.original, res.optimized
    out: list[Negative] = []
    classes_by_turn: dict[int, str] = {}
    for ch in res.changes:
        if ch.turn_idx >= 0 and ch.turn_idx not in classes_by_turn:
            classes_by_turn[ch.turn_idx] = ch.error_class
    div = _first_divergence(orig, opt)
    if div is not None:
        i, j = div
        cls = classes_by_turn.get(i, "serialized_calls" if orig.capability_refs else "redundant_read")
        out.append(Negative(orig.id, i, orig.turns[i], opt.turns[j], cls, "optimizer", {"kind": orig.meta.get("kind"), "refs": orig.capability_refs, "first_divergence": True}))
    # further removed single-call turns: good = next kept action (literal args) or the final answer
    first_i = div[0] if div else -1
    for i, cls in sorted(classes_by_turn.items()):
        if i <= first_i:
            continue
        t = orig.turns[i]
        if not t.is_action or len(t.tool_calls) != 1:
            continue
        nxt = None
        for k in range(i + 1, len(orig.turns)):
            u = orig.turns[k]
            if u.role == "assistant" and (not u.is_action or k not in classes_by_turn):
                nxt = u
                break
        if nxt is None or _same_turn(nxt, t):
            continue
        if orig.capability_refs and nxt.is_action:
            continue  # in program mode the proper continuation is a merged turn; only the first divergence is reliable
        out.append(Negative(orig.id, i, t, nxt, cls, "optimizer", {"kind": orig.meta.get("kind"), "refs": orig.capability_refs, "first_divergence": False}))
    return out
