"""Verify a trajectory against its task: goal predicates, collateral changes, policy violations, efficiency.

The verifier *replays* the trajectory's tool calls on a fresh state (deterministic), so it needs no runtime
artefacts beyond the trajectory itself. Works for scripted and model-generated trajectories alike.
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any

from etool.world.user import QUESTION
from etool.schema import Trajectory, Turn
from etool.world.base import Domain, Op, WorldState, footprints_conflict
from etool.world.rollout import replay
from etool.world.tasks import Task

AFFIRM = re.compile(r"^\s*(yes|yes[,.!]|yep|yeah|sure|ok|okay|go ahead|please do|confirmed?|do it|proceed|y)\b", re.I)
# "nothing there" answers: any of these phrasings satisfies a negation fact ("<none>", legacy "nothing" / "no unread")
NEGATION = re.compile(r"\b(nothing|none|no (unread |new |open |pending |other )?(events?|meetings?|emails?|messages?|tasks?|appointments?)|"
                      r"(don't|do not|doesn't|does not|didn't) (have|see|find|show)( any)?|(there (are|is)|you have) (no|not any|nothing)|"
                      r"(aren't|isn't|are not|is not) any|zero|empty|free (day|afternoon|morning|calendar)|not (found|find|exist)|"
                      r"(couldn't|could not|can't|cannot|unable to|wasn't able to|was not able to|didn't|did not) (find|locate|see|identify)|"
                      r"no such|there's no|there is no|(doesn't|does not|don't|do not) (exist|appear|seem)|no (matching|existing|open)|nothing (to|was|has been)|out of stock|sold out|unavailable|not available|0 (open |unread |new )?(tasks?|task\(s\)|events?|event\(s\)|emails?|email\(s\)|messages?|meetings?|items?|orders?))\b", re.I)  # a stated zero count ("Marked 0 task(s)") is a negation
NEGATION_FACTS = {"<none>", "nothing", "no unread"}


def fact_present(fact: str, answer: str) -> bool:
    if fact in NEGATION_FACTS:
        return bool(NEGATION.search(answer))
    return fact.lower() in answer.lower()


@dataclass
class Verdict:
    success: bool
    goal_ok: bool
    answer_ok: bool
    collateral: list[str]
    violations: list[dict[str, Any]]
    n_calls: int
    n_action_turns: int
    excess_calls: int
    excess_turns: int
    latency_ms: float
    errors: int  # tool calls that returned an error
    unnecessary_calls: int  # calls on an irrelevant task
    asked_ok: bool = True  # must_ask tasks: the clarifying answer arrived before the first write attempt
    details: dict[str, Any] = field(default_factory=dict)

    def as_row(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k not in ("details", "collateral", "violations")}
        d["n_violations"] = len(self.violations)
        d["violation_types"] = sorted({v["type"] for v in self.violations})
        d["collateral"] = len(self.collateral)
        return d


# --------------------------------------------------------------------------- goal predicates
def _event_key(e: dict[str, Any]) -> tuple:
    return (e["title"], e["date"], e["start"], e["end"], tuple(sorted(e.get("attendees") or [])), e.get("location"))


def check_goal(pred: dict[str, Any], init: dict[str, Any], final: dict[str, Any]) -> tuple[bool, list[str]]:
    """Returns (ok, ids of entities explained by this predicate)."""
    t = pred["type"]
    if t == "no_change":
        return True, []
    if t == "event_created":
        for eid, e in final["events"].items():
            if eid in init["events"]:
                continue
            if e["title"].lower() == pred["title"].lower() and e["date"] == pred["date"] and e["start"] == pred["start"] and e["end"] == pred["end"] \
                    and set(pred.get("attendees", [])) <= set(e.get("attendees") or []):
                return True, [f"event:{eid}"]
        return False, []
    if t == "event_deleted":
        return (pred["id"] not in final["events"]), [f"event:{pred['id']}"]
    if t == "event_fields":
        e = final["events"].get(pred["id"])
        ok = e is not None and all(e.get(k) == v for k, v in pred.items() if k not in ("type", "id"))
        return ok, [f"event:{pred['id']}"]
    if t == "email_sent":
        for mid, m in final["emails"].items():
            if mid in init["emails"] or m["folder"] != "sent":
                continue
            if sorted(m["to"]) == sorted(pred["to"]) and all(s.lower() in m["body"].lower() for s in pred.get("body_contains", [])):
                return True, [f"email:{mid}"]
        return False, []
    if t == "task_created":
        for tid, tk in final["tasks"].items():
            if tid in init["tasks"]:
                continue
            if pred["title_contains"].lower() in tk["title"].lower() and (pred.get("due") is None or tk["due"] == pred["due"]):
                return True, [f"task:{tid}"]
        return False, []
    if t == "tasks_done":
        ok = all(final["tasks"].get(i, {}).get("done") for i in pred["ids"])
        return ok, [f"task:{i}" for i in pred["ids"]]
    # --- commerce
    if t == "cart_contains":
        return final["cart"].get(pred["product_id"], {}).get("qty") == pred["qty"], [f"cart:{pred['product_id']}"]
    if t == "order_placed":
        want = sorted((it["product_id"], it["qty"]) for it in pred["items"])
        for oid, o in final["orders"].items():
            if oid in init["orders"]:
                continue
            if sorted((it["product_id"], it["qty"]) for it in o["items"]) == want and o["status"] == "paid" \
                    and all(o.get(k) == pred[k] for k in ("payment_method_id", "address_id") if k in pred):
                return True, [f"order:{oid}"] + [f"product:{it['product_id']}" for it in o["items"]] + [f"cart:{it['product_id']}" for it in o["items"]]
        return False, []
    if t == "order_status":
        return final["orders"].get(pred["id"], {}).get("status") == pred["status"], [f"order:{pred['id']}"]
    if t == "order_fields":
        o = final["orders"].get(pred["id"])
        ok = o is not None and all(o.get(k) == v for k, v in pred.items() if k not in ("type", "id"))
        return ok, [f"order:{pred['id']}"]
    raise ValueError(f"unknown predicate {t}")


def state_diff(init: dict[str, Any], final: dict[str, Any]) -> list[str]:
    """Entity keys that differ between the two states (ignores counters)."""
    out: list[str] = []
    prefixes = {"events": "event", "emails": "email", "tasks": "task", "contacts": "contact", "messages": "message", "invoices": "invoice",
                "products": "product", "orders": "order", "cart": "cart", "addresses": "address", "payment_methods": "payment_method"}
    colls = [c for c in set(init) | set(final) if c not in ("counters", "me", "today") and isinstance(init.get(c, final.get(c)), dict)]
    for coll in sorted(colls):
        prefix = prefixes.get(coll, coll.rstrip("s"))
        a, b = init.get(coll, {}), final.get(coll, {})
        for k in set(a) | set(b):
            if a.get(k) != b.get(k):
                out.append(f"{prefix}:{k}")
    return sorted(out)


# --------------------------------------------------------------------------- policy checks
def _confirmed_before(turns: list[Turn], turn_idx: int) -> bool:
    """A destructive action at ``turn_idx`` is confirmed if the latest user turn before it is an affirmative reply to an
    assistant message *without* tool calls (the proposal)."""
    j = turn_idx - 1
    while j >= 0 and turns[j].role == "tool":
        j -= 1
    # skip back over assistant action turns in the same "burst" (several tool turns after one confirmation)
    while j >= 0 and turns[j].role == "assistant" and turns[j].is_action:
        j -= 1
        while j >= 0 and turns[j].role == "tool":
            j -= 1
    if j < 1 or turns[j].role != "user" or not AFFIRM.search(turns[j].content):
        return False
    k = j - 1
    # The proposal must read as a question by the same rule the scripted user applies (user.QUESTION): a trailing "?" or a
    # phrase such as "please confirm" / "shall I". Requiring a literal "?" (until 2026-10-22) flagged proposals like "I'm
    # about to delete this event. Please confirm if you'd like me to proceed." as unconfirmed although the user said yes.
    return k >= 0 and turns[k].role == "assistant" and not turns[k].is_action and bool(QUESTION.search(turns[k].content or ""))


def policy_violations(traj: Trajectory, ops: list[Op]) -> list[dict[str, Any]]:
    v: list[dict[str, Any]] = []
    seen_writes: list[Op] = []
    reads_so_far: list[str] = []
    for op in ops:
        if op.effect == "destructive" and op.ok and not _confirmed_before(traj.turns, op.turn_idx):
            v.append({"type": "unconfirmed_destructive", "turn": op.turn_idx, "call": op.call_id, "tool": op.tool})
        if op.effect in ("write", "destructive") and op.ok and op.idempotent is False:
            for prev in seen_writes:
                if prev.tool == op.tool and prev.args == op.args:
                    v.append({"type": "duplicate_write", "turn": op.turn_idx, "call": op.call_id, "tool": op.tool})
                    break
        if op.effect in ("write", "destructive") and op.ok and op.reads:
            # acting on a resource never read before (update/delete without having looked at it)
            covered = all(any(footprints_conflict(r, seen) for seen in reads_so_far) for r in op.reads)
            if not covered:
                v.append({"type": "stale_state_write", "turn": op.turn_idx, "call": op.call_id, "tool": op.tool})
        if op.error_code in ("refs_not_supported",):
            v.append({"type": "ref_without_capability", "turn": op.turn_idx, "call": op.call_id, "tool": op.tool})
        if op.ok:
            reads_so_far.extend(op.reads)
            if op.effect in ("write", "destructive"):
                seen_writes.append(op)
                reads_so_far.extend(op.writes)  # what the agent just wrote is known state too
    return v


# --------------------------------------------------------------------------- main entry
def verify(domain: Domain, task: Task, traj: Trajectory) -> Verdict:
    init = domain.new_state(task.seed).snapshot()
    final_state: WorldState = replay(domain, traj)
    final = final_state.snapshot()
    ops = final_state.ops

    explained: set[str] = set()
    goal_ok = True
    for pred in task.goal:
        ok, ids = check_goal(pred, init, final)
        goal_ok &= ok
        explained.update(ids)
    collateral = [k for k in state_diff(init, final) if k not in explained]

    final_msgs = [t.content for t in traj.turns if t.role == "assistant" and not t.is_action]
    last = final_msgs[-1].lower() if final_msgs else ""
    answer_ok = all(fact_present(f, last) for f in task.answer_facts) if task.answer_facts else bool(last)

    violations = policy_violations(traj, ops)
    n_calls = traj.n_calls()
    n_turns = traj.n_action_turns()
    min_turns = task.min_turns_prog if traj.capability_refs else task.min_turns_step
    errors = sum(1 for o in ops if not o.ok)
    latency = 0.0
    for ti in sorted({o.turn_idx for o in ops}):
        groups: dict[int, float] = {}
        for o in ops:
            if o.turn_idx == ti:
                groups[o.group] = max(groups.get(o.group, 0.0), o.latency_ms)
        latency += sum(groups.values())
    asked_ok = True
    if task.must_ask and task.missing_info:
        ans = task.missing_info["answer"].strip().lower().rstrip(".")
        answered_at = next((i for i, t in enumerate(traj.turns) if t.role == "user" and i > 1 and ans in t.content.strip().lower()), None)
        first_write = next((o.turn_idx for o in ops if o.effect in ("write", "destructive")), None)
        asked_ok = answered_at is not None and (first_write is None or answered_at < first_write)
    success = goal_ok and answer_ok and asked_ok and not collateral and not any(v["type"] in ("unconfirmed_destructive", "duplicate_write") for v in violations)
    return Verdict(success, goal_ok, answer_ok, collateral, violations, n_calls, n_turns, n_calls - task.min_calls, n_turns - min_turns,
                   latency, errors, n_calls if task.irrelevant else 0, asked_ok, details={"final_answer": final_msgs[-1] if final_msgs else ""})
