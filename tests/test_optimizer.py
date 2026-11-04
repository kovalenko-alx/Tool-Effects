import collections

import pytest

from etool.optimize import optimize
from etool.world.domains.workspace import Workspace
from etool.world.domains.workspace_tasks import generate
from etool.world.rollout import run_plan
from etool.world.verifier import verify

DOM = Workspace()


def _tasks(n=72):
    return [g for g in (generate(DOM, s) for s in range(n)) if g is not None]


@pytest.mark.parametrize("refs", [True, False])
def test_optimizer_recovers_optimal_from_naive(refs):
    classes = collections.Counter()
    for task, opt_plan, naive_plan in _tasks():
        naive, _ = run_plan(DOM, task, naive_plan, refs=refs, source="naive")
        r = optimize(DOM, naive, conservative=False)
        assert r.accepted, (task.kind, r.reason)
        v = verify(DOM, task, r.optimized)
        assert v.success and v.violations == [], (task.kind, v)
        assert r.optimized.n_calls() == task.min_calls, (task.kind, refs, r.optimized.n_calls(), task.min_calls, r.classes)
        if refs:
            assert r.optimized.n_action_turns() == task.min_turns_prog, (task.kind, r.optimized.n_action_turns(), task.min_turns_prog)
        if not task.irrelevant:
            assert r.changes, task.kind
        classes.update(r.classes)
    assert {"redundant_read", "dead_call", "unbatched"} <= set(classes), classes
    if refs:
        assert "serialized_calls" in classes


@pytest.mark.parametrize("refs", [True, False])
def test_optimizer_is_identity_on_optimal(refs):
    for task, opt_plan, _ in _tasks():
        gold, _ = run_plan(DOM, task, opt_plan, refs=refs, source="optimal")
        r = optimize(DOM, gold)
        assert r.accepted
        assert r.optimized.n_calls() == gold.n_calls() and r.optimized.n_action_turns() == gold.n_action_turns(), (task.kind, r.classes)


def test_fold_pass_removes_iso_lookup():
    task, opt_plan, naive_plan = next(g for g in _tasks(300) if g[0].kind == "whats_on" and g[0].slots["expr"] == g[0].slots["date"])
    assert task.min_calls == 1  # the user gave an ISO date: no resolve_date needed
    naive, _ = run_plan(DOM, task, naive_plan, refs=False, source="naive")
    r = optimize(DOM, naive, conservative=False)
    assert r.accepted and "lookup_of_known_value" in r.classes and r.optimized.n_calls() == 1, r.classes


def test_conservative_keeps_decision_driving_reads():
    """A read whose result only drives an in-head decision (filtering) is kept in conservative mode and dropped otherwise."""
    from etool.schema import ToolCall, Trajectory, Turn
    from etool.world.executor import execute_turn
    task, _, _ = next(g for g in _tasks(100) if g[0].kind == "unread_from")
    state = DOM.new_state(task.seed)
    turns = [Turn("system", DOM.system_prompt(False)), Turn("user", task.utterance)]
    for call in (ToolCall("search_contacts", {"query": task.slots["contact"].split()[0]}), ToolCall("list_emails", {"folder": "inbox", "unread_only": True})):
        t = Turn("assistant", "", [call]); turns.append(t)
        turns.extend(execute_turn(DOM, state, t, turn_idx=len(turns) - 1, capability_refs=False).tool_turns)
    import json as _json
    email = _json.loads(turns[3].content)["contacts"][0]["email"]
    subjects = [m["subject"] for m in _json.loads(turns[5].content)["emails"] if m["from"] == email]
    turns.append(Turn("assistant", "Unread from them: " + "; ".join(subjects) if subjects else "Nothing unread from them."))
    traj = Trajectory("x", "test", DOM.specs(), turns, False, task.to_dict(), meta={"seed": task.seed})
    assert optimize(DOM, traj).optimized.n_calls() == 2
    assert optimize(DOM, traj, conservative=False).optimized.n_calls() == 1
