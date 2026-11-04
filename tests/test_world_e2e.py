"""End-to-end: generate tasks, run optimal/naive plans in both modes, verify."""
import collections

import pytest

from etool.world.domains.workspace import Workspace
from etool.world.domains.workspace_tasks import GENERATORS, generate
from etool.world.rollout import run_plan
from etool.world.verifier import verify

DOM = Workspace()


def _all(seeds):
    out = []
    for s in seeds:
        g = generate(DOM, s)
        if g is not None:
            out.append(g)
    return out


@pytest.mark.parametrize("refs", [True, False])
def test_optimal_plans_succeed_without_violations(refs):
    kinds = collections.Counter()
    for task, opt, _ in _all(range(0, 72)):
        traj, _ = run_plan(DOM, task, opt, refs=refs, source="optimal")
        v = verify(DOM, task, traj)
        assert v.success, (task.kind, task.utterance, v)
        assert v.violations == [], (task.kind, v.violations)
        assert v.excess_calls == 0 and v.excess_turns == 0, (task.kind, v.n_calls, task.min_calls, v.n_action_turns)
        assert v.errors == 0
        kinds[task.kind] += 1
    assert set(kinds) == set(GENERATORS), kinds


@pytest.mark.parametrize("refs", [True, False])
def test_naive_plans_succeed_but_are_inefficient(refs):
    worse = 0
    for task, opt, nai in _all(range(0, 72)):
        traj, _ = run_plan(DOM, task, nai, refs=refs, source="naive")
        v = verify(DOM, task, traj)
        assert v.success, (task.kind, v)
        assert v.excess_calls >= 0 and v.excess_turns >= 0
        if not task.irrelevant:
            assert v.excess_calls > 0 or v.excess_turns > 0, task.kind
            worse += 1
    assert worse > 40


def test_verifier_catches_bad_behaviour():
    from etool.schema import ToolCall, Turn
    task, opt, _ = next(g for g in _all(range(0, 72)) if g[0].kind == "cancel_event")
    traj, _ = run_plan(DOM, task, opt, refs=False, source="optimal")
    # strip the confirmation exchange -> unconfirmed destructive
    turns = [t for t in traj.turns if not (t.role == "user" and t.content.startswith("Yes")) and not (t.role == "assistant" and "Shall I" in t.content)]
    traj.turns = turns
    v = verify(DOM, task, traj)
    assert not v.success and any(x["type"] == "unconfirmed_destructive" for x in v.violations)
    # delete a second, unrelated event -> collateral change
    task2, opt2, _ = next(g for g in _all(range(0, 72)) if g[0].kind == "whats_on")
    traj2, _ = run_plan(DOM, task2, opt2, refs=False, source="optimal")
    some_event = next(iter(DOM.new_state(task2.seed).data["events"]))
    traj2.turns.insert(2, Turn("assistant", "", [ToolCall("delete_event", {"event_id": some_event})]))
    v2 = verify(DOM, task2, traj2)
    assert v2.collateral == [f"event:{some_event}"] and not v2.success
    # refs while the capability is off -> ref_without_capability
    task3, opt3, _ = next(g for g in _all(range(0, 72)) if g[0].kind == "whats_on")
    traj3, _ = run_plan(DOM, task3, opt3, refs=True, source="optimal")
    traj3.capability_refs = False
    v3 = verify(DOM, task3, traj3)
    assert any(x["type"] == "ref_without_capability" for x in v3.violations)


def test_scripted_user():
    from etool.world.user import ScriptedUser
    task, _, _ = next(g for g in _all(range(0, 72)) if g[0].kind == "cancel_event")
    u = ScriptedUser(task)
    assert u.respond("I'm about to delete X. Shall I proceed?") == "Yes, please go ahead."
    assert u.respond("Done, deleted it.") is None
    task_m = next(g[0] for g in _all(range(0, 200)) if g[0].missing_info)
    u2 = ScriptedUser(task_m)
    assert u2.respond("What time should the meeting start?") == task_m.missing_info["answer"]
