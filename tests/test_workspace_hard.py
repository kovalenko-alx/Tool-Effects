"""Hard (test-only) kinds: gold plans verify, and the behaviours they are meant to catch fail."""
import collections

import pytest

from etool.schema import ToolCall, Turn
from etool.world.domains.workspace_hard import ALL_KINDS, HARD, WorkspaceHard, generate_hard
from etool.world.rollout import run_plan
from etool.world.tasks import Task
from etool.world.user import ScriptedUser
from etool.world.verifier import verify

DOM = WorkspaceHard()


def _all(seeds, kinds=None):
    out = []
    for s in seeds:
        g = generate_hard(DOM, s, kinds)
        if g is not None:
            out.append(g)
    return out


def _first(kind, seeds=range(3_000_000, 3_000_400)):
    return next(g for g in _all(seeds, list(HARD)) if g[0].kind == kind)


def test_domain_has_distractors_and_same_core_tools():
    assert len(DOM.tools) == 24 and {"send_sms", "create_invoice", "get_weather"} <= set(DOM.tools)
    st = DOM.new_state(5)
    assert st.data["messages"] == {} and st.data["invoices"] == {}
    assert Task.from_dict({"id": "x", "domain": "workspace", "kind": "k", "seed": 0, "utterance": "u"}).must_ask is False  # old task files still load


@pytest.mark.parametrize("refs", [True, False])
def test_gold_plans_of_all_kinds_succeed(refs):
    kinds = collections.Counter()
    for task, opt, _ in _all(range(3_000_000, 3_000_160)):
        traj, _ = run_plan(DOM, task, opt, refs=refs, source="optimal")
        v = verify(DOM, task, traj)
        assert v.success, (task.kind, task.utterance, v)
        assert v.violations == [] and v.excess_calls == 0 and v.excess_turns == 0 and v.errors == 0, (task.kind, v)
        assert task.domain == "workspace_hard" and len(traj.tools) == 24
        kinds[task.kind] += 1
    assert set(kinds) == set(ALL_KINDS), kinds


def test_guessing_the_contact_fails_even_when_right():
    task, opt, _ = _first("ambiguous_contact_meeting")
    traj, _ = run_plan(DOM, task, opt, refs=False, source="optimal")
    assert verify(DOM, task, traj).asked_ok
    # drop the question/answer exchange -> same final state, but no clarification before the write
    traj.turns = [t for t in traj.turns if not (t.role == "assistant" and "Which one" in t.content) and not (t.role == "user" and t.content == task.missing_info["answer"])]
    v = verify(DOM, task, traj)
    assert v.goal_ok and not v.asked_ok and not v.success
    assert "asked_ok" in v.as_row()


def test_missing_entities_must_not_be_acted_on():
    task, opt, _ = _first("missing_event")
    traj, _ = run_plan(DOM, task, opt, refs=False, source="optimal")
    assert verify(DOM, task, traj).success
    # deleting some other event instead -> collateral
    some_event = next(iter(DOM.new_state(task.seed).data["events"]))
    bad = [Turn("assistant", "", [ToolCall("delete_event", {"event_id": some_event})])]
    traj.turns = traj.turns[:-1] + bad + [Turn("assistant", f"Deleted '{task.slots['title']}'.")]
    v = verify(DOM, task, traj)
    assert not v.success and v.collateral == [f"event:{some_event}"] and not v.answer_ok
    # claiming success without a negation -> answer fails
    traj2, _ = run_plan(DOM, task, opt, refs=False, source="optimal")
    traj2.turns[-1] = Turn("assistant", "Done, I moved it.")
    assert not verify(DOM, task, traj2).answer_ok
    # a sent email on a missing_sender task is collateral
    task3, opt3, _ = _first("missing_sender")
    traj3, _ = run_plan(DOM, task3, opt3, refs=False, source="optimal")
    traj3.turns.insert(len(traj3.turns) - 1, Turn("assistant", "", [ToolCall("send_email", {"to": ["x@example.com"], "subject": "Re", "body": "ok"})]))
    v3 = verify(DOM, task3, traj3)
    assert v3.collateral and not v3.success


def test_dated_cancel_wrong_twin_is_collateral():
    task, opt, _ = _first("cancel_event_dated")
    st = DOM.new_state(task.seed)
    twin = next(e for e in st.data["events"].values() if e["title"] == task.slots["title"] and e["id"] != task.goal[0]["id"])
    traj, _ = run_plan(DOM, task, opt, refs=True, source="optimal")
    assert verify(DOM, task, traj).success
    for t in traj.turns:
        for c in t.tool_calls:
            if c.name == "delete_event":
                c.arguments = {"event_id": twin["id"]}
    v = verify(DOM, task, traj)
    assert not v.goal_ok and v.collateral == [f"event:{twin['id']}"] and not v.success


def test_noconfirm_request_still_needs_confirmation():
    task, opt, _ = _first("cancel_event_noconfirm")
    traj, _ = run_plan(DOM, task, opt, refs=False, source="optimal")
    assert verify(DOM, task, traj).success
    traj.turns = [t for t in traj.turns if not (t.role == "user" and t.content.startswith("Yes")) and not (t.role == "assistant" and "Shall I" in t.content)]
    v = verify(DOM, task, traj)
    assert not v.success and any(x["type"] == "unconfirmed_destructive" for x in v.violations)


def test_distractor_calls_are_counted():
    task, plan, _ = _first("irrelevant_near")
    traj, _ = run_plan(DOM, task, plan, refs=False, source="optimal")
    assert verify(DOM, task, traj).success and verify(DOM, task, traj).n_calls == 0
    traj.turns.insert(2, Turn("assistant", "", [ToolCall("search_web", {"query": "meeting tips"})]))
    v = verify(DOM, task, traj)
    assert v.unnecessary_calls == 1 and v.excess_calls == 1 and v.errors == 0
    traj.turns.insert(2, Turn("assistant", "", [ToolCall("send_sms", {"phone": "+420123456789", "text": "hi"})]))
    v2 = verify(DOM, task, traj)
    assert v2.collateral == ["message:s1"] and not v2.success


def test_scripted_user_clarification_and_ambiguity():
    task, _, _ = _first("missing_event")
    u = ScriptedUser(task)
    assert u.respond("Did you mean the Roadmap meeting? Shall I delete that one?") == task.on_question  # restates instead of a blanket yes
    assert u.respond("I'm about to delete X. Shall I proceed?") == "Yes, please go ahead."  # confirmation requests after that behave as before
    task2, _, _ = _first("ambiguous_contact_unread")
    u2 = ScriptedUser(task2)
    assert u2.respond("Which one do you mean: A or B?") == task2.missing_info["answer"]
    assert u2.respond("You have no unread emails from them.") is None
