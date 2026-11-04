"""Commerce domain (test-only): gold plans verify in both modes; the behaviours it is meant to catch fail."""
import collections

import pytest

from etool.optimize import optimize
from etool.schema import ToolCall, Turn
from etool.world.domains.commerce import Commerce
from etool.world.domains.commerce_tasks import GENERATORS, generate
from etool.world.executor import execute_turn
from etool.world.rollout import run_plan
from etool.world.user import ScriptedUser
from etool.world.verifier import verify

DOM = Commerce()


def _all(seeds):
    return [g for g in (generate(DOM, s) for s in seeds) if g is not None]


def _first(kind, seeds=range(4_000_000, 4_000_300)):
    return next(g for g in _all(seeds) if g[0].kind == kind)


def test_tools_and_state():
    s1, s2 = DOM.new_state(11), DOM.new_state(11)
    assert s1.data == s2.data and len(DOM.tools) == 17
    assert {t.spec.effect for t in DOM.tools.values()} == {"pure", "read", "write", "destructive"}
    s = DOM.new_state(11)
    s.data["cart"].clear()
    pid = next(p["id"] for p in s.data["products"].values() if p["stock"] > 0)
    r = execute_turn(DOM, s, Turn("assistant", "", [ToolCall("get_product", {"product_id": pid}), ToolCall("list_payment_methods", {}), ToolCall("list_addresses", {}),
                                                   ToolCall("add_to_cart", {"product_id": "$1.id", "qty": 1}),
                                                   ToolCall("place_order", {"payment_method_id": "$2.payment_methods[0].id", "address_id": "$3.addresses[1].id"})]),
                     turn_idx=2, capability_refs=True)
    assert r.ok, r.errors
    assert r.groups == [["1", "2", "3"], ["4"], ["5"]]
    assert s.data["cart"] == {} and s.data["orders"]["o5"]["status"] == "paid" and s.data["orders"]["o5"]["address_id"] == "a2"
    # shipped orders cannot be cancelled; delivered ones can be refunded
    shipped = next((o for o in s.data["orders"].values() if o["status"] == "shipped"), None)
    if shipped:
        r2 = execute_turn(DOM, s, Turn("assistant", "", [ToolCall("cancel_order", {"order_id": shipped["id"]})]), turn_idx=4, capability_refs=False)
        assert r2.errors == {"1": "invalid_state"} and s.data["orders"][shipped["id"]]["status"] == "shipped"


@pytest.mark.parametrize("refs", [True, False])
def test_gold_plans_succeed(refs):
    kinds = collections.Counter()
    for task, opt, _ in _all(range(4_000_000, 4_000_200)):
        traj, _ = run_plan(DOM, task, opt, refs=refs, source="optimal")
        v = verify(DOM, task, traj)
        assert v.success, (task.kind, task.utterance, v)
        assert v.violations == [] and v.excess_calls == 0 and v.excess_turns == 0 and v.errors == 0, (task.kind, v)
        r = optimize(DOM, traj)
        assert r.accepted and r.optimized.n_calls() == task.min_calls, (task.kind, r.classes)
        kinds[task.kind] += 1
    assert set(kinds) == set(GENERATORS), kinds


def test_program_mode_saves_turns():
    task, opt, _ = _first("add_several")
    tp, _ = run_plan(DOM, task, opt, refs=True, source="optimal")
    ts, _ = run_plan(DOM, task, opt, refs=False, source="optimal")
    assert tp.n_action_turns() == 1 and ts.n_action_turns() == 4 and tp.n_calls() == ts.n_calls() == 4
    task2, opt2, _ = _first("buy_now")
    tp2, _ = run_plan(DOM, task2, opt2, refs=True, source="optimal")
    assert tp2.n_action_turns() == 2 and verify(DOM, task2, tp2).success


def test_unconfirmed_pay_and_wrong_order_fail():
    task, opt, _ = _first("buy_now")
    traj, _ = run_plan(DOM, task, opt, refs=False, source="optimal")
    traj.turns = [t for t in traj.turns if not (t.role == "user" and t.content.startswith("Yes")) and not (t.role == "assistant" and "Shall I" in t.content)]
    v = verify(DOM, task, traj)
    assert not v.success and any(x["type"] == "unconfirmed_destructive" for x in v.violations)
    task2, opt2, _ = _first("cancel_order")
    traj2, _ = run_plan(DOM, task2, opt2, refs=False, source="optimal")
    st = DOM.new_state(task2.seed)
    other = next(o["id"] for o in st.data["orders"].values() if o["id"] != task2.goal[0]["id"] and o["status"] in ("pending", "paid"))
    for t in traj2.turns:
        for c in t.tool_calls:
            if c.name == "cancel_order":
                c.arguments = {"order_id": other}
    v2 = verify(DOM, task2, traj2)
    assert not v2.goal_ok and v2.collateral == [f"order:{other}"] and not v2.success


def test_nothing_to_do_kinds():
    task, opt, _ = _first("cancel_shipped")
    traj, _ = run_plan(DOM, task, opt, refs=False, source="optimal")
    assert verify(DOM, task, traj).success
    traj.turns[-1] = Turn("assistant", "Done, your order has been cancelled.")  # claims success without the status word
    assert not verify(DOM, task, traj).answer_ok
    task2, opt2, _ = _first("missing_product")
    traj2, _ = run_plan(DOM, task2, opt2, refs=True, source="optimal")
    assert verify(DOM, task2, traj2).success
    u = ScriptedUser(task2)
    assert u.respond("Did you mean the espresso machine?") == task2.on_question
    task3, _, _ = _first("buy_now")
    u3 = ScriptedUser(task3)
    assert u3.respond("I'll charge your Visa $45 for the skillet. Do you want me to place the order?") == "Yes, please go ahead."
