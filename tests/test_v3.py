"""v3 data levers: the missing_capability kind (decline when the needed tool is absent), effect hints and tool-name dialects."""
import json

from etool.data.dialects import DIALECTS, apply_dialect, apply_dialect_pair, with_effect_hints
from etool.negatives.pairs import PairRecord
from etool.schema import ToolCall, Turn
from etool.world.domains.workspace import Workspace
from etool.world.domains.workspace_hard import ALL_KINDS
from etool.world.domains.workspace_tasks import MISSING_CAPABILITY, generate
from etool.world.executor import execute_turn
from etool.world.rollout import run_plan
from etool.world.verifier import verify


def _missing_tasks(n=12):
    dom = Workspace()
    out = []
    seed = 0
    while len(out) < n and seed < 400:
        g = generate(dom, seed, kinds=["missing_capability"])
        if g is not None:
            out.append(g)
        seed += 1
    return dom, out


def test_missing_capability_declines_and_hides_the_tool():
    dom, gens = _missing_tasks()
    assert len(gens) >= 8
    for task, opt, nai in gens:
        removed = MISSING_CAPABILITY[task.slots["base_kind"]][0]
        assert task.tools_excluded == removed and task.irrelevant and task.min_calls == 0
        for refs in (True, False):
            traj, _ = run_plan(dom, task, opt, refs=refs, source="optimal")
            names = {t.name for t in traj.tools}
            assert not (names & set(removed)) and len(names) == 18 - len(removed)
            v = verify(dom, task, traj)
            assert v.success and v.n_calls == 0, (task.kind, v.details)
            assert "not available" in traj.turns[-1].content


def test_excluded_tool_is_uncallable():
    dom, gens = _missing_tasks(3)
    task = gens[0][0]
    state = dom.new_state(task.seed)
    turn = Turn("assistant", "", [ToolCall(task.tools_excluded[0], {}, "1")])
    res = execute_turn(dom, state, turn, turn_idx=2, capability_refs=False, excluded=frozenset(task.tools_excluded))
    obs = json.loads(res.tool_turns[0].content)
    assert "unknown_tool" in json.dumps(obs)


def test_missing_capability_not_in_hard_split():
    assert "missing_capability" not in ALL_KINDS


def test_effect_hints_are_idempotent_and_describe_effects():
    dom = Workspace()
    spec = dom.tools["delete_event"].spec
    h = spec.with_effect_hint()
    assert h.description.endswith("[effect: destructive; idempotent]") and spec.description == dom.tools["delete_event"].spec.description
    assert h.with_effect_hint().description == h.description
    assert dom.tools["send_email"].spec.with_effect_hint().description.endswith("[effect: destructive; not idempotent]")
    traj, _ = run_plan(dom, *generate(dom, 3)[:2], refs=True, source="optimal")
    assert all("[effect: " in t.description for t in with_effect_hints(traj).tools)


def test_dialects_cover_all_tools_and_rename_consistently():
    dom = Workspace()
    names = set(dom.tools)
    for m in DIALECTS:
        assert set(m) == names and len(set(m.values())) == len(names) and not (set(m.values()) & names)
    task, opt, _ = generate(dom, 5)
    traj, _ = run_plan(dom, task, opt, refs=False, source="optimal")
    d = apply_dialect(traj, DIALECTS[0], 0)
    assert {t.name for t in d.tools} == set(DIALECTS[0].values())
    for t in d.turns:
        for c in t.tool_calls:
            assert c.name in DIALECTS[0].values()
        if t.role == "tool":
            assert t.name in DIALECTS[0].values()
    batch = [t for t in d.tools if t.batch_of]
    assert batch and all(b.batch_of in DIALECTS[0].values() for b in batch)
    assert d.meta["dialect"] == 0 and traj.turns[2].tool_calls[0].name in names  # the original is untouched
    rec = PairRecord("p", traj.id, 2, traj.turns[:2], traj.tools, False, traj.turns[2], Turn("assistant", "", [ToolCall("delete_event", {"event_id": "e1"})]), "wrong_tool", "rule")
    r = apply_dialect_pair(rec, DIALECTS[1], 1)
    assert r.rejected.tool_calls[0].name == "cancel_agenda_item" and r.chosen.tool_calls[0].name in DIALECTS[1].values()
