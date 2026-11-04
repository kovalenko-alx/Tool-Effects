"""Run a scripted plan against a domain to produce a full ``Trajectory`` (gold or naive)."""
from __future__ import annotations

import json
from typing import Any, Optional

from etool.schema import ToolCall, Trajectory, Turn
from etool.world.base import Domain, WorldState
from etool.world.executor import execute_turn
from etool.world.plan import Plan, materialize, turn_groups
from etool.world.tasks import Task

YES = "Yes, please go ahead."


def run_plan(domain: Domain, task: Task, plan: Plan, *, refs: bool, source: str) -> tuple[Trajectory, WorldState]:
    state = domain.new_state(task.seed)
    excluded = frozenset(task.tools_excluded)
    turns: list[Turn] = [Turn("system", domain.system_prompt(refs)), Turn("user", task.utterance)]
    results: dict[str, Any] = {}
    for stage in plan.stages:
        if stage.ask is not None:
            q, a = stage.ask
            turns.append(Turn("assistant", q))
            turns.append(Turn("user", a))
        if stage.confirm is not None:
            turns.append(Turn("assistant", stage.confirm))
            turns.append(Turn("user", YES))
        for group in turn_groups(stage, refs):
            same_turn = {s.key: i + 1 for i, s in enumerate(group)}
            calls = [ToolCall(s.tool, materialize(s.args, same_turn, results, refs), str(i + 1)) for i, s in enumerate(group)]
            turn = Turn("assistant", "", calls)
            turn_idx = len(turns)
            turns.append(turn)
            res = execute_turn(domain, state, turn, turn_idx=turn_idx, capability_refs=refs, excluded=excluded)
            for s, tt in zip(group, res.tool_turns):
                obs = json.loads(tt.content)
                if tt.meta.get("error"):
                    raise RuntimeError(f"scripted plan failed at step {s.key}: {tt.content}")
                results[s.key] = obs
            turns.extend(res.tool_turns)
    turns.append(Turn("assistant", plan.answer(results)))
    traj = Trajectory(
        id=f"{task.id}:{source}:{'prog' if refs else 'step'}", source=f"world:{domain.name}:{source}",
        tools=[t for t in domain.specs() if t.name not in excluded],
        turns=turns, capability_refs=refs, task=task.to_dict(), meta={"seed": task.seed, "kind": task.kind, "policy": source},
    )
    return traj, state


def replay(domain: Domain, traj: Trajectory, *, stop_at: Optional[int] = None) -> WorldState:
    """Re-execute every assistant tool-call turn of ``traj`` on a fresh state; returns the state (with ops)."""
    seed = int(traj.meta.get("seed", traj.task["seed"] if traj.task else 0))
    excluded = frozenset((traj.task or {}).get("tools_excluded") or [])
    state = domain.new_state(seed)
    for i, t in enumerate(traj.turns):
        if stop_at is not None and i >= stop_at:
            break
        if t.is_action:
            execute_turn(domain, state, t, turn_idx=i, capability_refs=traj.capability_refs, excluded=excluded)
    return state
