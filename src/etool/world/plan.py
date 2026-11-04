"""Symbolic plans for scripted agents (gold / naive trajectories).

A ``Plan`` is a list of ``Stage``s. Each stage may first *ask* the user something (missing info) or request
*confirmation* (before a destructive stage), then executes its ``Step``s. Step arguments may contain ``R(key, path)``
placeholders referring to the result of an earlier step. When several steps are emitted in one assistant turn and
the trajectory runs in program mode, an ``R`` to a step of the *same turn* becomes a ``$n`` reference; otherwise it is
materialised to the literal value obtained when that step was executed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from etool.refs import follow, parse_ref


@dataclass(frozen=True)
class R:
    key: str
    path: str = ""  # e.g. ".events[0].id", ".tasks[*].id"

    def literal(self, results: dict[str, Any]) -> Any:
        ref = parse_ref("$1" + self.path)
        if ref is None:
            raise ValueError(f"bad path {self.path!r}")
        return follow(results[self.key], ref.path, f"{self.key}{self.path}")


@dataclass
class Step:
    key: str
    tool: str
    args: dict[str, Any]
    needed: bool = True  # False = result is never used (naive dead call); informational only


@dataclass
class Stage:
    steps: list[Step]
    confirm: Optional[str] = None  # assistant asks this, user says yes, then the stage runs
    ask: Optional[tuple[str, str]] = None  # (assistant question, user answer) before the stage
    one_per_turn: bool = False  # force one call per turn even in program mode (naive serialisation)


@dataclass
class Plan:
    stages: list[Stage]
    answer: Callable[[dict[str, Any]], str]  # results-by-key -> final assistant message

    def steps(self) -> list[Step]:
        return [s for st in self.stages for s in st.steps]

    def n_calls(self) -> int:
        return len(self.steps())


def walk(value: Any, fn: Callable[[R], Any]) -> Any:
    if isinstance(value, R):
        return fn(value)
    if isinstance(value, dict):
        return {k: walk(v, fn) for k, v in value.items()}
    if isinstance(value, list):
        return [walk(v, fn) for v in value]
    return value


def materialize(args: dict[str, Any], same_turn: dict[str, int], results: dict[str, Any], refs: bool) -> dict[str, Any]:
    """Replace R placeholders: ``$n`` strings for same-turn steps in program mode, literals otherwise."""

    def sub(r: R) -> Any:
        if refs and r.key in same_turn:
            return f"${same_turn[r.key]}{r.path}"
        return r.literal(results)

    return walk(args, sub)


def turn_groups(stage: Stage, refs: bool) -> list[list[Step]]:
    if refs and not stage.one_per_turn:
        return [stage.steps] if stage.steps else []
    return [[s] for s in stage.steps]
