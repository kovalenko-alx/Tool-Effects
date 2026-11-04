"""Mock tool world: runtime tool definitions, effects, footprints, state and argument validation.

A *domain* bundles typed tools (``ToolSpec`` + Python implementation), a seeded state generator and the
natural-language policies that go into the system prompt. Every executed call is recorded as an ``Op`` with
its resolved footprints, which the verifier (policy checks) and the optimizer (legality) consume.
"""
from __future__ import annotations

import copy
import fnmatch
import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from etool.schema import ToolSpec

REF_CAPABILITY_TEXT = (
    "You may issue several tool calls in one turn. A later call may use the result of an earlier call in the "
    "same turn by writing \"$N\" (the whole result of call N, counted from 1 within the turn) or a path such as "
    "\"$N.field[0].subfield\" as an argument value. Calls that do not depend on each other run in parallel."
)
NO_REF_TEXT = "Call tools one step at a time: wait for a tool result before using it in another call."


class ToolError(Exception):
    """Raised by tool implementations; becomes a structured error observation."""

    def __init__(self, code: str, message: str, **extra: Any):
        super().__init__(f"{code}: {message}")
        self.code, self.message, self.extra = code, message, extra

    def observation(self) -> dict[str, Any]:
        return {"error": {"code": self.code, "message": self.message, **self.extra}}


@dataclass
class Op:
    """One executed (or attempted) tool call, for verifier / optimizer / metrics."""

    turn_idx: int
    call_id: str
    tool: str
    effect: Optional[str]
    idempotent: Optional[bool]
    reads: list[str]
    writes: list[str]
    ok: bool
    error_code: Optional[str]
    latency_ms: float
    args: dict[str, Any]
    group: int = 0  # parallel group index within the turn


class WorldState:
    """JSON-like state plus the op log. ``data`` is domain-specific."""

    def __init__(self, data: dict[str, Any], seed: int = 0):
        self.data = data
        self.ops: list[Op] = []
        self.rng = random.Random(seed)

    def snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(self.data)

    def clone(self) -> "WorldState":
        c = WorldState(copy.deepcopy(self.data), 0)
        c.ops = list(self.ops)
        c.rng.setstate(self.rng.getstate())
        return c

    def writes_since(self, op_index: int) -> list[str]:
        """Resource keys written by successful effectful ops with index >= op_index."""
        out: list[str] = []
        for op in self.ops[op_index:]:
            if op.ok:
                out.extend(op.writes)
        return out


ToolFn = Callable[[WorldState, dict[str, Any]], Any]


class _Star(dict):
    def __missing__(self, key: str) -> str:  # unresolved placeholder -> wildcard
        return "*"


def format_footprint(pattern: str, args: dict[str, Any]) -> str:
    return pattern.format_map(_Star({k: v for k, v in args.items() if isinstance(v, (str, int, float, bool))}))


def footprints_conflict(a: str, b: str) -> bool:
    """Two resource keys conflict if either matches the other as a glob (``*`` wildcards)."""
    return fnmatch.fnmatchcase(a, b) or fnmatch.fnmatchcase(b, a)


@dataclass
class ToolDef:
    spec: ToolSpec
    fn: ToolFn

    def footprints(self, args: dict[str, Any]) -> tuple[list[str], list[str]]:
        return ([format_footprint(p, args) for p in self.spec.reads], [format_footprint(p, args) for p in self.spec.writes])


# --------------------------------------------------------------------------- argument validation
_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,), "integer": (int,), "number": (int, float), "boolean": (bool,), "array": (list,), "object": (dict,),
}


def validate_args(spec: ToolSpec, args: dict[str, Any]) -> list[str]:
    """Minimal JSON-schema check: required, unknown, type, enum. Returns a list of problems (empty = ok)."""
    problems: list[str] = []
    props: dict[str, Any] = spec.parameters.get("properties", {}) or {}
    for r in spec.required:
        if r not in args:
            problems.append(f"missing required argument '{r}'")
    for k, v in args.items():
        if k not in props:
            problems.append(f"unknown argument '{k}'")
            continue
        p = props[k]
        t = p.get("type")
        if t in _JSON_TYPES and v is not None:
            ok_types = _JSON_TYPES[t]
            if isinstance(v, bool) and t in ("integer", "number"):
                problems.append(f"argument '{k}' must be {t}, got boolean")
            elif not isinstance(v, ok_types):
                problems.append(f"argument '{k}' must be {t}, got {type(v).__name__}")
        if "enum" in p and v not in p["enum"]:
            problems.append(f"argument '{k}' must be one of {p['enum']}")
        if t == "array" and isinstance(v, list) and "items" in p and p["items"].get("type") in _JSON_TYPES:
            it = _JSON_TYPES[p["items"]["type"]]
            if any(not isinstance(x, it) for x in v):
                problems.append(f"argument '{k}' items must be {p['items']['type']}")
    return problems


# --------------------------------------------------------------------------- domain
class Domain:
    name: str = "base"
    description: str = ""
    policies: list[str] = []

    def __init__(self) -> None:
        self.tools: dict[str, ToolDef] = {}

    def add(self, spec: ToolSpec, fn: ToolFn) -> None:
        if spec.name in self.tools:
            raise ValueError(f"duplicate tool {spec.name}")
        self.tools[spec.name] = ToolDef(spec, fn)

    def specs(self) -> list[ToolSpec]:
        return [t.spec for t in self.tools.values()]

    def new_state(self, seed: int) -> WorldState:  # pragma: no cover - abstract
        raise NotImplementedError

    _ID_RE = re.compile(r"^[a-z]{1,3}\d{1,6}$")

    def is_identifier(self, value: str) -> bool:
        """Does this (normalised) scalar name an entity: ids like 'e3', 'c12', email addresses, uuids."""
        return bool(self._ID_RE.match(value)) or "@" in value or (len(value) >= 32 and "-" in value)

    def fold(self, tool: str, args: dict[str, Any]) -> Optional[Any]:
        """Constant folding hook: the result of a *pure* call when it is trivially derivable from literal args, else None."""
        return None

    def system_prompt(self, capability_refs: bool) -> str:
        lines = [f"You are an assistant for {self.description or self.name}. Use the provided tools to help the user."]
        if self.policies:
            lines.append("Policies:")
            lines += [f"- {p}" for p in self.policies]
        lines.append(REF_CAPABILITY_TEXT if capability_refs else NO_REF_TEXT)
        return "\n".join(lines)


DEFAULT_POLICIES = [
    "Before any irreversible action (deleting, sending, paying, cancelling), state exactly what you are about to do and wait for the user's explicit confirmation.",
    "Never repeat an action that creates, sends or pays unless the user asks again.",
    "Read the current data before modifying it; do not act on assumptions.",
    "If information required by a tool is missing, ask the user instead of guessing.",
    "Do not call tools when the request can be answered without them.",
]
