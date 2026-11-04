"""Shared trajectory schema.

Used by the mock world, public-data loaders, optimizer, negative generators, trainer and evals.
Everything is a plain dataclass serialisable to JSONL via ``to_dict``/``from_dict``.

Conventions
-----------
* A *trajectory* is a list of turns: system / user / assistant / tool.
* An assistant turn may carry ``tool_calls``; each call has a 1-based ``id`` ("1", "2", …) that is
  its position inside the turn. Later calls in the *same* turn may reference earlier results inside
  argument values with ``"$1"`` or ``"$1.items[0].id"`` (see ``etool.refs``). This is only legal when
  the trajectory's system prompt advertises the capability (``capability_refs=True``).
* A tool turn answers exactly one call (``tool_call_id``) and carries the serialised result.
* Effects / footprints on ``ToolSpec`` are known for mock-world tools and ``None`` (unknown) for
  public datasets.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Iterator, Literal, Optional

Effect = Literal["pure", "read", "write", "destructive"]
EFFECTS: tuple[str, ...] = ("pure", "read", "write", "destructive")
Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON-schema object (``{"type": "object", "properties": ..., "required": [...]}``)
    effect: Optional[Effect] = None  # None = unknown (public tools)
    idempotent: Optional[bool] = None
    reads: list[str] = field(default_factory=list)  # footprint patterns, e.g. "calendar:{calendar_id}"
    writes: list[str] = field(default_factory=list)
    latency_ms: float = 50.0  # simulated latency (mock world)
    batch_of: Optional[str] = None  # if this tool is the batch variant of another tool, its name
    returns: Optional[dict[str, Any]] = None  # optional JSON schema of the result

    def to_openai(self) -> dict[str, Any]:
        """OpenAI / HF-chat-template function schema (what the model sees)."""
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }

    EFFECT_HINT_MARK = " [effect: "

    def with_effect_hint(self) -> "ToolSpec":
        """Copy whose description ends with an MCP-style annotation of the declared effect, e.g. ``[effect: destructive; not
        idempotent]`` (v3b data, 2026-10-22). Public tools (effect None) and already-annotated specs are returned unchanged, so
        applying it twice (data build + eval) is safe."""
        if self.effect is None or self.EFFECT_HINT_MARK in (self.description or ""):
            return self
        idem = "" if self.idempotent is None else ("; idempotent" if self.idempotent else "; not idempotent")
        return replace(self, description=f"{(self.description or '').rstrip()}{self.EFFECT_HINT_MARK}{self.effect}{idem}]")

    @property
    def required(self) -> list[str]:
        return list(self.parameters.get("required", []))

    @property
    def is_effectful(self) -> bool:
        return self.effect in ("write", "destructive")


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    id: Optional[str] = None  # "1", "2", … position inside the assistant turn

    def to_openai(self) -> dict[str, Any]:
        d: dict[str, Any] = {"type": "function", "function": {"name": self.name, "arguments": self.arguments}}
        if self.id is not None:
            d["id"] = self.id
        return d


@dataclass
class Turn:
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)  # assistant only
    tool_call_id: Optional[str] = None  # tool role: which call this result answers
    name: Optional[str] = None  # tool role: tool name
    meta: dict[str, Any] = field(default_factory=dict)  # latency_ms, error, origin, …

    @property
    def is_action(self) -> bool:
        return self.role == "assistant" and bool(self.tool_calls)

    def to_message(self) -> dict[str, Any]:
        """HF chat-template message dict (Qwen3 / hermes style)."""
        m: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.role == "assistant" and self.tool_calls:
            m["tool_calls"] = [c.to_openai() for c in self.tool_calls]
        if self.role == "tool":
            if self.name is not None:
                m["name"] = self.name
            if self.tool_call_id is not None:
                m["tool_call_id"] = self.tool_call_id
        return m


@dataclass
class Trajectory:
    id: str
    source: str  # "world:workspace", "world:commerce", "nemotron", "toolace", "hermes", "teacher", …
    tools: list[ToolSpec]
    turns: list[Turn]
    capability_refs: bool = False  # system prompt advertises $ref programs
    task: Optional[dict[str, Any]] = None  # mock-world task spec (for the verifier)
    gold_final_state: Optional[dict[str, Any]] = None
    meta: dict[str, Any] = field(default_factory=dict)

    # -- convenience -------------------------------------------------------
    def to_messages(self) -> list[dict[str, Any]]:
        return [t.to_message() for t in self.turns]

    def tools_openai(self) -> list[dict[str, Any]]:
        return [t.to_openai() for t in self.tools]

    def tool_by_name(self) -> dict[str, ToolSpec]:
        return {t.name: t for t in self.tools}

    def assistant_indices(self) -> list[int]:
        return [i for i, t in enumerate(self.turns) if t.role == "assistant"]

    def calls(self) -> list[tuple[int, ToolCall]]:
        """All tool calls as (turn_idx, call) in order."""
        return [(i, c) for i, t in enumerate(self.turns) if t.role == "assistant" for c in t.tool_calls]

    def n_calls(self) -> int:
        return sum(len(t.tool_calls) for t in self.turns if t.role == "assistant")

    def n_action_turns(self) -> int:
        return sum(1 for t in self.turns if t.is_action)

    def prefix(self, turn_idx: int) -> "Trajectory":
        """Copy of this trajectory truncated to turns[:turn_idx] (the context before turn ``turn_idx``)."""
        return Trajectory(
            id=self.id, source=self.source, tools=self.tools, turns=self.turns[:turn_idx],
            capability_refs=self.capability_refs, task=self.task, gold_final_state=self.gold_final_state,
            meta=dict(self.meta),
        )


Origin = Literal["rule", "optimizer", "teacher", "mined", "planner"]


@dataclass
class Negative:
    """A bad assistant action at a given prefix, optionally paired with the good action at the same prefix."""

    traj_id: str
    turn_idx: int  # index into Trajectory.turns of the assistant turn being replaced
    bad_turn: Turn
    good_turn: Optional[Turn]
    error_class: str  # see etool.negatives.classes
    origin: Origin
    meta: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- JSON (de)serialisation
_DATACLASSES = {c.__name__: c for c in (ToolSpec, ToolCall, Turn, Trajectory, Negative)}


def to_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        d = asdict(obj)
        d["__type__"] = type(obj).__name__
        return d
    return obj


def _from_dict_typed(cls: type, d: dict[str, Any]) -> Any:
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in d:
            continue
        v = d[f.name]
        if cls is Trajectory and f.name == "tools":
            v = [from_dict(x, ToolSpec) for x in v]
        elif cls is Trajectory and f.name == "turns":
            v = [from_dict(x, Turn) for x in v]
        elif cls is Turn and f.name == "tool_calls":
            v = [from_dict(x, ToolCall) for x in v]
        elif cls is Negative and f.name in ("bad_turn", "good_turn") and v is not None:
            v = from_dict(v, Turn)
        kwargs[f.name] = v
    return cls(**kwargs)


def from_dict(d: dict[str, Any], cls: Optional[type] = None) -> Any:
    if cls is None:
        cls = _DATACLASSES[d["__type__"]]
    return _from_dict_typed(cls, d)


def dumps(obj: Any) -> str:
    return json.dumps(to_dict(obj), ensure_ascii=False, separators=(",", ":"))


def loads(s: str, cls: Optional[type] = None) -> Any:
    return from_dict(json.loads(s), cls)


def write_jsonl(path: str | Path, objs: Iterator[Any] | list[Any]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for o in objs:
            f.write(dumps(o) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path, cls: Optional[type] = None) -> Iterator[Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield loads(line, cls)
