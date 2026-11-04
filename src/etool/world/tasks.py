"""Task specification shared by all domains."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class Task:
    id: str
    domain: str
    kind: str
    seed: int  # state seed
    utterance: str
    slots: dict[str, Any] = field(default_factory=dict)
    goal: list[dict[str, Any]] = field(default_factory=list)  # predicates for the verifier
    answer_facts: list[str] = field(default_factory=list)  # strings the final answer must contain
    requires_confirmation: bool = False
    missing_info: Optional[dict[str, Any]] = None  # {"question_hint": str, "answer": str}
    irrelevant: bool = False  # no tool call is appropriate
    must_ask: bool = False  # the request is ambiguous: the agent must ask (missing_info) before its first write; guessing right still fails
    on_question: Optional[str] = None  # scripted reply to the agent's first clarifying question when no confirmation/missing info is expected
    tools_excluded: list[str] = field(default_factory=list)  # tools hidden from the prompt and uncallable for this task (missing_capability kind)
    min_calls: int = 0  # optimal number of tool calls
    min_turns_prog: int = 0  # optimal number of action turns in program mode
    min_turns_step: int = 0  # ... in stepwise mode

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Task":
        return Task(**d)
