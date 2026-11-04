"""Assemble self-contained training records from trajectories and negatives.

A pair record holds the prefix (turns before the action), the tools, the capability flag, the good (chosen) and bad
(rejected) assistant turns and the error class. SFT records are whole trajectories; pair records feed suppression and
DPO/ORPO training.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from etool.schema import Negative, ToolSpec, Trajectory, Turn, from_dict, to_dict


@dataclass
class PairRecord:
    id: str
    traj_id: str
    turn_idx: int
    prefix: list[Turn]
    tools: list[ToolSpec]
    capability_refs: bool
    chosen: Turn
    rejected: Turn
    error_class: str
    origin: str
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        d = {"id": self.id, "traj_id": self.traj_id, "turn_idx": self.turn_idx, "prefix": [to_dict(t) for t in self.prefix],
             "tools": [to_dict(t) for t in self.tools], "capability_refs": self.capability_refs, "chosen": to_dict(self.chosen),
             "rejected": to_dict(self.rejected), "error_class": self.error_class, "origin": self.origin, "meta": self.meta}
        return json.dumps(d, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def from_json(s: str) -> "PairRecord":
        d = json.loads(s)
        return PairRecord(d["id"], d["traj_id"], d["turn_idx"], [from_dict(t, Turn) for t in d["prefix"]], [from_dict(t, ToolSpec) for t in d["tools"]],
                          d["capability_refs"], from_dict(d["chosen"], Turn), from_dict(d["rejected"], Turn), d["error_class"], d["origin"], d.get("meta", {}))


def build_pairs(trajs: dict[str, Trajectory], negatives: Iterable[Negative]) -> list[PairRecord]:
    out: list[PairRecord] = []
    for n, neg in enumerate(negatives):
        traj = trajs.get(neg.traj_id)
        if traj is None or neg.good_turn is None:
            continue
        out.append(PairRecord(f"{neg.traj_id}#{neg.turn_idx}#{neg.error_class}#{n}", neg.traj_id, neg.turn_idx, traj.turns[: neg.turn_idx], traj.tools,
                              traj.capability_refs, neg.good_turn, neg.bad_turn, neg.error_class, neg.origin, dict(neg.meta)))
    return out
