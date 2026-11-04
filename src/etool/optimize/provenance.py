"""Value provenance: which earlier tool result an argument value came from.

Stepwise traces copy values literally from tool results into later calls; to optimize (or to re-express a trace as a
``$ref`` program) we infer those data dependencies by value matching. Values that appear in the user's own text are
"known" and need no dependency.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Optional

Path = tuple[Any, ...]  # keys (str), indices (int) or "*"


def leaves(value: Any, path: Path = ()) -> Iterator[tuple[Path, Any]]:
    """All scalar leaves (str/int/float; not bool/None) with their paths."""
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (str, int, float)):
        if value != "":
            yield path, value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from leaves(v, path + (k,))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from leaves(v, path + (i,))


def leaf_set(value: Any) -> set[str]:
    return {_norm(v) for _, v in leaves(value)}


def _norm(v: Any) -> str:
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return str(v).strip().lower()


def known_in_text(value: Any, texts: list[str]) -> bool:
    s = _norm(value)
    if len(s) < 2:
        return True  # single characters/digits carry no provenance
    return any(s in t.lower() for t in texts)


def find_path(result: Any, target: Any) -> Optional[Path]:
    """First path in ``result`` whose scalar equals ``target`` (normalised string comparison)."""
    t = _norm(target)
    for p, v in leaves(result):
        if _norm(v) == t:
            return p
    return None


def _lists(value: Any, path: Path = ()) -> Iterator[tuple[Path, list]]:
    if isinstance(value, list):
        yield path, value
        for i, v in enumerate(value):
            yield from _lists(v, path + (i,))
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from _lists(v, path + (k,))


def find_list_path(result: Any, target: list) -> Optional[Path]:
    """Path producing ``target`` as a whole: a list equal to it, or a ``[*].key`` projection of a list of dicts."""
    tn = [_norm(x) for x in target]
    for p, lst in _lists(result):
        if len(lst) == len(target) and all(not isinstance(x, (dict, list)) for x in lst) and [_norm(x) for x in lst] == tn:
            return p
    for p, lst in _lists(result):
        if lst and all(isinstance(x, dict) for x in lst) and len(lst) == len(target):
            keys = set(lst[0]).intersection(*[set(x) for x in lst[1:]]) if len(lst) > 1 else set(lst[0])
            for k in sorted(keys):
                col = [x.get(k) for x in lst]
                if all(not isinstance(c, (dict, list)) and c is not None for c in col) and [_norm(c) for c in col] == tn:
                    return p + ("*", k)
    return None


def program_expressible(path: Path) -> bool:
    """A ``$ref`` path may only index lists at position 0 or with ``[*]``: anything else is a data-dependent selection
    that an agent cannot make before seeing the result."""
    return all((not isinstance(p, int)) or p == 0 for p in path)


def path_text(path: Path) -> str:
    out = ""
    for p in path:
        out += f"[{p}]" if (isinstance(p, int) or p == "*") else f".{p}"
    return out


@dataclass(frozen=True)
class Dep:
    src: int  # global call index of the producer
    path: Path
    composed: bool = False  # the value *contains* a result string (agent-composed text): not expressible as a $ref

    @property
    def in_program(self) -> bool:
        return (not self.composed) and program_expressible(self.path)


MIN_COMPOSED_LEAF = 6
MIN_COMPOSED_VALUE = 8


def infer_arg_template(args: Any, producers: list[tuple[int, Any]], texts: list[str]) -> tuple[Any, list[Dep]]:
    """Replace literal values in ``args`` by ``Dep`` objects where they provably come from an earlier result.

    ``producers`` are (call_index, result) of earlier *kept* calls, earliest first (earliest producer wins).
    Returns (template, deps). Values known from the conversation text stay literal.
    """
    deps: list[Dep] = []

    def scalar(v: Any) -> Any:
        if isinstance(v, bool) or v is None or known_in_text(v, texts):
            return v
        for idx, res in producers:
            p = find_path(res, v)
            if p is not None:
                d = Dep(idx, p)
                deps.append(d)
                return d
        # composed strings: the agent built this text from a result (e.g. "Follow up: <subject>") -> hidden dependency
        if isinstance(v, str) and len(v) >= MIN_COMPOSED_VALUE:
            low = v.lower()
            for idx, res in producers:
                for p, leaf in leaves(res):
                    if isinstance(leaf, str) and len(leaf) >= MIN_COMPOSED_LEAF and leaf.lower() in low and not known_in_text(leaf, texts):
                        deps.append(Dep(idx, p, composed=True))
                        return v
        return v

    def walk(v: Any) -> Any:
        if isinstance(v, dict):
            return {k: walk(x) for k, x in v.items()}
        if isinstance(v, list):
            if v and all(not isinstance(x, (dict, list)) for x in v) and not all(known_in_text(x, texts) for x in v):
                for idx, res in producers:
                    p = find_list_path(res, v)
                    if p is not None:
                        d = Dep(idx, p)
                        deps.append(d)
                        return d
            return [walk(x) for x in v]
        return scalar(v)

    return walk(args), deps
