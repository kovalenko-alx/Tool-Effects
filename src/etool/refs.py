"""``$ref`` syntax: referencing earlier tool results inside one assistant turn.

Grammar (whole-value references only, v1)::

    ref      := "$" index path*
    index    := [1-9][0-9]*                 # 1-based position of the call inside the same turn
    path     := "." identifier | "[" integer "]" | "[*]"      # [*] maps the rest of the path over a list

Examples: ``"$1"``, ``"$2.items[0].id"``, ``"$3.result.email"``, ``"$2.tasks[*].id"`` (-> list of ids).
A reference is only recognised when the *entire* argument value is a string matching the grammar.
(Embedding ``$1`` inside a longer string is deliberately *not* a reference — it stays literal.)
References may appear at any depth inside argument values (dicts / lists are walked).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterator

_REF_RE = re.compile(r"^\$(?P<idx>[1-9][0-9]*)(?P<path>(?:\.[A-Za-z_][A-Za-z0-9_]*|\[[0-9]+\]|\[\*\])*)$")
_PATH_RE = re.compile(r"\.([A-Za-z_][A-Za-z0-9_]*)|\[([0-9]+|\*)\]")


@dataclass(frozen=True)
class Ref:
    index: int  # 1-based call index within the turn
    path: tuple[str | int, ...]  # keys / list positions

    @property
    def text(self) -> str:
        out = f"${self.index}"
        for p in self.path:
            out += f"[{p}]" if (isinstance(p, int) or p == "*") else f".{p}"
        return out


class RefError(ValueError):
    """Unresolvable reference (unknown call, missing key/index, forward/self reference)."""


def parse_ref(value: Any) -> Ref | None:
    """Return a Ref if ``value`` is a whole-value reference string, else None."""
    if not isinstance(value, str):
        return None
    m = _REF_RE.match(value)
    if not m:
        return None
    path: list[str | int] = []
    for key, pos in _PATH_RE.findall(m.group("path")):
        path.append(("*" if pos == "*" else int(pos)) if key == "" else key)
    return Ref(int(m.group("idx")), tuple(path))


def iter_refs(value: Any) -> Iterator[Ref]:
    """All references inside a (possibly nested) argument value."""
    r = parse_ref(value)
    if r is not None:
        yield r
    elif isinstance(value, dict):
        for v in value.values():
            yield from iter_refs(v)
    elif isinstance(value, list):
        for v in value:
            yield from iter_refs(v)


def refs_in_arguments(arguments: dict[str, Any]) -> list[Ref]:
    return list(iter_refs(arguments))


def dependencies(arguments: dict[str, Any]) -> set[int]:
    """Set of call indices this call's arguments depend on."""
    return {r.index for r in iter_refs(arguments)}


def follow(result: Any, path: tuple[str | int, ...], ref_text: str = "") -> Any:
    cur = result
    for i, p in enumerate(path):
        if p == "*":
            if not isinstance(cur, list):
                raise RefError(f"{ref_text}: [*] applied to a non-list")
            return [follow(x, path[i + 1:], ref_text) for x in cur]
        if isinstance(p, int):
            if not isinstance(cur, list) or p >= len(cur):
                raise RefError(f"{ref_text}: index [{p}] out of range")
            cur = cur[p]
        else:
            if not isinstance(cur, dict) or p not in cur:
                raise RefError(f"{ref_text}: no field '{p}'")
            cur = cur[p]
    return cur


def resolve(value: Any, results: dict[int, Any]) -> Any:
    """Substitute every reference in ``value`` using ``results`` (call index -> parsed tool result).

    Raises RefError for unknown indices or bad paths. Returns a new structure; the input is not mutated.
    """
    r = parse_ref(value)
    if r is not None:
        if r.index not in results:
            raise RefError(f"{r.text}: result of call {r.index} is not available")
        return follow(results[r.index], r.path, r.text)
    if isinstance(value, dict):
        return {k: resolve(v, results) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve(v, results) for v in value]
    return value


def has_refs(arguments: dict[str, Any]) -> bool:
    return any(True for _ in iter_refs(arguments))
