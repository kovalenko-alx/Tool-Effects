"""Training datasets: SFT trajectories and (prefix, good, bad) pairs, pre-tokenised with the chat template."""
from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

import torch

from etool.negatives.pairs import PairRecord
from etool.schema import Trajectory, read_jsonl
from etool.templates import ChatFormatter


@dataclass
class SFTExample:
    input_ids: list[int]
    labels: list[int]
    source: str


@dataclass
class PairExample:
    good_ids: list[int]
    good_span: tuple[int, int]  # token span of the good turn (loss target for retain / chosen logp)
    bad_ids: list[int]
    bad_span: tuple[int, int]  # diverging span of the bad turn, common suffix trimmed (suppression target)
    bad_full_span: tuple[int, int]  # whole bad turn (for DPO rejected logp)
    error_class: str
    source: str


def _cache_path(cache_dir: Optional[str], files: list[str], tag: str, max_len: int, extra: str = "") -> Optional[Path]:
    """``extra`` goes into the hash but not the file name (a class list spelled out in the name exceeded NAME_MAX)."""
    if not cache_dir:
        return None
    h = hashlib.sha1()
    for f in files:
        st = os.stat(f)
        h.update(f"{f}:{st.st_size}:{int(st.st_mtime)}".encode())
    h.update(f"{tag}:{max_len}".encode())
    h.update(extra.encode())
    return Path(cache_dir) / f"{tag}_{h.hexdigest()[:16]}.pt"


def _profile_key(fmt: ChatFormatter) -> str:
    """Tokenisation depends on the model family; Qwen/hermes keeps the historical key, other families get their own."""
    return "" if fmt.profile.name == "hermes" else f"|{fmt.profile.name}:{len(fmt.tok)}"


def load_sft(files: list[str], fmt: ChatFormatter, cache_dir: Optional[str] = None, limit_per_file: Optional[int] = None) -> list[SFTExample]:
    cp = _cache_path(cache_dir, files, f"sft{limit_per_file or ''}", fmt.max_len, extra=_profile_key(fmt))
    if cp and cp.exists():
        return torch.load(cp, weights_only=False)
    out: list[SFTExample] = []
    skipped = 0
    for f in files:
        n = 0
        for traj in read_jsonl(f, Trajectory):
            try:
                enc = fmt.encode(traj)
            except ValueError:  # the family's template cannot render a turn as a prefix (Falcon3 drops the header of an
                skipped += 1  # assistant turn with neither content nor calls; 5 ToolACE trajectories in v2/v3a)
                continue
            if enc.truncated or enc.n_assistant_tokens == 0:
                continue
            out.append(SFTExample(enc.input_ids, enc.labels, traj.source))
            n += 1
            if limit_per_file and n >= limit_per_file:
                break
    if skipped:
        print(f"load_sft: skipped {skipped} trajectories the {fmt.profile.name} template cannot render", flush=True)
    if cp:
        cp.parent.mkdir(parents=True, exist_ok=True)
        torch.save(out, cp)
    return out


def load_pairs(files: list[str], fmt: ChatFormatter, cache_dir: Optional[str] = None, classes: Optional[set[str]] = None,
               max_per_class: Optional[int] = None, seed: int = 0) -> list[PairExample]:
    # "pairs2": spans with the common suffix trimmed (2026-10-21); "pairs" caches hold the untrimmed round-1 spans
    cp = _cache_path(cache_dir, files, f"pairs2{f'c{len(classes)}' if classes else ''}{max_per_class or ''}", fmt.max_len,
                     extra=("-".join(sorted(classes)) if classes else "") + _profile_key(fmt))
    if cp and cp.exists():
        return torch.load(cp, weights_only=False)
    recs: list[PairRecord] = []
    for f in files:
        with open(f) as fh:
            for line in fh:
                if line.strip():
                    r = PairRecord.from_json(line)
                    if classes is None or r.error_class in classes:
                        recs.append(r)
    if max_per_class:
        rng = random.Random(seed)
        rng.shuffle(recs)
        seen: dict[str, int] = {}
        kept = []
        for r in recs:
            if seen.get(r.error_class, 0) < max_per_class:
                kept.append(r)
                seen[r.error_class] = seen.get(r.error_class, 0) + 1
        recs = kept
    out: list[PairExample] = []
    for r in recs:
        tools = [t.to_openai() for t in r.tools]
        good_ids, gs, ge = fmt.diverging_span(r.prefix, tools, r.rejected, r.chosen)  # whole good turn (ge = its end)
        bad_ids, bs, be = fmt.diverging_span(r.prefix, tools, r.chosen, r.rejected, trim_common_suffix=True)
        pre_len = len(fmt.tok(fmt.render(r.prefix, tools, add_generation_prompt=True), add_special_tokens=False)["input_ids"])
        if max(len(good_ids), len(bad_ids)) > fmt.max_len or be <= bs:
            continue
        out.append(PairExample(good_ids, (pre_len, ge), bad_ids, (bs, be), (pre_len, be), r.error_class, r.meta.get("kind", "") or ""))
    if cp:
        cp.parent.mkdir(parents=True, exist_ok=True)
        torch.save(out, cp)
    return out


def pad_batch(seqs: list[list[int]], pad_id: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
        att[i, : len(s)] = 1
    return ids.to(device), att.to(device)


def collate_sft(batch: list[SFTExample], pad_id: int, device: torch.device) -> dict[str, torch.Tensor]:
    ids, att = pad_batch([b.input_ids for b in batch], pad_id, device)
    labels = torch.full_like(ids, -100)
    for i, b in enumerate(batch):
        labels[i, : len(b.labels)] = torch.tensor(b.labels)
    return {"input_ids": ids, "attention_mask": att, "labels": labels}


def span_mask(seqs: list[list[int]], spans: list[tuple[int, int]], device: torch.device) -> torch.Tensor:
    L = max(len(s) for s in seqs)
    m = torch.zeros((len(seqs), L), dtype=torch.bool)
    for i, (a, b) in enumerate(spans):
        m[i, a:b] = True
    return m.to(device)


class EpochSampler:
    """Deterministic shuffled order with a resumable position."""

    def __init__(self, n: int, seed: int):
        self.n, self.seed, self.epoch, self.pos = n, seed, 0, 0
        self._order = self._perm()

    def _perm(self) -> list[int]:
        rng = random.Random(self.seed * 1000 + self.epoch)
        order = list(range(self.n))
        rng.shuffle(order)
        return order

    def take(self, k: int) -> list[int]:
        out = []
        while len(out) < k:
            if self.pos >= self.n:
                self.epoch += 1
                self.pos = 0
                self._order = self._perm()
            out.append(self._order[self.pos])
            self.pos += 1
        return out

    def state(self) -> dict[str, int]:
        return {"epoch": self.epoch, "pos": self.pos}

    def load(self, st: dict[str, int]) -> None:
        self.epoch, self.pos = st["epoch"], st["pos"]
        self._order = self._perm()
