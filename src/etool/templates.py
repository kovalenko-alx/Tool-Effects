"""Chat-template handling for training and parsing.

* ``render``: the exact string the model is trained/served on (``tokenizer.apply_chat_template`` with tools).
* ``assistant_spans``: character spans of every assistant turn, found by incremental prefix rendering (Qwen templates
  carry no ``{% generation %}`` markers), so the loss can be masked to assistant tokens only.
* ``encode``: input_ids + labels (-100 outside assistant spans), optionally restricted to selected assistant turns.
* ``diverging_span``: for a (good, bad) pair of assistant turns at the same prefix, the token span of the bad turn from
  the first differing token to its end — the LUNE-style target span for suppression losses.
* ``parse_assistant``: model text -> ``Turn`` (content + tool calls), tolerant to ``<think>`` blocks and minor JSON noise.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional

from etool.schema import ToolCall, Trajectory, Turn

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
PHI_CALL_RE = re.compile(r"<\|tool_call\|>(.*?)(?:<\|/tool_call\|>|$)", re.S)
BARE_CALL_RE = re.compile(r"\{\s*\"name\"\s*:\s*\"[^\"]+\"\s*,\s*\"(?:arguments|parameters)\"\s*:\s*\{(?:[^{}]|\{[^{}]*\})*\}\s*\}", re.S)
THINK_RE = re.compile(r"<think>.*?</think>\s*", re.S)
IM_END = "<|im_end|>"


@dataclass(frozen=True)
class ModelProfile:
    """What differs between model families in how a conversation with tools is rendered and parsed (2026-10-22, for the
    multi-family sweep). ``hermes`` is the Qwen3 convention this project was built on; ``phi4mini`` mirrors BFCL's
    PhiFCHandler byte for byte (tool list as a JSON string inside the system message between <|tool|> and <|/tool|>, one
    ``json.dumps(func)`` per line; calls as <|tool_call|>[{"name","arguments"}]<|/tool_call|>; tool results as role
    ``tool`` rendered <|tool|>...<|end|>)."""
    name: str
    end_token: str  # closes an assistant turn; the supervised span and the diverging span end here
    closing_tag: str  # closes a tool-call block; excluded from suppression masks
    stop: tuple[str, ...]  # stop strings for the completions API
    tools_in_template: bool  # True: apply_chat_template(tools=...); False: tool JSON goes into the system message (Phi)


FALCON_CALL_RE = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.S)

PROFILES = {
    "hermes": ModelProfile("hermes", IM_END, "</tool_call>", (IM_END, "<|endoftext|>"), True),
    # falcon3 (BFCL Falcon3FCHandler, 2026-10-22): the template takes the *unwrapped* function dicts via tools=, writes its own
    # system block with an output-format instruction, renders assistant turns as <|assistant|>\n{content}<|endoftext|>\n and tool
    # results as <|assistant|>\n<tool_response>...</tool_response>; calls are a JSON list inside one <tool_call> tag. <tool_call>
    # tags are not single tokens in its tokenizer, so no closing-tag exclusion in the suppression mask.
    "falcon3": ModelProfile("falcon3", "<|endoftext|>", "</tool_call>", ("<|endoftext|>",), True),
    # <|/tool_call|> is a stop string too: the base model keeps generating after a call (probe 11844179), the parser accepts a missing closer
    "phi4mini": ModelProfile("phi4mini", "<|end|>", "<|/tool_call|>", ("<|end|>", "<|endoftext|>", "<|/tool_call|>"), False),
}
PHI_DEFAULT_SYSTEM = "You are a helpful assistant with some tools."


def detect_profile(tokenizer: Any) -> ModelProfile:
    vocab = tokenizer.get_vocab()
    if "Falcon assistant" in (getattr(tokenizer, "chat_template", None) or ""):
        return PROFILES["falcon3"]
    if "<|tool_call|>" in vocab and "<|im_start|>" not in vocab:
        return PROFILES["phi4mini"]
    return PROFILES["hermes"]


@dataclass
class Encoded:
    input_ids: list[int]
    labels: list[int]
    n_assistant_tokens: int
    truncated: bool
    spans: list[tuple[int, int]]  # token spans (start, end) per supervised assistant turn


class ChatFormatter:
    def __init__(self, tokenizer: Any, max_len: int = 4096, template_kwargs: Optional[dict[str, Any]] = None,
                 profile: Optional[ModelProfile] = None):
        """``template_kwargs`` are passed to ``apply_chat_template`` (e.g. ``{"enable_thinking": False}`` for hybrid
        Qwen3 checkpoints; the 2507 Instruct models ignore it). ``profile`` is detected from the tokenizer unless given."""
        self.tok = tokenizer
        self.max_len = max_len
        self.template_kwargs = dict(template_kwargs or {})
        self.profile = profile or detect_profile(tokenizer)
        if getattr(self.tok, "is_fast", False) is False:
            raise ValueError("a fast tokenizer is required for offset mapping")

    # ------------------------------------------------------------------ rendering
    def _messages(self, turns: list[Turn], tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.profile.name == "falcon3":
            # calls go into the content in the template's own list format (an assistant turn with tool_calls but empty content
            # would get no role header); a newline after our system text, which the template otherwise glues to its own block
            msgs = []
            for t in turns:
                if t.role == "assistant" and t.tool_calls:
                    block = "<tool_call>\n" + json.dumps([{"name": c.name, "arguments": c.arguments} for c in t.tool_calls], indent=2) + "\n</tool_call>"
                    msgs.append({"role": "assistant", "content": (t.content.rstrip() + "\n" if t.content else "") + block})
                elif t.role == "system":
                    msgs.append({"role": "system", "content": t.content.rstrip() + "\n\n"})
                else:
                    msgs.append({"role": t.role, "content": t.content})
            return msgs
        if self.profile.tools_in_template:
            return [t.to_message() for t in turns]
        # phi4mini: tools as a JSON string in the system message (BFCL PhiFCHandler: one json.dumps(func) per line, trailing
        # newline); assistant tool calls serialised into the content; tool results as plain role=tool messages.
        funcs = [t.get("function", t) for t in (tools or [])]
        tool_str = "".join(json.dumps(f) + "\n" for f in funcs)
        msgs: list[dict[str, Any]] = []
        if not turns or turns[0].role != "system":
            msgs.append({"role": "system", "content": PHI_DEFAULT_SYSTEM})
        for t in turns:
            if t.role == "assistant" and t.tool_calls:
                block = "<|tool_call|>" + json.dumps([{"name": c.name, "arguments": c.arguments} for c in t.tool_calls]) + "<|/tool_call|>"
                msgs.append({"role": "assistant", "content": (t.content.rstrip() + "\n" if t.content else "") + block})
            else:
                msgs.append({"role": t.role, "content": t.content})
        if funcs:
            msgs[0] = dict(msgs[0], tools=tool_str)
        return msgs

    def render(self, turns: list[Turn], tools: list[dict[str, Any]], add_generation_prompt: bool = False) -> str:
        msgs = self._messages(turns, tools)
        if self.profile.name == "falcon3":
            tools = [t.get("function", t) for t in (tools or [])]  # BFCL passes the function dicts unwrapped
        kw = {"tools": tools or None} if self.profile.tools_in_template else {}
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=add_generation_prompt, **kw, **self.template_kwargs)

    def render_traj(self, traj: Trajectory, add_generation_prompt: bool = False) -> str:
        return self.render(traj.turns, traj.tools_openai(), add_generation_prompt)

    def assistant_spans(self, turns: list[Turn], tools: list[dict[str, Any]]) -> list[tuple[int, int, int]]:
        """(turn_idx, char_start, char_end) for each assistant turn; the span covers content + tool calls + <|im_end|>."""
        spans = []
        for k, t in enumerate(turns):
            if t.role != "assistant":
                continue
            prefix = self.render(turns[:k], tools, add_generation_prompt=True)
            full = self.render(turns[: k + 1], tools, add_generation_prompt=False)
            if not full.startswith(prefix):
                raise ValueError(f"template is not prefix-preserving at turn {k}")
            seg = full[len(prefix):]
            E = self.profile.end_token
            end = len(prefix) + (seg.rfind(E) + len(E) if E in seg else len(seg))
            spans.append((k, len(prefix), end))
        return spans

    # ------------------------------------------------------------------ encoding with labels
    def encode(self, traj: Trajectory, supervise: Optional[set[int]] = None) -> Encoded:
        text = self.render_traj(traj)
        spans = self.assistant_spans(traj.turns, traj.tools_openai())
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets = enc["input_ids"], enc["offset_mapping"]
        labels = [-100] * len(ids)
        tok_spans: list[tuple[int, int]] = []
        for turn_idx, cs, ce in spans:
            if supervise is not None and turn_idx not in supervise:
                continue
            first = last = None
            for i, (a, b) in enumerate(offsets):
                if b <= cs or a >= ce or a == b:
                    continue
                if first is None:
                    first = i
                last = i
            if first is None:
                continue
            for i in range(first, last + 1):
                labels[i] = ids[i]
            tok_spans.append((first, last + 1))
        truncated = len(ids) > self.max_len
        if truncated:
            ids, labels = ids[: self.max_len], labels[: self.max_len]
            tok_spans = [(a, min(b, self.max_len)) for a, b in tok_spans if a < self.max_len]
        return Encoded(ids, labels, sum(1 for l in labels if l != -100), truncated, tok_spans)

    def diverging_span(self, prefix_turns: list[Turn], tools: list[dict[str, Any]], good: Turn, bad: Turn,
                       trim_common_suffix: bool = False) -> tuple[list[int], int, int]:
        """Token ids of prefix+bad turn, and the (start, end) token span of the bad turn from its first token that
        differs from the good turn to the end of the bad turn (the suppression target).

        With ``trim_common_suffix`` the span also stops before the longest token suffix the two turns share, so the
        closing ``</tool_call>`` / ``<|im_end|>`` the good turn needs as well are never suppression targets; at least
        one token is kept, e.g. the premature ``<|im_end|>`` of a turn that stops before a required call. Round-1 arms
        (s1_*) trained on untrimmed spans and learned not to close calls or turns (RESULTS §7)."""
        pre = self.render(prefix_turns, tools, add_generation_prompt=True)
        g = self.render(prefix_turns + [good], tools)[len(pre):]
        b = self.render(prefix_turns + [bad], tools)[len(pre):]
        E = self.profile.end_token
        g = g[: g.rfind(E) + len(E)] if E in g else g
        b = b[: b.rfind(E) + len(E)] if E in b else b
        pre_ids = self.tok(pre, add_special_tokens=False)["input_ids"]
        g_ids = self.tok(g, add_special_tokens=False)["input_ids"]
        b_ids = self.tok(b, add_special_tokens=False)["input_ids"]
        k = 0
        while k < min(len(g_ids), len(b_ids)) and g_ids[k] == b_ids[k]:
            k += 1
        end = len(b_ids)
        if trim_common_suffix and k < len(b_ids):
            s = 0  # only the bad span must stay non-empty; the good turn may be entirely prefix + suffix (an extra call)
            while s < len(b_ids) - k - 1 and s < len(g_ids) and g_ids[-1 - s] == b_ids[-1 - s]:
                s += 1
            end = len(b_ids) - s
        return pre_ids + b_ids, len(pre_ids) + k, len(pre_ids) + end

    # ------------------------------------------------------------------ parsing
    def parse(self, text: str) -> Turn:
        """Model text -> Turn for this formatter's profile."""
        if self.profile.name == "phi4mini":
            return self.parse_phi(text)
        if self.profile.name == "falcon3":
            return self.parse_falcon(text)
        return self.parse_assistant(text)

    @staticmethod
    def parse_falcon(text: str) -> Turn:
        calls: list[ToolCall] = []
        for m in FALCON_CALL_RE.finditer(text):
            body = m.group(1).strip()
            if body.startswith("```"):
                body = body.strip("`").strip()
                if body.startswith("json"):
                    body = body[4:].strip()
            try:
                objs = json.loads(body) if body else []
            except json.JSONDecodeError:
                objs = _lenient_json(body) or []
            for obj in objs if isinstance(objs, list) else [objs]:
                if isinstance(obj, dict) and "name" in obj:
                    args = obj.get("arguments", {})
                    calls.append(ToolCall(str(obj["name"]), args if isinstance(args, dict) else {"_raw": args}, str(len(calls) + 1)))
        content = FALCON_CALL_RE.sub("", text).replace("<|endoftext|>", "").strip()
        return Turn("assistant", content, calls)

    @staticmethod
    def parse_phi(text: str) -> Turn:
        calls: list[ToolCall] = []
        text = re.sub(r"(?:<\|tool_call\|>\s*){2,}", "<|tool_call|>", text)  # the base model often doubles the opening tag (BFCL run 11843962)
        for m in PHI_CALL_RE.finditer(text):
            body = m.group(1).strip()
            if body and not body.startswith("["):
                body = "[" + body + "]"
            try:
                objs = json.loads(body) if body else []
            except json.JSONDecodeError:
                objs = _lenient_json(body) or []
            for obj in objs if isinstance(objs, list) else [objs]:
                if isinstance(obj, dict) and "name" in obj:
                    args = obj.get("arguments", {})
                    calls.append(ToolCall(str(obj["name"]), args if isinstance(args, dict) else {"_raw": args}, str(len(calls) + 1)))
        content = PHI_CALL_RE.sub("", text).replace("<|end|>", "").strip()
        if not calls:  # the base model often writes the call as bare JSON without the tags (300 of 1,438 turns, 2026-10-22)
            for m in BARE_CALL_RE.finditer(text):
                obj = _lenient_json(m.group(0)) if _lenient_json(m.group(0)) is not None else None
                if isinstance(obj, dict) and "name" in obj and isinstance(obj.get("arguments", obj.get("parameters", {})), dict):
                    calls.append(ToolCall(str(obj["name"]), obj.get("arguments", obj.get("parameters", {})), str(len(calls) + 1)))
            if calls:
                content = BARE_CALL_RE.sub("", content).replace("```json", "").replace("```", "").strip()
        return Turn("assistant", content, calls)

    @staticmethod
    def parse_assistant(text: str) -> Turn:
        text = THINK_RE.sub("", text)
        calls: list[ToolCall] = []
        for i, m in enumerate(TOOL_CALL_RE.finditer(text)):
            try:
                obj = json.loads(m.group(1))
            except json.JSONDecodeError:
                obj = _lenient_json(m.group(1))
            if not isinstance(obj, dict) or "name" not in obj:
                continue
            args = obj.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": args}
            calls.append(ToolCall(str(obj["name"]), args if isinstance(args, dict) else {"_raw": args}, str(len(calls) + 1)))
        content = TOOL_CALL_RE.sub("", text).strip()
        content = content.replace(IM_END, "").strip()
        return Turn("assistant", content, calls)


def _lenient_json(s: str) -> Any:
    """Best-effort repair for trailing commas / single quotes."""
    s2 = re.sub(r",\s*([}\]])", r"\1", s)
    try:
        return json.loads(s2)
    except json.JSONDecodeError:
        try:
            return json.loads(s2.replace("'", '"'))
        except json.JSONDecodeError:
            return None


def load_tokenizer(path_or_id: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path_or_id)
