import glob
import os

import pytest

from etool.schema import ToolCall, Turn
from etool.templates import ChatFormatter, load_tokenizer
from etool.world.domains.workspace import Workspace
from etool.world.domains.workspace_tasks import generate
from etool.world.rollout import run_plan

SNAPS = glob.glob(os.path.expanduser("~/EfficeintTool/hf/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/*/tokenizer.json"))
pytestmark = pytest.mark.skipif(not SNAPS, reason="base tokenizer not downloaded")


@pytest.fixture(scope="module")
def fmt():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    return ChatFormatter(load_tokenizer(os.path.dirname(SNAPS[0])), max_len=4096)


def _gold(refs):
    dom = Workspace()
    task, opt, _ = generate(dom, 3)  # schedule_meeting-ish or any kind
    traj, _ = run_plan(dom, task, opt, refs=refs, source="optimal")
    return traj


def test_render_and_spans_cover_assistant_turns_only(fmt):
    traj = _gold(True)
    text = fmt.render_traj(traj)
    assert "<tools>" in text and "<tool_call>" in text and text.count("<|im_start|>assistant") == len(traj.assistant_indices())
    spans = fmt.assistant_spans(traj.turns, traj.tools_openai())
    assert [k for k, _, _ in spans] == traj.assistant_indices()
    for k, a, b in spans:
        seg = text[a:b]
        assert seg.endswith("<|im_end|>") and "<|im_start|>" not in seg
        if traj.turns[k].tool_calls:
            assert seg.count("<tool_call>") == len(traj.turns[k].tool_calls)


def test_encode_labels(fmt):
    traj = _gold(False)
    enc = fmt.encode(traj)
    assert len(enc.input_ids) == len(enc.labels) and enc.n_assistant_tokens > 0 and not enc.truncated
    # decoded supervised tokens == concatenation of assistant segments
    sup = [i for i, l in zip(enc.input_ids, enc.labels) if l != -100]
    decoded = fmt.tok.decode(sup)
    assert decoded.count("<|im_end|>") == len(traj.assistant_indices())
    assert "<|im_start|>" not in decoded and "<tool_response>" not in decoded
    # supervise a single turn
    k = traj.assistant_indices()[0]
    enc1 = fmt.encode(traj, supervise={k})
    assert 0 < enc1.n_assistant_tokens < enc.n_assistant_tokens and len(enc1.spans) == 1


def test_parse_roundtrip(fmt):
    traj = _gold(True)
    for k in traj.assistant_indices():
        t = traj.turns[k]
        seg = fmt.render(traj.turns[: k + 1], traj.tools_openai())[len(fmt.render(traj.turns[:k], traj.tools_openai(), True)):]
        parsed = ChatFormatter.parse_assistant(seg)
        assert [c.name for c in parsed.tool_calls] == [c.name for c in t.tool_calls]
        assert [c.arguments for c in parsed.tool_calls] == [c.arguments for c in t.tool_calls]
        assert parsed.content == t.content
    p = ChatFormatter.parse_assistant('<think>hmm</think>Sure.\n<tool_call>\n{"name": "x", "arguments": {"a": 1,}}\n</tool_call>')
    assert p.content == "Sure." and p.tool_calls[0].arguments == {"a": 1}


def test_diverging_span(fmt):
    traj = _gold(False)
    k = traj.assistant_indices()[0]
    good = traj.turns[k]
    bad = Turn("assistant", "", [ToolCall("delete_event", {"event_id": "e1"})]) if good.tool_calls else Turn("assistant", "Nope.")
    ids, s, e = fmt.diverging_span(traj.turns[:k], traj.tools_openai(), good, bad)
    assert 0 < s < e == len(ids)
    tail = fmt.tok.decode(ids[s:e])
    assert tail.endswith("<|im_end|>") and ("delete_event" in tail or "Nope" in tail)
    # identical turns -> empty span at the end
    ids2, s2, e2 = fmt.diverging_span(traj.turns[:k], traj.tools_openai(), good, good)
    assert s2 == e2 == len(ids2)


def test_diverging_span_trims_common_suffix(fmt):
    """The suppression span must never cover the closing tokens the good turn needs too (RESULTS §7)."""
    traj = _gold(False)
    k = traj.assistant_indices()[0]
    good = traj.turns[k]
    assert good.tool_calls
    prefix, tools = traj.turns[:k], traj.tools_openai()
    im_end, tc_close = fmt.tok.convert_tokens_to_ids("<|im_end|>"), fmt.tok.convert_tokens_to_ids("</tool_call>")
    c = good.tool_calls[0]
    extra = ToolCall("list_events", {"start": "2026-04-01", "end": "2026-04-02"})

    def span(g, b):
        ids, s, e = fmt.diverging_span(prefix, tools, g, b, trim_common_suffix=True)
        assert s < e <= len(ids)
        return ids[s:e]

    # wrong argument value: only the differing value tokens remain, no closing tags
    args = dict(c.arguments)
    args[next(iter(args))] = "zzz-wrong-value"
    sp = span(good, Turn("assistant", "", [ToolCall(c.name, args)] + list(good.tool_calls[1:])))
    assert im_end not in sp and tc_close not in sp and "wrong-value" in fmt.tok.decode(sp)
    # extra call appended: the span is the extra call without its closing tag / end of turn
    sp = span(good, Turn("assistant", "", list(good.tool_calls) + [extra]))
    assert im_end not in sp and tc_close not in sp and "list_events" in fmt.tok.decode(sp)
    # missing call: the only bad token is the premature end of turn, which must be kept
    sp = span(Turn("assistant", "", list(good.tool_calls) + [extra]), good)
    assert sp == [im_end]
    # text instead of a call: the whole text, without the end of turn
    sp = span(good, Turn("assistant", "I don't have access to tools that can do this."))
    assert im_end not in sp and "access" in fmt.tok.decode(sp)
    # untrimmed spans still run to the end of the turn
    ids, s, e = fmt.diverging_span(prefix, tools, good, Turn("assistant", "", list(good.tool_calls) + [extra]))
    assert ids[e - 1] == im_end
