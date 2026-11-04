"""Model profiles: the Phi-4-mini rendering must match BFCL's PhiFCHandler format; parsing must round-trip; hermes unchanged."""
import glob
import json
import os

import pytest

from etool.schema import ToolCall, Turn
from etool.templates import PROFILES, ChatFormatter, detect_profile, load_tokenizer

PHI = glob.glob(os.path.expanduser("~/EfficeintTool/hf/hub/models--microsoft--Phi-4-mini-instruct/snapshots/*/tokenizer.json"))
QWEN = glob.glob(os.path.expanduser("~/EfficeintTool/hf/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/*/tokenizer.json"))

TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Weather for a city.",
                                            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
TURNS = [Turn("system", "You are a workspace assistant."), Turn("user", "Weather in Prague?"),
         Turn("assistant", "", [ToolCall("get_weather", {"city": "Prague"}, "1")]),
         Turn("tool", '{"temp": 12}', tool_call_id="1", name="get_weather"), Turn("assistant", "It is 12 degrees in Prague.")]


@pytest.mark.skipif(not PHI, reason="Phi-4-mini not downloaded")
def test_phi_render_matches_bfcl_handler_format():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    fmt = ChatFormatter(load_tokenizer(os.path.dirname(PHI[0])), max_len=4096)
    assert fmt.profile.name == "phi4mini"
    tools_str = json.dumps(TOOLS[0]["function"]) + "\n"
    expected = ("<|system|>You are a workspace assistant.<|tool|>" + tools_str + "<|/tool|><|end|>"
                "<|user|>Weather in Prague?<|end|><|assistant|>")
    assert fmt.render(TURNS[:2], TOOLS, add_generation_prompt=True) == expected
    full = fmt.render(TURNS, TOOLS)
    assert '<|assistant|><|tool_call|>[{"name": "get_weather", "arguments": {"city": "Prague"}}]<|/tool_call|><|end|>' in full
    assert '<|tool|>{"temp": 12}<|end|>' in full and full.endswith("It is 12 degrees in Prague.<|end|><|endoftext|>")
    spans = fmt.assistant_spans(TURNS, TOOLS)
    assert [k for k, _, _ in spans] == [2, 4]
    assert all(full[a:b].endswith("<|end|>") for _, a, b in spans)
    assert fmt.tok.convert_tokens_to_ids(fmt.profile.closing_tag) == 200026 and "<|/tool_call|>" in fmt.profile.stop
    # parsing: list form, bare dict form, missing closing tag, plain text
    t = fmt.parse('<|tool_call|>[{"name": "get_weather", "arguments": {"city": "Prague"}}]<|/tool_call|>')
    assert t.tool_calls[0].name == "get_weather" and t.tool_calls[0].arguments == {"city": "Prague"} and t.content == ""
    t = fmt.parse('<|tool_call|>{"name": "get_weather", "arguments": {"city": "Brno"}}')
    assert t.tool_calls and t.tool_calls[0].arguments == {"city": "Brno"}
    t = fmt.parse("Sure, 12 degrees.<|end|>")
    assert not t.tool_calls and t.content == "Sure, 12 degrees."
    # diverging span works with the phi end token
    ids, s, e = fmt.diverging_span(TURNS[:2], TOOLS, TURNS[2], Turn("assistant", "", [ToolCall("get_weather", {"city": "Brno"}, "1")]), trim_common_suffix=True)
    assert 0 < s < e <= len(ids) and 200026 not in ids[s:e]


@pytest.mark.skipif(not QWEN, reason="Qwen tokenizer not downloaded")
def test_hermes_profile_is_default_and_unchanged():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    tok = load_tokenizer(os.path.dirname(QWEN[0]))
    assert detect_profile(tok) is PROFILES["hermes"]
    fmt = ChatFormatter(tok, max_len=4096)
    assert fmt.profile.stop == ("<|im_end|>", "<|endoftext|>") and fmt.tok.convert_tokens_to_ids("</tool_call>") != tok.unk_token_id
    assert fmt.parse('<tool_call>\n{"name": "x", "arguments": {"a": 1}}\n</tool_call>').tool_calls[0].arguments == {"a": 1}


def test_phi_parse_accepts_untagged_json_calls():
    t = ChatFormatter.parse_phi('Let me check.\n\n{"name": "get_weather", "arguments": {"city": "Prague"}}')
    assert t.tool_calls and t.tool_calls[0].name == "get_weather" and t.tool_calls[0].arguments == {"city": "Prague"} and t.content == "Let me check."
    t = ChatFormatter.parse_phi('{"name": "resolve_date", "description": "Convert a relative expression", "parameters": {"type": "object"}}')
    assert not t.tool_calls  # an echoed schema (has "description", no arguments) is not a call
    t = ChatFormatter.parse_phi("The area is 25 square units.")
    assert not t.tool_calls and t.content == "The area is 25 square units."


def test_phi_parse_tolerates_doubled_tag_and_trailing_text():
    t = ChatFormatter.parse_phi('<|tool_call|><|tool_call|>[{"name": "math.gcd", "arguments": {"num1": 40, "num2": 50}}]<|/tool_call|>output: 10')
    assert [c.name for c in t.tool_calls] == ["math.gcd"] and t.tool_calls[0].arguments == {"num1": 40, "num2": 50}
    t = ChatFormatter.parse_phi('<|tool_call|><|tool_call|>[{"name": "solve", "arguments": {"a": 2}}]')
    assert t.tool_calls and t.tool_calls[0].arguments == {"a": 2}


FALCON = glob.glob(os.path.expanduser("~/EfficeintTool/hf/hub/models--tiiuae--Falcon3-3B-Instruct/snapshots/*/tokenizer.json"))


@pytest.mark.skipif(not FALCON, reason="Falcon3 not downloaded")
def test_falcon_render_matches_bfcl_handler_format():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    tok = load_tokenizer(os.path.dirname(FALCON[0]))
    fmt = ChatFormatter(tok, max_len=4096)
    assert fmt.profile.name == "falcon3"
    call_text = '<tool_call>\n' + json.dumps([{"name": "get_weather", "arguments": {"city": "Prague"}}], indent=2) + '\n</tool_call>'
    msgs = [{"role": "system", "content": "You are a workspace assistant.\n\n"}, {"role": "user", "content": "Weather in Prague?"},
            {"role": "assistant", "content": call_text}, {"role": "tool", "content": '{"temp": 12}'}, {"role": "assistant", "content": "It is 12 degrees in Prague."}]
    expected = tok.apply_chat_template(msgs, tools=[TOOLS[0]["function"]], tokenize=False, add_generation_prompt=False)  # the handler's call
    assert fmt.render(TURNS, TOOLS) == expected
    assert fmt.render(TURNS[:2], TOOLS, add_generation_prompt=True).endswith("<|assistant|>\n")
    spans = fmt.assistant_spans(TURNS, TOOLS)
    full = fmt.render(TURNS, TOOLS)
    assert [k for k, _, _ in spans] == [2, 4] and all(full[a:b].endswith("<|endoftext|>") for _, a, b in spans)
    t = fmt.parse(call_text + "<|endoftext|>")
    assert t.tool_calls[0].name == "get_weather" and t.tool_calls[0].arguments == {"city": "Prague"} and t.content == ""
    t = fmt.parse('<tool_call>\n```json\n{"name": "get_weather", "arguments": {"city": "Brno"}}\n```\n</tool_call>')
    assert t.tool_calls and t.tool_calls[0].arguments == {"city": "Brno"}
    t = fmt.parse("<tool_call>\n[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Brno\"}}]")  # missing closer
    assert t.tool_calls
    assert not fmt.parse("Sure, 12 degrees.").tool_calls
    ids, s, e = fmt.diverging_span(TURNS[:2], TOOLS, TURNS[2], Turn("assistant", "", [ToolCall("get_weather", {"city": "Brno"}, "1")]), trim_common_suffix=True)
    assert 0 < s < e <= len(ids)


@pytest.mark.skipif(not FALCON, reason="Falcon3 not downloaded")
def test_load_sft_skips_turns_the_template_cannot_render(tmp_path):
    """Falcon3's template emits no header for an assistant turn with neither content nor calls (5 ToolACE trajectories),
    so the trajectory is not prefix-preserving; load_sft skips it instead of failing the whole cache build."""
    from etool.schema import ToolSpec, Trajectory, write_jsonl
    from etool.train.data import load_sft
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    fmt = ChatFormatter(load_tokenizer(os.path.dirname(FALCON[0])), max_len=4096)
    spec = ToolSpec("get_weather", "Weather for a city.", TOOLS[0]["function"]["parameters"])
    good = Trajectory("g", "test", [spec], TURNS)
    empty = Trajectory("e", "test", [spec], [TURNS[0], TURNS[1], Turn("assistant", ""), Turn("user", "Hello?"), TURNS[4]])
    f = tmp_path / "sft.jsonl"
    write_jsonl(f, [good, empty, good])
    out = load_sft([str(f)], fmt)
    assert len(out) == 2 and all(e.source == "test" for e in out)
