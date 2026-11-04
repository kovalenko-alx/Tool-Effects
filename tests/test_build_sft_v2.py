"""Data v2 helpers: refusal filtering and public-prefix refusal pairs."""
from etool.data.build_sft import first_turn_calls, is_refusal_text, public_ok, public_refusal_pairs
from etool.schema import ToolCall, ToolSpec, Trajectory, Turn
import random

TOOL = ToolSpec("get_weather", "Weather", {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]})


def _traj(tid, turns):
    return Trajectory(id=tid, source="nemotron:tool_calling", tools=[TOOL], turns=turns)


def test_refusal_filter_and_pairs():
    good = _traj("g", [Turn("user", "Weather in Prague?"), Turn("assistant", "", [ToolCall("get_weather", {"city": "Prague"}, "1")]), Turn("tool", "{}", tool_call_id="1", name="get_weather"), Turn("assistant", "Sunny.")])
    refusal = _traj("r", [Turn("user", "Weather in Prague?"), Turn("assistant", "I don't have access to a weather tool.")])
    noisy_user = _traj("n", [Turn("user", "I do not have access to tools that can do that."), Turn("assistant", "", [ToolCall("get_weather", {"city": "Prague"}, "1")])])
    chit = _traj("c", [Turn("user", "Hi!"), Turn("assistant", "Hello! How can I help?"), Turn("user", "Weather in Prague?"), Turn("assistant", "", [ToolCall("get_weather", {"city": "Prague"}, "1")])])
    assert public_ok(good) and public_ok(chit) and not public_ok(refusal) and not public_ok(noisy_user)
    assert first_turn_calls(good) and not first_turn_calls(chit)
    assert is_refusal_text("My current capabilities are limited to the tools listed above.") and not is_refusal_text("Here is the forecast.")
    pairs = public_refusal_pairs([good, chit], 5, random.Random(0))
    assert len(pairs) == 1 and pairs[0].error_class == "refusal" and pairs[0].chosen.tool_calls and not pairs[0].rejected.tool_calls
    assert pairs[0].prefix == good.turns[:1] and pairs[0].turn_idx == 1
