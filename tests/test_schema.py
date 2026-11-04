from etool.schema import Negative, ToolCall, ToolSpec, Trajectory, Turn, dumps, loads


def _traj() -> Trajectory:
    tools = [ToolSpec("get_item", "Get one item", {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]},
                      effect="read", idempotent=True, reads=["item:{id}"])]
    turns = [
        Turn("system", "You are a helpful assistant."),
        Turn("user", "Show item 42."),
        Turn("assistant", "", tool_calls=[ToolCall("get_item", {"id": "42"}, id="1")]),
        Turn("tool", '{"id":"42","name":"lamp"}', tool_call_id="1", name="get_item"),
        Turn("assistant", "Item 42 is a lamp."),
    ]
    return Trajectory(id="t1", source="test", tools=tools, turns=turns)


def test_roundtrip():
    t = _traj()
    t2 = loads(dumps(t))
    assert t2 == t
    assert t2.tools[0].effect == "read" and t2.turns[2].tool_calls[0].id == "1"


def test_helpers():
    t = _traj()
    assert t.n_calls() == 1 and t.n_action_turns() == 1 and t.assistant_indices() == [2, 4]
    assert t.prefix(2).turns == t.turns[:2]
    msgs = t.to_messages()
    assert msgs[2]["tool_calls"][0]["function"]["name"] == "get_item"
    assert msgs[3]["role"] == "tool" and msgs[3]["tool_call_id"] == "1"
    assert t.tools_openai()[0]["function"]["parameters"]["required"] == ["id"]


def test_negative_roundtrip():
    t = _traj()
    n = Negative(traj_id=t.id, turn_idx=2, bad_turn=Turn("assistant", "", [ToolCall("get_item", {"id": "43"}, "1")]),
                 good_turn=t.turns[2], error_class="wrong_arg_value", origin="rule")
    assert loads(dumps(n)) == n
