import json

from etool.schema import ToolCall, ToolSpec, Turn
from etool.world.base import DEFAULT_POLICIES, Domain, ToolError, WorldState, footprints_conflict, validate_args
from etool.world.executor import execute_turn, plan_levels


class Toy(Domain):
    name = "toy"
    description = "a toy inventory"
    policies = DEFAULT_POLICIES

    def __init__(self):
        super().__init__()
        obj = lambda props, req: {"type": "object", "properties": props, "required": req}
        self.add(ToolSpec("list_items", "List items", obj({"owner": {"type": "string"}}, ["owner"]), effect="read",
                          idempotent=True, reads=["items:{owner}:*"], latency_ms=100), self._list)
        self.add(ToolSpec("get_item", "Get item", obj({"id": {"type": "string"}}, ["id"]), effect="read", idempotent=True,
                          reads=["item:{id}"], latency_ms=50), self._get)
        self.add(ToolSpec("delete_item", "Delete item", obj({"id": {"type": "string"}}, ["id"]), effect="destructive",
                          idempotent=False, reads=["item:{id}"], writes=["item:{id}"], latency_ms=80), self._delete)
        self.add(ToolSpec("add", "Add numbers", obj({"a": {"type": "number"}, "b": {"type": "number"}}, ["a", "b"]),
                          effect="pure", idempotent=True, latency_ms=1), lambda s, a: a["a"] + a["b"])

    def new_state(self, seed):
        return WorldState({"items": {"i1": {"id": "i1", "owner": "ann", "price": 3}, "i2": {"id": "i2", "owner": "ann", "price": 4}}}, seed)

    def _list(self, s, a):
        return {"items": [v for v in s.data["items"].values() if v["owner"] == a["owner"]]}

    def _get(self, s, a):
        if a["id"] not in s.data["items"]:
            raise ToolError("not_found", f"no item {a['id']}")
        return s.data["items"][a["id"]]

    def _delete(self, s, a):
        s.data["items"].pop(a["id"], None)
        return {"deleted": a["id"]}


def test_levels_and_parallel_latency():
    d, s = Toy(), Toy().new_state(0)
    turn = Turn("assistant", "", [
        ToolCall("list_items", {"owner": "ann"}),               # 1
        ToolCall("get_item", {"id": "$1.items[0].id"}),         # 2 depends on 1
        ToolCall("get_item", {"id": "$1.items[1].id"}),         # 3 depends on 1
        ToolCall("add", {"a": "$2.price", "b": "$3.price"}),    # 4 depends on 2,3
    ])
    r = execute_turn(d, s, turn, turn_idx=2, capability_refs=True)
    assert r.ok and r.groups == [["1"], ["2", "3"], ["4"]]
    assert json.loads(r.tool_turns[3].content) == 7
    lat_by_id = {o.call_id: o.latency_ms for o in r.ops}
    assert abs(r.latency_ms - (lat_by_id["1"] + max(lat_by_id["2"], lat_by_id["3"]) + lat_by_id["4"])) < 1e-6
    assert [o.reads for o in r.ops][:2] == [["items:ann:*"], ["item:i1"]]
    assert all(t.tool_call_id == str(i + 1) for i, t in enumerate(r.tool_turns))


def test_refs_disabled_and_invalid():
    d, s = Toy(), Toy().new_state(0)
    turn = Turn("assistant", "", [ToolCall("list_items", {"owner": "ann"}), ToolCall("get_item", {"id": "$1.items[0].id"})])
    r = execute_turn(d, s, turn, turn_idx=2, capability_refs=False)
    assert r.errors == {"2": "refs_not_supported"} and "1" not in r.errors
    turn2 = Turn("assistant", "", [ToolCall("get_item", {"id": "$2.id"}), ToolCall("get_item", {"id": "i1"})])
    r2 = execute_turn(d, s, turn2, turn_idx=2, capability_refs=True)
    assert r2.errors == {"1": "invalid_ref"} and r2.groups == [["2"]]
    turn3 = Turn("assistant", "", [ToolCall("list_items", {"owner": "bob"}), ToolCall("get_item", {"id": "$1.items[0].id"})])
    r3 = execute_turn(d, s, turn3, turn_idx=2, capability_refs=True)
    assert r3.errors == {"2": "unresolved_ref"}


def test_validation_and_tool_errors_and_writes():
    d, s = Toy(), Toy().new_state(0)
    turn = Turn("assistant", "", [
        ToolCall("get_item", {}), ToolCall("get_item", {"id": 5}), ToolCall("nope", {}), ToolCall("get_item", {"id": "zz"}),
        ToolCall("delete_item", {"id": "i1"}),
    ])
    r = execute_turn(d, s, turn, turn_idx=4, capability_refs=True)
    assert r.errors == {"1": "invalid_arguments", "2": "invalid_arguments", "3": "unknown_tool", "4": "not_found"}
    assert json.loads(r.tool_turns[0].content)["error"]["code"] == "invalid_arguments"
    assert "i1" not in s.data["items"]
    op = [o for o in s.ops if o.call_id == "5"][0]
    assert op.ok and op.writes == ["item:i1"] and op.effect == "destructive" and op.idempotent is False
    assert len(plan_levels([])[0]) == 0


def test_footprints_and_validate():
    assert footprints_conflict("items:ann:*", "items:ann:i1") and not footprints_conflict("item:i1", "item:i2")
    spec = ToolSpec("t", "t", {"type": "object", "properties": {"n": {"type": "integer"}, "k": {"type": "string", "enum": ["a", "b"]}}, "required": ["n"]})
    assert validate_args(spec, {"n": 1, "k": "a"}) == []
    assert len(validate_args(spec, {"k": "c", "x": 1, "n": True})) == 3
    assert "Policies" in Toy().system_prompt(True) and "$N" in Toy().system_prompt(True) and "$N" not in Toy().system_prompt(False)
