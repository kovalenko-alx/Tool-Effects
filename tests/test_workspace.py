import datetime as dt
import json

from etool.schema import ToolCall, Turn
from etool.world.domains.workspace import Workspace, resolve_relative_date
from etool.world.executor import execute_turn


def test_state_is_deterministic_and_tools_run():
    d = Workspace()
    s1, s2 = d.new_state(7), d.new_state(7)
    assert s1.data == s2.data and len(d.tools) == 18
    assert all(t.spec.effect in ("pure", "read", "write", "destructive") for t in d.tools.values())
    today = s1.data["today"]
    r = execute_turn(d, s1, Turn("assistant", "", [
        ToolCall("resolve_date", {"expression": "tomorrow"}),
        ToolCall("list_events", {"date": "$1.date"}),
        ToolCall("create_event", {"title": "Sync", "date": "$1.date", "start": "09:00", "end": "09:30"}),
    ]), turn_idx=2, capability_refs=True)
    assert r.ok, r.errors
    assert r.groups == [["1"], ["2"], ["3"]]  # list_events reads event:*, create_event writes event:* -> footprint edge serialises them
    created = json.loads(r.tool_turns[2].content)["created"]
    assert created["date"] == (dt.date.fromisoformat(today) + dt.timedelta(days=1)).isoformat()
    assert created["id"] in s1.data["events"]
    op = [o for o in r.ops if o.call_id == "3"][0]
    assert op.effect == "write" and op.writes == ["event:*"]


def test_destructive_and_batch_tools():
    d = Workspace(); s = d.new_state(3)
    open_ids = [t["id"] for t in s.data["tasks"].values() if not t["done"]][:2]
    r = execute_turn(d, s, Turn("assistant", "", [ToolCall("complete_tasks", {"task_ids": open_ids}),
                                                  ToolCall("send_email", {"to": ["x@example.com"], "subject": "hi", "body": "there"}),
                                                  ToolCall("delete_event", {"event_id": "nope"})]), turn_idx=2, capability_refs=False)
    assert r.errors == {"3": "not_found"}
    assert all(s.data["tasks"][i]["done"] for i in open_ids)
    assert any(m["folder"] == "sent" for m in s.data["emails"].values())
    assert d.tools["complete_tasks"].spec.batch_of == "complete_task"


def test_relative_dates():
    mon = dt.date(2026, 3, 2)  # Monday
    assert resolve_relative_date(mon, "tomorrow") == dt.date(2026, 3, 3)
    assert resolve_relative_date(mon, "friday") == dt.date(2026, 3, 6)
    assert resolve_relative_date(mon, "next monday") == dt.date(2026, 3, 9)
    assert resolve_relative_date(mon, "in 3 days") == dt.date(2026, 3, 5)
    assert resolve_relative_date(mon, "2026-04-01") == dt.date(2026, 4, 1)
