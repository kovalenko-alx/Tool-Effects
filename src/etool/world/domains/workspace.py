"""Workspace domain: contacts, calendar, email, tasks. 16 tools with effects, footprints and batch variants.

State layout (all ids are short strings)::

    today: "YYYY-MM-DD"
    me: {"name", "email"}
    contacts: {id: {id, name, email, team}}
    events:   {id: {id, title, date, start, end, attendees: [email], location}}
    emails:   {id: {id, folder: inbox|sent, from, to: [email], subject, body, date, read: bool}}
    tasks:    {id: {id, title, due: date|None, assignee: email|None, done: bool}}
"""
from __future__ import annotations

import datetime as _dt
import random
from typing import Any

from etool.schema import ToolSpec
from etool.world.base import DEFAULT_POLICIES, Domain, ToolError, WorldState

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
FIRST = ["Ann", "Ben", "Carla", "Dana", "Eli", "Farah", "Gus", "Hana", "Ivan", "Jules", "Kim", "Lior", "Mona", "Nils", "Omar", "Pia"]
LAST = ["Novak", "Svoboda", "Dvorak", "Cerny", "Prochazka", "Kucera", "Vesely", "Horak", "Nemec", "Marek", "Pokorny", "Pospisil"]
TEAMS = ["engineering", "design", "sales", "finance"]
TOPICS = ["roadmap", "budget review", "hiring plan", "launch checklist", "quarterly report", "customer feedback",
          "security audit", "onboarding", "retrospective", "vendor contract", "offsite planning", "design review"]
LOCATIONS = ["Room A", "Room B", "Zoom", "Cafeteria", "Room 3.14", None]
SUBJECTS = ["Invoice #{n} overdue", "Re: {topic} notes", "Draft agenda for {topic}", "Question about {topic}",
            "Reminder: {topic} tomorrow", "Slides for the {topic}", "Lunch on Friday?", "Access request: {topic} repo"]
BODIES = ["Hi, could you have a look at this before the end of the week? Thanks!",
          "Attached are my notes from today. Let me know if I missed anything.",
          "Can we move our discussion to a later slot? Something came up.",
          "Please confirm the numbers below are final.",
          "Short version: we are on track. Longer version in the doc."]


def _obj(props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": required}


def _s(desc: str, **kw: Any) -> dict[str, Any]:
    return {"type": "string", "description": desc, **kw}


def _parse_date(s: str) -> _dt.date:
    try:
        return _dt.date.fromisoformat(s)
    except ValueError:
        raise ToolError("invalid_date", f"'{s}' is not a YYYY-MM-DD date")


def _parse_time(s: str) -> _dt.time:
    try:
        h, m = s.split(":")
        return _dt.time(int(h), int(m))
    except Exception:
        raise ToolError("invalid_time", f"'{s}' is not an HH:MM time")


def resolve_relative_date(today: _dt.date, expr: str) -> _dt.date:
    e = expr.strip().lower()
    if e in ("today", "now"):
        return today
    if e == "tomorrow":
        return today + _dt.timedelta(days=1)
    if e in ("day after tomorrow", "the day after tomorrow"):
        return today + _dt.timedelta(days=2)
    if e == "yesterday":
        return today - _dt.timedelta(days=1)
    if e.startswith("in ") and e.endswith(" days"):
        return today + _dt.timedelta(days=int(e[3:-5]))
    for prefix in ("next ", "this ", "on ", ""):
        if e.startswith(prefix) and e[len(prefix):] in WEEKDAYS:
            target = WEEKDAYS.index(e[len(prefix):])
            delta = (target - today.weekday()) % 7
            if delta == 0 or prefix == "next ":
                delta = delta or 7
            return today + _dt.timedelta(days=delta)
    try:
        return _dt.date.fromisoformat(e)
    except ValueError:
        raise ToolError("unknown_expression", f"cannot resolve date expression '{expr}'")


class Workspace(Domain):
    name = "workspace"
    description = "the user's personal workspace: contacts, calendar, email and tasks"
    policies = DEFAULT_POLICIES

    def __init__(self) -> None:
        super().__init__()
        add = self.add
        # --- pure
        add(ToolSpec("get_current_date", "Return today's date and weekday.", _obj({}, []), effect="pure", idempotent=True,
                     latency_ms=1, returns={"type": "object"}), lambda s, a: {"date": s.data["today"], "weekday": WEEKDAYS[_parse_date(s.data["today"]).weekday()]})
        add(ToolSpec("resolve_date", "Convert a relative expression such as 'tomorrow', 'next friday' or 'in 3 days' into a YYYY-MM-DD date.",
                     _obj({"expression": _s("Relative date expression or ISO date")}, ["expression"]), effect="pure", idempotent=True, latency_ms=1),
            lambda s, a: {"date": resolve_relative_date(_parse_date(s.data["today"]), a["expression"]).isoformat()})
        # --- contacts
        add(ToolSpec("search_contacts", "Find contacts whose name, email or team contains the query (case-insensitive).",
                     _obj({"query": _s("Name, email fragment or team")}, ["query"]), effect="read", idempotent=True,
                     reads=["contacts:*"], latency_ms=60), self._search_contacts)
        add(ToolSpec("get_contact", "Get one contact by id.", _obj({"contact_id": _s("Contact id")}, ["contact_id"]), effect="read",
                     idempotent=True, reads=["contact:{contact_id}"], latency_ms=40), self._get_contact)
        add(ToolSpec("get_contacts", "Get several contacts by id in one call.", _obj({"contact_ids": {"type": "array", "items": {"type": "string"}}}, ["contact_ids"]),
                     effect="read", idempotent=True, reads=["contact:*"], latency_ms=50, batch_of="get_contact"), self._get_contacts)
        # --- calendar
        add(ToolSpec("list_events", "List calendar events on a given date.", _obj({"date": _s("YYYY-MM-DD")}, ["date"]), effect="read",
                     idempotent=True, reads=["event:*"], latency_ms=80), self._list_events)
        add(ToolSpec("find_events", "Find events whose title contains the query, optionally restricted to a date range.",
                     _obj({"query": _s("Title fragment"), "from_date": _s("YYYY-MM-DD, inclusive"), "to_date": _s("YYYY-MM-DD, inclusive")}, ["query"]),
                     effect="read", idempotent=True, reads=["event:*"], latency_ms=90), self._find_events)
        add(ToolSpec("get_event", "Get one event by id.", _obj({"event_id": _s("Event id")}, ["event_id"]), effect="read", idempotent=True,
                     reads=["event:{event_id}"], latency_ms=40), self._get_event)
        add(ToolSpec("create_event", "Create a calendar event.",
                     _obj({"title": _s("Event title"), "date": _s("YYYY-MM-DD"), "start": _s("HH:MM"), "end": _s("HH:MM"),
                           "attendees": {"type": "array", "items": {"type": "string"}, "description": "Attendee email addresses"},
                           "location": _s("Optional location")}, ["title", "date", "start", "end"]),
                     effect="write", idempotent=False, writes=["event:*"], latency_ms=120), self._create_event)
        add(ToolSpec("update_event", "Change fields of an existing event (only the provided fields are changed).",
                     _obj({"event_id": _s("Event id"), "title": _s("New title"), "date": _s("New YYYY-MM-DD"), "start": _s("New HH:MM"),
                           "end": _s("New HH:MM"), "location": _s("New location"),
                           "add_attendees": {"type": "array", "items": {"type": "string"}, "description": "Emails to add"}}, ["event_id"]),
                     effect="write", idempotent=True, reads=["event:{event_id}"], writes=["event:{event_id}"], latency_ms=100), self._update_event)
        add(ToolSpec("delete_event", "Delete an event permanently.", _obj({"event_id": _s("Event id")}, ["event_id"]), effect="destructive",
                     idempotent=True, reads=["event:{event_id}"], writes=["event:{event_id}"], latency_ms=90), self._delete_event)
        # --- email
        add(ToolSpec("list_emails", "List emails in a folder, newest first; optionally only unread and/or from one sender address.",
                     _obj({"folder": _s("inbox or sent", enum=["inbox", "sent"]), "unread_only": {"type": "boolean"},
                           "sender": _s("Only emails from this address"), "limit": {"type": "integer"}}, []),
                     effect="read", idempotent=True, reads=["email:*"], latency_ms=90), self._list_emails)
        add(ToolSpec("get_email", "Get the full email (including body) by id.", _obj({"email_id": _s("Email id")}, ["email_id"]), effect="read",
                     idempotent=True, reads=["email:{email_id}"], latency_ms=40), self._get_email)
        add(ToolSpec("send_email", "Send an email from the user's account. This cannot be undone.",
                     _obj({"to": {"type": "array", "items": {"type": "string"}, "description": "Recipient emails"}, "subject": _s("Subject"), "body": _s("Body")},
                          ["to", "subject", "body"]),
                     effect="destructive", idempotent=False, writes=["email:*"], latency_ms=150), self._send_email)
        # --- tasks
        add(ToolSpec("list_tasks", "List tasks, optionally filtered by assignee email; open tasks only unless include_done.",
                     _obj({"assignee": _s("Assignee email"), "include_done": {"type": "boolean"}}, []), effect="read", idempotent=True,
                     reads=["task:*"], latency_ms=70), self._list_tasks)
        add(ToolSpec("create_task", "Create a task.", _obj({"title": _s("Task title"), "due": _s("YYYY-MM-DD"), "assignee": _s("Assignee email")}, ["title"]),
                     effect="write", idempotent=False, writes=["task:*"], latency_ms=100), self._create_task)
        add(ToolSpec("complete_task", "Mark one task as done.", _obj({"task_id": _s("Task id")}, ["task_id"]), effect="write", idempotent=True,
                     reads=["task:{task_id}"], writes=["task:{task_id}"], latency_ms=80), self._complete_task)
        add(ToolSpec("complete_tasks", "Mark several tasks as done in one call.", _obj({"task_ids": {"type": "array", "items": {"type": "string"}}}, ["task_ids"]),
                     effect="write", idempotent=True, reads=["task:*"], writes=["task:*"], latency_ms=90, batch_of="complete_task"), self._complete_tasks)

    def fold(self, tool: str, args: dict[str, Any]) -> Any:
        if tool == "resolve_date":
            try:
                return {"date": _dt.date.fromisoformat(str(args.get("expression", "")).strip()).isoformat()}
            except ValueError:
                return None
        return None

    # ------------------------------------------------------------------ state
    def new_state(self, seed: int) -> WorldState:
        rng = random.Random(seed)
        today = _dt.date(2026, 3, 2) + _dt.timedelta(days=rng.randrange(0, 60))  # a Monday + offset
        me = {"name": "Alex Kovar", "email": "alex.kovar@example.com"}
        names = rng.sample([f"{f} {l}" for f in FIRST for l in LAST], 9)
        contacts = {}
        for i, n in enumerate(names):
            cid = f"c{i+1}"
            contacts[cid] = {"id": cid, "name": n, "email": n.lower().replace(" ", ".") + "@example.com", "team": rng.choice(TEAMS)}
        emails_of = [c["email"] for c in contacts.values()]
        events = {}
        for i in range(7):
            eid = f"e{i+1}"
            d = today + _dt.timedelta(days=rng.randrange(-1, 6))
            h = rng.choice([9, 10, 11, 13, 14, 15, 16])
            events[eid] = {"id": eid, "title": rng.choice(TOPICS).title(), "date": d.isoformat(), "start": f"{h:02d}:00",
                           "end": f"{h+1:02d}:00", "attendees": rng.sample(emails_of, rng.randrange(1, 4)), "location": rng.choice(LOCATIONS)}
        emails = {}
        for i in range(9):
            mid = f"m{i+1}"
            sender = rng.choice(list(contacts.values()))
            subj = rng.choice(SUBJECTS).format(n=rng.randrange(1000, 9999), topic=rng.choice(TOPICS))
            emails[mid] = {"id": mid, "folder": "inbox", "from": sender["email"], "to": [me["email"]], "subject": subj,
                           "body": rng.choice(BODIES), "date": (today - _dt.timedelta(days=rng.randrange(0, 5))).isoformat(),
                           "read": rng.random() < 0.5}
        tasks = {}
        assignee_pool = emails_of[:4] + [me["email"], None]  # concentrated so that some people own several open tasks
        for i in range(10):
            tid = f"t{i+1}"
            tasks[tid] = {"id": tid, "title": f"{rng.choice(['Prepare', 'Review', 'Send', 'Update'])} {rng.choice(TOPICS)}",
                          "due": (today + _dt.timedelta(days=rng.randrange(0, 10))).isoformat() if rng.random() < 0.8 else None,
                          "assignee": rng.choice(assignee_pool), "done": rng.random() < 0.3}
        return WorldState({"today": today.isoformat(), "me": me, "contacts": contacts, "events": events, "emails": emails, "tasks": tasks,
                           "counters": {"e": 7, "m": 9, "t": 10}}, seed)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _next_id(s: WorldState, kind: str) -> str:
        s.data["counters"][kind] += 1
        return f"{kind}{s.data['counters'][kind]}"

    @staticmethod
    def _public_event(e: dict[str, Any]) -> dict[str, Any]:
        return dict(e)

    @staticmethod
    def _email_summary(m: dict[str, Any]) -> dict[str, Any]:
        return {k: m[k] for k in ("id", "from", "to", "subject", "date", "read")}

    # ------------------------------------------------------------------ tool impls
    def _search_contacts(self, s: WorldState, a: dict[str, Any]) -> Any:
        q = a["query"].lower().strip()
        hits = [c for c in s.data["contacts"].values() if q in c["name"].lower() or q in c["email"].lower() or q == c["team"]]
        return {"contacts": hits}

    def _get_contact(self, s: WorldState, a: dict[str, Any]) -> Any:
        c = s.data["contacts"].get(a["contact_id"])
        if c is None:
            raise ToolError("not_found", f"no contact {a['contact_id']}")
        return c

    def _get_contacts(self, s: WorldState, a: dict[str, Any]) -> Any:
        missing = [i for i in a["contact_ids"] if i not in s.data["contacts"]]
        if missing:
            raise ToolError("not_found", f"no contact(s) {missing}")
        return {"contacts": [s.data["contacts"][i] for i in a["contact_ids"]]}

    def _list_events(self, s: WorldState, a: dict[str, Any]) -> Any:
        _parse_date(a["date"])
        evs = sorted((e for e in s.data["events"].values() if e["date"] == a["date"]), key=lambda e: e["start"])
        return {"date": a["date"], "events": evs}

    def _find_events(self, s: WorldState, a: dict[str, Any]) -> Any:
        q = a["query"].lower().strip()
        lo = a.get("from_date") or "0000-00-00"
        hi = a.get("to_date") or "9999-99-99"
        evs = sorted((e for e in s.data["events"].values() if q in e["title"].lower() and lo <= e["date"] <= hi), key=lambda e: (e["date"], e["start"]))
        return {"events": evs}

    def _get_event(self, s: WorldState, a: dict[str, Any]) -> Any:
        e = s.data["events"].get(a["event_id"])
        if e is None:
            raise ToolError("not_found", f"no event {a['event_id']}")
        return e

    def _create_event(self, s: WorldState, a: dict[str, Any]) -> Any:
        _parse_date(a["date"])
        if _parse_time(a["start"]) >= _parse_time(a["end"]):
            raise ToolError("invalid_time_range", "end must be after start")
        eid = self._next_id(s, "e")
        e = {"id": eid, "title": a["title"], "date": a["date"], "start": a["start"], "end": a["end"],
             "attendees": list(a.get("attendees") or []), "location": a.get("location")}
        s.data["events"][eid] = e
        return {"created": e}

    def _update_event(self, s: WorldState, a: dict[str, Any]) -> Any:
        e = s.data["events"].get(a["event_id"])
        if e is None:
            raise ToolError("not_found", f"no event {a['event_id']}")
        for k in ("title", "date", "start", "end", "location"):
            if a.get(k) is not None:
                if k == "date":
                    _parse_date(a[k])
                if k in ("start", "end"):
                    _parse_time(a[k])
                e[k] = a[k]
        if _parse_time(e["start"]) >= _parse_time(e["end"]):
            raise ToolError("invalid_time_range", "end must be after start")
        for em in a.get("add_attendees") or []:
            if em not in e["attendees"]:
                e["attendees"].append(em)
        return {"updated": e}

    def _delete_event(self, s: WorldState, a: dict[str, Any]) -> Any:
        if a["event_id"] not in s.data["events"]:
            raise ToolError("not_found", f"no event {a['event_id']}")
        del s.data["events"][a["event_id"]]
        return {"deleted": a["event_id"]}

    def _list_emails(self, s: WorldState, a: dict[str, Any]) -> Any:
        folder = a.get("folder") or "inbox"
        ms = [m for m in s.data["emails"].values() if m["folder"] == folder and (not a.get("unread_only") or not m["read"])
              and (not a.get("sender") or m["from"].lower() == str(a["sender"]).lower())]
        ms.sort(key=lambda m: (m["date"], m["id"]), reverse=True)
        return {"emails": [self._email_summary(m) for m in ms[: a.get("limit") or 20]]}

    def _get_email(self, s: WorldState, a: dict[str, Any]) -> Any:
        m = s.data["emails"].get(a["email_id"])
        if m is None:
            raise ToolError("not_found", f"no email {a['email_id']}")
        return m  # reading does not mutate state (keeps footprints honest)

    def _send_email(self, s: WorldState, a: dict[str, Any]) -> Any:
        if not a["to"]:
            raise ToolError("invalid_arguments", "'to' must contain at least one recipient")
        bad = [t for t in a["to"] if "@" not in t]
        if bad:
            raise ToolError("invalid_recipient", f"not an email address: {bad}")
        mid = self._next_id(s, "m")
        m = {"id": mid, "folder": "sent", "from": s.data["me"]["email"], "to": list(a["to"]), "subject": a["subject"], "body": a["body"],
             "date": s.data["today"], "read": True}
        s.data["emails"][mid] = m
        return {"sent": self._email_summary(m)}

    def _list_tasks(self, s: WorldState, a: dict[str, Any]) -> Any:
        ts = [t for t in s.data["tasks"].values() if (a.get("include_done") or not t["done"]) and (not a.get("assignee") or t["assignee"] == a["assignee"])]
        ts.sort(key=lambda t: (t["due"] or "9999", t["id"]))
        return {"tasks": ts}

    def _create_task(self, s: WorldState, a: dict[str, Any]) -> Any:
        if a.get("due"):
            _parse_date(a["due"])
        tid = self._next_id(s, "t")
        t = {"id": tid, "title": a["title"], "due": a.get("due"), "assignee": a.get("assignee"), "done": False}
        s.data["tasks"][tid] = t
        return {"created": t}

    def _complete_task(self, s: WorldState, a: dict[str, Any]) -> Any:
        t = s.data["tasks"].get(a["task_id"])
        if t is None:
            raise ToolError("not_found", f"no task {a['task_id']}")
        t["done"] = True
        return {"updated": t}

    def _complete_tasks(self, s: WorldState, a: dict[str, Any]) -> Any:
        missing = [i for i in a["task_ids"] if i not in s.data["tasks"]]
        if missing:
            raise ToolError("not_found", f"no task(s) {missing}")
        for i in a["task_ids"]:
            s.data["tasks"][i]["done"] = True
        return {"updated": [s.data["tasks"][i] for i in a["task_ids"]]}
