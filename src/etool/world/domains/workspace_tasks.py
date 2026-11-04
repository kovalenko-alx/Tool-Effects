"""Task generators for the workspace domain: each returns (Task, optimal Plan, naive Plan).

Naive plans are *correct* but inefficient: per-item calls instead of batch tools, redundant re-reads after writes,
duplicate reads, lookups of values the user already gave, one call per turn. The optimizer should map each naive
plan back to the optimal one; the diff labels the negative spans.
"""
from __future__ import annotations

import datetime as _dt
import random
from typing import Any, Callable

from etool.world.base import WorldState
from etool.world.domains.workspace import WEEKDAYS, Workspace, resolve_relative_date
from etool.world.plan import Plan, R, Stage, Step
from etool.world.tasks import Task

Gen = Callable[[Workspace, WorldState, random.Random, str], tuple[Task, Plan, Plan] | None]
GENERATORS: dict[str, Gen] = {}


def gen(kind: str):
    def deco(fn: Gen) -> Gen:
        GENERATORS[kind] = fn
        return fn
    return deco


def _first(name: str) -> str:
    return name.split()[0]


def _ref_name(contact: dict[str, Any], state: WorldState, rng: random.Random, p_first: float = 0.6) -> str:
    """How the user refers to a contact: first name when it is unique among contacts, else the full name."""
    first = _first(contact["name"])
    unique = sum(1 for c in state.data["contacts"].values() if _first(c["name"]) == first) == 1
    return first if unique and rng.random() < p_first else contact["name"]


def _day_expr(rng: random.Random, today: _dt.date, target: _dt.date) -> str:
    """A natural expression for ``target``; sometimes the ISO date itself."""
    delta = (target - today).days
    opts = [target.isoformat()]
    if delta == 0:
        opts += ["today"]
    elif delta == 1:
        opts += ["tomorrow"]
    elif delta == 2:
        opts += ["the day after tomorrow"]
    if 0 < delta < 7:
        opts += [WEEKDAYS[target.weekday()], "on " + WEEKDAYS[target.weekday()]]
    return rng.choice(opts)


def _is_iso(expr: str) -> bool:
    try:
        _dt.date.fromisoformat(expr)
        return True
    except ValueError:
        return False


def _date_steps(expr: str, naive: bool) -> tuple[list[Step], Any]:
    """Steps that produce the target date, and how later steps refer to it.

    Optimal: no call at all when the user already gave an ISO date (the literal is used); otherwise ``resolve_date``.
    Naive: always calls ``resolve_date`` (a lookup of a known value for ISO input) and a dead ``get_current_date``.
    """
    if not naive and _is_iso(expr):
        return [], expr
    steps = []
    if naive:
        steps.append(Step("today", "get_current_date", {}, needed=False))
    steps.append(Step("d", "resolve_date", {"expression": expr}))
    return steps, R("d", ".date")


def _end_time(start: str, minutes: int) -> str:
    h, m = map(int, start.split(":"))
    t = h * 60 + m + minutes
    return f"{t // 60:02d}:{t % 60:02d}"


def _fmt_events(evs: list[dict[str, Any]]) -> str:
    return "; ".join(f"{e['title']} {e['start']}-{e['end']}" + (f" ({e['location']})" if e.get("location") else "") for e in evs)


# ----------------------------------------------------------------------------- read-only kinds
@gen("whats_on")
def whats_on(dom, state, rng, tid):
    today = _dt.date.fromisoformat(state.data["today"])
    target = today + _dt.timedelta(days=rng.randrange(0, 5))
    expr = _day_expr(rng, today, target)
    evs = sorted((e for e in state.data["events"].values() if e["date"] == target.isoformat()), key=lambda e: e["start"])
    utt = rng.choice([f"What's on my calendar {expr}?", f"What do I have scheduled {expr}?", f"Show me my meetings {expr}."])
    facts = [e["title"] for e in evs] or ["<none>"]

    date_iso = target.isoformat()

    def answer(res):
        ev = res["list"]["events"]
        return f"On {date_iso} you have {len(ev)} event(s): {_fmt_events(ev)}." if ev else f"You have nothing scheduled on {date_iso}."

    osteps, odate = _date_steps(expr, False)
    opt = Plan([Stage(osteps + [Step("list", "list_events", {"date": odate})])], answer)
    nsteps, ndate = _date_steps(expr, True)
    naive_steps = nsteps + [Step("list", "list_events", {"date": ndate})]
    if rng.random() < 0.5:
        naive_steps.append(Step("list2", "list_events", {"date": ndate}, needed=False))  # duplicate read
    nai = Plan([Stage(naive_steps, one_per_turn=True)], answer)
    task = Task(tid, "workspace", "whats_on", 0, utt,
                slots={"expr": expr, "date": target.isoformat()}, goal=[{"type": "no_change"}], answer_facts=facts)
    return task, opt, nai


@gen("unread_from")
def unread_from(dom, state, rng, tid):
    senders = {m["from"] for m in state.data["emails"].values() if m["folder"] == "inbox"}
    contact = rng.choice([c for c in state.data["contacts"].values() if c["email"] in senders] or list(state.data["contacts"].values()))
    first = _ref_name(contact, state, rng, p_first=1.0)
    unread = [m for m in state.data["emails"].values() if m["folder"] == "inbox" and m["from"] == contact["email"] and not m["read"]]
    utt = rng.choice([f"Do I have any unread emails from {first}?", f"Anything unread from {contact['name']}?", f"Check if {first} sent me anything I haven't read yet."])
    facts = [m["subject"] for m in unread] or ["<none>"]

    def answer(res):
        ms = res["mail"]["emails"]
        return (f"You have {len(ms)} unread email(s) from {first}: " + "; ".join(f"'{m['subject']}' ({m['date']})" for m in ms) + ".") if ms \
            else f"You have no unread emails from {first}."

    mail = Step("mail", "list_emails", {"folder": "inbox", "unread_only": True, "sender": R("who", ".contacts[0].email")})
    opt = Plan([Stage([Step("who", "search_contacts", {"query": first}), mail])], answer)
    nsteps = [Step("who", "search_contacts", {"query": first}), Step("c", "get_contact", {"contact_id": R("who", ".contacts[0].id")}, needed=False), mail]
    if rng.random() < 0.5:
        nsteps.append(Step("mail2", "list_emails", {"folder": "inbox", "unread_only": True, "sender": R("who", ".contacts[0].email")}, needed=False))
    nai = Plan([Stage(nsteps, one_per_turn=True)], answer)
    task = Task(tid, "workspace", "unread_from", 0, utt, slots={"contact": contact["name"]}, goal=[{"type": "no_change"}], answer_facts=facts)
    return task, opt, nai


# ----------------------------------------------------------------------------- write kinds
@gen("schedule_meeting")
def schedule_meeting(dom, state, rng, tid):
    today = _dt.date.fromisoformat(state.data["today"])
    target = today + _dt.timedelta(days=rng.randrange(1, 6))
    expr = _day_expr(rng, today, target)
    contact = rng.choice(list(state.data["contacts"].values()))
    start = rng.choice(["09:00", "10:00", "11:00", "13:30", "14:00", "15:00", "16:30"])
    dur = rng.choice([30, 45, 60])
    topic = rng.choice(["roadmap", "budget review", "hiring plan", "launch checklist", "design review", "vendor contract"]).title()
    missing = rng.random() < 0.25
    name = _ref_name(contact, state, rng, p_first=0.4)
    if missing:
        utt = rng.choice([f"Set up a {dur}-minute meeting with {name} {expr} about {topic}.", f"Book {name} for a {dur} min {topic} meeting {expr}."])
    else:
        utt = rng.choice([f"Set up a {dur}-minute meeting with {name} {expr} at {start} about {topic}.",
                          f"Schedule '{topic}' with {name} {expr}, {start}, {dur} minutes.",
                          f"Put a {dur} min {topic} meeting with {name} on my calendar {expr} at {start}."])
    end = _end_time(start, dur)

    def answer(res):
        e = res["new"]["created"]
        return f"Done. I scheduled '{e['title']}' on {e['date']} from {e['start']} to {e['end']} with {contact['name']} ({contact['email']})."

    ask = (f"Sure. What time should the {topic} meeting start?", f"At {start}.") if missing else None
    osteps, odate = _date_steps(expr, False)
    create_o = Step("new", "create_event", {"title": topic, "date": odate, "start": start, "end": end, "attendees": [R("who", ".contacts[0].email")]})
    opt = Plan([Stage(osteps + [Step("who", "search_contacts", {"query": name}), create_o], ask=ask)], answer)
    nsteps, ndate = _date_steps(expr, True)
    create_n = Step("new", "create_event", {"title": topic, "date": ndate, "start": start, "end": end, "attendees": [R("who", ".contacts[0].email")]})
    nsteps = nsteps + [Step("who", "search_contacts", {"query": name}), Step("c", "get_contact", {"contact_id": R("who", ".contacts[0].id")}, needed=False)]
    nai = Plan([Stage(nsteps, ask=ask, one_per_turn=True),
                Stage([create_n, Step("recheck", "list_events", {"date": ndate}, needed=False)], one_per_turn=True)], answer)
    goal = [{"type": "event_created", "title": topic, "date": target.isoformat(), "start": start, "end": end, "attendees": [contact["email"]]}]
    task = Task(tid, "workspace", "schedule_meeting", 0, utt, slots={"contact": contact["name"], "expr": expr, "start": start, "dur": dur, "topic": topic},
                goal=goal, missing_info={"question_hint": "time", "answer": f"At {start}."} if missing else None)
    return task, opt, nai


@gen("task_from_email")
def task_from_email(dom, state, rng, tid):
    today = _dt.date.fromisoformat(state.data["today"])
    mail = rng.choice([m for m in state.data["emails"].values() if m["folder"] == "inbox"])
    sender = next(c for c in state.data["contacts"].values() if c["email"] == mail["from"])
    first = _ref_name(sender, state, rng, p_first=1.0)
    target = today + _dt.timedelta(days=rng.randrange(1, 8))
    expr = _day_expr(rng, today, target)
    frag = mail["subject"].split(":")[-1].strip()
    utt = rng.choice([f"Create a task to follow up on {first}'s email about '{frag}', due {expr}.",
                      f"Add a to-do due {expr}: follow up on the email from {first} regarding {frag}."])
    title = f"Follow up: {mail['subject']}"

    def answer(res):
        t = res["task"]["created"]
        found = next((m for m in res["inbox"]["emails"] if m["id"] == mail["id"]), None)
        ref = f", for {first}'s email of {found['date']}" if found else ""
        return f"Created task '{t['title']}' due {t['due']}{ref}."

    # The agent reads the inbox, finds the email by sender + fragment, then creates the task with the exact subject.
    osteps, odate = _date_steps(expr, False)
    create_o = Step("task", "create_task", {"title": title, "due": odate})
    if frag.lower() == mail["subject"].lower():  # the user quoted the full subject: the title needs no inbox lookup first
        opt = Plan([Stage(osteps + [Step("inbox", "list_emails", {"folder": "inbox"}), create_o])], answer)
    else:  # the exact subject is only known after reading the inbox -> barrier before create_task
        opt = Plan([Stage(osteps + [Step("inbox", "list_emails", {"folder": "inbox"})]), Stage([create_o])], answer)
    nsteps, ndate = _date_steps(expr, True)
    create_n = Step("task", "create_task", {"title": title, "due": ndate})
    nsteps = nsteps + [Step("inbox", "list_emails", {"folder": "inbox"}), Step("full", "get_email", {"email_id": mail["id"]}, needed=False)]
    nai = Plan([Stage(nsteps, one_per_turn=True), Stage([create_n, Step("recheck", "list_tasks", {}, needed=False)], one_per_turn=True)], answer)
    task = Task(tid, "workspace", "task_from_email", 0, utt, slots={"sender": sender["name"], "subject": mail["subject"], "expr": expr},
                goal=[{"type": "task_created", "title_contains": frag, "due": target.isoformat()}])
    return task, opt, nai


@gen("complete_assignee_tasks")
def complete_assignee_tasks(dom, state, rng, tid):
    by_assignee: dict[str, list[dict]] = {}
    for t in state.data["tasks"].values():
        if not t["done"] and t["assignee"]:
            by_assignee.setdefault(t["assignee"], []).append(t)
    cands = [c for c in state.data["contacts"].values() if c["email"] in by_assignee]
    if not cands:
        return None
    contact = rng.choice(cands)
    ids = [t["id"] for t in by_assignee[contact["email"]]]
    name = _ref_name(contact, state, rng, p_first=0.5)
    utt = rng.choice([f"Mark all of {name}'s open tasks as done.", f"Close every open task assigned to {name}.", f"{name} finished everything; complete all their tasks."])

    def answer(res):
        n = len(res["done"]["updated"]) if "updated" in res["done"] else len(ids)
        return f"Marked {n} task(s) assigned to {contact['name']} as done."

    opt = Plan([Stage([Step("who", "search_contacts", {"query": name}), Step("ts", "list_tasks", {"assignee": R("who", ".contacts[0].email")}),
                       Step("done", "complete_tasks", {"task_ids": R("ts", ".tasks[*].id")})])], answer)
    per_item = [Step(f"done{k}", "complete_task", {"task_id": R("ts", f".tasks[{k}].id")}) for k in range(len(ids))]
    nai = Plan([Stage([Step("who", "search_contacts", {"query": name}), Step("c", "get_contact", {"contact_id": R("who", ".contacts[0].id")}, needed=False)], one_per_turn=True),
                Stage([Step("ts", "list_tasks", {"assignee": R("who", ".contacts[0].email")})]),
                Stage(per_item + [Step("recheck", "list_tasks", {"assignee": R("who", ".contacts[0].email")}, needed=False)], one_per_turn=True)],
               lambda res: f"Marked {len(ids)} task(s) assigned to {contact['name']} as done.")
    task = Task(tid, "workspace", "complete_assignee_tasks", 0, utt, slots={"contact": contact["name"], "n": len(ids)}, goal=[{"type": "tasks_done", "ids": ids}])
    return task, opt, nai


# ----------------------------------------------------------------------------- destructive kinds (need confirmation)
@gen("cancel_event")
def cancel_event(dom, state, rng, tid):
    today = _dt.date.fromisoformat(state.data["today"])
    titles: dict[str, list[dict]] = {}
    for e in state.data["events"].values():
        titles.setdefault(e["title"], []).append(e)
    uniq = [es[0] for es in titles.values() if len(es) == 1 and _dt.date.fromisoformat(es[0]["date"]) >= today]
    if not uniq:
        return None
    ev = rng.choice(uniq)
    target = _dt.date.fromisoformat(ev["date"])
    expr = _day_expr(rng, today, target)
    utt = rng.choice([f"Cancel the {ev['title']} meeting {expr}.", f"Please delete '{ev['title']}' from my calendar ({expr}).", f"Drop the {ev['title']} {expr}, it's no longer needed."])
    confirm = f"I'm about to permanently delete '{ev['title']}' on {ev['date']} at {ev['start']}. Shall I proceed?"
    answer = lambda res: f"Deleted '{ev['title']}' on {ev['date']}."
    opt = Plan([Stage([Step("find", "find_events", {"query": ev["title"]})]), Stage([Step("del", "delete_event", {"event_id": R("find", ".events[0].id")})], confirm=confirm)], answer)
    nsteps = _date_steps(expr, True)[0] + [Step("find", "find_events", {"query": ev["title"]}), Step("ev", "get_event", {"event_id": R("find", ".events[0].id")}, needed=False)]
    nai = Plan([Stage(nsteps, one_per_turn=True), Stage([Step("del", "delete_event", {"event_id": R("find", ".events[0].id")}),
                                                           Step("recheck", "list_events", {"date": ev["date"]}, needed=False)], confirm=confirm, one_per_turn=True)], answer)
    task = Task(tid, "workspace", "cancel_event", 0, utt, slots={"title": ev["title"], "expr": expr}, goal=[{"type": "event_deleted", "id": ev["id"]}], requires_confirmation=True)
    return task, opt, nai


@gen("move_event_and_notify")
def move_event_and_notify(dom, state, rng, tid):
    today = _dt.date.fromisoformat(state.data["today"])
    titles: dict[str, list[dict]] = {}
    for e in state.data["events"].values():
        titles.setdefault(e["title"], []).append(e)
    uniq = [es[0] for es in titles.values() if len(es) == 1 and es[0]["attendees"] and _dt.date.fromisoformat(es[0]["date"]) >= today]
    if not uniq:
        return None
    ev = rng.choice(uniq)
    new_start = rng.choice([t for t in ["09:00", "10:00", "11:00", "14:00", "15:00", "16:00"] if t != ev["start"]])
    new_end = _end_time(new_start, 60)
    utt = rng.choice([f"Move the {ev['title']} to {new_start} and let the attendees know by email.",
                      f"Shift '{ev['title']}' to {new_start}; email everyone who's invited about the new time."])
    confirm = (f"I'll move '{ev['title']}' on {ev['date']} to {new_start}-{new_end} and email {len(ev['attendees'])} attendee(s) about the change. "
               f"Shall I proceed?")
    subject = f"Rescheduled: {ev['title']}"
    body = f"Hi all, '{ev['title']}' on {ev['date']} now starts at {new_start} (until {new_end}). Thanks!"
    answer = lambda res: f"Moved '{ev['title']}' to {new_start}-{new_end} on {ev['date']} and emailed the attendees."
    upd = Step("upd", "update_event", {"event_id": R("find", ".events[0].id"), "start": new_start, "end": new_end})
    mail = Step("mail", "send_email", {"to": R("find", ".events[0].attendees"), "subject": subject, "body": body})
    opt = Plan([Stage([Step("find", "find_events", {"query": ev["title"]})]), Stage([upd, mail], confirm=confirm)], answer)
    nai = Plan([Stage([Step("find", "find_events", {"query": ev["title"]}), Step("ev", "get_event", {"event_id": R("find", ".events[0].id")}, needed=False)], one_per_turn=True),
                Stage([upd, Step("ev2", "get_event", {"event_id": R("find", ".events[0].id")}, needed=False), mail], confirm=confirm, one_per_turn=True)], answer)
    goal = [{"type": "event_fields", "id": ev["id"], "start": new_start, "end": new_end},
            {"type": "email_sent", "to": sorted(ev["attendees"]), "body_contains": [new_start]}]
    task = Task(tid, "workspace", "move_event_and_notify", 0, utt, slots={"title": ev["title"], "new_start": new_start}, goal=goal, requires_confirmation=True)
    return task, opt, nai


@gen("reply_email")
def reply_email(dom, state, rng, tid):
    mail = rng.choice([m for m in state.data["emails"].values() if m["folder"] == "inbox"])
    sender = next(c for c in state.data["contacts"].values() if c["email"] == mail["from"])
    first = _ref_name(sender, state, rng, p_first=1.0)
    frag = mail["subject"].split(":")[-1].strip()
    text = rng.choice(["Sounds good, let's do that.", "I'll have it ready by Thursday.", "Can we discuss this on Monday?", "Approved, go ahead."])
    utt = rng.choice([f"Reply to {first}'s email about '{frag}' saying: {text}", f"Answer {sender['name']}'s message regarding {frag} with \"{text}\""])
    # only information the agent has actually seen: the user's words and the inbox listing (sender address, subject)
    confirm = f"I'll reply to {first} ({sender['email']}) with subject 'Re: {mail['subject']}' saying \"{text}\". Shall I send it?"
    answer = lambda res: f"Sent the reply to {first} ({sender['email']})."
    # the agent lists the inbox to find the message; the plan then uses the sender address it found there
    send = Step("send", "send_email", {"to": [sender["email"]], "subject": f"Re: {mail['subject']}", "body": text})
    opt = Plan([Stage([Step("inbox", "list_emails", {"folder": "inbox"})]), Stage([send], confirm=confirm)], answer)
    nai = Plan([Stage([Step("inbox", "list_emails", {"folder": "inbox"}), Step("full", "get_email", {"email_id": mail["id"]}, needed=False),
                       Step("who", "search_contacts", {"query": first}, needed=False)], one_per_turn=True),
                Stage([send, Step("recheck", "list_emails", {"folder": "sent"}, needed=False)], confirm=confirm, one_per_turn=True)], answer)
    task = Task(tid, "workspace", "reply_email", 0, utt, slots={"sender": sender["name"], "subject": mail["subject"], "text": text},
                goal=[{"type": "email_sent", "to": [sender["email"]], "body_contains": [text.rstrip('.')]}], requires_confirmation=True)
    return task, opt, nai


# ----------------------------------------------------------------------------- no-tool kind
@gen("irrelevant")
def irrelevant(dom, state, rng, tid):
    # v2 (2026-10-21): no arithmetic here — "how many minutes in three hours" answered without a tool taught S1 to answer BFCL
    # calculation prompts in prose instead of calling the provided calculator tool. Keep only requests no tool could serve.
    utt, ans = rng.choice([
        ("What's a good name for a grey cat?", "How about Smokey, Ash or Pebble?"),
        ("Thanks, that's all for now.", "You're welcome! Let me know if you need anything else."),
        ("Any tips for writing a clear meeting agenda?", "Keep it to three to five items, give each a time box and an owner, and share it a day ahead."),
        ("Can you remind me what a calendar invite is?", "A calendar invite is an event sent to attendees so it appears on their calendars; they can accept or decline it."),
        ("Write a one-line birthday wish for a colleague.", "Happy birthday! Wishing you a great year ahead."),
    ])
    plan = Plan([], lambda res: ans)
    task = Task(tid, "workspace", "irrelevant", 0, utt, goal=[{"type": "no_change"}], irrelevant=True)
    return task, plan, plan


# ----------------------------------------------------------------------------- missing-capability kind (v3a, 2026-10-22)
# A normal request whose required tool is *absent* from the tool list: the right move is to decline, naming the missing
# capability, without calling anything. Teaches the discrimination BFCL's irrelevance categories test (the request is fine, the
# available functions cannot serve it) as a positive, the counterpart of the ``refusal`` negatives (tool present -> call).
MISSING_CAPABILITY: dict[str, tuple[list[str], str]] = {
    "schedule_meeting": (["create_event"], "create calendar events"),
    "cancel_event": (["delete_event"], "delete calendar events"),
    "reply_email": (["send_email"], "send emails"),
    "task_from_email": (["create_task"], "create tasks"),
    "complete_assignee_tasks": (["complete_task", "complete_tasks"], "mark tasks as done"),
    "move_event_and_notify": (["update_event"], "change existing calendar events"),
}


@gen("missing_capability")
def missing_capability(dom, state, rng, tid):
    base_kind = rng.choice(sorted(MISSING_CAPABILITY))
    out = GENERATORS[base_kind](dom, state, rng, tid)
    if out is None:
        return None
    base_task, _, _ = out
    removed, phrase = MISSING_CAPABILITY[base_kind]
    ans = rng.choice([
        f"Sorry, that is not available here: none of the tools I have can {phrase}.",
        f"I can't do that with the tools available to me - there is no tool to {phrase}, so it is not available.",
        f"That action is not available: the tools provided here do not include a way to {phrase}.",
    ])
    plan = Plan([], lambda res: ans)
    task = Task(tid, "workspace", "missing_capability", 0, base_task.utterance, slots={"base_kind": base_kind, "removed": list(removed)},
                goal=[{"type": "no_change"}], answer_facts=["<none>"], irrelevant=True, tools_excluded=list(removed))
    return task, plan, plan


def generate(dom: Workspace, seed: int, kinds: list[str] | None = None, tid_prefix: str = "ws") -> tuple[Task, Plan, Plan] | None:
    """Deterministically generate one task for ``seed`` (kind chosen by the seed unless given)."""
    rng = random.Random(seed * 7919 + 17)
    state = dom.new_state(seed)
    kinds = kinds or list(GENERATORS)
    kind = kinds[seed % len(kinds)]
    out = GENERATORS[kind](dom, state, rng, f"{tid_prefix}-{seed}-{kind}")
    if out is None:
        return None
    task, opt, nai = out
    task.seed = seed
    task.min_calls = opt.n_calls()
    task.min_turns_prog = sum(1 for st in opt.stages if st.steps)
    task.min_turns_step = opt.n_calls()
    return task, opt, nai
