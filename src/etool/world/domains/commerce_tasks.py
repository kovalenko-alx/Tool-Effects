"""Task generators for the commerce domain (test-only): each returns (Task, optimal Plan, same Plan).

Kinds: product_price, compare_prices (parallel reads), order_status, add_to_cart, add_several (batch tool), buy_now
(destructive: pay after confirmation), cancel_order / refund_delivered (destructive after confirmation),
change_address (read-before-write), cancel_shipped (tool would refuse: report, no destructive call), missing_product /
out_of_stock (nothing to do: report, no writes), irrelevant.
"""
from __future__ import annotations

import random
from typing import Any, Callable

from etool.world.base import WorldState
from etool.world.domains.commerce import Commerce
from etool.world.plan import Plan, R, Stage, Step
from etool.world.tasks import Task

Gen = Callable[[Commerce, WorldState, random.Random, str], tuple[Task, Plan, Plan] | None]
GENERATORS: dict[str, Gen] = {}

FRAG = {"Noise-cancelling headphones": "headphones", "USB-C charger 65W": "USB-C charger", "Wireless mouse": "wireless mouse",
        "Mechanical keyboard": "mechanical keyboard", "4K monitor 27-inch": "4K monitor", "Webcam 1080p": "webcam", "Espresso machine": "espresso machine",
        "Cast iron skillet": "skillet", "Chef's knife 8-inch": "chef's knife", "Electric kettle": "kettle", "Blender 1200W": "blender",
        "Bamboo cutting board": "cutting board", "Yoga mat": "yoga mat", "Running shoes": "running shoes", "Resistance bands set": "resistance bands",
        "Insulated water bottle": "water bottle", "Jump rope": "jump rope", "Dumbbell pair 5 kg": "dumbbell", "Fantasy novel box set": "fantasy novel",
        "Python programming guide": "Python programming guide", "Travel atlas": "travel atlas", "Cookbook: weeknight dinners": "cookbook",
        "Desk lamp LED": "desk lamp", "Standing desk converter": "standing desk"}


def gen(kind: str):
    def deco(fn: Gen) -> Gen:
        GENERATORS[kind] = fn
        return fn
    return deco


def _frag(p: dict[str, Any]) -> str:
    return FRAG[p["name"]]


def _price(x: float) -> str:
    return f"{x:.2f}".rstrip("0").rstrip(".")


def _products(state: WorldState, min_stock: int = 1) -> list[dict[str, Any]]:
    return [p for p in state.data["products"].values() if p["stock"] >= min_stock]


def _orders(state: WorldState, statuses: tuple[str, ...]) -> list[dict[str, Any]]:
    return [o for o in state.data["orders"].values() if o["status"] in statuses]


def _listing_index(state: WorldState, oid: str) -> int:
    """Position of an order in list_orders() output (newest first)."""
    os_ = sorted(state.data["orders"].values(), key=lambda o: (o["date"], o["id"]), reverse=True)
    return next(i for i, o in enumerate(os_) if o["id"] == oid)


def _oname(state: WorldState, o: dict[str, Any]) -> dict[str, Any]:
    return state.data["products"][o["items"][0]["product_id"]]


# ----------------------------------------------------------------------------- read-only
@gen("product_price")
def product_price(dom, state, rng, tid):
    p = rng.choice(list(state.data["products"].values()))
    frag = _frag(p)
    utt = rng.choice([f"How much is the {frag}?", f"What does the {frag} cost, and is it in stock?", f"Price check: {frag}."])
    opt = Plan([Stage([Step("s", "search_products", {"query": frag})])],
               lambda res: f"The {p['name']} costs ${_price(p['price'])}; {p['stock']} in stock." if p["stock"] else f"The {p['name']} costs ${_price(p['price'])} but it is currently out of stock.")
    task = Task(tid, dom.name, "product_price", 0, utt, slots={"product": p["name"], "frag": frag}, goal=[{"type": "no_change"}], answer_facts=[_price(p["price"])])
    return task, opt, opt


@gen("compare_prices")
def compare_prices(dom, state, rng, tid):
    a, b = rng.sample(list(state.data["products"].values()), 2)
    if a["price"] == b["price"]:
        return None
    fa, fb = _frag(a), _frag(b)
    utt = rng.choice([f"Which is cheaper, the {fa} or the {fb}?", f"Compare the prices of the {fa} and the {fb}.", f"Is the {fa} more expensive than the {fb}?"])
    cheap, dear = (a, b) if a["price"] < b["price"] else (b, a)
    opt = Plan([Stage([Step("a", "search_products", {"query": fa}), Step("b", "search_products", {"query": fb})])],
               lambda res: f"The {cheap['name']} is cheaper: ${_price(cheap['price'])} vs ${_price(dear['price'])} for the {dear['name']}.")
    task = Task(tid, dom.name, "compare_prices", 0, utt, slots={"a": a["name"], "b": b["name"]}, goal=[{"type": "no_change"}],
                answer_facts=[_price(a["price"]), _price(b["price"])])
    return task, opt, opt


@gen("order_status")
def order_status(dom, state, rng, tid):
    o = rng.choice(list(state.data["orders"].values()))
    p = _oname(state, o)
    frag = _frag(p)
    utt = rng.choice([f"What's the status of my order for the {frag}?", f"Has my {frag} shipped yet?", f"Where is the {frag} I ordered?"])
    opt = Plan([Stage([Step("os", "list_orders", {})])], lambda res: f"Your order {o['id']} for the {p['name']} (placed {o['date']}) is {o['status']}.")
    task = Task(tid, dom.name, "order_status", 0, utt, slots={"product": p["name"], "frag": frag}, goal=[{"type": "no_change"}], answer_facts=[o["status"]])
    return task, opt, opt


# ----------------------------------------------------------------------------- writes
@gen("add_to_cart")
def add_to_cart(dom, state, rng, tid):
    cands = _products(state, 3)
    if not cands:
        return None
    p = rng.choice(cands)
    qty = rng.choice([1, 2, 3])
    frag = _frag(p)
    utt = rng.choice([f"Add {qty} {frag} to my cart.", f"Put {qty} of the {frag} in my basket.", f"I'll take {qty} {frag} - add them to the cart."])
    have = state.data["cart"].get(p["id"], {"qty": 0})["qty"]
    opt = Plan([Stage([Step("s", "search_products", {"query": frag}), Step("add", "add_to_cart", {"product_id": R("s", ".products[0].id"), "qty": qty})])],
               lambda res: f"Added {qty} x {p['name']} to your cart (now {res['add']['cart']['subtotal']} in total).")
    task = Task(tid, dom.name, "add_to_cart", 0, utt, slots={"product": p["name"], "frag": frag, "qty": qty},
                goal=[{"type": "cart_contains", "product_id": p["id"], "qty": have + qty}], answer_facts=[p["name"]])
    return task, opt, opt


@gen("add_several")
def add_several(dom, state, rng, tid):
    cands = _products(state, 1)
    if len(cands) < 3:
        return None
    ps = rng.sample(cands, 3)
    frags = [_frag(p) for p in ps]
    utt = rng.choice([f"Add the {frags[0]}, the {frags[1]} and the {frags[2]} to my cart, one each.",
                      f"Put one {frags[0]}, one {frags[1]} and one {frags[2]} in my basket."])
    steps = [Step(f"s{i}", "search_products", {"query": f}) for i, f in enumerate(frags)]
    batch = Step("add", "add_items_to_cart", {"items": [{"product_id": R(f"s{i}", ".products[0].id"), "qty": 1} for i in range(3)]})
    opt = Plan([Stage(steps + [batch])], lambda res: "Added " + ", ".join(p["name"] for p in ps) + " to your cart.")
    goal = [{"type": "cart_contains", "product_id": p["id"], "qty": state.data["cart"].get(p["id"], {"qty": 0})["qty"] + 1} for p in ps]
    task = Task(tid, dom.name, "add_several", 0, utt, slots={"products": [p["name"] for p in ps]}, goal=goal, answer_facts=[p["name"] for p in ps])
    return task, opt, opt


@gen("change_address")
def change_address(dom, state, rng, tid):
    cands = _orders(state, ("pending", "paid"))
    if not cands:
        return None
    o = rng.choice(cands)
    p = _oname(state, o)
    frag = _frag(p)
    target = "a2" if o["address_id"] == "a1" else "a1"
    label = state.data["addresses"][target]["label"]
    utt = rng.choice([f"Ship my order for the {frag} to my {label} address instead.", f"Change the delivery address of the {frag} order to my {label} address."])
    k = _listing_index(state, o["id"])
    ak = 0 if target == "a1" else 1
    opt = Plan([Stage([Step("os", "list_orders", {}), Step("ad", "list_addresses", {}),
                       Step("upd", "update_order_address", {"order_id": R("os", f".orders[{k}].id"), "address_id": R("ad", f".addresses[{ak}].id")})])],
               lambda res: f"Order {o['id']} ({p['name']}) will now be delivered to your {label} address.")
    task = Task(tid, dom.name, "change_address", 0, utt, slots={"product": p["name"], "label": label}, goal=[{"type": "order_fields", "id": o["id"], "address_id": target}],
                answer_facts=[label])
    return task, opt, opt


# ----------------------------------------------------------------------------- destructive (confirmation)
@gen("buy_now")
def buy_now(dom, state, rng, tid):
    if state.data["cart"]:
        return None  # placing an order ships the whole cart: keep the gold unambiguous
    cands = _products(state, 1)
    if not cands:
        return None
    p = rng.choice(cands)
    frag = _frag(p)
    pm = rng.choice(list(state.data["payment_methods"].values()))
    addr = rng.choice(list(state.data["addresses"].values()))
    utt = rng.choice([f"Order one {frag} with my {pm['label']} and ship it to my {addr['label']} address.",
                      f"Buy a {frag} now, pay with the {pm['label']}, deliver to my {addr['label']} address."])
    confirm = (f"I'll place an order for 1 x {p['name']} (${_price(p['price'])}) paid with your {pm['label']} ending {pm['last4']}, delivered to "
               f"{addr['street']}, {addr['city']}. Shall I place the order?")
    pk = 0 if pm["id"] == "pm1" else 1
    ak = 0 if addr["id"] == "a1" else 1
    opt = Plan([Stage([Step("s", "search_products", {"query": frag}), Step("pm", "list_payment_methods", {}), Step("ad", "list_addresses", {}),
                       Step("add", "add_to_cart", {"product_id": R("s", ".products[0].id"), "qty": 1})]),
                Stage([Step("ord", "place_order", {"payment_method_id": R("pm", f".payment_methods[{pk}].id"), "address_id": R("ad", f".addresses[{ak}].id")})], confirm=confirm)],
               lambda res: f"Order {res['ord']['placed']['id']} placed: 1 x {p['name']} for ${_price(p['price'])}, paid with {pm['label']}, shipping to your {addr['label']} address.")
    task = Task(tid, dom.name, "buy_now", 0, utt, slots={"product": p["name"], "frag": frag, "card": pm["label"], "address": addr["label"]},
                goal=[{"type": "order_placed", "items": [{"product_id": p["id"], "qty": 1}], "payment_method_id": pm["id"], "address_id": addr["id"]}],
                requires_confirmation=True, answer_facts=[p["name"]])
    return task, opt, opt


@gen("cancel_order")
def cancel_order(dom, state, rng, tid):
    cands = _orders(state, ("pending", "paid"))
    if not cands:
        return None
    o = rng.choice(cands)
    p = _oname(state, o)
    frag = _frag(p)
    utt = rng.choice([f"Cancel my order for the {frag}.", f"I no longer need the {frag} I ordered - please cancel that order.", f"Please cancel the {frag} order and refund me."])
    confirm = f"I'm about to cancel order {o['id']} ({p['name']}, ${_price(o['total'])}) and refund the charge. Shall I proceed?"
    k = _listing_index(state, o["id"])
    opt = Plan([Stage([Step("os", "list_orders", {})]), Stage([Step("c", "cancel_order", {"order_id": R("os", f".orders[{k}].id")})], confirm=confirm)],
               lambda res: f"Order {o['id']} ({p['name']}) has been cancelled and ${_price(o['total'])} will be refunded.")
    task = Task(tid, dom.name, "cancel_order", 0, utt, slots={"product": p["name"], "frag": frag}, goal=[{"type": "order_status", "id": o["id"], "status": "cancelled"}],
                requires_confirmation=True)
    return task, opt, opt


@gen("refund_delivered")
def refund_delivered(dom, state, rng, tid):
    cands = _orders(state, ("delivered",))
    if not cands:
        return None
    o = rng.choice(cands)
    p = _oname(state, o)
    frag = _frag(p)
    reason = rng.choice(["it arrived damaged", "it is the wrong size", "it stopped working after two days"])
    utt = rng.choice([f"I want a refund for the {frag} I received; {reason}.", f"The {frag} from my order needs to be refunded - {reason}."])
    confirm = f"I'm about to refund order {o['id']} ({p['name']}, ${_price(o['total'])}). Shall I proceed?"
    k = _listing_index(state, o["id"])
    opt = Plan([Stage([Step("os", "list_orders", {})]), Stage([Step("r", "refund_order", {"order_id": R("os", f".orders[{k}].id"), "reason": reason})], confirm=confirm)],
               lambda res: f"Order {o['id']} ({p['name']}) has been refunded: ${_price(o['total'])} will be returned to your card.")
    task = Task(tid, dom.name, "refund_delivered", 0, utt, slots={"product": p["name"], "frag": frag, "reason": reason},
                goal=[{"type": "order_status", "id": o["id"], "status": "refunded"}], requires_confirmation=True)
    return task, opt, opt


# ----------------------------------------------------------------------------- nothing to do: report, no writes
@gen("cancel_shipped")
def cancel_shipped(dom, state, rng, tid):
    cands = _orders(state, ("shipped",))
    if not cands:
        return None
    o = rng.choice(cands)
    p = _oname(state, o)
    frag = _frag(p)
    utt = rng.choice([f"Cancel my order for the {frag}.", f"Please cancel the {frag} order."])
    opt = Plan([Stage([Step("os", "list_orders", {})])],
               lambda res: f"Order {o['id']} ({p['name']}) has already shipped, so it cannot be cancelled. Once it is delivered I can request a refund for you.")
    task = Task(tid, dom.name, "cancel_shipped", 0, utt, slots={"product": p["name"], "frag": frag}, goal=[{"type": "no_change"}], answer_facts=["shipped"],
                on_question=f"The {frag} order. If it can't be cancelled any more, just tell me.")
    return task, opt, opt


@gen("missing_product")
def missing_product(dom, state, rng, tid):
    present = {p["name"] for p in state.data["products"].values()}
    cands = [n for n in FRAG if n not in present]
    name = rng.choice(cands)
    frag = FRAG[name]
    utt = rng.choice([f"Add a {frag} to my cart.", f"How much is the {frag}?", f"Order one {frag} with my Visa, delivered to my home address."])
    opt = Plan([Stage([Step("s", "search_products", {"query": frag})])], lambda res: f"I couldn't find any '{frag}' in the catalogue, so nothing was changed.")
    task = Task(tid, dom.name, "missing_product", 0, utt, slots={"frag": frag}, goal=[{"type": "no_change"}], answer_facts=["<none>"],
                on_question=f"I mean the {frag}. If it isn't in the catalogue, just tell me.")
    return task, opt, opt


@gen("out_of_stock")
def out_of_stock(dom, state, rng, tid):
    cands = [p for p in state.data["products"].values() if p["stock"] == 0 and p["id"] not in state.data["cart"]]
    if not cands:
        return None
    p = rng.choice(cands)
    frag = _frag(p)
    utt = rng.choice([f"Add a {frag} to my cart.", f"Put one {frag} in my basket."])
    opt = Plan([Stage([Step("s", "search_products", {"query": frag})])], lambda res: f"The {p['name']} (${_price(p['price'])}) is out of stock at the moment, so I couldn't add it to your cart.")
    task = Task(tid, dom.name, "out_of_stock", 0, utt, slots={"product": p["name"], "frag": frag}, goal=[{"type": "no_change"}], answer_facts=["stock"])
    return task, opt, opt


@gen("irrelevant")
def irrelevant(dom, state, rng, tid):
    utt, ans = rng.choice([
        ("What's a good gift for a ten-year-old who likes science?", "A beginner's microscope or a chemistry set are popular choices."),
        ("Thanks, that's all for today.", "You're welcome! Let me know if you need anything else."),
        ("Is a 4K monitor worth it for office work?", "For text-heavy work a 4K monitor gives noticeably sharper text; for email and spreadsheets a 1440p screen is usually enough."),
        ("How should I store a cast iron skillet?", "Dry it fully, wipe a thin film of oil on it and keep it in a dry place."),
        ("What does 'pending' mean on an order?", "Pending means the order has been received but the payment has not been captured yet."),
    ])
    plan = Plan([], lambda res: ans)
    task = Task(tid, dom.name, "irrelevant", 0, utt, goal=[{"type": "no_change"}], irrelevant=True)
    return task, plan, plan


def generate(dom: Commerce, seed: int, kinds: list[str] | None = None, tid_prefix: str = "cm") -> tuple[Task, Plan, Plan] | None:
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
