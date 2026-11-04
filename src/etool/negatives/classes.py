"""Error-class taxonomy for negative examples and its mapping onto BFCL categories.

origin: how we obtain negatives of that class — rule (perturbation of gold), optimizer (diff naive→optimal),
verifier (policy checks on teacher/model rollouts).
"""
from __future__ import annotations

CLASSES: dict[str, dict] = {
    # --- correctness (rule perturbations of gold actions)
    "wrong_tool":             {"group": "correctness", "origin": "rule", "bfcl": ["simple", "multiple", "live_simple", "live_multiple"]},
    "hallucinated_tool":      {"group": "correctness", "origin": "rule", "bfcl": ["simple", "multiple", "live_multiple"]},
    "missing_required_arg":   {"group": "correctness", "origin": "rule", "bfcl": ["simple", "parallel", "live_simple"]},
    "wrong_arg_value":        {"group": "correctness", "origin": "rule", "bfcl": ["simple", "multiple", "parallel", "multi_turn_base"]},
    "wrong_arg_type":         {"group": "correctness", "origin": "rule", "bfcl": ["simple", "live_simple"]},
    "unnecessary_call":       {"group": "over-calling", "origin": "rule", "bfcl": ["irrelevance", "live_irrelevance"]},
    "missing_call":           {"group": "under-calling", "origin": "rule", "bfcl": ["live_relevance"]},
    "premature_answer":       {"group": "under-calling", "origin": "rule", "bfcl": ["multi_turn_base"]},
    "guessed_missing_info":   {"group": "under-calling", "origin": "rule", "bfcl": ["multi_turn_miss_param", "multi_turn_miss_func"]},
    "refusal":                {"group": "under-calling", "origin": "rule", "bfcl": ["simple", "multiple", "parallel", "live_simple", "live_multiple", "live_relevance"]},  # "I don't have access to a tool for that" on a public prefix whose gold is a call (S1 failure mode, data v2)
    "ref_without_capability": {"group": "format", "origin": "rule", "bfcl": ["simple", "parallel", "multi_turn_base"]},
    # --- policy / effects (verifier on rollouts, or rule perturbations)
    "unconfirmed_destructive": {"group": "policy", "origin": "rule|verifier", "bfcl": [], "tau2": True},
    "duplicate_write":         {"group": "policy", "origin": "rule|verifier", "bfcl": ["multi_turn_base"], "tau2": True},
    "stale_state_write":       {"group": "policy", "origin": "rule|verifier", "bfcl": ["multi_turn_base"], "tau2": True},
    # --- efficiency (optimizer diff)
    "redundant_read":          {"group": "efficiency", "origin": "optimizer", "bfcl": ["multi_turn_base"]},
    "dead_call":               {"group": "efficiency", "origin": "optimizer", "bfcl": ["multi_turn_base"]},
    "unbatched":               {"group": "efficiency", "origin": "optimizer", "bfcl": ["parallel", "parallel_multiple"]},
    "serialized_calls":        {"group": "efficiency", "origin": "optimizer", "bfcl": ["parallel", "parallel_multiple"]},
    "lookup_of_known_value":   {"group": "efficiency", "origin": "optimizer", "bfcl": ["simple", "multi_turn_base"]},
}

GROUPS = sorted({v["group"] for v in CLASSES.values()})


def bfcl_categories(error_class: str) -> list[str]:
    return list(CLASSES[error_class]["bfcl"])
