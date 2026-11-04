import collections

from etool.negatives.classes import CLASSES
from etool.negatives.perturb import PERTURBERS, perturb_trajectory
from etool.world.domains.workspace import Workspace
from etool.world.domains.workspace_tasks import generate
from etool.world.rollout import run_plan
from etool.world.verifier import verify

DOM = Workspace()


def test_perturbations_cover_classes_and_are_actually_bad():
    counts = collections.Counter()
    bad_verified = collections.Counter()
    for seed in range(0, 90):
        g = generate(DOM, seed)
        if g is None:
            continue
        task, opt, _ = g
        step, _ = run_plan(DOM, task, opt, refs=False, source="optimal")
        prog, _ = run_plan(DOM, task, opt, refs=True, source="optimal")
        fa = next((t for t in prog.turns if t.is_action), None)
        if fa:
            step.meta["program_first_action"] = [{"name": c.name, "arguments": c.arguments, "id": c.id} for c in fa.tool_calls]
        for traj in (step, prog):
            for n in perturb_trajectory(traj, task, seed=seed):
                counts[n.error_class] += 1
                assert n.error_class in CLASSES and n.bad_turn != n.good_turn
                # splice the bad turn in and check the verifier no longer reports a clean success (where decidable)
                t2 = traj.prefix(n.turn_idx)
                t2.turns = traj.turns[: n.turn_idx] + [n.bad_turn]
                v = verify(DOM, task, t2)
                if not v.success or v.violations or v.errors or v.unnecessary_calls:
                    bad_verified[n.error_class] += 1
    assert set(counts) >= set(PERTURBERS) - {"ignored_tool_error"}, (set(PERTURBERS) - set(counts))
    for cls in ("wrong_tool", "hallucinated_tool", "missing_required_arg", "wrong_arg_type", "unnecessary_call", "unconfirmed_destructive", "ref_without_capability", "stale_state_write"):
        assert bad_verified[cls] >= 0.8 * counts[cls], (cls, bad_verified[cls], counts[cls])
