import json
import subprocess
import sys

from etool.negatives.perturb import perturb_trajectory
from etool.schema import write_jsonl
from etool.world.domains.workspace import Workspace
from etool.world.domains.workspace_tasks import generate
from etool.world.rollout import run_plan

DOM = Workspace()


def test_from_rollouts_on_synthetic_rollouts(tmp_path):
    rollouts = []
    for s in range(40):
        g = generate(DOM, s)
        if g is None:
            continue
        task, opt, nai = g
        gold, _ = run_plan(DOM, task, opt, refs=False, source="model:teacher")
        naive, _ = run_plan(DOM, task, nai, refs=True, source="model:teacher")
        rollouts += [gold, naive]
        # a failing rollout: unconfirmed destructive / wrong tool spliced in
        for n in perturb_trajectory(gold, task, seed=s, classes=["unconfirmed_destructive", "wrong_tool"]):
            bad = gold.prefix(n.turn_idx)
            bad.id = gold.id + f":bad:{n.error_class}"
            bad.turns = gold.turns[: n.turn_idx] + [n.bad_turn]
            rollouts.append(bad)
    write_jsonl(tmp_path / "trajectories.jsonl", rollouts)
    r = subprocess.run([sys.executable, "-m", "etool.data.from_rollouts", "--rollouts", str(tmp_path / "trajectories.jsonl"), "--out", str(tmp_path / "out")],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    st = json.loads((tmp_path / "out" / "stats.json").read_text())
    assert st["positives"] > 0 and st["pairs_optimizer"] > 0 and st["pairs_unconfirmed"] > 0 and st.get("failed", 0) > 0, st
    assert (tmp_path / "out" / "pairs.jsonl").stat().st_size > 0
