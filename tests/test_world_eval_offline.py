"""Drive world_eval.run_task with a fake completion client that replays gold trajectories (no server, no GPU)."""
import glob
import os
import threading
from types import SimpleNamespace

import pytest

from etool.eval.world_eval import run_task, summarize
from etool.templates import ChatFormatter, load_tokenizer
from etool.world.domains.workspace import Workspace
from etool.world.domains.workspace_tasks import generate
from etool.world.rollout import run_plan

SNAPS = glob.glob(os.path.expanduser("~/EfficeintTool/hf/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/*/tokenizer.json"))
pytestmark = pytest.mark.skipif(not SNAPS, reason="base tokenizer not downloaded")


class ReplayClient:
    """Returns, for each prompt, the next assistant segment of the gold trajectory whose user utterance is in the prompt."""

    def __init__(self, fmt, golds):
        self.fmt, self.golds = fmt, golds  # golds: {(utterance, refs): Trajectory}
        self.pos: dict = {}
        self.lock = threading.Lock()
        self.completions = SimpleNamespace(create=self._create)

    def _create(self, model, prompt, max_tokens, temperature, stop):
        refs = "$N" in prompt.split("<|im_start|>user")[0]
        key = next(k for k in self.golds if k[0] in prompt and k[1] == refs)
        g = self.golds[key]
        with self.lock:
            i = self.pos.get(key, 0)
            idxs = g.assistant_indices()
            k = idxs[min(i, len(idxs) - 1)]
            self.pos[key] = i + 1
        seg = self.fmt.render(g.turns[: k + 1], g.tools_openai())[len(self.fmt.render(g.turns[:k], g.tools_openai(), True)):]
        text = seg[: seg.rfind("<|im_end|>")] if "<|im_end|>" in seg else seg
        n = len(self.fmt.tok(text)["input_ids"])
        return SimpleNamespace(choices=[SimpleNamespace(text=text, finish_reason="stop")], usage=SimpleNamespace(completion_tokens=n, prompt_tokens=len(prompt) // 4))


def test_eval_loop_reproduces_gold():
    fmt = ChatFormatter(load_tokenizer(os.path.dirname(SNAPS[0])), max_len=10**9)
    dom = Workspace()
    golds, tasks = {}, []
    for s in range(24):
        g = generate(dom, s)
        if g is None:
            continue
        task, opt, _ = g
        tasks.append(task)
        for refs in (True, False):
            golds[(task.utterance, refs)] = run_plan(dom, task, opt, refs=refs, source="optimal")[0]
    client = ReplayClient(fmt, golds)
    rows = []
    for task in tasks:
        for refs in (True, False):
            res = run_task(client, "m", fmt, task, refs, max_turns=12, max_tokens=512, temperature=0.0, tag="replay")
            rows.append(res["row"])
            assert res["row"]["success"], (task.kind, refs, res["row"])
            assert res["row"]["excess_calls"] == 0 and res["row"]["excess_turns"] == 0, (task.kind, refs, res["row"])
            assert res["row"]["out_tokens"] > 0 and res["row"]["opt_accepted"]
    s = summarize(rows)
    assert s["success"] == 1.0 and s["excess_calls"] == 0 and set(s["by_kind"]) == {t.kind for t in tasks}
