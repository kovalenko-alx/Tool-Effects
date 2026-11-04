"""CPU test of the training loop on a tiny random Qwen3 model with the real tokenizer."""
import glob
import json
import math
import os
import subprocess
import sys

import pytest

SNAPS = glob.glob(os.path.expanduser("~/EfficeintTool/hf/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/*/tokenizer.json"))
pytestmark = pytest.mark.skipif(not SNAPS, reason="base tokenizer not downloaded")


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    import torch
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

    d = tmp_path_factory.mktemp("tiny")
    tok = AutoTokenizer.from_pretrained(os.path.dirname(SNAPS[0]))
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      head_dim=16, max_position_embeddings=8192, tie_word_embeddings=True)
    torch.manual_seed(0)
    m = Qwen3ForCausalLM(cfg)
    m.save_pretrained(d / "model")
    tok.save_pretrained(d / "model")
    # data
    from etool.world.domains.workspace import Workspace
    from etool.world.domains.workspace_tasks import generate
    from etool.world.rollout import run_plan
    from etool.schema import write_jsonl
    from etool.negatives.perturb import perturb_trajectory
    from etool.negatives.pairs import build_pairs
    dom = Workspace()
    trajs, negs = [], []
    for s in range(12):
        g = generate(dom, s)
        if g is None:
            continue
        task, opt, _ = g
        t, _ = run_plan(dom, task, opt, refs=False, source="optimal")
        used = {c.name for turn in t.turns for c in turn.tool_calls} | {"search_contacts", "list_events"}
        t.tools = [s for s in t.tools if s.name in used]  # short prompts: the CPU test is about control flow, not capacity
        trajs.append(t)
        negs += perturb_trajectory(t, task, seed=s, classes=["wrong_tool", "wrong_arg_value", "unnecessary_call", "premature_answer"])
    write_jsonl(d / "sft.jsonl", trajs)
    pairs = build_pairs({t.id: t for t in trajs}, negs)
    with open(d / "pairs.jsonl", "w") as f:
        for p in pairs:
            f.write(p.to_json() + "\n")
    return d


def _run(args, timeout=900):
    env = dict(os.environ, HF_HUB_OFFLINE="1", CUDA_VISIBLE_DEVICES="")
    return subprocess.run([sys.executable, "-m", "etool.train.trainer", *args], capture_output=True, text=True, timeout=timeout, env=env)


@pytest.mark.parametrize("objective", ["sft", "suppress_ul", "suppress_npo", "dpo"])
def test_objectives_run_and_resume(tiny, objective, tmp_path):
    out = tmp_path / objective
    base = ["--model", str(tiny / "model"), "--out", str(out), "--objective", objective, "--sft_files", str(tiny / "sft.jsonl"),
            "--pair_files", str(tiny / "pairs.jsonl"), "--val_files", str(tiny / "sft.jsonl"), "--val_n", "4", "--micro_batch", "2", "--accum", "2",
            "--max_steps", "2", "--log_every", "1", "--val_every", "2", "--lora_r", "4", "--lora_alpha", "4", "--device", "cpu", "--max_len", "2048"]
    # 1) time budget hit immediately -> exit 3 with a checkpoint (margin > budget)
    r = _run(base + ["--max_seconds", "1", "--margin_seconds", "100"])
    assert r.returncode == 3, r.stdout[-2000:] + r.stderr[-2000:]
    assert (out / "last" / "state.pt").exists()
    # 2) resume and finish
    r = _run(base)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert (out / "DONE").exists() and (out / "adapter" / "adapter_model.safetensors").exists()
    recs = [json.loads(l) for l in (out / "train_log.jsonl").read_text().splitlines()]
    assert recs and all(all(v == v for v in rec.values() if isinstance(v, float)) for rec in recs)  # no NaN
    if objective != "sft":
        assert any("neg" in rec for rec in recs)


def test_init_adapter_continuation(tiny, tmp_path):
    base = ["--model", str(tiny / "model"), "--sft_files", str(tiny / "sft.jsonl"), "--pair_files", str(tiny / "pairs.jsonl"), "--micro_batch", "2",
            "--accum", "1", "--log_every", "1", "--val_every", "0", "--lora_r", "4", "--lora_alpha", "4", "--device", "cpu", "--max_len", "2048"]
    r = _run(base + ["--out", str(tmp_path / "a"), "--objective", "sft", "--max_steps", "1"])
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    r = _run(base + ["--out", str(tmp_path / "b"), "--objective", "suppress_ul", "--max_steps", "1", "--init_adapter", str(tmp_path / "a" / "adapter"),
                     "--lora_r", "16"])  # the CLI rank is overridden by the adapter config (r=4)
    assert r.returncode == 0 and "init_adapter: weights loaded" in r.stdout, r.stdout[-1500:] + r.stderr[-1500:]
    assert json.loads((tmp_path / "b" / "args.json").read_text())["lora_r"] == 4


@pytest.mark.parametrize("objective,expected", [("suppress_npo", 20 * math.log(2)), ("dpo", math.log(2))])
def test_ref_init_starts_at_log2(tiny, tmp_path, objective, expected):
    """--ref init: the reference is the frozen --init_adapter, so at step 0 policy == reference and the negative loss
    is exactly (2/beta) ln 2 (NPO) or ln 2 (DPO) with LoRA dropout off. Also checks resume and that the ref adapter is
    never saved."""
    base = ["--model", str(tiny / "model"), "--sft_files", str(tiny / "sft.jsonl"), "--pair_files", str(tiny / "pairs.jsonl"), "--micro_batch", "2",
            "--accum", "1", "--log_every", "1", "--val_every", "0", "--lora_r", "4", "--lora_alpha", "4", "--lora_dropout", "0", "--device", "cpu",
            "--max_len", "2048"]
    r = _run(base + ["--out", str(tmp_path / "a"), "--objective", "sft", "--max_steps", "1"])
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    arm = base + ["--out", str(tmp_path / "b"), "--objective", objective, "--beta", "0.1", "--sft_mix", "0", "--max_steps", "1",
                  "--init_adapter", str(tmp_path / "a" / "adapter"), "--ref", "init"]
    r = _run(arm + ["--max_seconds", "1", "--margin_seconds", "100"])  # budget hit before step 1 -> checkpoint with both adapters loaded
    assert r.returncode == 3 and "ref adapter:" in r.stdout, r.stdout[-1500:] + r.stderr[-1500:]
    assert not (tmp_path / "b" / "last" / "adapter" / "ref").exists()
    r = _run(arm)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    rec = json.loads((tmp_path / "b" / "train_log.jsonl").read_text().splitlines()[0])
    assert rec["step"] == 1 and abs(rec["neg"] - expected) < 1e-3, rec
    assert rec["grad_norm"] > 0 and not (tmp_path / "b" / "adapter" / "ref").exists()
