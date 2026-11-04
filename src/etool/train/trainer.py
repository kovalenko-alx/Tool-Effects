"""Single-GPU LoRA trainer with four objectives and SLURM-friendly time budgeting.

Objectives (``--objective``):
  sft           NLL on assistant spans of trajectories
  suppress_ul   retain NLL (good turn) + lambda * unlikelihood on the diverging span of the bad turn (common suffix trimmed,
                closing </tool_call> tags inside the span excluded)
  suppress_npo  retain NLL (good turn) + lambda * NPO on the diverging span (reference --ref: base model with the adapter
                disabled, or the --init_adapter weights as a frozen second adapter — the SFT model the arm starts from)
  dpo           DPO on (good, bad) turns at the same prefix (reference = base model), + optional retain NLL (lambda_sft)
Pair objectives mix in SFT micro-batches with probability ``--sft_mix`` to keep general tool-calling ability (retain set).

Time budget: ``--max_seconds``; the loop saves ``<out>/last`` and exits with code 3 when the budget is nearly spent.
On start it resumes from ``<out>/last`` if present. ``<out>/DONE`` marks completion. LUNE defaults for LoRA/optimizer.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch

from etool.templates import ChatFormatter, load_tokenizer
from etool.train.data import EpochSampler, PairExample, SFTExample, collate_sft, load_pairs, load_sft, pad_batch, span_mask
from etool.train.losses import dpo_loss, drop_tokens, nll_loss, nll_lp, npo_loss, seq_logp, seq_logp_lp, token_logps, unlikelihood_lp

EXIT_RESUME = 3


def build_model(model_path: str, lora: dict[str, Any], device: torch.device, grad_ckpt: bool, attn: str = "sdpa"):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, attn_implementation=attn)
    model.to(device)
    if grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    cfg = LoraConfig(r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"], target_modules=lora["targets"], bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    return model


def save_state(model, opt, sched, step: int, samplers: dict[str, EpochSampler], out: Path, tag: str = "last") -> None:
    d = out / tag
    tmp = out / (tag + ".tmp")
    if tmp.exists():
        import shutil
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    model.save_pretrained(tmp / "adapter", selected_adapters=["default"])  # never the frozen "ref" adapter
    torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(), "step": step, "samplers": {k: v.state() for k, v in samplers.items()},
                "rng": {"py": random.getstate(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}},
               tmp / "state.pt")
    if d.exists():
        import shutil
        shutil.rmtree(d)
    tmp.rename(d)


def load_state(model, opt, sched, samplers: dict[str, EpochSampler], out: Path) -> int:
    d = out / "last"
    if not (d / "state.pt").exists():
        return 0
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    sd = load_file(str(d / "adapter" / "adapter_model.safetensors"))
    set_peft_model_state_dict(model, sd)
    st = torch.load(d / "state.pt", weights_only=False)
    opt.load_state_dict(st["opt"])
    sched.load_state_dict(st["sched"])
    for k, v in st["samplers"].items():
        if k in samplers:
            samplers[k].load(v)
    random.setstate(st["rng"]["py"])
    torch.set_rng_state(st["rng"]["torch"])
    if st["rng"]["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["rng"]["cuda"])
    return int(st["step"])


def ref_logits(model, ids: torch.Tensor, att: torch.Tensor, ref: str) -> torch.Tensor:
    """Reference-policy logits (no grad): the base model (adapter disabled) or the frozen "ref" adapter loaded from
    --init_adapter. The ref forward runs in eval mode so LoRA dropout does not make the reference stochastic; no KV cache
    (eval mode would otherwise build one: ~1 GB at 4k context)."""
    with torch.no_grad():
        if ref == "init":
            model.eval()
            model.set_adapter("ref")
            try:
                return model(input_ids=ids, attention_mask=att, use_cache=False).logits
            finally:
                model.set_adapter("default")
                model.train()
        with model.disable_adapter():
            return model(input_ids=ids, attention_mask=att, use_cache=False).logits


def pair_losses(model, batch: list[PairExample], objective: str, beta: float, pad_id: int, device: torch.device, avg_logp: bool,
                ref: str = "base", drop_ids: Optional[torch.Tensor] = None) -> dict[str, torch.Tensor]:
    """Memory order matters on a 40 GB card (job 11835829 OOMed at 35 GB): reference log-probs first (no graph, logits freed
    at once), then the two policy forwards, and one token-level log-softmax per policy forward reused by every term."""
    good_ids, good_att = pad_batch([b.good_ids for b in batch], pad_id, device)
    bad_ids, bad_att = pad_batch([b.bad_ids for b in batch], pad_id, device)
    gmask = span_mask([b.good_ids for b in batch], [b.good_span for b in batch], device)
    bmask = span_mask([b.bad_ids for b in batch], [b.bad_span for b in batch], device)
    if drop_ids is not None:  # closing tags inside the span are not suppression targets (DPO keeps whole turns)
        bmask = drop_tokens(bmask, bad_ids, drop_ids)
    bfull = span_mask([b.bad_ids for b in batch], [b.bad_full_span for b in batch], device)
    out: dict[str, torch.Tensor] = {}
    rb = rg = None
    if objective in ("suppress_npo", "dpo"):
        rb = seq_logp(ref_logits(model, bad_ids, bad_att, ref), bad_ids, bmask if objective == "suppress_npo" else bfull, avg_logp)
        if objective == "dpo":
            rg = seq_logp(ref_logits(model, good_ids, good_att, ref), good_ids, gmask, avg_logp)
    lp_g = token_logps(model(input_ids=good_ids, attention_mask=good_att).logits, good_ids)
    out["retain"] = nll_lp(lp_g, gmask)
    lp_b = token_logps(model(input_ids=bad_ids, attention_mask=bad_att).logits, bad_ids)
    if objective == "suppress_ul":
        out["neg"] = unlikelihood_lp(lp_b, bmask)
    elif objective == "suppress_npo":
        out["neg"] = npo_loss(seq_logp_lp(lp_b, bmask, avg_logp), rb, beta)
    elif objective == "dpo":
        out["neg"] = dpo_loss(seq_logp_lp(lp_g, gmask, avg_logp), rg, seq_logp_lp(lp_b, bfull, avg_logp), rb, beta)
    with torch.no_grad():
        out["p_bad_span"] = seq_logp_lp(lp_b, bmask, True).exp().mean()
    return out


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--objective", choices=["sft", "suppress_ul", "suppress_npo", "dpo"], default="sft")
    ap.add_argument("--sft_files", nargs="*", default=[])
    ap.add_argument("--pair_files", nargs="*", default=[])
    ap.add_argument("--val_files", nargs="*", default=[])
    ap.add_argument("--pair_classes", nargs="*", default=None)
    ap.add_argument("--max_per_class", type=int, default=None)
    ap.add_argument("--limit_per_file", type=int, default=None)
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--max_len", type=int, default=4096)
    # LUNE defaults
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_targets", default="q_proj,k_proj,v_proj,o_proj,up_proj,down_proj")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_frac", type=float, default=0.05)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--micro_batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--lam", type=float, default=0.3, help="weight of the negative term")
    ap.add_argument("--beta", type=float, default=0.1, help="NPO / DPO beta")
    ap.add_argument("--avg_logp", action="store_true", help="length-normalised sequence log-probs for NPO/DPO (beta then acts per token: "
                    "round 1 used it with the papers' beta=0.1 and pushed the bad spans ~30 nats per token, RESULTS §7)")
    ap.add_argument("--ref", choices=["base", "init"], default="base",
                    help="NPO/DPO reference policy: the base model (adapter disabled) or the --init_adapter weights (frozen second adapter)")
    ap.add_argument("--sft_mix", type=float, default=0.5, help="fraction of micro-batches drawn from SFT data for pair objectives")
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--val_every", type=int, default=100)
    ap.add_argument("--val_n", type=int, default=64)
    ap.add_argument("--save_every_min", type=float, default=10.0)
    ap.add_argument("--max_seconds", type=float, default=None)
    ap.add_argument("--margin_seconds", type=float, default=240.0)
    ap.add_argument("--init_adapter", default=None, help="start from an existing LoRA adapter dir (its r/alpha/dropout/targets override the CLI); fresh optimizer")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    if args.init_adapter:
        cfg = json.loads((Path(args.init_adapter) / "adapter_config.json").read_text())
        args.lora_r, args.lora_alpha, args.lora_dropout = int(cfg["r"]), int(cfg["lora_alpha"]), float(cfg.get("lora_dropout", 0.0))
        tm = cfg.get("target_modules")
        if isinstance(tm, (list, set, tuple)):
            args.lora_targets = ",".join(sorted(tm))
        print(f"init_adapter: {args.init_adapter} -> r={args.lora_r} alpha={args.lora_alpha} targets={args.lora_targets}", flush=True)

    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "DONE").exists():
        print("already DONE")
        return 0
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    tok = load_tokenizer(args.model)
    fmt = ChatFormatter(tok, max_len=args.max_len)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    tc_close = tok.convert_tokens_to_ids(fmt.profile.closing_tag)
    drop_ids = None if tc_close is None or tc_close == tok.unk_token_id else torch.tensor([tc_close], device=device)
    sft = load_sft(args.sft_files, fmt, args.cache_dir, args.limit_per_file) if args.sft_files else []
    pairs = load_pairs(args.pair_files, fmt, args.cache_dir, set(args.pair_classes) if args.pair_classes else None, args.max_per_class, args.seed) if args.pair_files else []
    val = load_sft(args.val_files, fmt, args.cache_dir, args.val_n) if args.val_files else []
    print(f"data: sft={len(sft)} pairs={len(pairs)} val={len(val)}  load {time.time()-t0:.0f}s", flush=True)
    use_pairs = args.objective != "sft"
    if use_pairs and not pairs:
        raise SystemExit("pair objective without pair data")
    if not sft and (not use_pairs or args.sft_mix > 0):
        raise SystemExit("no SFT data")

    # steps
    per_step_sft = args.micro_batch * args.accum * (args.sft_mix if use_pairs else 1.0)
    per_step_pair = args.micro_batch * args.accum * (1 - args.sft_mix) if use_pairs else 0
    n_examples = len(sft) + len(pairs)
    total_steps = args.max_steps or max(1, int(math.ceil(args.epochs * n_examples / (args.micro_batch * args.accum))))

    model = build_model(args.model, {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout, "targets": args.lora_targets.split(",")},
                        device, args.grad_ckpt, args.attn)
    if args.ref == "init" and use_pairs:
        if not args.init_adapter:
            raise SystemExit("--ref init needs --init_adapter")
        model.load_adapter(args.init_adapter, adapter_name="ref", is_trainable=False)
        model.set_adapter("default")
        print(f"ref adapter: {args.init_adapter} loaded as the frozen NPO/DPO reference", flush=True)
    params = [p for p in model.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    warm = max(1, int(args.warmup_frac * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) if s < warm else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total_steps - warm)))))
    samplers = {"sft": EpochSampler(max(1, len(sft)), args.seed), "pair": EpochSampler(max(1, len(pairs)), args.seed + 1)}
    if args.init_adapter and not (out / "last" / "state.pt").exists():
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        res = set_peft_model_state_dict(model, load_file(str(Path(args.init_adapter) / "adapter_model.safetensors")))
        unexpected = getattr(res, "unexpected_keys", [])
        if unexpected:
            raise SystemExit(f"init_adapter: unexpected keys {list(unexpected)[:5]}")
        print("init_adapter: weights loaded", flush=True)
    step = load_state(model, opt, sched, samplers, out)
    print(f"model ready: trainable={n_tr/1e6:.1f}M total_steps={total_steps} resume_step={step}  {time.time()-t0:.0f}s", flush=True)

    log = (out / "train_log.jsonl").open("a")
    last_save = time.time()
    model.train()

    def out_of_time() -> bool:
        return args.max_seconds is not None and (time.time() - t0) > (args.max_seconds - args.margin_seconds)

    def validate() -> Optional[float]:
        if not val:
            return None
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(val), args.micro_batch):
                b = collate_sft(val[i: i + args.micro_batch], pad_id, device)
                logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
                m = b["labels"] != -100
                tot += nll_loss(logits, b["input_ids"], m).item() * m.sum().item()
                n += m.sum().item()
        model.train()
        return tot / max(1, n)

    while step < total_steps:
        if out_of_time():
            save_state(model, opt, sched, step, samplers, out)
            print(f"out of time at step {step}; saved, exit {EXIT_RESUME}", flush=True)
            log.close()
            return EXIT_RESUME
        opt.zero_grad(set_to_none=True)
        acc: dict[str, float] = {}
        for micro in range(args.accum):
            draw_pair = use_pairs and (random.random() >= args.sft_mix)
            if draw_pair:
                batch = [pairs[i] for i in samplers["pair"].take(args.micro_batch)]
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    ls = pair_losses(model, batch, args.objective, args.beta, pad_id, device, args.avg_logp, args.ref, drop_ids)
                loss = ls["retain"] * (1.0 if args.objective != "dpo" else args.lam) + ls["neg"] * (args.lam if args.objective != "dpo" else 1.0)
                acc["retain"] = acc.get("retain", 0) + ls["retain"].item() / args.accum
                acc["neg"] = acc.get("neg", 0) + ls["neg"].item() / args.accum
                acc["p_bad"] = acc.get("p_bad", 0) + ls["p_bad_span"].item() / args.accum
            else:
                batch = [sft[i] for i in samplers["sft"].take(args.micro_batch)]
                b = collate_sft(batch, pad_id, device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
                    loss = nll_loss(logits, b["input_ids"], b["labels"] != -100)
                acc["sft"] = acc.get("sft", 0) + loss.item() / args.accum
            (loss / args.accum).backward()
        gn = torch.nn.utils.clip_grad_norm_(params, args.clip).item()
        opt.step()
        sched.step()
        step += 1
        if step % args.log_every == 0 or step == 1:
            rec = {"step": step, "lr": sched.get_last_lr()[0], "grad_norm": gn, "elapsed_s": round(time.time() - t0), **{k: round(v, 5) for k, v in acc.items()}}
            if device.type == "cuda":
                rec["mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            if args.val_every and step % args.val_every == 0:
                rec["val_nll"] = validate()
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
        if (time.time() - last_save) > args.save_every_min * 60:
            save_state(model, opt, sched, step, samplers, out)
            last_save = time.time()

    v = validate()
    save_state(model, opt, sched, step, samplers, out)
    model.save_pretrained(out / "adapter", selected_adapters=["default"])
    tok.save_pretrained(out / "adapter")
    (out / "DONE").write_text(json.dumps({"steps": step, "val_nll": v, "elapsed_s": round(time.time() - t0), "trainable": n_tr}))
    print(f"DONE steps={step} val_nll={v} elapsed={time.time()-t0:.0f}s", flush=True)
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
