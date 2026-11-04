"""Merge a LoRA adapter into the base model and save a standalone bf16 checkpoint (for vLLM / BFCL)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(args.out)
    (Path(args.out) / "MERGED_FROM.json").write_text(json.dumps({"base": args.base, "adapter": args.adapter}))
    print("merged ->", args.out)


if __name__ == "__main__":
    main()
