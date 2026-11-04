# Tool-Effect

Effect aware tool-calling agent: LoRA adapters trained on generated agentic flows, with negative examples used for
suppression (LUNE-style), plus effect-typed tools, `$ref` tool programs and a compiler-style trace optimizer.

* `PLAN.md` — the plan, decisions, experiment matrix, task board · `RESULTS.md` — ledger of every job and number
* `JOBS.md` — script catalog and environments · `CLUSTER.md` — cluster facts and gotchas (portable)

## Quick start (login / `amdfast` node)

```bash
source env/activate.sh                 # module stack + .venv-train (torch 2.13, transformers 5.17, vllm 0.31, peft, trl)
python -m pytest -q                    # ~50 CPU tests (world incl. hard/commerce domains, optimizer, negatives, data v2 filter, templates, trainer loop, eval loop)
sbatch --export=ALL,VERSION=v1 batch/build_data.batch            # data: world + public + SFT mix (amdfast, ~5 min)
sbatch batch/smoke.batch                                          # G0: GPU smoke test
sbatch --export=ALL,ARM=sft_v1 batch/train.batch                  # S1: chained LoRA SFT (configs/arms/*.args)
sbatch --export=ALL,TAG=sft_v1,ADAPTER=outputs/train/sft_v1/adapter batch/eval_world.batch              # add SPLIT=test_hard for the OOD split
sbatch --export=ALL,TAG=sft_v1,ADAPTER=outputs/train/sft_v1/adapter batch/eval_bfcl.batch   # needs .venv-clients
bash scripts/submit_round.sh sft_v2 s2_cont_sft s2_sup_ul s2_sup_npo s2_dpo   # one SFT + negative arms + all evals, fully chained
bash scripts/status.sh                                            # queue, runs, summaries
```

## Layout

`src/etool/` — `schema.py` (trajectories), `refs.py` (`$n.path` references), `templates.py` (chat template, spans,
parser), `world/` (domains: `workspace` for training; test-only `workspace_hard` (OOD kinds + distractor tools) and `commerce` (unseen tool set); executor, planner plans, verifier, scripted user), `optimize/` (passes + legality),
`negatives/` (taxonomy, perturbations, optimizer diff, pairs), `data/` (world/public builders, SFT mix, teacher,
rollouts, decontam), `train/` (LoRA trainer: sft / suppress_ul / suppress_npo / dpo, time budget + resume),
`serve/` (vLLM launcher, merge), `eval/` (world eval, BFCL runner, NESTful, paired bootstrap).
