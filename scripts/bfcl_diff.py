"""Per-category BFCL accuracy vs the base, and the prompts the base gets right but a run gets wrong (by error type).
    python scripts/bfcl_diff.py sft_v2 sft_v3a    (paper Sec. "When fine-tuning beats the base model on BFCL")
"""
import glob, json, sys, collections
def load(tag):
    out={}
    for f in glob.glob(f"outputs/bfcl/{tag}/score/*/*/BFCL_v4_*_score.json"):
        cat=f.split("BFCL_v4_")[1].replace("_score.json","")
        L=[json.loads(l) for l in open(f) if l.strip()]
        head=L[0]; errs={e["id"]:e for e in L[1:]}
        out[cat]=(head["total_count"], head["correct_count"], errs)
    return out
A=load("base"); tags=sys.argv[1:]
B={t:load(t) for t in tags}
print(f"{'category':28s} {'n':>5s} {'base':>6s} " + " ".join(f"{t[:10]:>10s}" for t in tags))
for c in sorted(A):
    n,ca,_=A[c]
    print(f"{c:28s} {n:5d} {100*ca/n:6.1f} " + " ".join(f"{(100*B[t][c][1]/B[t][c][0] if c in B[t] else float('nan')):10.1f}" for t in tags))
for t in tags:
    print(f"\n== {t}: prompts base right, {t} wrong (by category, error type)")
    tot=collections.Counter(); rev=collections.Counter()
    for c in A:
        if c not in B[t] or c.startswith("multi_turn"): continue
        ea=A[c][2]; eb=B[t][c][2]
        for i,e in eb.items():
            if i not in ea: tot[(c,e.get("error_type","?"))]+=1
        for i in ea:
            if i not in eb: rev[c]+=1
    for (c,et),k in tot.most_common(14): print(f"  {k:4d}  {c:26s} {et}")
    print("  lost total", sum(tot.values()), " | gained (base wrong, ours right) total", sum(rev.values()))
