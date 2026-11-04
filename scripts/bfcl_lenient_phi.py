"""Re-score a model's BFCL single-turn answers with a lenient decoder (diagnostic for Phi-4-mini's base model).

    python scripts/bfcl_lenient_phi.py outputs/bfcl/base_phi4mini outputs/bfcl/base_phi4mini_lenient
Copies the result files and replaces every raw-text answer from which a call can be recovered by the decoded form BFCL
stores for parsed calls ([{name: {arg: value}}]). Recovered forms: <|tool_call|> blocks (doubled tag collapsed, missing
closer tolerated), untagged {"name": ..., "arguments": ...} JSON, and calls written as Python code (`f({...})` or
`f(a=1, b="x")`, dotted names allowed). Multi-turn results are copied unchanged. Then run `bfcl evaluate` on the copy.
"""
import ast, glob, json, os, re, shutil, sys

TAG = re.compile(r"(?:<\|tool_call\|>)+\s*(.*?)\s*(?:<\|/tool_call\|>|$)", re.S)
CODE = re.compile(r"```(?:python|json)?\s*(.*?)```", re.S)

def _jsonish(txt):
    try:
        v = json.loads(txt)
    except Exception:
        return None
    v = v if isinstance(v, list) else [v]
    out = []
    for c in v:
        if isinstance(c, dict) and "name" in c:
            out.append({c["name"]: c.get("arguments", c.get("parameters", {})) or {}})
    return out or None

def _pycalls(code):
    try:
        tree = ast.parse(code)
    except Exception:
        return None
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            if name in ("print", "len", "range", "int", "float", "str", "round"):  # Python builtins, not tools
                continue
            try:
                args = {k.arg: ast.literal_eval(k.value) for k in node.keywords if k.arg}
                if not args and len(node.args) == 1 and isinstance(node.args[0], ast.Dict):
                    args = ast.literal_eval(node.args[0])
                elif node.args:
                    continue  # positional arguments cannot be mapped to parameter names
            except Exception:
                continue
            out.append({name: args})
    return out or None

def decode(text):
    m = TAG.search(text)
    if m:
        r = _jsonish(m.group(1))
        if r: return r
    for m in re.finditer(r"\{[^{}]*\"name\"[^{}]*\"arguments\"\s*:\s*\{.*?\}\s*\}", text, re.S):
        r = _jsonish(m.group(0))
        if r: return r
    for block in CODE.findall(text):
        r = _jsonish(block.strip()) or _pycalls(block)
        if r: return r
    return None

def main(src, dst):
    if os.path.exists(dst): shutil.rmtree(dst)
    shutil.copytree(os.path.join(src, "result"), os.path.join(dst, "result"))
    n = rec = 0
    for f in glob.glob(os.path.join(dst, "result", "*", "*", "BFCL_v4_*_result.json")):
        if "/multi_turn/" in f: continue
        rows = [json.loads(l) for l in open(f) if l.strip()]
        for r in rows:
            if isinstance(r["result"], str):
                n += 1
                d = decode(r["result"])
                if d is not None:
                    r["result"] = d; rec += 1
        with open(f, "w") as fh:
            for r in rows: fh.write(json.dumps(r) + "\n")
    print(f"raw-text answers: {n}, recovered as calls: {rec}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
