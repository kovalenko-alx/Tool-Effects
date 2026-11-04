"""Near-miss decline builder (recipe SFT_near-miss): name normalisation, TF-IDF nearness, and the one-change mix."""
import json
from types import SimpleNamespace

from etool.data.nearmiss import Tfidf, _norm_name, cmd_mix
from etool.schema import ToolCall, ToolSpec, Trajectory, Turn, read_jsonl, write_jsonl


def test_norm_name_merges_spellings():
    assert _norm_name("getWeather") == _norm_name("get_weather") == _norm_name("Get Weather") == "weather"


def test_tfidf_ranks_same_topic_tool_first():
    tf = Tfidf(["calculate_mortgage_payment monthly mortgage payment loan interest rate", "get_weather current weather city",
                "translate_text translate text language"])
    q = tf.transform(["compound interest on a loan amount"])
    sims = (q @ tf.matrix.T).toarray().ravel()
    assert sims.argmax() == 0 and sims[1] == 0


def _traj(i, source, call=True):
    spec = ToolSpec("f", "d", {"type": "object", "properties": {}})
    turns = [Turn("user", f"q{i}"), Turn("assistant", "", [ToolCall("f", {}, "1")] if call else [])]
    return Trajectory(f"{source}:{i}", source, [spec], turns)


def test_mix_replaces_exactly_n_public(tmp_path):
    clean, out = tmp_path / "clean", tmp_path / "out"
    clean.mkdir()
    write_jsonl(clean / "sft_train.jsonl", [_traj(i, "world:workspace:gold") for i in range(10)] + [_traj(i, "toolace") for i in range(20)])
    for n in ("sft_val.jsonl", "pairs_train.jsonl", "pairs_val.jsonl"):
        (clean / n).write_text("")
    (clean / "stats.json").write_text(json.dumps({"n_sft_train": 30}))
    kept = tmp_path / "kept.jsonl"
    write_jsonl(kept, [_traj(i, "nearmiss:toolace", call=False) for i in range(8)])
    cmd_mix(SimpleNamespace(clean=str(clean), kept=str(kept), out=str(out), n=5, seed=0))
    mixed = list(read_jsonl(out / "sft_train.jsonl", Trajectory))
    src = [t.source for t in mixed]
    assert len(mixed) == 30 and src.count("world:workspace:gold") == 10 and src.count("toolace") == 15 and src.count("nearmiss:toolace") == 5
