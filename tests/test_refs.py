import pytest

from etool.refs import Ref, RefError, dependencies, has_refs, parse_ref, resolve


def test_parse():
    assert parse_ref("$1") == Ref(1, ())
    assert parse_ref("$2.items[0].id") == Ref(2, ("items", 0, "id"))
    assert parse_ref("$12[3]") == Ref(12, (3,))
    assert parse_ref("$0") is None  # 1-based
    assert parse_ref("price is $1") is None  # embedded -> literal
    assert parse_ref("$1.") is None and parse_ref(1) is None and parse_ref("$a") is None
    assert Ref(2, ("items", 0, "id")).text == "$2.items[0].id"
    assert parse_ref("$2.tasks[*].id") == Ref(2, ("tasks", "*", "id")) and Ref(2, ("tasks", "*", "id")).text == "$2.tasks[*].id"
    assert resolve({"ids": "$2.tasks[*].id"}, {2: {"tasks": [{"id": "a"}, {"id": "b"}]}}) == {"ids": ["a", "b"]}


def test_dependencies_and_resolve():
    args = {"id": "$1.items[0].id", "tags": ["x", "$2"], "nested": {"owner": "$1.owner"}, "n": 3}
    assert dependencies(args) == {1, 2}
    assert has_refs(args) and not has_refs({"a": 1})
    results = {1: {"items": [{"id": "it-7"}], "owner": "ann"}, 2: "tag-b"}
    out = resolve(args, results)
    assert out == {"id": "it-7", "tags": ["x", "tag-b"], "nested": {"owner": "ann"}, "n": 3}
    assert args["id"] == "$1.items[0].id"  # not mutated


def test_resolve_errors():
    with pytest.raises(RefError):
        resolve({"a": "$3"}, {1: {}})
    with pytest.raises(RefError):
        resolve({"a": "$1.missing"}, {1: {"x": 1}})
    with pytest.raises(RefError):
        resolve({"a": "$1[2]"}, {1: [0, 1]})
