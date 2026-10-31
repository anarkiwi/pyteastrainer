import ast
import json

import numpy as np
import pytest
from coverage import CoverageData
from scipy import sparse

from pyteastrainer import cover
from pyteastrainer.fingerprint import fingerprint, parametrize_groups, suggest


def _matrix(rows, cost=None, exempt=None):
    n, e = len(rows), 1 + max(max(r) for r in rows)
    a = sparse.lil_matrix((n, e), dtype=bool)
    for i, r in enumerate(rows):
        a[i, list(r)] = True
    merged, weight, column = cover.merge_columns(a.tocsr())
    return cover.Matrix(
        [f"t{i}" for i in range(n)],
        np.asarray(cost if cost is not None else np.ones(n), float),
        merged,
        weight,
        np.asarray(exempt if exempt is not None else np.zeros(n), bool),
        efile=np.zeros(e, np.int64),
        eline=np.arange(e) + 1,
        column=column,
        files=["mod.py"],
    )


def _kept(m, sol):
    return [t for t, s in zip(m.tests, sol.selected) if s]


DECOY = [{0, 1, 2}, {3, 4, 5}, {0, 1, 3, 4}]


def test_exact_cover_beats_greedy_on_a_decoy():
    m = _matrix(DECOY)
    assert cover.summary(m, cover.greedy(m))["tests"] == 3
    sol = cover.min_cover(m)
    assert _kept(m, sol) == ["t0", "t1"]
    assert sol.optimal and sol.gap == 0
    assert cover.summary(m, sol.selected)["coverage"] == 1.0


def test_greedy_takes_free_tests_first():
    m = _matrix([{0, 1, 2}, {0}, {1, 2}], cost=[1, 0, 0])
    assert cover.greedy(m).tolist() == [False, True, True]


def test_runtime_breaks_ties_between_equal_size_covers():
    m = _matrix([{0, 1}, {2, 3}, {0, 2}, {1, 3}], cost=[5, 5, 1, 1])
    assert _kept(m, cover.min_cover(m, ("count", "time"))) == ["t2", "t3"]


def test_time_first_trades_count_for_runtime():
    m = _matrix([{0, 1, 2, 3}, {0, 1}, {2, 3}], cost=[10, 1, 1])
    assert _kept(m, cover.min_cover(m, ("count", "time"))) == ["t0"]
    fast = cover.min_cover(m, ("time", "count")).selected
    assert cover.summary(m, fast) == {"tests": 2, "runtime": 2.0, "coverage": 1.0}


def test_exempt_is_forced_and_makes_its_twin_redundant():
    rows, cost = [{0, 1}, {0, 1}, {2}], [5, 1, 1]
    assert _kept(_matrix(rows, cost), cover.min_cover(_matrix(rows, cost))) == [
        "t1",
        "t2",
    ]
    m = _matrix(rows, cost, exempt=[True, False, False])
    assert _kept(m, cover.min_cover(m)) == ["t0", "t2"]


@pytest.mark.parametrize("accepted,dropped", [(0, "t0"), (1, "t1")])
def test_baseline_tie_break_drops_the_accepted_test(accepted, dropped):
    m = _matrix([{0, 1}, {0, 1}, {2}])
    baseline = np.arange(3) == accepted
    kept = _kept(m, cover.min_cover(m, baseline=baseline))
    assert dropped not in kept and len(kept) == 2


def test_slow_stage_replaces_a_slow_test_by_fast_ones():
    m = _matrix([{0, 1, 2}, {0}, {1, 2}], cost=[1, 1, 1])
    assert _kept(m, cover.min_cover(m)) == ["t0"]
    slow = np.array([True, False, False])
    assert _kept(m, cover.min_cover(m, slow=slow)) == ["t1", "t2"]


def test_slow_stage_keeps_a_slow_test_nothing_else_replaces():
    m = _matrix([{0, 1, 2}, {0}, {0, 1}], cost=[9, 1, 1])
    slow = np.array([True, False, False])
    kept = cover.min_cover(m, slow=slow).selected
    assert kept.tolist() == [True, False, False]
    fast = np.asarray(m.a[~slow].sum(axis=0)).ravel() > 0
    cols = m.a[0].indices
    assert cover.ranges(m, cols[~fast[cols]]) == ["mod.py:3"]


def test_all_essential_skips_the_solver(monkeypatch):
    def boom(*_, **__):
        raise AssertionError("solver called")

    monkeypatch.setattr(cover, "milp", boom)
    m = _matrix([{0, 1}, {1, 2}, {3}])
    sol = cover.min_cover(m, baseline=np.array([True, False, False]))
    assert sol.selected.all() and sol.status == "every test essential"
    with pytest.raises(AssertionError):
        cover.min_cover(_matrix([{0, 1}, {1}]))


def test_partial_cover_picks_the_cheap_subset():
    m = _matrix([{0, 1, 2, 3, 4}, {0, 1, 2, 3}, {4}], cost=[10, 1, 1])
    assert _kept(m, cover.partial_cover(m, 0.8)) == ["t1"]
    assert _kept(m, cover.partial_cover(m, 1.0)) == ["t1", "t2"]


def test_identical_columns_merge_with_their_weight():
    m = _matrix([{0, 1, 2}, {2}])
    assert m.a.shape[1] == 2
    assert sorted(m.weight.tolist()) == [1.0, 2.0]
    assert m.column[0] == m.column[1] != m.column[2]
    assert m.elements == 3


def test_essential_counts_weighted_unique_elements():
    m = _matrix([{0, 1, 2}, {2, 3}, {3}])
    assert cover.essential(m).tolist() == [2.0, 0.0, 0.0]
    assert cover.essential(m, np.array([True, False, True])).tolist() == [3.0, 0, 1]


def test_explain_classifies_same_subset_and_union():
    m = _matrix([{0, 1}, {2, 3}, {0, 1}, {0}, {1, 2}])
    why = cover.explain(m, np.array([True, True, False, False, False]))
    assert why[2] == ("same", [(0, 2)])
    assert why[3] == ("subset", [(0, 1)])
    assert why[4] == ("union", [(0, 1), (1, 1)])


def test_ranges_compacts_lines_per_file():
    m = _matrix([{0, 1, 2, 4}])
    m.files, m.efile = ["a.py", "b.py"], np.array([0, 0, 0, 0, 1])
    m.eline = np.array([3, 1, 2, 5, 7])
    assert cover.ranges(m, np.unique(m.column)) == ["a.py:1-3,5", "b.py:7"]


def test_npz_round_trip(tmp_path):
    m = _matrix(DECOY, cost=[1, 2, 3], exempt=[False, True, False])
    cover.save(tmp_path / "m.npz", m)
    back = cover.load_npz(tmp_path / "m.npz")
    assert back.tests == m.tests and back.branch
    assert (back.a != m.a).nnz == 0
    assert back.cost.tolist() == [1, 2, 3] and back.exempt.tolist() == [0, 1, 0]


def _db(tmp_path, arcs):
    data = CoverageData(str(tmp_path / "c.db"))
    src, tst = str(tmp_path / "src.py"), str(tmp_path / "test_src.py")
    for ctx, body in [
        ("t.py::a|setup", {src: [1, 2]}),
        ("t.py::a|run", {src: [2, 3, 9], tst: [1]}),
        ("t.py::b|run", {src: [9]}),
        ("t.py::gone|run", {src: [5]}),
        ("", {src: [7]}),
    ]:
        data.set_context(ctx)
        if arcs:
            data.add_arcs({f: [(-1, n) for n in ns] for f, ns in body.items()})
        else:
            data.add_lines(body)
    data.write()
    return str(tmp_path / "c.db")


@pytest.mark.parametrize("arcs", [False, True])
def test_load_lines_and_arcs(tmp_path, arcs):
    m = cover.load(
        _db(tmp_path, arcs),
        tests=["t.py::a", "t.py::b"],
        durations={"t.py::a": 2.0},
        exempt=["t.py::b"],
        keep_file=lambda p: not p.endswith("test_src.py"),
        root=str(tmp_path) + "/",
    )
    assert m.branch is arcs and m.unit == ("branches" if arcs else "lines")
    assert m.tests == ["t.py::a", "t.py::b"]
    assert m.cost.tolist() == [2.0, 0.0] and m.exempt.tolist() == [False, True]
    assert m.elements == 4 and m.a.shape[1] == 2
    assert cover.ranges(m, np.unique(m.column)) == ["src.py:1-3,9"]


def test_context_nodeid_strips_only_pytest_cov_phases():
    assert cover.context_nodeid("t.py::a[x|y]|run") == "t.py::a[x|y]"
    assert cover.context_nodeid("t.py::a[x|y]") == "t.py::a[x|y]"


def test_cli_on_recorded_contexts(project, monkeypatch, capsys):
    pytester = project("""
def test_neg():
    assert classify(-1) == "neg"

def test_zero():
    assert classify(0) == "zero"

def test_all():
    assert [classify(n) for n in (-1, 0, 1)] == ["neg", "zero", "pos"]

def test_broken():
    assert classify(1) == "neg"
""")
    monkeypatch.setenv("COVERAGE_FILE", str(pytester.path / "c.db"))
    monkeypatch.setenv("COVERAGE_CORE", "ctrace")
    pytester.runpytest_subprocess(
        "-p",
        "no:pyteastrainer",
        "--cov=mod",
        "--cov-branch",
        "--cov-context=test",
        "--cov-report=",
        "--junitxml=j.xml",
    )
    capsys.readouterr()
    out = pytester.path / "r.json"
    args = [str(pytester.path / "c.db"), "--junit", str(pytester.path / "j.xml")]
    assert cover.main(args + ["--pareto", "0.5", "--report", str(out)]) == 0
    report = json.loads(out.read_text())
    assert report["matrix"]["excluded"] == ["test_mod.py::test_broken"]
    assert report["count,time"]["selected"] == ["test_mod.py::test_all"]
    assert report["count,time"]["coverage"] == 1.0
    assert report["pareto"][0]["coverage"] >= 0.5
    assert json.loads(capsys.readouterr().out)["matrix"]["excluded"] == 1


GROUP = '''
def test_a(tmp_path):
    """Doc."""
    result = classify(3, mode="x")
    assert result == "pos"

def test_b(tmp_path):
    assert classify(5, mode="y") == "pos"

def test_c(tmp_path):
    assert classify(-7, mode="z") == "pos"

def test_other_fixture(capsys):
    assert classify(9, mode="w") == "pos"

@pytest.mark.parametrize("n", [1])
def test_already(n):
    assert classify(n, mode="v") == "pos"

def test_d(tmp_path):
    assert classify(5, mode="y") == "neg"
'''


def test_suggest_parametrizes_the_differing_literals():
    fns = ast.parse(GROUP).body[:3]
    assert suggest(fns) == (
        "@pytest.mark.parametrize('p0, mode', [(3, 'x'), (5, 'y'), (-7, 'z')], "
        "ids=['test_a', 'test_b', 'test_c'])\n"
        "def test_a(tmp_path, p0, mode):\n"
        "    assert classify(p0, mode=mode) == 'pos'"
    )
    assert suggest([fns[1], fns[1]]) is None


def test_suggest_keeps_original_names_when_shapes_match():
    fns = ast.parse(
        "def t1():\n    r = f(1)\n    assert r\ndef t2():\n    q = f(2)\n    assert q\n"
    )
    assert suggest(fns.body).splitlines()[1:] == [
        "def t1(p0):",
        "    r = f(p0)",
        "    assert r",
    ]


def test_parametrize_groups_respect_arguments_and_existing_parametrize(tmp_path):
    (tmp_path / "test_g.py").write_text("import pytest\n" + GROUP)
    ids = [
        f"test_g.py::{n}"
        for n in ("test_a", "test_b", "test_other_fixture", "test_already[1]", "test_d")
    ]
    ((members, code),) = parametrize_groups(tmp_path, ids + ["test_g.py::missing"])
    assert members == ["test_g.py::test_a", "test_g.py::test_b", "test_g.py::test_d"]
    assert "ids=['test_a', 'test_b', 'test_d']" in code


def test_exact_fingerprint_ignores_local_names_and_temporaries():
    a, b, c = ast.parse(
        "def f():\n    x = g(1)\n    assert x\n"
        "def h():\n    assert g(1)\n"
        "def k():\n    assert other(1)\n"
    ).body
    assert fingerprint(a) == fingerprint(b) != fingerprint(c)
