import ast
import json
import re
import sys

import pytest

from conftest import DUPES

OFF = ("--strain-parametrize", "off")

SUITE = {
    "test_neg": 'def test_neg():\n    assert classify(-1) == "neg"\n',
    "test_zero": 'def test_zero():\n    assert classify(0) == "zero"\n',
    "test_all": (
        "def test_all():\n"
        '    assert [classify(n) for n in (-1, 0, 1)] == ["neg", "zero", "pos"]\n'
    ),
    "test_count": "def test_count():\n    assert count_to(3) == 3\n",
    "test_count_zero": "def test_count_zero():\n    assert count_to(0) == 0\n",
    "test_cases": (
        '@pytest.mark.parametrize("n", [1, 2])\n'
        'def test_cases(n):\n    assert classify(n) == "pos"\n'
    ),
}


def _report(pytester, name="r.json"):
    return json.loads((pytester.path / name).read_text())


def _without(source: str, names) -> str:
    tree = ast.parse(source)
    tree.body = [
        n for n in tree.body if not (isinstance(n, ast.FunctionDef) and n.name in names)
    ]
    return ast.unparse(tree)


def test_red_lists_actions_and_deleting_them_turns_green(project):
    pytester = project("\n".join(SUITE.values()))
    r = pytester.runpytest("--strain", "--strain-report", "r.json", *OFF)
    r.assert_outcomes(passed=7)
    assert r.ret == 1
    r.stdout.fnmatch_lines(
        [
            "*pyteastrainer: RED*",
            "7 passed tests cover * branches; 2 of them (* of *) cover all of it. "
            "To make teastrainer green:",
            "  delete test_mod.py::test_cases (all 2 cases): every branch they reach "
            "is reached by test_mod.py::test_all",
            "  delete test_mod.py::test_count_zero: every branch it reaches is reached "
            "by test_mod.py::test_count",
            "  delete test_mod.py::test_neg: every branch it reaches is reached by "
            "test_mod.py::test_all",
            "Removing these leaves a suite with the same coverage, 2 tests, *",
            "speed (advisory)*",
        ]
    )
    report = _report(pytester)
    assert report["kept"] == ["test_mod.py::test_all", "test_mod.py::test_count"]
    assert report["metrics"]["minimal_tests"] == 2 and report["metrics"]["optimal"]
    assert report["gate"] == {
        "passed": False,
        "violations": [
            "test_mod.py::test_cases[1]",
            "test_mod.py::test_cases[2]",
            "test_mod.py::test_count_zero",
            "test_mod.py::test_neg",
            "test_mod.py::test_zero",
        ],
    }
    gone = {a["nodeid"].split("::")[1].split("[")[0] for a in report["actions"]}
    src = (pytester.path / "test_mod.py").read_text()
    (pytester.path / "test_mod.py").write_text(_without(src, gone))
    r = pytester.runpytest("--strain", *OFF)
    assert r.ret == 0
    r.stdout.fnmatch_lines(
        [
            "*pyteastrainer: GREEN*",
            "every one of the 2 tests covers something no other test does",
        ]
    )


def test_duplicate_action(project):
    pytester = project(DUPES + 'def test_neg():\n    assert classify(-1) == "neg"\n')
    r = pytester.runpytest("--strain", *OFF)
    assert r.ret == 1
    r.stdout.fnmatch_lines(
        ["  delete test_mod.py::test_pos* (duplicate of test_mod.py::test_pos*)"]
    )


def test_union_action(project):
    pytester = project("""
def test_neg_zero():
    assert classify(-1) == "neg" and classify(0) == "zero"

def test_zero_pos():
    assert classify(0) == "zero" and classify(1) == "pos"

def test_neg_pos():
    assert classify(-1) == "neg" and classify(1) == "pos"
""")
    r = pytester.runpytest("--strain", *OFF)
    assert r.ret == 1
    r.stdout.fnmatch_lines(
        [
            "  delete test_mod.py::test_*: its * branches are all reached by "
            "test_mod.py::test_* (*), test_mod.py::test_* (*)",
        ]
    )


def test_same_coverage_different_body(project):
    pytester = project(
        'def test_a():\n    assert classify(3) == "pos"\n'
        'def test_b():\n    assert classify(4) != "neg"\n'
    )
    r = pytester.runpytest("--strain", *OFF)
    r.stdout.fnmatch_lines(
        ["*delete test_mod.py::test_?: same branches as test_mod.py::test_?"]
    )


def test_no_coverage_test_is_a_violation(project):
    pytester = project(
        'def test_pos():\n    assert classify(1) == "pos"\n'
        "def test_nothing():\n    assert True\n"
    )
    r = pytester.runpytest("--strain", "--strain-report", "r.json")
    assert r.ret == 1
    r.stdout.fnmatch_lines(
        [
            "  test_mod.py::test_nothing covers no measured source: add the code it "
            "tests to --strain-source / --cov, or delete it"
        ]
    )
    report = _report(pytester)
    assert report["no_coverage"] == ["test_mod.py::test_nothing"]
    assert report["actions"][0]["kind"] == "no_coverage"


def test_exempt_test_with_no_coverage_is_kept(project):
    pytester = project(
        'def test_pos():\n    assert classify(1) == "pos"\n'
        '@pytest.mark.strain_exempt(reason="reads source")\n'
        "def test_nothing():\n    assert True\n"
    )
    r = pytester.runpytest("--strain", "--strain-report", "r.json")
    assert r.ret == 0
    report = _report(pytester)
    assert report["no_coverage"] == [] and report["gate"]["passed"]


def test_exempt_twin_makes_the_copy_redundant(project):
    pytester = project(
        DUPES.replace(
            "def test_pos():", '@pytest.mark.strain_exempt(reason="x")\ndef test_pos():'
        )
    )
    r = pytester.runpytest("--strain", "--strain-report", "r.json", *OFF)
    assert r.ret == 1
    assert _report(pytester)["kept"] == ["test_mod.py::test_pos"]
    r.stdout.fnmatch_lines(["*delete test_mod.py::test_positive_number (duplicate of*"])


def test_failed_and_xfailed_tests_are_ignored(project):
    pytester = project("""
def test_pos():
    assert classify(1) == "pos"

def test_broken():
    assert classify(1) == "neg"

@pytest.mark.xfail
def test_x():
    assert classify(2) == "neg"
""")
    r = pytester.runpytest("--strain")
    r.assert_outcomes(passed=1, failed=1, xfailed=1)
    assert r.ret == 1
    r.stdout.fnmatch_lines(["*pyteastrainer: GREEN*"])


def test_warn_only_keeps_exit_status(project):
    pytester = project(DUPES)
    r = pytester.runpytest("--strain", "--strain-warn-only")
    assert r.ret == 0
    r.stdout.fnmatch_lines(["*pyteastrainer: RED (warn only)*"])


def test_show_truncates_action_lines(project):
    pytester = project(DUPES + 'def test_three():\n    assert classify(3) == "pos"\n')
    r = pytester.runpytest("--strain", "--strain-show", "1")
    r.stdout.fnmatch_lines(["  ... 1 more (pass --strain-report FILE for all)"])
    r = pytester.runpytest(
        "--strain", "--strain-show", "1", "--strain-report", "x.json"
    )
    r.stdout.fnmatch_lines(["  ... 1 more in x.json"])


def test_baseline_cycle(project):
    pytester = project(DUPES)
    r = pytester.runpytest(
        "--strain", "--strain-baseline", "b.json", "--strain-update-baseline"
    )
    assert r.ret == 0
    doc = json.loads((pytester.path / "b.json").read_text())
    assert len(doc["accepted"]) == 1 and doc["slow"] == []
    r = pytester.runpytest("--strain", "--strain-baseline", "b.json")
    assert r.ret == 0
    r.stdout.fnmatch_lines(["1 flagged tests remain, all accepted in the baseline"])

    src = (pytester.path / "test_mod.py").read_text()
    (pytester.path / "test_mod.py").write_text(
        src + '\ndef test_more():\n    assert classify(9) == "pos"\n'
    )
    r = pytester.runpytest(
        "--strain", "--strain-baseline", "b.json", "--strain-report", "r.json"
    )
    assert r.ret == 1
    assert len(_report(pytester)["gate"]["violations"]) == 1

    accepted = doc["accepted"][0].split("::")[1]
    pytester.makepyfile(test_mod=_without(src, {accepted}))
    r = pytester.runpytest("--strain", "--strain-baseline", "b.json")
    assert r.ret == 0
    r.stdout.fnmatch_lines(
        ["*1 baseline entries are fixed; lock in with --strain-update-baseline"]
    )


def test_missing_baseline_is_noted(project):
    pytester = project(DUPES)
    r = pytester.runpytest("--strain", "--strain-baseline", "nope.json")
    assert r.ret == 1
    r.stdout.fnmatch_lines(["*baseline * does not exist*"])


def test_select_deselects_outside_the_optimal_suite(project):
    pytester = project("\n".join(SUITE.values()))
    pytester.runpytest("--strain", *OFF)
    r = pytester.runpytest("--strain-select")
    r.assert_outcomes(passed=2, deselected=5)
    r.stdout.fnmatch_lines(
        [
            "pyteastrainer select: running 2 of 7 recorded tests, * of *, 100.00% of branches"
        ]
    )
    assert "pyteastrainer: " not in r.stdout.str()
    r = pytester.runpytest("--strain-select", "--strain-select-coverage", "0.5")
    outcomes = r.parseoutcomes()
    assert outcomes["passed"] == 1 and outcomes["deselected"] == 6
    r = pytester.runpytest("--strain-select", "--strain", *OFF)
    r.stdout.fnmatch_lines(["*pyteastrainer: GREEN*"])


def test_select_without_recording(project):
    pytester = project(DUPES)
    r = pytester.runpytest("--strain-select")
    r.assert_outcomes(passed=2)
    r.stdout.fnmatch_lines(["pyteastrainer select: no recording; run --strain first"])


def test_max_time_drops_replaceable_and_flags_irreplaceable(project):
    pytester = project("""
import time

def test_slow_but_replaceable():
    time.sleep(0.3)
    assert classify(-1) == "neg"

def test_fast_neg():
    assert classify(-2) == "neg"

def test_slow_only_count():
    time.sleep(0.3)
    assert count_to(2) == 2

def test_zero_pos():
    assert classify(0) == "zero" and classify(1) == "pos"
""")
    args = ("--strain", "--strain-max-time", "0.2", "--strain-report", "r.json", *OFF)
    r = pytester.runpytest(*args)
    assert r.ret == 1
    r.stdout.fnmatch_lines(
        [
            "  too slow: test_mod.py::test_slow_only_count takes * > max 0.20 s; only slow "
            "tests reach mod.py:10-13 — write a test under 0.20 s that reaches these, "
            "then delete test_mod.py::test_slow_only_count",
            "  delete test_mod.py::test_slow_but_replaceable (exceeds max time): same "
            "branches as test_mod.py::test_fast_neg",
        ]
    )
    report = _report(pytester)
    assert not any(s["nodeid"].endswith("slow_only_count") for s in report["slow"])
    pytester.runpytest(*args, "--strain-baseline", "b.json", "--strain-update-baseline")
    assert json.loads((pytester.path / "b.json").read_text())["slow"] == [
        "test_mod.py::test_slow_only_count"
    ]


def test_parametrize_suggestion_applies_and_turns_green(project):
    pytester = project('''
def test_neg():
    assert classify(-1) == "neg"

def test_neg_again():
    assert classify(-5) == "neg"

def test_zero():
    """Zero."""
    result = classify(0)
    assert result == "zero"

def test_pos():
    assert classify(3) == "pos"
''')
    r = pytester.runpytest("--strain", "--strain-report", "r.json")
    assert r.ret == 1
    r.stdout.fnmatch_lines(
        [
            "  parametrize: merge test_mod.py::test_neg*, test_pos, test_zero into one test:"
        ]
    )
    report = _report(pytester)
    ((merge,),) = [report["parametrize"]]
    assert len(merge["members"]) == 3
    assert not {"test_mod.py::test_neg", "test_mod.py::test_neg_again"} <= set(
        merge["members"]
    )
    names = {a["nodeid"].split("::")[1] for a in report["actions"]}
    names |= {m.split("::")[1] for m in merge["members"]}
    src = _without((pytester.path / "test_mod.py").read_text(), names)
    pytester.makepyfile(test_mod=src + "\n\n" + merge["code"] + "\n")
    r = pytester.runpytest("--strain")
    assert r.ret == 0, r.stdout.str()
    r.stdout.fnmatch_lines(["*pyteastrainer: GREEN*"])
    r = pytester.runpytest("--strain", "--strain-parametrize", "warn")
    assert r.ret == 0


def test_parametrize_warn_does_not_fail(project):
    pytester = project(
        'def test_a():\n    assert classify(-1) == "neg"\n'
        'def test_b():\n    assert classify(1) == "pos"\n'
    )
    assert pytester.runpytest("--strain").ret == 1
    r = pytester.runpytest("--strain", "--strain-parametrize", "warn")
    assert r.ret == 0
    r.stdout.fnmatch_lines(
        [
            "*pyteastrainer: GREEN*",
            "  parametrize: merge test_mod.py::test_a, test_b into one test:",
            "      @pytest.mark.parametrize('p0, p1', [(-1, 'neg'), (1, 'pos')], "
            "ids=['test_a', 'test_b'])",
            "      def test_a(p0, p1):",
            "          assert classify(p0) == p1",
        ]
    )


def test_lines_mode_and_ini(project):
    pytester = project(DUPES)
    pytester.makeini(
        "[pytest]\npythonpath = .\nstrain_lines = true\nstrain_source = mod\n"
        "strain_show = 0\nstrain_order = time,count\n"
    )
    r = pytester.runpytest("--strain")
    assert r.ret == 1
    r.stdout.fnmatch_lines(
        ["2 passed tests cover 3 lines; 1 of them*", "  ... 1 more*"]
    )
    assert "delete" not in r.stdout.str()


def test_no_cacheprovider_still_gates(project):
    pytester = project(DUPES)
    r = pytester.runpytest("--strain", "-p", "no:cacheprovider")
    assert r.ret == 1
    r = pytester.runpytest("--strain-select", "-p", "no:cacheprovider")
    r.assert_outcomes(passed=2)


@pytest.mark.parametrize(
    "args,message",
    [
        (("--strain-order", "speed"), "*--strain-order must be*"),
        (("--strain-select-coverage", "0"), "*--strain-select-coverage must be*"),
        (("--strain-parametrize", "maybe"), "*--strain-parametrize must be*"),
        (("-n", "2"), "*add --cov=SRC --cov-branch --cov-context=test*"),
    ],
)
def test_usage_errors(project, args, message):
    pytester = project(DUPES)
    r = pytester.runpytest_subprocess("--strain", *args)
    assert r.ret == pytest.ExitCode.USAGE_ERROR
    r.stderr.fnmatch_lines([message])


def test_pytest_cov_without_contexts_is_rejected(project):
    pytester = project(DUPES)
    r = pytester.runpytest_subprocess("--strain", "--cov=mod")
    r.stderr.fnmatch_lines(["*not recording per-test contexts*"])


@pytest.mark.skipif(sys.version_info < (3, 12), reason="sys.monitoring needs 3.12")
def test_pytest_cov_sysmon_core_is_rejected(project, monkeypatch):
    monkeypatch.setenv("COVERAGE_CORE", "sysmon")
    pytester = project(DUPES)
    r = pytester.runpytest_subprocess("--strain", "--cov=mod", "--cov-context=test")
    r.stderr.fnmatch_lines(["*sys.monitoring core*"])


def test_gate_under_xdist_with_pytest_cov(project, monkeypatch):
    monkeypatch.setenv("COVERAGE_CORE", "ctrace")
    pytester = project("\n".join(SUITE.values()))
    r = pytester.runpytest_subprocess(
        "-n",
        "2",
        "--cov=mod",
        "--cov-branch",
        "--cov-context=test",
        "--cov-report=",
        "--strain",
        "--strain-report",
        "r.json",
        *OFF,
    )
    assert r.ret == 1
    r.stdout.fnmatch_lines(
        ["*pyteastrainer: RED*", "*delete test_mod.py::test_count_zero*"]
    )
    report = _report(pytester)
    assert report["source"] == "pytest-cov"
    assert report["kept"] == ["test_mod.py::test_all", "test_mod.py::test_count"]
    assert re.search(r"7 passed tests cover", r.stdout.str())
