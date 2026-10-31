import json
import sys

import pytest
from conftest import DUPES as DUPES_BODY


def test_duplicate_fails_run(pytester, project):
    project(DUPES_BODY)
    r = pytester.runpytest("--strain", "--strain-report", "r.json")
    r.assert_outcomes(passed=2)
    assert r.ret == 1
    r.stdout.fnmatch_lines(
        [
            "*duplicate*test_mod.py::test_positive_number*",
            "*FAIL: duplicates 1 > max 0*",
        ]
    )
    report = json.loads((pytester.path / "r.json").read_text())
    assert report["metrics"]["duplicates"] == 1
    assert report["findings"][0]["because_of"] == ["test_mod.py::test_pos"]


def test_clone_is_reported_not_failed_by_default(pytester, project):
    project(
        """
def test_a():
    assert classify(3) == "pos"

def test_b():
    assert classify(99) == "pos"
""",
    )
    r = pytester.runpytest("--strain")
    assert r.ret == 0
    r.stdout.fnmatch_lines(["*clone *1*"])
    r = pytester.runpytest("--strain", "--strain-max-clones", "0")
    assert r.ret == 1


def test_distinct_paths_are_essential(pytester, project):
    project(
        """
def test_neg():
    assert classify(-1) == "neg"

def test_zero():
    assert classify(0) == "zero"

def test_pos():
    assert classify(1) == "pos"
""",
    )
    r = pytester.runpytest("--strain")
    assert r.ret == 0
    r.stdout.fnmatch_lines(["*essential *3*"])


def test_exempt_marker_is_never_flagged(pytester, project):
    project(
        """
def test_pos():
    assert classify(3) == "pos"

@pytest.mark.strain_exempt(reason="deliberate smoke copy")
def test_pos_again():
    assert classify(3) == "pos"
""",
    )
    r = pytester.runpytest("--strain")
    assert r.ret == 0
    r.stdout.fnmatch_lines(["*exempt *1*"])


def test_parametrize_duplicate_row(pytester, project):
    project(
        """
@pytest.mark.parametrize("n,exp", [(1, "pos"), (2, "pos"), (1, "pos")])
def test_p(n, exp):
    assert classify(n) == exp
""",
    )
    r = pytester.runpytest("--strain", "--strain-report", "r.json")
    assert r.ret == 1
    report = json.loads((pytester.path / "r.json").read_text())
    assert report["metrics"]["duplicates"] == 1
    assert report["metrics"]["clones"] == 0
    assert [(f["kind"], f["nodeid"], f["because_of"]) for f in report["findings"]][
        0
    ] == ("duplicate", "test_mod.py::test_p[1-pos1]", ["test_mod.py::test_p[1-pos0]"])


def test_parametrize_rows_are_not_clones_of_each_other(pytester, project):
    project(
        """
@pytest.mark.parametrize("n", [1, 2, 3])
def test_p(n):
    assert classify(n) == "pos"

@pytest.mark.parametrize("n", [5])
def test_q(n):
    assert classify(n) == "pos"
""",
    )
    pytester.runpytest("--strain", "--strain-report", "r.json")
    report = json.loads((pytester.path / "r.json").read_text())
    clones = [f for f in report["findings"] if f["kind"] == "clone"]
    assert [(f["nodeid"], f["because_of"]) for f in clones] == [
        ("test_mod.py::test_q[5]", ["test_mod.py::test_p[1]"])
    ]
    assert report["metrics"]["duplicates"] == 0


def test_runs_without_cacheprovider(pytester, project):
    project(DUPES_BODY)
    r = pytester.runpytest("-p", "no:cacheprovider", "--strain")
    assert r.ret == 1
    r.stdout.fnmatch_lines(["*duplicate *1*"])
    assert not (pytester.path / ".pytest_cache").exists()


def test_tests_without_coverage_are_counted(pytester, project):
    project(
        """
def test_pos():
    assert classify(3) == "pos"

def test_nothing():
    assert 1 + 1 == 2
""",
    )
    r = pytester.runpytest("--strain", "--strain-report", "r.json")
    r.stdout.fnmatch_lines(
        ["*not analyzed   1   passed but recorded no measured coverage*"]
    )
    metrics = json.loads((pytester.path / "r.json").read_text())["metrics"]
    assert (metrics["analyzed"], metrics["not_analyzed"]) == (1, 1)
    r = pytester.runpytest("--strain", "-k", "test_pos")
    r.stdout.no_fnmatch_line("*not analyzed*")


def test_baseline_ratchet(pytester, project):
    project(DUPES_BODY)
    r = pytester.runpytest(
        "--strain", "--strain-baseline", "b.json", "--strain-update-baseline"
    )
    assert r.ret == 0
    assert (
        json.loads((pytester.path / "b.json").read_text())["metrics"]["duplicates"] == 1
    )
    # unchanged suite passes against the baseline even though duplicates > 0
    assert pytester.runpytest("--strain", "--strain-baseline", "b.json").ret == 0
    # one more copy regresses the ratchet
    project(
        DUPES_BODY + """
def test_pos_three():
    x = classify(3)
    assert x == "pos"
""",
    )
    r = pytester.runpytest("--strain", "--strain-baseline", "b.json")
    assert r.ret == 1
    r.stdout.fnmatch_lines(["*duplicates 2 > baseline 1*"])


def test_failed_tests_are_ignored(pytester, project):
    project(
        """
def test_pos():
    assert classify(3) == "pos"

def test_broken():
    assert classify(3) == "nope"
""",
    )
    r = pytester.runpytest("--strain")
    r.assert_outcomes(passed=1, failed=1)
    r.stdout.fnmatch_lines(["*analyzed 1 passed tests*"])


SUBSUMED = """
def test_neg():
    assert classify(-1) == "neg"

def test_zero_path():
    assert classify(0) == "zero"
    assert classify(-1) == "neg"
"""


def test_warn_only_keeps_exit_status(pytester, project):
    project(DUPES_BODY)
    r = pytester.runpytest("--strain", "--strain-warn-only")
    assert r.ret == 0
    r.stdout.fnmatch_lines(["*WARN: duplicates 1 > max 0*"])
    r.stdout.no_fnmatch_line("*Delete or merge*")


def test_subsumed_and_redundancy_gates(pytester, project):
    project(SUBSUMED)
    r = pytester.runpytest(
        "--strain", "--strain-max-subsumed", "0", "--strain-max-redundancy", "0.1"
    )
    assert r.ret == 1
    r.stdout.fnmatch_lines(
        [
            "*subsumed  test_mod.py::test_neg*",
            "*<= test_mod.py::test_zero_path*",
            "*FAIL: subsumed 1 > max 0*",
            "*FAIL: redundancy 50.0% > max 10.0%*",
        ]
    )


def test_ini_options_drive_the_gate(pytester, project):
    project(SUBSUMED)
    pytester.makeini(
        "[pytest]\npythonpath = .\nstrain_max_subsumed = 0\nstrain_report = out/r.json\n"
    )
    r = pytester.runpytest("--strain")
    assert r.ret == 1
    report = json.loads((pytester.path / "out" / "r.json").read_text())
    assert report["gate"]["violations"] == ["subsumed 1 > max 0"]
    assert report["mode"] == "line"


def test_show_limits_findings(pytester, project):
    project(DUPES_BODY + SUBSUMED)
    r = pytester.runpytest("--strain", "--strain-show", "1")
    r.stdout.fnmatch_lines(["*... 1 more (see --strain-report)*"])


def test_missing_baseline_falls_back_to_hard_default(pytester, project):
    project(DUPES_BODY)
    r = pytester.runpytest("--strain", "--strain-baseline", "nope.json")
    assert r.ret == 1
    r.stdout.fnmatch_lines(["*note: baseline*nope.json does not exist*"])


def test_leaner_than_baseline_suggests_update(pytester, project):
    project(DUPES_BODY)
    assert pytester.runpytest("--strain", "--strain-update-baseline").ret == 0
    baseline = pytester.path / ".strain-baseline.json"
    assert json.loads(baseline.read_text())["flagged"] == [
        "test_mod.py::test_positive_number"
    ]
    project('def test_pos():\n    assert classify(3) == "pos"\n')
    r = pytester.runpytest("--strain", "--strain-baseline", str(baseline))
    assert r.ret == 0
    r.stdout.fnmatch_lines(["*suite got leaner than the baseline*"])


def test_no_passed_tests_is_reported(pytester, project):
    project('def test_bad():\n    assert classify(3) == "neg"\n')
    r = pytester.runpytest("--strain")
    r.assert_outcomes(failed=1)
    r.stdout.fnmatch_lines(
        ["*pyteastrainer*", "*no passed tests with coverage to analyze*"]
    )


def test_branch_mode_counts_arcs(pytester, project):
    project(SUBSUMED)
    r = pytester.runpytest("--strain", "--strain-branch", "--strain-report", "r.json")
    r.stdout.fnmatch_lines(["*analyzed 2 passed tests over * branches*"])
    assert json.loads((pytester.path / "r.json").read_text())["mode"] == "branch"


def test_exempt_copy_is_not_a_duplicate(pytester, project):
    project(
        DUPES_BODY + '\n@pytest.mark.strain_exempt(reason="x")\ndef test_again():\n'
        '    assert classify(3) == "pos"\n',
    )
    pytester.runpytest("--strain", "--strain-report", "r.json")
    report = json.loads((pytester.path / "r.json").read_text())
    assert report["metrics"]["exempt"] == 1
    assert [(f["nodeid"], f["because_of"]) for f in report["findings"]] == [
        ("test_mod.py::test_positive_number", ["test_mod.py::test_pos"])
    ]


def test_xdist_is_rejected(pytester, project):
    project(DUPES_BODY)
    r = pytester.runpytest("--strain", "-n", "2")
    assert r.ret == pytest.ExitCode.USAGE_ERROR
    r.stderr.fnmatch_lines(["*does not support xdist*"])


def test_pytest_cov_contexts_are_reused(pytester, project, monkeypatch):
    monkeypatch.setenv("COVERAGE_CORE", "ctrace")
    project(DUPES_BODY)
    r = pytester.runpytest_subprocess(
        "--strain", "--strain-report", "r.json", "--cov=.", "--cov-context=test"
    )
    assert r.ret == 1
    r.stdout.fnmatch_lines(["*FAIL: duplicates 1 > max 0*"])
    assert json.loads((pytester.path / "r.json").read_text())["source"] == "pytest-cov"


def test_pytest_cov_without_contexts_is_rejected(pytester, project):
    project(DUPES_BODY)
    r = pytester.runpytest_subprocess("--strain", "--cov=.")
    assert r.ret == pytest.ExitCode.USAGE_ERROR
    r.stderr.fnmatch_lines(["*not recording per-test contexts*"])


def test_union_cover_is_redundant(pytester, project):
    project(
        """
def test_neg_zero():
    assert classify(-1) == "neg"
    assert classify(0) == "zero"

def test_zero_pos():
    assert classify(0) == "zero"
    assert classify(1) == "pos"

def test_neg_pos():
    assert classify(-1) == "neg"
    assert classify(1) == "pos"
""",
    )
    r = pytester.runpytest("--strain")
    assert r.ret == 0
    r.stdout.fnmatch_lines(["*suite needs 2 (redundancy 33.3%)*", "*redundant *3*"])


def test_pytest_cov_sysmon_core_is_rejected(pytester, project, monkeypatch):
    if not hasattr(sys, "monitoring"):
        pytest.skip("sys.monitoring needs Python 3.12+")
    monkeypatch.setenv("COVERAGE_CORE", "sysmon")
    project(DUPES_BODY)
    r = pytester.runpytest_subprocess("--strain", "--cov=.", "--cov-context=test")
    assert r.ret == pytest.ExitCode.USAGE_ERROR
    r.stderr.fnmatch_lines(["*sys.monitoring core*COVERAGE_CORE=ctrace*"])
