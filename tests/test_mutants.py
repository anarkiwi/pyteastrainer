import json

import pytest
from conftest import DUPES


def _run(project, tests: str, *args):
    pytester = project(tests)
    r = pytester.runpytest(
        "--strain-mutants", "--strain-jobs", "4", "--strain-report", "r.json", *args
    )
    return r, json.loads((pytester.path / "r.json").read_text())


BROAD_BUT_BLIND = """
def test_broad():
    assert len([classify(n) for n in (-1, 0, 1)]) == 3

def test_focused():
    assert classify(1) == "pos"
"""


def test_coverage_subset_with_a_stronger_oracle_is_essential(project):
    r, report = _run(project, BROAD_BUT_BLIND)
    assert r.ret == 0
    assert report["mode"] == "mutant"
    assert report["metrics"]["essential"] == 1
    assert report["metrics"]["subsumed"] == 0
    assert report["mutants"]["weak"] == ["test_mod.py::test_broad"]
    r.stdout.fnmatch_lines(
        ["*essential *1*kills a mutant nothing else does*", "*weak *test_broad*"]
    )


def test_coverage_mode_calls_the_same_focused_test_subsumed(project):
    pytester = project(BROAD_BUT_BLIND)
    r = pytester.runpytest("--strain", "--strain-report", "r.json")
    report = json.loads((pytester.path / "r.json").read_text())
    assert [f["nodeid"] for f in report["findings"] if f["kind"] == "subsumed"] == [
        "test_mod.py::test_focused"
    ]
    r.stdout.fnmatch_lines(["*analyzed 2 passed tests over * lines*"])


def test_equal_kills_with_one_body_is_a_duplicate(project):
    r, report = _run(project, DUPES)
    assert r.ret == 1
    assert report["metrics"]["duplicates"] == 1
    r.stdout.fnmatch_lines(
        ["*identical kills and body*", "*FAIL: duplicates 1 > max 0*"]
    )


def test_a_mutant_that_hangs_times_out_and_counts_as_a_kill(project):
    r, report = _run(
        project,
        """
def test_count():
    assert count_to(3) == 3
""",
        "--strain-timeout",
        "1",
    )
    assert r.ret == 0
    assert report["mutants"]["timeouts"] >= 1
    assert report["mutants"]["killed"] == report["metrics"]["elements"] > 0
    assert report["mutants"]["weak"] == []


@pytest.mark.parametrize(
    "args,message",
    [
        (("--strain",), "*separate modes*"),
        (("-n", "2"), "*forks its own workers*"),
    ],
)
def test_incompatible_options_are_rejected(project, args, message):
    pytester = project("def test_x():\n    pass\n")
    r = pytester.runpytest("--strain-mutants", *args)
    assert r.ret == pytest.ExitCode.USAGE_ERROR
    r.stderr.fnmatch_lines([message])
