import json

import pytest

pytest_plugins = ["pytester"]

SRC = '''
def classify(n):
    if n < 0:
        return "neg"
    if n == 0:
        return "zero"
    return "pos"
'''


def _project(pytester, tests: str):
    pytester.makepyfile(mod=SRC)
    pytester.makeini("[pytest]\npythonpath = .\n")
    pytester.makepyfile(test_mod="import pytest\nfrom mod import classify\n" + tests)


def test_duplicate_fails_run(pytester):
    _project(pytester, '''
def test_pos():
    assert classify(3) == "pos"

def test_positive_number():
    result = classify(3)
    assert result == "pos"
''')
    r = pytester.runpytest_subprocess("--strain", "--strain-report", "r.json")
    r.assert_outcomes(passed=2)
    assert r.ret == 1
    r.stdout.fnmatch_lines(["*duplicate*test_mod.py::test_positive_number*", "*FAIL: duplicates 1 > max 0*"])
    report = json.loads((pytester.path / "r.json").read_text())
    assert report["metrics"]["duplicates"] == 1
    assert report["findings"][0]["because_of"] == ["test_mod.py::test_pos"]


def test_clone_is_reported_not_failed_by_default(pytester):
    _project(pytester, '''
def test_a():
    assert classify(3) == "pos"

def test_b():
    assert classify(99) == "pos"
''')
    r = pytester.runpytest_subprocess("--strain")
    assert r.ret == 0
    r.stdout.fnmatch_lines(["*clone *1*"])
    r = pytester.runpytest_subprocess("--strain", "--strain-max-clones", "0")
    assert r.ret == 1


def test_distinct_paths_are_essential(pytester):
    _project(pytester, '''
def test_neg():
    assert classify(-1) == "neg"

def test_zero():
    assert classify(0) == "zero"

def test_pos():
    assert classify(1) == "pos"
''')
    r = pytester.runpytest_subprocess("--strain")
    assert r.ret == 0
    r.stdout.fnmatch_lines(["*essential *3*"])


def test_exempt_marker_is_never_flagged(pytester):
    _project(pytester, '''
def test_pos():
    assert classify(3) == "pos"

@pytest.mark.strain_exempt(reason="deliberate smoke copy")
def test_pos_again():
    assert classify(3) == "pos"
''')
    r = pytester.runpytest_subprocess("--strain")
    assert r.ret == 0
    r.stdout.fnmatch_lines(["*exempt *1*"])


def test_parametrize_duplicate_row(pytester):
    _project(pytester, '''
@pytest.mark.parametrize("n,exp", [(1, "pos"), (2, "pos"), (1, "pos")])
def test_p(n, exp):
    assert classify(n) == exp
''')
    r = pytester.runpytest_subprocess("--strain")
    assert r.ret == 1
    r.stdout.fnmatch_lines(["*duplicate *1*", "*clone *1*"])


def test_baseline_ratchet(pytester):
    _project(pytester, '''
def test_pos():
    assert classify(3) == "pos"

def test_positive_number():
    result = classify(3)
    assert result == "pos"
''')
    r = pytester.runpytest_subprocess("--strain", "--strain-baseline", "b.json", "--strain-update-baseline")
    assert r.ret == 0
    assert json.loads((pytester.path / "b.json").read_text())["metrics"]["duplicates"] == 1
    # unchanged suite passes against the baseline even though duplicates > 0
    assert pytester.runpytest_subprocess("--strain", "--strain-baseline", "b.json").ret == 0
    # one more copy regresses the ratchet
    _project(pytester, '''
def test_pos():
    assert classify(3) == "pos"

def test_positive_number():
    result = classify(3)
    assert result == "pos"

def test_pos_three():
    x = classify(3)
    assert x == "pos"
''')
    r = pytester.runpytest_subprocess("--strain", "--strain-baseline", "b.json")
    assert r.ret == 1
    r.stdout.fnmatch_lines(["*duplicates 2 > baseline 1*"])


def test_failed_tests_are_ignored(pytester):
    _project(pytester, '''
def test_pos():
    assert classify(3) == "pos"

def test_broken():
    assert classify(3) == "nope"
''')
    r = pytester.runpytest_subprocess("--strain")
    r.assert_outcomes(passed=1, failed=1)
    r.stdout.fnmatch_lines(["*analyzed 1 passed tests*"])
