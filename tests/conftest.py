"""Shared pytester project; imports heavy modules before pytester snapshots ``sys.modules``."""

import coverage  # pylint: disable=unused-import
import pytest

import pyteastrainer.plugin  # pylint: disable=unused-import

pytest_plugins = ["pytester"]

SRC = """
def classify(n):
    if n < 0:
        return "neg"
    if n == 0:
        return "zero"
    return "pos"


def count_to(n):
    i = 0
    while i < n:
        i += 1
    return i
"""

DUPES = """
def test_pos():
    assert classify(3) == "pos"

def test_positive_number():
    result = classify(3)
    assert result == "pos"
"""


@pytest.fixture
def project(pytester):
    def make(tests: str):
        pytester.makepyfile(mod=SRC)
        pytester.makeini("[pytest]\npythonpath = .\n")
        pytester.makepyfile(
            test_mod="import pytest\nfrom mod import classify, count_to\n" + tests
        )
        return pytester

    return make
