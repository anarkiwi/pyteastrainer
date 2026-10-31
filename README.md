# pyteastrainer


Strains the redundant tests out of a pytest suite.
A pytest plugin that **fails the run when the suite has grown redundant**, so an
agent cannot land a test that proves nothing new.

```
pip install -e .                 # not on PyPI yet
pytest --strain
```

```
================================= pyteastrainer =================================
  analyzed 22 passed tests over 24 lines; a coverage-equivalent suite needs 9 (redundancy 59.1%)
  duplicate      3   identical coverage and body              -> delete
  clone          3   same path & assertions, other literals   -> parametrize
  subsumed       7   strict subset of one other test          -> review/delete
  redundant      0   nothing beyond the union of others       -> review
  essential      8   covers something nothing else does       -> keep
  exempt         1   @pytest.mark.strain_exempt                -> keep

  duplicate tests/test_cart.py::test_add_single_item
            == tests/test_cart.py::test_add_new_item
  ...
  FAIL: duplicates 3 > max 0
```

Exit status 1. The run's own tests still pass; the *gate* failed.

## Why not TestIQ / recov?

Both exist and both find the same thing with coverage contexts. They are
analyzers you run afterwards and read. This is a gate: it runs inside pytest,
sets the exit status, and ratchets against a baseline, so it can live in CI
and in an agent's pre-commit loop. It also looks at the test body, not just
coverage, which is what separates "copy-pasted" from "same path, different
oracle".

## How it decides

For every **passed** test it records two things:

1. **Coverage fingerprint** — the set of production lines (or branch arcs with
   `--strain-branch`) the test executed. Test files, `conftest.py`, venvs and
   site-packages are excluded, so a test's own body never makes it "unique".
2. **Static fingerprint** — its body with docstrings dropped, single-use
   temporaries inlined, identifiers alpha-renamed; once with literals kept
   (`exact`) and once with literals abstracted to their type (`shape`).
   Parametrized instances mix their runtime parameter values into `exact`.

Then each test gets at most one label, strongest first:

| label | meaning | default gate |
|---|---|---|
| `duplicate` | same coverage **and** same `exact` body as a kept sibling | fail if > 0 |
| `clone` | same coverage and same `shape` as another test function; only literals differ | report |
| `subsumed` | coverage is a strict subset of **one** other test | report |
| `redundant` | nothing beyond the union of all other tests | report |
| `essential` | covers at least one element nothing else does | keep |

It also computes a **greedy minimal coverage-equivalent suite** and reports
`redundancy = 1 - minimal/analyzed`. On a 3,000-test synthetic suite the
analysis adds well under a second.

Only `duplicate` fails by default, because the other three have honest false
positives: a boundary test (`total() at exactly 100`) takes the same path as a
bigger test and is still worth keeping. Those are what the thresholds, the
ratchet and the marker are for.

## The ratchet (the part that stops agents)

On an existing suite you don't want to fail on day one. Record where you are:

```
pytest --strain --strain-baseline .strain-baseline.json --strain-update-baseline
```

Commit the file. From then on, plain `pytest --strain --strain-baseline .strain-baseline.json`
fails if **any** count (`duplicates`, `clones`, `subsumed`, `redundant`) or the
redundancy ratio is worse than the baseline, and names the tests that are new
since the baseline. When the suite gets leaner it tells you to re-run
`--strain-update-baseline` to lock the gain in. The baseline can only move down
unless a human regenerates it.

With a baseline present the `duplicates > 0` default is suspended (the
baseline owns the gate); pass `--strain-max-duplicates` explicitly to keep both.

## Configuration

CLI flags or `pytest.ini` / `pyproject.toml` `[tool.pytest.ini_options]`:

```ini
[pytest]
addopts = --strain --strain-baseline .strain-baseline.json --strain-report strain-report.json
strain_source = src/mypkg          # default: rootdir minus test files
strain_branch = true               # branch arcs instead of lines
strain_max_duplicates = 0
strain_max_clones = 10
strain_max_subsumed = 50
strain_max_redundancy = 0.4        # 1 - minimal_suite/analyzed
strain_tolerance = 0.01            # slack over the baseline ratio
strain_show = 20                   # findings printed in the terminal
```

`--strain-warn-only` reports without touching the exit status.

Mark a test that asserts something coverage can't see (rounding, ordering,
an error message, a boundary) and it is never flagged, never used to flag a
twin, and always kept in the minimal suite:

```python
@pytest.mark.strain_exempt(reason="checks cent rounding, same path as test_total")
def test_total_rounds_to_cents(): ...
```

## For agents

`--strain-report strain-report.json` writes the full result:

```json
{
  "metrics": {"analyzed": 22, "not_analyzed": 0, "duplicates": 3, "clones": 3,
              "subsumed": 7, "redundant": 0, "essential": 8, "minimal_suite": 9,
              "redundancy_ratio": 0.5909},
  "gate": {"passed": false, "violations": ["duplicates 3 > max 0"]},
  "findings": [
    {"nodeid": "tests/test_cart.py::test_add_single_item",
     "kind": "duplicate",
     "because_of": ["tests/test_cart.py::test_add_new_item"],
     "detail": "identical coverage and identical body (modulo names)",
     "elements": 7, "unique_elements": 0}
  ],
  "kept_for": {"tests/test_cart.py::test_add_new_item": ["tests/test_cart.py::test_add_single_item"]}
}
```

A sensible agent instruction: *"If the pyteastrainer gate fails, delete every
`duplicate`, merge `clone` groups into one `parametrize`, and justify or delete
`subsumed` tests. Do not add `strain_exempt` to make the gate pass."*

## Working with pytest-cov

If pytest-cov is active, pyteastrainer does not start a second tracer; it reads
pytest-cov's data. You must record contexts:

```
pytest --strain --cov=src --cov-context=test [--cov-branch]
```

Without `--cov-context=test` it refuses to run rather than silently analyze
nothing.

## Mutation mode (prototype)

```
pip install -e .[mutants]
pytest --strain-mutants --strain-source src/pkg
```

Coverage says where a test went, not what it checked. `--strain-mutants`
loads the modules under `--strain-source` through
[mutmut](https://github.com/boxed/mutmut)'s trampolines, records which tests
reach which function, then runs every reached mutant in a forked child against
only those tests. A test's elements become the mutants it kills, so the same
categories mean: `duplicate` kills the same mutants with the same body,
`subsumed` kills a strict subset of another test's mutants, `essential` kills
one nothing else does. Tests that reach mutants but kill none are listed as
`weak`.

| Option | Meaning |
|---|---|
| `--strain-jobs N` | concurrent mutant processes (default: CPU count) |
| `--strain-timeout S` | base seconds per mutant, plus ten times its tests' runtime; a timeout counts as a kill |

Limits: decorated functions (including numba `@njit`) are not mutated; a
verdict only covers code inside `--strain-source`; `-n` is refused because the
mode forks its own workers.

## Limitations

- No `pytest-xdist` yet (`-n` is refused). Per-worker data would need combining.
- Coverage is not semantics. Two tests on the same path with different
  assertions are both reported as `clone`/`subsumed`; use the marker, the
  thresholds or the ratchet rather than treating every finding as deletable.
- Only passed tests are analyzed; a failing test's coverage is partial. Passed
  tests that execute no measured source are counted as `not analyzed`.
- Static fingerprints come from the test's own function body. Tests whose
  behaviour lives entirely in fixtures look alike to the `exact`/`shape`
  comparison; coverage still separates them.

## Demo

```
cd demo
pytest --strain                                   # fails: 3 duplicates
pytest --strain --strain-branch --strain-max-clones 0   # also fails on clones
```

`demo/tests/test_cart.py` has a legit block and a labelled BLOAT block so you
can see which finding each pattern produces.

## Development

```
pip install -e .[test,mutants]
pytest -n auto --cov
black --check . && pylint pyteastrainer tests
```

Releases: [docs/releasing.md](docs/releasing.md).
