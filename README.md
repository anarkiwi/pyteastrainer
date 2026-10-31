# pyteastrainer

A pytest plugin that fails the run when tests add no coverage, are too slow,
or should be one parametrized test, and prints the exact changes that make it
green. The minimal suite is solved exactly as an integer program (HiGHS).

```
pip install git+https://github.com/anarkiwi/pyteastrainer
pytest --strain
```

## Output

```
============================== pyteastrainer: RED ==============================
25 passed tests cover 41 branches; 9 of them (0.01 s of 0.02 s) cover all of it. To make teastrainer green:
  delete tests/test_cart.py::test_bad_coupon_rejected (duplicate of tests/test_cart.py::test_unknown_coupon)
  delete tests/test_cart.py::test_coupon_is_stored: every branch it reaches is reached by tests/test_cart.py::test_pct_coupon
  drop cases [0-1.0], [1--1.0] from the parametrize of tests/test_cart.py::test_add_rejects: ...
  parametrize: merge tests/test_cart.py::test_abs_coupon_floors_at_zero, test_pct_coupon into one test:
      @pytest.mark.parametrize('p0, p1, p2', [(2.0, 'FIVEOFF', 5.0), (200.0, 'TEN', 180.0)], ids=[...])
      def test_abs_coupon_floors_at_zero(p0, p1, p2):
          ...
Removing these leaves a suite with the same coverage, 9 tests, 0.01 s, that teastrainer passes.
```

Apply every action and rerun. A test that asserts something coverage cannot
see is marked `@pytest.mark.strain_exempt(reason="...")` and always kept.
`cd demo && pytest --strain` reproduces the full output.

## Options

| Option (ini) | Default | Meaning |
|---|---|---|
| `--strain` | off | Record per-test coverage and gate |
| `--strain-source PATH` (`strain_source`) | rootdir minus tests | Production code to measure |
| `--strain-lines` (`strain_lines`) | branches | Measure lines instead of branch arcs |
| `--strain-order` (`strain_order`) | `count,time` | Fewest tests then least runtime, or `time,count` |
| `--strain-max-time S` (`strain_max_time`) | off | Tests slower than S are RED |
| `--strain-parametrize` (`strain_parametrize`) | `fail` | Parametrize merge suggestions: `fail`, `warn`, `off` |
| `--strain-baseline FILE` (`strain_baseline`) | none | Accepted findings; only new ones fail |
| `--strain-update-baseline` | | Write current findings to the baseline |
| `--strain-time-limit S` (`strain_time_limit`) | 60 | HiGHS seconds per solver stage |
| `--strain-report FILE` (`strain_report`) | none | JSON report |
| `--strain-show N` / `--strain-show-slow N` | 20 / 3 | Action lines / advisory speed lines |
| `--strain-warn-only` | | Report without changing the exit status |
| `--strain-select` | | Run only the minimal suite of the last recording |
| `--strain-select-coverage F` (`strain_select_coverage`) | 1 | With select: least runtime reaching coverage fraction F |

## pytest-cov and xdist

```
pytest -n auto --cov=src --cov-branch --cov-context=test --strain
```

Per-test contexts are required. On Python 3.14 set `COVERAGE_CORE=ctrace`.

## Docs

- [docs/algorithm.md](docs/algorithm.md): formulations, convergence, parametrize detection, benchmark
- [docs/releasing.md](docs/releasing.md)

## Development

```
pip install -e .[test]
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 pytest -n auto --cov
black --check . && pylint pyteastrainer tests
```
