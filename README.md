# pyteastrainer

A pytest plugin that fails the run when the suite contains tests that add no
coverage, and says exactly which tests to remove or change to make it green.
It records per-test coverage, solves the exact minimum covering suite with
HiGHS, and gates on every test outside it.

```
pip install git+https://github.com/anarkiwi/pyteastrainer
pytest --strain
```

## Example

```
============================== pyteastrainer: RED ==============================
25 passed tests cover 41 branches; 9 of them (0.01 s of 0.02 s) cover all of it. To make teastrainer green:
  drop cases [0-1.0], [1--1.0] from the parametrize of tests/test_cart.py::test_add_rejects: every branch they reach is reached by tests/test_cart.py::test_add_rejects[-1-1.0], tests/test_cart.py::test_add_rejects[1--5.0]
  delete tests/test_cart.py::test_bad_coupon_rejected (duplicate of tests/test_cart.py::test_unknown_coupon)
  delete tests/test_cart.py::test_coupon_is_stored: every branch it reaches is reached by tests/test_cart.py::test_pct_coupon
  delete tests/test_cart.py::test_total_adds_shipping_under_100: same branches as tests/test_cart.py::test_total_rounds_to_cents
  delete tests/test_cart.py::test_total_free_shipping_at_100: its 15 branches are all reached by tests/test_cart.py::test_pct_coupon (14), tests/test_cart.py::test_total_rounds_to_cents (1)
  ...
  parametrize: merge tests/test_cart.py::test_abs_coupon_floors_at_zero, test_pct_coupon into one test:
      @pytest.mark.parametrize('p0, p1, p2', [(2.0, 'FIVEOFF', 5.0), (200.0, 'TEN', 180.0)], ids=['test_abs_coupon_floors_at_zero', 'test_pct_coupon'])
      def test_abs_coupon_floors_at_zero(p0, p1, p2):
          ...
Removing these leaves a suite with the same coverage, 9 tests, 0.01 s, that teastrainer passes. If a test asserts something coverage can't see, mark it @pytest.mark.strain_exempt(reason=...).
speed (advisory): slowest tests in the minimal suite; a faster test reaching these makes it redundant
  0.00 s (16%) tests/test_cart.py::test_add_rejects[-1-1.0]: 2 branches only it reaches: shop/cart.py:26-27
```

What to do: apply every action line (delete the test, drop the parametrize
cases, paste the merged test), then rerun. The result is a minimum cover in
which every test reaches something no other test does, so it is GREEN and
stays GREEN. A test that asserts something coverage cannot see gets
`@pytest.mark.strain_exempt(reason="...")`: it is always kept and may make
others redundant. `cd demo && pytest --strain` reproduces the output above.

## Options

| Option (ini) | Default | Meaning |
|---|---|---|
| `--strain` | off | Record per-test coverage and gate |
| `--strain-source PATH` (`strain_source`) | rootdir minus tests | Production code to measure |
| `--strain-lines` (`strain_lines`) | branches | Measure lines instead of branch arcs |
| `--strain-order` (`strain_order`) | `count,time` | Fewest tests then least runtime, or `time,count` |
| `--strain-time-limit S` (`strain_time_limit`) | 60 | HiGHS seconds per solver stage |
| `--strain-max-time S` (`strain_max_time`) | off | Tests slower than S are RED; the cover avoids them where it can |
| `--strain-parametrize` (`strain_parametrize`) | `fail` | Same-shape kept tests: suggest a parametrize merge (`fail`, `warn`, `off`) |
| `--strain-baseline FILE` (`strain_baseline`) | none | Accepted redundant / slow tests (ratchet) |
| `--strain-update-baseline` | | Write the current findings to the baseline; never fails |
| `--strain-report FILE` (`strain_report`) | none | Full JSON report |
| `--strain-show N` / `--strain-show-slow N` | 20 / 3 | Action lines / advisory speed lines |
| `--strain-warn-only` | | Report without changing the exit status |
| `--strain-select` | | Run only the optimal suite of the last `--strain` recording |
| `--strain-select-coverage F` (`strain_select_coverage`) | 1 | With select: least runtime reaching fraction F of coverage |

The exit status becomes 1 on RED; an existing failure status is never changed.

## pytest-cov and xdist

With pytest-cov active, pyteastrainer reads its data instead of tracing. Per-test
contexts are required, and are the only way to run under xdist:

```
pytest -n auto --cov=src --cov-branch --cov-context=test --strain
```

On Python 3.14 set `COVERAGE_CORE=ctrace` (the sys.monitoring core cannot record
contexts). The controller combines worker data and runs the gate.

## Legacy suites

```
pytest --strain --strain-baseline .strain-baseline.json --strain-update-baseline
pytest --strain --strain-baseline .strain-baseline.json
```

Only findings not in the baseline fail. When baseline entries disappear the
run says so; lock the gain in with `--strain-update-baseline`.

## Compared to pytest-maxcov

pytest-maxcov picks tests with Chvátal's greedy rule, which can keep far more
tests than needed and gives no bound on how far off it is. pyteastrainer solves
the set cover exactly as an integer program and reports the HiGHS status and
gap, so a minimal suite is proven minimal. Because the gate fails exactly when
some test covers nothing unique, deleting the reported tests gives a stable
GREEN state rather than a moving target. It is a gate with an exit status and
per-test actions, not only a selector, it runs under xdist through pytest-cov,
and its threshold select mode (`--strain-select-coverage`) solves maxcov's
coverage-fraction selection optimally.

## More

- [docs/algorithm.md](docs/algorithm.md): formulations, convergence, benchmark
- [docs/releasing.md](docs/releasing.md)
- CLI on a recorded database: `python -m pyteastrainer.cover .coverage --junit junit.xml --pareto 0.95,0.99`

## Development

```
pip install -e .[test]
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 pytest -n auto --cov
black --check . && pylint pyteastrainer tests
```
