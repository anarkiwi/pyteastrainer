# Adopting pyteastrainer on an existing suite

An established suite usually comes back RED with most of its tests listed. Gate
it in this order: make the measurement trustworthy, triage what it finds, then
gate new findings against a baseline.

## 1. Record once, with nothing failing

```
pytest -n auto --cov=SRC --cov-branch --cov-context=test \
  --strain --strain-parametrize=warn --strain-show 0 --strain-report strain.json
```

Run it in the environment CI uses. `strain.json` holds every action, the kept
suite and the slowest essential tests; read it rather than the terminal.

## 2. Make what a test reaches a property of the test

The gate compares per-test coverage, so that coverage has to be the same
whatever ran first and on every run. Check this before acting on findings:
record twice with different `-n`, then diff each test's arcs between the two
coverage databases (`context` and `arc` tables; strip pytest-cov's `|run`
suffix). Any test whose arcs differ needs one of these fixes.

| Symptom | Cause | Fix |
|---|---|---|
| A module's top-level lines credited to whichever test imported it first | lazy imports | import every package module at collection, in `conftest.py`, only when `config.getoption("--strain")` is set, so jobs without every optional extra still collect |
| A module- or session-scoped fixture's lines credited to its first test | fixture scope | run with `--dist loadfile`, so a file's tests stay together and in order at any `-n` |
| A cached function's body credited to its first caller | `functools.lru_cache` and hand-rolled memo dicts | an autouse fixture that calls `cache_clear()` on each module-level cache and clears module dicts that are empty on import |
| A once-only branch (`if not _LOADED:`) credited once per worker | module-level flags | reset them in the same fixture |
| Branches that vary run to run inside one test | seeds from the OS | draw seeds through one helper that is fixed by an environment variable plus the test id when the gated job sets it, and from the OS otherwise; keep one job that runs with OS seeds |
| Arcs that vary with timing | polling loops, signals, deadlines | leave them; with a baseline they matter only if a kept test's only unique arcs are timing arcs |

A test that covers nothing usually tests code outside the measured source, such
as scripts. Add that source to `--cov` before treating the test as redundant.

## 3. Triage

Do these in order, because forced tests change which others are redundant.

1. **Exempt** what coverage cannot judge, and record again: conformance vectors
   from another implementation, rows of a standard's table, one case per rule,
   and tests that read source (AST rules). Mark them
   `@pytest.mark.strain_exempt(reason="...")`.
2. **Same-branch pairs** (`same`, `duplicate`): if the flagged test checks a
   different input or property, fold that check into its twin; otherwise delete
   it.
3. **Parametrize cases** (`case`): cut interchangeable samples (seeds, SNRs,
   offsets, lengths) to the cases in `kept`, plus any boundary the test names,
   and reword an "every" the test no longer checks.
4. **Slow essential tests** (`slow` in the report): shrink their inputs to the
   smallest scenario that still reaches the lines only they reach. Check those
   lines with a single-test coverage run before and after.
5. **Subset and union** findings take the most judgement. A test whose branches
   others reach can still be the only one asserting a behaviour; fold that
   assertion in before deleting it.

Coverage is blind to values: two cases that take the same branches with
different data look identical to it. When unsure between trimming and exempting,
exempt.

## 4. Gate new findings

```
pytest ... --cov-context=test --strain --strain-update-baseline   # once, in CI's configuration
```

Commit `.strain-baseline.json` and point `strain_baseline` at it. CI then fails
only on findings outside it. Generate the baseline with CI's exact command:
worker count, seed variable and `COVERAGE_CORE=ctrace`. Pick the seed by running
the whole suite under it first; a statistical test can fail under one fixed seed
and pass under the next.

- A test that fails during the recording has no coverage context, so it is
  missing from the baseline, and the next green run reports a finding or a
  "fixed" entry for it. Record only from a run with nothing failing.
- The image CI runs must contain the baseline file (add it to the Dockerfile's
  `COPY`) and every file the tests read, or CI collects a different suite.

- Equal-coverage tests are told apart by runtime, so two recordings can accept
  different members of a pair. The solver minimises baseline tests kept before
  anything else, so it usually keeps the unaccepted member and the gate stays
  green. A slower CI runner can still surface a new finding from a flip; exempt
  the test if it checks a contract of its own, otherwise regenerate the baseline.
- Parametrize merge suggestions are not baselined; set `strain_parametrize = warn`
  until they are dealt with.
- When the run says baseline entries are fixed, rerun with
  `--strain-update-baseline` and commit, so the ratchet only tightens.
