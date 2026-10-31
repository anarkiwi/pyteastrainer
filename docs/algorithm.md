# Algorithm

## Model

A run produces a boolean matrix `A` (tests × elements). Rows are tests that
passed in every phase (not xfail); elements are branch arcs when the data has
arcs, else lines, restricted to production files (test files, `conftest.py`,
site-packages and the plugin are excluded). A test's duration `d_t` is the sum
of its setup, call and teardown report durations.

Line data is decoded without per-row Python loops: the numbits blobs of all
rows are concatenated, unpacked once with
`np.unpackbits(..., bitorder="little")`, and bit positions are mapped back to
rows with `searchsorted` over the blob offsets.

### Column merging

Two elements reached by exactly the same tests are interchangeable in every
formulation below: covering one covers the other. Columns are grouped by
random 64-bit row-weight sums (wrapping integer arithmetic, one sparse
mat-vec per hash), and the grouping is then verified exactly by comparing each
column with its group representative; while any group mixes distinct
columns another hash is added. Each merged column `e` carries a weight `w_e`,
the number of original elements it stands for. Per-original-element arrays
(file, line, merged column) are kept to decode results into `file:a-b` ranges.

## Minimum cover

Variables `x_t ∈ {0, 1}`; exempt tests and nothing else are fixed to 1.

```
A^T x ≥ 1                      every element of the full suite stays covered
```

The objectives are solved lexicographically, each stage adding
`c_k · x ≤ c_k · x*_k` for the previous optimum before minimising the next:

1. `Σ_{t slow} x_t` — only with `--strain-max-time`; slow means `d_t` above the limit.
2. the configured order: `Σ x_t` then `Σ d_t x_t` (`count,time`), or the reverse.
3. `Σ_{t ∈ baseline} x_t` — only with a baseline: among optimal covers, keep
   the tests that are not already accepted, so accepted tests are dropped first.

Every stage runs HiGHS with `mip_rel_gap = 0`. The drop list is `{t : x_t = 0}`.
If `--strain-time-limit` stops a stage, the incumbent still satisfies
`A^T x ≥ 1`, so the drop list is still safe to apply; the report says
"not proven minimal" and gives the gap.

### GREEN ⇔ every test is essential

Let `u_t = Σ_e w_e A_te [Σ_s A_se = 1]`, the weight of elements only `t` covers
(`unique = A @ (count == 1)`).

* If every non-exempt test has `u_t > 0`, each one is forced by some element,
  and exempt tests are forced by their bounds, so `x = 1` is the only feasible
  point and nothing is dropped. The plugin returns this without calling the
  solver (whatever the baseline or slow sets, since there is nothing to choose).
* If some non-exempt `t` has `u_t = 0`, `x = 1 - e_t` is feasible. It is
  strictly better on the count objective and no worse on time and slow count,
  so every lexicographic optimum drops at least one test: RED.

### One-step convergence

Let `S` be the kept set of an optimal solve. Deleting the drop list leaves
exactly `S`, which covers every element. If some `t ∈ S` had `u_t = 0` within
`S`, then `S \ {t}` would cover everything with no more slow tests, no more
runtime and one test fewer, contradicting the optimality of `S` at whichever
stage first separates them. So every test of `S` is essential, and by the
previous section the rerun is GREEN regardless of timing noise (with
`--strain-max-time`, a test that newly crosses the limit is a new finding). The slow stage
comes first precisely so that dropping a test can never increase the slow
count, which keeps this argument valid.

Parametrize merges are proposed only among kept tests, after the deletes. A
merged test keeps one row per old test, with the same body and the same
coverage, so every row is still essential and the merged file is a fixed point
as well.

## Partial cover (select mode)

`--strain-select-coverage F` with `F < 1` solves

```
x_t ∈ {0,1},  y_e ∈ [0,1]
y_e ≤ Σ_t A_te x_t           (y ≤ A^T x)
Σ_e w_e y_e ≥ F Σ_e w_e
minimise Σ d_t x_t, then Σ x_t
```

which is pytest-maxcov's coverage-threshold selection, solved optimally.
`F = 1` uses the minimum cover above. Select mode is deterministic (same
matrix, same solve), so the controller and every xdist worker agree.

## Explanations

For each dropped test `t`, with kept set `K`:

* `same` — some kept row has exactly `t`'s columns; it is a `duplicate` when the
  exact AST fingerprint of the bodies also matches.
* `subset` — the smallest kept row containing all of `t`'s columns.
* `union` — otherwise, a greedy cover of `t`'s columns by kept rows (largest
  remaining weight first). This is only an explanation, so greedy suffices; the
  overlap counts come from one sparse product `A[drop] · A[K]^T`.

A slow test that the cover must keep is reported with the elements it alone
among fast tests reaches: `A[t] ∧ ¬(A[fast].sum(0) > 0)`, decoded to ranges.

## Parametrize detection

Kept, non-exempt, unparametrized test functions are grouped by
(module, class scope, argument list, shape fingerprint). The shape fingerprint
is the body with docstrings dropped, single-use temporaries inlined, locally
bound names alpha-renamed, and constants replaced by their type. Within a
group the constant nodes align in `ast.walk` order; positions whose values
differ become parameters (one per distinct value tuple, named after the
keyword they are passed to when there is one). The first member's original
body is used when the members' un-inlined shapes match, otherwise the inlined
form, and old function names become the `ids`.

## Baseline and CLI

`--strain-baseline FILE` lists accepted redundant and slow tests; a final
solver stage minimises the kept baseline tests, so accepted tests are dropped
in preference to new ones and only findings outside the baseline fail.
`python -m pyteastrainer.cover .coverage --junit junit.xml --pareto 0.95,0.99`
runs the solver offline on a recorded database.

## Benchmark

sigmfalyzer: 3049 passed tests with coverage, 27,884 branches merged to
2,947 columns.

| Selection | Tests | Runtime |
|---|---|---|
| Full suite | 3049 | 2884 s |
| Greedy (maxcov's algorithm) | 860 | 909 s |
| Exact, count first | 555 | 829 s |
| Exact, time first | 570 | 790 s |

All exact runs solved in about 20 s with gap 0. Partial coverage from the
runtime-budget sweep: 5% of the runtime reaches 97.9% of branches, 10% reaches
99.6%.

pytest-maxcov selects with Chvátal's greedy rule, the "Greedy" row above, and
gives no optimality bound; pyteastrainer reports the HiGHS status and gap.
