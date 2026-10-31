"""Minimal test suite as an exact set cover over per-test coverage, solved with HiGHS.

Loads a coverage.py per-test-context database into a tests x elements matrix, then
finds lexicographically optimal covers, threshold partial covers and explanations.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import xml.etree.ElementTree as ET
from contextlib import closing
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp
from tqdm import tqdm

PHASES = ("setup", "run", "teardown")
ORDERS = (("count", "time"), ("time", "count"))


@dataclass
class Matrix:
    """Tests x merged-elements incidence; ``weight`` counts original elements per column.

    ``efile``/``eline``/``column`` give each original element's file, line and merged column.
    """

    tests: List[str]
    cost: np.ndarray
    a: sparse.csr_matrix
    weight: np.ndarray
    exempt: np.ndarray
    branch: bool = True
    files: List[str] = field(default_factory=list)
    efile: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    eline: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    column: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    excluded: List[str] = field(default_factory=list)

    @property
    def elements(self) -> int:
        return int(self.weight.sum())

    @property
    def unit(self) -> str:
        return "branches" if self.branch else "lines"


@dataclass
class Solution:
    selected: np.ndarray
    status: str
    gap: float
    optimal: bool


def context_nodeid(name: str) -> str:
    """Strip pytest-cov's ``|setup``/``|run``/``|teardown`` suffix from a context name."""
    base, sep, phase = name.rpartition("|")
    return base if sep and phase in PHASES else name


def _nodeid_key(nodeid: str) -> str:
    path, _, rest = nodeid.partition("::")
    parts = rest.split("::")
    return (
        ".".join([path.removesuffix(".py").replace("/", ".")] + parts[:-1])
        + "::"
        + parts[-1]
    )


def read_junit(path: str) -> Tuple[Dict[str, float], set]:
    """Per-test durations keyed like ``_nodeid_key``, and keys of tests that did not pass."""
    times: Dict[str, float] = {}
    bad = set()
    for case in ET.parse(path).iter("testcase"):
        key = f"{case.get('classname')}::{case.get('name')}"
        times[key] = times.get(key, 0.0) + float(case.get("time") or 0.0)
        if any(case.find(tag) is not None for tag in ("failure", "error", "skipped")):
            bad.add(key)
    return times, bad


def _unpack_lines(blobs: Sequence[bytes]) -> Tuple[np.ndarray, np.ndarray]:
    """Row index and line number of every set bit in a list of little-endian numbits."""
    sizes = np.fromiter(map(len, blobs), np.int64, len(blobs)) * 8
    bits = np.unpackbits(np.frombuffer(b"".join(blobs), np.uint8), bitorder="little")
    pos = bits.nonzero()[0]
    start = np.concatenate([[0], np.cumsum(sizes)])
    row = np.searchsorted(start, pos, side="right") - 1
    return row, pos - start[row]


def load(
    data_file: str,
    *,
    tests: Optional[Iterable[str]] = None,
    durations: Optional[Dict[str, float]] = None,
    exempt: Iterable[str] = (),
    keep_file: Optional[Callable[[str], bool]] = None,
    junit: Optional[str] = None,
    root: Optional[str] = None,
) -> Matrix:
    """Read a coverage.py database recorded with one dynamic context per test.

    Rows are restricted to ``tests`` (and to junit passes); files to ``keep_file``.
    Elements are arcs when the database has arcs, else lines.
    """
    with closing(sqlite3.connect(f"file:{data_file}?mode=ro", uri=True)) as con:
        ctx = con.execute("SELECT id, context FROM context").fetchall()
        files = con.execute("SELECT id, path FROM file").fetchall()
        branch = bool(con.execute("SELECT EXISTS(SELECT 1 FROM arc)").fetchone()[0])
        rows = con.execute(
            "SELECT context_id, file_id, fromno, tono FROM arc"
            if branch
            else "SELECT context_id, file_id, numbits FROM line_bits"
        ).fetchall()
    times, bad = read_junit(junit) if junit else ({}, set())
    wanted = None if tests is None else set(tests)
    names = [context_nodeid(c) for _, c in ctx]
    excluded = sorted({n for n in names if n and _nodeid_key(n) in bad})
    ok = [
        bool(n) and (wanted is None or n in wanted) and _nodeid_key(n) not in bad
        for n in names
    ]
    test_names, tix = np.unique(
        np.array([n for n, k in zip(names, ok) if k], dtype=object).astype(str),
        return_inverse=True,
    )
    ctx_ids = np.array([i for i, _ in ctx], np.int64)
    ctx_test = np.full(np.r_[0, ctx_ids].max() + 1, -1, np.int64)
    ctx_test[ctx_ids[np.array(ok, bool)]] = tix.ravel()
    fids = np.array([i for i, _ in files], np.int64)
    file_ok = np.zeros(np.r_[0, fids].max() + 1, bool)
    file_ok[fids] = [keep_file is None or keep_file(p) for _, p in files]
    paths = [""] * len(file_ok)
    for i, p in files:
        paths[i] = os.path.relpath(p, root) if root and p.startswith(root) else p

    c = np.fromiter((r[0] for r in rows), np.int64, len(rows))
    f = np.fromiter((r[1] for r in rows), np.int64, len(rows))
    keep = (ctx_test[c] >= 0) & file_ok[f] if len(rows) else np.zeros(0, bool)
    if branch:
        arcs = np.array([r[2:] for r in rows], np.int64).reshape(-1, 2)[keep]
        t, f = ctx_test[c[keep]], f[keep]
        cols = np.column_stack([f, arcs])
        line = np.where(arcs[:, 0] > 0, arcs[:, 0], np.abs(arcs[:, 1]))
    else:
        r, line = _unpack_lines([row[2] for row, k in zip(rows, keep) if k])
        t, f = ctx_test[c[keep]][r], f[keep][r]
        cols = np.column_stack([f, line])
    _, first, e = np.unique(cols, axis=0, return_index=True, return_inverse=True)
    e = e.ravel()
    n, ne = len(test_names), len(first)
    pair = np.unique(t * ne + e)
    a = sparse.csr_matrix(
        (np.ones(len(pair), bool), (pair // max(ne, 1), pair % max(ne, 1))),
        shape=(n, ne),
    )
    present = np.unique(t)
    a = a[present]
    names_out = [str(s) for s in test_names[present]]
    src = durations if durations is not None else {}
    cost = np.array(
        [src.get(s, times.get(_nodeid_key(s), 0.0)) for s in names_out], float
    )
    exempt_set = set(exempt)
    merged, weight, column = merge_columns(a)
    return Matrix(
        names_out,
        cost,
        merged,
        weight,
        np.array([s in exempt_set for s in names_out], bool),
        branch=branch,
        files=paths,
        efile=f[first],
        eline=line[first],
        column=column,
        excluded=excluded,
    )


def merge_columns(a) -> Tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    """Collapse elements covered by exactly the same tests; covering one covers all.

    Columns are grouped by random 64-bit row-weight sums and the grouping is verified
    exactly, adding hashes until no group mixes distinct columns.
    """
    csc = sparse.csc_matrix(a, dtype=bool)
    rng = np.random.default_rng(0)
    keys = np.diff(csc.indptr)[:, None].astype(np.uint64)
    while True:
        h = rng.integers(
            0, np.iinfo(np.uint64).max, csc.shape[0], np.uint64, endpoint=True
        )
        keys = np.column_stack([keys, csc.T @ h])
        _, first, inv, counts = np.unique(
            keys, axis=0, return_index=True, return_inverse=True, return_counts=True
        )
        inv = inv.ravel()
        if (csc[:, first[inv]] != csc).nnz == 0:
            return csc[:, first].tocsr(), counts.astype(float), inv


def essential(m: Matrix, kept: Optional[np.ndarray] = None) -> np.ndarray:
    """Weighted count of elements each (kept) test covers and no other (kept) test does."""
    a = m.a if kept is None else sparse.diags(kept.astype(float)) @ m.a
    count = np.asarray(a.sum(axis=0)).ravel()
    return np.asarray(a @ (m.weight * (count == 1))).ravel()


def _lexicographic(
    objectives, constraints, integrality, lb, time_limit: float, desc: str
) -> Tuple[np.ndarray, str, float, bool]:
    x, messages, gap, optimal = None, [], 0.0, True
    for c in tqdm(objectives, desc=desc, file=sys.stderr, disable=None, leave=False):
        res = milp(
            c,
            constraints=constraints,
            integrality=integrality,
            bounds=Bounds(lb, 1),
            options={"time_limit": time_limit, "mip_rel_gap": 0.0, "disp": False},
        )
        if res.x is None:
            raise RuntimeError(f"HiGHS returned no solution: {res.message}")
        x = np.where(integrality > 0, np.round(res.x), res.x)
        messages.append(res.message)
        gap = max(gap, float(getattr(res, "mip_gap", 0.0) or 0.0))
        optimal &= res.status == 0
        constraints = constraints + [LinearConstraint(c[None, :], -np.inf, c @ x)]
    return x, messages[-1] if optimal else "; ".join(messages), gap, optimal


def min_cover(
    m: Matrix,
    order: Sequence[str] = ("count", "time"),
    time_limit: float = 60.0,
    baseline: Optional[np.ndarray] = None,
    slow: Optional[np.ndarray] = None,
) -> Solution:
    """Keep every element covered; lexicographically minimise slow tests kept, baseline
    tests kept, then ``order``. Exempt tests are forced in. When every other test is
    essential the full suite is the only cover, so no solve is needed.
    """
    n = len(m.tests)
    if (essential(m)[~m.exempt] > 0).all():
        return Solution(np.ones(n, bool), "every test essential", 0.0, True)
    stages = [slow, baseline] + [
        {"count": np.ones(n), "time": m.cost}[o] for o in order
    ]
    objectives = [np.asarray(c, float) for c in stages if c is not None and c.any()]
    x, status, gap, optimal = _lexicographic(
        objectives,
        [LinearConstraint(m.a.T.astype(float).tocsr(), lb=1)],
        np.ones(n),
        m.exempt.astype(float),
        time_limit,
        "pyteastrainer cover",
    )
    return Solution(x > 0.5, status, gap, optimal)


def partial_cover(m: Matrix, fraction: float, time_limit: float = 60.0) -> Solution:
    """Least runtime, then fewest tests, covering at least ``fraction`` of element weight.

    x binary per test, y in [0, 1] per merged column, y <= A^T x, w.y >= fraction * sum(w).
    """
    n, e = m.a.shape
    zeros = np.zeros(e)
    x, status, gap, optimal = _lexicographic(
        [np.concatenate([m.cost, zeros]), np.concatenate([np.ones(n), zeros])],
        [
            LinearConstraint(
                sparse.hstack([-m.a.T.astype(float), sparse.identity(e)]).tocsr(),
                ub=0,
            ),
            LinearConstraint(
                np.concatenate([np.zeros(n), m.weight])[None, :],
                lb=fraction * m.weight.sum(),
            ),
        ],
        np.concatenate([np.ones(n), zeros]),
        np.concatenate([m.exempt.astype(float), zeros]),
        time_limit,
        "pyteastrainer partial cover",
    )
    return Solution(x[:n] > 0.5, status, gap, optimal)


def explain(
    m: Matrix, kept: np.ndarray
) -> Dict[int, Tuple[str, List[Tuple[int, int]]]]:
    """Why each dropped test is redundant given ``kept``: ``same``, ``subset`` or ``union``.

    Returns ``(kind, [(kept_test, elements), ...])``; ``same`` lists every identical kept
    test, ``subset`` the smallest superset, ``union`` a greedy cover of the test's elements.
    """
    k_idx, d_idx = np.flatnonzero(kept), np.flatnonzero(~kept)
    ak = m.a[k_idx].astype(float).tocsc()
    size = m.a.getnnz(axis=1)
    over = (m.a[d_idx].astype(float) @ ak.T).tocsr()
    out: Dict[int, Tuple[str, List[Tuple[int, int]]]] = {}
    steps = tqdm(d_idx, desc="pyteastrainer explain", file=sys.stderr, disable=None)
    for i, t in enumerate(steps):
        cols = m.a.indices[m.a.indptr[t] : m.a.indptr[t + 1]]
        total = int(m.weight[cols].sum())
        row = over[i]
        sup = row.indices[row.data == size[t]]
        if sup.size:
            sizes = size[k_idx[sup]]
            same = sup[sizes == size[t]]
            if same.size:
                out[t] = ("same", [(int(k_idx[j]), total) for j in same])
            else:
                out[t] = ("subset", [(int(k_idx[sup[np.argmin(sizes)]]), total)])
            continue
        sub = ak[:, cols].tocsr()
        left = m.weight[cols].copy()
        picks = []
        while left.any():
            gain = sub @ left
            j = int(np.argmax(gain))
            if gain[j] == 0:
                break
            picks.append((int(k_idx[j]), int(gain[j])))
            left[sub.indices[sub.indptr[j] : sub.indptr[j + 1]]] = 0
        out[t] = ("union", picks)
    return out


def ranges(m: Matrix, cols: np.ndarray) -> List[str]:
    """Decode merged columns to compact ``file:a-b,c`` source ranges, one per file."""
    sel = np.isin(m.column, cols)
    pairs = np.unique(np.column_stack([m.efile[sel], m.eline[sel]]), axis=0)
    f, line = pairs[:, 0], pairs[:, 1]
    new_file = np.concatenate([[True], f[1:] != f[:-1]])
    start = np.flatnonzero(
        new_file | np.concatenate([[True], line[1:] != line[:-1] + 1])
    )
    end = np.concatenate([start[1:], [len(line)]]) - 1
    out: Dict[int, List[str]] = {}
    for s, e in zip(start, end):
        span = f"{line[s]}-{line[e]}" if line[e] != line[s] else f"{line[s]}"
        out.setdefault(int(f[s]), []).append(span)
    return [f"{m.files[k]}:{','.join(v)}" for k, v in out.items()]


def greedy(m: Matrix) -> np.ndarray:
    """Chvatal's weighted greedy: repeatedly take the max of new elements per runtime."""
    a = m.a.astype(float)
    left = m.weight.copy()
    chosen = np.zeros(len(m.tests), bool)
    while left.any():
        gain = np.asarray(a @ left).ravel()
        gain[chosen] = 0
        ratio = np.divide(
            gain, m.cost, out=np.where(gain > 0, np.inf, 0.0), where=m.cost > 0
        )
        best = np.flatnonzero(ratio == ratio.max())
        t = best[np.argmax(gain[best])]
        chosen[t] = True
        left[a.indices[a.indptr[t] : a.indptr[t + 1]]] = 0
    return chosen


def summary(m: Matrix, sel: np.ndarray) -> dict:
    covered = float(m.weight[np.asarray(m.a[sel].sum(axis=0)).ravel() > 0].sum())
    return {
        "tests": int(sel.sum()),
        "runtime": round(float(m.cost[sel].sum()), 3),
        "coverage": round(covered / max(m.weight.sum(), 1.0), 6),
    }


def save(path, m: Matrix) -> None:
    """Store the rows, costs, exemptions and merged CSR matrix for select mode."""
    np.savez_compressed(
        path,
        tests=np.array(m.tests, dtype=str),
        cost=m.cost,
        exempt=m.exempt,
        weight=m.weight,
        indptr=m.a.indptr,
        indices=m.a.indices,
        shape=np.array(m.a.shape),
        branch=m.branch,
    )


def load_npz(path) -> Matrix:
    with np.load(path) as z:
        a = sparse.csr_matrix(
            (np.ones(len(z["indices"]), bool), z["indices"], z["indptr"]),
            shape=tuple(z["shape"]),
        )
        return Matrix(
            list(map(str, z["tests"])),
            z["cost"],
            a,
            z["weight"],
            z["exempt"],
            branch=bool(z["branch"]),
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "data_file", help="coverage.py database recorded with --cov-context=test"
    )
    p.add_argument(
        "--junit", help="junit XML of the same run: durations, and failures to exclude"
    )
    p.add_argument(
        "--order", default="count,time", choices=[",".join(o) for o in ORDERS]
    )
    p.add_argument(
        "--time-limit", type=float, default=60.0, help="HiGHS seconds per stage"
    )
    p.add_argument(
        "--pareto", default="", help="coverage fractions to reach, e.g. 0.95,0.99"
    )
    p.add_argument("--report", help="write the JSON report here")
    args = p.parse_args(argv)

    m = load(args.data_file, junit=args.junit)
    sol = min_cover(m, args.order.split(","), args.time_limit)
    out = {
        "matrix": {
            "tests": len(m.tests),
            "elements": m.elements,
            "merged_elements": m.a.shape[1],
            "excluded": m.excluded,
        },
        "full": summary(m, np.ones(len(m.tests), bool)),
        "greedy": summary(m, greedy(m)),
        args.order: summary(m, sol.selected)
        | {"status": sol.status, "gap": sol.gap, "optimal": sol.optimal},
        "pareto": [
            {"coverage_target": f}
            | summary(m, partial_cover(m, f, args.time_limit).selected)
            for f in map(float, filter(None, args.pareto.split(",")))
        ],
    }
    out[args.order]["selected"] = [t for t, s in zip(m.tests, sol.selected) if s]
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
    brief = dict(out)
    brief["matrix"] = out["matrix"] | {"excluded": len(m.excluded)}
    brief[args.order] = {k: v for k, v in out[args.order].items() if k != "selected"}
    print(json.dumps(brief, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
