"""Redundancy analysis over per-test coverage fingerprints.

Input: for each passed test, the set of coverage *elements* it executed
(lines, or arcs when branch tracing is on) in the production source, plus its
static fingerprint.  Output: a classified list of findings and a metrics
block suitable for gating and ratcheting.

Classification (strongest first; each test gets at most one):

  duplicate   same coverage AND same exact fingerprint as a kept sibling.
              Pure copy.  Safe to delete.
  clone       same coverage AND same shape fingerprint (different literals)
              as a kept test of another function.  Same assertion down the
              same path - one parametrized test.  Rows of one parametrized
              function are never clones of each other.
  subsumed    coverage is a strict subset of ONE other test's coverage.
              It cannot fail for any reason the bigger test can't also fail
              for, unless it asserts something the bigger one doesn't.
  redundant   every element it covers is covered by some other test, but no
              single test subsumes it.  Deleting it costs no coverage.
  essential   covers at least one element nothing else covers.

Only ``duplicate`` is a hard default gate; the others are reported and can
be gated by threshold or ratchet.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set

from .fingerprint import StaticFingerprint


@dataclass
class TestRecord:
    nodeid: str
    elements: FrozenSet[int]
    static: Optional[StaticFingerprint]
    exempt: bool = False
    duration: float = 0.0

    @property
    def base(self) -> str:
        """nodeid with the parametrize id stripped."""
        return self.nodeid.split("[", 1)[0]


@dataclass
class Finding:
    nodeid: str
    kind: str  # duplicate | clone | subsumed | redundant
    because_of: List[str]  # the sibling(s) that make it redundant
    detail: str
    elements: int
    unique_elements: int = 0


@dataclass
class Metrics:
    tests: int = 0
    analyzed: int = 0
    not_analyzed: int = 0
    exempt: int = 0
    duplicates: int = 0
    clones: int = 0
    subsumed: int = 0
    redundant: int = 0
    essential: int = 0
    minimal_suite: int = 0
    elements: int = 0
    redundancy_ratio: float = 0.0  # 1 - minimal_suite / analyzed

    def as_dict(self) -> Dict[str, float]:
        return dict(self.__dict__)


@dataclass
class Analysis:
    metrics: Metrics
    findings: List[Finding] = field(default_factory=list)
    kept_for: Dict[str, List[str]] = field(
        default_factory=dict
    )  # kept nodeid -> flagged siblings


def _prefer_keep(t: TestRecord) -> tuple:
    """Sort key: which test survives a duplicate/clone cluster."""
    n_asserts = t.static.n_asserts if t.static else 0
    return (-n_asserts, "[" in t.nodeid, t.nodeid)


def analyze(tests: List[TestRecord]) -> Analysis:
    m = Metrics(tests=len(tests))
    recs = [t for t in tests if t.elements]
    m.analyzed = len(recs)
    m.not_analyzed = m.tests - m.analyzed
    m.exempt = sum(t.exempt for t in recs)
    an = Analysis(metrics=m)
    if not recs:
        return an

    # Inverted index: element -> set of test indices covering it.
    index: Dict[int, Set[int]] = defaultdict(set)
    for i, t in enumerate(recs):
        for e in t.elements:
            index[e].add(i)
    m.elements = len(index)

    flagged: Dict[int, Finding] = {}

    # --- 1. duplicate / clone clusters (identical coverage) -----------------
    by_cov: Dict[FrozenSet[int], List[int]] = defaultdict(list)
    for i, t in enumerate(recs):
        by_cov[t.elements].append(i)

    for members in by_cov.values():
        if len(members) < 2:
            continue
        # exact clusters first
        by_exact: Dict[str, List[int]] = defaultdict(list)
        by_shape: Dict[str, List[int]] = defaultdict(list)
        for i in members:
            s = recs[i].static
            if s is None or recs[i].exempt:
                continue
            by_exact[s.exact].append(i)
            by_shape[s.shape].append(i)

        for group in by_exact.values():
            if len(group) < 2:
                continue
            group.sort(key=lambda i: _prefer_keep(recs[i]))
            keep = group[0]
            for i in group[1:]:
                flagged[i] = Finding(
                    recs[i].nodeid,
                    "duplicate",
                    [recs[keep].nodeid],
                    "identical coverage and identical body (modulo names)",
                    len(recs[i].elements),
                )
                an.kept_for.setdefault(recs[keep].nodeid, []).append(recs[i].nodeid)

        for group in by_shape.values():
            group = [i for i in group if i not in flagged]
            if len(group) < 2:
                continue
            group.sort(key=lambda i: _prefer_keep(recs[i]))
            keep = group[0]
            for i in group[1:]:
                if recs[i].base == recs[keep].base:
                    continue
                flagged[i] = Finding(
                    recs[i].nodeid,
                    "clone",
                    [recs[keep].nodeid],
                    "identical coverage and identical body shape; only literals differ "
                    "(merge into one @pytest.mark.parametrize)",
                    len(recs[i].elements),
                )
                an.kept_for.setdefault(recs[keep].nodeid, []).append(recs[i].nodeid)

    # --- 2. subsumed / redundant / essential ------------------------------
    # Re-index without the duplicate/clone members so a survivor of a cluster
    # is not then declared redundant because of the copies it just beat, and
    # without exempt tests, which must never be the reason another is flagged.
    live: Dict[int, Set[int]] = defaultdict(set)
    for i, t in enumerate(recs):
        if i not in flagged and not t.exempt:
            for e in t.elements:
                live[e].add(i)
    unique_count: List[int] = [0] * len(recs)
    for e, covering in live.items():
        if len(covering) == 1:
            unique_count[next(iter(covering))] += 1

    for i, t in enumerate(recs):
        if i in flagged or t.exempt:
            continue
        if unique_count[i]:
            continue  # essential
        # Find tests whose coverage is a superset of t's, via rarest element.
        rarest = min(t.elements, key=lambda e: len(live[e]))
        cands = set(live[rarest])
        cands.discard(i)
        for e in t.elements:
            if e == rarest:
                continue
            cands &= live[e]
            if not cands:
                break
        strict = [j for j in cands if len(recs[j].elements) > len(t.elements)]
        if strict:
            strict.sort(key=lambda j: (len(recs[j].elements), recs[j].nodeid))
            big = recs[strict[0]]
            flagged[i] = Finding(
                t.nodeid,
                "subsumed",
                [big.nodeid],
                f"every line/branch it executes is also executed by {big.nodeid} "
                f"({len(t.elements)} of {len(big.elements)} elements)",
                len(t.elements),
            )
        else:
            # Covered by the union of others but no single superset.
            others = set()
            for e in t.elements:
                others |= live[e]
            others.discard(i)
            flagged[i] = Finding(
                t.nodeid,
                "redundant",
                sorted(recs[j].nodeid for j in others)[:5],
                "contributes no coverage beyond the union of other tests",
                len(t.elements),
            )

    # --- 3. greedy minimal equivalent suite ----------------------------------
    remaining: Set[int] = set(index)
    chosen: Set[int] = set()
    # Exempt tests are always in the suite.
    for i, t in enumerate(recs):
        if t.exempt:
            chosen.add(i)
            remaining -= t.elements
    gain = {
        i: len(t.elements & remaining) for i, t in enumerate(recs) if i not in chosen
    }
    while remaining:
        best = max(
            gain, key=lambda i: (gain[i], -len(recs[i].elements), recs[i].nodeid)
        )
        if gain[best] == 0:
            break
        chosen.add(best)
        newly = recs[best].elements & remaining
        remaining -= newly
        del gain[best]
        touched: Set[int] = set()
        for e in newly:
            touched |= index[e]
        for j in touched:
            if j in gain:
                gain[j] = len(recs[j].elements & remaining)
    m.minimal_suite = len(chosen)

    # --- metrics -------------------------------------------------------------
    attr = {
        "duplicate": "duplicates",
        "clone": "clones",
        "subsumed": "subsumed",
        "redundant": "redundant",
    }
    for f in flagged.values():
        setattr(m, attr[f.kind], getattr(m, attr[f.kind]) + 1)
    m.essential = m.analyzed - len(flagged) - m.exempt
    m.redundancy_ratio = (
        round(1 - m.minimal_suite / m.analyzed, 4) if m.analyzed else 0.0
    )
    for i, f in flagged.items():
        f.unique_elements = unique_count[i]
    order = {"duplicate": 0, "clone": 1, "subsumed": 2, "redundant": 3}
    an.findings = sorted(flagged.values(), key=lambda f: (order[f.kind], f.nodeid))
    return an
