"""pytest plugin: fail the run when tests add no coverage, and say which ones to remove.

Records per-test coverage (own tracer, or pytest-cov with ``--cov-context=test``), solves
the exact minimal covering suite with ``cover`` and gates on every test outside it.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import coverage
import numpy as np
import pytest

from . import __version__, cover
from .fingerprint import clear_cache, fingerprint_for, parametrize_groups

TEST_FILE_OMIT = [
    "*/tests/*",
    "*/test/*",
    "*/test_*.py",
    "*_test.py",
    "*/conftest.py",
    "*/.venv/*",
    "*/venv/*",
    "*/site-packages/*",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "*"),
]
_SELF_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/") + "/"
NO_COVERAGE = (
    "covers no measured source: add the code it tests to --strain-source / --cov, "
    "or delete it"
)
EXEMPT_HINT = (
    "If a test asserts something coverage can't see, mark it "
    "@pytest.mark.strain_exempt(reason=...)."
)

BASELINE_KEYS = ("accepted", "slow")
VALUED = {
    "order": (
        str,
        "count,time",
        "ORDER",
        "Objectives in order: count,time or time,count.",
    ),
    "time_limit": (float, 60.0, "SECONDS", "HiGHS time limit per solver stage."),
    "baseline": (str, None, "FILE", "JSON ratchet of accepted redundant tests."),
    "report": (str, None, "FILE", "Write the full JSON report here."),
    "show": (int, 20, "N", "Action lines to print."),
    "show_slow": (int, 3, "N", "Slowest minimal-suite tests to print (advisory)."),
    "max_time": (
        float,
        None,
        "SECONDS",
        "Fail on tests slower than this; the cover avoids them where it can.",
    ),
    "parametrize": (
        str,
        "fail",
        "MODE",
        "Same-shape test functions: fail, warn or off.",
    ),
    "select_coverage": (
        float,
        1.0,
        "F",
        "With --strain-select, least-runtime subset reaching this coverage fraction.",
    ),
}
FLAGS = {
    "": "Record per-test coverage and fail when tests add no coverage.",
    "-lines": "Measure lines instead of branch arcs (own tracer only).",
    "-update-baseline": "Write the current drop list to the baseline; never fails.",
    "-warn-only": "Report but never change the exit status.",
    "-select": "Deselect tests outside the optimal suite of the last --strain recording.",
}


def pytest_addoption(parser: pytest.Parser) -> None:
    g = parser.getgroup("pyteastrainer", "minimal-suite gate")
    for name, text in FLAGS.items():
        g.addoption(f"--strain{name}", action="store_true", default=False, help=text)
    g.addoption(
        "--strain-source",
        action="append",
        default=None,
        metavar="PATH",
        help="Production source to measure (repeatable). Default: rootdir minus tests.",
    )
    for name, (kind, default, metavar, text) in VALUED.items():
        g.addoption(
            "--strain-" + name.replace("_", "-"),
            type=kind,
            default=None,
            metavar=metavar,
            help=text if default is None else f"{text} Default: {default}.",
        )
        parser.addini("strain_" + name, default="", help=text)
    parser.addini("strain_source", type="linelist", help="Production source paths.")
    parser.addini("strain_lines", type="bool", default=False, help=FLAGS["-lines"])


def _opt(config: pytest.Config, name: str):
    kind, default, _, _ = VALUED[name]
    cli = config.getoption("--strain-" + name.replace("_", "-"))
    if cli is not None:
        return cli
    raw = config.getini("strain_" + name)
    return kind(raw) if raw not in ("", None) else default


def _resolve(config: pytest.Config, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else config.rootpath / path


def _looks_like_test_file(path: str) -> bool:
    p = path.replace("\\", "/")
    name = p.rsplit("/", 1)[-1]
    return (
        "/tests/" in p
        or "/test/" in p
        or name.startswith("test_")
        or name.endswith("_test.py")
        or name == "conftest.py"
        or "/site-packages/" in p
        or p.startswith(_SELF_DIR)
    )


def _s(seconds: float) -> str:
    return f"{seconds:,.2f} s"


def _one(unit: str) -> str:
    return unit[:-2] if unit.endswith("ches") else unit[:-1]


def _n(count: int, unit: str) -> str:
    return f"{count:,} {_one(unit) if count == 1 else unit}"


def _matrix_path(config: pytest.Config) -> Optional[Path]:
    cache = getattr(config, "cache", None)
    return None if cache is None else cache.mkdir("pyteastrainer") / "matrix.npz"


class Gate:
    """Controller-side recorder, solver and gate."""

    def __init__(self, config: pytest.Config) -> None:
        self.config = config
        self.cov: Optional[coverage.Coverage] = None
        self.data_file: Optional[Path] = None
        self.source = "pyteastrainer"
        self.durations: Dict[str, float] = defaultdict(float)
        self.passed: set = set()
        self.bad: set = set()
        self.exempt: set = set()
        self.report: Optional[dict] = None
        self.lines: List[tuple] = []
        self.notes: List[str] = []
        self.red = False

    def _pytest_cov(self) -> Optional[coverage.Coverage]:
        plug = self.config.pluginmanager.getplugin("_cov")
        cov = getattr(getattr(plug, "cov_controller", None), "cov", None)
        if cov is None:
            return None
        if not (
            self.config.getoption("cov_context", None) == "test"
            or cov.config.dynamic_context
        ):
            raise pytest.UsageError(
                "pyteastrainer: pytest-cov is active but not recording per-test "
                "contexts. Add --cov-context=test (or drop --cov to let "
                "pyteastrainer trace on its own)."
            )
        if dict(cov.sys_info()).get("core") == "SysMonitor":
            raise pytest.UsageError(
                "pyteastrainer: pytest-cov is using coverage's sys.monitoring core, "
                "which cannot record per-test contexts. Set COVERAGE_CORE=ctrace "
                "(or drop --cov to let pyteastrainer trace on its own)."
            )
        return cov

    def pytest_sessionstart(self) -> None:
        clear_cache()
        if self._pytest_cov() is not None:
            self.source = "pytest-cov"
            return
        cache = _matrix_path(self.config)
        folder = cache.parent if cache else Path(tempfile.mkdtemp(prefix="strain-"))
        self.data_file = folder / "coverage.sqlite"
        self.data_file.unlink(missing_ok=True)
        lines = self.config.getoption("--strain-lines") or self.config.getini(
            "strain_lines"
        )
        self.cov = coverage.Coverage(
            data_file=str(self.data_file),
            branch=not lines,
            source=self.config.getoption("--strain-source")
            or self.config.getini("strain_source")
            or [str(self.config.rootpath)],
            omit=TEST_FILE_OMIT,
            config_file=False,
        )
        self.cov.set_option("run:core", "ctrace")
        self.cov.start()
        self.cov.switch_context("")

    @pytest.hookimpl(wrapper=True)
    def pytest_runtest_protocol(self, item: pytest.Item):
        if self.cov is not None:
            self.cov.switch_context(item.nodeid)
        try:
            return (yield)
        finally:
            if self.cov is not None:
                self.cov.switch_context("")

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        nid = report.nodeid
        self.durations[nid] += report.duration
        if "strain_exempt" in report.keywords:
            self.exempt.add(nid)
        if report.failed or report.skipped or hasattr(report, "wasxfail"):
            self.bad.add(nid)
        elif report.when == "call":
            self.passed.add(nid)

    def _data_path(self) -> Optional[Path]:
        if self.cov is not None:
            self.cov.stop()
            self.cov.save()
            return self.data_file
        cov = self._pytest_cov()
        return None if cov is None else Path(cov.config.data_file).resolve()

    def _baseline(self) -> Optional[Dict[str, set]]:
        name = _opt(self.config, "baseline")
        if not name:
            return None
        path = _resolve(self.config, name)
        if not path.exists():
            if not self.config.getoption("--strain-update-baseline"):
                self.notes.append(
                    f"baseline {path} does not exist; "
                    "create it with --strain-update-baseline"
                )
            return None
        doc = json.loads(path.read_text(encoding="utf-8"))
        return {k: set(doc.get(k, [])) for k in BASELINE_KEYS}

    def pytest_sessionfinish(self, session: pytest.Session) -> None:
        self._gate(session, self._data_path())

    def _gate(self, session: pytest.Session, path: Optional[Path]) -> None:
        cfg = self.config
        passed = self.passed - self.bad
        if path is None or not path.exists():
            self.notes.append("no coverage data was recorded")
            return
        m = cover.load(
            str(path),
            tests=passed,
            durations=self.durations,
            exempt=self.exempt,
            keep_file=lambda p: not _looks_like_test_file(p),
            root=str(cfg.rootpath) + os.sep,
        )
        baseline = self._baseline()
        base = baseline or {k: set() for k in BASELINE_KEYS}
        limit = _opt(cfg, "max_time")
        slow = m.cost > (np.inf if limit is None else limit)
        sol = cover.min_cover(
            m,
            _opt(cfg, "order").split(","),
            _opt(cfg, "time_limit"),
            (
                np.array([t in base["accepted"] for t in m.tests], bool)
                if baseline
                else None
            ),
            slow,
        )
        kept = sol.selected
        no_cov = sorted(passed - set(m.tests) - self.exempt)
        actions = self._actions(m, kept, slow, limit, passed) + [
            {
                "nodeid": t,
                "kind": "no_coverage",
                "action": f"{t} {NO_COVERAGE}",
                "because_of": [],
                "elements": 0,
            }
            for t in no_cov
        ]
        mode = _opt(cfg, "parametrize")
        merges = (
            []
            if mode == "off"
            else [
                {"members": members, "code": code}
                for members, code in parametrize_groups(
                    cfg.rootpath, [t for t, k in zip(m.tests, kept & ~m.exempt) if k]
                )
            ]
        )
        lists = {
            k: sorted(
                a["nodeid"] for a in actions if (a["kind"] == "slow") == (k == "slow")
            )
            for k in BASELINE_KEYS
        }
        fixed = sum(len(base[k] - set(lists[k])) for k in BASELINE_KEYS)
        if fixed:
            self.notes.append(
                f"{fixed} baseline entries are fixed; lock in with --strain-update-baseline"
            )
        update = cfg.getoption("--strain-update-baseline")
        if update:
            bp = _resolve(cfg, _opt(cfg, "baseline") or ".strain-baseline.json")
            bp.write_text(
                json.dumps({"version": __version__} | lists, indent=2) + "\n",
                encoding="utf-8",
            )
            self.notes.append(f"baseline written to {bp}")
            base = {k: set(v) for k, v in lists.items()}
        for a in actions:
            a["baseline"] = (
                a["nodeid"] in base["slow" if a["kind"] == "slow" else "accepted"]
            )
        violations = [a["nodeid"] for a in actions if not a["baseline"]]
        if mode == "fail" and not update:
            violations += ["parametrize: " + ", ".join(g["members"]) for g in merges]
        self.red = bool(violations)
        speed = self._speed(m, kept & ~slow)
        self.report = {
            "version": __version__,
            "mode": "branch" if m.branch else "line",
            "source": self.source,
            "metrics": {
                "tests": len(m.tests),
                "elements": m.elements,
                "merged_elements": int(m.a.shape[1]),
                "minimal_tests": int(kept.sum()),
                "minimal_runtime": float(m.cost[kept].sum()),
                "full_runtime": float(m.cost.sum()),
                "status": sol.status,
                "gap": sol.gap,
                "optimal": sol.optimal,
            },
            "gate": {"passed": not self.red, "violations": violations},
            "actions": actions,
            "parametrize": merges,
            "no_coverage": no_cov,
            "slow": speed,
            "kept": [t for t, k in zip(m.tests, kept) if k],
        }
        self._render(m, passed, sol, actions, speed, merges if mode != "off" else [])
        rp = _opt(cfg, "report")
        if rp:
            out = _resolve(cfg, rp)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(self.report, indent=2), encoding="utf-8")
            self.notes.append(f"report: {out}")
        npz = _matrix_path(cfg)
        if npz is not None and not cfg.getoption("--strain-select"):
            cover.save(npz, m)
        if (
            self.red
            and not cfg.getoption("--strain-warn-only")
            and session.exitstatus == pytest.ExitCode.OK
        ):
            session.exitstatus = pytest.ExitCode.TESTS_FAILED

    def _actions(self, m, kept, slow, limit, passed) -> List[dict]:
        unit, one, root = m.unit, _one(m.unit), self.config.rootpath
        rows = []
        for t, (kind, why) in cover.explain(m, kept).items():
            nid, names = m.tests[t], [m.tests[k] for k, _ in why]
            n = int(m.weight[m.a.indices[m.a.indptr[t] : m.a.indptr[t + 1]]].sum())
            own = fingerprint_for(root, nid) if "[" not in nid else None
            dup = next(
                (y for y in names if own and own == fingerprint_for(root, y)), None
            )
            if kind == "same" and dup:
                kind, names, tail = "duplicate", [dup], f" (duplicate of {dup})"
            elif kind == "same":
                names, tail = names[:1], f": same {unit} as {names[0]}"
            elif kind == "subset":
                tail = f": every {one} it reaches is reached by {names[0]}"
            else:
                tail = f": its {_n(n, unit)} are all reached by " + ", ".join(
                    f"{y} ({k:,})" for y, (_, k) in zip(names, why)
                )
            reason = " (exceeds max time)" if slow[t] else ""
            rows.append(
                {
                    "nodeid": nid,
                    "kind": kind,
                    "action": f"delete {nid}{reason}{tail}",
                    "because_of": names,
                    "elements": n,
                }
            )
        groups: Dict[str, List[dict]] = defaultdict(list)
        for r in rows:
            if "[" in r["nodeid"]:
                groups[r["nodeid"].split("[", 1)[0]].append(r)
        cases_of = Counter(p.split("[", 1)[0] for p in passed if "[" in p)
        for fn, rs in groups.items():
            names = ", ".join(dict.fromkeys(y for r in rs for y in r["because_of"]))
            they = "they reach" if len(rs) > 1 else "it reaches"
            if len(rs) == cases_of[fn]:
                head = f"delete {fn} (all {len(rs)} cases)"
            else:
                cases = ", ".join("[" + r["nodeid"].split("[", 1)[1] for r in rs)
                head = f"drop case{'s' if len(rs) > 1 else ''} {cases} from the parametrize of {fn}"
            for r in rs:
                r["kind"], r["action"] = (
                    "case",
                    f"{head}: every {one} {they} is reached by {names}",
                )
        fast = np.asarray(m.a[~slow].sum(axis=0)).ravel() > 0
        for t in np.flatnonzero(kept & slow):
            nid, cap = m.tests[t], _s(limit)
            cols = m.a.indices[m.a.indptr[t] : m.a.indptr[t + 1]]
            where = "; ".join(cover.ranges(m, cols[~fast[cols]]))
            fix = (
                f"only slow tests reach {where} — write a test under {cap} that reaches "
                f"these, then delete {nid}"
                if where
                else "make it faster"
            )
            rows.append(
                {
                    "nodeid": nid,
                    "kind": "slow",
                    "action": f"too slow: {nid} takes {_s(m.cost[t])} > max {cap}; {fix}",
                    "because_of": [],
                    "elements": int(m.weight[cols[~fast[cols]]].sum()),
                }
            )
        return sorted(rows, key=lambda r: (r["kind"] != "slow", r["nodeid"]))

    def _speed(self, m: cover.Matrix, kept: np.ndarray) -> List[dict]:
        count = np.asarray(m.a[kept].sum(axis=0)).ravel()
        unique = cover.essential(m, kept)
        cand = np.flatnonzero(kept & ~m.exempt & (unique > 0))
        top = cand[np.argsort(-m.cost[cand], kind="stable")][
            : _opt(self.config, "show_slow")
        ]
        total = float(m.cost[kept].sum()) or 1.0
        out = []
        for t in top:
            cols = m.a.indices[m.a.indptr[t] : m.a.indptr[t + 1]]
            out.append(
                {
                    "nodeid": m.tests[t],
                    "duration": float(m.cost[t]),
                    "share": float(m.cost[t]) / total,
                    "unique": int(unique[t]),
                    "ranges": cover.ranges(m, cols[count[cols] == 1]),
                }
            )
        return out

    def _render(self, m, passed, sol, actions, speed, merges) -> None:
        unit, kept, out = m.unit, sol.selected, self.lines
        km, ks, fs = int(kept.sum()), _s(m.cost[kept].sum()), _s(m.cost.sum())
        out.append(
            (
                f"{len(passed):,} passed tests cover {_n(m.elements, unit)}; {km:,} of them "
                f"({ks} of {fs}) cover all of it."
                + (" To make teastrainer green:" if self.red else ""),
                {},
            )
        )
        if not sol.optimal:
            out.append(
                (
                    f"solver: {sol.status}, gap {sol.gap:.2%}: not proven minimal; "
                    "raise --strain-time-limit",
                    {"yellow": True},
                )
            )
        todo = [a for a in actions if not a["baseline"]]
        lines = list(dict.fromkeys(a["action"] for a in todo))
        show = _opt(self.config, "show")
        out.extend((f"  {text}", {"red": True}) for text in lines[:show])
        if len(lines) > show:
            rp = _opt(self.config, "report")
            where = f"in {rp}" if rp else "(pass --strain-report FILE for all)"
            out.append((f"  ... {len(lines) - show} more {where}", {}))
        for g in merges:
            first, *rest = g["members"]
            names = ", ".join([first] + [r.rpartition("::")[2] for r in rest])
            out.append(
                (f"  parametrize: merge {names} into one test:", {"red": self.red})
            )
            out.extend((f"      {line}", {}) for line in g["code"].splitlines())
        accepted = len(actions) - len(todo)
        if any(a["kind"] != "slow" for a in todo):
            also = f" and the {accepted} baseline-accepted tests" if accepted else ""
            out.append(
                (
                    f"Removing these{also} leaves a suite with the same coverage, "
                    f"{_n(km, 'tests')}, {ks}, that teastrainer passes. {EXEMPT_HINT}",
                    {},
                )
            )
        if not self.red and accepted:
            out.append(
                (f"{accepted} flagged tests remain, all accepted in the baseline", {})
            )
        elif not self.red:
            free = len(m.tests) - int(m.exempt.sum())
            out.append(
                (
                    f"every one of the {free:,} tests covers something no other test does"
                    + (f" ({int(m.exempt.sum())} exempt)" if m.exempt.any() else ""),
                    {"green": True},
                )
            )
        if speed:
            out.append(
                (
                    "speed (advisory): slowest tests in the minimal suite; "
                    "a faster test reaching these makes it redundant",
                    {},
                )
            )
            out.extend(
                (
                    f"  {_s(s['duration'])} ({s['share']:.0%}) {s['nodeid']}: "
                    f"{_n(s['unique'], unit)} only it reaches: "
                    + "; ".join(s["ranges"]),
                    {},
                )
                for s in speed
            )

    def pytest_terminal_summary(self, terminalreporter) -> None:
        tr = terminalreporter
        if self.report is None:
            tr.write_sep("=", "pyteastrainer", yellow=True)
        else:
            warn = " (warn only)" if self.config.getoption("--strain-warn-only") else ""
            title = f"pyteastrainer: RED{warn}" if self.red else "pyteastrainer: GREEN"
            tr.write_sep("=", title, red=self.red, green=not self.red)
        for text, markup in self.lines:
            tr.write_line(text, **markup)
        for note in self.notes:
            tr.write_line(f"note: {note}")


class Selector:
    """Deselects tests outside the optimal suite of the last recorded matrix."""

    def __init__(self, config: pytest.Config) -> None:
        self.config = config
        self.selection: Optional[tuple] = None

    def _select(self) -> tuple:
        if self.selection is None:
            path = _matrix_path(self.config)
            if path is None or not path.exists():
                self.selection = (
                    set(),
                    "pyteastrainer select: no recording; run --strain first",
                )
                return self.selection
            m = cover.load_npz(path)
            frac = _opt(self.config, "select_coverage")
            limit = _opt(self.config, "time_limit")
            sol = (
                cover.min_cover(m, _opt(self.config, "order").split(","), limit)
                if frac >= 1
                else cover.partial_cover(m, frac, limit)
            )
            s = cover.summary(m, sol.selected)
            self.selection = (
                {t for t, k in zip(m.tests, sol.selected) if not k},
                f"pyteastrainer select: running {s['tests']:,} of {len(m.tests):,} recorded "
                f"tests, {_s(s['runtime'])} of {_s(m.cost.sum())}, "
                f"{s['coverage']:.2%} of {m.unit}",
            )
        return self.selection

    def pytest_report_header(self) -> str:
        return self._select()[1]

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, config: pytest.Config, items) -> None:
        drop = self._select()[0]
        gone = [i for i in items if i.nodeid in drop]
        if gone:
            items[:] = [i for i in items if i.nodeid not in drop]
            config.hook.pytest_deselected(items=gone)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "strain_exempt(reason): always keep this test in the minimal suite; it still "
        "counts toward coverage and can make other tests redundant.",
    )
    strain, select = config.getoption("--strain"), config.getoption("--strain-select")
    if not (strain or select):
        return
    if _opt(config, "order") not in {",".join(o) for o in cover.ORDERS}:
        raise pytest.UsageError("--strain-order must be count,time or time,count")
    if _opt(config, "parametrize") not in ("fail", "warn", "off"):
        raise pytest.UsageError("--strain-parametrize must be fail, warn or off")
    if not 0 < _opt(config, "select_coverage") <= 1:
        raise pytest.UsageError("--strain-select-coverage must be in (0, 1]")
    if select:
        config.pluginmanager.register(Selector(config), "pyteastrainer_select")
    if not strain or hasattr(config, "workerinput"):
        return
    distributed = getattr(config.option, "numprocesses", None) or (
        getattr(config.option, "dist", "no") != "no"
    )
    if distributed and not getattr(config.option, "cov_source", None):
        raise pytest.UsageError(
            "pyteastrainer: with -n, record coverage through pytest-cov: "
            "add --cov=SRC --cov-branch --cov-context=test"
        )
    config.pluginmanager.register(Gate(config), "pyteastrainer_gate")
