"""pyteastrainer: fail the run when the suite has grown redundant.

Collects per-test coverage (its own coverage.py tracer, or pytest-cov's when
that plugin is active with ``--cov-context=test``), fingerprints every test's
body, classifies redundancy, and sets a non-zero exit status when the gate
fails.  Designed to sit in CI so that an agent cannot land a test that adds
nothing.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional

import coverage
import pytest
from coverage.numbits import numbits_to_nums

from .analysis import Analysis, TestRecord, analyze
from .fingerprint import clear_cache, fingerprint_for, mix_params

REPORT_VERSION = 1

TEST_FILE_OMIT = [
    "*/tests/*",
    "*/test/*",
    "*/test_*.py",
    "*_test.py",
    "*/conftest.py",
    "*/.venv/*",
    "*/venv/*",
    "*/site-packages/*",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "*"),  # this plugin
]
_SELF_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/") + "/"


# --------------------------------------------------------------------------- #
# Options
# --------------------------------------------------------------------------- #
def pytest_addoption(parser: pytest.Parser) -> None:
    g = parser.getgroup("pyteastrainer", "redundant-test gate")
    g.addoption(
        "--strain",
        action="store_true",
        default=False,
        help="Collect per-test coverage and fail on redundant tests.",
    )
    g.addoption(
        "--strain-source",
        action="append",
        default=None,
        metavar="PATH",
        help="Production source to measure (repeatable). Default: rootdir "
        "minus test files.",
    )
    g.addoption(
        "--strain-branch",
        action="store_true",
        default=False,
        help="Fingerprint on branch arcs instead of lines (slower, stricter).",
    )
    g.addoption(
        "--strain-max-duplicates",
        type=int,
        default=None,
        metavar="N",
        help="Fail if more than N duplicate tests. Default 0.",
    )
    g.addoption(
        "--strain-max-clones",
        type=int,
        default=None,
        metavar="N",
        help="Fail if more than N clone tests (same path, different literals).",
    )
    g.addoption(
        "--strain-max-subsumed",
        type=int,
        default=None,
        metavar="N",
        help="Fail if more than N tests are strict subsets of another test.",
    )
    g.addoption(
        "--strain-max-redundancy",
        type=float,
        default=None,
        metavar="RATIO",
        help="Fail if 1 - minimal_suite/analyzed exceeds RATIO (0-1).",
    )
    g.addoption(
        "--strain-baseline",
        default=None,
        metavar="FILE",
        help="Ratchet file: fail if any redundancy metric is worse than it.",
    )
    g.addoption(
        "--strain-update-baseline",
        action="store_true",
        default=False,
        help="Write current metrics to the baseline file (never fails).",
    )
    g.addoption(
        "--strain-tolerance",
        type=float,
        default=None,
        metavar="RATIO",
        help="Slack allowed over the baseline redundancy ratio. Default 0.",
    )
    g.addoption(
        "--strain-report",
        default=None,
        metavar="FILE",
        help="Write the full machine-readable report (JSON) here.",
    )
    g.addoption(
        "--strain-show",
        type=int,
        default=None,
        metavar="N",
        help="How many findings to print in the terminal. Default 20.",
    )
    g.addoption(
        "--strain-warn-only",
        action="store_true",
        default=False,
        help="Report but never change the exit status.",
    )

    parser.addini("strain_source", type="linelist", help="Production source paths.")
    parser.addini("strain_branch", type="bool", default=False, help="Use branch arcs.")
    parser.addini("strain_max_duplicates", default="", help="Max duplicate tests.")
    parser.addini("strain_max_clones", default="", help="Max clone tests.")
    parser.addini("strain_max_subsumed", default="", help="Max subsumed tests.")
    parser.addini("strain_max_redundancy", default="", help="Max redundancy ratio.")
    parser.addini("strain_baseline", default="", help="Baseline (ratchet) file.")
    parser.addini("strain_tolerance", default="0", help="Ratio slack over baseline.")
    parser.addini("strain_report", default="", help="JSON report path.")
    parser.addini("strain_show", default="20", help="Findings to print.")


def _opt(config: pytest.Config, name: str, cast=str):
    cli = config.getoption("--strain-" + name.replace("_", "-"), default=None)
    if cli is not None:
        return cli
    raw = config.getini("strain_" + name)
    if raw in ("", None, []):
        return None
    return cast(raw)


# --------------------------------------------------------------------------- #
# Plugin
# --------------------------------------------------------------------------- #
class StrainerPlugin:
    def __init__(self, config: pytest.Config) -> None:
        self.config = config
        self.cov = None  # our own coverage.Coverage
        self.cov_from_pytest_cov = False
        self.data_file: Optional[Path] = None
        self.branch = bool(
            config.getoption("--strain-branch") or config.getini("strain_branch")
        )
        self.params: Dict[str, str] = {}  # nodeid -> repr(callspec.params)
        self.exempt: set = set()
        self.passed: set = set()
        self.durations: Dict[str, float] = {}
        self.analysis: Optional[Analysis] = None
        self.violations: List[str] = []
        self.notes: List[str] = []
        self.report_path: Optional[Path] = None
        self.baseline_written = False

    # ---- coverage lifecycle ------------------------------------------------
    def _pytest_cov_data_file(self) -> Optional[Path]:
        plug = self.config.pluginmanager.getplugin("_cov")
        ctrl = getattr(plug, "cov_controller", None)
        cov = getattr(ctrl, "cov", None)
        if cov is None:
            return None
        per_test = self.config.getoption("cov_context", None) == "test" or getattr(
            cov.config, "dynamic_context", None
        )
        if not per_test:
            raise pytest.UsageError(
                "pyteastrainer: pytest-cov is active but not recording per-test "
                "contexts. Add --cov-context=test (or drop --cov to let "
                "pyteastrainer trace on its own)."
            )
        self.cov_from_pytest_cov = True
        self.branch = bool(cov.config.branch)
        return Path(cov.get_data().data_filename())

    def pytest_sessionstart(self) -> None:
        clear_cache()
        if self._pytest_cov_data_file() is not None:
            return
        cache_dir = self.config.cache.mkdir("pyteastrainer")
        self.data_file = cache_dir / "coverage.sqlite"
        if self.data_file.exists():
            self.data_file.unlink()
        source = _opt(self.config, "source", list) or [str(self.config.rootpath)]
        self.cov = coverage.Coverage(
            data_file=str(self.data_file),
            branch=self.branch,
            source=source,
            omit=TEST_FILE_OMIT,
            config_file=False,
        )
        self.cov.set_option("run:relative_files", False)
        self.cov.start()
        self.cov.switch_context("")

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_protocol(self, item: pytest.Item):
        callspec = getattr(item, "callspec", None)
        if callspec is not None:
            self.params[item.nodeid] = repr(sorted(callspec.params.items()))
        if item.get_closest_marker("strain_exempt"):
            self.exempt.add(item.nodeid)
        if self.cov is not None:
            self.cov.switch_context(item.nodeid)
        t0 = time.perf_counter()
        yield
        self.durations[item.nodeid] = time.perf_counter() - t0
        if self.cov is not None:
            self.cov.switch_context("")

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if report.when == "call" and report.passed and not hasattr(report, "wasxfail"):
            self.passed.add(report.nodeid)
        elif report.failed or report.skipped:
            self.passed.discard(report.nodeid)

    # ---- read coverage data ------------------------------------------------
    def _per_test_elements(self) -> Dict[str, set]:
        if self.cov is not None:
            self.cov.stop()
            self.cov.save()
            path = self.data_file
        else:
            path = self._pytest_cov_data_file()
        if path is None or not path.exists():
            return {}

        con = sqlite3.connect(str(path))
        try:
            files = dict(con.execute("SELECT id, path FROM file"))
            contexts = dict(con.execute("SELECT id, context FROM context"))
            keep_file = {
                fid for fid, p in files.items() if not _looks_like_test_file(p)
            }
            per: Dict[str, set] = {}

            def ctx_to_nodeid(c: str) -> Optional[str]:
                c = c.split("|", 1)[0]  # pytest-cov appends |setup/|run/|teardown
                return c or None

            if self.branch:
                rows = con.execute("SELECT file_id, context_id, fromno, tono FROM arc")
                for fid, cid, a, b in rows:
                    if fid not in keep_file:
                        continue
                    nid = ctx_to_nodeid(contexts[cid])
                    if nid is None:
                        continue
                    per.setdefault(nid, set()).add(
                        (fid << 42) | ((a + (1 << 20)) << 21) | (b + (1 << 20))
                    )
            else:
                rows = con.execute("SELECT file_id, context_id, numbits FROM line_bits")
                for fid, cid, nb in rows:
                    if fid not in keep_file:
                        continue
                    nid = ctx_to_nodeid(contexts[cid])
                    if nid is None:
                        continue
                    s = per.setdefault(nid, set())
                    base = fid << 24
                    s.update(base | ln for ln in numbits_to_nums(nb))
            return per
        finally:
            con.close()

    # ---- analysis + gate ----------------------------------------------------
    def pytest_sessionfinish(self, session: pytest.Session) -> None:
        try:
            per = self._per_test_elements()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self.notes.append(f"could not read coverage data: {exc!r}")
            return
        self._finish(session, per)

    def _finish(self, session: pytest.Session, per: Dict[str, set]) -> None:
        records: List[TestRecord] = []
        for item in session.items:
            nid = item.nodeid
            if nid not in self.passed:
                continue
            path = str(getattr(item, "path", None) or getattr(item, "fspath", ""))
            qual = "::".join(p for p in nid.split("::")[1:]).split("[", 1)[0]
            static = fingerprint_for(path, qual)
            if static is not None:
                static = mix_params(static, self.params.get(nid, ""))
            records.append(
                TestRecord(
                    nodeid=nid,
                    elements=frozenset(per.get(nid, ())),
                    static=static,
                    exempt=nid in self.exempt,
                    duration=self.durations.get(nid, 0.0),
                )
            )
        if not records:
            self.notes.append("no passed tests with coverage to analyze")
            return

        self.analysis = analyze(records)
        self._gate()
        self._write_outputs()

        if self.violations and not self.config.getoption("--strain-warn-only"):
            if session.exitstatus == 0:
                session.exitstatus = pytest.ExitCode.TESTS_FAILED

    def _gate(self) -> None:
        m = self.analysis.metrics
        c = self.config
        if c.getoption("--strain-update-baseline"):
            return  # baseline runs never fail

        baseline_path = _opt(c, "baseline")
        baseline_exists = bool(baseline_path) and _resolve(c, baseline_path).exists()

        max_dup = _opt(c, "max_duplicates", int)
        if max_dup is None and not baseline_exists:
            max_dup = 0  # hard default; a baseline takes over on legacy suites
        if max_dup is not None and m.duplicates > max_dup:
            self.violations.append(f"duplicates {m.duplicates} > max {max_dup}")
        max_clones = _opt(c, "max_clones", int)
        if max_clones is not None and m.clones > max_clones:
            self.violations.append(f"clones {m.clones} > max {max_clones}")
        max_sub = _opt(c, "max_subsumed", int)
        if max_sub is not None and m.subsumed > max_sub:
            self.violations.append(f"subsumed {m.subsumed} > max {max_sub}")
        max_red = _opt(c, "max_redundancy", float)
        if max_red is not None and m.redundancy_ratio > max_red:
            self.violations.append(
                f"redundancy {m.redundancy_ratio:.1%} > max {max_red:.1%}"
            )

        if baseline_path:
            p = _resolve(c, baseline_path)
            if not p.exists():
                self.notes.append(
                    f"baseline {p} does not exist; run once with --strain-update-baseline"
                )
            else:
                doc = json.loads(p.read_text(encoding="utf-8"))
                base = doc.get("metrics", {})
                tol = _opt(c, "tolerance", float) or 0.0
                improved = False
                for key in ("duplicates", "clones", "subsumed", "redundant"):
                    if key in base and getattr(m, key) > base[key]:
                        self.violations.append(
                            f"{key} {getattr(m, key)} > baseline {base[key]}"
                        )
                    elif key in base and getattr(m, key) < base[key]:
                        improved = True
                if (
                    "redundancy_ratio" in base
                    and m.redundancy_ratio > base["redundancy_ratio"] + tol
                ):
                    self.violations.append(
                        f"redundancy {m.redundancy_ratio:.1%} > baseline "
                        f"{base['redundancy_ratio']:.1%}"
                        + (f" (+{tol:.1%} tolerance)" if tol else "")
                    )
                known = set(doc.get("flagged", []))
                new = [
                    f.nodeid for f in self.analysis.findings if f.nodeid not in known
                ]
                if new and self.violations:
                    self.notes.append(
                        "new since baseline: "
                        + ", ".join(_short(n, 70) for n in new[:10])
                    )
                if improved and not self.violations:
                    self.notes.append(
                        "suite got leaner than the baseline; "
                        "lock it in with --strain-update-baseline"
                    )

    def _write_outputs(self) -> None:
        c = self.config
        an = self.analysis
        payload = {
            "version": REPORT_VERSION,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "mode": "branch" if self.branch else "line",
            "source": "pytest-cov" if self.cov_from_pytest_cov else "pyteastrainer",
            "metrics": an.metrics.as_dict(),
            "gate": {
                "passed": not self.violations,
                "violations": list(self.violations),
            },
            "findings": [f.__dict__ for f in an.findings],
            "kept_for": an.kept_for,
        }
        rp = _opt(c, "report")
        if rp:
            self.report_path = Path(rp) if os.path.isabs(rp) else c.rootpath / rp
            self.report_path.parent.mkdir(parents=True, exist_ok=True)
            self.report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        if c.getoption("--strain-update-baseline"):
            bp = _resolve(c, _opt(c, "baseline") or ".strain-baseline.json")
            bp.write_text(
                json.dumps(
                    {
                        "version": REPORT_VERSION,
                        "metrics": an.metrics.as_dict(),
                        "flagged": sorted(f.nodeid for f in an.findings),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            self.notes.append(f"baseline written to {bp}")
            self.baseline_written = True

    # ---- terminal ----------------------------------------------------------
    def pytest_terminal_summary(self, terminalreporter) -> None:
        tr = terminalreporter
        an = self.analysis
        if an is None:
            if self.notes:
                tr.write_sep("=", "pyteastrainer", yellow=True)
                for n in self.notes:
                    tr.write_line(f"  {n}")
            return
        m = an.metrics
        failed = bool(self.violations) and not self.config.getoption(
            "--strain-warn-only"
        )
        tr.write_sep("=", "pyteastrainer", red=failed, green=not self.violations)
        mode = "branches" if self.branch else "lines"
        tr.write_line(
            f"  analyzed {m.analyzed} passed tests over {m.elements} {mode}; "
            f"a coverage-equivalent suite needs {m.minimal_suite} "
            f"(redundancy {m.redundancy_ratio:.1%})"
        )
        rows = [
            ("duplicate", m.duplicates, "identical coverage and body", "delete"),
            (
                "clone",
                m.clones,
                "same path & assertions, other literals",
                "parametrize",
            ),
            (
                "subsumed",
                m.subsumed,
                "strict subset of one other test",
                "review/delete",
            ),
            ("redundant", m.redundant, "nothing beyond the union of others", "review"),
            ("essential", m.essential, "covers something nothing else does", "keep"),
            ("exempt", m.exempt, "@pytest.mark.strain_exempt", "keep"),
        ]
        for name, n, what, action in rows:
            tr.write_line(f"  {name:<10}{n:>6}   {what:<40} -> {action}")

        show = _opt(self.config, "show", int)
        show = 20 if show is None else show
        if an.findings and show:
            tr.write_line("")
            for f in an.findings[:show]:
                sib = f.because_of[0] if f.because_of else "-"
                tr.write_line(f"  {f.kind:<10}{_short(f.nodeid)}")
                rel = "==" if f.kind in ("duplicate", "clone") else "<="
                tr.write_line(f"            {rel} {_short(sib)}")
            if len(an.findings) > show:
                tr.write_line(
                    f"  ... {len(an.findings) - show} more (see --strain-report)"
                )

        for n in self.notes:
            tr.write_line(f"  note: {n}")
        if self.report_path:
            tr.write_line(f"  report: {self.report_path}")
        if self.violations:
            tr.write_line("")
            verdict = "WARN" if self.config.getoption("--strain-warn-only") else "FAIL"
            for v in self.violations:
                tr.write_line(
                    f"  {verdict}: {v}",
                    red=not self.config.getoption("--strain-warn-only"),
                    yellow=self.config.getoption("--strain-warn-only"),
                )
            if failed:
                tr.write_line(
                    "  Delete or merge the flagged tests, or mark a test "
                    "@pytest.mark.strain_exempt(reason=...) if it asserts something "
                    "coverage can't see.",
                    red=True,
                )


def _resolve(config: pytest.Config, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else config.rootpath / path


def _short(nodeid: str, width: int = 96) -> str:
    return nodeid if len(nodeid) <= width else "..." + nodeid[-(width - 3) :]


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


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "strain_exempt(reason): never flag this test as redundant, and never use it "
        "to flag a copy; it still counts toward the coverage universe.",
    )
    if not config.getoption("--strain"):
        return
    if (
        config.pluginmanager.hasplugin("xdist")
        and config.getoption("dist", "no") != "no"
    ):
        raise pytest.UsageError(
            "pyteastrainer does not support xdist workers yet; run without -n."
        )
    config.pluginmanager.register(StrainerPlugin(config), "pyteastrainer_gate")
