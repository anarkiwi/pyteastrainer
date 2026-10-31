"""Mutation mode: classify tests by the mutants they kill instead of the lines they run.

Source modules load through mutmut's trampolines; each mutant runs in a forked
child against only the tests that reached its function, and a test's elements
become the set of mutants it kills.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import json
import os
import select
import signal
import sys
import time
import types
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, NoReturn, Optional, Set, Tuple

import pytest
from _pytest.runner import runtestprotocol

from .fingerprint import clear_cache
from .plugin import StrainerPlugin, _looks_like_test_file

_SUFFIX = "__mutmut_"


def _mutmut():
    """Import mutmut lazily (optional extra) with the config its mutator reads."""
    # pylint: disable=import-outside-toplevel,protected-access
    import mutmut.configuration as mconf
    from mutmut.mutation import trampoline
    from mutmut.mutation.file_mutation import mutate_file_contents

    if mconf._config is None:
        mconf._config = types.SimpleNamespace(do_not_mutate_patterns=[])
    return trampoline, mutate_file_contents


class MutantRegistry(importlib.abc.MetaPathFinder):
    """Serve trampolined source for modules under ``roots``; remember their mutants."""

    def __init__(self, roots: List[str]) -> None:
        self.roots = [os.path.abspath(r).rstrip(os.sep) for r in roots]
        self.by_function: Dict[str, List[str]] = defaultdict(list)
        self.skipped: Dict[str, str] = {}

    def wants(self, origin: Optional[str]) -> bool:
        if not origin or not origin.endswith(".py"):
            return False
        p = os.path.abspath(origin)
        return not _looks_like_test_file(p) and any(
            p == r or p.startswith(r + os.sep) for r in self.roots
        )

    def find_spec(self, fullname, path, target=None):  # pylint: disable=unused-argument
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and self.wants(spec.origin):
            spec.loader = _MutantLoader(fullname, spec.origin, self)
        return spec

    def mutate(self, module: str, path: str, source: str) -> str:
        _, mutate_file_contents = _mutmut()
        try:
            mutated = mutate_file_contents(path, source)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self.skipped[path] = repr(exc)
            return source
        for name in mutated.mutant_names:
            self.by_function[f"{module}.{name.rsplit(_SUFFIX, 1)[0]}"].append(
                f"{module}.{name}"
            )
        return mutated.code


class _MutantLoader(importlib.machinery.SourceFileLoader):
    def __init__(self, fullname: str, path: str, registry: MutantRegistry) -> None:
        super().__init__(fullname, path)
        self.registry = registry

    def get_code(self, fullname):
        source = Path(self.path).read_text(encoding="utf-8")
        code = self.registry.mutate(fullname, self.path, source)
        return compile(code, self.path, "exec", dont_inherit=True)


def _set_mutant(trampoline, name: str) -> None:
    trampoline.set_mutant_under_test(name)
    os.environ["MUTANT_UNDER_TEST"] = name


class MutationStrainer(StrainerPlugin):
    """StrainerPlugin whose per-test elements are killed mutants."""

    WORDS = {
        "equiv": "mutation-equivalent",
        "same": "kills",
        "unique": "kills a mutant nothing else does",
        "none": "killed no mutant",
    }
    mode = "mutant"

    def __init__(self, config: pytest.Config, registry: MutantRegistry) -> None:
        super().__init__(config)
        self.registry = registry
        self.hits: Dict[str, Set[str]] = defaultdict(set)
        self.kills: Dict[str, Set[int]] = defaultdict(set)
        self.mutant_ids: Dict[str, int] = {}
        self.weak: List[str] = []
        self.timeouts = 0
        self.jobs = config.getoption("--strain-jobs") or os.cpu_count() or 1
        self.timeout = config.getoption("--strain-timeout")
        self._current: Optional[Set[str]] = None
        self.trampoline, _ = _mutmut()

    def pytest_sessionstart(self) -> None:
        clear_cache()
        self.trampoline.record_trampoline_hit = self._record
        _set_mutant(self.trampoline, "stats")

    def _record(self, name: str, **_) -> None:
        if self._current is not None:
            self._current.add(name)

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_protocol(self, item: pytest.Item):
        self._current = self.hits[item.nodeid]
        yield from super().pytest_runtest_protocol(item)
        self._current = None

    @pytest.hookimpl(wrapper=True)
    def pytest_runtestloop(self, session: pytest.Session):
        result = yield
        _set_mutant(self.trampoline, "")
        if not session.shouldfail and not session.shouldstop:
            self._kill_matrix(session)
        return result

    def _jobs(self, session: pytest.Session) -> List[Tuple[str, List[pytest.Item]]]:
        items = {i.nodeid: i for i in session.items if i.nodeid in self.passed}
        tests_by_fn: Dict[str, List[str]] = defaultdict(list)
        for nid in items:
            for fn in self.hits.get(nid, ()):
                tests_by_fn[fn].append(nid)
        jobs = []
        for fn, nids in tests_by_fn.items():
            nids.sort(key=lambda n: self.durations.get(n, 0.0))
            for m in self.registry.by_function.get(fn, ()):
                self.mutant_ids.setdefault(m, len(self.mutant_ids))
                jobs.append((m, [items[n] for n in nids]))
        return jobs

    def _kill_matrix(self, session: pytest.Session) -> None:
        from tqdm import tqdm  # pylint: disable=import-outside-toplevel

        queue = self._jobs(session)
        running: Dict[int, list] = {}
        progress = tqdm(total=len(queue), desc="mutants", unit="mut", file=sys.stderr)
        while queue or running:
            while queue and len(running) < self.jobs:
                mutant, items = queue.pop()
                budget = self.timeout + 10.0 * sum(
                    self.durations.get(i.nodeid, 0.0) for i in items
                )
                pid, fd = self._fork(mutant, items)
                running[fd] = [pid, mutant, items, 0, time.monotonic() + budget, b""]
            ready, _, _ = select.select(list(running), [], [], 0.5)
            for fd in ready:
                job = running[fd]
                chunk = os.read(fd, 65536)
                if chunk:
                    *lines, job[5] = (job[5] + chunk).split(b"\n")
                    for line in lines:
                        nid, killed = line.decode().rsplit(" ", 1)
                        if killed == "1":
                            self.kills[nid].add(self.mutant_ids[job[1]])
                        job[3] += 1
                    continue
                os.close(fd)
                os.waitpid(job[0], 0)
                del running[fd]
                progress.update()
            now = time.monotonic()
            for fd, (pid, mutant, items, done, deadline, _) in list(running.items()):
                if now < deadline:
                    continue
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
                os.close(fd)
                del running[fd]
                self.timeouts += 1
                self.kills[items[done].nodeid].add(self.mutant_ids[mutant])
                if done + 1 < len(items):
                    queue.append((mutant, items[done + 1 :]))
                else:
                    progress.update()
        progress.close()
        self.weak = sorted(
            nid
            for nid in self.passed
            if not self.kills.get(nid)
            and any(fn in self.registry.by_function for fn in self.hits.get(nid, ()))
        )

    def _fork(self, mutant: str, items: List[pytest.Item]) -> Tuple[int, int]:
        r, w = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(r)
            self._child(mutant, items, w)
        os.close(w)
        return pid, r

    def _child(self, mutant: str, items: List[pytest.Item], w: int) -> NoReturn:
        code = 0
        try:
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, 1)
            os.dup2(devnull, 2)
            _set_mutant(self.trampoline, mutant)
            for item in items:
                reports = runtestprotocol(item, nextitem=None, log=False)
                os.write(
                    w,
                    f"{item.nodeid} {int(any(rep.failed for rep in reports))}\n".encode(),
                )
        except BaseException:  # pylint: disable=broad-exception-caught
            code = 1
        finally:
            os._exit(code)  # pylint: disable=protected-access

    def _per_test_elements(self) -> Dict[str, set]:
        return self.kills

    def _summary(self) -> dict:
        return {
            "reached": len(self.mutant_ids),
            "killed": len(set().union(*self.kills.values())) if self.kills else 0,
            "timeouts": self.timeouts,
            "weak": self.weak,
            "skipped_files": self.registry.skipped,
        }

    def _write_outputs(self) -> None:
        super()._write_outputs()
        if self.report_path:
            doc = json.loads(self.report_path.read_text(encoding="utf-8"))
            doc["mutants"] = self._summary()
            self.report_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    def pytest_terminal_summary(self, terminalreporter) -> None:
        super().pytest_terminal_summary(terminalreporter)
        s = self._summary()
        terminalreporter.write_line(
            f"  mutants   {s['reached']} reached, {s['killed']} killed, {s['timeouts']} timeouts; "
            f"{len(self.weak)} weak tests kill none of the mutants they reach"
        )
        show = self.config.getoption("--strain-show")
        for nid in self.weak[: 20 if show is None else show]:
            terminalreporter.write_line(f"  weak      {nid}")


def pytest_addoption(parser: pytest.Parser) -> None:
    g = parser.getgroup("pyteastrainer")
    g.addoption(
        "--strain-mutants",
        action="store_true",
        default=False,
        help="Classify tests by the mutants they kill (needs the 'mutants' extra).",
    )
    g.addoption(
        "--strain-jobs",
        type=int,
        default=None,
        metavar="N",
        help="Concurrent mutant processes in --strain-mutants mode. Default: CPU count.",
    )
    g.addoption(
        "--strain-timeout",
        type=float,
        default=10.0,
        metavar="SECONDS",
        help="Base time per mutant on top of ten times its tests' own duration.",
    )


_REGISTRY: Optional[MutantRegistry] = None


@pytest.hookimpl(tryfirst=True)
def pytest_load_initial_conftests(early_config):
    global _REGISTRY  # pylint: disable=global-statement
    ns = early_config.known_args_namespace
    if not getattr(ns, "strain_mutants", False):
        return
    roots = getattr(ns, "strain_source", None) or [str(early_config.rootpath)]
    _REGISTRY = MutantRegistry(roots)
    sys.meta_path.insert(0, _REGISTRY)


def pytest_configure(config: pytest.Config) -> None:
    if _REGISTRY is None or not config.getoption("--strain-mutants"):
        return
    if config.getoption("--strain"):
        raise pytest.UsageError(
            "--strain and --strain-mutants are separate modes; pick one."
        )
    if config.getoption("dist", "no") != "no":
        raise pytest.UsageError(
            "--strain-mutants forks its own workers; run without -n."
        )
    config.pluginmanager.register(
        MutationStrainer(config, _REGISTRY), "pyteastrainer_mutation_gate"
    )


def pytest_unconfigure() -> None:
    global _REGISTRY  # pylint: disable=global-statement
    if _REGISTRY is not None and _REGISTRY in sys.meta_path:
        sys.meta_path.remove(_REGISTRY)
    _REGISTRY = None
