"""Static fingerprints of test functions.

Coverage alone can't tell "copy-pasted test" from "same code path, different
oracle", so every test also gets two structural fingerprints computed from its
AST:

* ``exact``  - alpha-renamed body with constants kept.  Two tests with the
               same ``exact`` fingerprint are the same test with the variable
               names changed (the classic LLM paraphrase).
* ``shape``  - alpha-renamed body with constants abstracted to their type.
               Same ``shape`` + same coverage means "same assertion against
               the same path with different literals": a parametrize
               candidate, which we call a *clone*.

Parametrized instances share a body, so the runtime parameter values are mixed
into the ``exact`` fingerprint by the plugin (see ``mix_params``).
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional


@dataclass(frozen=True)
class StaticFingerprint:
    exact: str
    shape: str
    n_asserts: int
    lineno: int


class _Normalizer(ast.NodeTransformer):
    """Alpha-rename identifiers and (optionally) abstract constants."""

    def __init__(self, abstract_constants: bool) -> None:
        self.abstract = abstract_constants
        self.names: Dict[str, str] = {}

    def _rename(self, name: str) -> str:
        if name.startswith("__") and name.endswith("__"):
            return name
        if name not in self.names:
            self.names[name] = f"v{len(self.names)}"
        return self.names[name]

    def visit_Name(self, node: ast.Name) -> ast.AST:
        return ast.copy_location(ast.Name(id=self._rename(node.id), ctx=node.ctx), node)

    def visit_arg(self, node: ast.arg) -> ast.AST:
        node.arg = self._rename(node.arg)
        node.annotation = None
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.name = self._rename(node.name)
        node.returns = None
        self.generic_visit(node)
        return node

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if self.abstract:
            return ast.copy_location(
                ast.Constant(value=type(node.value).__name__), node
            )
        return node

    def visit_Expr(self, node: ast.Expr) -> Optional[ast.AST]:
        # Drop bare string expressions (docstrings / commentary).
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        return self.generic_visit(node)


def _digest(s: str) -> str:
    return hashlib.blake2b(s.encode(), digest_size=10).hexdigest()


def _fingerprint_node(fn: ast.AST) -> StaticFingerprint:
    n_asserts = sum(isinstance(n, ast.Assert) for n in ast.walk(fn))
    n_asserts += sum(
        isinstance(n, ast.withitem)
        and isinstance(n.context_expr, ast.Call)
        and ast.unparse(n.context_expr.func).endswith("raises")
        for n in ast.walk(fn)
    )
    body_only = ast.Module(body=list(fn.body), type_ignores=[])  # type: ignore[attr-defined]
    body_only = _inline_temporaries(_clone(body_only))
    exact = ast.dump(_Normalizer(False).visit(_clone(body_only)), annotate_fields=False)
    shape = ast.dump(_Normalizer(True).visit(_clone(body_only)), annotate_fields=False)
    return StaticFingerprint(
        _digest(exact), _digest(shape), n_asserts, getattr(fn, "lineno", 0)
    )


def _clone(node: ast.AST) -> ast.AST:
    return ast.parse(ast.unparse(node))


def _inline_temporaries(mod: ast.Module) -> ast.Module:
    """Fold ``x = expr`` into its single use so that

        result = f(3); assert result == 1      and      assert f(3) == 1

    fingerprint the same.  Only names assigned once and loaded once, at the
    top level of the body, are folded; this is a normalisation for comparing
    tests, not a semantics-preserving transform.
    """
    while True:
        assigns: Dict[str, int] = {}
        loads: Dict[str, int] = {}
        for n in ast.walk(mod):
            if isinstance(n, ast.Name):
                if isinstance(n.ctx, ast.Load):
                    loads[n.id] = loads.get(n.id, 0) + 1
                else:
                    assigns[n.id] = assigns.get(n.id, 0) + 1
        target = None
        for i, stmt in enumerate(mod.body):
            if (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and assigns.get(stmt.targets[0].id) == 1
                and loads.get(stmt.targets[0].id) == 1
            ):
                target = (i, stmt.targets[0].id, stmt.value)
                break
        if target is None:
            return mod
        idx, name, value = target

        class _Sub(ast.NodeTransformer):
            def visit_Name(self, node: ast.Name) -> ast.AST:
                if node.id == name and isinstance(node.ctx, ast.Load):
                    return value
                return node

        del mod.body[idx]
        mod = _Sub().visit(mod)


@lru_cache(maxsize=None)
def _index_file(path: str) -> Dict[str, StaticFingerprint]:
    """Map ``Class::method`` / ``function`` names to fingerprints for a test file."""
    try:
        src = Path(path).read_text(encoding="utf-8")
        tree = ast.parse(src)
    except (OSError, SyntaxError):
        return {}
    out: Dict[str, StaticFingerprint] = {}

    def walk(nodes, prefix: str) -> None:
        for n in nodes:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out[prefix + n.name] = _fingerprint_node(n)
            elif isinstance(n, ast.ClassDef):
                walk(n.body, prefix + n.name + "::")

    walk(tree.body, "")
    return out


clear_cache = _index_file.cache_clear


def fingerprint_for(path: str, qualname: str) -> Optional[StaticFingerprint]:
    return _index_file(path).get(qualname)


def mix_params(fp: StaticFingerprint, params_repr: str) -> StaticFingerprint:
    """Fold runtime parameter values into the exact fingerprint."""
    if not params_repr:
        return fp
    return StaticFingerprint(
        _digest(fp.exact + "|" + params_repr), fp.shape, fp.n_asserts, fp.lineno
    )
