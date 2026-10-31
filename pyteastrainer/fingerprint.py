"""Static fingerprints of test bodies and parametrize suggestions for same-shape tests.

Bodies are normalised (docstrings dropped, single-use temporaries inlined, locally bound
names alpha-renamed); ``exact`` keeps constants, ``shape`` abstracts them to their type.
"""

from __future__ import annotations

import ast
import copy
import hashlib
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

FUNCS = (ast.FunctionDef, ast.AsyncFunctionDef)


class _Normalizer(ast.NodeTransformer):
    """Alpha-rename locally bound names and optionally abstract constants to their type."""

    def __init__(self, bound: set, abstract: bool) -> None:
        self.bound, self.abstract = bound, abstract
        self.names: Dict[str, str] = {}

    def _rename(self, name: str) -> str:
        if name not in self.bound:
            return name
        return self.names.setdefault(name, f"v{len(self.names)}")

    def visit_Name(self, node: ast.Name) -> ast.AST:
        return ast.copy_location(ast.Name(id=self._rename(node.id), ctx=node.ctx), node)

    def visit_arg(self, node: ast.arg) -> ast.AST:
        node.arg = self._rename(node.arg)
        node.annotation = None
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if self.abstract:
            return ast.copy_location(ast.Constant(type(node.value).__name__), node)
        return node


_clone = copy.deepcopy


class _FoldSign(ast.NodeTransformer):
    """Fold ``-3`` into one constant so signed literals align with unsigned ones."""

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        self.generic_visit(node)
        if (
            isinstance(node.op, ast.USub)
            and isinstance(node.operand, ast.Constant)
            and type(node.operand.value) in (int, float, complex)
        ):
            return ast.copy_location(ast.Constant(-node.operand.value), node)
        return node


def _body(fn: ast.AST) -> ast.Module:
    """The function body without bare string expressions (docstrings, commentary)."""
    keep = [
        s
        for s in fn.body
        if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))
    ]
    return _FoldSign().visit(ast.Module(body=_clone(keep), type_ignores=[]))


def _dump(mod: ast.Module, abstract: bool) -> str:
    mod = _clone(mod)
    bound = {
        n.id
        for n in ast.walk(mod)
        if isinstance(n, ast.Name) and not isinstance(n.ctx, ast.Load)
    } | {n.arg for n in ast.walk(mod) if isinstance(n, ast.arg)}
    return ast.dump(_Normalizer(bound, abstract).visit(mod), annotate_fields=False)


def _digest(s: str) -> str:
    return hashlib.blake2b(s.encode(), digest_size=10).hexdigest()


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
def _index_file(path: str) -> Dict[str, ast.AST]:
    """Map ``Class::method`` / ``function`` names to their AST nodes for a test file."""
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return {}
    out: Dict[str, ast.AST] = {}

    def walk(nodes, prefix: str) -> None:
        for n in nodes:
            if isinstance(n, FUNCS):
                out[prefix + n.name] = n
            elif isinstance(n, ast.ClassDef):
                walk(n.body, prefix + n.name + "::")

    walk(tree.body, "")
    return out


clear_cache = _index_file.cache_clear


def _function(root: Path, nodeid: str) -> Optional[ast.AST]:
    path, _, qual = nodeid.partition("::")
    return _index_file(str(root / path)).get(qual.split("[", 1)[0])


def fingerprint(fn: ast.AST, abstract: bool = False) -> str:
    return _digest(_dump(_inline_temporaries(_body(fn)), abstract))


def fingerprint_for(root: Path, nodeid: str) -> Optional[str]:
    fn = _function(root, nodeid)
    return None if fn is None else fingerprint(fn)


def _param_name(node: ast.AST, parents: dict, taken: set) -> str:
    parent = parents.get(id(node))
    name = parent.arg if isinstance(parent, ast.keyword) else None
    if not name or name in taken:
        name = next(f"p{i}" for i in range(len(taken) + 1) if f"p{i}" not in taken)
    taken.add(name)
    return name


def suggest(fns: List[ast.AST]) -> Optional[str]:
    """One parametrized test replacing same-shape ``fns``: differing constants become params."""
    bodies = [_body(fn) for fn in fns]
    if len({_dump(b, True) for b in bodies}) != 1:
        bodies = [_inline_temporaries(b) for b in bodies]
    consts = [[n for n in ast.walk(b) if isinstance(n, ast.Constant)] for b in bodies]
    columns = sorted(
        (c for c in zip(*consts) if len({repr(n.value) for n in c}) > 1),
        key=lambda c: (c[0].lineno, c[0].col_offset),
    )
    if not columns:
        return None
    parents = {id(c): p for p in ast.walk(bodies[0]) for c in ast.iter_child_nodes(p)}
    taken = {n.id for n in ast.walk(bodies[0]) if isinstance(n, ast.Name)} | {
        a.arg for a in fns[0].args.args
    }
    params: Dict[tuple, str] = {}
    values: Dict[str, tuple] = {}
    replace: Dict[int, str] = {}
    for col in columns:
        key = tuple(repr(n.value) for n in col)
        if key not in params:
            params[key] = _param_name(col[0], parents, taken)
            values[params[key]] = col
        replace[id(col[0])] = params[key]

    class _Sub(ast.NodeTransformer):
        def visit_Constant(self, node: ast.Constant) -> ast.AST:
            name = replace.get(id(node))
            return ast.Name(id=name, ctx=ast.Load()) if name else node

    body = _Sub().visit(bodies[0])
    rows = [
        [ast.Constant(c[j].value) for c in values.values()] for j in range(len(fns))
    ]
    fn = _clone(fns[0])
    fn.body = body.body
    fn.args.args.extend(ast.arg(arg=name) for name in params.values())
    fn.decorator_list.append(
        ast.Call(
            func=ast.parse("pytest.mark.parametrize", mode="eval").body,
            args=[
                ast.Constant(", ".join(params.values())),
                ast.List(
                    [
                        r[0] if len(params) == 1 else ast.Tuple(r, ast.Load())
                        for r in rows
                    ],
                    ast.Load(),
                ),
            ],
            keywords=[
                ast.keyword(
                    "ids", ast.List([ast.Constant(f.name) for f in fns], ast.Load())
                )
            ],
        )
    )
    return ast.unparse(fn)


def parametrize_groups(
    root: Path, nodeids: Iterable[str]
) -> List[Tuple[List[str], str]]:
    """Same-scope, same-arguments, same-shape unparametrized test functions, with a merge."""
    groups: Dict[tuple, List[Tuple[str, ast.AST]]] = defaultdict(list)
    for nid in sorted(set(nodeids)):
        fn = _function(root, nid)
        if (
            "[" in nid
            or fn is None
            or any("parametrize" in ast.unparse(d) for d in fn.decorator_list)
        ):
            continue
        path, _, qual = nid.partition("::")
        key = (
            path,
            qual.rpartition("::")[0],
            ast.unparse(fn.args),
            fingerprint(fn, True),
        )
        groups[key].append((nid, fn))
    out = []
    for members in groups.values():
        code = suggest([fn for _, fn in members]) if len(members) > 1 else None
        if code:
            out.append(([nid for nid, _ in members], code))
    return sorted(out)
