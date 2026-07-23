"""De-dynamization: expand pointless dynamic code into plain static code.

Rewrites (all require the dynamic builtin to really be the builtin, see the
all-or-nothing rule below):

* module level ``eval('<constant expression>')``  -> the parsed expression
  (rejected if the string does not parse or contains a walrus -- inside
  ``eval`` a walrus writes to a discarded namespace, inlined it would bind
  for real);
* module level ``exec('<constant statements>')`` used as a statement -> the
  parsed statements spliced in place (``from __future__ import`` and
  ``__debug__`` bindings are rejected; an empty string becomes ``pass``);
* module level ``globals()['x'] = value`` / ``vars()['x'] = value`` ->
  ``x = value`` (reads are *not* rewritten: ``globals()['x']`` raises
  ``KeyError`` where a bare name raises ``NameError``);
* ``getattr(obj, 'attr')`` -> ``obj.attr`` (any scope);
* ``setattr(name, 'attr', value)`` / ``delattr(name, 'attr')`` statements ->
  ``name.attr = value`` / ``del name.attr`` (the object must be a plain name
  so evaluation order of side effects is preserved).

``eval``/``exec``/``globals`` are only expanded at module top level: inside a
function ``eval``/``exec`` see a *snapshot* of the locals (an unbound-yet
local falls back to globals; ``exec('x = 1')`` cannot rebind a real local),
so inlining there would change semantics.  Attribute names must be constant
strings that are valid non-keyword identifiers.

All-or-nothing rule
-------------------
The rewrites are only committed when the *result* module is completely free
of dynamic constructs (per :mod:`opast.safety`) and none of the relied-upon
builtin names (``getattr``/``setattr``/``delattr`` for performed rewrites) is
bound anywhere in the result.  Otherwise the module is returned untouched:
any remaining ``exec``/``eval``/``import *`` could rebind the builtins our
rewrites depend on, so a partial expansion would be unsound.  On success the
module becomes fully static and every other pass (including inlining, which
requires a taint-free module) is unlocked.
"""

from __future__ import annotations

import ast
import copy
import keyword

from ..analysis import all_bound_names
from ..safety import tree_has_dynamic

#: Names that make it worth attempting this pass at all.
_TRIGGER_NAMES = frozenset(
    {"eval", "exec", "globals", "vars", "getattr", "setattr", "delattr"}
)

_FAMILY = frozenset({"getattr", "setattr", "delattr"})

_INNER_SCOPES = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.Lambda,
    ast.ClassDef,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)


def _const_identifier(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.isidentifier()
        and not keyword.iskeyword(node.value)
        and node.value != "__debug__"  # assignable via dict, not via name
    )


def _is_builtin_call(node: ast.AST, names: frozenset | set, argc: int) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in names
        and len(node.args) == argc
        and not node.keywords
        and not any(isinstance(a, ast.Starred) for a in node.args)
    )


def _const_str_arg(call: ast.Call) -> str | None:
    arg = call.args[0]
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return arg.value
    return None


def _relocate(node: ast.AST, anchor: ast.AST) -> ast.AST:
    """Point every location in *node* (parsed from a string) at *anchor*."""
    for child in ast.walk(node):
        if "lineno" in getattr(child, "_attributes", ()):
            ast.copy_location(child, anchor)
    return node


class _Rewriter(ast.NodeTransformer):
    def __init__(self, family_allowed: frozenset[str]) -> None:
        self.family_allowed = family_allowed
        self.family_used: set[str] = set()
        self.changes = 0
        self._depth = 0  # 0 == module region

    def _visit_inner_scope(self, node: ast.AST) -> ast.AST:
        self._depth += 1
        try:
            return self.generic_visit(node)
        finally:
            self._depth -= 1

    visit_FunctionDef = _visit_inner_scope
    visit_AsyncFunctionDef = _visit_inner_scope
    visit_Lambda = _visit_inner_scope
    visit_ClassDef = _visit_inner_scope
    visit_ListComp = _visit_inner_scope
    visit_SetComp = _visit_inner_scope
    visit_DictComp = _visit_inner_scope
    visit_GeneratorExp = _visit_inner_scope

    # -- eval / getattr (expression positions) ---------------------------
    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = self.generic_visit(node)
        if self._depth == 0 and _is_builtin_call(node, {"eval"}, 1):
            text = _const_str_arg(node)
            if text is not None:
                expr = self._parse_eval(text)
                if expr is not None:
                    self.changes += 1
                    return _relocate(expr, node)
        if (
            "getattr" in self.family_allowed
            and _is_builtin_call(node, {"getattr"}, 2)
            and _const_identifier(node.args[1])
        ):
            self.changes += 1
            self.family_used.add("getattr")
            new = ast.Attribute(value=node.args[0], attr=node.args[1].value,
                                ctx=ast.Load())
            return ast.copy_location(new, node)
        return node

    @staticmethod
    def _parse_eval(text: str) -> ast.expr | None:
        try:
            parsed = ast.parse(text, mode="eval")
        except (SyntaxError, ValueError):
            return None  # runtime SyntaxError preserved by not rewriting
        if any(isinstance(n, ast.NamedExpr) for n in ast.walk(parsed)):
            return None
        return parsed.body

    # -- exec / setattr / delattr (statement positions) -------------------
    def visit_Expr(self, node: ast.Expr):
        node = self.generic_visit(node)
        if not isinstance(node, ast.Expr):
            return node
        value = node.value
        if self._depth == 0 and _is_builtin_call(value, {"exec"}, 1):
            text = _const_str_arg(value)
            if text is not None:
                stmts = self._parse_exec(text)
                if stmts is not None:
                    self.changes += 1
                    return [_relocate(s, node) for s in stmts]
        if (
            "setattr" in self.family_allowed
            and _is_builtin_call(value, {"setattr"}, 3)
            and isinstance(value.args[0], ast.Name)
            and _const_identifier(value.args[1])
        ):
            self.changes += 1
            self.family_used.add("setattr")
            target = ast.Attribute(value=value.args[0], attr=value.args[1].value,
                                   ctx=ast.Store())
            new = ast.Assign(targets=[target], value=value.args[2])
            return _relocate(new, node)
        if (
            "delattr" in self.family_allowed
            and _is_builtin_call(value, {"delattr"}, 2)
            and isinstance(value.args[0], ast.Name)
            and _const_identifier(value.args[1])
        ):
            self.changes += 1
            self.family_used.add("delattr")
            target = ast.Attribute(value=value.args[0], attr=value.args[1].value,
                                   ctx=ast.Del())
            return _relocate(ast.Delete(targets=[target]), node)
        return node

    @staticmethod
    def _parse_exec(text: str) -> list[ast.stmt] | None:
        try:
            parsed = ast.parse(text, mode="exec")
        except (SyntaxError, ValueError):
            return None
        for n in ast.walk(parsed):
            if isinstance(n, ast.ImportFrom) and n.module == "__future__":
                return None  # must stay at file top; splicing would misplace it
            if (
                isinstance(n, ast.Name)
                and isinstance(n.ctx, (ast.Store, ast.Del))
                and n.id == "__debug__"
            ):
                return None  # legal through exec, SyntaxError as plain code
        if not parsed.body:
            return [ast.Pass()]  # keep enclosing blocks non-empty
        return parsed.body

    # -- globals()['x'] = value -------------------------------------------
    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        node = self.generic_visit(node)
        if self._depth != 0 or len(node.targets) != 1:
            return node
        target = node.targets[0]
        if not isinstance(target, ast.Subscript):
            return node
        if not _is_builtin_call(target.value, {"globals", "vars"}, 0):
            return node
        if not _const_identifier(target.slice):
            return node
        self.changes += 1
        name = ast.copy_location(
            ast.Name(id=target.slice.value, ctx=ast.Store()), target
        )
        return ast.copy_location(ast.Assign(targets=[name], value=node.value), node)


class DeDynamize:
    """Pipeline pass (runs first). Not a :class:`ScopedTransformer`: it must
    run precisely on modules the other passes would skip as dynamic."""

    name = "de-dynamize"

    def __init__(self) -> None:
        self.changes = 0
        self.skipped_scopes = 0  # counts aborted (rolled back) attempts

    def run(self, tree: ast.Module) -> ast.Module:
        if not any(
            isinstance(n, ast.Name) and n.id in _TRIGGER_NAMES
            for n in ast.walk(tree)
        ):
            return tree
        family_allowed = frozenset(_FAMILY - all_bound_names(tree))
        work = copy.deepcopy(tree)
        rewriter = _Rewriter(family_allowed)
        work = rewriter.visit(work)
        if rewriter.changes == 0:
            return tree
        # All-or-nothing: commit only if the rewritten module is fully static
        # and nothing (e.g. spliced exec content) binds a builtin we relied on.
        if tree_has_dynamic(work):
            self.skipped_scopes += 1
            return tree
        if rewriter.family_used & all_bound_names(work):
            self.skipped_scopes += 1
            return tree
        ast.fix_missing_locations(work)
        self.changes = rewriter.changes
        return work
