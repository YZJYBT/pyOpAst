"""Conservative detection of dynamic constructs.

opast only transforms code it can reason about statically.  A *region* is the
executable code that belongs directly to one scope (module, function, lambda
or class body), excluding the bodies of nested scopes but *including* the
parts of nested definitions that evaluate in the enclosing scope (decorators,
default values, annotations, base classes).

A region is considered *dynamic* (tainted) when it contains any of:

* a use of a name in :data:`DYNAMIC_NAMES` (``eval``, ``exec``, ``globals``,
  ``locals``, ``vars``, ``compile``, ``__import__``, ``breakpoint``) -- in any
  context, so a mere reference or a local shadowing assignment also taints;
* an attribute access whose attribute name is in :data:`DYNAMIC_ATTRIBUTES`
  (e.g. ``builtins.eval``, ``sys._getframe``, ``frame.f_locals``,
  ``importlib.import_module``, ``inspect.currentframe``);
* ``from module import *`` (unknown names may enter the scope);
* ``from builtins import eval`` style imports of dynamic names.

The check is purely name based and deliberately over-approximates: a harmless
local variable called ``eval`` still disables optimisation of that scope.
Tainted scopes are skipped entirely by every pass; a tainted module top level
disables all optimisation of the file; the inlining pass additionally requires
the whole module to be free of dynamic constructs (because ``exec``/``eval``
executed anywhere can rebind module globals).
"""

from __future__ import annotations

import ast

DYNAMIC_NAMES = frozenset({
    "eval",
    "exec",
    "compile",
    "globals",
    "locals",
    "vars",
    "__import__",
    "breakpoint",
})

DYNAMIC_ATTRIBUTES = frozenset({
    "eval",
    "exec",
    "compile",
    "globals",
    "locals",
    "vars",
    "__import__",
    "_getframe",
    "f_locals",
    "f_globals",
    "f_back",
    "settrace",
    "setprofile",
    "gi_frame",
    "cr_frame",
    "currentframe",
    "import_module",
})

SCOPE_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


def _arg_region(args: ast.arguments):
    """Expressions of an ``arguments`` node evaluated in the enclosing scope."""
    for default in args.defaults:
        yield default
    for default in args.kw_defaults:
        if default is not None:
            yield default
    all_args = list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)
    if args.vararg is not None:
        all_args.append(args.vararg)
    if args.kwarg is not None:
        all_args.append(args.kwarg)
    for a in all_args:
        if a.annotation is not None:
            yield a.annotation


def _boundary_children(node: ast.AST):
    """Children of a nested scope node that still belong to the *enclosing*
    region (bodies are excluded -- they form their own region)."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        yield from node.decorator_list
        yield from _arg_region(node.args)
        if node.returns is not None:
            yield node.returns
    elif isinstance(node, ast.Lambda):
        yield from _arg_region(node.args)
    elif isinstance(node, ast.ClassDef):
        yield from node.decorator_list
        yield from node.bases
        yield from node.keywords


def iter_region(scope: ast.AST):
    """Yield every AST node belonging to *scope*'s own executable region.

    Nested scope nodes themselves are yielded (a nested ``def`` is a statement
    of this region) but their bodies are not descended into.  Comprehensions
    are conservatively treated as part of the enclosing region.
    """
    if isinstance(scope, ast.Lambda):
        start = [scope.body]
    else:
        # Module, FunctionDef, AsyncFunctionDef, ClassDef
        start = list(scope.body)
    stack = list(start)
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, SCOPE_NODES):
            stack.extend(_boundary_children(node))
        else:
            stack.extend(ast.iter_child_nodes(node))


def _node_is_dynamic(node: ast.AST) -> bool:
    if isinstance(node, ast.Name) and node.id in DYNAMIC_NAMES:
        return True
    if isinstance(node, ast.Attribute) and node.attr in DYNAMIC_ATTRIBUTES:
        return True
    if isinstance(node, ast.ImportFrom):
        for alias in node.names:
            if alias.name == "*" or alias.name in DYNAMIC_NAMES:
                return True
    return False


def region_is_dynamic(scope: ast.AST) -> bool:
    """True if *scope*'s own region contains a dynamic construct."""
    return any(_node_is_dynamic(n) for n in iter_region(scope))


def tree_has_dynamic(tree: ast.AST) -> bool:
    """True if *any* node of the whole tree is a dynamic construct."""
    return any(_node_is_dynamic(n) for n in ast.walk(tree))
