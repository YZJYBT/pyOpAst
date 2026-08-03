"""Common subexpression elimination (CSE).

Within one statement block, an expression appearing two or more times is
computed once into a ``_opast_cse_N`` temporary inserted before the
statement of its first occurrence, and every occurrence is replaced by the
temp.  Eligible expressions are exactly the LICM ones
(:func:`opast.analysis.hoistable_int_expr`): provably pure and total int
arithmetic over proven-int names, plus ``len()`` of fresh never-escaping
builtin containers (module-wide gate on no dynamic constructs / unshadowed
builtins).  Purity+totality is what makes this sound without occurrence
dominance: even if a later occurrence sits inside a branch or nested loop,
evaluating the expression once up front is unobservable.

Additional requirements per expression:

* all names are definitely bound before the first occurrence's statement
  (same straight-line dominance scan as LICM);
* no name is rebound anywhere in the statement span from the first to the
  last occurrence (full-subtree binding scan, nested ``global``/``nonlocal``
  included);
* it contains at least one name and at least one operation or ``len`` call
  (repeated plain names/constants are free anyway).

Under the aggressive ``pure-calls`` option, calls to trusted-pure module
functions (see :func:`opast.purity.trusted_pure_functions`; never shadowed
in the enclosing scope) also qualify when every argument is a constant, a
plain name, a hoistable numeric expression, or another such call --
argument names still pass the same boundness and no-rebind-in-span checks.
The option's stated assumption covers the semantic deltas: a later
occurrence inside a branch may now be evaluated up front, and the merged
occurrences share one result object.

Occurrences are matched *maximally* (a recorded expression's subexpressions
are not counted separately) and a candidate that is a subexpression of
another selected candidate is deferred to a later pipeline iteration.
Blocks are processed independently; nested scope bodies are never entered.
Function scopes only.
"""

from __future__ import annotations

import ast
import copy

from ..analysis import (
    AnnotationTrust,
    bound_names,
    builtin_gate,
    contains_name,
    fresh_container_names,
    hoistable_num_expr,
    infer_float_names,
    infer_int_names,
    is_len_call,
    param_names,
)
from ..safety import region_is_dynamic
from .base import ScopedTransformer
from .licm import (
    _LOOPS,
    container_gate,
    definite_bindings,
    subtree_bindings,
    unbound_risk_names,
)

_TEMP_PREFIX = "_opast_cse"


def _worthy(node: ast.AST, containers) -> bool:
    """Repeating this expression actually saves work."""
    if not contains_name(node):
        return False
    return isinstance(node, (ast.BinOp, ast.UnaryOp)) or is_len_call(node, containers)


class _Collector(ast.NodeVisitor):
    """Records maximal eligible expressions per statement index."""

    def __init__(self, ints, floats, containers, pure=frozenset()) -> None:
        self.ints = ints
        self.floats = floats
        self.containers = containers
        self.pure = pure
        self.table: dict[str, list[tuple[int, ast.AST]]] = {}
        self._idx = 0

    def collect(self, idx: int, stmt: ast.stmt) -> None:
        self._idx = idx
        self.visit(stmt)

    def _skip(self, node: ast.AST) -> None:
        pass

    visit_FunctionDef = _skip
    visit_AsyncFunctionDef = _skip
    visit_Lambda = _skip
    visit_ClassDef = _skip

    def _pure_call(self, node: ast.AST) -> bool:
        """Aggressive ``pure-calls`` eligibility.  Argument names are
        accepted here; the selection loop still checks their boundness and
        rebinding over the occurrence span."""
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in self.pure
        ):
            return False
        return all(self._arg_ok(a) for a in node.args) and all(
            k.arg is not None and self._arg_ok(k.value) for k in node.keywords
        )

    def _arg_ok(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Constant):
            return True
        if isinstance(node, ast.Name):
            return isinstance(node.ctx, ast.Load)
        if isinstance(node, ast.Call):
            return self._pure_call(node)
        return hoistable_num_expr(node, self.ints, self.floats, self.containers)

    def _record(self, node: ast.AST) -> None:
        if (
            hoistable_num_expr(node, self.ints, self.floats, self.containers)
            and _worthy(node, self.containers)
        ) or self._pure_call(node):
            self.table.setdefault(ast.dump(node), []).append((self._idx, node))
            # keep descending: a shared subexpression may repeat inside
            # otherwise-distinct larger expressions
        self.generic_visit(node)

    visit_BinOp = _record
    visit_UnaryOp = _record
    visit_Call = _record


class _Replacer(ast.NodeTransformer):
    """Replaces selected maximal expressions with their temp names, using
    the same traversal shape as :class:`_Collector`."""

    def __init__(self, active: dict[str, str], ints, floats, containers, owner) -> None:
        self.active = active
        self.ints = ints
        self.floats = floats
        self.containers = containers
        self.owner = owner

    def _skip(self, node: ast.AST) -> ast.AST:
        return node

    visit_FunctionDef = _skip
    visit_AsyncFunctionDef = _skip
    visit_Lambda = _skip
    visit_ClassDef = _skip

    def _replace(self, node: ast.AST) -> ast.AST:
        temp = self.active.get(ast.dump(node))
        if temp is not None:
            return ast.copy_location(ast.Name(id=temp, ctx=ast.Load()), node)
        return self.generic_visit(node)

    visit_BinOp = _replace
    visit_UnaryOp = _replace
    visit_Call = _replace


class CommonSubexpressionElimination(ScopedTransformer):
    name = "cse"

    def run(self, tree: ast.Module) -> ast.Module:
        if region_is_dynamic(tree):
            self.skipped_scopes += 1
            return tree
        self._len_ok = container_gate(tree)
        self._range_ok = builtin_gate(tree, "range")
        self._trust = AnnotationTrust(
            tree if "annotations" in self.aggressive else None
        )
        self._counter = 0
        for n in ast.walk(tree):
            ident = None
            if isinstance(n, ast.Name):
                ident = n.id
            elif isinstance(n, ast.arg):
                ident = n.arg
            if ident and ident.startswith(_TEMP_PREFIX + "_"):
                suffix = ident[len(_TEMP_PREFIX) + 1:]
                if suffix.isdigit():
                    self._counter = max(self._counter, int(suffix) + 1)
        return self.visit(tree)

    def _fresh_name(self) -> str:
        name = f"{_TEMP_PREFIX}_{self._counter}"
        self._counter += 1
        return name

    def _visit_function(self, node: ast.AST) -> ast.AST:
        if region_is_dynamic(node):
            self.skipped_scopes += 1
            return node
        node = self.generic_visit(node)  # nested functions first
        containers = fresh_container_names(node) if self._len_ok else frozenset()
        ints = infer_int_names(
            node,
            containers,
            range_ok=self._range_ok,
            trusted=self._trust.ints(node),
            typed_calls=self._trust.int_returns,
        )
        floats = infer_float_names(
            node,
            ints=ints,
            containers=containers,
            trusted=self._trust.floats(node),
            typed_calls=self._trust.float_returns,
        )
        # Trusted-pure callees must resolve to the module global: any local
        # binding of the name in this scope shadows it.
        self._scope_pure = (
            self.pure_calls - bound_names(node) if self.pure_calls else frozenset()
        )
        if ints or floats or containers or self._scope_pure:
            # Parameters are bound on entry (same reason as in LICM).
            node.body = self._process_block(
                node.body, set(param_names(node)), ints, floats, containers
            )
        return node

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function

    # -- one block ----------------------------------------------------------
    def _process_block(self, stmts, bound: set[str], ints, floats, containers) -> list:
        stmts = self._eliminate(stmts, bound, ints, floats, containers)

        # Recurse into nested blocks for expressions local to them.
        tracked = set(bound)
        for stmt in stmts:
            if isinstance(stmt, _LOOPS):
                inner = tracked - unbound_risk_names(stmt)
                if isinstance(stmt, (ast.For, ast.AsyncFor)):
                    # Loop targets are (re)bound before every body run.
                    inner |= {
                        n.id
                        for n in ast.walk(stmt.target)
                        if isinstance(n, ast.Name)
                        and isinstance(n.ctx, ast.Store)
                    }
                stmt.body = self._process_block(stmt.body, inner, ints, floats, containers)
            elif isinstance(stmt, ast.If):
                stmt.body = self._process_block(stmt.body, tracked, ints, floats, containers)
                stmt.orelse = self._process_block(
                    stmt.orelse, tracked, ints, floats, containers
                )
            elif isinstance(stmt, (ast.With, ast.AsyncWith)):
                stmt.body = self._process_block(stmt.body, tracked, ints, floats, containers)
            elif isinstance(stmt, ast.Try) or (
                hasattr(ast, "TryStar") and isinstance(stmt, ast.TryStar)
            ):
                stmt.body = self._process_block(stmt.body, tracked, ints, floats, containers)
                for handler in stmt.handlers:
                    handler.body = self._process_block(
                        handler.body, tracked, ints, floats, containers
                    )
                stmt.orelse = self._process_block(
                    stmt.orelse, tracked, ints, floats, containers
                )
                stmt.finalbody = self._process_block(
                    stmt.finalbody, tracked, ints, floats, containers
                )
            elif isinstance(stmt, ast.Match):
                for case in stmt.cases:
                    case.body = self._process_block(
                        case.body, tracked, ints, floats, containers
                    )
            tracked |= definite_bindings(stmt)
            tracked -= unbound_risk_names(stmt)
        return stmts

    def _eliminate(self, stmts, bound: set[str], ints, floats, containers) -> list:
        collector = _Collector(ints, floats, containers, pure=self._scope_pure)
        bound_at: list[set[str]] = []
        rebinds: list[set[str]] = []
        running = set(bound)
        for idx, stmt in enumerate(stmts):
            bound_at.append(set(running))
            collector.collect(idx, stmt)
            rebinds.append(subtree_bindings(stmt))
            running |= definite_bindings(stmt)
            running -= unbound_risk_names(stmt)

        selected: dict[str, tuple[int, str, ast.AST]] = {}
        # Names exempt from boundness/rebinding checks: trusted-pure callees
        # resolve to the module global (shadowing already excluded) and the
        # ``len`` name is builtin-stable via the module-wide gate.  Every
        # other name in a candidate -- proven-numeric operands and pure-call
        # arguments alike -- must be definitely bound before the first
        # occurrence and never rebound across the span.
        stable = set(self._scope_pure)
        if self._len_ok:
            stable.add("len")
        for key, occurrences in collector.table.items():
            if len(occurrences) < 2:
                continue
            first = occurrences[0][0]
            last = occurrences[-1][0]
            sample = occurrences[0][1]
            names = {
                n.id for n in ast.walk(sample) if isinstance(n, ast.Name)
            } - stable
            if not names <= bound_at[first]:
                continue
            if any(names & rebinds[i] for i in range(first, last + 1)):
                continue
            selected[key] = (first, "", sample)
        if not selected:
            return list(stmts)

        # A candidate that is a subexpression of another selected candidate
        # is deferred to a later pipeline iteration (dump() of a subtree is a
        # substring of its parent's dump()).
        for key in list(selected):
            if any(key != other and key in other for other in selected):
                del selected[key]

        for key, (first, _, sample) in list(selected.items()):
            selected[key] = (first, self._fresh_name(), sample)

        out: list[ast.stmt] = []
        for idx, stmt in enumerate(stmts):
            for key, (first, temp, sample) in selected.items():
                if first == idx:
                    target = ast.copy_location(
                        ast.Name(id=temp, ctx=ast.Store()), stmt
                    )
                    assign = ast.copy_location(
                        ast.Assign(targets=[target], value=copy.deepcopy(sample)),
                        stmt,
                    )
                    out.append(assign)
                    self.changes += 1
            active = {
                key: temp
                for key, (first, temp, _) in selected.items()
                if first <= idx
            }
            if active:
                stmt = _Replacer(active, ints, floats, containers, self).visit(stmt)
            out.append(stmt)
        return out
