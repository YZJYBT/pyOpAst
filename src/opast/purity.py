"""Trusted-pure function inference (aggressive ``pure-calls``).

Computes the set of module-level functions that the ``pure-calls``
aggressive option lets LICM, CSE and unused-elimination treat as *pure*:
same argument values give the same result and evaluation has no observable
effect.  With that trust, a call whose arguments are loop-invariant can be
hoisted, two identical calls can be computed once, and a call whose result
is dead can be dropped -- the exact rewrites those passes already perform
for proven int/float arithmetic.

This is an evidence check, not a proof (that is why it is aggressive):
every *visible* side-effect channel disqualifies a function, and the user's
one stated assumption covers the invisible ones (dunder dispatch such as
``__len__``/``__getitem__``/``__hash__`` doing effectful work, runtime
monkey-patching of the module, callers mutating argument objects across the
optimized region).  What must not depend on the assumption -- and therefore
is checked structurally -- is that the *body* writes nothing it did not
create and reads nothing that can change behind its back:

* module top-level ``def`` only, no decorators, name bound exactly once in
  the whole module (no ``global f`` rebinds, no ``f = wrap(f)``, no
  ``del f``), no ``f.attr = ...`` store anywhere;
* immutable parameter defaults (a shared mutable default is hidden state);
* body contains no nested scope (``def``/``lambda``/``class``), no
  ``yield``/``await``, no generator expression (a returned generator is a
  stateful object), no ``global``/``nonlocal``, no ``import``, no ``with``
  (context managers are effects by design);
* no attribute access at all, and no subscript *store*/*delete* -- the only
  writable places left are the function's own locals;
* every name the body reads resolves to one of: its own parameters/locals,
  another trusted-pure candidate (mutual recursion converges via the
  fixpoint below), an unshadowed builtin from :data:`PURE_BUILTINS`, or a
  module-level constant bound exactly once to an immutable literal
  (mutable module state may change between calls; immutable state cannot).

Comprehension targets live in their own runtime scope, so a name bound
*only* as a comprehension target counts as local inside that comprehension
and as an unresolvable global read outside it.

The set is decided **once, on the pristine module, before the fixpoint
loop** (same reasoning as ``slots``): rejections rest on module-wide
evidence -- a second binding, an attribute store, a ``setattr``-style
dynamic construct -- that other passes may legitimately erase, and a
rejection must stay a rejection.
"""

from __future__ import annotations

import ast

from .analysis import binding_names
from .safety import tree_has_dynamic

#: Builtins whose calls are deterministic value computations: no I/O, no
#: mutation of their arguments, result a function of the argument values.
#: They gate what a candidate *body* may call -- the pure-calls machinery
#: itself never hoists or dedupes builtin calls (``len`` already has its own
#: proof-backed path).  ``iter``/``next`` are deliberately absent: consuming
#: an iterator is mutation.
PURE_BUILTINS = frozenset({
    "abs", "all", "any", "bool", "bytes", "chr", "complex", "dict",
    "divmod", "enumerate", "filter", "float", "format", "frozenset",
    "hash", "hex", "int", "isinstance", "issubclass", "len", "list",
    "map", "max", "min", "oct", "ord", "pow", "range", "repr",
    "reversed", "round", "set", "sorted", "str", "sum", "tuple", "zip",
})

#: Additional builtin names a candidate body may *read* (exception classes
#: for ``raise``/``except``, plus ``NotImplemented``): stable, and either
#: deterministic to construct or never called at all.
STABLE_BUILTIN_READS = frozenset({
    "ArithmeticError", "AssertionError", "AttributeError", "BaseException",
    "Exception", "IndexError", "KeyError", "LookupError", "NameError",
    "NotImplemented", "NotImplementedError", "OverflowError",
    "RecursionError", "RuntimeError", "StopIteration", "TypeError",
    "UnboundLocalError", "ValueError", "ZeroDivisionError",
})

_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp)


def immutable_literal(node: ast.AST) -> bool:
    """A literal expression whose value is immutable all the way down."""
    if isinstance(node, ast.Constant):
        return True  # str/bytes/num/bool/None/Ellipsis -- all immutable
    if isinstance(node, ast.Tuple) and isinstance(node.ctx, ast.Load):
        return all(immutable_literal(e) for e in node.elts)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return isinstance(node.operand, ast.Constant)
    return False


def _binding_counts(tree: ast.Module) -> dict[str, int]:
    """How many times each name is bound anywhere in the tree (parameters
    and nested scopes included -- one extra binding anywhere is enough to
    distrust a module-level name)."""
    counts: dict[str, int] = {}
    for node in ast.walk(tree):
        for name in binding_names(node):
            counts[name] = counts.get(name, 0) + 1
        if isinstance(node, ast.arg):
            counts[node.arg] = counts.get(node.arg, 0) + 1
    return counts


def _attr_stored_names(tree: ast.Module) -> set[str]:
    """Names whose *object* visibly gains or loses an attribute somewhere
    (``f.cache = ...`` / ``del f.cache``)."""
    stored: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.ctx, (ast.Store, ast.Del))
            and isinstance(node.value, ast.Name)
        ):
            stored.add(node.value.id)
    return stored


def _module_constants(tree: ast.Module, counts: dict[str, int]) -> frozenset[str]:
    """Module-level names bound exactly once, to an immutable literal."""
    consts: set[str] = set()
    for stmt in tree.body:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and counts.get(stmt.targets[0].id, 0) == 1
            and immutable_literal(stmt.value)
        ):
            consts.add(stmt.targets[0].id)
    return frozenset(consts)


def _immutable_defaults(args: ast.arguments) -> bool:
    for default in list(args.defaults) + list(args.kw_defaults):
        if default is not None and not immutable_literal(default):
            return False
    return True


class _BodyScan(ast.NodeVisitor):
    """Structural purity scan of one candidate body.

    Records every global (non-local) name read; sets ``rejected`` on the
    first visible effect channel.  Comprehension targets are tracked as a
    scope stack so ``[x for x in xs]`` reads ``x`` locally while a stray
    ``x`` outside the comprehension is treated as a global read.
    """

    def __init__(self, fn: ast.FunctionDef) -> None:
        self.rejected = False
        self.reads: set[str] = set()
        self._comp_stack: list[frozenset[str]] = []
        self._locals, self._comp_only = self._collect_locals(fn)

    @staticmethod
    def _collect_locals(fn: ast.FunctionDef) -> tuple[set[str], set[str]]:
        params = set()
        a = fn.args
        for p in [*a.posonlyargs, *a.args, *a.kwonlyargs, a.vararg, a.kwarg]:
            if p is not None:
                params.add(p.arg)
        comp_target_nodes: set[int] = set()
        comp_targets: set[str] = set()
        for node in ast.walk(fn):
            if isinstance(node, _COMPREHENSIONS):
                for gen in node.generators:
                    for n in ast.walk(gen.target):
                        if isinstance(n, ast.Name):
                            comp_target_nodes.add(id(n))
                            comp_targets.add(n.id)
        other: set[str] = set(params)
        for node in ast.walk(fn):
            if isinstance(node, ast.Name) and id(node) in comp_target_nodes:
                continue
            other.update(binding_names(node))
        return other | comp_targets, comp_targets - other

    def visit(self, node: ast.AST) -> None:  # early-out on rejection
        if self.rejected:
            return
        super().visit(node)

    def _reject(self, node: ast.AST) -> None:
        self.rejected = True

    visit_FunctionDef = _reject
    visit_AsyncFunctionDef = _reject
    visit_Lambda = _reject
    visit_ClassDef = _reject
    visit_Yield = _reject
    visit_YieldFrom = _reject
    visit_Await = _reject
    visit_GeneratorExp = _reject
    visit_Global = _reject
    visit_Nonlocal = _reject
    visit_Import = _reject
    visit_ImportFrom = _reject
    visit_With = _reject
    visit_AsyncWith = _reject
    visit_Attribute = _reject  # any access: reads may be properties too

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.rejected = True
            return
        self.generic_visit(node)

    def _visit_comp(self, node: ast.AST) -> None:
        targets: set[str] = set()
        for gen in node.generators:
            for n in ast.walk(gen.target):
                if isinstance(n, ast.Name):
                    targets.add(n.id)
        self._comp_stack.append(frozenset(targets))
        try:
            self.generic_visit(node)
        finally:
            self._comp_stack.pop()

    visit_ListComp = _visit_comp
    visit_SetComp = _visit_comp
    visit_DictComp = _visit_comp

    def visit_Name(self, node: ast.Name) -> None:
        if not isinstance(node.ctx, ast.Load):
            return
        name = node.id
        if any(name in scope for scope in self._comp_stack):
            return
        if name in self._locals and name not in self._comp_only:
            return
        # Includes names bound only as comprehension targets but read
        # outside any comprehension binding them: at runtime that read is a
        # global (or an error), so it must pass the global gate.
        self.reads.add(name)


def _scan_body(fn: ast.FunctionDef) -> set[str] | None:
    """Global reads of *fn*'s body, or ``None`` if structurally impure."""
    scan = _BodyScan(fn)
    for stmt in fn.body:
        scan.visit(stmt)
        if scan.rejected:
            return None
    return scan.reads


def trusted_pure_functions(tree: ast.Module) -> frozenset[str]:
    """The set of module-level function names trusted as pure under the
    ``pure-calls`` assumption.  Empty when the module has any dynamic
    construct (``eval``/``exec``/frame introspection could rebind or
    observe anything)."""
    if tree_has_dynamic(tree):
        return frozenset()
    counts = _binding_counts(tree)
    attr_stored = _attr_stored_names(tree)
    consts = _module_constants(tree, counts)
    builtins_ok = (PURE_BUILTINS | STABLE_BUILTIN_READS) - set(counts)
    candidates: dict[str, set[str]] = {}
    for stmt in tree.body:
        if (
            isinstance(stmt, ast.FunctionDef)
            and not stmt.decorator_list
            and counts.get(stmt.name, 0) == 1
            and stmt.name not in attr_stored
            and _immutable_defaults(stmt.args)
        ):
            reads = _scan_body(stmt)
            if reads is not None:
                candidates[stmt.name] = reads
    # Optimistic fixpoint: drop every candidate reading a global outside the
    # allowed set, until stable.  Mutually recursive pure functions survive;
    # a call into anything impure sinks the whole strongly-connected group.
    changed = True
    while changed:
        changed = False
        allowed = builtins_ok | consts | set(candidates)
        for name in list(candidates):
            if not candidates[name] <= allowed:
                del candidates[name]
                changed = True
    return frozenset(candidates)
