"""``__slots__`` injection (aggressive option ``slots``).

    class Point:                     class Point:
        def __init__(self, x, y):        __slots__ = ('x', 'y', '__weakref__')
            self.x = x            ->     def __init__(self, x, y):
            self.y = y                       self.x = x
                                             self.y = y

A slotted class stores attributes in fixed offsets instead of a per-
instance ``__dict__``: attribute access gets faster and every instance
drops a dict allocation.  The rewrite is observable -- assigning an
attribute outside the slot set raises ``AttributeError``, ``vars(obj)``
stops working, protocol-0/1 pickling of instances fails -- which is why
the whole pass sits behind the aggressive ``slots`` option: enabling it
asserts that instances never receive attributes beyond those the class's
own methods assign and that no code relies on instance ``__dict__``.

What stays *proven* even under the assumption:

* only module-level classes with **no bases** (a non-slotted base would
  re-introduce ``__dict__`` and void the benefit), no decorators (a
  wrapper like ``@dataclass`` builds its own contract), no metaclass
  keywords, and no pre-existing ``__slots__``;
* the module has zero dynamic constructs (``vars(obj)`` is a dynamic
  construct, so surviving modules cannot introspect ``__dict__``), and
  neither ``setattr`` nor ``delattr`` is referenced anywhere in it (their
  targets cannot be enumerated statically);
* the class defines none of ``__setattr__`` / ``__getattr__`` /
  ``__getattribute__`` / ``__delattr__`` (self-managed attribute stores
  cannot be enumerated), no method carries a decorator other than
  ``staticmethod``/``classmethod``/``property``/accessor forms
  (``functools.cached_property`` stores through the instance ``__dict__``
  at runtime, invisibly), and no name collides between a class-level
  binding and an instance attribute (``__slots__`` would raise at class
  creation);
* the slot set is the union of every ``self.attr`` store/delete in the
  class's methods (first positional parameter as the instance name;
  ``@staticmethod``/``@classmethod``/``__new__`` excluded), so an unset
  slot read raises the same ``AttributeError`` a missing dict entry did;
* a module-visible extra attribute (``p = Point(1); p.tag = 1`` with
  ``tag`` outside the set) rejects the class outright -- the assumption
  is for what the analysis cannot see, not for what it can;
* the *class object* itself must stay confined to provably
  slot-compatible positions -- ``C(...)`` calls, gated
  ``isinstance``/``issubclass`` second arguments, single-base
  ``class D(C):`` -- because ``C.x = 5`` (directly or through an alias)
  overwrites the injected slot descriptor and breaks every later
  ``self.x`` store, and even a class-attribute *read* is observable;
* ``'__weakref__'`` is always included, so ``weakref.ref(obj)`` keeps
  working.

The pass runs **once, before the fixpoint loop, on the pristine module**
(the pipeline special-cases it like the jit pass): its rejections rest on
module-wide evidence -- a ``setattr`` reference, an alias, a visible
extra attribute -- and later passes may legitimately erase that evidence
(``unused`` removing a never-called function), which must not turn a
rejection into an injection on the next iteration.
"""

from __future__ import annotations

import ast

from ..analysis import builtin_gate, tree_has_dynamic


class SlotsInjection:
    """Pass driver."""

    name = "slotify"

    #: Set by the pipeline.
    aggressive: frozenset = frozenset()

    def __init__(self) -> None:
        self.changes = 0
        self.skipped_scopes = 0

    def run(self, tree: ast.Module) -> ast.Module:
        if "slots" not in self.aggressive:
            return tree
        if tree_has_dynamic(tree):
            self.skipped_scopes += 1
            return tree
        if any(
            isinstance(node, ast.Name) and node.id in ("setattr", "delattr")
            for node in ast.walk(tree)
        ):
            return tree
        candidates: dict[str, tuple[ast.ClassDef, frozenset]] = {}
        for stmt in tree.body:
            if isinstance(stmt, ast.ClassDef):
                attrs = _slot_attrs(stmt)
                if attrs is not None:
                    candidates[stmt.name] = (stmt, attrs)
        if not candidates:
            return tree
        for name in _escaping_class_objects(tree, candidates):
            del candidates[name]
        for name in _visibly_extended(tree, candidates):
            del candidates[name]
        for cd, attrs in candidates.values():
            slots = ast.Assign(
                targets=[ast.Name(id="__slots__", ctx=ast.Store())],
                value=ast.Constant((*sorted(attrs), "__weakref__")),
            )
            ast.copy_location(slots, cd)
            index = 1 if _has_docstring(cd) else 0
            cd.body.insert(index, slots)
            self.changes += 1
        return tree


def _has_docstring(cd: ast.ClassDef) -> bool:
    return bool(
        cd.body
        and isinstance(cd.body[0], ast.Expr)
        and isinstance(cd.body[0].value, ast.Constant)
        and isinstance(cd.body[0].value.value, str)
    )


_SELF_MANAGED = frozenset(
    {"__setattr__", "__getattr__", "__getattribute__", "__delattr__"}
)


def _slot_attrs(cd: ast.ClassDef) -> frozenset | None:
    """The instance-attribute slot set for *cd*, or None when the class
    must not be slotted."""
    if cd.bases or cd.keywords or cd.decorator_list:
        return None
    class_level: set[str] = set()
    attrs: set[str] = set()
    for stmt in cd.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                for node in ast.walk(target):
                    if isinstance(node, ast.Name):
                        class_level.add(node.id)
        elif isinstance(stmt, ast.AnnAssign):
            if isinstance(stmt.target, ast.Name):
                class_level.add(stmt.target.id)
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if stmt.name in _SELF_MANAGED:
                return None
            for deco in stmt.decorator_list:
                if isinstance(deco, ast.Name) and deco.id in (
                    "staticmethod", "classmethod", "property",
                ):
                    continue
                if isinstance(deco, ast.Attribute) and deco.attr in (
                    "setter", "deleter", "getter",
                ):
                    continue
                # Unknown method decorator: functools.cached_property and
                # friends store through the instance __dict__ at runtime,
                # invisibly to this analysis -- refuse the whole class.
                return None
            class_level.add(stmt.name)
            method_attrs = _method_attrs(stmt)
            if method_attrs is None:
                continue  # no instance parameter to observe
            attrs |= method_attrs
        elif isinstance(stmt, ast.ClassDef):
            class_level.add(stmt.name)
        elif isinstance(stmt, (ast.Expr, ast.Pass)):
            continue
        else:
            # if/try/with/loops at class level: bindings there are hard to
            # enumerate exactly -- refuse rather than guess.
            return None
    if "__slots__" in class_level or "__dict__" in attrs:
        return None
    if attrs & class_level:
        # ``__slots__`` naming a class variable raises at class creation.
        return None
    return frozenset(attrs)


def _method_attrs(method) -> set[str] | None:
    """``self.attr`` stores/deletes in *method*, or None when the first
    parameter is not an instance (static/class methods, ``__new__``)."""
    for deco in method.decorator_list:
        if isinstance(deco, ast.Name) and deco.id in (
            "staticmethod", "classmethod",
        ):
            return None
    if method.name == "__new__":
        return None
    args = method.args
    positional = args.posonlyargs + args.args
    if not positional:
        return None
    self_name = positional[0].arg
    out: set[str] = set()
    for node in ast.walk(method):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == self_name
            and isinstance(node.ctx, (ast.Store, ast.Del))
        ):
            out.add(node.attr)
    return out


def _escaping_class_objects(tree: ast.Module, candidates: dict) -> set[str]:
    """Candidate classes whose *class object* is used anywhere beyond the
    three provably slot-compatible positions:

    * ``C(...)`` -- the func of a call;
    * ``isinstance(x, C)`` / ``issubclass(x, C)`` second argument (bare or
      inside a tuple), when those builtin names pass the module gate;
    * the single base of a ``class D(C):`` definition (multi-base use is
      rejected: two injected slot sets would raise a lay-out conflict at
      class creation that the original never had).

    Everything else rejects: ``C.attr = v`` / ``del C.attr`` overwrite or
    drop a slot descriptor (breaking every later instance store), and even
    a *read* is observable (``C.x`` raises ``AttributeError`` without
    slots but returns the descriptor with them); an alias ``D = C``, an
    argument position ``g(C)`` or a container display hand the class
    object to code that can do any of the above invisibly."""
    gate = {
        name: builtin_gate(tree, name)
        for name in ("isinstance", "issubclass")
    }
    allowed: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                allowed.add(id(node.func))
                if (
                    node.func.id in gate
                    and gate[node.func.id]
                    and len(node.args) == 2
                    and not node.keywords
                ):
                    second = node.args[1]
                    if isinstance(second, ast.Name):
                        allowed.add(id(second))
                    elif isinstance(second, ast.Tuple):
                        for elt in second.elts:
                            if isinstance(elt, ast.Name):
                                allowed.add(id(elt))
        elif isinstance(node, ast.ClassDef):
            if len(node.bases) == 1 and isinstance(node.bases[0], ast.Name):
                allowed.add(id(node.bases[0]))
    escaping: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in candidates
            and id(node) not in allowed
        ):
            escaping.add(node.id)
    return escaping


def _visibly_extended(tree: ast.Module, candidates: dict) -> set[str]:
    """Classes whose instances visibly receive attributes outside the slot
    set somewhere in the module: ``p = C(...)`` then ``p.tag = 1``.  Flow
    is ignored on purpose -- any name ever bound from ``C(...)`` taints
    every attribute store through that name, which can only over-reject."""
    producers: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id in candidates
        ):
            producers.setdefault(node.targets[0].id, set()).add(
                node.value.func.id
            )
    rejected: set[str] = set()
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Attribute)
            and isinstance(node.ctx, (ast.Store, ast.Del))
        ):
            continue
        base = node.value
        classes: set[str] = set()
        if isinstance(base, ast.Name):
            classes = producers.get(base.id, set())
        elif (
            isinstance(base, ast.Call)
            and isinstance(base.func, ast.Name)
            and base.func.id in candidates
        ):
            classes = {base.func.id}
        for cls in classes:
            if cls not in rejected and node.attr not in candidates[cls][1]:
                rejected.add(cls)
    return rejected
