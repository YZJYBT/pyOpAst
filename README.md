# opast

> **English** | [中文](README-ZH.md) (the Chinese document carries the exhaustive per-pass safety conditions; this one is the canonical overview)

[![CI](https://github.com/YZJYBT/pyOpAst/actions/workflows/ci.yml/badge.svg)](https://github.com/YZJYBT/pyOpAst/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/opast)](https://pypi.org/project/opast/)
[![Python versions](https://img.shields.io/pypi/pyversions/opast)](https://pypi.org/project/opast/)

**opast** ("OPtimizing AST") is a conservative AST-level source optimizer for Python. It rewrites your script into an equivalent but faster one and runs it on the very interpreter that invoked opast (plain CPython by default). Every transformation is backed by a static proof of semantic preservation; anything the optimizer cannot prove, it leaves alone.

- PyPI / import package / CLI command: `opast` — GitHub repository: **pyOpAst**
- Requires Python ≥ 3.10. Zero runtime dependencies; `numba` only for the opt-in `--jit` extra.

```powershell
pip install opast            # or from source: pip install -e .

opast script.py a b c        # optimize, then run (same as python -m opast)
opast --show --no-run script.py     # print the optimized source only
opast --report script.py            # run + per-pass statistics on stderr
opast -c "print(sum(i for i in range(10)))"   # inline code string, like python -c
opast --disable inline,licm script.py         # skip passes by name
```

## Measured results

CPython 3.14.2 (Windows), 21 built-in workloads, best of 5 runs per variant, outputs verified identical — see [Benchmarks](#benchmarks) for the full table and how to reproduce:

| Mode | Geomean speedup |
| --- | --- |
| Default tier — proof-backed passes only, plain CPython output | **2.23x** |
| Aggressive `-O3` — every assumption-backed option, numba jit included | **18.7x** |

A few individual rows: constant `eval`/`getattr` de-dynamization **34.5x**, inline→fold cascades **2.4x** (50x under `-O3`), loop-invariant motion **1.9x**, a "daily-style" report script **1.3x** — and numeric kernels that numba can take whole reach **50–1200x** under `-O3`. The `control` workload has nothing to optimize and gets **0 rewrites**: its run-to-run wobble (0.91x–1.03x across runs) is the harness's noise floor, not an effect.

### It stacks with PyPy

The default tier emits plain Python source with no runtime of its own, so it is not an alternative to a faster interpreter — it composes with one. Running the default-tier output on **PyPy 7.3.21** instead of the original source is a **1.98x** geomean over the same 21 workloads (best of 3, fresh process per run). The wins there come from the rewrites PyPy's tracing JIT cannot do for you — de-dynamization, optimize-time loop folding, comprehension and hoisting rewrites — while a few workloads come out **slower** (worst measured 0.62x), because the JIT sometimes traces the shape you wrote better than the one opast produced. `--jit` is CPython-only (numba), so on PyPy stay on the default tier.

### Self-hosting check

opast optimizes **itself** (`python -m opast.build src/opast -o out/opast`): 841 rewrites across 57 files, zero fallbacks — and the self-optimized copy then produces **byte-identical output** to the original over the whole benchmark corpus and passes the full acceptance suite, the classic compiler bootstrap test. Modules that genuinely use `exec`/`compile` (the runner, the import hook) are skipped whole by the dynamic-code rule, exactly as documented. Honest footnote: the self-optimized optimizer is *not faster* (0.98x, within noise) — opast's own hot paths are method-heavy visitor classes, which is precisely the territory the proof-backed tier leaves alone. Semantics preservation survives its harshest user; the speedups live in code shaped like the benchmarks above.

## Design

Three principles drive every pass:

1. **Prove, then rewrite.** By default every pass fires only on facts established by static analysis — no assumption about your program is required. (Assumption-backed optimization exists, but lives behind the opt-in [`--aggressive`](#aggressive-mode-opt-in---aggressive---o3) tier, which states each assumption it makes.) The facts come from: per-function *proven-int* type inference (a greatest-fixpoint over all bindings), interval (value-range) analysis with widening, escape analysis for *fresh containers* (built locally, never leaked, never mutated), straight-line dominance scans for definite binding, and module-wide stability checks for names (bound exactly once, never `global`-declared, no dynamic constructs anywhere).
2. **Dynamic code disables optimization, scope by scope.** Any appearance of `eval` / `exec` / `globals` / `locals` / `vars` / `compile` / `__import__` / frame-introspection attributes / `from m import *` taints the enclosing scope; tainted scopes (and everything nested inside) are skipped entirely. A tainted module top level disables the whole file. The check is name-based and deliberately over-conservative.
3. **Zero runtime overhead.** The static passes emit ordinary Python source — no guards, no helper runtime. The only runtime machinery lives behind the opt-in `--jit` flag, and it degrades to plain Python on any failure.

## Optimization passes

The pipeline iterates to a fixpoint (default ≤ 8 rounds): de-dynamize → folding → constant propagation → algebraic → loop-fold → cond-narrow → dead code → tail-recursion (needs `tail-calls`) → range-to-iter → LICM → attr-hoist (needs `attrs`) → CSE → unused → inline → scalar-repl → loop-to-comp → comp-to-map → module-loop outlining (needs `loop-state`) → localize. Each pass feeds the next; collapsed constants cascade outward across iterations. See [README-ZH.md](README-ZH.md) for the full safety-condition spec of each pass.

| Pass | What it does |
| --- | --- |
| `de-dynamize` | Expands *pointless* dynamic code into static equivalents: top-level `eval("<const expr>")` → the expression, `exec("<const stmts>")` → the statements, `globals()['x'] = v` → `x = v`, `getattr(o, 'a')` → `o.a`, statement-form `setattr`/`delattr` → attribute syntax. All-or-nothing: commits only if the whole module ends up free of dynamic constructs, otherwise rolls back — un-tainting the module unlocks every other pass. |
| `constant-folding` | Folds constant arithmetic / strings / comparisons / boolean short-circuits / subscripts. **Constant containers** participate: a tuple display of constants becomes one constant tuple, and constant subscripts/slices/`len()`/membership tests on it fold; `f(*(a, b))` flattens to `f(a, b)` and `f(**{'k': v})` to `f(k=v)` — displays evaluate elements left-to-right at the argument's own position, so this is exact even for impure elements, and rejected `**` keys (non-identifier/duplicate/colliding) are left alone to keep the runtime `TypeError`. Expressions that would raise at runtime are preserved; guardrails cap optimization-time work and literal sizes, with partial sub-expression folding allowed. |
| `const-prop` | Substitutes constants for variables along four routes: single-binding names (whole scope), **cross-scope** module constants into later-defined functions, **span propagation** for multiply-bound names (from an assignment up to the first statement that could rebind), and **copy propagation** `y = x` between plain locals (function scopes only). Constant tuples (from folding) propagate like scalars, including through flat unpacks `x, y = t`, feeding subscript/`len`/`*`-expansion folds downstream. |
| `algebraic` | Identity cleanup (`x+0`, `x*1`, `-(-x)`, …) plus strength reduction — `E % 2**k → E & mask`, `E // 2**k → E >> k`, `E ** 2 → E * E` — and interval-analysis-backed `abs(E) → E` for provably non-negative `E`. Proven-`float` expressions get only the **bit-exact** identities (`F - 0`, `F * 1`, `F / 1`, `F ** 1`, `+F`, `-(-F)`); `F + 0` and `F // 1` are deliberately excluded because they change `-0.0` and floor respectively. Never duplicates or drops effects. |
| `loop-fold` | **Closed-form loop evaluation**: a `for i in range(<const>)` loop whose body is pure int arithmetic (no calls) is simulated exactly at optimization time and replaced by its final constant assignments. Step and magnitude budgets; any simulated exception keeps the loop. A successful fold proves the loop cannot raise, so it is legal even inside `try`/`with`. |
| `cond-narrow` | Decides comparisons between proven-`int` expressions whose intervals settle the outcome and replaces them with `True`/`False` for `dead-code` to reap. **Chained comparisons** participate: one provably-false leg falsifies the chain, all-true folds it, provably-true legs peel off the ends (`0 <= i < n` → `i < n`), and a true chain narrows through every leg. Any operand a rewrite stops evaluating must also be provably non-raising (`10 // x < 5 < 0` keeps its chain when `x` can be 0). Interval-decided two-argument `min`/`max` fold in `algebraic` (`min(i, 255)` → `i` when the ranges prove it; the dropped operand must be pure and total). **Guard clauses** count too — when an `if` body always exits, the negated condition holds for everything below it, so `if n <= 0: return` establishes `n >= 1` — `if i < 0:` inside `for i in range(n)` disappears. Combines the flow-insensitive base intervals with **path-sensitive narrowing** (inside `if c:` the condition holds, so `if k > 10: ... if k > 5:` decides the inner test) and straight-line assignment transfer. Sound by construction: any name a statement binds is reset to its base interval afterwards, and entering a loop body resets everything the loop rebinds before applying the test as a fact. Only pure int expressions are folded, so no side effect can be removed. |
| `dead-code` | Unreachable statements after `return`/`raise`/`break`/`continue`, constant-condition `if`/`while` (with `else` semantics), `assert True`, useless constant expression statements, redundant `pass`. |
| `range-to-iter` | `for i in range(len(x))` over a provably fresh *sequence* becomes `for v in x` (index dead) or `for i, v in enumerate(x)` (index live); per-iteration `BINARY_SUBSCR` lookups disappear. Only exact `x[i]` loads are replaced; the enumerate form keeps `x` and `i` bound, so leftovers stay correct. |
| `licm` | Hoists provably pure-and-total loop-invariant int **and float** expressions (plus `len()` of fresh containers) out of loop bodies and `while` tests into pre-loop temporaries, with dominance-checked definite binding. Float admits `+`/`-`/`*` (overflow yields `inf`, never raises) and division by a non-zero constant; `**` is excluded (`(-8.0) ** 0.5` is complex), and the int/float name sets stay separate so int-only forms like `F << 2` cannot slip through. |
| `cse` | Merges repeated provably pure int/float expressions (same criteria as LICM) within a statement block into a temporary; speculative evaluation is sound because the expressions cannot raise. |
| `tail-recursion` | **Opt-in via `--aggressive=tail-calls`.** A self-call that *is* the whole return value becomes a parameter rebind plus a jump, measured **4.6x** faster than the recursion (**2.6x** with the depth counter that keeps `RecursionError` alive). Partial elimination is supported — in divide-and-conquer code only the tail call becomes a jump and the other recursion stays, which is the textbook way to halve quicksort's stack. Rejected inside `try`/`with` (`finally` runs per frame, not per iteration), inside nested loops, for functions whose parameters a closure captures, and whenever a non-parameter local could be read before the iteration assigns it — a reused frame would hand back the previous iteration's value where a fresh frame raised `UnboundLocalError`. |
| `unused` | Removes unused imports, unused module-level functions (post-inlining helpers), and dead local stores (effectful right-hand sides are downgraded to bare expressions). Module-level variable assignments are deliberately kept — module globals are observable API. |
| `inline` | Two shapes: expression-body functions (`def f(...): return <expr>`) inline at any call site with constant/name arguments; straight-line statement bodies (≤ 6 assignments + `return`) inline with arbitrary arguments — **positional or keyword, evaluated in call-site written order** — at statement positions, in `if` tests, and **buried inside expressions** (`acc = (x + f(y)) * 2`): subtrees evaluated before the call become their own temps so effects and exceptions keep their exact positions, while short-circuit and conditional positions are never touched. Requires module-wide name stability; the leftover `def` is cleaned by `unused`. |
| `scalar-repl` | **Scalar replacement of small local aggregates.** A tuple/list local whose every use is a constant, in-bounds subscript (or `len()`) never needs to exist: each element becomes its own local (measured **1.3x**; the allocation and every `BINARY_SUBSCR` disappear, and const-prop/CSE/unused then treat the elements as plain scalars). Same for instances of a *simple class* — module-level, no bases/decorators, body is a single plain `__init__` of `self.attr = expr` — used only through `p.attr` (measured **2.7x**). Any escape rejects: a bare read, `del`, slice, non-constant or out-of-bounds index, unknown attribute, iteration, or any use in a nested scope, so `IndexError`/`AttributeError`/`TypeError` (`t[0] = 1` on a tuple) all survive; the binding must **dominate** every use (only statements after it in its own block qualify — a conditional binding, a prior-loop-position read or an `except`/`finally` use rejects, preserving `UnboundLocalError`); element/argument expressions are re-emitted in evaluation order, so effects keep their positions. One name binding does not freeze the class *object* — `P.__init__ = fn` monkey-patches without rebinding `P` — so every Load of the class name module-wide must be the func of a call: any attribute store/read, alias, argument position or base-class use disqualifies the class. Free names in `__init__` must not be shadowed at the call site, calling before the class statement keeps its `NameError`, and generated temps avoid every identifier in the function. Element count is budgeted (8; `--aggressive=budgets` raises it). |
| `loop-to-comp` | `x = []` + an adjacent append-accumulation loop (nested `for`/guard-`if` chains allowed) becomes a list comprehension (`set()`/`.add` → set comprehension): dedicated `LIST_APPEND`/`SET_ADD` bytecode instead of a per-iteration attribute lookup + method call. Rejected inside `try`/`with` (a mid-loop exception would expose the partially-built list). **Module-level rewriting needs `--aggressive=loop-state`** — module globals outlive an escaping exception, so an outer `try` can tell a half-filled list from an assignment that never happened; inside a function that state dies with the frame. |
| `comp-to-map` | `(f(x) for x in it)` → `map(f, it)`, with `filter` for guards, restricted to positions where the generator/`map` identity difference is unobservable and `f` is provably stable. |
| `localize` | Per-iteration reads of stable globals/builtins inside loops become pre-loop locals (`LOAD_GLOBAL` → `LOAD_FAST`). Runs last so other passes claim names first. |

LICM/CSE additionally support **`len()` caching for fresh containers** — sound only because escape analysis guarantees no reference ever leaves the function, so no call can mutate the container.

## Aggressive mode (opt-in): `--aggressive` / `-O3`

Everything above is proof-backed: no assumption about your program is
required. Aggressive mode is the second tier — each option buys extra
optimization by resting on **one stated assumption**, and `--report` prints
exactly which bets are in play.

```powershell
opast --aggressive script.py                  # everything below
opast -O3 script.py                           # same thing
opast --aggressive=all script.py              # explicit spelling of the same
opast --aggressive=annotations script.py      # pick a subset
opast --aggressive --disable jit script.py    # all but one
opast -O2 script.py                           # aggressive, third-party-free
opast --aggressive=stdlib,numpy script.py     # stdlib group plus one back
```

`-O2` (`--aggressive=stdlib`) is the same tier with the two options whose
output leans on a third-party package left out — `jit` (numba) and `numpy`.
Both of those degrade to plain Python when the package is missing, but the
emitted source still mentions it; `-O2` keeps the optimized code runnable on
a bare interpreter, which is what you want for freezing, for shipping a
single file, or on a machine where neither package is installed. `stdlib` is
a *group alias* usable anywhere an option name is, so it composes:
`--aggressive=stdlib,numpy` adds one back.

| Option | What it buys | What it assumes |
| --- | --- | --- |
| `annotations` | Parameters and call results annotated `int`/`float` become typed, which unblocks strength reduction, LICM/CSE hoisting and interval narrowing across the function boundary | Annotations reflect runtime types; no `bool` where `int` is annotated (identities only) |
| `attrs` | **Attribute hoisting**: loop-invariant attribute chains move to pre-loop locals — dotted module calls (`math.floor(i)`) measured **1.77x**, attribute value reads 1.24x. Instance *method calls* are deliberately not touched: bound-method caching measured 0.90x on 3.14, slower than the interpreter's specialized call path. Writes to any prefix of a chain inside the loop cancel its hoists | Attributes read inside loops are stable: no property/`__getattr__` side effects, nothing rebinds them mid-loop; a zero-iteration loop may now perform the lookup once |
| `budgets` | Raises every "is this worth it" limit: loop-fold simulates up to 20M steps instead of 200k (about a second of optimization time), constants may fold to far larger literals, statement-body inlining accepts 24 assignments instead of 6, scalar replacement accepts 64 elements/attributes instead of 8 | **Nothing semantic** — only that longer optimization and bigger output are acceptable |
| `fastmath` | Float rewrites that are not bit-exact: `F + 0` → `F`, `F ** 2` → `F * F` (~3x cheaper per operation), `F / c` → `F * (1/c)` for any constant | `-0.0` may become `0.0`; overflow may yield `inf` where `**` raised `OverflowError`; reciprocal multiplication may differ in the last ulp |
| `jit` | The numba path described below | numba's int64 arithmetic does not wrap for your values |
| `loop-state` | **Module-level loop outlining** (below), plus module-level `loop-to-comp` | When an exception escapes a module-level loop, the module's globals may hold their pre-loop values instead of partial results |
| `module-locals` | Refines `loop-state`: loop counters and temporaries need no write-back, so loops with unknown trip counts become outlinable | Module-level names that nothing in the module reads are private temporaries, not API |
| `numpy` | **numpy vectorization**: after the fixpoint, comprehension maps (`acc = [f(v) for v in xs]`) and `+=` reductions over proven-fresh lists/tuples or `range(...)` are extracted into generated helper pairs — the exact original Python plus a numpy version — joined by a runtime dispatcher: the first call runs both and compares, and any mismatch, exception, unsupported dtype (heterogeneous lists included) or absent numpy permanently selects the Python path for that site; a zero divisor re-raises the exact original exception. Results return through `.tolist()`/`.item()`, so element types stay exact Python ints/floats. Measured (best-of-7): range reductions **8.8–23.8x**, range maps ~1.5x, list maps/reductions with heavy (≥4-op) elements ~2–3x; light list elements measured 0.61–0.92x and are statically skipped | Numeric loop intermediates become fixed-width int64/float64 (large ints wrap instead of growing), a float reduction is reassociated into one partial sum, and numpy must be importable for the fast path (plain Python otherwise) |
| `opt-imports` | The import hook described above | Optimized imported modules are not monkeypatched |
| `pure-calls` | **Trusted-pure module functions**: calls to your own helper functions become LICM/CSE/unused material — a call with loop-invariant arguments moves out of the loop, duplicate calls compute once, a call whose result is dead disappears, and inlining/attr-hoist cascade with all three. Trust is evidence-based and decided once on the pristine module: top-level `def`, name bound exactly once (no `f = wrap(f)`, no `f.attr = ...`), immutable defaults, and a body with no nested scope, `yield`/`await`, `global`, `import`, `with`, attribute access, or subscript write, reading nothing but its own locals, other trusted functions, whitelisted pure builtins, and immutable module constants (mutual recursion converges via a fixpoint) | Trusted functions are pure and never rebound at runtime: their calls may run at different times or not at all (exceptions and non-termination inside them may move or vanish), same-argument calls may share one result object, and argument objects are not mutated across the optimized region |
| `slots` | **`__slots__` injection** for module-level base-less classes: every instance drops its `__dict__` (measured **1.2x smaller** at 4 attributes — more with fewer; attribute speed is ~1.0x on 3.14, whose inline caches already make dict access fast — the win is memory, plus speed on older interpreters). `'__weakref__'` is always included so `weakref.ref` keeps working. Proven even under the assumption: no decorators/bases/metaclass/existing `__slots__`/`__setattr__` family, no unknown *method* decorators (`functools.cached_property` stores through the instance `__dict__` invisibly), no class-variable collision, `setattr`/`delattr` nowhere in the module, a module-visible extra attribute (`p.tag = 1`) rejects the class, and the class *object* itself may only appear as `C(...)`, a gated `isinstance`/`issubclass` argument, or a single base — `C.x = 5` (directly or via an alias) would overwrite the injected slot descriptor. Runs once on the pristine module before the fixpoint loop, so rejections cannot be undone by passes erasing their evidence | Instances never receive attributes beyond those the class's own methods assign, and nothing relies on instance `__dict__` (`vars(obj)`, protocol-0/1 pickle) |
| `tail-calls` | **Tail-recursion elimination** (2.6x with the counter) | `RecursionError` is approximated by a depth counter: it fires at a somewhat different depth, and follows `sys.getrecursionlimit()` rather than the interpreter's own stack |
| `unbounded-recursion` | Refines `tail-calls`: drops the counter (4.6x, and deep tail recursion simply works) | Tail recursion may run past the recursion limit instead of raising `RecursionError` |

### Module-level loop outlining (`loop-state`)

Module-level code compiles to `LOAD_NAME`/`STORE_NAME` — a chain of dict
lookups — while the same code inside a function uses `LOAD_FAST`/`STORE_FAST`
array slots. Measured on CPython 3.14, a top-level accumulator loop runs
**about twice as fast** once moved into a function, which is why "put it in
a function" is folklore advice. Scripts, opast's main target, routinely do
all their work at module level, so this is the widest-reaching aggressive
option:

```python
total = 0                          def _opast_outline_0(total):
for i in range(1_000_000):             for i in range(1_000_000):
    total = total + i * 2      ->          total = total + i * 2
                                       return (total, i)
                                   total = 0
                                   (total, i) = _opast_outline_0(0)
```

The analysis is a per-iteration store-before-load scan: names written before
being read are loop-local, names read while unwritten become parameters and
must be definitely bound before the loop. A stored name that **any nested
scope reads** rejects the outline outright — a function called from the body
would otherwise observe the stale pre-loop value. Loops under `try`/`with`,
or containing `def`/`class`/`import`/`global`/`yield`, are rejected too.

The write-back happens once after the loop, which is exactly the assumption
the option names: if an exception escapes, the globals keep their pre-loop
values. That is invisible to a script that dies on the exception, and
visible to anyone who wrapped the module in a `try` — hence opt-in.

Note that `F / c` **is** rewritten by default when `c` is a power of two —
the reciprocal is then exact, so `x / 4.0` → `x * 0.25` is bit-identical
for every input and needs no assumption. `fastmath` only adds the divisors
whose reciprocal rounds.

**Why `annotations` matters most**: a parameter is normally *never* provably
typed — a caller may pass `Decimal`, a numpy array or a subclass — so the
type-driven passes stop at the function boundary. Annotations lift that:

```python
def integrate(steps: int, dt: float) -> float:      # default: nothing fires
    x, v, g = 0.0, 1.5, 9.81
    for i in range(steps):
        a = g * dt * dt
        v = v + g * dt
        x = x + v * dt + a
    return x
```
```python
def integrate(steps: int, dt: float) -> float:      # --aggressive=annotations
    x, v = 0.0, 1.5
    _opast_cse_0 = 9.81 * dt                        # hoisted out of the loop
    _opast_inv_0 = _opast_cse_0 * dt
    for i in range(steps):
        v = v + _opast_cse_0
        x = x + v * dt + _opast_inv_0
    return x
```

Two conventions make annotations looser than the analysis needs, and both are
handled rather than assumed away: `bool` is a subclass of `int` (so it only
affects the algebraic identities, never hoisting or narrowing), and PEP 484
explicitly allows an `int` argument where `float` is annotated (so the
`F / 1` identity is switched off whenever annotations are trusted). Only bare
`int`/`float` annotations count — `Optional[int]`, `int | None`, `list[int]`
and friends are ignored, as are `*args`/`**kwargs` annotations (they describe
the elements, not the parameter). Return annotations are trusted only for
module-level functions bound exactly once and never declared `global`, and
they only *type* a value: a call is still never hoisted, because purity is a
separate question an annotation says nothing about.

## Importing modules through the optimizer (`--opt-imports`)

```powershell
opast --opt-imports script.py            # also optimize modules imported from the script's directory
opast --opt-imports-under src script.py  # add extra roots (repeatable)
```

A `sys.meta_path` hook runs imported modules through the full pipeline before compilation. Optimized bytecode never touches `__pycache__` (a later plain `python` run must not pick it up); results go to a content-addressed private cache instead (`OPAST_NO_IMPORT_CACHE=1` bypasses). Semantics boundary: modules whose globals get monkeypatched from outside (e.g. `unittest.mock.patch`) silently lose the stability assumptions several passes rely on — do not enable this for such modules, which is why it is opt-in. `unused` is force-disabled for imported modules (their "unused" definitions are the export surface). Python API: `from opast.importhook import install, uninstall`.

## Benchmarks

```powershell
python -m opast.bench            # all built-in workloads, both modes, best-of-3, results verified identical
python -m opast.bench --mode default daily     # one tier only (detailed table: opt-cost, changes)
python -m opast.bench --mode aggressive -r 5
python -m opast.bench --list
```

21 built-in workloads, each measured in **two modes** — default (proof-backed passes only) and aggressive (`-O3`: every assumption-backed option, jit included; numba recommended, the warmup run absorbs import + compile cost) — in the same interpreter with GC disabled and a `RESULT` equality check per variant. One combined table with both speedups and per-mode geomeans; several workloads (`annotated`, `fastmath`, `tailrec`, `jitlazy`) are ~1.0x in the default tier by design and show their real numbers in the aggressive column. Note that CPython's own compiler already does trivial constant folding — opast's wins come from what CPython does *not* do: inlining, type-proven algebraic rewrites, loop rewrites, de-dynamization.

Full run on CPython 3.14.2 (Windows, best of 5, `python -m opast.bench -r 5`):

```text
workload       original    default  speedup  aggressive  speedup  changes  result
---------------------------------------------------------------------------------
algebra         299.4ms    136.8ms    2.19x      10.8ms   35.60x      8/9  OK
annotated       119.8ms    122.6ms    0.98x       0.9ms  134.17x     8/21  OK
attrhoist        46.2ms     35.2ms    1.32x      34.3ms    1.31x     9/18  OK
comptomap       420.1ms    179.3ms    2.34x     181.6ms    2.22x     11/8  OK
condnarrow       80.2ms     50.1ms    1.60x       1.7ms   51.32x      7/8  OK
control         113.0ms    124.0ms    0.91x      81.1ms    1.53x      0/2  OK
daily           178.1ms    136.5ms    1.30x     145.0ms    1.28x    47/48  OK
dedynamize      817.3ms     23.7ms   34.52x      10.0ms   65.39x      2/5  OK
fastmath         79.8ms     76.5ms    1.04x       1.0ms   71.01x      0/5  OK
inline          615.7ms    437.5ms    1.41x      12.0ms   50.05x    10/13  OK
inlinestmt      156.8ms    125.5ms    1.25x       4.0ms   38.33x      3/4  OK
jitlazy         241.4ms    236.4ms    1.02x       0.2ms 1035.42x      1/2  OK
lencache         68.4ms     58.9ms    1.16x      54.0ms    1.00x      2/3  OK
licm            112.7ms     60.0ms    1.88x      61.9ms    1.92x      6/7  OK
localize        203.7ms    194.4ms    1.05x     201.4ms    1.02x      4/2  OK
loopfold        230.9ms      0.1ms 3761.35x       0.0ms 17574.18x    21/38  OK
looptocomp       59.4ms     57.1ms    1.04x      52.2ms    1.10x      4/5  OK
mixed           243.6ms    102.4ms    2.38x       9.1ms   25.16x    19/29  OK
rangeiter        80.1ms     76.0ms    1.05x      59.6ms    1.32x      2/3  OK
strength        405.3ms    340.1ms    1.19x      14.4ms   26.56x      7/8  OK
tailrec         226.3ms    227.7ms    0.99x       0.2ms 1209.97x      1/4  OK
---------------------------------------------------------------------------------
geomean: default 2.23x | aggressive 18.72x
```

Reading the table honestly:

- **`loopfold` (thousands of x) is a constructed extreme**, not a typical win: the whole computation happens *at optimization time* (closed-form loop evaluation) and the run-time script is just constant assignments. The real accounting is "pay ~0.13s once at optimize time, save ~0.23s every run". `dedynamize` (34.5x) is similar — a hot loop calling constant `eval`/`getattr` collapses to static code.
- **Middle rows are the representative ones**: `mixed` 2.38x, `comptomap` 2.34x, `algebra` 2.19x, `licm` 1.88x, `condnarrow` 1.60x, `daily` (a realistic order-settlement report script touching many passes at once) 1.30x.
- **Aggressive triple-digit rows are numba doing what numba does** — the interesting part is that opast's rewrites make the functions *jittable* (de-dynamized, inlined, typed) without you restructuring anything, and any numba failure falls back to plain Python at runtime.
- Sub-100ms rows carry a few percent of run-to-run noise (`control` has bounced between 0.92x and 1.09x across runs); treat ±0.05x as measurement jitter. Numbers are from one machine — run the command above on yours.

## Batch build (PyInstaller & co)

```powershell
python -m opast.build src/ -o build_opt/ --entry main.py   # optimize a whole tree
pyinstaller build_opt/main.py                              # freeze the optimized tree
python -m opast.build src/ -o build_opt/ -O2              # aggressive minus numba/numpy
```

Optimizes every `.py` under a directory into a mirrored output tree (other files copied verbatim; `__pycache__`/VCS dirs skipped, `--exclude` adds more). A file the optimizer cannot process is copied unchanged with a warning — the build never breaks (`--strict` makes fallbacks fatal instead). Directory builds treat every file as a **library module** — its whole top level is public API, so unused-elimination and the aggressive `module-locals` option stay off (the import-hook contract, which now enforces the same); mark entry scripts with `--entry` for full script-mode cleanup. The default tier emits plain Python with **no runtime dependency on opast**, which is what makes the output suitable for freezing; so does `-O2`, the aggressive tier minus the numba/numpy options. `--jit` output imports `opast.jitsupport` at runtime — bundle opast and numba if you freeze it. Comments and formatting are not preserved (a leading `#!` shebang is); line numbers shift.

## IPython / Jupyter

```
%load_ext opast

%%opast --report --disable licm
total = 0
for i in range(50_000):
    total += i * 2
total
```

Options mirror the CLI, `-O3`/`-O2` included (`%%opast -O2` is the aggressive tier without the numba/numpy options — handy in a kernel that has neither). The cell executes in the user namespace, so assignments persist. Analyses are cell-scoped — see README-ZH for the notebook caveats.

## Experimental: `--jit` (numba, off by default; part of `--aggressive`)

```powershell
pip install opast[jit]
opast --jit hot_script.py
```

After the static fixpoint, a one-shot pass decorates hot numeric functions with a guarded `numba.njit` wrapper. Measured on CPython 3.14 (8M-iteration numeric kernel): 1.22 s pure Python vs 0.011 s steady-state (~110×), first call 0.79 s including compilation.

- **Static hotness** (constant loop bounds ≥ 10 000 or nested loops) compiles at decoration time; a strict whitelist predicts numba compatibility (int/float arithmetic, `range` loops, `math.*`, tuples with constant-index reads and `len`, proven single-binding numeric module constants — numba freezes globals, and the single binding makes the frozen value exact; no strings, no lists/dicts/sets — reflected containers copy per call and are deprecated). With a single-binding `import numpy` in the module, candidates may also call whitelisted `np.*` functions, read subscripts with variable indices and slices, read `.size`/`.shape`/`.ndim`, `raise` builtin exceptions with constant messages, and subscript-store into locals provably created by `np.zeros`/`ones`/`empty_like`/… — array-heavy kernels compile whole. Under the aggressive `annotations` option, parameters annotated `np.ndarray`/`npt.NDArray` (module-level aliases included) count as proven arrays too, unlocking in-place algorithms (sorts, partitions); such argument-mutating candidates skip the first-call comparison — re-running an in-place function on the same arguments is itself unsound — and rely on the whitelist plus numba's typing. Lazy candidates may now call each other: peer aliases are backfilled at trigger time, so a variable-bound merge kernel calling `_gallop_*` helpers compiles as a unit (measured on a 200k-element sorting benchmark: merge 5.3x, three-way-partition quicksort 17.8x steady-state).

**`DynArray`** (`from opast import DynArray`) is a plain `list` subclass — outside a jitted function, and in any interpreter that never runs opast, it *is* a list. Using it states a contract no analysis can prove for you: *the elements are homogeneous numbers, nothing depends on the container's identity or concrete type, and it does not leave the function that built it.* Inside a compiled function that lets opast pick the representation numba is fastest with, in the compiled copy only — the Python fallback keeps running the list, so a failed compilation or a numba-free interpreter stays exactly correct:

| You write | Compiled as | Measured |
| --- | --- | --- |
| `DynArray()` + `append`/`pop` | a numba list (`[]`) — dynamic capacity, which arrays cannot provide | monotonic stack over 400k values **32x** |
| `DynArray.zeros(n)` / `.full(n, v)` | `np.zeros(n)` / `np.full(n, v)` when the module imports numpy, else a list | 1M-update histogram **37x** |

`dtype` is explicit (`zeros(n)` → `0.0`/float64, `zeros(n, dtype=int)` → `0`/int64) so both representations agree element for element. The optimizer verifies the mechanical half of the contract — bound exactly once, never returned, never passed on, never aliased, growth methods only on the growable form, and truth tests rewritten to `len(x) > 0` in the compiled copy — and rejects the whole function if anything else touches the container. A container that is created and discarded per iteration is not worth it (a short-lived stack rebuilt 30k times measured 1.07x: allocation dominates); keep it alive across the hot loop. Under `--jit` a `numpy`/`math` import is kept even when nothing else reads it, since the rewrite above is what will read it.
- **Loop outlining** extracts hot whitelisted loops out of mixed functions and module top level into fresh compiled functions, with proven input/output sets.
- **njit inter-calls**: candidate functions may call each other (fixpoint selection, call cycles dropped, compiled copies call raw dispatchers).
- **Runtime lazy compilation** covers *variable* loop bounds (`for i in range(n)`): the wrapper observes plain-Python calls and compiles when a trigger fires — bound argument ≥ `OPAST_JIT_LAZY_BOUND` (default 10 000), a single call ≥ 0.1 s, or ≥ 10 calls totalling ≥ 0.3 s. The triggering call's Python result doubles as the verification expectation, and `numba` itself is imported only on the first compilation attempt — a script whose lazy candidates never get hot pays nothing.
- **First-call verification**: whitelisted functions are pure, so the first call runs both versions and compares results; a divergence (in practice: int64 wraparound, which no static filter can rule out) triggers a permanent fallback to Python instead of silently wrong answers. `OPAST_JIT_NO_VERIFY=1` opts out.
- **Layered degradation**: no numba / incompatible interpreter / `OPAST_DISABLE_JIT` → original function; any numba error at compile or call time → permanent Python fallback. `OPAST_JIT_DEBUG=1` explains every fallback and lazy trigger on stderr.

⚠️ Opt-in semantic caveat: numba integers are fixed 64-bit — intermediate values beyond ±9.2e18 wrap silently. This is why `--jit` is not on by default and not part of the semantic-preservation contract; the verification above is a safety net, not a proof.

## Experimental: `--cython` (correct first, fast where it can prove)

```powershell
opast -O2 --cython -o build/hot.py hot.py
cythonize -i build/hot.py                  # your own build step
```

`--cython` emits Cython pure-Python-mode annotations into the optimized source: `@cython.locals(...)` where a C type is **proven**, and `@cython.infer_types(False)` on every function. The output is still plain Python — a guarded import shim keeps it running on an interpreter with no Cython installed — so this only matters if you go on to compile it.

It started as a correctness fix, and that half still matters most. **Stock `cythonize` is not semantics-preserving**, and two of its deviations are reproducible in this repo's own benchmark suite:

| Plain, correct Python | Stock Cython | opast `--cython` |
| --- | --- | --- |
| `[j * j % 97 for j in range(0, 400000, 4)]` → last element `54` | **`20`** — it infers a C `long` for `j`, which is 32 bits under MSVC, and `399996²` wraps | `54` |
| reading a conditionally-bound local raises `UnboundLocalError` | **returns `0`** — a C variable has no unbound state | raises |

Both come from Cython's own "safe" type inference, which has no interval analysis to consult; opast's does (`j * j` is provably in `(0, 159996800016)`, so it needs 64 bits, so `j` must not be a 32-bit `long`). Turning that inference off is what restores the semantics, and the types opast can prove go back in explicitly.

Speed, on the 21 benchmark workloads with both sides compiled by Cython: **1.29x geomean, 1.05x median** (best-of-15). That average is not a broad win — it is two large ones. Coverage is still narrow — 4 of the 36 functions in the benchmark workloads get typed at all — so most rows are ~1.0x, and the payoff lands where a hot function can be typed end to end:

| | `-O2` + stock cythonize | `--cython` | |
| --- | --- | --- | --- |
| `annotated` | 22.7 ms | **1.4 ms** | **16.6x** |
| `strength` | 163.4 ms | **11.1 ms** | **14.8x** |

(best-of-13 interleaved, identical results, and the same ratios by median — 17.0x and 11.7x — so this is not a min-of-N artifact.)

Treat the *small* numbers with suspicion, though: the unit here is one module import in one process, because a CPython extension cannot be re-executed in-process, and process spawn on Windows carries first-touch scanner spikes that best-of-N does not fully suppress. Rows near 1.0x have moved by ±0.4x between runs; two rows the full run reported as 0.71x and 0.72x measured 1.04x and 1.01x when re-run on their own. Only differences of the order above are worth reading.

What does not depend on that timing at all:

- Where nothing gets typed, `--cython` output differs from plain `-O2` **only** by the directive, so any loss there is the price of correctness by construction. A four-variant comparison of one workload prices the halves separately: the directive costs **0.74–0.80x**.
- Partial typing is worse than none — a typed value meeting an untyped operand is boxed back into a Python object at every boundary (0.72x on a loop whose counter was typed and whose accumulator was not). Hence the all-or-nothing rule below, and hence the narrow coverage: opast declines far more often than it types.
- *Which* functions qualify depends on the option set, because the pass runs last and reads whatever shape the earlier passes left. The same 21 workloads yield 4 typed functions under `-O2` and 4 under a bare `--aggressive=annotations`, but not the same 4 (`fastmath.work` in the first, `tailrec.work` in the second); with no aggressive options at all it is 2.
- The worst row of every run is the miscompiled comprehension (0.49x–0.61x) — the wrong answer really was about twice as fast as the right one.

What a name must satisfy to be typed: proven float (a Python float *is* a C double), or proven int with an interval inside int64; definitely bound before every read (one binding site must dominate every use — a name assigned in both arms of an `if` does not qualify); not captured by a nested scope or comprehension; not `del`eted, `global`/`nonlocal`, or compared with `is`; and every operand it meets in arithmetic must be typed too. The gap to the ceiling is the interval analysis, not the mechanism — hand-typing all five locals of a loop opast declines to type measured **11.35x**.

### Annotated integers: checked mode

An `int` annotation says "integer", never "small", so no interval can be derived from it — which is why, on its own, a declared name is *not* typed. Under `--aggressive=annotations` it gets a second route: when a believed `int` has no derivable interval, its function switches to **checked mode**, where every proven int in it is typed whether or not it is bounded, and the function carries `@cython.overflowcheck(True)`.

Both parameters and locals are believed, `x: int = compute()` included — but a local only when the declaration is that name's **only** binding, since a name rebound afterwards (a loop target, a second assignment) may hold something the annotation never described and nothing re-checks it at runtime. The belief has to reach the fixpoint rather than just the declared name: `acc = (acc + i * x) % M` is provably int only once `x` is, so one opaque-but-declared local would otherwise drag the whole function down through the all-or-nothing rule.

```python
def b(): x: int = math.floor(2.5)   # believed -> checked mode, x/acc/i all typed
def c(): x: float = math.sqrt(2.0)  # believed -> typed double, no check needed
def d(): x: int = 5                 # provably bounded -> typed, no check needed
def e(): x: int = f(); …; for x in r: …   # not the only binding -> nothing typed
```

That directive is the whole reason this is offered at all, because the two ways a C integer can go wrong are not equally loud. Converting an argument that does not fit int64 already raises `OverflowError` — Cython checks that boundary. Arithmetic *between* C integers does not: `4e9 * 4e9` in a typed function measured `-2446744073709551616`. `overflowcheck` makes that raise too, at a cost of about 0.79x. So the bet reads: **these values fit in 64 bits, and if they do not you get an exception rather than a wrong number** — a strictly louder contract than `--jit` offers for the same arithmetic, where int64 wraparound is silent.

Measured on one annotated function (3M iterations, best-of-11, both sides Cython-compiled):

```
def hot(rounds: int, seed: int) -> int:      -O2 + stock cythonize   310.07 ms
    total = 0                                --cython + annotations   34.47 ms   9.00x
    for i in range(rounds):
        total += (seed + i) * (seed + i) % 1000003
    return total
```

with `hot(3_000_000, 7)` byte-identical to pure Python, and `seed=4e9` — which Python computes via bignums — raising `OverflowError` instead.

## Python API

```python
from opast import optimize_source, optimize_file, run_path, run_source, PASS_NAMES

result = optimize_file("script.py")
print(result.source)             # optimized source (ast.unparse)
print(result.report.summary())   # per-pass statistics
run_path("script.py", argv=("--flag",))

optimize_file("script.py", disable="inline,licm")   # every entry point accepts `disable`
```

## Tests

```powershell
python tests/verify_opast.py     # full acceptance suite: per-pass behavior assertions
                                 # + original-vs-optimized output comparison
```

Case sources are generated into `tests/cases/` on demand (not tracked); the bench harness doubles as a correctness check.

## Prior art

- [pyastop](https://github.com/xiaonanln/pyastop) (2017–2018): an early AST-optimizer prototype built around whole-project analysis and comment hints. opast instead proves each rewrite safe per pass, with no user annotations.
- [fatoptimizer](https://github.com/vstinner/fatoptimizer) (FAT Python, PEPs 509/510/511): runtime guards + function specialization, abandoned over guard overhead. opast's static passes have zero runtime overhead; runtime machinery exists only behind the opt-in `--jit`.
- CPython's built-in AST/peephole optimizer folds constants only; opast's gains come from everything beyond that.

## Known limitations

- Optimized code is executed via `compile(optimized_ast, original_filename)`; traceback line numbers reuse original positions where possible but may drift slightly on rewritten lines. Inlining removes call frames from tracebacks (documented per pass).
- All transformations require static proof; anything unprovable is left untouched. Documented edge observables (e.g. the partially-built-list window excluded via `try`/`with` checks) are listed per pass in [README-ZH.md](README-ZH.md).
- Python ≥ 3.10.

## License

MIT
