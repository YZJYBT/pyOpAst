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

## Design

Three principles drive every pass:

1. **Prove, then rewrite.** By default every pass fires only on facts established by static analysis — no assumption about your program is required. (Assumption-backed optimization exists, but lives behind the opt-in [`--aggressive`](#aggressive-mode-opt-in---aggressive---o3) tier, which states each assumption it makes.) The facts come from: per-function *proven-int* type inference (a greatest-fixpoint over all bindings), interval (value-range) analysis with widening, escape analysis for *fresh containers* (built locally, never leaked, never mutated), straight-line dominance scans for definite binding, and module-wide stability checks for names (bound exactly once, never `global`-declared, no dynamic constructs anywhere).
2. **Dynamic code disables optimization, scope by scope.** Any appearance of `eval` / `exec` / `globals` / `locals` / `vars` / `compile` / `__import__` / frame-introspection attributes / `from m import *` taints the enclosing scope; tainted scopes (and everything nested inside) are skipped entirely. A tainted module top level disables the whole file. The check is name-based and deliberately over-conservative.
3. **Zero runtime overhead.** The static passes emit ordinary Python source — no guards, no helper runtime. The only runtime machinery lives behind the opt-in `--jit` flag, and it degrades to plain Python on any failure.

## Optimization passes

The pipeline iterates to a fixpoint (default ≤ 8 rounds): de-dynamize → folding → constant propagation → algebraic → loop-fold → cond-narrow → dead code → tail-recursion (needs `tail-calls`) → range-to-iter → LICM → CSE → unused → inline → loop-to-comp → comp-to-map → module-loop outlining (needs `loop-state`) → localize. Each pass feeds the next; collapsed constants cascade outward across iterations. See [README-ZH.md](README-ZH.md) for the full safety-condition spec of each pass.

| Pass | What it does |
| --- | --- |
| `de-dynamize` | Expands *pointless* dynamic code into static equivalents: top-level `eval("<const expr>")` → the expression, `exec("<const stmts>")` → the statements, `globals()['x'] = v` → `x = v`, `getattr(o, 'a')` → `o.a`, statement-form `setattr`/`delattr` → attribute syntax. All-or-nothing: commits only if the whole module ends up free of dynamic constructs, otherwise rolls back — un-tainting the module unlocks every other pass. |
| `constant-folding` | Folds constant arithmetic / strings / comparisons / boolean short-circuits / subscripts. Expressions that would raise at runtime are preserved; guardrails cap optimization-time work and literal sizes, with partial sub-expression folding allowed. |
| `const-prop` | Substitutes constants for variables along four routes: single-binding names (whole scope), **cross-scope** module constants into later-defined functions, **span propagation** for multiply-bound names (from an assignment up to the first statement that could rebind), and **copy propagation** `y = x` between plain locals (function scopes only). |
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
```

| Option | What it buys | What it assumes |
| --- | --- | --- |
| `annotations` | Parameters and call results annotated `int`/`float` become typed, which unblocks strength reduction, LICM/CSE hoisting and interval narrowing across the function boundary | Annotations reflect runtime types; no `bool` where `int` is annotated (identities only) |
| `attrs` | **Attribute hoisting**: loop-invariant attribute chains move to pre-loop locals — dotted module calls (`math.floor(i)`) measured **1.77x**, attribute value reads 1.24x. Instance *method calls* are deliberately not touched: bound-method caching measured 0.90x on 3.14, slower than the interpreter's specialized call path. Writes to any prefix of a chain inside the loop cancel its hoists | Attributes read inside loops are stable: no property/`__getattr__` side effects, nothing rebinds them mid-loop; a zero-iteration loop may now perform the lookup once |
| `budgets` | Raises every "is this worth it" limit: loop-fold simulates up to 20M steps instead of 200k (about a second of optimization time), constants may fold to far larger literals, statement-body inlining accepts 24 assignments instead of 6 | **Nothing semantic** — only that longer optimization and bigger output are acceptable |
| `fastmath` | Float rewrites that are not bit-exact: `F + 0` → `F`, `F ** 2` → `F * F` (~3x cheaper per operation), `F / c` → `F * (1/c)` for any constant | `-0.0` may become `0.0`; overflow may yield `inf` where `**` raised `OverflowError`; reciprocal multiplication may differ in the last ulp |
| `jit` | The numba path described below | numba's int64 arithmetic does not wrap for your values |
| `loop-state` | **Module-level loop outlining** (below), plus module-level `loop-to-comp` | When an exception escapes a module-level loop, the module's globals may hold their pre-loop values instead of partial results |
| `module-locals` | Refines `loop-state`: loop counters and temporaries need no write-back, so loops with unknown trip counts become outlinable | Module-level names that nothing in the module reads are private temporaries, not API |
| `opt-imports` | The import hook described above | Optimized imported modules are not monkeypatched |
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
python -m opast.bench            # all built-in workloads, best-of-3, results verified identical
python -m opast.bench --jit daily
python -m opast.bench --list
```

21 built-in workloads, each executed twice per measurement (original vs optimized) in the same interpreter with GC disabled and a `RESULT` equality check. Several (`annotated`, `fastmath`, `tailrec`, `attrhoist`, `jitlazy`) are ~1.0x by default and only meaningful with the matching `--aggressive` option or `--jit`; `python -m opast.bench --aggressive ...` passes the options through. Note that CPython's own compiler already does trivial constant folding — opast's wins come from what CPython does *not* do: inlining, type-proven algebraic rewrites, loop rewrites, de-dynamization.

## IPython / Jupyter

```
%load_ext opast

%%opast --report --disable licm
total = 0
for i in range(50_000):
    total += i * 2
total
```

Options mirror the CLI; the cell executes in the user namespace, so assignments persist. Analyses are cell-scoped — see README-ZH for the notebook caveats.

## Experimental: `--jit` (numba, off by default; part of `--aggressive`)

```powershell
pip install opast[jit]
opast --jit hot_script.py
```

After the static fixpoint, a one-shot pass decorates hot numeric functions with a guarded `numba.njit` wrapper. Measured on CPython 3.14 (8M-iteration numeric kernel): 1.22 s pure Python vs 0.011 s steady-state (~110×), first call 0.79 s including compilation.

- **Static hotness** (constant loop bounds ≥ 10 000 or nested loops) compiles at decoration time; a strict whitelist predicts numba compatibility (int/float arithmetic, `range` loops, `math.*`, no containers/strings/attributes).
- **Loop outlining** extracts hot whitelisted loops out of mixed functions and module top level into fresh compiled functions, with proven input/output sets.
- **njit inter-calls**: candidate functions may call each other (fixpoint selection, call cycles dropped, compiled copies call raw dispatchers).
- **Runtime lazy compilation** covers *variable* loop bounds (`for i in range(n)`): the wrapper observes plain-Python calls and compiles when a trigger fires — bound argument ≥ `OPAST_JIT_LAZY_BOUND` (default 10 000), a single call ≥ 0.1 s, or ≥ 10 calls totalling ≥ 0.3 s. The triggering call's Python result doubles as the verification expectation, and `numba` itself is imported only on the first compilation attempt — a script whose lazy candidates never get hot pays nothing.
- **First-call verification**: whitelisted functions are pure, so the first call runs both versions and compares results; a divergence (in practice: int64 wraparound, which no static filter can rule out) triggers a permanent fallback to Python instead of silently wrong answers. `OPAST_JIT_NO_VERIFY=1` opts out.
- **Layered degradation**: no numba / incompatible interpreter / `OPAST_DISABLE_JIT` → original function; any numba error at compile or call time → permanent Python fallback. `OPAST_JIT_DEBUG=1` explains every fallback and lazy trigger on stderr.

⚠️ Opt-in semantic caveat: numba integers are fixed 64-bit — intermediate values beyond ±9.2e18 wrap silently. This is why `--jit` is not on by default and not part of the semantic-preservation contract; the verification above is a safety net, not a proof.

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
