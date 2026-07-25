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

1. **Prove, then rewrite.** Passes only fire on facts established by static analysis: per-function *proven-int* type inference (a greatest-fixpoint over all bindings), interval (value-range) analysis with widening, escape analysis for *fresh containers* (built locally, never leaked, never mutated), straight-line dominance scans for definite binding, and module-wide stability checks for names (bound exactly once, never `global`-declared, no dynamic constructs anywhere).
2. **Dynamic code disables optimization, scope by scope.** Any appearance of `eval` / `exec` / `globals` / `locals` / `vars` / `compile` / `__import__` / frame-introspection attributes / `from m import *` taints the enclosing scope; tainted scopes (and everything nested inside) are skipped entirely. A tainted module top level disables the whole file. The check is name-based and deliberately over-conservative.
3. **Zero runtime overhead.** The static passes emit ordinary Python source — no guards, no helper runtime. The only runtime machinery lives behind the opt-in `--jit` flag, and it degrades to plain Python on any failure.

## Optimization passes

The pipeline iterates to a fixpoint (default ≤ 8 rounds): de-dynamize → folding → constant propagation → algebraic → loop-fold → dead code → range-to-iter → LICM → CSE → unused → inline → loop-to-comp → comp-to-map → localize. Each pass feeds the next; collapsed constants cascade outward across iterations. See [README-ZH.md](README-ZH.md) for the full safety-condition spec of each pass.

| Pass | What it does |
| --- | --- |
| `de-dynamize` | Expands *pointless* dynamic code into static equivalents: top-level `eval("<const expr>")` → the expression, `exec("<const stmts>")` → the statements, `globals()['x'] = v` → `x = v`, `getattr(o, 'a')` → `o.a`, statement-form `setattr`/`delattr` → attribute syntax. All-or-nothing: commits only if the whole module ends up free of dynamic constructs, otherwise rolls back — un-tainting the module unlocks every other pass. |
| `constant-folding` | Folds constant arithmetic / strings / comparisons / boolean short-circuits / subscripts. Expressions that would raise at runtime are preserved; guardrails cap optimization-time work and literal sizes, with partial sub-expression folding allowed. |
| `const-prop` | Substitutes constants for variables along four routes: single-binding names (whole scope), **cross-scope** module constants into later-defined functions, **span propagation** for multiply-bound names (from an assignment up to the first statement that could rebind), and **copy propagation** `y = x` between plain locals (function scopes only). |
| `algebraic` | Identity cleanup (`x+0`, `x*1`, `-(-x)`, …) plus strength reduction — `E % 2**k → E & mask`, `E // 2**k → E >> k`, `E ** 2 → E * E` — and interval-analysis-backed `abs(E) → E` for provably non-negative `E`. Proven-`float` expressions get only the **bit-exact** identities (`F - 0`, `F * 1`, `F / 1`, `F ** 1`, `+F`, `-(-F)`); `F + 0` and `F // 1` are deliberately excluded because they change `-0.0` and floor respectively. Never duplicates or drops effects. |
| `loop-fold` | **Closed-form loop evaluation**: a `for i in range(<const>)` loop whose body is pure int arithmetic (no calls) is simulated exactly at optimization time and replaced by its final constant assignments. Step and magnitude budgets; any simulated exception keeps the loop. A successful fold proves the loop cannot raise, so it is legal even inside `try`/`with`. |
| `dead-code` | Unreachable statements after `return`/`raise`/`break`/`continue`, constant-condition `if`/`while` (with `else` semantics), `assert True`, useless constant expression statements, redundant `pass`. |
| `range-to-iter` | `for i in range(len(x))` over a provably fresh *sequence* becomes `for v in x` (index dead) or `for i, v in enumerate(x)` (index live); per-iteration `BINARY_SUBSCR` lookups disappear. Only exact `x[i]` loads are replaced; the enumerate form keeps `x` and `i` bound, so leftovers stay correct. |
| `licm` | Hoists provably pure-and-total loop-invariant int **and float** expressions (plus `len()` of fresh containers) out of loop bodies and `while` tests into pre-loop temporaries, with dominance-checked definite binding. Float admits `+`/`-`/`*` (overflow yields `inf`, never raises) and division by a non-zero constant; `**` is excluded (`(-8.0) ** 0.5` is complex), and the int/float name sets stay separate so int-only forms like `F << 2` cannot slip through. |
| `cse` | Merges repeated provably pure int/float expressions (same criteria as LICM) within a statement block into a temporary; speculative evaluation is sound because the expressions cannot raise. |
| `unused` | Removes unused imports, unused module-level functions (post-inlining helpers), and dead local stores (effectful right-hand sides are downgraded to bare expressions). Module-level variable assignments are deliberately kept — module globals are observable API. |
| `inline` | Two shapes: expression-body functions (`def f(...): return <expr>`) inline at any call site with constant/name arguments; straight-line statement bodies (≤ 6 assignments + `return`) inline at statement positions with arbitrary positional arguments via ordered temporaries. Requires module-wide name stability; the leftover `def` is cleaned by `unused`. |
| `loop-to-comp` | `x = []` + an adjacent append-accumulation loop (nested `for`/guard-`if` chains allowed) becomes a list comprehension (`set()`/`.add` → set comprehension): dedicated `LIST_APPEND`/`SET_ADD` bytecode instead of a per-iteration attribute lookup + method call. Rejected inside `try`/`with` (a mid-loop exception would expose the partially-built list). |
| `comp-to-map` | `(f(x) for x in it)` → `map(f, it)`, with `filter` for guards, restricted to positions where the generator/`map` identity difference is unobservable and `f` is provably stable. |
| `localize` | Per-iteration reads of stable globals/builtins inside loops become pre-loop locals (`LOAD_GLOBAL` → `LOAD_FAST`). Runs last so other passes claim names first. |

LICM/CSE additionally support **`len()` caching for fresh containers** — sound only because escape analysis guarantees no reference ever leaves the function, so no call can mutate the container.

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

16 built-in workloads, each executed twice per measurement (original vs optimized) in the same interpreter with GC disabled and a `RESULT` equality check. Note that CPython's own compiler already does trivial constant folding — opast's wins come from what CPython does *not* do: inlining, type-proven algebraic rewrites, loop rewrites, de-dynamization.

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

## Experimental: `--jit` (numba, off by default)

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
