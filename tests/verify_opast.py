from __future__ import annotations

import os
import re
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = Path(__file__).resolve().parent / "cases"
PYTHON = sys.executable


@dataclass
class Area:
    name: str
    failures: list[str] = field(default_factory=list)
    repros: list[str] = field(default_factory=list)

    def fail(self, msg: str, *commands: str) -> None:
        self.failures.append(msg)
        self.repros.extend(commands)


def env() -> dict[str, str]:
    out = os.environ.copy()
    out["PYTHONPATH"] = str(ROOT / "src")
    return out


def write_case(name: str, source: str) -> Path:
    CASE_DIR.mkdir(parents=True, exist_ok=True)
    path = CASE_DIR / f"{name}.py"
    path.write_text(textwrap.dedent(source).lstrip(), encoding="utf-8")
    return path


def run(cmd: list[str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*cmd, *args],
        cwd=ROOT,
        env=env(),
        text=True,
        capture_output=True,
    )


def command_text(cmd: list[str], *args: str) -> str:
    parts = [*cmd, *args]
    return "$env:PYTHONPATH='src'; " + " ".join(
        f'"{p}"' if any(ch in p for ch in " :\\") else p for p in parts
    )


def assert_same_output(area: Area, path: Path, *args: str) -> None:
    original_cmd = [PYTHON, str(path)]
    optimized_cmd = [PYTHON, "-m", "opast", str(path)]
    a = run(original_cmd, *args)
    b = run(optimized_cmd, *args)
    if (a.returncode, a.stdout, a.stderr) != (b.returncode, b.stdout, b.stderr):
        area.fail(
            f"{path.name}: original and optimized run differ\n"
            f"  original rc={a.returncode} stdout={a.stdout!r} stderr={a.stderr!r}\n"
            f"  opast    rc={b.returncode} stdout={b.stdout!r} stderr={b.stderr!r}",
            command_text(original_cmd, *args),
            command_text(optimized_cmd, *args),
        )


def assert_same_success_output(area: Area, path: Path, *args: str) -> None:
    original_cmd = [PYTHON, str(path)]
    optimized_cmd = [PYTHON, "-m", "opast", str(path)]
    a = run(original_cmd, *args)
    b = run(optimized_cmd, *args)
    if a.returncode != 0 or b.returncode != 0 or a.stdout != b.stdout:
        area.fail(
            f"{path.name}: successful stdout comparison failed\n"
            f"  original rc={a.returncode} stdout={a.stdout!r} stderr={a.stderr!r}\n"
            f"  opast    rc={b.returncode} stdout={b.stdout!r} stderr={b.stderr!r}",
            command_text(original_cmd, *args),
            command_text(optimized_cmd, *args),
        )


def show(path: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return run([PYTHON, "-m", "opast", "--show", "--no-run", *extra, str(path)])


def verify_jit() -> Area:
    area = Area("jit")

    hot = write_case(
        "jit_hot_numeric",
        """
        import sys

        def hot(n):
            total = 0
            for i in range(12000):
                total += (i % 17) * (i % 19) + n
            return total + n

        n = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
        print(hot(n))
        """,
    )
    assert_same_success_output(area, hot, "20000")
    jit_run = run([PYTHON, "-m", "opast", "--jit", str(hot)], "20000")
    original = run([PYTHON, str(hot)], "20000")
    if (jit_run.returncode, jit_run.stdout) != (original.returncode, original.stdout):
        area.fail(
            f"--jit runtime output differed: original={original.stdout!r}/{original.returncode}, jit={jit_run.stdout!r}/{jit_run.returncode}",
            command_text([PYTHON, str(hot)], "20000"),
            command_text([PYTHON, "-m", "opast", "--jit", str(hot)], "20000"),
        )

    normal_show = show(hot)
    jit_show = show(hot, "--jit")
    if normal_show.returncode != 0 or "_opast_jit" in normal_show.stdout:
        area.fail("JIT helper/decorator appeared without --jit", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(hot)]))
    if jit_show.returncode != 0 or "import opast.jitsupport as _opast_jit" not in jit_show.stdout or "@_opast_jit.maybe_njit" not in jit_show.stdout:
        area.fail("--jit --show did not inject jitsupport import/decorator", command_text([PYTHON, "-m", "opast", "--show", "--no-run", "--jit", str(hot)]))

    report = run([PYTHON, "-m", "opast", "--jit", "--report", "--no-run", str(hot)])
    if report.returncode != 0 or "jit:" not in report.stderr:
        area.fail("--jit --report did not include jit pass stats", command_text([PYTHON, "-m", "opast", "--jit", "--report", "--no-run", str(hot)]))

    fallback = write_case(
        "jit_runtime_fallback",
        """
        def hot(n, x):
            total = 0
            for i in range(12000):
                total += i % 3 + n
            return x * 2 or total

        print(hot(12000, 7))
        print(hot(12000, "ab"))
        print(getattr(hot, "opast_compiled", None) is not None)
        """,
    )
    jit_fb = run([PYTHON, "-m", "opast", "--jit", str(fallback)])
    if jit_fb.returncode != 0 or jit_fb.stdout != "14\nabab\nTrue\n":
        area.fail(
            f"JIT runtime fallback failed: rc={jit_fb.returncode} stdout={jit_fb.stdout!r} stderr={jit_fb.stderr!r}",
            command_text([PYTHON, "-m", "opast", "--jit", str(fallback)]),
        )

    materialized = write_case(
        "jit_materialized_source",
        """
        import pathlib
        import sys

        def hot(n):
            total = 0
            for i in range(12000):
                total += i + n
            return total + n

        print(hot(12000))
        print(pathlib.Path(__file__).resolve() == pathlib.Path(sys.argv[0]).resolve())
        print("opast-jit" in pathlib.Path(hot.__wrapped__.__code__.co_filename).parts)
        print(pathlib.Path(hot.__wrapped__.__code__.co_filename).exists())
        """,
    )
    mat = run([PYTHON, "-m", "opast", "--jit", str(materialized)])
    if mat.returncode != 0 or mat.stdout.splitlines()[-3:] != ["True", "True", "True"]:
        area.fail(
            f"JIT source materialization/__file__ check failed: rc={mat.returncode} stdout={mat.stdout!r} stderr={mat.stderr!r}",
            command_text([PYTHON, "-m", "opast", "--jit", str(materialized)]),
        )

    dynamic = write_case(
        "jit_dynamic_skip",
        """
        def hot(n):
            total = 0
            for i in range(n):
                total += i
            return total

        code = "1"
        eval(code)
        print(hot(12000))
        """,
    )
    assert_same_success_output(area, dynamic)
    s = show(dynamic, "--jit")
    if s.returncode != 0 or "_opast_jit" in s.stdout:
        area.fail("JIT injection ran despite remaining dynamic construct", command_text([PYTHON, "-m", "opast", "--show", "--no-run", "--jit", str(dynamic)]))

    cold = write_case(
        "jit_cold_skip",
        """
        def cold(n):
            total = 0
            for i in range(10):
                total += i
            return total + n

        print(cold(5))
        """,
    )
    s = show(cold, "--jit")
    if s.returncode != 0 or "_opast_jit" in s.stdout:
        area.fail("JIT injection decorated a cold function", command_text([PYTHON, "-m", "opast", "--show", "--no-run", "--jit", str(cold)]))
    return area


def verify_benchmark() -> Area:
    area = Area("benchmark")
    bench_cmd = [PYTHON, "-m", "opast.bench"]

    listed = run([*bench_cmd, "--list"])
    expected = {
        "algebra",
        "control",
        "dedynamize",
        "inline",
        "mixed",
        "licm",
        "lencache",
        "comptomap",
    }
    if listed.returncode != 0 or not expected.issubset({line.split()[0] for line in listed.stdout.splitlines() if line.strip()}):
        area.fail("--list did not report all built-in workloads", command_text([*bench_cmd, "--list"]))

    pattern = re.compile(r"^(algebra|control|dedynamize|inline|mixed|licm|lencache|comptomap)\s+.*?\s+([0-9.]+)x\s+.*?\s+(\d+)\s+(OK|MISMATCH!)$", re.M)
    rows = {}
    cp = None
    timing_errors: list[str] = []
    for _ in range(3):
        cp = run([*bench_cmd, "-r", "2"])
        if cp.returncode != 0:
            timing_errors = [f"benchmark -r 2 exited non-zero: {cp.stderr!r}"]
            continue
        rows = {
            name: {"speedup": float(speedup), "changes": int(changes), "result": result}
            for name, speedup, changes, result in pattern.findall(cp.stdout)
        }
        timing_errors = []
        if set(rows) != expected:
            timing_errors.append(f"benchmark did not report exactly all 5 workload rows: {sorted(rows)}")
        for name in expected:
            if rows.get(name, {}).get("result") != "OK":
                timing_errors.append(f"{name} workload result was not OK")
        if "control" in rows:
            if rows["control"]["changes"] != 0:
                timing_errors.append("control workload did not report 0 changes")
            if not (0.9 <= rows["control"]["speedup"] <= 1.1):
                timing_errors.append(f"control speedup {rows['control']['speedup']:.2f}x outside 0.9-1.1")
        for name in ("dedynamize", "inline", "mixed", "algebra", "licm", "lencache", "comptomap"):
            if name in rows and rows[name]["speedup"] <= 1.0:
                timing_errors.append(f"{name} speedup was not >1x: {rows[name]['speedup']:.2f}x")
        if not timing_errors:
            break
    for err in timing_errors:
        area.fail(err, command_text([*bench_cmd, "-r", "2"]))

    single = run([*bench_cmd, "-r", "2", "inline"])
    if single.returncode != 0 or "inline" not in single.stdout or " OK" not in single.stdout or "1 workload(s)" not in single.stdout:
        area.fail("single workload selection by name failed", command_text([*bench_cmd, "-r", "2", "inline"]))

    external = write_case("bench_external", "total = 0\nfor i in range(1000):\n    total += i\nRESULT = total\n")
    ext = run([*bench_cmd, "-r", "2", str(external)])
    if ext.returncode != 0 or "bench_external" not in ext.stdout or " OK" not in ext.stdout:
        area.fail("external .py workload path failed", command_text([*bench_cmd, "-r", "2", str(external)]))

    harness = (ROOT / "src" / "opast" / "bench" / "__main__.py").read_text(encoding="utf-8")
    methodology_terms = ["Warmup runs double as the correctness check", "gc.disable()", "for _ in range(repeat):", "orig_times.append", "opt_times.append", "min(orig_times)"]
    for term in methodology_terms:
        if term not in harness:
            area.fail(f"benchmark methodology check missing {term!r}", str(ROOT / "src" / "opast" / "bench" / "__main__.py"))
    return area


def verify_constant_propagation() -> Area:
    area = Area("constant propagation")

    basic = write_case(
        "constprop_basic",
        """
        x = 2 + 3
        print(x + 4)

        def f():
            y = "ab"
            print(y + "cd")
        f()
        """,
    )
    assert_same_output(area, basic)
    s = show(basic)
    if s.returncode != 0 or "print(9)" not in s.stdout or "print('abcd')" not in s.stdout:
        area.fail("fold -> const-prop cascade did not produce folded constants", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(basic)]))

    after_only = write_case(
        "constprop_after_only",
        """
        def f():
            try:
                print(x)
            except Exception as e:
                print(type(e).__name__)
            x = 7
            print(x)
        f()
        """,
    )
    assert_same_output(area, after_only)
    s = show(after_only)
    if s.returncode != 0 or "print(x)" not in s.stdout or "print(7)" not in s.stdout:
        area.fail("constant propagation changed a use before assignment or missed later use", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(after_only)]))

    nested = write_case(
        "constprop_nested_scope",
        """
        def outer():
            x = 11
            def inner():
                return x
            return inner()
        print(outer())
        """,
    )
    assert_same_output(area, nested)
    s = show(nested)
    if s.returncode != 0 or "return x" not in s.stdout:
        area.fail("constant propagation entered a nested closure scope", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(nested)]))

    multiple = write_case(
        "constprop_multiple_bindings",
        """
        def f(flag):
            x = 1
            if flag:
                x = 2
            return x + 3
        print(f(False), f(True))
        """,
    )
    assert_same_output(area, multiple)
    s = show(multiple)
    if s.returncode != 0 or "x + 3" not in s.stdout:
        area.fail("name with multiple bindings was propagated", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(multiple)]))

    dynamic = write_case(
        "constprop_dynamic_skip",
        """
        x = 4
        code = "1"
        eval(code)
        print(x)
        """,
    )
    assert_same_output(area, dynamic)
    s = show(dynamic)
    if s.returncode != 0 or "x = 4" not in s.stdout or "print(4)" in s.stdout:
        area.fail("module-level propagation ran despite remaining dynamic code", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(dynamic)]))

    global_decl = write_case(
        "constprop_global_decl",
        """
        x = 5
        def g():
            global x
            x = 6
        g()
        print(x)
        """,
    )
    assert_same_output(area, global_decl)
    s = show(global_decl)
    if s.returncode != 0 or "print(x)" not in s.stdout:
        area.fail("module constant with global declaration was propagated", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(global_decl)]))

    nonlocal_decl = write_case(
        "constprop_nonlocal_decl",
        """
        def outer():
            x = 5
            def inner():
                nonlocal x
                x = 6
            inner()
            return x
        print(outer())
        """,
    )
    assert_same_output(area, nonlocal_decl)
    s = show(nonlocal_decl)
    if s.returncode != 0 or "return x" not in s.stdout:
        area.fail("function constant with nested nonlocal declaration was propagated", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(nonlocal_decl)]))

    long_str = write_case(
        "constprop_long_string",
        """
        def f():
            x = "abcdefghijklmnopqrstuvwxyz" * 6
            return len(x)
        print(f())
        """,
    )
    assert_same_output(area, long_str)
    s = show(long_str)
    if s.returncode != 0 or "len(x)" not in s.stdout:
        area.fail("long string constant was duplicated by propagation", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(long_str)]))

    tuple_unpack = write_case(
        "constprop_tuple_unpack",
        """
        a, b = 2, 3
        print(a + b)

        def f():
            x, y = (4, 5)
            return x * y
        print(f())
        """,
    )
    assert_same_output(area, tuple_unpack)
    s = show(tuple_unpack)
    if s.returncode != 0 or "print(5)" not in s.stdout or "return 20" not in s.stdout:
        area.fail("constant propagation did not handle flat constant tuple unpacking", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(tuple_unpack)]))

    tuple_partial = write_case(
        "constprop_tuple_unpack_rejects",
        """
        def make():
            print("make")
            return 9

        x, y = 1, make()
        print(x, y)
        """,
    )
    assert_same_output(area, tuple_partial)
    s = show(tuple_partial)
    tuple_assignment_kept = (
        "x, y = (1, make())" in s.stdout
        or "x, y = 1, make()" in s.stdout
    )
    if s.returncode != 0 or not tuple_assignment_kept or "print(x, y)" not in s.stdout:
        area.fail("tuple unpacking with non-constant element was propagated", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(tuple_partial)]))

    tuple_star = write_case(
        "constprop_tuple_unpack_starred",
        """
        a, *rest = 1, 2, 3
        print(a, rest)
        """,
    )
    assert_same_output(area, tuple_star)
    s = show(tuple_star)
    if s.returncode != 0 or "a, *rest" not in s.stdout or "print(a, rest)" not in s.stdout:
        area.fail("starred tuple unpacking was propagated", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(tuple_star)]))
    return area


def verify_licm() -> Area:
    area = Area("loop-invariant motion")

    basic = write_case(
        "licm_basic",
        """
        def f():
            factor = 0
            k = 0
            while k < 5:
                factor = factor * 3 + k
                k += 1
            acc = 0
            i = 0
            while i < 10:
                acc = acc + factor * 7 - (factor << 3) + factor // 5
                i += 1
            return acc
        print(f())
        """,
    )
    assert_same_output(area, basic)
    s = show(basic)
    if s.returncode != 0 or "_opast_inv_" not in s.stdout or "factor * 7" not in s.stdout or "factor << 3" not in s.stdout or "factor // 5" not in s.stdout:
        area.fail("LICM did not hoist invariant int expressions before loop", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(basic)]))
    elif "acc = acc + factor * 7" in s.stdout or "factor << 3) + factor // 5" in s.stdout:
        area.fail("LICM left original invariant expressions in loop body", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(basic)]))

    test_expr = write_case(
        "licm_while_test",
        """
        def f():
            limit = 0
            k = 0
            while k < 3:
                limit = limit + 2
                k += 1
            i = 0
            total = 0
            while i < limit + 1:
                total += i
                i += 1
            return total
        print(f())
        """,
    )
    assert_same_output(area, test_expr)
    s = show(test_expr)
    if s.returncode != 0 or "_opast_inv_" not in s.stdout or "while i < _opast_inv_" not in s.stdout:
        area.fail("LICM did not hoist invariant expression from while test", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(test_expr)]))

    module_loop = write_case(
        "licm_module_level_skip",
        """
        factor = 7
        i = 0
        acc = 0
        while i < 3:
            acc += factor * 2
            i += 1
        print(acc)
        """,
    )
    assert_same_output(area, module_loop)
    s = show(module_loop)
    if s.returncode != 0 or "_opast_inv_" in s.stdout:
        area.fail("LICM hoisted at module level", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(module_loop)]))

    zero_iter_unbound = write_case(
        "licm_not_definitely_bound",
        """
        def f(flag):
            if flag:
                x = 2
            i = 0
            n = 0
            while i < n:
                print(x + 1)
                i += 1
            return "ok"
        print(f(False))
        """,
    )
    assert_same_output(area, zero_iter_unbound)
    s = show(zero_iter_unbound)
    if s.returncode != 0 or "_opast_inv_" in s.stdout:
        area.fail("LICM hoisted expression using name not definitely bound before zero-iteration loop", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(zero_iter_unbound)]))

    unsafe_ops = write_case(
        "licm_unsafe_ops",
        """
        def f():
            x = 10
            d = 2
            i = 0
            total = 0
            while i < 3:
                total += x // d
                i += 1
            return total
        print(f())
        """,
    )
    assert_same_output(area, unsafe_ops)
    s = show(unsafe_ops)
    if s.returncode != 0 or "_opast_inv_" in s.stdout:
        area.fail("LICM hoisted floor-division with non-constant divisor", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(unsafe_ops)]))

    loop_mutated = write_case(
        "licm_loop_mutated_name",
        """
        def f():
            x = 1
            i = 0
            total = 0
            while i < 3:
                x = x + 1
                total += x * 2
                i += 1
            return total
        print(f())
        """,
    )
    assert_same_output(area, loop_mutated)
    s = show(loop_mutated)
    if s.returncode != 0 or "_opast_inv_" in s.stdout:
        area.fail("LICM hoisted expression using loop-mutated name", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(loop_mutated)]))

    dynamic = write_case(
        "licm_dynamic_skip",
        """
        def f():
            factor = 3
            code = "1"
            eval(code)
            i = 0
            acc = 0
            while i < 3:
                acc += factor * 2
                i += 1
            return acc
        print(f())
        """,
    )
    assert_same_output(area, dynamic)
    s = show(dynamic)
    if s.returncode != 0 or "_opast_inv_" in s.stdout:
        area.fail("LICM ran inside dynamic function scope", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(dynamic)]))
    return area


def verify_cse_lencache() -> Area:
    area = Area("common subexpression / len cache")

    basic = write_case(
        "cse_basic",
        """
        def f():
            a = 0
            i = 0
            while i < 4:
                a = a + i
                i += 1
            return (a * 7) + (a * 7)
        print(f())
        """,
    )
    assert_same_output(area, basic)
    s = show(basic)
    if s.returncode != 0 or "_opast_cse_" not in s.stdout or "a * 7" not in s.stdout or "return _opast_cse_" not in s.stdout:
        area.fail("CSE did not cache repeated pure int subexpression", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(basic)]))

    rebind_span = write_case(
        "cse_rebind_span",
        """
        def f():
            a = 0
            i = 0
            while i < 4:
                a = a + i
                i += 1
            x = a * 7
            a = a + 1
            y = a * 7
            return x + y
        print(f())
        """,
    )
    assert_same_output(area, rebind_span)
    s = show(rebind_span)
    if s.returncode != 0 or "_opast_cse_" in s.stdout:
        area.fail("CSE crossed a name rebind between repeated expressions", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(rebind_span)]))

    len_loop = write_case(
        "lencache_licm_loop",
        """
        def f():
            data = [1, 2, 3, 4]
            total = 0
            i = 0
            while i < 4:
                total = total + len(data) + len(data)
                i += 1
            return total
        print(f())
        """,
    )
    assert_same_output(area, len_loop)
    s = show(len_loop)
    if s.returncode != 0 or "_opast_inv_" not in s.stdout or "len(data)" not in s.stdout or "while i < 4" not in s.stdout:
        area.fail("LICM did not hoist fresh-container len expression out of loop", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(len_loop)]))
    elif "total = total + len(data)" in s.stdout:
        area.fail("fresh-container len expression remained in loop body after LICM", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(len_loop)]))

    len_cse = write_case(
        "lencache_cse_block",
        """
        def f():
            data = list(range(5))
            a = len(data) + 1
            b = len(data) + 2
            return a + b
        print(f())
        """,
    )
    assert_same_output(area, len_cse)
    s = show(len_cse)
    if s.returncode != 0 or "_opast_cse_" not in s.stdout or "len(data)" not in s.stdout:
        area.fail("CSE did not cache repeated fresh-container len calls in a block", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(len_cse)]))

    bound_builtin = write_case(
        "lencache_bound_builtin_gate",
        """
        def f():
            data = [1, 2, 3]
            a = len(data) + 1
            b = len(data) + 2
            return a + b
        len = len
        print(f())
        """,
    )
    assert_same_output(area, bound_builtin)
    s = show(bound_builtin)
    if s.returncode != 0 or "_opast_cse_" in s.stdout or "_opast_inv_" in s.stdout:
        area.fail("len/cache machinery ran despite module-level len binding", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(bound_builtin)]))

    escaped = write_case(
        "lencache_container_escape",
        """
        def consume(x):
            x.append(4)
            return 0

        def f():
            data = [1, 2, 3]
            consume(data)
            return len(data) + len(data)
        print(f())
        """,
    )
    assert_same_output(area, escaped)
    s = show(escaped)
    if s.returncode != 0 or "_opast_cse_" in s.stdout or "_opast_inv_" in s.stdout:
        area.fail("len/cache machinery trusted an escaped mutable container", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(escaped)]))

    mutated = write_case(
        "lencache_container_mutation",
        """
        def f():
            data = [1, 2, 3]
            data[0] = 9
            return len(data) + len(data)
        print(f())
        """,
    )
    assert_same_output(area, mutated)
    s = show(mutated)
    if s.returncode != 0 or "_opast_cse_" in s.stdout or "_opast_inv_" in s.stdout:
        area.fail("len/cache machinery trusted a container with subscript write", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(mutated)]))

    dynamic = write_case(
        "lencache_dynamic_gate",
        """
        def f():
            data = [1, 2, 3]
            code = "1"
            eval(code)
            return len(data) + len(data)
        print(f())
        """,
    )
    assert_same_output(area, dynamic)
    s = show(dynamic)
    if s.returncode != 0 or "_opast_cse_" in s.stdout or "_opast_inv_" in s.stdout:
        area.fail("len/cache machinery ran in module with remaining dynamic construct", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(dynamic)]))
    return area


def verify_comprehension_to_map() -> Area:
    area = Area("comprehension-to-map")

    builtins_case = write_case(
        "comp_builtin_rewrites",
        """
        data = [-2, 0, 3]
        print(sum(abs(x) for x in data))
        print(sum(x for x in data if bool(x)))
        print(len([float(x) for x in data]))
        print(sorted({str(x) for x in data}))
        """,
    )
    assert_same_output(area, builtins_case)
    s = show(builtins_case)
    if s.returncode != 0:
        area.fail(f"builtin comprehension rewrite --show failed: {s.stderr!r}", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(builtins_case)]))
    else:
        for expected in ["sum(map(abs, data))", "sum(filter(bool, data))", "list(map(float, data))", "set(map(str, data))"]:
            if expected not in s.stdout:
                area.fail(f"missing expected comprehension rewrite {expected!r}", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(builtins_case)]))

    map_filter_case = write_case(
        "comp_map_filter_chain",
        """
        data = [-2, 0, 3]
        print(list(abs(x) for x in data if bool(x)))
        """,
    )
    assert_same_output(area, map_filter_case)
    s = show(map_filter_case)
    if s.returncode != 0 or "list(map(abs, filter(bool, data)))" not in s.stdout:
        area.fail("map(filter(...)) chain was not generated for f(x) with if g(x)", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(map_filter_case)]))

    module_fn_gen = write_case(
        "comp_module_fn_genexpr",
        """
        def f(x):
            y = x + 1
            return y

        data = [1, 2, 3]
        print(sum(f(x) for x in data))
        """,
    )
    assert_same_output(area, module_fn_gen)
    s = show(module_fn_gen)
    if s.returncode != 0 or "sum(map(f, data))" not in s.stdout:
        area.fail("generator expression with stable module function was not rewritten", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(module_fn_gen)]))

    module_fn_list = write_case(
        "comp_module_fn_listcomp_blocked",
        """
        def f(x):
            y = x + 1
            return y

        data = [1, 2, 3]
        print([f(x) for x in data])
        """,
    )
    assert_same_output(area, module_fn_list)
    s = show(module_fn_list)
    if s.returncode != 0 or "list(map(f, data))" in s.stdout or "[f(x) for x in data]" not in s.stdout:
        area.fail("list comprehension used module function despite builtins-only rule", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(module_fn_list)]))

    gen_identity = write_case(
        "comp_gen_identity_blocked",
        """
        data = [-1, 2]
        g = (abs(x) for x in data)
        print(type(g).__name__, hasattr(g, "send"), list(g))
        """,
    )
    assert_same_output(area, gen_identity)
    s = show(gen_identity)
    if s.returncode != 0 or "map(abs, data)" in s.stdout or "(abs(x) for x in data)" not in s.stdout:
        area.fail("generator expression in identity-observable position was rewritten", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(gen_identity)]))

    for_iter = write_case(
        "comp_for_iter_rewrite",
        """
        data = [-1, 2]
        total = 0
        for y in (abs(x) for x in data):
            total += y
        print(total)
        """,
    )
    assert_same_output(area, for_iter)
    s = show(for_iter)
    if s.returncode != 0 or "for y in map(abs, data):" not in s.stdout:
        area.fail("generator expression in for-iter position was not rewritten", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(for_iter)]))

    join_case = write_case(
        "comp_join_rewrite",
        """
        data = [1, 2, 3]
        print(",".join(str(x) for x in data))
        """,
    )
    assert_same_output(area, join_case)
    s = show(join_case)
    if s.returncode != 0 or "','.join(map(str, data))" not in s.stdout:
        area.fail("generator expression consumed by literal join was not rewritten", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(join_case)]))

    helper_shadow = write_case(
        "comp_helper_shadowing",
        """
        map = lambda f, it: ["bad"]
        data = [-1, 2]
        print(sum(abs(x) for x in data))
        """,
    )
    assert_same_output(area, helper_shadow)
    s = show(helper_shadow)
    if s.returncode != 0 or "map(abs, data)" in s.stdout:
        area.fail("comprehension rewrite used shadowed map helper", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(helper_shadow)]))

    consumer_shadow = write_case(
        "comp_consumer_shadowing",
        """
        sum = lambda it: "custom:" + type(it).__name__
        data = [-1, 2]
        print(sum(abs(x) for x in data))
        """,
    )
    assert_same_output(area, consumer_shadow)
    s = show(consumer_shadow)
    if s.returncode != 0 or "map(abs, data)" in s.stdout:
        area.fail("generator rewrite fired under shadowed consumer builtin", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(consumer_shadow)]))

    dynamic = write_case(
        "comp_dynamic_skip",
        """
        code = "1"
        eval(code)
        data = [-1, 2]
        print(sum(abs(x) for x in data))
        """,
    )
    assert_same_output(area, dynamic)
    s = show(dynamic)
    if s.returncode != 0 or "map(abs, data)" in s.stdout:
        area.fail("comprehension rewrite ran in module with remaining dynamic construct", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(dynamic)]))

    inline_shadow = write_case(
        "inline_comp_target_shadow_fix",
        """
        n = 10
        def f(v):
            return v + n

        print([f(1) for n in [100]])
        """,
    )
    assert_same_output(area, inline_shadow)
    s = show(inline_shadow)
    if s.returncode != 0 or "f(1)" not in s.stdout or "1 + n" in s.stdout:
        area.fail("FunctionInlining ignored comprehension target shadowing of free name", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(inline_shadow)]))

    return area


def verify_unused_elimination() -> Area:
    area = Area("unused elimination")

    imports = write_case(
        "unused_imports",
        """
        import math, json
        from collections import Counter, deque
        print(math.sqrt(9), Counter("aba")["a"])
        """,
    )
    assert_same_output(area, imports)
    s = show(imports)
    if s.returncode != 0:
        area.fail(f"unused imports --show failed: {s.stderr!r}", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(imports)]))
    else:
        if "json" in s.stdout or "deque" in s.stdout or "import math" not in s.stdout or "Counter" not in s.stdout:
            area.fail("unused import aliases were not trimmed correctly", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(imports)]))

    keep_imports = write_case(
        "unused_keep_future_star",
        """
        from __future__ import annotations
        from math import *
        print(sqrt(4))
        """,
    )
    assert_same_output(area, keep_imports)
    s = show(keep_imports)
    if s.returncode != 0 or "from __future__ import annotations" not in s.stdout or "from math import *" not in s.stdout:
        area.fail("__future__ or import * was removed", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(keep_imports)]))

    local_assign = write_case(
        "unused_local_assignments",
        """
        def side():
            print("side")
            return 3

        def f():
            pure = (1, 2, "x")
            impure = side()
            live = 4
            return live

        print(f())
        """,
    )
    assert_same_output(area, local_assign)
    s = show(local_assign)
    if s.returncode != 0:
        area.fail(f"unused local assignments --show failed: {s.stderr!r}", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(local_assign)]))
    else:
        if "pure =" in s.stdout or "impure =" in s.stdout or "side()" not in s.stdout or "return 4" not in s.stdout:
            area.fail("unused local assignments were not removed/downgraded as expected", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(local_assign)]))

    inline_cleanup = write_case(
        "unused_inline_cleanup",
        """
        def add(a, b):
            return a + b
        print(add(2, 3))
        """,
    )
    assert_same_output(area, inline_cleanup)
    s = show(inline_cleanup)
    if s.returncode != 0 or "def add" in s.stdout or "print(5)" not in s.stdout:
        area.fail("unused function helper was not removed after inlining/folding", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(inline_cleanup)]))

    module_var = write_case(
        "unused_module_var_kept",
        """
        x = 123
        print("ok")
        """,
    )
    assert_same_output(area, module_var)
    s = show(module_var)
    if s.returncode != 0 or "x = 123" not in s.stdout:
        area.fail("module-level variable assignment was removed", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(module_var)]))

    closure = write_case(
        "unused_closure_read_kept",
        """
        def outer():
            x = 8
            def inner():
                return x
            return inner()
        print(outer())
        """,
    )
    assert_same_output(area, closure)
    s = show(closure)
    if s.returncode != 0 or "x = 8" not in s.stdout:
        area.fail("local assignment read by nested closure was removed", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(closure)]))

    dunder_all = write_case(
        "unused_dunder_all_keeps",
        """
        __all__ = ["exported"]
        def exported():
            return 1
        def hidden():
            return 2
        print("module")
        """,
    )
    assert_same_output(area, dunder_all)
    s = show(dunder_all)
    if s.returncode != 0 or "def exported" not in s.stdout or "def hidden" in s.stdout:
        area.fail("__all__ did not keep exported function or hidden function survived", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(dunder_all)]))

    unknown_all = write_case(
        "unused_unknown_all_disables_module",
        """
        names = ["hidden"]
        __all__ = names
        def hidden():
            return 2
        print("module")
        """,
    )
    assert_same_output(area, unknown_all)
    s = show(unknown_all)
    if s.returncode != 0 or "def hidden" not in s.stdout:
        area.fail("non-literal __all__ did not disable module-level function removal", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(unknown_all)]))

    dynamic = write_case(
        "unused_dynamic_skip",
        """
        import math
        code = "1"
        eval(code)
        print("ok")
        """,
    )
    assert_same_output(area, dynamic)
    s = show(dynamic)
    if s.returncode != 0 or "import math" not in s.stdout:
        area.fail("unused elimination ran despite remaining dynamic constructs", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(dynamic)]))

    global_store = write_case(
        "unused_global_store_escape",
        """
        counter = 0
        def f():
            global counter
            counter = 1
        f()
        print(counter)
        """,
    )
    assert_same_output(area, global_store)
    s = show(global_store)
    if s.returncode != 0 or "counter = 1" not in s.stdout:
        area.fail("unused elimination removed a global-declared escaping store", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(global_store)]))

    nonlocal_store = write_case(
        "unused_nonlocal_store_escape",
        """
        def outer():
            counter = 0
            def f():
                nonlocal counter
                counter = 1
            f()
            return counter
        print(outer())
        """,
    )
    assert_same_output(area, nonlocal_store)
    s = show(nonlocal_store)
    if s.returncode != 0 or "counter = 1" not in s.stdout:
        area.fail("unused elimination removed a nonlocal-declared escaping store", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(nonlocal_store)]))

    global_import = write_case(
        "unused_global_import_escape",
        """
        def f():
            global os
            import os
        f()
        print(os.path.basename("a/b.txt"))
        """,
    )
    assert_same_output(area, global_import)
    s = show(global_import)
    if s.returncode != 0 or "import os" not in s.stdout:
        area.fail("unused elimination removed an import bound under global declaration", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(global_import)]))
    return area


def assert_exception(area: Area, path: Path, expected: str) -> None:
    a = run([PYTHON, str(path)])
    b = run([PYTHON, "-m", "opast", str(path)])
    if a.returncode == 0 or b.returncode == 0 or expected not in a.stderr or expected not in b.stderr:
        area.fail(
            f"{path.name}: expected {expected} from both original and optimized\n"
            f"  original rc={a.returncode} stderr={a.stderr!r}\n"
            f"  opast    rc={b.returncode} stderr={b.stderr!r}",
            command_text([PYTHON, str(path)]),
            command_text([PYTHON, "-m", "opast", str(path)]),
        )


def verify_greatest_fixpoint_analysis() -> Area:
    area = Area("greatest-fixpoint int inference")

    acc = write_case(
        "gfp_accumulator",
        """
        def f(n):
            acc = 0
            i = 0
            while i < n:
                acc = acc + i * 1
                acc = acc << 0
                i += 1
            return acc
        print(f(200))
        """,
    )
    assert_same_output(area, acc)
    s = show(acc)
    if s.returncode != 0 or "i * 1" in s.stdout or "acc << 0" in s.stdout:
        area.fail("self-referential int accumulator identities did not simplify", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(acc)]))

    branch_float = write_case(
        "gfp_branch_float_contamination",
        """
        def f(flag):
            x = 0
            if flag:
                x = 1.5
            return x + 0
        print(f(False))
        try:
            print(f(True))
        except Exception as e:
            print(type(e).__name__)
        """,
    )
    assert_same_output(area, branch_float)
    s = show(branch_float)
    if s.returncode != 0 or "x + 0" not in s.stdout:
        area.fail("float branch contamination did not preserve x + 0", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(branch_float)]))

    branch_call = write_case(
        "gfp_branch_call_contamination",
        """
        def some_call():
            return "s"
        def f(flag):
            x = 0
            if flag:
                x = some_call()
            return x + 0
        print(f(False))
        try:
            print(f(True))
        except Exception as e:
            print(type(e).__name__)
        """,
    )
    assert_same_output(area, branch_call)
    s = show(branch_call)
    if s.returncode != 0 or "x + 0" not in s.stdout:
        area.fail("call-result contamination did not preserve x + 0", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(branch_call)]))

    mutual = write_case("gfp_mutual_unbound", "def f():\n    a = b\n    b = a\n    return a + 0\nf()\n")
    assert_exception(area, mutual, "UnboundLocalError")

    self_only = write_case("gfp_self_only_unbound", "def f():\n    x = x\n    return x * 1\nf()\n")
    assert_exception(area, self_only, "UnboundLocalError")

    aug_only = write_case("gfp_aug_only_unbound", "def f():\n    x += 1\n    return x * 1\nf()\n")
    assert_exception(area, aug_only, "UnboundLocalError")

    div_contam = write_case("gfp_float_aug_contamination", "def f():\n    x = 1\n    x /= 2\n    return x * 1\nprint(f())\n")
    assert_same_output(area, div_contam)
    s = show(div_contam)
    if s.returncode != 0 or "x * 1" not in s.stdout:
        area.fail("float aug-assign contamination simplified x * 1", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(div_contam)]))

    bool_case = write_case(
        "gfp_bool_excluded",
        """
        def f(flag):
            x = True
            if flag:
                x = False
            return x * 1
        print(f(False), f(True))
        """,
    )
    assert_same_output(area, bool_case)
    s = show(bool_case)
    if s.returncode != 0 or "x * 1" not in s.stdout:
        area.fail("bool binding was treated as plain int", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(bool_case)]))

    try_case = write_case(
        "gfp_try_except_contamination",
        """
        def f(flag):
            x = 1
            try:
                if flag:
                    raise ValueError
            except ValueError:
                x = "s"
            return x + 0
        print(f(False))
        try:
            print(f(True))
        except Exception as e:
            print(type(e).__name__)
        """,
    )
    assert_same_output(area, try_case)
    s = show(try_case)
    if s.returncode != 0 or "x + 0" not in s.stdout:
        area.fail("try/except non-int rebinding did not preserve x + 0", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(try_case)]))
    return area


def verify_dedynamize_eval() -> Area:
    area = Area("de-dynamize eval")
    p = write_case(
        "dedyn_eval_const",
        """
        print(eval("'ab' + 'cd'"))
        """,
    )
    assert_same_success_output(area, p)
    s = show(p)
    if s.returncode != 0 or "eval" in s.stdout or "abcd" not in s.stdout:
        area.fail("module-level eval constant expression was not inlined/folded", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(p)]))

    rollback = write_case(
        "dedyn_eval_rollback_nonconst",
        """
        code = "40 + 2"
        print(eval("'ok'"))
        print(eval(code))
        """,
    )
    assert_same_success_output(area, rollback)
    s = show(rollback)
    if s.returncode != 0 or "eval(" not in s.stdout or "'ok'" not in s.stdout or "eval(code)" not in s.stdout:
        area.fail("non-constant eval did not roll back all eval rewrites", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(rollback)]))

    walrus = write_case("dedyn_eval_walrus", "print(eval('(x:=5)'))\nprint('x' in globals())\n")
    assert_same_success_output(area, walrus)
    s = show(walrus)
    if s.returncode != 0 or "eval('(x:=5)')" not in s.stdout:
        area.fail("eval string containing walrus was expanded", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(walrus)]))

    syntax = write_case("dedyn_eval_syntax", "print(eval('1+'))\n")
    a = run([PYTHON, str(syntax)])
    b = run([PYTHON, "-m", "opast", str(syntax)])
    if a.returncode == 0 or b.returncode == 0 or "SyntaxError" not in a.stderr or "SyntaxError" not in b.stderr:
        area.fail(
            "eval('1+') did not stay as runtime SyntaxError",
            command_text([PYTHON, str(syntax)]),
            command_text([PYTHON, "-m", "opast", str(syntax)]),
        )
    s = show(syntax)
    if s.returncode != 0 or "eval('1+')" not in s.stdout:
        area.fail("eval syntax-error string was expanded or removed", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(syntax)]))

    multiarg = write_case("dedyn_eval_multiarg", "print(eval('x', {'x': 7}))\n")
    assert_same_success_output(area, multiarg)
    s = show(multiarg)
    if s.returncode != 0 or "eval('x'" not in s.stdout:
        area.fail("2-arg eval was expanded", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(multiarg)]))
    multiarg3 = write_case("dedyn_eval_multiarg3", "print(eval('x', {'x': 7}, {}))\n")
    assert_same_success_output(area, multiarg3)
    s = show(multiarg3)
    if s.returncode != 0 or "eval('x'" not in s.stdout:
        area.fail("3-arg eval was expanded", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(multiarg3)]))
    return area


def verify_dedynamize_exec() -> Area:
    area = Area("de-dynamize exec")
    p = write_case(
        "dedyn_exec_splice",
        """
        exec('a = 2\\nb = a + 3')
        print(a, b)
        """,
    )
    assert_same_success_output(area, p)
    s = show(p)
    if s.returncode != 0 or "exec" in s.stdout or "a = 2" not in s.stdout or "b = 5" not in s.stdout or "print(2, 5)" not in s.stdout:
        area.fail("module-level exec constant statements were not spliced and optimized", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(p)]))

    in_func = write_case(
        "dedyn_exec_in_function",
        """
        x = 1
        def f():
            exec('x = 2')
            return x
        try:
            print(f())
        except Exception as e:
            print(type(e).__name__)
        print(x)
        """,
    )
    assert_same_success_output(area, in_func)
    s = show(in_func)
    if s.returncode != 0 or "exec('x = 2')" not in s.stdout:
        area.fail("exec inside function was expanded", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(in_func)]))

    empty_if = write_case(
        "dedyn_exec_empty_if",
        """
        flag = True
        if flag:
            exec('')
        print('done')
        """,
    )
    assert_same_success_output(area, empty_if)
    s = show(empty_if)
    if s.returncode != 0 or "exec" in s.stdout or "print('done')" not in s.stdout:
        area.fail("exec('') inside if-block was not safely removed by later passes", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(empty_if)]))

    future = write_case("dedyn_exec_future", "exec('from __future__ import annotations')\nprint('ok')\n")
    assert_same_success_output(area, future)
    s = show(future)
    if s.returncode != 0 or "exec('from __future__ import annotations')" not in s.stdout:
        area.fail("exec string containing __future__ import was expanded", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(future)]))
    return area


def verify_dedynamize_globals_vars() -> Area:
    area = Area("de-dynamize globals/vars")
    p = write_case(
        "dedyn_globals_vars_write",
        """
        globals()['x'] = 2
        vars()['y'] = x + 3
        print(x, y)
        """,
    )
    assert_same_success_output(area, p)
    s = show(p)
    if s.returncode != 0 or "globals()" in s.stdout or "vars()" in s.stdout or "x = 2" not in s.stdout or "y = 5" not in s.stdout or "print(2, 5)" not in s.stdout:
        area.fail("module-level globals/vars writes were not rewritten and optimized", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(p)]))

    read = write_case("dedyn_globals_read", "globals()['x'] = 1\nprint(globals()['x'])\n")
    assert_same_success_output(area, read)
    s = show(read)
    if s.returncode != 0 or "globals()['x']" not in s.stdout or "globals()['x'] = 1" not in s.stdout:
        area.fail("globals read was rewritten or write did not roll back with remaining dynamic read", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(read)]))

    in_func = write_case(
        "dedyn_globals_in_function",
        """
        def f():
            globals()['x'] = 4
        f()
        print(x)
        """,
    )
    assert_same_success_output(area, in_func)
    s = show(in_func)
    if s.returncode != 0 or "globals()['x'] = 4" not in s.stdout:
        area.fail("function-scope globals write was expanded", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(in_func)]))
    return area


def verify_dedynamize_attributes() -> Area:
    area = Area("de-dynamize attributes")
    p = write_case(
        "dedyn_attr_family",
        """
        class O:
            pass
        o = O()
        setattr(o, 'x', 5)
        print(getattr(o, 'x'))
        delattr(o, 'x')
        print(hasattr(o, 'x'))
        """,
    )
    assert_same_success_output(area, p)
    s = show(p)
    if s.returncode != 0 or "getattr" in s.stdout or "setattr" in s.stdout or "delattr" in s.stdout or "o.x = 5" not in s.stdout or "del o.x" not in s.stdout or "print(o.x)" not in s.stdout:
        area.fail("getattr/setattr/delattr family was not rewritten to attribute syntax", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(p)]))

    blocked_cases = {
        "dedyn_getattr_default": "class O: pass\no=O()\nprint(getattr(o, 'a', 9))\n",
        "dedyn_getattr_keyword": "class O: pass\no=O()\nprint(getattr(o, 'class'))\n",
        "dedyn_getattr_nonidentifier": "class O: pass\no=O()\nprint(getattr(o, 'not-ident'))\n",
        "dedyn_setattr_call_obj": "class O: pass\ndef make(): return O()\nsetattr(make(), 'x', 1)\n",
        "dedyn_getattr_rebound": "def my_func(o, n): return 3\ngetattr = my_func\nprint(getattr(1, 'real'))\n",
        "dedyn_getattr_param": "def f(getattr):\n    return getattr(1, 'x')\nprint(f(lambda o, n: 'ok'))\n",
        "dedyn_exec_binds_getattr": "class O: pass\no=O()\nexec('getattr = lambda o, n: 99')\nprint(getattr(o, 'x'))\n",
    }
    for name, src in blocked_cases.items():
        case = write_case(name, src)
        s = show(case)
        if s.returncode != 0:
            area.fail(f"{name}: --show failed: {s.stderr!r}", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(case)]))
            continue
        if name == "dedyn_getattr_default" and "getattr(o, 'a', 9)" not in s.stdout:
            area.fail("3-arg getattr was rewritten", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(case)]))
        if name == "dedyn_getattr_keyword" and "getattr(o, 'class')" not in s.stdout:
            area.fail("keyword attribute string was rewritten", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(case)]))
        if name == "dedyn_getattr_nonidentifier" and "getattr(o, 'not-ident')" not in s.stdout:
            area.fail("non-identifier attribute string was rewritten", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(case)]))
        if name == "dedyn_setattr_call_obj" and ("setattr(" not in s.stdout or ".x =" in s.stdout):
            area.fail("setattr with call object was rewritten", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(case)]))
        if name == "dedyn_getattr_rebound" and "getattr(1, 'real')" not in s.stdout:
            area.fail("getattr was rewritten despite module rebinding", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(case)]))
        if name == "dedyn_getattr_param" and "getattr(1, 'x')" not in s.stdout:
            area.fail("getattr was rewritten despite function parameter rebinding", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(case)]))
        if name == "dedyn_exec_binds_getattr" and ("exec(" not in s.stdout or "getattr(o, 'x')" not in s.stdout):
            area.fail("exec binding getattr did not abort family rewrites", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(case)]))
    return area


def verify_dedynamize_cascade_rollback() -> Area:
    area = Area("de-dynamize cascade/rollback")
    p = write_case(
        "dedyn_cascade",
        """
        exec('def add(a, b):\\n    return a + b')
        globals()['x'] = 1 + 2
        print(add(x, 4))
        """,
    )
    assert_same_success_output(area, p)
    s = show(p)
    if s.returncode != 0 or "exec" in s.stdout or "globals()" in s.stdout or "add(" in s.stdout or "x = 3" not in s.stdout or "print(7)" not in s.stdout:
        area.fail("successful de-dynamize did not unlock folding and inlining in the same run", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(p)]))

    rollback = write_case(
        "dedyn_rollback_purity",
        """
        code = "2 + 2"
        print(eval("'static'"))
        print(eval(code))
        globals()['x'] = 5
        print('done')
        """,
    )
    assert_same_success_output(area, rollback)
    s = show(rollback)
    if s.returncode != 0:
        area.fail(f"rollback purity --show failed: {s.stderr!r}", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(rollback)]))
    else:
        checks = [("eval(", "rollback purity lost eval calls"), ("'static'", "rollback purity lost expandable eval payload"), ("eval(code)", "rollback purity lost non-constant eval"), ("globals()['x'] = 5", "rollback purity lost globals write")]
        for expected, msg in checks:
            if expected not in s.stdout:
                area.fail(msg, command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(rollback)]))
    return area


def verify_constant_folding() -> Area:
    area = Area("constant folding")
    p = write_case(
        "constant_folding",
        """
        print(1 + 2 * 3)
        print("a" * 3, "abcdef"[2:5])
        try:
            print(1 / 0)
        except Exception as e:
            print(type(e).__name__)
        try:
            print(1 < "a")
        except Exception as e:
            print(type(e).__name__)
        print("done")
        """,
    )
    assert_same_output(area, p)
    s = show(p)
    if s.returncode != 0:
        area.fail(f"{p.name}: --show failed: {s.stderr!r}", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(p)]))
    else:
        if "1 / 0" not in s.stdout:
            area.fail("1/0 was folded or rewritten away", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(p)]))
        if "1 < 'a'" not in s.stdout and '1 < "a"' not in s.stdout:
            area.fail("1<'a' was folded or rewritten away", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(p)]))
    huge = write_case("huge_literal_guard", "print(10 ** 10 ** 10)\n")
    try:
        s = subprocess.run(
            [PYTHON, "-m", "opast", "--show", "--no-run", str(huge)],
            cwd=ROOT,
            env=env(),
            text=True,
            capture_output=True,
            timeout=3,
        )
    except subprocess.TimeoutExpired:
        area.fail("huge literal guard did not complete quickly", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(huge)]))
    else:
        if s.returncode != 0:
            area.fail(f"huge literal guard --show failed: {s.stderr!r}", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(huge)]))
        elif "**" not in s.stdout:
            area.fail("outer power was fully evaluated instead of preserved", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(huge)]))
        elif any(len(m.group(0)) > 1300 for m in re.finditer(r"\d+", s.stdout)):
            area.fail("optimized source emitted an integer literal longer than 1300 digits", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(huge)]))
    return area


def verify_dead_code() -> Area:
    area = Area("dead code")
    p = write_case(
        "dead_code",
        '''
        """module doc"""
        def f():
            """function doc"""
            print("before")
            return "ret"
            print("after")

        if False:
            print("bad-if-false")
        else:
            print("if-false-else")
        if True:
            print("if-true")
        else:
            print("bad-if-true-else")
        while False:
            print("bad-while")
        else:
            print("while-else")
        assert True
        print(f(), __doc__)
        ''',
    )
    assert_same_output(area, p)
    s = show(p)
    if s.returncode != 0:
        area.fail(f"{p.name}: --show failed: {s.stderr!r}", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(p)]))
    else:
        for absent in ["bad-if-false", "bad-if-true-else", "bad-while", "after", "assert True"]:
            if absent in s.stdout:
                area.fail(f"{absent!r} survived dead-code elimination", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(p)]))
        for present in ['"""module doc"""', '"""function doc"""']:
            if present not in s.stdout and present.replace('"""', "'") not in s.stdout:
                area.fail(f"{present} docstring not preserved in optimized source", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(p)]))
    return area


def verify_algebraic() -> Area:
    area = Area("algebraic")
    p = write_case(
        "algebraic",
        """
        def int_case():
            x = 5
            print(x + 0, 0 + x, x * 1, +x, -(-x))

        def param_case(s):
            return s + 0

        def float_case():
            s = -0.0
            print(s + 0)

        class Weird:
            def __add__(self, other):
                print("add-called")
                return 99

        int_case()
        try:
            print(param_case("x"))
        except Exception as e:
            print(type(e).__name__)
        float_case()
        print(Weird() + 0)
        """,
    )
    assert_same_output(area, p)
    s = show(p)
    if s.returncode != 0:
        area.fail(f"{p.name}: --show failed: {s.stderr!r}", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(p)]))
    else:
        if "'x' + 0" not in s.stdout and '"x" + 0' not in s.stdout:
            area.fail("non-int string addition was simplified despite not being a provable int", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(p)]))
        if "Weird() + 0" not in s.stdout:
            area.fail("custom object addition was simplified despite not being a provable int", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(p)]))
        if "x + 0" in s.stdout or "0 + x" in s.stdout or "x * 1" in s.stdout:
            area.fail("provable int identities were not simplified", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(p)]))
    return area


def verify_inlining() -> Area:
    area = Area("inlining")
    p = write_case(
        "inlining",
        """
        def f(a, b):
            return a + b

        x = 4
        y = 5
        print(f(x, 3))
        print(f(1, y))
        """,
    )
    assert_same_output(area, p)
    s = show(p)
    if s.returncode != 0 or "f(x, 3)" in s.stdout or "f(1, y)" in s.stdout:
        area.fail("safe module-level f(a,b) calls were not inlined", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(p)]))

    side_blocked = write_case(
        "inlining_blocked_side_effect",
        """
        def side():
            print("side")
            return 1

        def f(a):
            return a + 1

        print(f(side()))
        """,
    )
    assert_same_output(area, side_blocked)
    s = show(side_blocked)
    if s.returncode != 0 or "f(side())" not in s.stdout:
        area.fail("call with side-effect argument was inlined or --show failed", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(side_blocked)]))

    rebound_blocked = write_case(
        "inlining_blocked_rebound",
        """
        def f(a):
            return a + 1
        f = lambda a: a + 100
        print(f(1))
        """,
    )
    assert_same_output(area, rebound_blocked)
    s = show(rebound_blocked)
    if s.returncode != 0 or "f(1)" not in s.stdout:
        area.fail("rebound function was inlined or --show failed", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(rebound_blocked)]))

    before_def_blocked = write_case(
        "inlining_blocked_before_def",
        """
        print(f(1))
        def f(a):
            return a + 1
        """,
    )
    s = show(before_def_blocked)
    if s.returncode != 0 or "f(1)" not in s.stdout:
        area.fail("call before def was inlined or --show failed", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(before_def_blocked)]))

    dynamic_blocked = write_case(
        "inlining_blocked_dynamic",
        """
        def f(a):
            return a + 1
        code = "1"
        eval(code)
        print(f(2))
        """,
    )
    assert_same_output(area, dynamic_blocked)
    s = show(dynamic_blocked)
    if s.returncode != 0 or "f(2)" not in s.stdout:
        area.fail("module containing eval still inlined a function call or --show failed", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(dynamic_blocked)]))
    return area


def verify_localize_regressions() -> Area:
    area = Area("localize regressions")

    basic = write_case(
        "localize_basic",
        """
        def helper(x):
            y = x + 1
            return y

        def f(n):
            total = 0
            for i in range(n):
                total += abs(-i) + min(i, 3) + helper(i)
            return total

        print(f(8))
        """,
    )
    assert_same_output(area, basic)
    s = show(basic)
    if s.returncode != 0 or s.stdout.count("_opast_glb_") < 3 or " = abs" not in s.stdout or " = min" not in s.stdout or " = helper" not in s.stdout:
        area.fail("localize did not bind abs/min/module helper to _opast_glb temps", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(basic)]))

    shadowed_builtin = write_case(
        "localize_shadowed_builtin_anywhere",
        """
        def shadow():
            abs = lambda x: 99
            return abs(-1)

        def f(n):
            total = 0
            for i in range(n):
                total += abs(i)
            return total

        print(shadow(), f(4))
        """,
    )
    assert_same_output(area, shadowed_builtin)
    s = show(shadowed_builtin)
    if s.returncode != 0 or " = abs" in s.stdout:
        area.fail("localize bound abs despite a shadowing binding elsewhere", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(shadowed_builtin)]))

    blocked_cases = {
        "localize_nondirect_binding": """
            if True:
                G = 4
            def f(n):
                total = 0
                for i in range(n):
                    total += G
                return total
            print(f(3))
        """,
        "localize_binding_after_def": """
            def f(n):
                total = 0
                for i in range(n):
                    total += G
                return total
            G = 4
            print(f(3))
        """,
        "localize_global_decl": """
            G = 4
            def mut():
                global G
                G = 5
            def f(n):
                total = 0
                for i in range(n):
                    total += G
                return total
            print(f(3))
        """,
        "localize_module_loop_untouched": """
            G = 4
            total = 0
            for i in range(3):
                total += G
            print(total)
        """,
        "localize_for_iter_only_skipped": """
            DATA = (1, 2, 3)
            def f():
                total = 0
                for i in DATA:
                    total += i
                return total
            print(f())
        """,
    }
    for name, src in blocked_cases.items():
        p = write_case(name, src)
        assert_same_output(area, p)
        s = show(p)
        if s.returncode != 0 or "_opast_glb_" in s.stdout:
            area.fail(f"{name}: localize introduced a temp for a blocked case", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(p)]))

    zero_iter = write_case(
        "localize_zero_iteration",
        """
        G = 4
        def f():
            total = 0
            while False:
                total += G
            return total
        print(f())
        """,
    )
    assert_same_output(area, zero_iter)
    return area


def verify_inline_statement_regressions() -> Area:
    area = Area("inline-stmt regressions")

    basic = write_case(
        "inlinestmt_basic",
        """
        def helper(a, b):
            "doc"
            x = a + 1
            y = b + 2
            return x * y

        def f(v):
            if v < 0:
                return 0
            return helper(v, 3)

        print(f(int("2")))
        """,
    )
    assert_same_output(area, basic)
    s = show(basic)
    if s.returncode != 0 or "_opast_in_" not in s.stdout or "helper(2, 3)" in s.stdout:
        area.fail("statement-body inlining did not introduce _opast_in temps and remove the call", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(basic)]))

    side_effects = write_case(
        "inlinestmt_arg_order_once",
        """
        events = []
        def arg(v):
            events.append(v)
            return v
        def helper(a, b):
            x = a * 10
            return x + b
        def f():
            return helper(arg(1), arg(2))
        print(f(), events)
        """,
    )
    assert_same_output(area, side_effects)
    s = show(side_effects)
    if s.returncode != 0 or s.stdout.count("arg(1)") != 1 or s.stdout.count("arg(2)") != 1:
        area.fail("statement-body inlining did not preserve one left-to-right evaluation of side-effecting args", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(side_effects)]))

    blocked_cases = {
        "inlinestmt_keyword_blocked": """
            def helper(a):
                x = a + 1
                return x
            def f():
                return helper(a=2)
            print(f())
        """,
        "inlinestmt_unboundlocal_preserved": """
            def helper():
                x = x + 1
                return x
            def f():
                try:
                    return helper()
                except Exception as e:
                    return type(e).__name__
            print(f())
        """,
        "inlinestmt_expression_positions_blocked": """
            def helper(a):
                x = a + 1
                return x
            def wrap(v):
                return v * 2
            def f():
                total = 1
                total += helper(2)
                return wrap(helper(total))
            print(f())
        """,
        "inlinestmt_shadowed_free_name": """
            import math
            SCALE = math.floor(10.5)
            def helper(a):
                x = a + SCALE
                return x
            def f(v):
                SCALE = v + 90
                if v < 0:
                    return 0
                out = helper(v)
                return out + SCALE
            print(f(int("2")))
        """,
        "inlinestmt_oversized_body": """
            def helper(a):
                x1 = a + 1
                x2 = x1 + 1
                x3 = x2 + 1
                x4 = x3 + 1
                x5 = x4 + 1
                x6 = x5 + 1
                x7 = x6 + 1
                return x7
            def f():
                return helper(1)
            print(f())
        """,
    }
    for name, src in blocked_cases.items():
        p = write_case(name, src)
        assert_same_output(area, p)
        s = show(p)
        if s.returncode != 0:
            area.fail(f"{name}: --show failed", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(p)]))
        elif name == "inlinestmt_keyword_blocked" and "helper(a=2)" not in s.stdout:
            area.fail("keyword call was statement-inlined", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(p)]))
        elif name == "inlinestmt_unboundlocal_preserved" and "helper()" not in s.stdout:
            area.fail("read-local-before-assign helper was statement-inlined", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(p)]))
        elif name == "inlinestmt_expression_positions_blocked" and ("helper(2)" not in s.stdout or "helper(total)" not in s.stdout):
            area.fail("expression-position statement-body call was rewritten", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(p)]))
        elif name == "inlinestmt_shadowed_free_name" and "helper(v)" not in s.stdout:
            area.fail("helper with shadowed free name was statement-inlined", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(p)]))
        elif name == "inlinestmt_oversized_body" and "helper(1)" not in s.stdout:
            area.fail("helper with 7 assigns was statement-inlined", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(p)]))

    constant_shadow_cascade = write_case(
        "inlinestmt_constant_shadow_cascade",
        """
        import math
        SCALE = math.floor(10.5)
        def helper(a):
            x = a + SCALE
            return x
        def f(v):
            SCALE = 100
            out = helper(v)
            return out + SCALE
        print(f(int("2")))
        """,
    )
    assert_same_output(area, constant_shadow_cascade)
    return area


def verify_strength_interval_regressions() -> Area:
    area = Area("strength/interval regressions")

    strength = write_case(
        "strength_interval_basic",
        """
        def f():
            out = []
            for i in range(-5, 6):
                out.append((i % 8, i // 4, i ** 2))
            return tuple(out)
        print(f())
        """,
    )
    assert_same_output(area, strength)
    s = show(strength)
    if s.returncode != 0 or "i & 7" not in s.stdout or "i >> 2" not in s.stdout or "i * i" not in s.stdout:
        area.fail("power-of-two strength rewrites did not fire for proven int range target", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(strength)]))

    pow_cap = write_case(
        "strength_pow_node_cap",
        """
        def f(n):
            total = 0
            for i in range(n):
                total += (i + i + i + i + i + i) ** 2
            return total
        print(f(4))
        """,
    )
    assert_same_output(area, pow_cap)
    s = show(pow_cap)
    if s.returncode != 0 or "** 2" not in s.stdout:
        area.fail("pow-2 duplication ignored the node cap", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(pow_cap)]))

    abs_cases = write_case(
        "strength_abs_ranges",
        """
        def nonnegative(n):
            total = 0
            for i in range(n):
                total += abs(i)
            return total
        def maybe_negative():
            total = 0
            for i in range(-3, 3):
                total += abs(i)
            return total
        print(nonnegative(5), maybe_negative())
        """,
    )
    assert_same_output(area, abs_cases)
    s = show(abs_cases)
    if s.returncode != 0 or "total += i" not in s.stdout or "_opast_glb_" not in s.stdout:
        area.fail("abs interval rewrite did not remove only the non-negative range-target case", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(abs_cases)]))

    shadow_abs = write_case(
        "strength_shadowed_abs",
        """
        abs = lambda x: 100
        def f(n):
            total = 0
            for i in range(n):
                total += abs(i)
            return total
        print(f(3))
        """,
    )
    assert_same_output(area, shadow_abs)
    s = show(shadow_abs)
    if s.returncode != 0 or "total += i" in s.stdout:
        area.fail("shadowed abs was eliminated", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(shadow_abs)]))

    range_args = write_case(
        "strength_range_weird_args",
        """
        class Weird:
            def __index__(self):
                return 3
        def f(n):
            out = []
            for i in range(n):
                out.append((i + 0, i * 1, i % 8, i // 4, abs(i)))
            for j in range(True):
                out.append((j + 0, j % 2))
            for k in range(Weird()):
                out.append((k * 1, k // 2))
            return tuple(out)
        print(f(4))
        """,
    )
    assert_same_output(area, range_args)
    s = show(range_args)
    if s.returncode != 0 or "i + 0" in s.stdout or "i * 1" in s.stdout or "i & 7" not in s.stdout or "i >> 2" not in s.stdout or "abs(i)" in s.stdout:
        area.fail("range(n)/range(True)/__index__ int-target rewrites did not fire as expected", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(range_args)]))

    reused_as_str = write_case(
        "strength_target_reused_as_str",
        """
        def f():
            total = 0
            for i in range(3):
                total += i
            i = "x"
            return i + "y"
        print(f())
        """,
    )
    assert_same_output(area, reused_as_str)
    s = show(reused_as_str)
    if s.returncode != 0 or "return 'xy'" in s.stdout:
        area.fail("range target reused as str was treated as a stable int after the loop", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(reused_as_str)]))

    widening = write_case(
        "strength_widening_accumulator",
        """
        def f(n):
            acc = 1
            for i in range(n):
                acc = acc * 2 + i
            return acc
        print(f(40))
        """,
    )
    assert_same_output(area, widening)
    return area


def verify_cross_scope_constprop() -> Area:
    area = Area("cross-scope const-prop")

    branch = write_case(
        "constprop_cross_debug_branch",
        """
        DEBUG = False
        def f():
            if DEBUG:
                return "debug"
            return "release"
        print(f())
        """,
    )
    assert_same_output(area, branch)
    s = show(branch)
    if s.returncode != 0 or "if DEBUG" in s.stdout or "'debug'" in s.stdout or '"debug"' in s.stdout:
        area.fail("cross-scope DEBUG branch was not eliminated", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(branch)]))

    def_before = write_case(
        "constprop_cross_def_before_binding",
        """
        def f():
            try:
                return DEBUG
            except Exception as e:
                return type(e).__name__
        print(f())
        DEBUG = False
        """,
    )
    assert_same_output(area, def_before)
    s = show(def_before)
    if s.returncode != 0 or "return DEBUG" not in s.stdout:
        area.fail("def-before-binding module constant was propagated", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(def_before)]))

    local_shadow = write_case(
        "constprop_cross_local_shadow",
        """
        DEBUG = False
        def f():
            DEBUG = True
            return DEBUG
        print(f())
        """,
    )
    assert_same_output(area, local_shadow)
    s = show(local_shadow)
    if s.returncode != 0 or "print(True)" not in s.stdout:
        area.fail("local shadow did not preserve local constant propagation semantics", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(local_shadow)]))
    elif "def f():\n    return False" in s.stdout:
        area.fail("module constant propagated through a local shadow", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(local_shadow)]))

    disabled = {
        "constprop_cross_global_decl": """
            DEBUG = False
            def mut():
                global DEBUG
                DEBUG = True
            def f():
                return DEBUG
            print(f())
        """,
        "constprop_cross_comp_target_shadow": """
            DEBUG = False
            def f():
                return [DEBUG for DEBUG in (1, 2)]
            print(f())
        """,
        "constprop_cross_del_disables": """
            DEBUG = False
            del DEBUG
            def f():
                return DEBUG
            try:
                print(f())
            except Exception as e:
                print(type(e).__name__)
        """,
    }
    for name, src in disabled.items():
        p = write_case(name, src)
        assert_same_output(area, p)
        s = show(p)
        if s.returncode != 0 or (name != "constprop_cross_comp_target_shadow" and "DEBUG" not in s.stdout):
            area.fail(f"{name}: --show failed or lost DEBUG name unexpectedly", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(p)]))
        if name == "constprop_cross_comp_target_shadow" and "for DEBUG in" not in s.stdout:
            area.fail("comprehension target shadow was substituted", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(p)]))

    nested = write_case(
        "constprop_cross_nested_class_lambda",
        """
        VALUE = 7
        class C:
            body = VALUE
            def method(self):
                def inner():
                    return VALUE
                return inner()
        lam = lambda: VALUE
        print(C.body, C().method(), lam())
        """,
    )
    assert_same_output(area, nested)
    s = show(nested)
    if s.returncode != 0 or "body = VALUE" in s.stdout or "return VALUE" in s.stdout or "lambda: VALUE" in s.stdout:
        area.fail("nested def/class method/lambda did not receive cross-scope module constant", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(nested)]))
    return area


def verify_dynamic_fallback() -> Area:
    area = Area("dynamic fallback")
    samples = {
        "dyn_eval": "x = 1 + 2\ncode = 'x'\neval(code)\nprint(x)\n",
        "dyn_exec": "x = 1 + 2\ncode = 'x = 9'\nexec(code)\nprint(x)\n",
        "dyn_globals": "x = 1 + 2\nprint(globals()['x'])\n",
        "dyn_locals": "def f():\n    x = 1 + 2\n    locals()\n    return x\nprint(f())\n",
        "dyn_vars": "x = 1 + 2\nvars()\nprint(x)\n",
        "dyn_compile": "x = 1 + 2\ncompile('1+1', '<x>', 'eval')\nprint(x)\n",
        "dyn_import": "x = 1 + 2\n__import__('math')\nprint(x)\n",
        "dyn_star": "from math import *\nx = 1 + 2\nprint(x)\n",
    }
    for name, src in samples.items():
        p = write_case(name, src)
        s = show(p)
        if s.returncode != 0:
            area.fail(f"{name}: --show --no-run failed: {s.stderr!r}", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(p)]))
            continue
        original = p.read_text(encoding="utf-8").strip()
        shown = s.stdout.strip()
        if "1 + 2" not in shown:
            area.fail(f"{name}: dynamic scope was optimized instead of left untouched\nshown={shown!r}", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(p)]))
        if name == "dyn_star" and "from math import *" not in shown:
            area.fail("import * dynamic file was unexpectedly rewritten", command_text([PYTHON, "-m", "opast", "--show", "--no-run", str(p)]))
    return area


def verify_cli() -> Area:
    area = Area("CLI")
    p = write_case(
        "cli_args",
        """
        import sys
        print("argv", sys.argv[1:])
        print(1 + 2)
        """,
    )
    assert_same_output(area, p, "alpha", "--beta", "3")
    out = CASE_DIR / "cli_args_optimized.py"
    cp = run([PYTHON, "-m", "opast", "--no-run", "--show", "--report", "-o", str(out), str(p)])
    if cp.returncode != 0:
        area.fail(f"combined CLI flags failed: {cp.stderr!r}", command_text([PYTHON, "-m", "opast", "--no-run", "--show", "--report", "-o", str(out), str(p)]))
    else:
        if "print('argv', sys.argv[1:])" not in cp.stdout and 'print("argv", sys.argv[1:])' not in cp.stdout:
            area.fail("--show did not print optimized source", command_text([PYTHON, "-m", "opast", "--no-run", "--show", str(p)]))
        if "constant-folding" not in cp.stderr or "opast:" not in cp.stderr:
            area.fail("--report did not include pass stats on stderr", command_text([PYTHON, "-m", "opast", "--no-run", "--report", str(p)]))
        if not out.exists() or "print" not in out.read_text(encoding="utf-8"):
            area.fail("-o did not write optimized source", command_text([PYTHON, "-m", "opast", "--no-run", "-o", str(out), str(p)]))
    return area


def main() -> int:
    areas = [
        verify_jit(),
        verify_benchmark(),
        verify_constant_propagation(),
        verify_licm(),
        verify_cse_lencache(),
        verify_comprehension_to_map(),
        verify_unused_elimination(),
        verify_greatest_fixpoint_analysis(),
        verify_dedynamize_eval(),
        verify_dedynamize_exec(),
        verify_dedynamize_globals_vars(),
        verify_dedynamize_attributes(),
        verify_dedynamize_cascade_rollback(),
        verify_constant_folding(),
        verify_dead_code(),
        verify_algebraic(),
        verify_inlining(),
        verify_localize_regressions(),
        verify_inline_statement_regressions(),
        verify_strength_interval_regressions(),
        verify_cross_scope_constprop(),
        verify_dynamic_fallback(),
        verify_cli(),
    ]
    failed = False
    for area in areas:
        status = "FAIL" if area.failures else "PASS"
        print(f"{status}: {area.name}")
        for failure in area.failures:
            print(f"  - {failure}")
        for repro in dict.fromkeys(area.repros):
            print(f"    repro: {repro}")
        failed = failed or bool(area.failures)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
