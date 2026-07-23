# opast

> **opast**("OPtimizing AST")— PyPI 包 / import 包 / CLI 命令均为 `opast`;GitHub 仓库名 **pyOpAst**。

基于 Python `ast` 的源码优化器。对脚本做静态优化后,用**运行 opast 的解释器**(默认 CPython)直接执行优化后的代码。

## 优化策略(第一批)

| Pass | 说明 |
| --- | --- |
| 去动态化 `de-dynamize` | 把"无意义的动态代码"展开为等价静态代码:模块顶层的 `eval('<常量表达式>')` → 表达式本身、`exec('<常量语句>')` → 语句展开、`globals()['x'] = v` / `vars()['x'] = v` → `x = v`;任意作用域的 `getattr(obj, 'attr')` → `obj.attr`、语句形式的 `setattr(名字, 'attr', v)` → `名字.attr = v`、`delattr(名字, 'attr')` → `del 名字.attr`。**全有或全无**:仅当改写后整个模块零动态构造残留、且所依赖的内置名(`getattr` 等)未在任何位置被绑定时才提交,否则整体回滚——因为残留的 `exec`/`eval`/`import *` 可能在运行期重绑定这些内置名。成功后模块变为纯静态,解锁其余全部 pass(含内联)。函数内的 `eval`/`exec` 永不展开(locals 快照语义不等价);`globals()['x']` 读取不改写(`KeyError` vs `NameError`);eval 字符串含 walrus、exec 字符串含 `from __future__ import` 或绑定 `__debug__` 均拒绝。 |
| 常量折叠 `constant-folding` | 折叠常量算术/字符串/比较/布尔短路/常量下标,如 `1+2*3 → 7`、`"a"*3 → "aaa"`、`True and x → x`。会在运行期抛异常的表达式(`1/0`、`1<"a"`)不折叠,保留原有异常;巨大结果(大整数、超长字符串)跳过。护栏契约是"优化期不做无界计算、不产出巨型字面量",**子表达式允许部分折叠**:如 `10 ** 10 ** 10 → 10 ** 10000000000`(内层在护栏内故折叠,外层指数超限故保留,运行期语义不变——正是内层先折叠,外层护栏才能看到真实指数)。 |
| 代数化简 `algebraic` | `x+0`、`x*1`、`x-0`、`x**1`、`x//1`、`x<<0`、`x\|0`、`+x`、`-(-x)` 等,外加**强度削减**:`E % 2^k → E & (2^k-1)`、`E // 2^k → E >> k`(Python 的 `//` 是地板除、int 是无限二补,两条对**全体** int 成立)与 `E ** 2 → E * E`(仅限小的纯 int 表达式——纯且确定性,重复求值不可观察,省下的 `pow` 远大于代价);以及**区间分析**支撑的 `abs(E) → E`(流不敏感不动点+加宽推出每个 proven-int 名字的取值区间,`E` 可证非负才改写,另要求 `abs` 全模块未绑定)。**仅**对能静态证明为纯 `int` 的局部变量生效(逐函数最小不动点推断;`range` 可信时 `for i in range(...)` 的目标也算 proven int),避免自定义 `__add__`、`-0.0`、`bool` 等语义陷阱;除 `E**2` 外被化简的表达式本身永远保留、不复制不丢弃,副作用与异常不变。 |
| 常量传播 `const-prop` | 变量在整个作用域内只绑定一次、且绑定是作用域体**顶层**的常量赋值(含扁平元组解包 `x, y = 1, 2`:两侧等长、左侧全为普通名字、无 Starred,每个名字独立判定)时,把赋值语句**之后**的使用直接代入常量("先用后赋"的 NameError 行为保留;不进入嵌套作用域,闭包引用继续读原变量)。模块层额外要求全模块零动态构造(被调用函数的 `exec` 可能改写全局)且该名字无 `global` 重绑定;函数层排除 `nonlocal`。超过 128 字符的 str/bytes 不复制。赋值语句本身保留,由 `unused` 清理。**跨度传播**(多次绑定的名字):同一语句块内语句严格顺序执行,故从 `x = <常量>` 起、到其后第一条可能重绑 `x` 的语句为止(语句子树内对 `x` 的任何绑定、以及嵌套作用域内 `global`/`nonlocal x` 声明——调用该闭包即可重绑——都算),之间对 `x` 的读取代入常量;全程不重绑 `x` 的复合语句整条(含循环体——每次迭代读到的都是该常量)代入,其内部块再递归开启自己的局部跨度;单名字赋值 `x = <expr>` 的右值先于存储求值,故 `x = x + 1` 的右值也可代入(值内 walrus 除外)。**复制传播**(仅函数作用域——内联临时变量所在之处;模块层的 `getattr = my_func` 式别名往往是有意的重命名,不动):`y = x` 且两侧均为本作用域普通名、`x` 在本作用域有绑定(读全局/内置的 `y` 冻结了值、`x` 是活读取,不等价)时,二者都未重绑期间 `y` 的读取改读 `x`;赋值本身保留,由 `unused` 清理——语句级内联的实参临时变量由此消失。被 `global`/`nonlocal` 声明的名字不参与跨度/复制传播;嵌套作用域照旧不进入。**跨作用域**:模块常量还会传播进所在顶层语句位于绑定**之后**的函数/lambda/类(作用域内代码只能在其顶层 `def`/`class` 执行后运行,即绑定之后;单次绑定+无 `global` 声明保证永不重绑)——函数里的 `if DEBUG:` 由此变成可消除的死分支;自己绑定该名字的作用域(参数、赋值、推导式目标,按保守的 bound_names)连同其内部整体排除,动态作用域照常跳过。 |
| 循环闭式折叠 `loop-fold` | `for i in range(<int 常量>)` 且循环体只有纯 int/bool 算术——单名字赋值、增量赋值、`if`/`elif`/`else`、`pass`,表达式限于常量/名字/算术/比较/布尔/条件表达式,**无任何调用**——时,优化期**精确模拟**整个循环,并用最终值的常量赋值替换循环:每个实际被存储的名字一条(按首次存储顺序),迭代 ≥1 时含循环变量终值,零迭代整体删除(`range(常量)` 求值是纯的)。初值取紧邻循环前连续的 `名 = <int/bool 常量>` 赋值(原赋值保留,函数内的死存储由 `unused` 清理);模拟中读到未知名字、任何运行期异常(除零、负移位)、步数超预算(200k,预先按迭代数×体积粗筛)或数值超 4096 位即放弃、循环原样保留。模拟执行的就是运行期同一份计算,结果逐位一致——语义精确,且成功折叠即证明循环不可能抛异常,故 try/with 内、模块层、`global` 声明名都允许(体内无调用,循环期间无人能观察中间态;线程并发观察不在语义契约内,与其他 pass 一致)。整个 pass 由 `range` 内置门控(全模块零动态构造且 `range` 未绑定);类体不做(元类 `__prepare__` 映射可观察逐迭代存储)。嵌套常量循环自底向上折叠:内层折成常量后,外层往往在下一轮不动点迭代中跟着折叠。 |
| 死代码消除 `dead-code` | `return/raise/break/continue` 之后的同块语句;常量条件 `if`/`while`(含 `else` 语义);`assert True`;无用的常量表达式语句(docstring 保留)与多余 `pass`。 |
| 循环不变量外提 `licm` | 把循环体(及 `while` 测试)中不随迭代变化的 int 表达式提到循环前的 `_opast_inv_N = <expr>` 临时变量。外提改变求值时机(零迭代循环甚至改变是否求值),所以仅限**可证明纯且不抛异常**的表达式:proven-int 名字(`range` 可信时含 `for i in range(...)` 目标,其在循环体内视为必定已绑定)+ int 常量的算术组合,`//`/`%` 要求非零 int 常量除数、移位要求 0..256 常量位数;名字须在循环子树内零绑定(含嵌套函数的 `global`/`nonlocal` 声明),且经直线支配分析确认循环前**必定已绑定**(`del`、`except as`、条件绑定都会取消资格)。不进入循环内嵌套作用域;仅函数作用域(模块全局可能被调用的函数改写)。 |
| 公共子表达式消除 `cse` | 同一语句块内重复出现(≥2 次)的可证明纯 int 表达式(与 LICM 同一判定,含新鲜容器的 `len()`)合并为 `_opast_cse_N` 临时变量,插在首次出现语句之前并替换所有出现。纯且不抛异常 → 允许投机求值,无需逐次支配,只要求名字在物化点已绑定、首末次出现之间零重绑定。按最大表达式匹配(子表达式不重复计数);仅函数作用域,不跨语句块、不进嵌套作用域。 |
| 未使用消除 `unused` | 删除:未使用的 import(始终保留 `__future__` 与 `import *`;`import a, b` 部分使用时只留用到的);未使用的函数定义(无装饰器、默认值/注解无副作用——内联后残留的 helper 由此清掉);函数内未使用的局部赋值(右值纯的整句删除,可能有副作用的降级为裸表达式语句,只去掉死存储)。**模块级变量赋值刻意不删**——模块全局是可观察面(`import module; module.x`、bench 读 `RESULT`)。`__all__` 中的字符串视为已使用,`__all__` 非字面量时模块级删除整体停用。"未使用"按整棵作用域子树判定(含闭包读取);`x += 1` 算读取。模块存在任何动态构造时本 pass 整体跳过(`eval`/帧内省可能观察被删名字)。注意:删除 import 会跳过被导入模块的副作用。删除以 `pass` 占位,下一轮死代码清理。 |
| 全局名局部化 `localize` | 循环体(及 `while` 测试)里逐迭代读取的稳定全局名,在循环前绑定为 `_opast_glb_N` 局部变量并替换循环内读取:LOAD_GLOBAL(模块字典+内置字典查找)变 LOAD_FAST。两类合格名字(均要求全模块零动态构造):**内置名**——全树任何作用域零绑定且确为 `builtins` 属性(永远已绑定、永远是内置);**稳定模块全局**——模块区域恰好一次绑定、且是**直接**顶层语句(`if`/`try` 里的条件绑定不算)、任何作用域无 `global` 声明、绑定语句位于该循环所在顶层 `def`/`class` 之前(调用只能发生在顶层 def 执行之后,故读取时必定已绑定、且永不可能重绑)。只在函数作用域内做(模块层临时量还是全局);不进嵌套作用域与推导式;`for` 的 iter 位读取不计入触发条件(只求值一次)。排在流水线最后,让 inline/comp-to-map 先认领它们理解的名字;`--jit` 时跳过 numba 白名单内置名(`abs`/`min`/`range` 等换成局部名会让候选函数过不了 numba 类型化)。 |
| 下标循环转直接迭代 `range-to-iter` | `for i in range(len(x)):` 且 `x` 是**新鲜序列**时消除逐迭代下标查找:`i` 仅用作 `x[i]` 下标 → `for <元素临时名> in x:`(直接迭代,计数器整个消失);`i` 另有用途 → `for i, <元素临时名> in enumerate(x):`(`i` 的绑定逐迭代、终值、零迭代不绑定都与原来一致)。只替换循环体内**恰为** `x[i]` 的读取(本作用域层;嵌套作用域、`else` 块、`x[i+1]` 这类复合下标原样保留——enumerate 形态下 `x` 与 `i` 都还在,残留照常正确;至少替换一处才改写)。`x` 须过 `fresh_container_names`(单次绑定、零逃逸零变异零重绑——循环期间长度与元素都不可能变,`x[i]` 才恒等于迭代元素)**且**绑定值可证为序列:list/tuple 字面量、列表推导式、`list`/`tuple`/`sorted`/`range`/`str`/`bytes` 构造调用(set/dict 排除:迭代顺序不可下标对应);循环目标为普通名字且循环子树内零绑定;直接迭代形态额外要求 `i` 在作用域内此外零出现。门控:`len`+容器构造器门(同 len 缓存)加 `range` 门,enumerate 形态还要 `enumerate` 全模块未绑定;函数与模块作用域,类体不做。排在 loop-to-comp 之前——`for i in range(len(x)): out.append(x[i]*2)` 可级联成推导式。 |
| 累加循环转推导式 `loop-to-comp` | `x = []` 紧随其后 `for v in it: x.append(f(v))`(允许嵌套 `for` 与卫式 `if` 链,唯一的最内层 append)→ `x = [f(v) for v in it ...]`;`x = set()` + `.add` 同形转集合推导式(额外要求 `set` 全模块未绑定)。推导式走 LIST_APPEND/SET_ADD 专用字节码,替代逐迭代的属性查找+方法调用,并接入下游 comp-to-map。安全条件:种子赋值**紧邻**循环且绑定普通名字;`x` 在循环子树内的**唯一**出现就是 append/add 的接收者(循环既观察不到也重绑不了 `x`;新鲜字面量接收者保证方法就是真的 `list.append`/`set.add`)、`x` 全模块无 `global`/`nonlocal` 声明、模块零动态构造(元素/条件/iter 表达式里的调用都无法背后重绑 `x`);循环目标名改写后变为推导式局部,故必须此后死亡——在所在作用域整个子树内(含嵌套作用域)循环之外零出现,后续闭包读取、复用同名的后续循环、参数同名都取消资格;循环子树无嵌套作用域(def/lambda/class)、无 yield/await/walrus,`for` 无 `else`;不在 try/with 内改写(中途异常时语句版留下的半成品 `x` 可被 handler/`__exit__` 观察,推导式版不会)。零迭代两版都得到空容器(种子字面量从未逃逸,身份无从比较);类体不做(元类 `__prepare__` 可观察逐迭代对 `x` 的读取)。 |
| 推导式转 map/filter `comp-to-map` | `(f(x) for x in it)` → `map(f, it)`、`(x for x in it if g(x))` → `filter(g, it)`、`(f(x) for x in it if g(x))` → `map(f, filter(g, it))`;列表/集合推导式同形则包一层 `list(...)`/`set(...)`。安全条件:① 名字查找时机——推导式逐迭代查 `f`,`map` 构造时捕获,故 `f` 须不可重绑定:内置可调用名(全模块零绑定)或(仅生成器表达式)模块级单绑定的 def/class/import/赋值、定义在使用点之前、未被外层函数/类/**外层推导式目标**遮蔽;列表/集合推导式按用户规格仅限内置函数。② 对象身份——generator 有 `send/throw/close/gi_frame`,map 没有,故生成器表达式仅在**纯迭代消费位置**改写(`for` 的 iter 位、`sum/list/sorted/any/...` 等只迭代参数的内置消费者的位置实参、`"字面量".join(...)`),裸赋值跳过;list/set 推导式产物类型不变,任意位置可改。③ 引入的 `map/filter/list/set` 及消费者名本身须全模块未遮蔽,模块零动态构造。 |
| 简单函数内联 `inline` | 两种形态。**表达式体**:模块级 `def f(...): return <expr>`(可带 docstring;默认值须为常量),任意调用位置替换为代入后的表达式,实参须为常量或普通名字(保证副作用求值次数与顺序不变)。**直线语句体**:docstring + 至多 6 条单目标名字赋值 + 末尾 `return <expr>`(无分支/循环);仅改写函数体内 `x = f(...)`、`return f(...)`、裸 `f(...)` 三种"调用即整个值"的语句位:实参按调用顺序逐个求值进 `_opast_in_<site>_<形参>` 临时变量(因此**任意表达式实参**都允许,仅限位置实参),函数体赋值重命名后按序展开,原语句保留重命名后的返回表达式;函数局部量对调用方本不可见,重命名临时变量是唯一可观察差异(traceback 少一帧,属已文档化限制)。共同要求:整个模块无动态构造、`f` 只被 `def` 绑定一次且未被 `global` 声明、调用点行号在定义之后、`f` 及函数体自由名在调用点未被局部遮蔽、不含 yield/await/walrus/lambda/推导式、无先读后赋的局部名、非递归。`def` 本体保留(闲置后由 `unused` 清理)。 |

此外 LICM 与 CSE 支持**新鲜容器的 `len()` 缓存**:`len(x)` 可参与外提/合并,当且仅当 `x` 在函数内只绑定一次、右值是字面量/推导式/内置构造器(新鲜对象),且全程未逃逸——仅允许 `len(x)`、下标**读取** `x[...]`、`for ... in x`、裸 `if x:`/`while x:` 四类使用;任何传参、别名赋值、属性/方法访问(含 `x.append`)、下标写入、放进容器、return 等都取消资格(逃逸的引用意味着任何后续调用都可能变动容器)。另要求全模块无动态构造、`len` 及容器构造器名未在任何位置被绑定。自定义对象上的 `len`/重复调用**明确不缓存**(`__len__` 可以有副作用,语义不等价)。

流水线按「去动态化 → 折叠 → 常量传播 → 代数化简 → 循环闭式折叠 → 死代码 → 下标循环转直接迭代 → 循环不变量外提 → 公共子表达式消除 → 未使用消除 → 内联 → 累加循环转推导式 → 推导式转 map → 全局名局部化」迭代到不动点(默认最多 8 轮;内联在前——能内联进推导式的简单函数比 map 更快,map 转换兜底接住不可内联的函数与内置;局部化垫底——让 inline/comp-to-map 先认领名字,comp-to-map 引入的 `map`/`filter` 由下一轮局部化接住):折叠喂给传播(`x = 2+3` 先折成 `x = 5` 再代入),传播喂给循环闭式折叠(`range(n)` 变 `range(1000)`)与代数化简、死代码(`if x:` 变 `if True:`),死代码释放最后一处使用后未使用消除接手,内联靠后——下一轮会折叠内联出的表达式、删除因此闲置的 helper 定义;loop-to-comp 排在内联之后、comp-to-map 之前,产出的推导式同轮即被 comp-to-map 接住。循环折叠出的常量又喂给下一轮的传播/内联,常量热核会逐轮向外坍缩(见 `loopfold` 负载)。

## 动态代码回退策略

优化以「作用域区域」为单位(模块顶层 / 函数体 / lambda / 类体,嵌套定义的装饰器、默认值、注解归外层区域)。某区域出现以下任意一种即视为动态、该作用域(连同其内部所有嵌套作用域)整体跳过、不做任何变换:

- 名字 `eval` `exec` `compile` `globals` `locals` `vars` `__import__` `breakpoint` 的任何出现(纯基于名字,故意过度保守);
- 属性名 `eval` `exec` `globals` `locals` `vars` `_getframe` `f_locals` `f_globals` `settrace` `import_module` `currentframe` 等;
- `from m import *` 或 `from builtins import eval` 之类导入。

模块顶层被污染则整个文件不优化;**内联**额外要求全模块零动态构造(任何位置的 `exec`/`eval` 都可能改写模块全局名)。

## 使用

```powershell
pip install -e .          # 或临时: $env:PYTHONPATH = "src"

opast script.py a b c     # 优化并运行(等价 python -m opast script.py a b c)
opast --show --no-run script.py        # 只看优化结果
opast --report script.py               # 运行并输出每个 pass 的统计(stderr)
opast -o optimized.py --no-run script.py
opast -c "print(sum(i for i in range(10)))" arg1   # 直接运行代码串(仿 python -c;sys.argv[0] 为 '-c')
opast --disable inline,licm script.py  # 跳过指定 pass(名字见 --help;'jit' 也可禁)
opast --opt-imports script.py          # 连同脚本目录下被导入的纯 Python 模块一起优化
opast --opt-imports-under src script.py   # 指定额外目录(可重复;与 --opt-imports 可并用)
```

### `--opt-imports`:优化被导入的模块(选择加入)

通过 `sys.meta_path` 钩子拦截源码位于指定目录下的模块导入,编译前先过一遍完整优化流水线(`--jit`/`--disable`/`--max-iterations` 同样作用于这些模块;`--report` 会在导入时逐模块输出统计)。默认范围是入口脚本所在目录及子目录(`-c` 时为当前目录);stdlib、site-packages 与 C 扩展不受影响,opast 自身被排除。

- **不污染 `__pycache__`**:优化后的字节码绝不会写入标准 pyc 缓存(否则之后不经 opast 的 `python` 运行会误加载优化版)。优化结果进独立的内容寻址缓存(`%TMP%/opast-imports`,按源码+版本+选项寻址),二次导入零优化开销;`OPAST_NO_IMPORT_CACHE=1` 可绕过。
- **⚠️ 语义边界**:库模块的全局可被外部改绑(`mod.f = patched`、`unittest.mock.patch`),这会让依赖模块内绑定稳定性的 pass(内联、常量传播、comp-to-map)静默失效——**会被运行期 monkeypatch 的模块不要开启本功能**。这是它做成选择加入的原因。
- **模块顶层整体视为公开接口**:库模块内"未被自己使用"的函数定义与 import 恰恰是给导入方消费的(`mod.calc()`、re-export),因此对被导入模块强制停用 `unused` pass(入口脚本不受影响)。
- 被优化模块内的 traceback 行号按优化后布局显示(文件路径仍指原 `.py`);优化失败的模块自动回退为原样导入,绝不阻断导入。
- Python API:`from opast.importhook import install, uninstall`。

基准测试(优化后 vs 纯 CPython,同解释器内计时、GC 关闭、校验两版本计算结果一致):

```powershell
python -m opast.bench            # 全部内置负载,每变体 best-of-3
python -m opast.bench -r 5 inline dedynamize   # 指定负载与重复次数
python -m opast.bench --list     # 列出内置负载
python -m opast.bench --jit daily      # 连同 jit pass 一起测(需 numba;预热轮吸收 import+编译)
python -m opast.bench my_hot_script.py         # 也可测任意脚本(可选定义 RESULT 供校验)
```

内置负载:`inline`(小函数热循环)、`inlinestmt`(多语句函数体的语句级内联)、`algebra`(可证明 int 的恒等式噪声)、`strength`(for-range 计数器上的强度削减与 `abs` 消除)、`dedynamize`(热循环里的常量 `eval`/`getattr`)、`licm`(热循环里的不变量表达式)、`lencache`(热循环里新鲜列表的 `len()`)、`rangeiter`(新鲜列表上的下标循环转直接迭代/enumerate)、`looptocomp`(append 累加循环转推导式)、`loopfold`(常量边界纯 int 热核的优化期折叠与向外坍缩)、`comptomap`(内置函数上的推导式转 map/filter)、`localize`(热循环里的内置名/模块全局名局部化)、`mixed`(内联→折叠级联+死代码)、`daily`(日常风格的订单结算日报,单个脚本同时覆盖全部 pass 的实用测试点)、`jitlazy`(变量边界数值核,`--jit` 下验证运行期 lazy 触发,普通模式预期 ~1x)、`control`(无可优化项,预期 ~1.00x,验证零回归)。注意:纯常量折叠类优化(`1+2`、`"a"*3`)CPython 编译器自己也会做,opast 的运行时收益主要来自 CPython 不做的部分——内联、需类型证明的代数化简、去动态化。

Python API:

```python
from opast import optimize_source, optimize_file, run_path, run_source, PASS_NAMES

result = optimize_file("script.py")
print(result.source)             # 优化后源码 (ast.unparse)
print(result.report.summary())   # 统计
run_path("script.py", argv=("--flag",))   # 优化 + 以 __main__ 运行
run_source("print(1 + 1)")                # 直接从源码字符串优化并运行

# 所有入口都支持 disable:可迭代对象或逗号分隔字符串,关闭指定 pass;
# 合法名字见 PASS_NAMES(另接受 "jit"),未知名字抛 ValueError。
optimize_file("script.py", disable="inline,licm")
run_source("x = 1 + 2", disable=("constant-folding", "const-prop"))
```

IPython / Jupyter(cell magic):

```
%load_ext opast

%%opast --report --disable licm
total = 0
for i in range(50_000):
    total += i * 2
total          # 末尾表达式照常显示、照常绑定到 _
```

选项与 CLI 一致(`--show/--report/--no-run/--jit/--disable/--max-iterations`);cell 在用户命名空间执行,赋值跨 cell 保留。**注意**:优化分析以单个 cell 为"模块"视角,其他 cell 里定义的名字对分析不可见——若之前的 cell 重绑过内置名(如 `range = ...`),依赖模块级判定的 pass(内联、去动态化、comp-to-map)可能误判,请对相关 cell 用 `--disable` 或不加 magic;cell 内不能混用其他 magic / `!` 转义(无法解析)。

## 实验性:`--jit`(numba 加速,默认关闭)

```powershell
pip install -e .[jit]          # 可选依赖(numba ≥0.65 已支持 Python 3.14)
opast --jit hot_script.py
```

实测(numba 0.65 / CPython 3.14.2,800 万次迭代的数值函数):纯 Python 单次调用 1.22s;`--jit` 首调 0.79s(含编译,已反超),**稳态单次 0.011s(≈110x)**。注意 numba 有每进程约 1s 的固定开销(import ~0.6s + 首调编译 ~0.5s,`cache=True` 的磁盘缓存可部分摊薄),数值热代码本身不足一两秒的脚本开 `--jit` 反而更慢。`OPAST_JIT_DEBUG=1` 可在 stderr 查看每个函数的编译失败/回退原因与 lazy 触发原因;静态热点启发只认**常量**循环边界(`range(8_000_000)` 命中),**变量边界**(`range(n)`)走下述 lazy 路径。

**变量边界的运行期 lazy 编译**:过白名单但边界只有运行期才知道的函数(`for i in range(n)`、`while i < n`)不再直接放弃,而是打上 `maybe_njit_lazy` 装饰:每次调用照常走纯 Python 并累积观察证据,满足任一触发条件才编译——① **规模触发**:静态侧能定位"边界来自第 k 个位置参数"时,调用实参 ≥ `OPAST_JIT_LAZY_BOUND`(默认 10000,与静态热点阈值一致)立即编译;② **单次耗时** ≥ 0.1s;③ **调用量**:≥ 10 次且累计 ≥ 0.3s。触发的那次调用本来就跑了纯 Python,其结果直接充当首调对照校验的期望值——观察、编译、校验在同一次调用内完成(且校验用的就是真实热参数,回绕最容易当场暴露),之后与 eager 路径共用同一套守卫分发器与永久回退。观察计数按代码身份进程级持久(bench 反复 exec 不清零);lazy 候选须无同伴调用(裸 Dispatcher 别名在装饰时解析,与延迟编译不相容)。配套地,**numba 的 import 也推迟到首次触发编译时**——lazy 候选始终没热起来的脚本连 0.6s 的 import 都不付。触发那次调用会付 ~0.5s 同步编译成本;交互式场景可调高 `OPAST_JIT_LAZY_BOUND` 或 `OPAST_DISABLE_JIT` 全关。

流水线收敛后运行一次性 `jit` pass:对**未被内联**的模块级函数,若同时满足——

- **热点启发**:含嵌套循环,或某循环迭代次数可静态估算 ≥ 10000(`range(常量)` / `while name < 常量`);
- **numba 白名单**:仅数值常量与 int/float 算术、`if`/`while`/`for range()`/`return`、调用仅限 `range/abs/min/max/round/divmod/int/float/bool`、`math.*` 与**其他白名单候选函数**(见下)、名字仅限参数和局部变量;无字符串/容器/下标/属性(math 除外)/闭包/全局读写/try/yield;
- 模块无动态构造、函数名只绑定一次且未被 `global` 声明——

则加上 `@_opast_jit.maybe_njit` 装饰(注入 `import opast.jitsupport`)。

**njit 间调用**:候选函数之间允许互相调用——候选资格按不动点收缩(调用了非候选者即出局),调用环(自递归/互递归)整体剔除(numba 递归支持不稳定);热函数作为种子,沿调用边把被调的冷候选一并选入(调用者的 nopython 编译需要它们)。实现上,调用者不能直接编译原函数体(运行期被调名指向的是回退**包装器**,numba 无法类型化),因此为每个调用了同伴的函数生成改写副本 `_opast_jitsrc_F`(同伴调用重定向到 `_opast_njit_G` 裸 Dispatcher 别名),编译走副本、回退走原 def(其调用命中包装器,坏类型时逐层安全回退);被调者编译失败时别名为 None,调用者首调时同样降级为纯 Python。

**热循环外提(outlining)**:热点数值代码往往不是现成的独立函数,而是写在模块顶层或混杂函数(如带 `print`/IO)内部的循环。第二阶段把这类「热 + 白名单兼容」的 `for`/`while` 循环外提成新的模块级函数 `_opast_jit_loop_N`(同样加 `maybe_njit`),原循环替换为 `out1, out2 = _opast_jit_loop_N(in1, in2)`:

- **入参** = 循环用到、且循环前**必定已绑定**的名字(与 LICM 同一套直线支配扫描),以及作用域内从未绑定的外部只读名(调用时捕获一次即可——白名单禁止调用用户代码,循环期间无人能改绑它们);读取**条件绑定**的局部名则整体拒绝(改成函数局部变量会把可行读取变成 `UnboundLocalError`);
- **出参** = 循环内有绑定且此后可观察的名字(函数内看后续使用;模块层全部全局名都是可观察面)。出参必须同时是入参(零迭代时原值传入原值返回,语义不变),或可证明每次运行必定被存储:常量 `range` 且迭代 ≥1 次的 `for` 目标、无 `break`/`continue` 时每迭代无条件赋值的名字;
- 循环内纯临时变量需通过「先存后读」的迭代内顺序证明;
- 直接拒绝:循环内含 `return`、带 `orelse`、位于 `try`/`with` 内(循环中途抛异常时单次写回不等价于逐迭代更新,handler 可能观察到循环前的旧值)、触及作用域 `global`/`nonlocal` 声明名;
- 组装出的函数最后仍要过与整函数相同的 `_is_hot` + `_numba_compatible` 双闸;类型不合适时由运行期分发器永久回退纯 Python 版兜底(回退版与原循环语义一致)。

外提新增的选择加入代价(int64 回绕之外):循环变量的写回从逐迭代变为循环结束后一次,且中途抛出的异常传播时外层变量保持循环前的值——两者对「循环期间无法运行任何用户代码」的单线程程序、且不在 `try`/`with` 内时均不可观察。

运行期分层降级:无 numba / 版本不兼容 / 设置了 `OPAST_DISABLE_JIT` → 原函数原样运行;编译或调用时抛任何 numba 异常 → **永久回退**纯 Python 版(稳态调用中的非 numba 异常照常传播——那是真实语义;**校验窗口内**编译版抛任何异常都算分歧:纯 Python 版已成功,回退并返回其结果,而不是把 Python 不会抛的错误交给用户;来自编译调用内部的 `ImportError` 同样视为 numba 基建故障——白名单代码不含 import,典型来源是陈旧的磁盘缓存条目);分发器永不把全局名重绑定为裸 numba Dispatcher——后来出现的不支持类型必须能回退,而不是抛 `TypingError`。

**首调对照校验**:白名单保证被 jit 的函数是纯函数,因此首次调用会把纯 Python 版与编译版**各跑一遍并比对结果**(int 精确相等、float 带 NaN 感知的微小容差、外提循环的元组逐元素),不一致(实践中即 int64 回绕)则永久回退纯 Python——把"静默算错"降级为"一次性开销后保持正确"。校验调用返回 Python 结果(语义精确),代价是首调多付一次纯 Python 执行;`OPAST_JIT_NO_VERIFY=1` 可关闭。注意这是启发式安全网而非证明:之后用更大参数调用仍可能回绕。编译产物与校验状态按代码身份(内容寻址的源码路径)进程级记忆,同一优化模块反复 exec(如 bench)只编译、校验一次;`njit(cache=True)` 仅在定义模块可导入(存在于 `sys.modules`)时启用——对 scratch globals 里 exec 的函数,numba 会把编译环境记成 `<dynamic>` 模块,后续进程加载该缓存条目会直接崩溃。`--jit` 时优化后源码会落到内容寻址的临时文件(numba 需经 `inspect` 读源码,且行号需精确;`__file__` 仍指向原脚本),配合 `njit(cache=True)` 二次运行免编译。

**⚠️ 选择加入的语义警告**:numba 的整数是定宽 int64,中间值超出 ±9.2e18 会**静默回绕**——这是 `--jit` 不默认开启、也不纳入"语义保持"承诺的原因。白名单只预测编译成功率,类型正确性由运行期回退兜底,但 int64 溢出编译器无法察觉。数值不会超界的热函数才应使用。

## 相关项目(prior art)

- [pyastop](https://github.com/xiaonanln/pyastop)(2017-2018,MIT):同为 AST 级源码优化器,主打"全项目全局分析 + 注释引导",实现了折叠/展开/死代码/小函数内联的早期原型后停更。opast 走的是**逐 pass 可证明语义保持**路线(动态构造污染检测、类型/区间/逃逸的静态证明),不依赖用户注释。
- [fatoptimizer](https://github.com/vstinner/fatoptimizer)(FAT Python 项目,PEP 509/510/511 配套,2016 年后停更):走"运行期 guard + 函数特化"路线,因 guard 开销与侵入性作罢。opast 的静态 pass **零运行期开销**(优化即普通源码),运行期机制仅存在于选择加入的 `--jit` 路径,且以永久回退兜底。
- CPython 自带的 AST/窥孔优化器:只做常量折叠等极保守变换。opast 的收益主要来自 CPython 不做的部分——内联、需类型证明的代数化简、循环改写、去动态化。

## 测试

```powershell
python tests/verify_opast.py    # 全量验收(逐 pass 行为断言 + 原始/优化双跑输出对照)
```

用例源码由脚本按需生成到 `tests/cases/`(不入库);bench(见上文)另带结果一致性校验。

## 已知限制

- 以 `compile(优化后AST, 原文件名)` 执行,traceback 行号已尽量沿用原始位置,但删除/改写过的行可能与原文件轻微错位。
- 所有变换以保守的静态证明为前提,证明不了的一律不动;traceback 帧数(内联)、异常时的部分累加状态(loop-to-comp,已按 try/with 排除)等已文档化的边缘可观测差异见各 pass 说明。
- 要求 Python ≥ 3.10。
