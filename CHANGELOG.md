# Changelog

## 0.2.1 (2026-07-27)

**修复:** 0.2.0 中 `opast -O3 script.py` / `opast --aggressive script.py`(README 的示例写法)会把脚本路径吞成选项值而报"缺少脚本路径";bench 的 `--aggressive 负载名` 同病。现在裸 `--aggressive`/`-O3` 在解析前改写为 `--aggressive=all`(该显式写法同时开放);改写止步于 `--` 与第一个位置参数,用户程序的 `sys.argv` 逐字节原样。空格传值形式(`--aggressive annotations`)不受影响。

**文档:** 双语 README 补齐流水线顺序中的尾递归/模块层外提、四个激进层 bench 负载与 `--aggressive=all` 写法。

## 0.2.0 (2026-07-27)

**新增:两层契约与 `--aggressive` / `-O3`。** 默认层保持"每条改写皆有静态证明";激进层每个选项以一条明确写出的假设换取更多优化,`--report` 逐条列出**实际生效**的假设(被 `--disable` 掉全部消费者的选项不虚报)。八个选项:

- `annotations` — 信任裸 `int`/`float` 参数与返回注解,强度削减 / LICM / CSE / 区间收窄跨过函数边界(`bool` 子类与 PEP 484 数值塔已各自处理);
- `budgets` — 纯资源交换:loop-fold 模拟步数 20 万 → 2000 万、折叠字面量与内联预算放大;
- `fastmath` — 非逐位精确的浮点改写:`F + 0`、`F ** 2 → F * F`(该运算约 3 倍)、任意常量除数转倒数乘;
- `loop-state` — 模块层热循环外提(`LOAD_NAME` → `LOAD_FAST`,实测 ~2 倍)及模块层 loop-to-comp;假设:循环中途异常时模块全局可能保持循环前值;
- `module-locals` — 修饰 `loop-state`:无人读取的模块级名字视为私有临时量;
- `tail-calls` — 尾递归消除(带深度计数器复现 `RecursionError`,2.6 倍;支持分治代码的部分消除);
- `unbounded-recursion` — 修饰 `tail-calls`:去掉计数器(4.6 倍,深层尾递归直接跑通);
- `jit` / `opt-imports` — 既有选择加入项纳入同一伞下,独立开关保留。

**新增默认层 pass:** `cond-narrow`(区间可判定的条件折叠:路径敏感收窄、赋值传递、守卫子句)。proven-float 分析接入 algebra/LICM/CSE;`F / 2^k → F * 2^-k`(逐位精确)进入默认层。

**修复:** LICM/CSE 的支配扫描此前从不把参数视为已绑定,涉及参数的表达式从未被外提过;dead-code 现在清理复制传播遗留的无用裸名字语句;超大整数除数(`10**400`)曾使代数化简崩溃。

**⚠️ 行为变更:** 模块层的 loop-to-comp 改写从默认层收紧至 `--aggressive=loop-state`——循环中途异常时"半满列表"与"从未赋值"的差异在模块层可被外部 `try` 观察,不再符合默认层的证明标准。函数内不受影响。

## 0.1.0 (2026-07-24)

首个公开版本。包名 `opast`("OPtimizing AST",PyPI/import/CLI 三者一致);GitHub 仓库 pyOpAst。

**优化 pass(14 个,迭代到不动点)**:去动态化、常量折叠、常量传播(单绑定 + 跨作用域 + 跨度 + 复制传播)、代数化简(含强度削减与区间分析支撑的 `abs` 消除)、循环闭式折叠(loop-fold)、死代码消除、下标循环转直接迭代/enumerate(range-to-iter)、循环不变量外提(LICM)、公共子表达式消除(CSE)、未使用消除、简单函数内联(表达式体 + 直线语句体)、累加循环转推导式(loop-to-comp)、推导式转 map/filter、全局名局部化。LICM/CSE 支持新鲜容器的 `len()` 缓存。

**动态代码回退**:以作用域为单位的保守污染检测;`eval`/`exec`/帧内省等出现即整域跳过。

**`--jit`(选择加入,需 numba)**:热数值函数 njit 装饰、热循环外提、njit 间调用、首调对照校验、运行期永久回退;变量循环边界的运行期 lazy 编译(规模/耗时/调用量三触发,numba import 延迟到首次编译)。

**工具链**:CLI(`opast script.py`,`--show/--report/--disable/-c/-o`)、`--opt-imports` 导入钩子(内容寻址缓存,不污染 `__pycache__`)、IPython cell magic、带结果校验的 bench 矩阵(16 个内置负载)、全量验收测试。
