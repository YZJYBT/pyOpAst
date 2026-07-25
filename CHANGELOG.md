# Changelog

## 0.1.0 (2026-07-24)

首个公开版本。包名 `opast`("OPtimizing AST",PyPI/import/CLI 三者一致);GitHub 仓库 pyOpAst。

**优化 pass(14 个,迭代到不动点)**:去动态化、常量折叠、常量传播(单绑定 + 跨作用域 + 跨度 + 复制传播)、代数化简(含强度削减与区间分析支撑的 `abs` 消除)、循环闭式折叠(loop-fold)、死代码消除、下标循环转直接迭代/enumerate(range-to-iter)、循环不变量外提(LICM)、公共子表达式消除(CSE)、未使用消除、简单函数内联(表达式体 + 直线语句体)、累加循环转推导式(loop-to-comp)、推导式转 map/filter、全局名局部化。LICM/CSE 支持新鲜容器的 `len()` 缓存。

**动态代码回退**:以作用域为单位的保守污染检测;`eval`/`exec`/帧内省等出现即整域跳过。

**`--jit`(选择加入,需 numba)**:热数值函数 njit 装饰、热循环外提、njit 间调用、首调对照校验、运行期永久回退;变量循环边界的运行期 lazy 编译(规模/耗时/调用量三触发,numba import 延迟到首次编译)。

**工具链**:CLI(`opast script.py`,`--show/--report/--disable/-c/-o`)、`--opt-imports` 导入钩子(内容寻址缓存,不污染 `__pycache__`)、IPython cell magic、带结果校验的 bench 矩阵(16 个内置负载)、全量验收测试。
