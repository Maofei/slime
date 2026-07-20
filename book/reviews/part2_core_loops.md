# 第二部"核心循环"评审（Ch 4 training loop + Ch 5 rollout loop）

## 总体判断

两章都已具备出版级骨架——叙事弧线清楚、回扣全书论点（"一条数据
路径"、"原生 engine 透传"）、Apply This 的 5 条都从章正文里长出来
而不是套话。但作为全书最重要的两章，**承载的复杂度还没有被完全
拆开给读者**：Ch 4 的 4 处放缩与 Ch 5 的 1485 行论证都在"把结论
告诉你"和"让你自己看明白"之间偏向了前者，缺了一两层中间步骤。两
章的图示密度（各 2 张）也明显低于其他核心章建议的 3-4 张。

风格上 Ch 4 比 Ch 5 更紧凑、论点更尖锐；Ch 5 在 5.1 的部分有一些
"教科书式列举"的段落（行数表、三层抽象表）拖慢了节奏。两章之间
**抽象层级基本对称**——都是"主接口 → 内部分层 → 关键机制 → 反
直觉决策 → 集成接口 → 深入剖析"，这种对称值得保留。

---

## 1. 开篇质量

**Ch 4**：开篇是合格的，"slime 几乎不重写 Megatron 的训练 step"
是一个**有立场的论点**，4 处放缩作为悬念预告也很好。衔接 Ch 3
那句 "上一章把 placement、引擎初始化、RolloutManager 都就绪了"
直接、精准。**问题在于"全书最重要"这个分量没有被声明出来**——
Ch 4 开篇说"这一章和下一章并列是全书最重要的两章"，但没有解释
**为什么**。读者在 Ch 1 已经被告诉过这一点，这里只是重复。建议
改成"为什么 training 这一侧值得整章讲"——比如"训练 step 是 RL 框
架里最容易被'写得像 supervised'但其实暗藏分布式数值陷阱的地方"。

**Ch 5**：开篇用"1485 行里只有 200 行业务逻辑"这个反差作为钩子
非常好，比 Ch 4 的开篇更抓人。衔接 Ch 4 的方式（"上一章是 Megatron
侧。这一章是 SGLang 侧"）有点过于平淡——可以反过来用"上一章 Megatron
侧的故事是'四处放缩组合成一行干净的梯度'；这一章 SGLang 侧的故事
是'1485 行只在管生命周期'"建立一个**对偶**，让两章的对称性立刻
可见。

---

## 2. 流畅性

**Ch 4 的 4 处放缩（4.2）讲清楚了吗**：基本讲清楚了，链路图 +
乘积验证那一段是高质量推导。但有一个**关键跳跃**没补上：第 81-88
行的伪代码下面紧接着说"它不是随手凑的，它是一个反向消除链路的一
个环节"——但读者还**没看到下游会做什么放缩**，就被告知"这是反向
消除"，会困惑。**应该先讲 Megatron 默认会做什么、DDP 默认会做什
么，再讲 slime 这两行是在抵消它们**。叙事顺序反了。

另一个不够清晰的点：第 117-119 行的乘积验证 `× num_mbs / step_gbs
× (dp × cp) ÷ num_mbs × 1 / (dp × cp) = 1 / step_gbs` 是对的，但
**为什么 `loss × (1 / step_gbs)` 就是 per-rollout-mean 梯度，要再
补一句**——读者需要知道 loss_function 输入的 `sum_of_sample_mean`
是"每个 sample 一个 mean 然后跨 sample 求和"，再乘以 `1/step_gbs`
就是 per-rollout-mean。这一步在笔记里 6.2 节有，但被章里省略了。

**Ch 5 的 1485 行真实分布段（5.1）论证力度**：表格本身没问题，但
**"启动逻辑占了最大头"那一段（96-99 行）有点拖**——端口、PG bundle、
NCCL 端口、router 注册端口连珠炮列出来，读者会陷入"细节淹没"。建
议留一两个最有代表性的（端口冲突、PG 跨 bundle），把其余压缩成
"三种端口在不同生命周期被使用"一句话带过。

**值得删的拖沓**：Ch 5 的 5.6 集成接口节是参考手册风格——三个子
节列出"manager 与 X 通过 Y、Z 接口耦合"，读起来像 API doc。这个
信息对深度读者有用，但作为正文叙事偏弱。**建议改成"两条通道：
HTTP 数据面 + Ray 控制面"作为单一论点展开**，把 Data Buffer 和
customization 那两段压缩成"接口表"放在节末，而不是各自展开。

---

## 3. 内容删减建议

| 位置 | 问题 | 建议 |
|---|---|---|
| Ch 4 第 33-38 行四层架构表 | 信息量适中，可以保留但表头"知道什么 / 不知道什么"比四个文件名更值得突出 | 把表压缩成"四层的关键边界"散文 2 段，文件路径放在脚注或行末括号 |
| Ch 4 第 81-88 行的伪代码 | 长度可以，但 `# loss.py:1285-1293 的核心` 这种注释把读者引向源码而不是模式 | 去掉源码行号注释，保留 `# 伪代码 —— illustrative` 即可 |
| Ch 5 第 86-92 行行数分布表 | 表格本身价值不高，"manager 213 行 / ServerGroup 315 行" 这种数据是参考性的 | 改成 1 句话："启动编排（端口 / PG bundle）占 ~340 行，metric 计算 ~250 行，业务逻辑 ~200 行——业务在最少数" |
| Ch 5 第 168-186 行的 `base_types.py` 伪代码 | 26 行的契约用 19 行伪代码展示有点冗余 | 压成 6-8 行只展示 `RolloutFnTrainOutput` + `call_rollout_fn` 的 fallback 逻辑 |
| Ch 5 第 193-202 行 6 种形态表 | 表格信息密度高，值得保留 | 保留，但"在解决什么"列的描述可以更短（例如 SFT 那行"不调 SGLang，纯走 mask generator" 已足够，不需再说"算 loss_mask"） |

---

## 4. 缺失内容

**Ch 4 的关键空白**：

1. **`compute_advantages_and_returns` 完全没出现**。架构图（45-68
   行）里画了 E "compute_advantages_and_returns"，但正文从头到尾没
   解释它在做什么、为什么放在 ref/teacher log_prob 之后、它的输出
   `advantages / returns` 如何就地写回 `rollout_data`。读者会困惑
   架构图上这个框是干嘛的。建议在 4.3 或 4.4 之间补一小节（200-400
   字）讲"advantage 计算的位置选择"。

2. **`can_reuse_log_probs_in_loss` 优化（笔记 6.4）缺失**。这是
   笔记里明确列为"出人意料的决策"的一条，章里没讲。这条揭示了
   RLHF "什么时候 old_log_prob 等于当前 actor log_prob"的 N 元
   布尔表达式，值得放在 4.4 多模型一节末尾——它正是"多模型切换的
   优化路径"。

3. **`StatelessAdam` 与 `with_defer` / `inverse_timer` 也都缺失**。
   笔记列出但章里没讲。前者是"一步式 rollout"的关键优化，后者是
   train_wait timer 的妙招。可以选其一放在 Apply This 之前补一条
   深入剖析框。

**Ch 5 的关键空白**：

1. **Sample 状态机（5.2）讲了 5 个状态，但 `PENDING → COMPLETED /
   TRUNCATED` 那条"常规生成路径"几乎一笔带过**——`PENDING` 阶段
   sample 内部发生了什么（HTTP 调度、token 累积、reward 计算）没说。
   读者看完只能记住 `ABORTED` 和 `FAILED`。建议补 1 段：常规路径
   下 sample 是怎么从 PENDING 走到 COMPLETED 的——HTTP 请求发出去
   → response token 逐步 append → finish_reason 决定 status。这个
   主线讲清后，ABORTED / FAILED 才作为"分支"凸显。

2. **`_post_process_rewards`（group-norm）缺失**。笔记 2.4 节明确
   列出了"对 GRPO/GSPO/CISPO 做 group-norm（reshape 成 `(rollout_batch_size,
   n_samples_per_prompt)` 后减均值除标准差）"，章里只在 5.4
   `rollout_mask_sums` 的伪代码 `train_data = {...}` 里被一笔带过。
   group-norm 是 RL 里"reward 归一化"最关键的一步，应该在 5.4 之
   前补一节讲。

3. **`Box` 与 `--rollout-data-transport` 没解释**。开篇说
   `rollout_data_ref` 是 `list[Box]`，每个 Box 对应一个 dp_rank，
   但 Box 到底是什么、为什么不直接用 `ray.ObjectRef`、
   nixl transport 是什么——都没讲。读者读到末尾仍不知道 Box 的角色。
   建议在 5.6 节集成接口里补 1 段。

---

## 5. 需要的图示

**Ch 4 应增图**：

1. **"4 处放缩"的时间轴图**（强烈建议）。当前的 LR mermaid 图
   （104-115 行）把 4 步放缩并列展示，**但没有突出"谁调用谁"的
   时间顺序**——slime pre-scale 是发生在 loss_function 内部、
   Megatron schedule 除法是 forward_backward_func 调用过程中、DDP
   reduce 是 backward 完成后。建议改成 `sequenceDiagram`：参与者
   是 `slime loss_function` / `Megatron schedules.py` / `DDP grad
   reducer`，每步标注它做的乘除因子，最末尾汇总成 `1/step_gbs`。
   这样"反向消除"的因果链才直观。

2. **"四层架构"图替代当前文字表**。`graph TD` 自上而下 4 层（主
   循环 / Ray 层 / Backend actor / Megatron 适配），每层画 1-2 个
   关键标识符（`train.py` / `MegatronTrainRayActor` / `forward_backward_func`），
   层间用箭头标注"知道什么/不知道什么"。这比文字表更易扫读。

3. **`TensorBackuper` 切换的"显存 vs CPU"对比图**（4.4 节）。一
   张简单 `graph LR`：左边"传统：actor + ref + teacher 三份 GPU
   显存"，右边"slime：actor GPU + ref/teacher/old_actor 在 CPU
   pinned"，箭头标注 `restore("ref")` 是 CPU→GPU 的 copy。这一对
   比图能立刻把节省说清楚。

**Ch 5 应增图**：

1. **`RolloutManager.generate` 的 14 行调用流程图**（5.1 节）。当
   前是用伪代码展示 14 行，但每行的"调用 → 返回 → 下一步"关系藏
   在代码里。改成 `sequenceDiagram`：参与者 `driver` / `RolloutManager`
   / `用户函数` / `data_source`，时间轴展示 `generate.remote()` →
   `health_monitoring_resume()` → `_get_rollout_data` → `_convert_samples_to_train_data`
   → `_split_train_data_by_dp` → 返回 `list[Box]`。这是全章的"主
   骨架图"，缺它读者很难把后面几节挂上去。

2. **`rollout_mask_sums` 的"广播分母"示意图**（5.4 节）。一张
   `graph TD`：上方画一个 rollout 包含 8 个 sample（标注 sum=120），
   下方画 micro-batch 切分把 8 个 sample 分到 3 个 mb（每个 mb 内
   局部 sum 不同），中间用箭头标注"每个 sample 都带 `rollout_mask_sum=120`
   进入训练侧"。当前文字描述 "DP + micro-batch packing 会把一个
   rollout 的多个 sample 切到不同 micro-batch" 用图比文字快 5 倍。

3. **三层引擎抽象图保留并强化**。当前 67-72 行有一个 ASCII 图，建
   议改成 mermaid `graph TD` 配上"一个 manager / N 个 server / M
   个 group / K 个 engine"的数量关系——读者会更清楚"多模型时多了
   什么、PD 解耦时多了什么"。

---

## 6. 跨章一致性

**对称性整体良好**：

- 抽象层级对称：两章都是"接口 → 内部分层 → 关键机制 → 反直觉决策
  → 集成 → 深入剖析"
- Apply This 格式对称：都是 5 条、每条"模式名 / 怎么改造适配 / 陷阱"
  三段式
- 章末"下一站"格式对称：都是简短一段，预告下一章

**不一致或可对齐的地方**：

1. Ch 4 用"4 处放缩"作为核心论点的具象表达；Ch 5 用"1485 行的真实
   分布"作为对应的具象数字。这是个**好的对称**，可以在 Ch 5 开篇
   显式说出来（"上一章的具象数字是 4 处放缩，这一章是 200 / 1485"）。

2. Ch 4 的 4.5 节后跟一个长长的"深入剖析：上游 OOM 的外科手术"框；
   Ch 5 的对应位置（5.6 后）是"深入剖析：CI 故障注入"。两个深入剖
   析都很好，但 Ch 5 的那个其实更轻——可以考虑把 CI fault injection
   提到 5.1 RolloutManager 那 14 行的解释里（毕竟它就在那 14 行
   里），把 5.6 后的深入剖析换成更有重量的话题，比如 `recover()`
   区分 updatable / non-updatable 的两条路径（笔记 6.9）。

3. Ch 4 的 Apply This 第 1 条"分布式数值放缩集中在一处"是从 4.2
   直接长出来的；Ch 5 的 Apply This 第 3 条"在能看到全量的位置预算
   reducer 元数据"也是从 5.4 长出来的——**这两条本质上是同一个
   抽象模式**（在数据流上游做集中决策）。可以在 Ch 5 那条末尾加一
   句"这条与 Ch 4 的'数值放缩集中在一处'是同一个模式在数据流不同
   位置的体现"，让两章呼应明显。

---

## 7. 具体修改清单（按严重程度排序）

**必须改（结构 / 事实 / 缺失）**：

1. **Ch 4 第 95-100 行的叙事顺序倒置**（必须改）
   > 原文："第一眼看这两行没有意义……它不是随手凑的。它是一个**反向
   > 消除链路**的一个环节——slime 想要最终的梯度等于'每个 rollout
   > 算一个 mean……"
   > **问题**：读者还没看到下游做什么，就被告知是"反向消除"。
   > **建议改写**：先用 1-2 句话讲 "Megatron 的 schedule 会对每个
   > mb 的 loss `/= num_microbatches`，DDP grad reduce 会把跨 dp/cp
   > 的梯度平均（等价于 `× 1/(dp×cp)`）——这两层放缩是 Megatron
   > 替你做了的"，再接"slime 想要的最终梯度是 per-rollout-mean，
   > 但 Megatron 的两层默认放缩并不会自动得到这个结果。slime 在
   > loss_function 里这两行就是反向消除——"。

2. **Ch 4 缺 `compute_advantages_and_returns` 的解释**（必须改）
   架构图里有 E "compute_advantages_and_returns"，但正文不解释。
   在 4.3 节末或 4.4 节前补 200-400 字小节："advantage 计算的位置
   选择"——为什么放在 ref/teacher log_prob 之后、为什么就地写回
   `rollout_data` 而不是返回值、它支持哪些算法（GRPO / GSPO /
   CISPO / PPO / REINFORCE++）。

3. **Ch 5 缺 `_post_process_rewards` group-norm 的讲解**（必须改）
   当前 5.4 节伪代码 `train_data = {...}` 里用 `...` 把它跳过了，
   但 group-norm 是 RL reward 归一化的关键步骤、是笔记里被明确标
   出的关键抽象之一。建议在 5.4 之前补 1 个小节 "5.4 reward 的
   group-norm 在哪里做"，说清"reshape 成 `(B, n_samples_per_prompt)`
   后减均值 / 可选除标准差"。

4. **Ch 5 第 117-128 行 Sample 状态机正常路径几乎为空**（必须改）
   > 状态机图清楚，但正文跳过了 `PENDING → COMPLETED / TRUNCATED`
   > 的正常路径解释，直接讲 ABORTED 和 FAILED。
   > **建议补**：在状态机图之后，先用 1 段讲常规路径——"PENDING
   > 状态下 sample 持续接收 SGLang 的 response token，每收到一段就
   > `append_response_tokens` 累计到 `tokens / response / log_probs`；
   > finish_reason 决定终态——stop → COMPLETED，length →
   > TRUNCATED"。再讲 ABORTED 和 FAILED 作为"非常规分支"。

5. **Ch 5 的 5.6 集成接口节风格过于参考手册**（必须改）
   > 三个子节（Data Buffer / SGLang engine / customization hook）
   > 都是"通过 X、Y 接口耦合"的列举，叙事弱。
   > **建议改写**：保留"数据面走 HTTP、控制面走 Ray"这条核心论点
   > 作为该节的唯一主线展开 200-300 字，把 Data Buffer 接口和
   > customization hook 压缩成节末的小表格（不超过 8 行）。

**建议改（润色 / 风格 / 局部）**：

6. **Ch 4 第 16-17 行 "这一章和下一章并列是全书最重要的两章" 重复**
   Ch 1 已经讲过这件事，这里再讲一遍是套话。**建议改为**："训练
   step 是 RL 框架里最容易'看似 supervised'但其实暗藏分布式数值
   陷阱的地方——这一章讲 slime 如何把它写得'看似平凡'，但平凡背后
   的几处关键设计支撑了整个 RL 训练的正确性。"

7. **Ch 5 第 5-7 行开篇过于平淡**
   > "上一章是 Megatron 侧。这一章是 SGLang 侧。"
   > **建议改写**：建立对偶——"上一章 Megatron 侧的故事是'4 处放
   > 缩组合成一行干净的梯度'；这一章 SGLang 侧的故事是'1485 行
   > 里 1200 行在管生命周期'。两者都是 slime '一条数据路径'赌注
   > 在两侧的具象表达。"

8. **Ch 5 第 96-100 行"启动逻辑占了最大头"段落臃肿**
   > 端口 / PG bundle / NCCL 端口 / router 注册端口连珠炮列举。
   > **建议改为**：选最有代表性的 1-2 个例子（同节点不同 group
   > 不能撞端口；PD 解耦的两阶段启动），其他压缩成"另有 N 类协调
   > 细节"。

9. **Ch 4 第 139-140 行的引用块措辞略弱**
   > "数值放缩这种错误**不会让训练崩**。它只会让 grad 算错一个
   > 常数倍……等到一个月后发现 reward curve 不收敛回头排查，已经
   > 烧了几千 GPU 小时。"
   > **建议加强**：明确指出"几千 GPU 小时"对应的具体规模（70B 模
   > 型 8 节点连续训 1 周 ≈ X GPU 小时），让数字有锚点；或者去掉
   > 具体数字，改成"等到下次大规模 sweep 才发现不收敛，沉没成本
   > 已经远超一个 5 秒 CI 测试的边际成本"。

10. **Ch 5 第 236-240 行 "compact rollout 是个轻量约定" 与上下文断
    裂**
    > 这段紧跟在 fully_async 后面，但 compact rollout 跟 fully_async
    > 没有逻辑联系——读者会困惑为什么突然讲这个。
    > **建议**：把这段移到 5.3 末尾（讲完 6 种形态之后），作为
    > "形态之上还有一个轻量形状约定"的补充；或者完全移到 5.2 Sample
    > 节末尾（毕竟它是 Sample 的 `rollout_id` 字段在做的事）。

11. **Ch 4 第 308 行的版本号写法**
    > "`inspect.signature` 在运行时检测 Megatron v0.13 与 v0.15rc7
    > 的 API 差异"
    > **问题**：codebase2book.md 阶段 4 一致性检查规则明确要求"避
    > 免会过时的定量陈述"——具体版本号会过时。
    > **建议**：改为"在运行时检测 Megatron 不同主版本的 API 差异
    > （参数个数从 2 变成 3）"，去掉具体版本号。

12. **Ch 5 第 22 行的具体行数 "1485 行"出现频率过高**
    全章一共出现 5 次（第 22、26、41、84、412 行）。同样违反"避免
    定量陈述"原则。**建议**：开篇保留 1 次作为反差钩子，其他位置
    改为"manager 这个最大的文件" / "1500 行级别的 manager"。

13. **Ch 4 第 232-248 行 `TensorBackuper` 伪代码偏长**
    19 行展示一个 5-15 行就能说清的模式。**建议压缩**：只保留
    `backup` 和 `restore` 各 3 行就够；`__init__` 那行 dict 注释可
    以去掉，因为读者看到 `snapshots: dict[str, list[Tensor]]` 就懂。

14. **Ch 5 第 309-326 行 `GenerateState` 伪代码偏长**
    18 行展示"挑最空闲 dp_rank"的模式。**建议**：把类定义和
    `@contextmanager` 留下，但把 `min_count` / `candidates` /
    `np.random.choice` 三步压缩成 1 句注释 + 2 行代码。

15. **Ch 4 Apply This 第 5 条"不 fork 上游"与第 4.5 节深入剖析重复**
    深入剖析框已经把 monkey-patch 讲透了，Apply This 第 5 条又复述
    一遍。**建议**：Apply This 第 5 条压缩成 3-4 句话，把"陷阱"那
    段（`slime 的 megatron_patch/ 目前只有 1 个补丁，这是健康信号`）
    保留，其他删掉。

---

## 评审小结

Ch 4 与 Ch 5 都是骨架坚实的核心章，可以在不重写的前提下通过 5 处
必须改 + 10 处建议改打磨到出版级。**最高优先级**是修复 Ch 4 的
"4 处放缩"叙事顺序（让读者先看到下游做什么再讲 slime 反向消除）、
补 Ch 5 的 Sample 常规路径与 group-norm、把 5.6 集成接口节从参考
手册风格改回叙事风格。**图示补充**（4 处放缩的 sequenceDiagram、
RolloutManager 14 行的 sequenceDiagram、rollout_mask_sums 的广播
示意图）是性价比最高的改动——3 张图能让两章的可读性整体上一个台
阶。
