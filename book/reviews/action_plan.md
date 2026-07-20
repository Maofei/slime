# slime 技术书修订行动计划

汇总 5 个评审智能体的反馈，按优先级排序。每条标注：
- 严重程度（P0 必改 / P1 高 / P2 中 / P3 低）
- 类型（结构 / 内容 / 图示 / 一致性 / 事实）
- 涉及章节
- 具体修改

---

## P0 — 必须修复的结构性 / 跨章问题

### A1. Ch 13 § 13.3 重复了 Ch 11/12 的 Apply This
**类型**：结构 + 内容
**问题**：结语章 12 条可迁移模式中第 10、11、12 条几乎逐字复制
Ch 11/12 的 Apply This。大纲承诺的"精炼提纯"没做到。结语章质量
决定全书最终印象。
**修改**：把这 12 条改成**跨章对照式融合**——每条点明它"在哪几
章被反复证明"，而不是孤立复述。删除与单章 Apply This 文字重合
的部分，只保留跨章主题（比如"CPU 守不变量"应综合 Ch 4 test_loss_cp_invariance
+ Ch 11 mp.spawn helper + Ch 10 plugin_contracts 三处共用模式）。

### A2. Ch 11 Apply This 完全缺失 metric 模式
**类型**：结构
**问题**：Ch 11 § 11.2 "两条独立管线 + 同一组公式 + 测试守 diverge"
是本章最独特的洞见，但 Apply This 5 条里完全没有对应模式。
**修改**：把现有 5 条之一替换为"双管线 + 同公式 + 双向断言"
模式（参考 Ch 10 contract test 双向校验的写法）。

### A3. Ch 12 §12.4 与 §12.6 重复（用 grad 流向作断言）
**类型**：结构
**问题**：§12.4 主体讲 cispo stop-gradient 用 grad 流向断言，§12.6
深入剖析框又讲了同一件事，重叠度 80%。
**修改**：保留 §12.4，把 §12.6 深入剖析框换成另一个 PyTorch
autograd 应用例子（比如 register_hook、no_grad context、retain_graph
中的某一个真实场景），或者直接删掉该框。

### A4. Hook 数量在 Ch 10 与 Ch 13 之间不一致
**类型**：一致性 + 事实
**问题**：Ch 10 内部"18 与 21 并用"（official 18 + 3 个 megatron
path），Ch 13 同样混用。
**修改**：全书统一用 **"18 个 customization hook"** 作为对外说法（与
docs/zh/get_started/customization.md 一致），3 个 megatron path 作为
脚注或附录说明"加上 megatron-* 系列就是 21 个加载点"。Ch 10、Ch 13
对应位置统一改写。

### A5. Ch 9 开篇 "6 个文件 1.4k 行" 数字与文件数对不上
**类型**：事实
**问题**：9.1 节表格只列了 trajectory.py 一个文件的行数，1.4k 总
行数来源不清。
**修改**：用 `wc -l slime/agent/**/*.py` 实际统计后写准确数字；
如果是 6 个文件就明确列出（trajectory / sandbox / parsing /
aiohttp_threaded / harness/__init__.py / adapters/__init__.py 等
有意义的 6 个），与表格一致。

### A6. Ch 8 RoutingReplay 深入剖析框位置错位
**类型**：结构
**问题**：RoutingReplay 的深入剖析框放在 §8.6（server_control 节）
后面，但话题无关。Ch 10 §10.6 又讲了一遍 routing replay，存在
轻微重复。
**修改**：把 Ch 8 的 RoutingReplay 深入剖析框移到 §8.4 后或独立
成 §8.7（与 server_control 不混）。Ch 10 §10.6 缩到只引用 Ch 8
"已经讲过"，重点放在 group_rm 这一个特殊形态上。

---

## P1 — 高优先级内容问题

### B1. Ch 4 第 95-100 行叙事顺序倒置
**类型**：内容
**问题**：读者还没看到 Megatron schedule 与 DDP 默认的放缩，就被
告知 slime 那两行是"反向消除"——因果链反了。
**修改**：调整 §4.2 段落顺序——先讲下游（Megatron schedule
`/= num_microbatches` + DDP `× 1/(dp×cp)`）做什么放缩，再讲
slime 在 loss_function 那两行是抵消它们。可以画一张时序图把"先发生"
和"后发生"的因果关系明确化。

### B2. Ch 5 Sample 状态机正常路径几乎为空 + group-norm 缺失
**类型**：内容
**问题**：§5.2 跳过了 `PENDING → COMPLETED` 主线直接讲 ABORTED/
FAILED 分支；§5.4 的 `_post_process_rewards` group-norm（GRPO 关键
步骤）被伪代码 `...` 吞掉。
**修改**：
- §5.2 在状态机图前补一段"正常路径"叙述（PENDING → COMPLETED
  的完整流程，1 段 100-150 字）
- §5.4 把 group-norm 的伪代码展开到 8-12 行可读形态：
  `samples.reshape(rollout_batch_size, n_samples_per_prompt) → -mean → /std → flatten`

### B3. Ch 1 §1.3 五个赌注全展开过度
**类型**：内容
**问题**：5 个赌注每段 4-6 行散文，会抢 Ch 5/8/10/11/12 开场的
"第一次说"张力。
**修改**：把 §1.3 五赌注节压缩成紧凑表格 + 一段 4-5 行的总结。
每条赌注一句话点题 + 一句"哪一章展开"，不要在导览章把所有论点都
讲完。

### B4. Ch 1 §1.4 与 Ch 2 §2.5 重复 "4 个 SFT 参数" 列表
**类型**：内容
**问题**：同一份 SFT 参数清单出现两次。
**修改**：保留 Ch 2 版本（启动脚本作为 recipe 是它的主场）；
Ch 1 §1.4 改成一句话："SFT 是 RL 主循环的特化形态，靠 4 个参数
表达——细节见 §2.5"。

### B5. Ch 6 §6.5 Sample 状态机图与 Ch 5 重复
**类型**：图示
**问题**：Ch 6 §6.5 画了完整 5 态状态机，但全节真正贡献的只是
`ABORTED → Buffer` 这一条边和方法引用切权限那段深入剖析。
**修改**：删完整图，留 3 节点小图（`Generation → ABORTED → Buffer`
回流 vs `Generation → COMPLETED → Train` 主线）。其余篇幅让给
方法引用切权限那个深入剖析。

### B6. Ch 7 §7.2 "共享层"论点松脱
**类型**：内容
**问题**：节里塞了中间层、10 个 converter 差异、注册机制三件事，
但开篇句没说"为什么这种分层是正确的"。
**修改**：把开篇句重写成明确论点——例如"四条 carrier 在 wire
层分叉，但**共享一套 Megatron→HF 转换**——这种'共享中间层 +
carrier 分叉'的设计意味着新增模型只改转换器、新增 carrier 只改
传输，互不污染"。让现有三件事都收束到这一句下。

### B7. Ch 10 §10.1 节奏过早辩护
**类型**：内容
**问题**：plugin / ABC / decorator 对比表放在第一节正中央，读者
还没理解 slime 选了什么就先看到"为什么不选其他"。
**修改**：先讲正面叙事（`load_function` 5 行 + zero-import 接入），
把对比表降级成节末的小字提示框（或者移到 Apply This 节的第 1 条
"陷阱"里讨论）。

---

## P2 — 关键图示补充

### C1. Ch 3 §3.4 末尾补进程拓扑图
**问题**：开篇承诺"读完应该能画出 slime 启动后的进程拓扑图"，
但全章没给。
**建议图**：`graph TD`，三色区分——
- 灰色：0-GPU CPU actor（RolloutManager、Lock actor）
- 黄色：fractional-GPU actor（actor_group 的训练 actor，num_gpus=0.4）
- 蓝色：SGLangEngine actor + 它 fork 出的 multiprocessing 子进程
节点间用箭头连接调用关系。能替代 §3.3 + §3.4 共约 60 行散文。

### C2. Ch 4 §4.2 改 sequenceDiagram 展示 4 处放缩时序
**问题**：当前用文字罗列 4 处放缩，时序关系不直观。
**建议图**：`sequenceDiagram`，5 个 participant（loss_function /
slime_pre_scale / megatron_schedule / ddp_reduce / final_grad），
按时间顺序画出每一步的放缩因子。

### C3. Ch 5 §5.1 RolloutManager 14 行配 sequenceDiagram
**问题**：RolloutManager 的 14 行 generate 主体是 Ch 5 最核心
代码，但只有伪代码没有时序图。
**建议图**：`sequenceDiagram`，Driver → RM → health_monitor →
user_fn → _convert_samples → _split_by_dp → return，按 14 行
实际调用顺序。

### C4. Ch 5 §5.4 rollout_mask_sums 广播配 graph TD
**问题**：分母广播这个反直觉但关键的设计仅靠散文论证。
**建议图**：`graph TD`，左侧 "rollout manager 看到全量 sample"，
中间 "为每个 rollout 算 total_mask_sum"，右侧 "复制到该 rollout
的每个 sample → 切到不同 mb"。

### C5. Ch 7 §7.3 ASCII pipeline 图换成 sequenceDiagram
**问题**：当前用空格缩进模拟"GPU compute / d2h_stream / h2d_stream
三者时间重叠"，读者无法看出重叠关系。
**建议图**：`sequenceDiagram`，3 个 stream 为 participant，画
chunk N 与 chunk N+1 在不同 stream 上的并行。

### C6. Ch 10 至少补 2 张图
**问题**：18 个 hook 章但图示偏少。
**建议图**：
- `flowchart TD`：18 hook 按 rollout 阶段嵌入位置图（参考 outline
  里的图示建议）
- `graph TD`：契约测试双向校验数据流（CI fallback 路径 vs 用户
  指定路径，最后断言代码相同）

---

## P3 — 一致性 / 术语 / 句子级修改

### D1. 全书术语统一
- **sid vs session_id**：Ch 5、Ch 8、Ch 9 写法不一致，统一为
  `session_id`（首次出现），后续可用 `sid` 简写但必须先有定义
- **18 vs 21 hook**：见 A4，全书改为"18 个 customization hook
  （加上 3 个 megatron-* 系列共 21 个加载点）"
- **rollout manager vs RolloutManager**：跨章统一用 `RolloutManager`
  代码名

### D2. Ch 3 事实精确化
`slime/backends/megatron_utils/sglang.py` 行数核对——研究笔记说
44 行，实际可能略有出入（用 `wc -l` 验证后填准确数字）。

### D3. Ch 1 与 Ch 3 关于 "SGLang Engine 持有 GPU" 措辞矛盾
**问题**：Ch 1 §1.2 sequenceDiagram 下方写 "Engine: SGLang Engine
——持有 GPU 的 Ray actor"，但 Ch 3 §3.4 强调"SGLangEngine actor
只是 HTTP 客户端，模型在 fork 子进程里"。
**修改**：Ch 1 改成"持有 GPU 资源的 Ray actor（模型在它 fork
出的子进程里跑，详见第 3 章）"，避免读者在 Ch 1 形成错觉。

### D4. 各部评审建议的剩余句子级修改
- Part 1：剩余 9 条句子级修改（详见 `book/reviews/part1_intro_setup.md`）
- Part 2：剩余 10 条建议改（详见 `book/reviews/part2_core_loops.md`）
- Part 3：剩余 8 条建议改（详见 `book/reviews/part3_dataflow_sync.md`）
- Part 4：剩余 10 条建议改（详见 `book/reviews/part4_deploy_extension.md`）
- Part 5：剩余 10 条 S 类修改（详见 `book/reviews/part5_engineering_finale.md`）

**处理策略**：P3 级别按部依次过一遍，每条 5 分钟内能改完的直接
改；需要重写整段的归入 P1 或 P2 重新评估。

---

## 修订执行顺序建议

1. **第一轮**（P0，约 1-2 小时）：A1-A6 六个结构性问题。完成后
   全书结构稳定。
2. **第二轮**（P1，约 2-3 小时）：B1-B7 七个内容问题。完成后正文
   叙事完整。
3. **第三轮**（P2，约 2 小时）：C1-C6 六张关键图示补充。完成后
   视觉密度合格。
4. **第四轮**（P3，按部依次过）：D1-D4 一致性扫尾。完成后术语
   统一、句子级润色到位。

之后进入**阶段 7：代码块审计**——逐块核对每章伪代码，确保
- 没有与源码逐字一致
- 都加了 `// Illustrative` 注释
- 保留指向真实源码路径的引用
