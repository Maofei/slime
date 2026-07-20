# Part 1 评审：入门 + 起步（第 1-3 章）

**评审范围**：
- 第 1 章 `ch01_main_loop.md` — 100 行的主循环（约 1500 字中文正文 + 表格/图）
- 第 2 章 `ch02_configuration.md` — 配置面（约 1400 字）
- 第 3 章 `ch03_runtime.md` — 运行时基础（约 1500 字）

**整体定位**：三章是全书的"入门"梯度——Ch 1 设论点（100 行主循环 + 5 大赌注），Ch 2 兑现"赌注 2"（透传胜过封装），Ch 3 落 placement + 引擎进程模型。叙事弧线清晰，三章互相回扣到位。下面按 7 个维度逐项评。

---

## 1. 开篇质量

**第 1 章**（全书首章）。开篇做得**好**：用"103 行 / 80 行"两个具体数字立刻给出张力——一个 SOTA 级 RL 框架的主循环居然能压到 100 行。第三段直接点出全书的核心问题"剩下的复杂度去哪了"，把后续 12 章定位为"回答这一个问题"。这是教科书级的开篇——具体数字 + 反直觉断言 + 设悬念。**无问题**。

**第 2 章**。开篇用"103 → 2010"的对照衔接 Ch 1，很自然。"任何一个工程师看到这个数字第一反应都是技术债"这句站位准确，立刻把读者代入。**无问题**。

**第 3 章**。开篇"从一行到几百个进程"用 `pgs = create_placement_groups(args)` 这一行做钩子，回指 Ch 2 末尾的"下一站"。**有一处可改进**：第 13-21 行那个 4 条 bullet 列表（"它已经做完了下面这些事"）信息密度过高且**预先剧透**——读者还没看到 3.1 节就先被告知"InfoActor 探测拓扑、按 IP+GPU id 排序"等细节，这些点在正文里反而没有展开（InfoActor 只在 3.1 末尾被一笔带过）。建议把这 4 条 bullet 压成 1-2 句话："这一行返回前，slime 已经决定了每个角色占哪几张 GPU，以及 colocate 和分离部署的拓扑差异。"细节留给 3.1。

---

## 2. 流畅性

**第 1 章**。整体节奏好，但 **1.3 节"100 行不告诉你的事"那张 12 行的章节地图表格**之后，紧接着出现 5 个"赌注 N"的大段落（200-247 行），每段 4-6 行。这一段读起来**最累**——5 个赌注全部叙述节奏一致（粗体小标题 + 一段散文 + 引向哪几章），且每个赌注都在 Ch 1 完整说明白，结果就是 Ch 1 抢了后续章节的"主场"。建议：把 5 个赌注压缩成 3-4 句话+一张表，或者只展开赌注 1-2（最核心），其余 3 个赌注只点名字 + 一句话 + 指引到对应章节。

**第 2 章**。**2.3 节"Megatron 透传：寄生策略"**有一处节奏问题——"第三个钉子是 Megatron parser 的 `ignore_unknown_args=True`"（第 210 行）独立成段。前面"两个钉子"是 `extra_args_provider` + `reset_arg`，但作者没明确说"第一个钉子"和"第二个钉子"是什么；读者要回头数才能理解"第三"指的什么。建议在 2.3 开头加一句"slime 用三招把自己的参数寄生到 Megatron parser 里"作为路标。

**第 3 章**。**3.1 节末尾"如果你打开 tests/test_placement_group.py"那段**（117-122 行）信息散——它讲的是测试覆盖了 10 个用例守住布局，但放在 3.1 末尾、3.2 开头之前，像个孤立的工程笔记，没有承上启下。建议要么删掉（细节归 Ch 11 工程基础设施），要么用一句话融进上一段。

---

## 3. 内容删减

**第 1 章**。**赌注 3-5 段落**（赌注 3 单 SGLang backend、赌注 4 customization、赌注 5 显式 dataflow + CI）是最大的删减候选。这三段在 Ch 1 全文展开会让 Ch 5、Ch 8、Ch 10、Ch 11、Ch 12 的开场失去"第一次说"的张力——读者已经知道答案了，后面章节只是补细节。建议把赌注 3-5 各压缩成 1-2 句话+章节指引。

**第 1 章 1.4 节"train.py vs train_async.py"**作为"深入剖析"框存在合理，但 "4 个 customization 参数复用同一个 train_async.py" 这段把 SFT/RL/PPO/GRPO 的差异说得很详——这其实是 Ch 2 §2.5 "启动脚本作为参数 recipe" 的话题。建议 Ch 1 这里只点出"任务类型不是文件名的事，是参数的事"这个结论，删掉 4 个具体参数的列表（保留到 Ch 2 §2.5 的"SFT 与 RL 的差异（节选）"代码块那里第一次说）。**必须改**——目前 Ch 1 和 Ch 2 §2.5 末尾确实重复了同一段（Ch 1 第 273-281 行 vs Ch 2 第 330-344 行讲的是同一件事）。

**第 2 章**。无明显冗余。260 行 validation 的解释（2.4 节末尾 248-271 行）很扎实，"断言 + 归一化 + 派生"三件事说得清楚，不要删。

**第 3 章**。**3.5 节"两套 model bridge 的并存"**和**3.4 节末尾"两个引擎都按自己的方式独立运行"+ 44 行的 sglang.py**——这两段都在讲"slime 不强求统一抽象"，主题重叠。建议把 3.5 压缩成 3.4 末尾的一个小段（3-4 句话即可），把"双轨制 bridge"从 5 节降级到 §3.4 一个段落，避免给一个 30 行的话题立独立小节。

---

## 4. 缺失内容

**第 1 章**。**1.2 节那张 5 角色 sequence 图**之后，正文说 "Hook：用户配置的 `custom_generate` 函数（如果有）"——但读者此时**不知道"hook"是什么**。Ch 1 没有解释这个词在 slime 里的具体含义，要等到 Ch 10 才正式定义。建议在第 156 行 "Hook" 这条 bullet 后加一句话："hook 是用户传入的 Python 函数路径（由 `--xxx-path` 参数指定），slime 用 18 个这种 hook 暴露扩展点；细节看 Ch 10。" 让读者知道"hook=可插拔函数"，而不是某种装饰器或事件机制。

**第 2 章**。**2.2 节末尾 skipped_args 白名单**（第 156-160 行）讲了"slime 接管不透传"，但没有解释**为什么是这几个**（`tp_size`、`port`、`nnodes`、`nccl_port`）。读者会问：为什么 `tp_size` 由 slime 算？建议加一句过渡："这几个的共同点是它们都依赖 slime 自己的并行拓扑——`tp_size` 由 `--rollout-num-gpus-per-engine` 推出来，`port`/`nccl_port` 由 Ray 分配。让用户传反而会和 slime 的 placement 冲突。"

**第 3 章**。**3.3 节 RolloutManager 起完之后**那段（225-242 行）讲了 3 件辅助动作（算 num_rollout_per_epoch、check_weights snapshot、offload_rollout）。但读者会困惑：**这 3 件事为什么不在主循环里？**Ch 1 §1.5 提过 `offload_rollout`，但没回答"什么时候执行"。建议在 3.3 节结尾加一句："这些是 for-loop 启动前必须就位的一次性状态——把它们留在主循环会让 Ch 1 那 100 行多出 10-15 行只在第一次迭代有用的代码。"明确点出 Ch 1 与 Ch 3 的设计权衡。

**跨章过渡**：Ch 2 → Ch 3 的"下一站"段落很短（438-443 行），只说"参数解析完之后调 `create_placement_groups`"。Ch 1 → Ch 2 的过渡（397-402 行）更有"诱饵感"——"你会看到 `--sglang-` 前缀是怎么用一个 monkey-patch 实现的"。建议 Ch 2 末尾的"下一站"也加一个具体的诱饵句，例如："你会看到 colocate 部署下 actor 和 rollout 怎么共享同一组 GPU 而不打架——答案是一个 `offset = 0`。"

---

## 5. 需要的图示

按"一张图能替代大段文字"的标准，**3 处建议补图，1 处建议删图**：

**建议补图 A：Ch 2 §2.1 三层 parser 的并行解析流程**

当前 Ch 2 §2.1 已经有一张 `flowchart TD`（62-78 行）讲三阶段，但它**只画了串行流**：CLI → Pre → SG → Mega → Merge → Val → Done。这张图没有体现"为什么是三个 parser 而不是一个"——也就是 SGLang 走独立 parser、Megatron+slime 共用 parser 的**职责切分**。

建议改为**一张分栏图**：左栏 SGLang parser（输入：`--sglang-*` 参数，输出：sglang_ns），右栏 Megatron parser（输入：所有其他参数 + `ignore_unknown_args=True` 容忍 SGLang 残余，输出：含 slime 自身 ~150 参数的 namespace），底部一个 merge 节点汇总到 `args`，再到 `slime_validate_args`。这样读者一眼看到"两个 parser 平行存在，merge 后才有 validate"。

```
建议类型：flowchart TD（带 subgraph 分栏）
节点：
  - subgraph "Phase 1: SGLang 独立 parser"
    - "ServerArgs.add_cli_args + monkey-patch add_argument"
    - "parse_known_args"
    - "sglang_ns（带 sglang_ 前缀）"
  - subgraph "Phase 2: Megatron+slime 共享 parser"
    - "Megatron 注册原生参数（~数百）"
    - "extra_args_provider=add_slime_arguments（~150 参数）"
    - "ignore_unknown_args=True"
    - "args（含 megatron+slime 参数）"
  - "merge sglang_ns into args"
  - "slime_validate_args：断言+归一化+派生"
  - "args 就绪"
```

**建议补图 B：Ch 3 §3.3 sidecar 拓扑图**

3.3 节讲 RolloutManager 是"0 GPU 的 CPU sidecar"，但 Ch 1 §1.2 的 sequence 图只展示了**调用关系**，没有展示**进程拓扑**——也就是"哪些是 Ray actor、哪些是它们 fork 出的子进程、谁占 GPU 谁不占"。这正是 Ch 3 开篇承诺的"读完这一章你应该能画出 slime 启动后的进程拓扑图"。

建议在 3.4 节末尾（或 3.5 之前）补一张 **`graph TD` 拓扑图**，画 3 类节点：
- **绿色（CPU only, 0 GPU）**：Driver (`train.py`)、RolloutManager actor、sgl-router subprocess、Lock actor、HealthMonitor
- **蓝色（GPU actor, fractional GPU=0.4）**：N 个 MegatronTrainActor、M 个 SGLangEngine actor
- **黄色（fork 出的子进程）**：每个 SGLangEngine 下面的 SGLang worker 子进程（scheduler、cache controller）

边只画"谁拥有谁"和"谁通过 HTTP/Ray 调用谁"两种关系。这一张图能替代 3.3 + 3.4 共约 60 行散文。

**建议补图 C：Ch 3 §3.1 colocate vs 分离的 bundle slice**

当前 3.1 节已有一张 `graph LR`（91-108 行）画了 colocate vs 分离，但它**只标了 bundle 数**，没有把"同一个 pg 用 offset 切片"这个关键洞见图形化。建议把这张图升级——用一个长条表示 PlacementGroup 的 bundle 数组，actor 角色用上方括号 `[0:N]`，rollout 角色用下方括号；colocate 时两个括号重叠（offset=0），分离时错开（offset=N）。Bundle 切片这个概念用"长条 + 区间括号"远比节点框直观。

**建议删的图**：Ch 1 §1.1 的 **gantt 图**（98-116 行）。这张图想展示"阶段 1-4 是顺序启动 + 阶段 5 是主循环"，但其实**读者在伪代码里已经看到这个顺序了**——5 个阶段用 5 个注释明确标出。gantt 图的"时间宽度"对启动序列毫无信息量（每段 1-3 秒都是随手编的），主循环那一段重复迭代也展示不出来。建议删掉，把"启动序列发生一次，主循环按 num_rollout 次数迭代"用一句话总结。

---

## 6. 跨章一致性

**Apply This 节 4 要素结构**（名称 → 解决什么问题 → 如何改造适配 → 陷阱）三章**完全一致**，每章 5 条模式，每条都有"怎么改造适配" + "陷阱"两个子标题。**无问题**——这是格式守得最好的一处。

**"伪代码"标注**。Ch 2、Ch 3 大量代码块标了 `# 伪代码 —— illustrative`，Ch 1 的伪代码（第 38-65 行 train.py 骨架）**只有标题里说"概念视图"，没有 illustrative 注释**。建议统一在 Ch 1 的 train.py 骨架代码块顶部也加一行 `# 伪代码 —— illustrative，省略 if/else 与异常处理`，和后两章一致。

**术语统一性**：
- "fractional GPU" 这个词 Ch 1（第 360 行）和 Ch 3（第 114、142 行）都用了，一致。
- Ch 1 把 RolloutManager 称为 "CPU sidecar"，Ch 3 §3.3 强化这个术语，一致。
- Ch 1 §1.3 表格里用 "1485 行的 RolloutManager"，Ch 3 §3.3 也说 "1485 行"，一致。
- **小不一致**：Ch 3 §3.4 末尾说 sglang.py 是 **44 行**（318 行），但实际是 43 行（已 `wc -l` 核对）。建议改成 "40 多行" 避免硬编码具体数字（codebase2book.md 也提醒"避免会过时的定量陈述"）。

**矛盾点**：
- Ch 1 §1.2 第 162-164 行说 "SGLang Engine 是另一个 Ray actor，它持有 GPU，但实际推理工作跑在它 `multiprocessing.fork` 出的子进程里"——这里 "持有 GPU" 措辞会让读者以为 Ray actor 进程内有模型权重。Ch 3 §3.4 第 288-294 行精确化为 "actor 进程里只起了一个 HTTP 客户端，模型实际跑在它 `multiprocessing.fork` 出来的子进程里"。**Ch 1 的措辞建议改为**："`SGLang Engine` 是另一个 Ray actor，它的 bundle 占着 GPU 资源声明，但实际推理跑在它 fork 出的子进程里——actor 进程本身只是个 HTTP 客户端。" 这样和 Ch 3 表述对齐，避免读者带着错误的 mental model 进入 Ch 3。

**赌注序号**：Ch 1 §1.3 列了 5 个赌注（赌注 1-5），Ch 2 开篇第 26 行说 "它是 slime 五条核心赌注里的第二条"。这是好的——赌注序号在两章间被引用，证明 Ch 1 的设论点起到了"全书定位"作用。建议 Ch 3 也明确回扣赌注（目前 Ch 3 §3.4 第 282-283 行用了 "native engine 透传赌注" 这个说法，但没标"赌注 2"）。统一在 Ch 3 出现透传相关段落时也加"（赌注 2）"括注，增强论点连贯性。

---

## 7. 具体修改（按问题严重程度排序）

下面列 12 条具体修改，前 4 条是**必须改**（结构/矛盾/事实），后 8 条是**建议改**（润色）。

### 必须改

**#1 [Ch 1 / Ch 2 重复段落] 删除 Ch 1 §1.4 的 4 个 customization 参数列表**

Ch 1 第 273-281 行：
> 至于跑什么任务，全靠 4 个 customization 参数复用同一个 `train_async.py`：
> - `--rollout-function-path`（rollout 形态）
> - `--loss-type`（loss 类型）
> - `--disable-compute-advantages-and-returns`（advantage 计算开关）
> - `--debug-train-only`（只跑训练侧）

这 4 条 bullet 和 Ch 2 第 332-338 行的 SFT 参数代码块讲同一件事。**保留 Ch 2 的版本**（那里是它的"主场"——讲启动脚本作为 recipe），**Ch 1 改为一句**："至于跑什么任务，全靠几个 customization 参数复用同一个 `train_async.py`——这是赌注 1（一条数据路径）在入口层的具体落点；第 2 章 §2.5 会给出具体参数清单。"

**#2 [Ch 1 §1.2 Ch 3 §3.4 矛盾措辞] 修正 SGLang Engine 持有 GPU 的描述**

Ch 1 第 162-164 行 "SGLang Engine 是另一个 Ray actor，它持有 GPU" → 改为 "SGLang Engine 是另一个 Ray actor，它的 bundle 声明占着 GPU 资源，但实际推理跑在它 `multiprocessing.fork` 出的子进程里——actor 进程本身只是个 HTTP 客户端。**第 3 章会展开这套'透传上游进程模型'的设计。**"

**#3 [Ch 3 §3.4 事实精确化] sglang.py 行数**

Ch 3 第 318 行 "整个文件 **44 行**" → 改为 "整个文件 40 多行"（实际 43 行），或者删掉"44 行"具体数字保留"整个文件极薄"的判断。

**#4 [Ch 1 §1.3 结构性删减] 压缩赌注 3-5**

Ch 1 第 226-247 行"赌注 3-5" 三段（每段 4-6 行散文）会抢后续章节的开场。建议改为一段紧凑总结：

> **赌注 3-5**：单一 SGLang backend（不在多推理引擎上做最小公约数，第 5、8 章）、customization 取代 fork（18 个 hook 让用户代码不 import slime ABC，第 10 章）、显式 dataflow + CI 一等公民（trace/metric/health 全做基础设施，第 11、12 章）。这三条共享同一种态度：**与其在框架里抽象，不如让框架把扩展点暴露为契约**——契约由参数声明，由测试验证。

把腾出来的 15 行用来强化赌注 1 和赌注 2 的总结。

### 建议改

**#5 [Ch 3 开篇剧透] 第 13-21 行的 4 条 bullet**

把"InfoActor 探测每个 bundle 实际落在哪个节点的哪张卡上 / 按 IP + GPU id 排序"这两条删掉（这些细节正文没展开），保留前 2 条即可。

**#6 [Ch 1 §1.2 hook 术语未定义] 第 156 行补一句过渡**

"Hook：用户配置的 `custom_generate` 函数（如果有）" 后加一句："hook 是用户提供的 Python 函数路径（由 `--xxx-path` 参数指定）——它们是 slime 暴露扩展点的统一形式，第 10 章详细讲。"

**#7 [Ch 2 §2.3 路标缺失] 2.3 开头补一句**

第 162 行 "Megatron 不能用 SGLang 那套技巧"之前补一句："slime 把自己的 ~150 个参数寄生到 Megatron 的 parser 里靠三招：`extra_args_provider`、`reset_arg`、`ignore_unknown_args=True`。下面依次看。" 然后让现有的三段自然对应三招，读者就不需要回头数"第三个钉子"。

**#8 [Ch 2 §2.2 skipped_args 解释]**

第 156-160 行末尾补一句："这几个参数的共同点是它们的值由 slime 自己的并行配置推出来（`tp_size` 由 `--rollout-num-gpus-per-engine` 算，`port`/`nccl_port` 由 Ray 分配）——让用户传反而会和 slime 的 placement 冲突。"

**#9 [Ch 3 §3.3 主循环窗口的解释]**

第 244-246 行那段 "这几件事都不在主循环里——它们是 for-loop 开始之前必须就位的状态" 已经讲了，但建议加一句把 Ch 1 的设计权衡显式化："slime 把一次性的协调动作放在 placement 与 train actor 创建之间这个窗口里，是它能让主循环保持 100 行的关键之一——主循环不需要 if `is_first_step` 来跑只在第一次有用的代码。"

**#10 [Ch 3 §3.5 降级]**

把"两套 model bridge 的并存"这一节降级为 3.4 末尾的一个小段，标题改成行内的强调而不是 §3.5。当前篇幅（10 行）撑不起一个独立小节。

**#11 [Ch 1 §1.1 gantt 图删除]**

按维度 5 的建议，删除第 98-116 行的 gantt 图，用一句话替代："启动序列只发生一次（阶段 1-4），主循环按 `num_rollout` 次数迭代（阶段 5）。前后总共 103 行代码。"

**#12 [Ch 2 末尾过渡诱饵]**

第 438-443 行 "下一站" 段落改为：

> 参数解析完之后，`train.py` 的下一行是 `pgs = create_placement_groups(args)`——这一行触发的是 slime 把 args 翻译成 Ray placement bundles + 启动 Megatron 与 SGLang 引擎的整套启动序列。下一章会看到 colocate 部署下 actor 和 rollout 怎么共享同一组 GPU 而不打架——答案是 `_get_placement_group_layout` 里 `offset = 0` 这一行。

---

## 评审小结

**整体质量：高**。三章读起来流畅，论点贯穿，Apply This 节质量稳定，伪代码符合写作规范（除了 Ch 1 漏了 `illustrative` 注释）。

**最值得修的 3 个问题**：
1. **Ch 1 §1.3 赌注 3-5 过度展开** → 抢了后续章节的开场，必须压缩（修改 #4）。
2. **Ch 1 / Ch 2 重复了"4 个 SFT 参数"段落** → 保留 Ch 2，Ch 1 改一句话（修改 #1）。
3. **缺一张进程拓扑图** → Ch 3 开篇承诺读完能画出拓扑，但全章没给图（建议补图 B）。

**不需要改的部分**：Apply This 节的格式与质量、Ch 2 §2.4 关于"validation = 断言+归一化+派生"的阐述、Ch 2 §2.5 启动脚本作为 recipe 的论证、Ch 3 §3.4 关于"两个引擎按自己的方式独立运行"的洞见——这些都是本部最扎实的内容，保留。
