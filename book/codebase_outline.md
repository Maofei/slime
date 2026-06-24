# 提示词：为 slime 框架生成技术书大纲

本提示词的目标是**只产出大纲**，不写正文。大纲质量直接决定后续写
作能否落地，所以这一步要做扎实：每一章都必须有清晰的问题、明确的
代码锚点、可参考的官方文档与示例，并且整本书要有一个贯穿的论点。

---

## 提示词

````
我要为位于 `/Users/quemaofei/Documents/Projects/slime` 的 slime 框架
写一本面向资深工程师的技术书（中文写作）。请你先**只产出一份大纲**，
写到 `book/outline.md`，等我评审通过后再讨论正文写作。

不是文档目录，不是教程列表。是一本技术书的目录——每一章都对应一个
**清晰的问题**，章节之间形成**累进的叙事**，整本书围绕**一个核心
论点**展开。

---

## 第一步：在动笔写大纲之前，先读这些

**官方论点（不要重新发明）**：
- `README_zh.md` 与 `docs/zh/index.rst` 的开头部分，特别是这几个论点：
  - 高性能训练 + 灵活的数据生成两大能力**彼此强化**，避免变成三套
    割裂的 trainer / rollout service / agent framework
  - 设计上就是 **native**：直接透传 Megatron 参数与 `--sglang-` 前缀
    参数，新上游优化无需 wrapper
  - 专注 SGLang 单一 rollout backend，避免 lowest-common-denominator
    抽象
  - Agentic workflow 就是数据生成——tool use、sandbox、verifier、
    multi-agent 都流经同一条 training / rollout / Data Buffer 路径，
    **不 fork training kernel**
  - 实战验证：GLM-4.5/4.6/4.7/5/5.1/5.2 背后的 RL 框架

**官方话题拓扑（可作为章节参考，但不要照抄目录）**：
- `docs/zh/get_started/`：quick_start、usage、customization、agent、qa
- `docs/zh/advanced/`：sglang-config、megatron-config、pd-disaggregation、
  external-rollout-engines、delta-weight-sync、low-precision、
  on-policy-distillation、speculative-decoding、reproducibility、
  fault-tolerance、observability、arch-support-beyond-megatron
- `docs/zh/developer_guide/`：ci、debug、trace、profiling
- `examples/`：coding_agent_rl、retool、multi_agent、search-r1、
  tau-bench、strands_sglang、fully_async、on_policy_distillation、
  geo3k_vlm、geo3k_vlm_multi_turn、delta_weight_sync、eval_multi_task、
  train_infer_mismatch_helper

**入口与主循环**：
- `train.py`（同步 RL，103 行）—— 整本书的"主线代码"，几乎所有概念
  都能挂到这个 100 行的循环上
- `train_async.py`（异步 RL，80 行）
- `slime/utils/arguments.py`（2010 行）—— 配置面

---

## 第二步：确认子系统边界

代码结构已经核对过，按下面这些桶组织（每个桶对应代码里真实存在的
目录或一组紧密相关的文件；用 `find` / `grep` 校验后再下笔）：

**核心三模块**（README 概念架构图所示，**注意 Data Buffer 不是独立
目录**）：
- **training (Megatron)**：`train.py` 主循环 → `slime/ray/train_actor.py`
  → `slime/backends/megatron_utils/actor.py` + `loss.py` + `data.py` +
  `model.py` + `cp_utils.py`；CKPT 在 `checkpoint.py`、`hf_checkpoint_saver.py`
- **rollout (SGLang + router)**：`slime/ray/rollout.py`（1485 行的
  rollout manager）+ `slime/rollout/sglang_rollout.py` +
  `sglang_streaming_rollout.py` + `sft_rollout.py` + `fully_async_rollout.py`
  + `on_policy_distillation.py` + `sleep_rollout.py`；router 是上游
  `sgl-router`（主仓没有 `slime/router/`）
- **Data Buffer**：`slime/rollout/data_source.py`（含 `RolloutDataSourceWithBuffer`）
  + `slime_plugins/rollout_buffer/buffer.py` + `generator/`；
  prompt 生命周期、`forge_load.py`、dynamic filter

**横切与基础设施**：
- **agent harness**：`slime/agent/` 整目录——`trajectory.py`、`sandbox.py`、
  `parsing.py`、`aiohttp_threaded.py`，以及 `harness/`（含 `claude_code.py`、
  `codex.py`、`common.py`）和 `adapters/`（`anthropic.py`、`openai.py`、
  `common.py`）。slime 自带两个 agent harness 实现，是讲 agentic RL
  最具体的素材。
- **backends 适配层**：
  - Megatron 侧：`slime/backends/megatron_utils/` 全部，重点是
    `initialize.py`、`arguments.py`、`actor.py`、`server/`、`megatron_patch/`
  - SGLang 侧：`slime/backends/sglang_utils/`——`sglang_engine.py`、
    `sglang_config.py`、`server_control.py`、`external.py`、`arguments.py`
  - 模型 bridge：`slime_plugins/mbridge/`、`slime_plugins/megatron_bridge/`
  - 模型特定 kernel：`slime_plugins/models/`（含 glm5 sparse_mla、
    qwen_gdn_backend、flash_dot_product_attention 等）
- **Ray 编排层**：`slime/ray/` 全部——`placement_group.py`（246 行，
  GPU 分配与 placement strategy）、`actor_group.py`（169 行）、
  `train_actor.py`（128 行）、`rollout.py`（1485 行）、`ray_actor.py`、
  `rollout_validation.py`
- **weight sync**：`slime/backends/megatron_utils/update_weight/` 五个
  文件分别对应不同传输路径——`update_weight_from_disk.py`、
  `update_weight_from_distributed.py`、`update_weight_from_tensor.py`、
  `update_weight_from_distributed_delta.py`、`common.py`、
  `hf_weight_iterator_*.py`；加上 `megatron_to_hf/` 里的 10 个模型转换器
  与 `processors/`（FP8、quantization）
- **deployment / topology**：`slime/backends/sglang_utils/sglang_config.py` +
  `external.py`；对应文档 `docs/zh/advanced/sglang-config.md`、
  `pd-disaggregation.md`、`external-rollout-engines.md`
- **customization 接口**：四类 hook
  - `--rollout-function-path` → 自定义 rollout
  - `--custom-generate-function-path` → 自定义 generate
  - `--custom-rm-path` → 自定义 reward（hub：`slime/rollout/rm_hub/`
    含 deepscaler / f1 / gpqa / ifbench / math_utils / math_dapo_utils）
  - dynamic filter → `slime/rollout/filter_hub/dynamic_sampling_filters.py`
  - eval dataset config → `slime/utils/eval_config.py`
- **CLI / 参数 / 启动入口**：`slime/utils/arguments.py`（2010 行）+
  顶层 `train.py` / `train_async.py` + `examples/` + `scripts/run-*.sh`
- **工程基础设施**：
  - 可观测：`slime/utils/trace_utils.py`、`metric_utils.py`、
    `train_metric_utils.py`、`health_monitor.py`、`profile_utils.py`、
    `timer.py`
  - 测试：`tests/` 反映了团队认为必须守住的不变量——cispo_loss、
    chunked_gae、cp_invariance、placement_group、metric_report_dist、
    full_disk_weight_update、external_sglang_engines 等
  - CI / debug / trace / profiling：`docs/zh/developer_guide/` 四篇

---

## 第三步：定义全书论点（thesis）

在列章节之前，**先用 3-5 句话写出全书的论点**。这个论点必须：
- 回答"读完这本书，读者能带走的最大洞见是什么"
- 与每一章直接挂钩（每章都应当回扣论点的一部分）
- 不重复 README 已经说过的话——但建立在那些已确立的论点之上

slime 的天然论点是："统一的 train ↔ rollout ↔ Data Buffer 路径
+ 原生 engine 透传，让 RL post-training 不再是三套割裂系统的胶水
代码"。但你可以提出更尖锐或更具体的角度，只要能贯穿全书。

---

## 第四步：产出大纲

把大纲写到 `book/outline.md`，结构如下：

```markdown
# slime 技术书大纲

## 全书论点
（3-5 句话，回答"读者能带走的最大洞见"）

## 目标读者
- 主要：……
- 次要：……
- 不适合：……（说清楚谁不应该读这本书，能反向定义读者）

## 阅读路径
- 技术领导者路径：第 X、Y、Z 章 + 各章 Apply This
- 资深工程师路径：通读
- 想做 agentic RL 的读者：第 A、B、C 章
（让两类读者都能找到自己的入口）

## 全书结构

### 第 N 部：部名
> 一句话引语，奠定本部基调

#### 第 N 章：章名
**问题**：本章要回答的问题（一句话）
**读者带走**：读完本章读者能做什么或理解什么（一句话）
**覆盖**：
- 内容要点 1
- 内容要点 2
- 内容要点 3（3-5 条）
**代码锚点**：
- 主要文件：`path/to/file.py`（含关键 class / function）
- 关键参数：`--xxx-yyy`
**官方文档参考**：`docs/zh/<path>.md`（如有）
**示例**：`examples/<name>/`（如适用）
**与其他章节的关系**：依赖第 M 章；被第 K 章引用

（对每一章重复上面的结构）
```

---

## 大纲质量自检（提交前过一遍）

- [ ] **论点清晰**：一句话能说出全书在讲什么、为什么值得读
- [ ] **章节累进**：第 N 章只依赖第 1..N-1 章的概念，不前向依赖
- [ ] **代码锚点真实**：每个引用的文件路径都用 `find` / `ls` 验证过
- [ ] **论点回扣**：每章的"读者带走"都可以追溯回全书论点的某个方面
- [ ] **可跳读**：技术领导者只读章首/章尾仍能 get 到关键信息
- [ ] **覆盖均衡**：核心循环（training loop / rollout loop / weight sync）
  得到足够篇幅；不重要的话题不要凑章
- [ ] **每章独立题目**：章名不重复（避免"深入 X"、"再谈 X" 这种）
- [ ] **示例真实**：每个引用的 `examples/<name>/` 目录都存在
- [ ] **不重复 docs/zh/**：本书不是把 `docs/zh/` 重排一遍——如果一章
  只是某个 doc 的复述，删掉或与其他章合并；本书的增量价值是叙事、
  横切模式、设计原理与可迁移经验

## 章节排序建议（可调整，但要说明理由）

代码与文档的自然顺序大致是：
1. **入口与配置**：`train.py` 100 行主循环 + `arguments.py` 配置面
2. **运行时基础**：Ray placement / actor group / 引擎初始化（backends
   适配层 + Ray 编排层）
3. **核心循环**：training loop + rollout loop（两章，是全书最重要的
   两章）
4. **数据桥梁**：Data Buffer 在 train ↔ rollout 之间如何流转
5. **训推同步**：weight sync 的四条传输路径与 delta sync
6. **部署能力**：SGLang config / PD disaggregation / external engines /
   router
7. **agent harness**：slime 内置 harness 的执行机制（claude_code、
   codex、trajectory、sandbox）
8. **扩展契约**：四类 hook 如何让用户接入自己的 agent / tool /
   verifier / filter
9. **工程基础设施**：trace、profiling、metric、health、CI、reproducibility、
   fault tolerance——为什么这些在 RL 项目里是一等问题
10. **性能与正确性**：BF16+FP8 混合精度、CP 一致性、loss 不变性、
    瓶颈分析
11. **结语**：综合提炼、可迁移经验、前瞻

如果你认为有更好的排序（比如把 agent harness 提前、把 weight sync
拆成两章），**列出来并说明理由**。

## 边界澄清（要在大纲里明确）

- **agent harness 章 vs. 扩展契约章**：harness 章讲 slime 内置实现
  （系统提供什么——trajectory 数据结构、sandbox 协议、HTTP harness
  并发模型）；扩展契约章讲用户怎么通过 hook 接入（系统暴露什么扩展
  点）。同一个 multi-turn 示例可以在两章用，视角不同。
- **backends 适配层 vs. weight sync 章**：backends 章讲引擎初始化
  与参数透传；weight sync 章讲已经初始化的两个引擎之间如何同步参数。
- **deployment 章 vs. backends 章**：backends 章讲单进程引擎；
  deployment 章讲跨节点的拓扑、router、PD 解耦、external engines。

## 部（Part）的分组

把上面的章节合并成 5-7 个"部"，每部一句话引语。建议合并思路：
- "起步"部：入口与配置 + 运行时基础（Ray + backends）
- "核心循环"部：training loop + rollout loop
- "数据流与同步"部：Data Buffer + weight sync
- "部署与扩展"部：deployment + agent harness + 扩展契约
- "工程实践"部：工程基础设施 + 性能与正确性
- 收尾：结语

（这只是建议，你可以自己分。说清理由。）

---

## 大纲输出之后

写完 `book/outline.md` 之后，**停下来等我评审**。不要直接开始写正文。
我会通读大纲、提出修改意见，定稿后再讨论正文。
````
