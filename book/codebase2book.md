# 提示词：将任意代码库转化为一本技术书籍

一个可复用的系统，用于通过并行 AI 智能体分析代码库并产出出版级别的技术书籍。设计目标是产出可与 O'Reilly 技术书媲美的成果。

---

## 提示词

````
我希望你分析位于 `/Users/quemaofei/Documents/Projects/slime` 的源代码，
并就其架构、模式和内部机制产出一本全面的技术书籍。

这本书的阅读体验应当像一本专业的技术出版物——那种资深工程师愿意
购买、用来深入理解某个系统的书。不是文档。不是教程。而是一本教
读者理解系统如何工作、每个决策为何如此、以及读者可以借鉴到自己
项目中的模式的书。

---

## 项目背景：slime 子系统提示

slime 是面向 RL scaling 的 LLM post-training 框架。按官方架构，核心是
`training (Megatron)` ↔ `Data Buffer` ↔ `rollout (SGLang + router)` 三个
模块互相驱动（注意：这是 README 的概念架构图，不等同于代码目录结构——
Data Buffer 并非独立目录，见下方说明）。建议阶段 1 的并行智能体按以下
子系统划分（先用 `find` / `grep` 实地核对目录结构再决定边界）：

**所有阶段产物输出到 `book/` 目录**：阶段 1 的研究笔记写到 `book/notes/`
（每个子系统一份 markdown），阶段 3 的大纲写到 `book/outline.md`，阶段 4
的最终书稿写到 `book/chapters/`（每章一个文件，文件名带章节号）。

**核心三模块**（README 架构图所示，需要查阅代码确认）：

- **training (Megatron)**：主训练循环、Megatron 参数透传、优化器与
  checkpoint、训练完成后向 rollout 同步参数。顶层入口是 `train.py`（RL）
  与 `train_async.py`（异步 RL），SFT 走的是 `slime/rollout/sft_rollout.py`
  内部分支，不在 entry script 层面分。
- **rollout (SGLang + router)**：样本生成、SGLang engine 对接、router
  policy（含 session affinity）、multi-turn / agentic loop、reward /
  verifier、custom generate。**注意：slime 主仓没有 `slime/router/`
  目录**，router 是上游 `sgl-router`，slime 这边的对接代码在
  `slime/rollout/sglang_rollout.py`——智能体需同时参考 `sgl-router`
  仓库和 `sglang_rollout.py` 才能讲清 router policy。
- **Data Buffer**：训练与 rollout 之间的桥梁、prompt 初始化、样本生命
  周期管理、dynamic filter。**注意：slime 没有独立的 `slime/data_buffer/`
  目录**，主实现散落在 `slime/rollout/data_source.py`
  （`RolloutDataSourceWithBuffer` 类）与 `slime_plugins/rollout_buffer/`，
  智能体需顺这两个入口横向追踪。

**横切与基础设施子系统**：

- **agent harness**：`slime/agent/` 下的 `harness/`、`trajectory.py`、
  `sandbox.py`、`adapters/`、`parsing.py`、`aiohttp_threaded.py`——agentic
  RL 与 tool use 的执行层，是 agentic RL 章的主要素材来源
- **backends 适配层**：`slime/backends/megatron_utils/` 与
  `slime/backends/sglang_utils/`，以及 `slime_plugins/mbridge/`、
  `slime_plugins/megatron_bridge/`——training / rollout 与具体引擎之间
  的适配层，也是"原生 engine 透传"论点的落点
- **Ray 编排层**：`slime/ray/` 下的 `placement_group.py`、`actor_group.py`、
  `train_actor.py`、`rollout.py`、`ray_actor.py`——横跨 training 与 rollout
  的编排层
- **weight sync**：Megatron → SGLang 参数同步，包含 delta weight sync、
  NCCL / 共享文件系统两条传输路径
- **deployment / topology**：SGLang Config（YAML 扩展）、PD Disaggregation、
  External Rollout Engines、heterogeneous server group
- **customization 接口**：`--rollout-function-path`、
  `--custom-generate-function-path`、`--custom-rm-path`，以及
  `slime/rollout/rm_hub/`（reward / verifier 实现）、
  `slime/rollout/filter_hub/`（dynamic filter hook）、eval dataset
  config——slime 的核心扩展契约
- **CLI / 参数 / 启动入口**：`slime/utils/arguments.py`、顶层 `train.py` /
  `train_async.py`、`examples/`——读者从命令行到第一个 step 的路径
- **工程基础设施**：CI、debug、trace viewer、profiling、reproducibility、
  fault tolerance——RL 项目里这些是一等问题

---

## 阶段 1：探索

启动并行智能体覆盖上方列出的全部子系统桶。子系统数较多时可以分批：
核心三模块 + agent harness + Ray 编排层 + backends 适配层 优先并行
（这几个互相耦合最紧，需要相互参考），weight sync / deployment /
customization / CLI / 工程基础设施 可以在第二批跑。每个智能体应通读
其负责子系统的核心文件；辅助/测试/边角文件可以只扫读，但在笔记里
注明哪些文件只扫读、哪些没看。记录：

- 架构和模块边界
- 关键抽象（类型、接口、核心类）
- 数据流（信息如何在系统中流动）
- 设计模式（使用了哪些模式以及原因）
- 集成点（这个模块如何与其他模块连接）
- 出人意料的决策（任何非显而易见或巧妙的设计）

为每个子系统产出一份原始分析文档。这些是研究笔记，不是最终的
书稿。

## 阶段 2：受众与定位

在构建书籍结构之前，先定义：

**主要受众**：这本书是为谁写的？他们已经掌握了什么？他们想学
什么？这本书应当同时服务于两类读者：
- 想要了解架构与设计原理的技术领导者（可以跳过代码块和深入剖析
  小节）
- 想要在实现层面理解系统的资深工程师（通读全文，包括深入剖析
  部分）

**核心论点**：关于这个系统，那个"唯一的大洞见"是什么？每一章
都应当回扣这个论点。对于一个代码库来说，通常是："这是这个系统
在架构上下的赌注，以及每个子系统是如何服务于这个赌注的。"

> slime 提示：在动笔之前先读 `README_zh.md` 的 "为什么这个设计
> 重要" 与 "原生 Engine 透传与 SGLang 部署" 两节——slime 的核心
> 论点（统一 train ↔ rollout ↔ Data Buffer 流转 + 原生 engine
> 透传 vs. lowest-common-denominator 中间层）已经被官方清晰阐述，
> 不要重新发明。

**为什么值得写成一本书**：为什么读者不能直接读源码？这本书的
价值在于：叙事（源码没有叙事）、横切模式（在源码中分散于多个
文件）、设计原理（在代码中根本没有体现），以及可迁移的经验
（需要源码无法提供的综合提炼）。

## 阶段 3：结构

把这本书组织得像读者要从零搭建这个系统一样。每一章都应当解决
一个清晰的问题，而下一章依赖于这个问题已被解决。尽量避免让读者
在前面的章节里被迫理解后面章节才会展开的概念——确实需要前向引用
时，用一句话点到即可，不要展开。

**部（Part）**：将章节分组为 5-7 个主题部分。每一部都有一句话
的引语来奠定基调。"部"的划分形成了天然的阅读断点，并让目录更
易于扫读。下方"章节排序原则"列出的章节桶可以在分部时合并
（例如 `Ray 编排` 与 `backends 适配层` 可合并为"运行时基础"部，
`数据桥梁` 与 `训推同步` 可合并为"数据流与同步"部，
`工程基础设施` 与 `性能与正确性` 可合并为"工程实践"部），不必
强行一桶一部。

**章节排序原则**（针对 slime 这样的 RL 训练框架；多数章节对应上方
"子系统提示"中的一个桶，training 与 rollout 合并到"核心循环"一条，
末尾的"性能与正确性"、"结语"是综合性章节，不来自子系统桶）：
- 先讲基础：参数系统与启动入口（`slime/utils/arguments.py` + `train.py` /
  `train_async.py`）
- Ray 编排层：placement_group、actor_group、train_actor、ray_actor——
  actor / rollout / ref 角色如何被拉起与协调
- backends 适配层：Megatron 与 SGLang engine 的初始化与对接
  （`slime/backends/*`、`slime_plugins/mbridge|megatron_bridge`），
  落实"原生 engine 透传"论点
- 接着是核心循环：training loop 与 rollout loop 分别讲清楚（通常是全书
  最重要的两章）
- 数据桥梁：Data Buffer（`data_source.py` + `slime_plugins/rollout_buffer/`），
  prompt 生命周期、样本如何在 train ↔ rollout 之间流动、dynamic filter
- 训推同步：weight sync、delta weight sync、NCCL 与共享文件系统两条
  传输路径
- 部署能力：SGLang Config、PD Disaggregation、External Rollout Engines、
  router policy 与 session affinity
- agent harness：`slime/agent/` 下的 harness / trajectory / sandbox /
  adapters / parsing——agentic RL 的执行层与生命周期
- 扩展契约：customization 接口（`--rollout-function-path` /
  `--custom-generate-function-path` / `--custom-rm-path`、rm_hub /
  filter_hub / eval dataset config），如何在不改 training kernel 的前提
  下接入 multi-turn、tool use、sandbox、verifier、multi-agent workflow
- 工程基础设施：CI、debug、trace viewer、profiling、reproducibility、
  fault tolerance——为什么这些在 RL 项目里是一等问题
- 性能与正确性：什么时候哪一环会成为瓶颈，slime 做过哪些权衡
- 结语：综合提炼、可迁移的经验教训、前瞻展望

> **agent harness 章与扩展契约章的边界**：agent harness 章讲 slime
> 内置 harness 的实现机制（系统提供什么——trajectory 数据结构、
> sandbox 协议、HTTP harness 的并发模型）；扩展契约章讲用户怎么通过
> hook 接入自己的 agent / tool / verifier（系统暴露什么扩展点）。两章
> 视角不同，可以共享同一个 multi-turn 示例——harness 章看它如何被
> 框架执行，扩展契约章看它如何被用户接入；这是视角切换，不违反"每个
> 概念唯一主场"原则。

**章节体量**：多数章节落在 6000-15000 字（约 300-800 行 markdown）是
舒适区。核心循环、weight sync 这类章可以更长；结语、概览性的章可以更
短。明显失衡时再考虑拆/合，不要为凑字数注水或硬切叙事。

完整展示大纲，包含部名、章名以及每章 2-3 句话的覆盖范围说明。
获得批准后再开始写作。

## 阶段 4：写作

把阶段 1 的分析当作素材库，不要直接拼接。每一章从零开始，按本章的
叙事弧线重新组织内容，让笔记里的事实流入散文。

### 章节模板

每一章都遵循以下结构：

1. **开篇**（简短，通常 2-4 段）
   - 这个层次/子系统解决什么问题？
   - 它为什么存在？没有它会出什么问题？
   - 它如何与读者已知的内容衔接？（明确回指上一章）
   - 读者在本章结束时将理解什么？

2. **正文**（核心内容）
   - 散文、图示、代码片段和表格的混合
   - 散文用于叙事和原理阐述（"为什么"）
   - 图示用于架构、数据流和状态机
   - 代码用于关键模式（仅伪代码——见下方规则）
   - 表格用于参考性资料（字段清单、配置选项）

3. **深入剖析小节**（可选，行内）
   - 提示框形式的小节，用于领导者可以跳过的实现细节
   - 比如 NCCL 通信细节、CUDA stream 调度、Ray actor 重启语义、
     某个 race condition 的处理——读懂主线不必看，但深度读者会想看
   - 应当能够独立阅读而不破坏本章的叙事

4. **Apply This（实战借鉴）**（收尾小节）
   - 从本章中提炼可迁移的模式，目标 3-7 个；宁可少而精，不要凑数
   - 每个模式：名称 → 解决什么问题 → 如何改造适配 → 需要注意
     的陷阱
   - 足够具体到可以立即上手，又足够抽象可以迁移到其他系统
   - 在不同章节中略微变化形式以避免单调

### 风格与语调

- **专家同行**：像一位资深工程师在为同事做深度技术评审。不学术、
  不教学、不营销。
- **直接且有立场**："这一点很巧妙，因为……" / "对于……这是错
  误的抽象" / "这个东西存在的原因是……"
- **没有水分**：每个句子要么在教读者一件事，要么在为下一句教学
  做铺垫。如果一句话没有挣得它的位置，就删掉。
- **展示权衡**：不要只描述构建了什么。解释那些**没有**构建的
  东西以及原因。未走的路往往更有启发性。

### 代码块

- **仅使用伪代码**：绝不照搬源码原文。展示**模式**，而不是
  实现。
- **少即是多**：典型章节 3-5 个代码块、每块 5-15 行已经够用；复杂章节
  （如 weight sync、Ray 编排）可以更多，但每多一块都要问自己"散文是
  否更清楚"。
- **使用不同的变量名**：用能说明概念的通用名字，而不是源码中
  的精确标识符。
- **明确标注为示例性质**：加上注释如 `// Pseudocode — illustrates the pattern`
  或 `// Simplified for clarity`。
- **前后铺垫**：代码块前用一句话说明它**展示了什么**。代码块
  后用一段话解释**为什么**这个模式重要。

### 图示

- **Mermaid 格式**：使用 ```mermaid 围栏代码块。这种格式在
  GitHub 和大多数 Web 框架中可以原生渲染。
- **图能替代散文时就画**：数据流、状态机、决策树、时间线、组件关系
  通常一张图胜过一段话；纯文字能讲清的概念则不必硬塞图。
- **可使用的图示类型**：
  - `graph TD` / `graph LR` 用于架构和数据流
  - `sequenceDiagram` 用于请求/响应流程和生命周期
  - `stateDiagram-v2` 用于状态机
  - `flowchart TD` 用于决策树和流水线
  - `gantt` 用于时间线和并行执行
- **图示密度**：典型章节 2-4 张图。复杂章节（training/rollout 核心循环、
  Ray 编排层、backends 适配层、weight sync、agent harness、customization
  接口、部署 topology）按需更多；聚焦章节可以只有 1 张甚至零张——一张
  不能替代散文的图不如不画。
- **Mermaid 语法常见陷阱**（写完一张图后渲染验证一次，避免下面这些）：
  - **节点标签里的双横线**：`--custom-generate-function-path` 这种带
    `--` 的字符串放进 `[...]` 节点标签会被 mermaid 当作箭头解析。改
    用引号包：`["--custom-generate-function-path"]`，或在文本里去掉
    前导双横线。
  - **节点标签含特殊字符**：标签里出现 `:` `/` `(` `)` `+` 等字符时
    用引号包，例如 `Ch3["Ch 3: Ray + engines"]` 而不是
    `Ch3[Ch 3: Ray + engines]`。
  - **边标签的多行写法**：边标签里 `<br/>` 在很多渲染器中不支持。要
    多行就改成节点；要单行用 `A -->|label| B` 形式，不要写
    `A -.label<br/>line2.-> B`。
  - **箭头指向 subgraph 名**：在旧版 mermaid 里 `A --> SubGraphName`
    可能不渲染。要稳妥就让箭头指向 subgraph 内的某个具体节点，而不是
    subgraph 本身。
  - **节点 ID 不要与 mermaid 关键字撞**：避免用 `end`、`graph`、
    `subgraph`、`style` 作节点 ID；**sequenceDiagram 里 `Actor`
    也是关键字**（与 `actor` 关键字 case-insensitive 冲突），不能
    用 `participant Actor` 这种写法，改用 `Train`、`TA` 等。
  - **sequenceDiagram 的 participant 别名要简洁**：`participant X as
    复杂名称` 在显示名称含空格、`<br/>`、括号时容易解析失败。最稳
    的写法是只用短 ID（`participant RM`），把全称放到图下方散文里
    说明。同理 `Note over` 块也建议省略，散文比图内注释更可靠。

### 交叉引用

- 每章开头都显式回指上一章（第一章除外，可改为定义全书的核心论点
  与阅读路径）
- 当某个概念会在后续展开时，使用前向引用："第 N 章会深入介绍
  这个内容"
- 每个概念有**唯一**的"主场"位置——其他章节只引用，不重复讲解。
  例外：同一段代码或同一个示例可以在不同章节出现，只要每次切换了
  视角（比如内部实现 vs. 用户接入）；这不是"重复讲解"

### 一致性检查

- 各章之间没有重复的修辞性套话
- "Apply This" 小节格式统一（标题、每个模式的四要素结构一致）
- 避免会过时的定量陈述（如"截至 N 个 example"、"共 M 个参数"、
  "支持 X 个 reward function"）；文件路径作为深度阅读锚点可以保留
- 全书术语保持一致

## 阶段 5：编辑评审

启动若干评审智能体（按"部"切分，让每个评审者负责的篇幅大致均衡）。
每个评审者要评估：

1. **开篇质量**：是否抓人？是否衔接上一章？
2. **流畅性**：是否有拖沓、重复、罗列事实但没有引向洞见的段落？
3. **内容删减**：哪些是参考手册式、不服务叙事的内容？哪些代码
   块太长？
4. **缺失内容**：是否有让读者困惑的空白？是否缺少过渡？
5. **需要的图示**：具体哪些地方一张图可以替代大段文字。详细
   描述每张图。
6. **跨章一致性**：风格、格式、术语、矛盾点。
7. **具体修改**：列出要重写的句子/段落（按问题严重程度排序），并给
   出原因。

每个评审者把意见写到 `book/reviews/<部名>.md`，最后汇总成
`book/reviews/action_plan.md`（按优先级排序的行动计划）。

## 阶段 6：修订

按 `action_plan.md` 的优先级依次应用评审反馈（结构性改动先于内容
微调，全书一致性问题最后兜底）：

- **结构性改动**：拆分/合并章节，修复断裂的引用，如有必要补上
  缺失的收尾章节
- **去重**：每个概念只讲解一次，其他地方交叉引用
- **内容删减**：删除单纯枚举（保留模式），精简臃肿的小节，把
  参考性内容压缩成表格
- **内容增补**：可上手的示例、真实的钩子示例、在指出的位置补
  图示
- **一致性**：统一 "Apply This" 小节，修正重复的套话，校核交叉
  引用

## 阶段 7：代码块审计（教学清晰度，非版权审计）

阶段 4 的"仅使用伪代码"规则已经在写作时执行；本阶段是出版前的 QA
兜底——slime 是开源项目，**不是为了"防止源码外泄"，而是为了教学清晰**：
逐字粘贴源码会把书退化成在线代码阅读器，失去叙事价值。逐块过一遍，
只修不达标的，不要重写已经合格的代码块：

- **替换**任何与源码逐字或几乎逐字一致的代码块，改写为更短、突出
  模式本身的伪代码（用不同的、能体现概念的变量名）
- **标注**函数 / 类型签名加上 `// Illustrative` 注释，让读者知道这是
  用来说明结构而非可直接运行的版本
- **保留**指向真实源码路径的引用（例如 `slime/utils/arguments.py`），
  让深度读者能跳回去读完整实现

代码块的价值在于**让模式可见**——能用 8 行说清楚的东西不要用 40 行。
````
