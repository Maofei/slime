# 研究笔记：customization 接口（18 个函数路径 hook + 契约测试）

> 对应章节：第 10 章 "扩展契约——18 个 hook 与契约测试"
> 范围：所有 `--*-path` 风格的 hook 注入点、`rm_hub` / `filter_hub` 内置实现、`tests/plugin_contracts/` 契约测试套件、`docs/zh/get_started/customization.md`。

## TL;DR

- slime 的扩展模型不是 plugin 系统、不是抽象基类、不是 decorator 注册表，而是**一套通过 `--xxx-path "module.sub.function"` 字符串路径在运行时 `importlib` 加载的 hook 函数**（`slime/utils/misc.py:37` 的 `load_function` 全栈通用）。一共 **18 个对外暴露的 hook 入口**，全部集中在 `slime/utils/arguments.py` 中注册为 CLI 参数，调用点散布在 rollout、reward、loss、Megatron 训练 step 等阶段。
- 18 个 hook 按生命周期可分成 9 组：rollout 编排、生成、奖励、过滤、数据源、训练 loss/advantage/TIS、Megatron 训练 step、日志、MoE routing replay。其中 **"core 4"**（`--custom-generate-function-path` + `--custom-rm-path` + `--dynamic-sampling-filter-path` + `--data-source-path`）覆盖了官方文档明确推荐的绝大多数 agentic / RL 训练扩展场景。
- 设计上最反常的一点是契约测试（`tests/plugin_contracts/`）：slime 通过环境变量 `SLIME_CONTRACT_*` 让**外部用户的自定义实现**走完全相同的一套断言（签名、返回结构、副作用），把"内置 hook"和"用户 hook"放进同一个 CPU-only 测试 harness 校验。这是把 hook-by-path 从"接口约定"变成"可机器化校验的契约"的关键。

## 1. 架构与模块边界

### 1.1 注册侧（arguments.py）

所有 18 个 hook 都以 `--xxx-path` 形式注册在 `slime/utils/arguments.py` 中：

- `arguments.py:307` `--rollout-function-path`（默认 `slime.rollout.sglang_rollout.generate_rollout`）
- `arguments.py:421` `--dynamic-sampling-filter-path`
- `arguments.py:453` `--custom-generate-function-path`
- `arguments.py:462` `--custom-rollout-log-function-path`
- `arguments.py:472` `--custom-eval-rollout-log-function-path`
- `arguments.py:483` `--buffer-filter-path`
- `arguments.py:515` `--rollout-data-postprocess-path`
- `arguments.py:605` `--data-source-path`（默认 `slime.rollout.data_source.RolloutDataSourceWithBuffer`）
- `arguments.py:747` `--eval-function-path`（若未设置，`arguments.py:1938-1939` 默认 fallback 到 `rollout_function_path`）
- `arguments.py:892` `--custom-loss-function-path`（搭配 `--loss-type custom_loss`）
- `arguments.py:935` `--custom-advantage-function-path`
- `arguments.py:1046` `--custom-tis-function-path`
- `arguments.py:1052` `--custom-pg-loss-reducer-function-path`
- `arguments.py:1327` `--custom-rm-path`（+ `--group-rm` 开关切换批量模式）
- `arguments.py:1337` `--custom-reward-post-process-path`
- `arguments.py:1345` `--custom-convert-samples-to-train-data-path`
- `arguments.py:1395` `--rollout-sample-filter-path`
- `arguments.py:1407` `--rollout-all-samples-process-path`
- `arguments.py:1424-1434` 三个 `--custom-megatron-*-path`

外加两个**非 -path 但属于 hook**的开关：`arguments.py:1059` `--use-routing-replay` 与 `--use-rollout-routing-replay`，它们控制 MoE 专家路由的录-放机制（见 §6.3）。

还有一对辅助 -path 在边缘场景上挂钩：`--custom-delta-pre-push-path`（`arguments.py:205`，weight sync 钩子）和 `--custom-model-provider-path`（`arguments.py:216`，Megatron 模型构造钩子）—— 这两条不在 customization.md 的 18 个清单里，但用的是同一套 hook-by-path 模式。

### 1.2 调用侧（按 rollout 阶段分 9 组）

把 18 个 hook 按 rollout step → reward → filter → training step → log 的时间轴归类：

| 组 | hook | 调用点 |
| --- | --- | --- |
| (1) rollout 编排 | `rollout_function_path` / `eval_function_path` | `slime/ray/rollout.py:441-442` |
| (2) 生成 | `custom_generate_function_path` | `slime/rollout/sglang_rollout.py:251-261`（per-sample dispatch） |
| (3) 数据源 | `data_source_path` | `slime/ray/rollout.py:438-439` |
| (4) 奖励 | `custom_rm_path` (+ `group_rm` 切批量) | `slime/rollout/rm_hub/__init__.py:55-110`（`async_rm` / `batched_async_rm`） |
| (5) 动态/缓冲过滤 | `dynamic_sampling_filter_path`, `buffer_filter_path`, `rollout_sample_filter_path`, `rollout_all_samples_process_path` | `slime/rollout/sglang_rollout.py:395-477` |
| (6) 训练数据后处理 | `rollout_data_postprocess_path`, `custom_reward_post_process_path`, `custom_convert_samples_to_train_data_path` | `slime/backends/megatron_utils/actor.py:180-182`、`slime/ray/rollout.py:687-688/717-718` |
| (7) 训练 loss | `custom_loss_function_path`, `custom_advantage_function_path`, `custom_tis_function_path`, `custom_pg_loss_reducer_function_path` | `slime/backends/megatron_utils/loss.py:711-712/1007-1008/1028-1029/1267` |
| (8) Megatron 训练 step | `custom_megatron_init_path`, `custom_megatron_before_log_prob_hook_path`, `custom_megatron_before_train_step_hook_path` | `slime/backends/megatron_utils/initialize.py:100-103`、`model.py:451-454/554-557` |
| (9) 日志 | `custom_rollout_log_function_path`, `custom_eval_rollout_log_function_path` | `slime/ray/rollout.py:1260/1292-1293` |
| (10) MoE routing replay | `--use-routing-replay` / `--use-rollout-routing-replay` | `slime/utils/routing_replay.py:57-93` |

为什么这么分组：上半部分（1-5）是 **rollout 进程** 内的串联，它们由 SGLang rollout actor 调用；下半部分（6-9）是 **trainer 进程** 内的串联，由 Megatron actor 与 Ray rollout manager 调用。同一个 `args` 对象通过 Ray 跨进程传递，让两边都能 `load_function(args.xxx_path)` 触发同一个 `importlib.import_module` 路径。

## 2. 关键抽象

不复述官方手册——只挑每组一个代表 hook，看签名背后的**契约形态**。

### 2.1 `load_function` 是唯一的加载器

`slime/utils/misc.py:37-45`：

```python
def load_function(path):
    module_path, _, attr = path.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, attr)
```

整个 customization 子系统只有这一个加载入口。这意味着：
- 路径里**最后一段是属性名**（函数或类），前面全是模块路径。
- 不接受 `module:attr` 这种 Setuptools 风格。
- 没有 lazy import / 缓存——每次 `load_function` 都会重新 `importlib.import_module`，但 Python 自身的 `sys.modules` 缓存让二次调用变成查表。

### 2.2 rm_hub：内置 + 用户共享同一接口的范本

`slime/rollout/rm_hub/__init__.py` 的 `async_rm` 是教科书级的"内置默认 + 用户覆盖"实现：

1. **最高优先级**：`sample.custom_rm_path`（per-sample，来自 eval dataset config，见 §5.2）。
2. **次优先级**：`args.custom_rm_path`（全局 CLI 注入）。
3. **fallback**：根据 `args.rm_type` 走内置的 `deepscaler / dapo / math / f1 / gpqa / ifbench / remote_rm / random` 七种实现，对应 `rm_hub/deepscaler.py` 等模块。

内置实现的形式是普通 Python 函数（不是类、不是子类），与用户自定义的 `custom_rm` 签名完全相同——这就是把"内置 reward"和"用户 reward"看成同一个接口的不同实例。`rm_hub/deepscaler.py:4` 的 `get_deepscaler_rule_based_reward(response, label)` 和 `rm_hub/gpqa.py:54` 的 `compute_gpqa_reward(response, label, metadata=None)` 都是 1-3 个原始入参，外面套一层 `async_rm` 做参数适配。

### 2.3 filter_hub：极简的契约 + 旧版兼容

`slime/rollout/filter_hub/__init__.py` 是**空的**——`filter_hub` 几乎只是命名空间。所有内容在两个文件里：

- `filter_hub/base_types.py`：`DynamicFilterOutput(keep: bool, reason: str | None)` + `call_dynamic_filter(fn, *args, **kwargs)` 辅助函数。**旧版用户**可能只返回 bool，`call_dynamic_filter` 在 `base_types.py:11-21` 显式包了一层 `if not isinstance(output, DynamicFilterOutput): output = DynamicFilterOutput(keep=output)`，做向后兼容。
- `filter_hub/dynamic_sampling_filters.py:9-15`：唯一的内置 filter `check_reward_nonzero_std`，5 行的 DAPO 实现（reward 全相同时丢弃）。

这个文件如此短小恰恰说明：filter 接口被设计成"用户高度自定义"的，slime 只提供一个 reference + 一个 `MetricGatherer` 收集过滤原因（`base_types.py:24-37`）。

### 2.4 契约测试通过环境变量动态加载

`tests/plugin_contracts/_shared.py` 的 `run_contract_test_for_file`（行 58-89）是整个套件的中枢：

1. 用 `ArgumentParser` 把每个 `--xxx-path` CLI 参数解析出来。
2. 把它存到环境变量 `SLIME_CONTRACT_XXX_PATH`。
3. `raise SystemExit(pytest.main([file, *remaining]))` 把控制权交给 pytest。

测试函数里调用 `get_contract_path("XXX_PATH", default_path)` 读环境变量；若用户没传，就用仓库内置的 reference 实现。**这让"测试内置 hook 是否符合契约"和"测试外部用户 hook 是否符合契约"是同一份测试代码**。

## 3. 数据流：一个 rollout step 里 18 个 hook 谁在何时被调用

```
[trainer 进程]
  RolloutController.__init__         → data_source_path, rollout_function_path,
  (slime/ray/rollout.py:438-450)        eval_function_path, custom_reward_post_process_path,
                                        custom_convert_samples_to_train_data_path 一次性 load

  ----------------------------- 每个 rollout step -----------------------------

[rollout 进程，由 generate_rollout 入口驱动]
  generate_rollout(args, rollout_id, data_source, evaluation)
    ├─ data_source.get_samples(...)          ← data_source_path
    ├─ for each group:
    │    generate_and_rm_group
    │      └─ generate_and_rm(sample)
    │           ├─ custom_generate_function_path（per-sample，可返回 list[Sample]）
    │           │     ← sglang_rollout.py:251-261
    │           └─ async_rm / batched_async_rm
    │                 ├─ sample.custom_rm_path     (eval-level)
    │                 ├─ args.custom_rm_path       (CLI-level)
    │                 └─ rm_type 内置 fallback
    ├─ call_dynamic_filter ← dynamic_sampling_filter_path  (per-group, 在 over-sampling 期间丢弃)
    ├─ rollout_sample_filter_path  (per-sample remove_sample 标记)
    └─ rollout_all_samples_process_path  (拿到含被丢弃样本的全集做统计)

[trainer 进程]
  RolloutController 收到 samples
    ├─ custom_rollout_log_function_path  /  custom_eval_rollout_log_function_path
    │     ← slime/ray/rollout.py:1260/1292
    ├─ custom_reward_post_process_func(samples) → raw_rewards, rewards
    │     ← slime/ray/rollout.py:687-688
    ├─ custom_convert_samples_to_train_data_func(samples) → train_data dict
    │     ← slime/ray/rollout.py:717-718
    └─ Megatron actor 拿到 train_data
         ├─ rollout_data_postprocess_path(args, rollout_id, rollout_data)
         │     ← actor.py:182（在 log_prob 之后，advantage 之前）
         ├─ custom_advantage_function_path(args, rollout_data)
         │     ← loss.py:711-712
         ├─ custom_megatron_before_train_step_hook_path
         │     ← model.py:557（每个 train step 前）
         ├─ custom_megatron_before_log_prob_hook_path
         │     ← model.py:454（每个 log_prob 计算前）
         ├─ loss 计算
         │    ├─ custom_tis_function_path           ← loss.py:1008
         │    ├─ custom_pg_loss_reducer_function_path ← loss.py:1029
         │    └─ custom_loss_function_path          ← loss.py:1267
         └─ MoE routing replay
              ├─ rollout 阶段：record/store top_indices
              └─ train 阶段：replay forward+backward（routing_replay.py:57-93）

  Megatron 初始化（仅一次）
    └─ custom_megatron_init_path  ← initialize.py:103
```

注意 4 个 filter hook 之间的微妙差异：
- `dynamic_sampling_filter_path` 决定一个 group 是否**进入训练 batch**（DAPO 风格）；
- `buffer_filter_path` 决定从 buffer 里**怎么 pop 样本**（默认 `pop_first` 见 `data_source.py:225`）；
- `rollout_sample_filter_path` 通过修改 `sample.remove_sample` 决定 sample **是否参与 loss**（但仍参与 advantage normalization）；
- `rollout_all_samples_process_path` 是**只读副作用**，可以观察被丢弃的样本做日志/统计。

## 4. 设计模式

### 4.1 为什么选 hook-by-path 而不是 plugin / 继承 / decorator

阅读源码后能看出这是一个**有意的工程权衡**：

- **plugin 系统**（如 setuptools entry points）需要用户把代码打成可发布的包，对 RL 用户来说太重。
- **继承基类**（如要求用户继承 `RewardModel` 抽象类）会产生稳定性义务——基类一改用户就崩。slime 选择不暴露任何 ABC，只暴露**函数签名约定 + 一个 `DataSource` "duck-typed" 类**（`data_source_path` 是唯一的类 hook，而且测试里也只检查 `__init__` 签名而非 isinstance，见 `test_plugin_path_loading_contracts.py:216`）。
- **decorator 注册表**（`@register_rm("my_rm")`）要求用户的模块被显式 import 才会注册，需要额外的 plugin discovery。
- **hook-by-path** 让用户写一个普通 Python 文件、在命令行里多加一行 `--custom-rm-path my_module.my_fn`，就完成接入。**用户的代码完全不需要 import slime 的任何东西就能成立**（虽然实际上会用到 `slime.utils.types.Sample`）。

代价是：**契约只在运行时（或测试时）被验证**。这才有了下一节的契约测试。

### 4.2 契约测试：把"约定"机器化

`tests/plugin_contracts/` 把 18 个 hook 按形态归并成 4 个测试文件：

- `test_plugin_rollout_contracts.py`：覆盖整个 rollout 函数（最重的一类），用 `inspect.signature` 断言 `("args", "rollout_id", "data_source", "evaluation")` 完全一致（行 126-135），并跑一遍 `RolloutFnTrainOutput` / `RolloutFnEvalOutput` 的结构校验。
- `test_plugin_generate_contracts.py`：覆盖 `custom_generate_function_path`，验证返回 `Sample` 或 `list[Sample]` 都合法（`group_rm` 模式下 `list[Sample]` 的 sibling 场景见行 176-198）。
- `test_plugin_path_loading_contracts.py`：参数化测试 7 个偏"加载即可"的 hook（`eval_function_path` / `custom_rm_path` / `dynamic_sampling_filter_path` / `buffer_filter_path` / `data_source_path` / `rollout_sample_filter_path` / `rollout_all_samples_process_path`），用 `SyncCase` dataclass 统一描述每个 hook 的环境变量名 + 默认路径 + 默认行为断言 + 自定义路径断言。
- `test_plugin_runtime_hook_contracts.py`：覆盖 5 个 runtime hook，**还顺便检查调用点 source code 是否包含某个字符串标记**（`runtime_marker`），这相当于做**"调用点契约"**：如果有人把 `self.custom_reward_post_process_func(self.args, samples)` 这一行从 `slime/ray/rollout.py` 里改掉，测试就会失败（见行 154、180-183）。这是一个非常聪明、低成本的"防止内部 refactor 破坏外部约定"的设计。

### 4.3 "core 4" 覆盖 80% 场景

`docs/zh/get_started/customization.md:32-47` 给出了一个"想做的事 → 应使用的接口"映射表，明确推荐：

- agentic / RAG / tool use → `--custom-generate-function-path`
- verifier / rule-based / 外部 reward → `--custom-rm-path`
- 替换整套 rollout（仅在 per-sample 不够用时）→ `--rollout-function-path`
- 自定义任务采样 / buffer → `--data-source-path`

这相当于官方给出了"扩展层级建议"：从最局部（`custom_generate`）到最全局（`rollout_function`）。**slime 不希望用户一上来就重写整个 rollout 函数**，因为那意味着失去内置的 partial rollout、abort、dynamic filter 编排。

## 5. 集成点

### 5.1 与 arguments.py 的注册关系

所有 hook 参数都是在 `arguments.py:1417` 的 `add_custom_megatron_plugins_arguments` 和其他 `add_*_arguments` 子函数里注册的。注册时只声明 `type=str, default=None`，没有额外校验——**校验完全推迟到运行时**（要么是 `load_function` 抛 ImportError，要么是契约测试发现签名不符）。

`arguments.py:1938-1939` 是一个微妙的默认值传染：

```python
if args.eval_function_path is None:
    args.eval_function_path = args.rollout_function_path
```

这意味着用户只要覆盖 `--rollout-function-path`，eval 路径会自动跟上——除非显式提供 `--eval-function-path`。

### 5.2 与 eval dataset config 的关系（per-sample hook）

`slime/utils/eval_config.py` 是 customization 系统里**唯一支持 per-sample hook**的地方：

- `EvalDatasetConfig.custom_rm_path`（`eval_config.py:126`）
- `EvalDatasetConfig.custom_generate_function_path`（`eval_config.py:151`）

调用点在 `sglang_rollout.py:580-581` 和 `rm_hub/__init__.py:57-59`：sample 上挂着 `custom_rm_path` / `generate_function_path` 属性，这两个属性会**覆盖全局 `args.*_path`**。这就让 eval 时不同 dataset 用不同 reward / generate function 成为可能——非常适合"主任务用 GRPO，eval 时跑 IFBench / GPQA / math 三个 dataset 各自的 verifier"这种场景。

### 5.3 与 agent harness 的关系

agent harness（`slime/agent/`）作为**独立子系统**存在，但它最终是被 `--custom-generate-function-path` 调用接入 slime 的。这就是为什么 `examples/multi_agent/rollout_with_multi_agents.py` 的最外层函数 `generate_with_multi_agents(args, sample, sampling_params, evaluation=False)` 签名严格符合 `custom_generate` 契约——agent harness 不是 slime 的第一公民，它只是"实现了 custom_generate 接口的一个用户工程"。

### 5.4 与 routing replay 的关系

MoE routing replay 是个奇特的 hook：它不通过 `--xxx-path` 注入，而是通过 `--use-routing-replay` 开关 + monkey-patch `compute_topk`（`routing_replay.py:57-82`）。原因是它需要在 **MoE expert routing 这种深度内嵌的 forward 路径**上插桩——用户代码无论如何写不出来，必须 slime 自己提供。**它属于"内置 hook 但通过 env var 与开关串联"的特殊形态**：`ENABLE_ROUTING_REPLAY` + `ROUTING_REPLAY_STAGE`（record / replay_forward / replay_backward / fallthrough）四态切换。

## 6. 出人意料的决策

### 6.1 契约测试用环境变量而非 pytest fixture 接收用户路径

最常规的做法是写一个 conftest fixture 或 pytest 命令行参数。slime 选了**环境变量 `SLIME_CONTRACT_*`**（`_shared.py:12` 的 `ENV_PREFIX`），原因藏在 `_shared.py:88-89`：测试可以**直接当脚本跑** `python tests/plugin_contracts/test_plugin_rollout_contracts.py --rollout-function-path my.func`，由 `run_contract_test_for_file` 内部 parse 这些 `--xxx-path` 参数，写入环境变量，再 `pytest.main(...)` 重新启动 pytest，pytest 里的测试代码 `get_contract_path("ROLLOUT_FUNCTION_PATH")` 从 env 读取。这种做法的好处是**让契约测试既能在 pytest 里运行，也能在 `run-ci-changed` 这种"直接执行 .py 文件"的 CI 工具里运行**，并且让用户验证自己实现的方式和验证仓库内置实现的方式完全一致——`customization.md:489-494` 显式记录了这个用法。

### 6.2 `custom_generate` 可以返回 `list[Sample]`，sibling samples 用同一个 rollout_id

`customization.md:87-117` 专门讲了这个：subagent / multi-agent / context compact 场景下，一次 prompt 自然拆成多个训练片段，`custom_generate` 可以直接返回 list。**关键约束**是这些 sibling samples 必须设置**相同的 `rollout_id`**，slime 在训练切分和 loss 聚合时会把它们当作同一次 rollout，避免重复计数。这是一个对"agentic RL"的特殊妥协——比起强行让一次 rollout 产出一个样本，承认 agent trajectory 可能被自然拆分更符合实际。这条契约在 `test_plugin_generate_contracts.py:176-198` 有显式的测试 (`test_generate_and_rm_group_rm_accepts_list_result_from_custom_generate`)。

### 6.3 `--group-rm` 让 reward 模型成为"组级"批处理

`rm_hub/__init__.py:99-110` 的 `batched_async_rm` 在 `args.custom_rm_path` 存在且 `args.group_rm=True` 时，把整个 group 作为 `list[Sample]` 一次性传给用户的 reward 函数。这覆盖了一类重要场景：**奖励本身需要 group 内对比**（例如"哪个回答比其他更好"的 pairwise 排序、或者群体一致性指标）。`generate_and_rm_group`（`sglang_rollout.py:294-333`）会延迟到 group 集齐后再统一调用 `batched_async_rm`。这是 18 个 hook 里唯一一对"signatures 不同但参数 path 共用"的 hook，靠 `--group-rm` 开关切换形态。

### 6.4 调用点字符串作为"反 refactor 守卫"

`test_plugin_runtime_hook_contracts.py:131-177` 的 `HOOK_CASES` 里每个 case 都带 `runtime_marker`（如 `"self.custom_reward_post_process_func(self.args, samples)"`），测试会去 `slime/ray/rollout.py` 等源文件里 `read_text()` 然后 `assert runtime_marker in source`。这相当于**用 grep 替代了完整的调用图分析**——非常便宜，但能在重构改动了调用点形式时立刻失败。这是工程务实的体现：与其写一个 AST 分析器，不如假设调用点写法稳定，让 string match 来守护。

### 6.5 MoE routing replay 是个"非 -path 但概念上是 hook"的特殊成员

之所以把它列进 customization 的 18 个 hook，是因为它**让训练时重放 rollout 时的专家路由决策**成为可能（R3 论文，arXiv:2510.11370）。技术上它不是用户代码注入点，但概念上它是 slime 给"训练-推理 MoE routing 一致性"留的扩展位。`routing_replay.py:14` 的 `all_routing_replays = []` 是一个类级别的全局收集，每个 MoE module register 时往里塞一个实例（行 86-92），训练时四种 stage（fallthrough / record / replay_forward / replay_backward）通过环境变量切换——又是一例"用 env var 而不是 args"的设计，让它对 Megatron forward 路径透明。

### 6.6 hook 加载没有 try/except，故意失败要快

`load_function` 不做错误兜底；`importlib.import_module` 失败、`getattr` 找不到属性都会直接抛异常。这与 slime 整体"配置错误即崩"的哲学一致：宁可在初始化时崩，也不要在跑了 8 小时之后某个 rollout step 才发现 hook 路径写错了。

## 附录

### A. 已通读的关键文件

- `/Users/quemaofei/Documents/Projects/slime/slime/utils/arguments.py`（所有 -path 注册，行 200-1502）
- `/Users/quemaofei/Documents/Projects/slime/slime/utils/misc.py:37-45`（`load_function`）
- `/Users/quemaofei/Documents/Projects/slime/slime/utils/eval_config.py`（per-sample hook 通过 dataset config）
- `/Users/quemaofei/Documents/Projects/slime/slime/utils/routing_replay.py`（MoE routing 重放）
- `/Users/quemaofei/Documents/Projects/slime/slime/rollout/rm_hub/__init__.py`（reward 加载与 dispatch）
- `/Users/quemaofei/Documents/Projects/slime/slime/rollout/rm_hub/{deepscaler,f1,gpqa}.py`（内置 reward 实现）
- `/Users/quemaofei/Documents/Projects/slime/slime/rollout/filter_hub/{__init__,base_types,dynamic_sampling_filters}.py`
- `/Users/quemaofei/Documents/Projects/slime/slime/rollout/sglang_rollout.py:241-479`（generate_and_rm + filter 编排）
- `/Users/quemaofei/Documents/Projects/slime/slime/ray/rollout.py:430-477, 687-718, 1260-1293`（hook 调用点）
- `/Users/quemaofei/Documents/Projects/slime/slime/backends/megatron_utils/{initialize,model,loss,actor}.py`（trainer 侧 hook 调用点）
- `/Users/quemaofei/Documents/Projects/slime/tests/plugin_contracts/`（4 个测试文件 + `_shared.py`）
- `/Users/quemaofei/Documents/Projects/slime/docs/zh/get_started/customization.md`（496 行官方参考）

### B. 已扫读的示例文件

- `examples/search-r1/generate_with_search.py`（`custom-generate` 示例）
- `examples/multi_agent/rollout_with_multi_agents.py`（`rollout-function` 示例）
- `examples/train_infer_mismatch_helper/mis.py`（`custom-tis-function` 示例，未深入）

### C. 未读但与本子系统有关的边界文件

- `slime/rollout/data_source.py`（属于 Data Buffer 子系统，仅核对了 `pop_first` 的位置 `:225`）
- `slime/agent/`（属于 agent harness 子系统，本笔记只标注它通过 `--custom-generate-function-path` 接入）
- `slime/backends/megatron_utils/loss.py` 完整 1300 行（仅看了 hook 调用点，未通读 loss 算法）

### D. 18 个 hook 默认值清单（便于书里做对照）

| hook | 默认值 |
| --- | --- |
| `--rollout-function-path` | `slime.rollout.sglang_rollout.generate_rollout` |
| `--eval-function-path` | `None`（fallback 到 rollout_function_path） |
| `--custom-generate-function-path` | `None` |
| `--data-source-path` | `slime.rollout.data_source.RolloutDataSourceWithBuffer` |
| `--custom-rm-path` | `None`（fallback 到 `--rm-type` 内置七种） |
| `--dynamic-sampling-filter-path` | `None`（示例：`slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std`） |
| `--buffer-filter-path` | `None`（默认 `slime.rollout.data_source.pop_first`） |
| `--rollout-sample-filter-path` | `None` |
| `--rollout-all-samples-process-path` | `None` |
| `--rollout-data-postprocess-path` | `None` |
| `--custom-reward-post-process-path` | `None`（默认 GRPO 归一化） |
| `--custom-convert-samples-to-train-data-path` | `None` |
| `--custom-rollout-log-function-path` | `None` |
| `--custom-eval-rollout-log-function-path` | `None` |
| `--custom-loss-function-path` | `None`（需 `--loss-type custom_loss`） |
| `--custom-advantage-function-path` | `None` |
| `--custom-tis-function-path` | `None` |
| `--custom-pg-loss-reducer-function-path` | `None` |
| `--custom-megatron-init-path` | `None` |
| `--custom-megatron-before-log-prob-hook-path` | `None` |
| `--custom-megatron-before-train-step-hook-path` | `None` |

> 注：官方文档列了 18 个槽位，但若严格数 `--*-path` 风格的 hook，加上 `--custom-megatron-*-path` 三个，共 21 个 path 注入点；customization.md 把 3 个 Megatron hook 合并叙述为"第 17 项"，把 routing replay 算作"第 18 项"（虽然它不是 -path 形态）。
