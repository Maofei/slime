# 研究笔记：CLI / 参数 / 启动入口子系统

## TL;DR

slime 的"配置面"是**整套架构里压力最大的一层**：要把 Megatron（数百个 native 参数）、SGLang（一整个 `ServerArgs` + `RouterArgs`）、以及 slime 自己的 ~150 个旋钮装进同一条 `python train.py` 命令行，并且要在解析时就把绝大多数语义错误拦下来。它的核心赌注是**透传胜过封装**——slime 不为 Megatron / SGLang 重新设计配置 schema，而是把它们的 argparse 原样借过来，再用三个动作把它们粘合：(1) 用 `reset_arg` 在 Megatron parser 上原地改默认值（不重新声明参数）；(2) 用一个"重写 `parser.add_argument`"的 wrapper，把 SGLang 所有参数都加 `--sglang-` 前缀注入到独立的 parser；(3) 在 `slime_validate_args` 里一次性做所有跨参数约束、推断与隐式开关计算（这一个函数 260 行）。`train.py` / `train_async.py` 各 80-100 行，只是这套配置的下游消费者——一旦 `args` 命名空间合法，主循环就是机械的。SFT vs RL 的差异**不在入口脚本层面分**，而是通过几个 hook 参数（`--rollout-function-path`、`--loss-type`、`--disable-compute-advantages-and-returns`、`--debug-train-only`）在同一个 `train.py` / `train_async.py` 里复用整套基础设施。

---

## 1. 架构与模块边界

参数面有**三层**，每一层各管一个解析阶段：

| 层 | 注册点 | 命名空间 | 解析时机 |
| --- | --- | --- | --- |
| **SGLang**（含 `sgl-router`）| `slime/backends/sglang_utils/arguments.py:35-138` | 全部以 `sglang_` 前缀写入 args | Phase 1：独立 parser，`parse_known_args` |
| **Megatron** + **slime 自身** | `slime/utils/arguments.py:36-1506` 把 slime args 作为 `extra_args_provider` 注入 Megatron 的 parser；Megatron 默认值用 `reset_arg` 改写 | 共用同一个 args 命名空间（slime args 直接命名，不加前缀） | Phase 2：`megatron_parse_args(...)` 套在 `_megatron_parse_args(ignore_unknown_args=True)` 上 |
| **pre-parse 模式开关** | `_pre_parse_mode`（`arguments.py:1509-1522`）| `train_backend / debug_rollout_only / debug_train_only / load_debug_rollout_data` | Phase 0：先于 SGLang 和 Megatron，因为它决定要不要跳过 SGLang 解析 |

三层最后合并到同一个 `argparse.Namespace`，由 `slime_validate_args`（`arguments.py:1748-2008`）做跨层一致性校验。`parse_args`（`arguments.py:1525-1568`）是唯一的入口，做的事情顺序固定：

```
configure_logger()
pre = _pre_parse_mode()             # Phase 0
skip_sglang = pre.debug_train_only or pre.load_debug_rollout_data is not None
sglang_ns = sglang_parse_args() if not skip_sglang else None    # Phase 1
args = megatron_parse_args(extra_args_provider=add_slime_arguments, ...)  # Phase 2
merge pre into args
merge sglang_ns into args
slime_validate_args(args)
if backend == megatron: megatron_validate_args(args)            # backends/megatron_utils
if not debug_train_only: sglang_validate_args(args)             # backends/sglang_utils
```

**`train.py` vs `train_async.py`**：两者长度几乎相同（103 vs 80），main 都是 `args = parse_args(); train(args)`，差异在 `train(args)` 内部：
- `train.py` 是 **sync rollout-then-train**：`for rollout_id: generate → train → update_weights → eval`。
- `train_async.py` 是 **rollout-训练 1 步流水线**：`generate(0)` 先发，然后 `for rollout_id: sync(rollout_id) → kick off generate(rollout_id+1) → train → 周期性 update_weights`。`train_async.py:11` 第一行就 `assert not args.colocate, "Colocation is not supported for async training."` —— 异步流水线要求 rollout 和 train 在不同 GPU 上，强一致性约束放在入口而不是 validate。
- 没有 `train_sft.py` / `train_pretrain.py` / `train_rl.py`，所有任务形态共享 `train.py` 或 `train_async.py`。SFT 是 `--rollout-function-path slime.rollout.sft_rollout.generate_rollout` + `--loss-type sft_loss` + `--disable-compute-advantages-and-returns` + `--debug-train-only` 的组合（见 `scripts/run-qwen3-4B-base-sft.sh:37-51`），完全在参数层完成"分支"。

---

## 2. 关键抽象

| 抽象 | 路径 | 说明 |
| --- | --- | --- |
| `parse_args` | `slime/utils/arguments.py:1525-1568` | 唯一对外入口；三阶段解析 + 三层 validate 的顺序在这里固定 |
| `get_slime_extra_args_provider` | `arguments.py:36-1506` | 返回一个 `add_slime_arguments(parser)` 闭包；Megatron 用 `extra_args_provider=` 调用它，从而把 slime 的 16 个分组全部注入 Megatron 的 parser |
| `reset_arg` | `arguments.py:20-33` | 没有公开 API 时"原地改 Megatron 默认值"的小工具：遍历 `parser._actions`，匹配 `option_strings` 后改 `default`；没找到才 `add_argument`。整个文件里出现 ~14 次，专门用来把 `--lr`、`--seed`、`--clip-grad`、`--eval-interval`、`--mtp-num-layers`、`--micro-batch-size`、`--global-batch-size` 等改成 slime 期望的 RL 默认 |
| `add_sglang_arguments` + wrapper | `slime/backends/sglang_utils/arguments.py:35-138` | 用 monkey-patch `parser.add_argument` 的方式，把 `ServerArgs.add_cli_args(parser)` 注入的所有 SGLang 参数自动加 `--sglang-` 前缀；同时维护 `skipped_args` 列表（如 `tp_size / port / nnodes / nccl_port` 等由 slime 自己算出来的）。这是"透传" 哲学最浓的一段代码 |
| `slime_validate_args` | `arguments.py:1748-2008` | 260 行的跨参数约束 + 隐式推断 + 模式开关求值；本子系统的"业务规则中心"。下面会单独抽出来讲 |
| `megatron_validate_args` | `slime/backends/megatron_utils/arguments.py:72-90` + `_hf_validate_args:93-144` | 调上游 `_megatron_validate_args`，再做 slime 特有的：强制 `variable_seq_lengths=True`、`allgather` dispatcher 自动换 `alltoall`、与 HF config 比对 hidden_size/num_heads/num_layers/intermediate_size/rope_theta 等 8 个字段 |
| `sglang_validate_args` | `slime/backends/sglang_utils/arguments.py:141-173` | 计算 `sglang_tp_size = rollout_num_gpus_per_engine // pp_size`、检查 `sglang-config / prefill_num_servers / rollout_external` 三个互斥的 deployment 模式 |
| `parse_megatron_role_args` / `_apply_megatron_role_overrides` | `arguments.py:1571-1646` | 支持 `--megatron-config-path` YAML：用 `deepcopy(base_args)` 后按 role（`actor` / `critic`）做覆盖。critic role 强制 `kl_coef=0`、`use_opd=False`、`custom_advantage_function_path=None`、`untie_embeddings_and_output_weights=True` —— 这条线让 actor / critic 可以共享一份 base CLI，再用 YAML 表达"critic 跟 actor 不一样的那几行" |
| `_resolve_eval_datasets` | `arguments.py:1654-1694` | 把 `--eval-config <yaml>` 或 `--eval-prompt-data <name> <path>` 两种输入归一成 `args.eval_datasets: list[EvalDatasetConfig]`；同时把 legacy 形态（单值默认成 `aime`）显式 warning 后接受 |
| `_pre_parse_mode` | `arguments.py:1509-1522` | "提前判路"：用一个无 help、`allow_abbrev=False` 的 mini parser 把决定**解析流程**的几个 flag 先抓出来。返回的 namespace 用 `setattr` 合并回主 args，避免在 `add_slime_arguments` 里重复注册 |
| `_validate_update_weight_args` + `_resolve_update_weight_disk_dir` | `arguments.py:1697-1745` | weight sync 模式的"提前 unify"：把 deprecated alias `--update-weight-delta-dir` 折叠成 `--update-weight-disk-dir`、检查 `disk` transport 必须配 dir、`delta` 模式不能配 `--colocate` |

---

## 3. 数据流：从 shell 脚本到 `for rollout_id in range(...)`

一次 `bash scripts/run-qwen3-4B.sh` 的完整路径（顺着 `train.py` 一直走到第一个 `actor_model.async_train`）：

```
scripts/run-qwen3-4B.sh
  │  ├─ source scripts/models/qwen3-4B.sh  # MODEL_ARGS=(...)
  │  ├─ 组装 7 个数组：CKPT_ARGS / ROLLOUT_ARGS / EVAL_ARGS / PERF_ARGS / GRPO_ARGS / OPTIMIZER_ARGS / SGLANG_ARGS / MISC_ARGS
  │  ├─ ray start --head --num-gpus ${NUM_GPUS} ...
  │  └─ ray job submit -- python3 train.py --actor-num-nodes ... ${MODEL_ARGS[@]} ${CKPT_ARGS[@]} ...
        │
        ▼
python3 train.py
  │
  ▼  args = parse_args()       # slime/utils/arguments.py:1525
  │  ├─ Phase 0: _pre_parse_mode()    # 抓 train_backend / debug_* / load_debug_rollout_data
  │  ├─ Phase 1: sglang_parse_args()  # 独立 parser，吃所有 --sglang-* 参数（除了被 skipped_args 屏蔽的）
  │  ├─ Phase 2: megatron_parse_args(extra_args_provider=add_slime_arguments)
  │  │     ↳ Megatron 的 parser 注册所有 Megatron 原生参数
  │  │     ↳ slime 的 16 个 add_*_arguments 函数注入到同一个 parser
  │  │     ↳ ignore_unknown_args=True 让 --sglang-* 不报错
  │  │     ↳ 解析完调 _hf_validate_args(args, AutoConfig.from_pretrained(hf_checkpoint))
  │  │       —— 这里就开始读 HF 模型 config 文件了
  │  │     ↳ _set_default_megatron_args(args)：use_distributed_optimizer=True / bf16=not fp16 / ...
  │  ├─ merge pre 和 sglang_ns 进 args
  │  ├─ slime_validate_args(args)     # 260 行业务规则，详见下面
  │  ├─ megatron_validate_args(args)  # 调上游 + 强制 variable_seq_lengths
  │  └─ sglang_validate_args(args)    # sglang_tp_size 推断 + 部署模式互斥检查
        │
        ▼
train(args)                    # train.py:9
  │  ├─ configure_logger()
  │  ├─ pgs = create_placement_groups(args)         # slime/ray/placement_group.py
  │  │     ↳ 这里第一次把 args 翻译成 Ray placement bundle
  │  │     ↳ 读 args.actor_num_nodes / actor_num_gpus_per_node / rollout_num_gpus / colocate / debug_*
  │  ├─ init_tracking(args)
  │  ├─ rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])
  │  ├─ actor_model, critic_model = create_training_models(args, pgs, rollout_manager)
  │  │     ↳ 内部会调 parse_megatron_role_args(args, args.megatron_config_path, role)
  │  │       —— 第二次"再解析"，但这次是 YAML 覆盖
  │  ├─ actor_model.update_weights()
  │  └─ for rollout_id in range(args.start_rollout_id, args.num_rollout): ...
```

要点：
- `parse_args` 返回后，`args` 不再有"未消费的 token"——SGLang 在 Phase 1 用 `parse_known_args` 吃自己的，Megatron 在 Phase 2 用 `ignore_unknown_args=True` 吃剩下的。这种"两个 parser 互相不干扰"的设计是为了让上游 SGLang 加新参数时**不需要 slime 同步**——`ServerArgs.add_cli_args(parser)` 总是把 SGLang 当前所有参数都注册一遍，wrapper 自动加前缀。
- `start_rollout_id` 在 `slime_validate_args:1804/1818` 里被悄悄改写：HF checkpoint 启动时强制为 0；从 Megatron `latest_checkpointed_iteration.txt` 续训时由后续 checkpoint 加载逻辑回填（不在这里设）。
- shell 脚本的"数组分组"约定（`CKPT_ARGS / ROLLOUT_ARGS / EVAL_ARGS / PERF_ARGS / GRPO_ARGS / OPTIMIZER_ARGS / SGLANG_ARGS / MISC_ARGS`）**和 `arguments.py` 的 `add_*_arguments` 分组不是一一对应**，但精神上接近——shell 脚本通过命名约定让 reviewer 一眼能知道哪一组参数是改 RL 算法、哪一组是改并行、哪一组是改 SGLang。这是社区脚本的事实标准，不是代码强制。

---

## 4. 设计模式

- **三层 parser、单一 namespace**：SGLang 走独立 parser + 前缀注入，Megatron 走自带 parser + slime 用 `extra_args_provider` 寄生，最后用 `setattr` 把三个 namespace 合并。**关键决定是 slime 不为自己的参数加前缀**，所以 `args.use_kl_loss` 和 Megatron 的 `args.lr` 都是 top-level 属性，写下游代码时不需要记 `args.slime_xxx`。代价：slime 参数名要保证全局不撞 Megatron——这一点靠 review 和测试守。
- **`reset_arg` 而不是 `add_argument(... default=...)`**：直接 `add_argument` 同名参数会让 argparse 抛 "conflicting option string"。`reset_arg` 走 `parser._actions` 改 default，是 argparse 唯一不需要 fork 父 parser 就能改默认值的方式。这是 slime "宁愿 monkey-patch 也不 fork Megatron" 哲学的另一处证据（同前一个笔记里讲的 `megatron_chunked_grad_coalesce_patch`）。
- **`add_*_arguments` 嵌套局部函数**：`get_slime_extra_args_provider` 内部把每个分组写成一个嵌套 `def add_xxx_arguments(parser)`，最后在 `add_slime_arguments` 末尾按固定顺序依次调用（`arguments.py:1478-1494`）。这套写法让 2010 行的文件**按分组浏览**仍然可读——读者跳到 `def add_rollout_arguments` 就知道这一段全是 rollout 相关；如果硬要平铺成一个 `add_argument` 大列表，2000 行真的没法读了。16 个分组：cluster / train / rollout / fault_tolerance / data / eval / algo / on_policy_distillation / wandb / tensorboard / debug / network / reward_model / rollout_buffer / mtp_training / ci / custom_megatron_plugins，每个分组对应书后续某一章的代码范围。
- **validation 集中而不是分散**：`slime_validate_args` 这一个函数处理了几乎所有"如果 X 设置了，那么 Y 必须如此"的约束。文件里没有 `@argparse.action`、没有 `type=callable_validator`、没有装饰器——validation 在解析完之后**统一一处做**，方便阅读、方便测试（`tests/test_megatron_argument_validation.py` 直接调 `slime_validate_args` 和 `_resolve_update_weight_disk_dir` 等子函数）。这种"解析阶段不做 validation，全部推后"的取舍在 RL 系统里特别重要：因为很多约束跨多个分组（如 `--use-opd` 和 `--opd-type` 和 `--opd-teacher-load` 三者），分散到 argparse action 里就把规则砍碎了。
- **shell 脚本的 boilerplate 模式**：每个 `scripts/run-*.sh` 都按固定模板写：(1) `pkill -9` 清场；(2) 探测 NVLink / GPU count；(3) `source scripts/models/<model>.sh` 拉 `MODEL_ARGS=(...)`；(4) 局部定义 7-8 个数组；(5) `ray start --head` 起头节点；(6) `ray job submit -- python3 train.py ${ALL_ARGS[@]}`。这套模板是 slime 实际的"用户接口"——没有 launcher 程序、没有 YAML、没有 hydra；shell 数组就是配置文件。读者在书里看到这套模板后，可以把自己的 recipe 当作"新加一个 `scripts/run-mymodel.sh`"来思考。
- **模型 specific override 用 `source scripts/models/*.sh`**：`MODEL_ARGS=(...)` 这一个数组完全描述了"模型架构"——`--num-layers`、`--hidden-size`、`--num-attention-heads`、`--rotary-base` 等。`scripts/models/qwen3-4B.sh` 只有 17 行（见 §6.4）。这种"模型描述就是一段 bash 数组" 的极简做法是因为模型架构必须和 HF config 一致（否则 `_hf_validate_args` 会拒），所以这些参数没有"调优空间"——它们是事实而不是选择。

---

## 5. 集成点

- **与 backends 的接口**（详见 backends 子系统笔记，本节只画连线）：
  - SGLang：`slime/utils/arguments.py:11-13` 直接 import `sglang_parse_args / validate_args / apply_external_engine_info_to_args`。Phase 1 调用 `sglang_parse_args()`，Phase 4 调用 `sglang_validate_args(args)`。
  - Megatron：`arguments.py:1543-1549` 延迟 import `megatron_parse_args / validate_args`，避免 `parse_args` 调用前就触发 Megatron 全家桶 import（关键——shell 脚本里 `pkill -9 python` 后立即重启时，Megatron import 慢，懒加载能让 SGLang 阶段先动）。
- **与 customization 的接口**（详见 customization 子系统笔记）：参数面定义了 18 个 `*-path` 参数，是 customization 的唯一注入点。本子系统不讨论这些 hook 的语义，只记录它们的注册位置：
  - rollout 阶段（add_rollout_arguments）：`--rollout-function-path`、`--dynamic-sampling-filter-path`、`--custom-generate-function-path`、`--custom-rollout-log-function-path`、`--custom-eval-rollout-log-function-path`、`--buffer-filter-path`、`--rollout-data-postprocess-path`
  - data（add_data_arguments）：`--data-source-path`
  - eval（add_eval_arguments）：`--eval-function-path`
  - algo（add_algo_arguments）：`--custom-loss-function-path`、`--custom-advantage-function-path`、`--custom-tis-function-path`、`--custom-pg-loss-reducer-function-path`
  - reward（add_reward_model_arguments）：`--custom-rm-path`、`--custom-reward-post-process-path`、`--custom-convert-samples-to-train-data-path`
  - train（add_train_arguments）：`--custom-delta-pre-push-path`、`--custom-model-provider-path`
  - rollout_buffer（add_rollout_buffer_arguments）：`--rollout-sample-filter-path`、`--rollout-all-samples-process-path`
  - custom_megatron_plugins：`--custom-megatron-init-path`、`--custom-megatron-before-log-prob-hook-path`、`--custom-megatron-before-train-step-hook-path`
  - 还有一个 `--custom-config-path`（`arguments.py:1495-1501`）——指向一个 YAML，`slime_validate_args:1983-1989` 在最后一步把里面的 key/value 全部 `setattr` 到 args 上，给 hook 函数读其私有配置。
- **与 Ray 编排的接口**：参数面把所有 GPU 拓扑参数（`actor_num_nodes`、`actor_num_gpus_per_node`、`rollout_num_gpus`、`rollout_num_gpus_per_engine`、`num_gpus_per_node`、`colocate`）放在 `add_cluster_arguments`。`slime_validate_args` 在最后做几条**关键的隐式推断**——`colocate=True` 时若 `rollout_num_gpus is None` 则补 `actor_num_gpus_per_node * actor_num_nodes`；`offload=True` 时自动设 `offload_train=True; offload_rollout=True` 然后 `del args.offload`（namespace 里只留细粒度的开关）；`use_critic = (advantage_estimator == "ppo")` 自动派生，critic 的 GPU count 强制等于 actor 的。这些推断把"用户写最少的开关、系统理解最多"做到极致，但代价是新 reader 必须读懂这 260 行才能预测 `args` 的最终形态——这也是 `tests/test_megatron_argument_validation.py:312-351` 那几个 `test_slime_validate_args_*` 存在的原因。
- **`args` 直接作为 Ray actor 的构造参数**：`actor.py / rollout.py` 等所有 Ray actor 都把 `args` 整个传过去（如 `MegatronTrainRayActor.init(args, ...)`）。这意味着 `args` 必须是 picklable 的纯数据——`argparse.Namespace` 满足这点；任何在 validation 阶段加进去的东西也都必须是 picklable。`args.eval_datasets`（一个 `list[EvalDatasetConfig]`）是这条约束的边界——`EvalDatasetConfig` 是个 dataclass，能 pickle。

---

## 6. 出人意料的决策（最重要部分）

### 6.1 2010 行 argparse 不拆，是因为"配置即文档"

slime 没有把 `arguments.py` 拆成 `arguments/cluster.py`、`arguments/rollout.py`、`arguments/algo.py` ——明明每一组的 `add_*_arguments` 是相互独立的局部函数，技术上拆开毫无障碍。**但作者反其道而行之**：所有分组都嵌套在 `get_slime_extra_args_provider` 的内层函数里，构成一个长达 1500 行的闭包。

读了几遍后才理解动机：这个文件就是 slime 的**唯一参数手册**。`docs/zh/` 下没有 "arguments reference" 那种把每个 flag 列一遍的文档（除了 `customization.md` 把 18 个 `*-path` hook 单独整理过），用户要查"`--rollout-max-context-len` 默认值是什么"或者"`--update-weight-encoding` 都支持哪些枚举"必须直接看代码。如果文件被拆成 16 个文件，用户就要先猜参数属于哪一组、再去对应文件搜。一个 2000 行的单文件 + 一个 `Cmd+F` 反而比拆开更高效。`help=` 字段里的文档密度证明了这一点——大量 `help=` 长达 6-10 行（如 `--update-weight-transport`、`--only-train-params-name-list`、`--rollout-batch-size`），实质上把参数说明写在源码里。

### 6.2 SGLang 透传用 monkey-patch `parser.add_argument`，不维护 allowlist

`add_sglang_arguments`（`sglang_utils/arguments.py:35-138`）的中间一段是这样的：

```python
old_add_argument = parser.add_argument
def new_add_argument_wrapper(*name_or_flags, **kwargs):
    # ... 检查 skipped_args，剩下的把 --foo-bar 改成 --sglang-foo-bar、dest 改成 sglang_foo_bar
    old_add_argument(*new_name_or_flags_list, **final_kwargs)

parser.add_argument = new_add_argument_wrapper
ServerArgs.add_cli_args(parser)   # 调用上游 SGLang 的注册函数
parser.add_argument = old_add_argument
```

也就是说，slime 不维护 "SGLang 有哪些参数"的列表——它直接调用上游 `ServerArgs.add_cli_args`，让 SGLang 自己注册所有当前支持的参数，wrapper 自动加 `--sglang-` 前缀。**上游 SGLang 新加任何参数，slime 第二天就支持**，零代码改动。`skipped_args`（`sglang_utils/arguments.py:45-63`）只列 14 个 slime 自己要算的参数（`tp_size / port / nccl_port / nnodes / random_seed / enable_memory_saver` 等）。

这是"native passthrough"赌注最纯粹的表达。代价是：(a) 用户必须用 `--sglang-` 前缀，不能直接抄 SGLang 文档里的 `--mem-fraction-static`；(b) `argparse` 的 `--help` 会把所有 SGLang 参数都列出来，让 `python train.py --help` 输出几百行（这也是 `add_sglang_router_arguments` 用 `use_router_prefix=True` 让 router 也走类似规则的原因）。

### 6.3 validation 集中且"先副作用、后断言"

`slime_validate_args` 里有大约一半内容**不是断言，而是改 `args`**。挑几条最反常的：

- `arguments.py:1881` `args.rollout_external = args.rollout_external_engine_addrs is not None` —— 不引入新参数，直接从已有参数派生一个 bool。
- `arguments.py:1886-1889` `args.use_critic = args.advantage_estimator == "ppo"` —— "PPO 必须用 critic" 这条事实直接写成派生属性。
- `arguments.py:1891-1894` `if args.offload: args.offload_train = True; args.offload_rollout = True; del args.offload` —— 把 user-facing 的粗粒度开关炸开后**删除原参数**，让下游代码只能看到细粒度的两个 bool。
- `arguments.py:1938-1939` `if args.eval_function_path is None: args.eval_function_path = args.rollout_function_path` —— 默认 eval 用 train 的 rollout function。
- `arguments.py:1949` `args.global_batch_size = rollout_batch_size * n_samples_per_prompt // num_steps_per_rollout` —— 派生计算 + 一致性断言并存。
- `arguments.py:1996` `args.rollout_max_prompt_len = args.rollout_max_context_len - 1` —— "至少留 1 个 token 出来产生 loss"的暗规则。
- `arguments.py:1870-1872` 把 `--dump-details` 自动展开成 `save_debug_rollout_data` 和 `save_debug_train_data` 两个路径模板。
- `arguments.py:1874-1879` `if args.load_debug_rollout_data is not None: args.debug_train_only = True` —— 入参互相隐含，**parse 阶段就强制 mode**。

这种"validation 同时是 normalization"的写法让下游代码极度简单：actor 看到 `args.use_critic` 就知道要不要建 critic；rollout 看到 `args.offload_train` 就知道是否要 offload；它们都不用关心 user 当初是写了 `--offload` 还是写了 `--offload-train`。代价是新 reader 看 `args.xxx` 不能光看 `add_argument` 那行的 default——必须把 `slime_validate_args` 通读一遍才知道运行时 `args.xxx` 的可能值。

### 6.4 SFT 不在 entry 层面分支，靠 4 个参数复用整套 RL 基础设施

`scripts/run-qwen3-4B-base-sft.sh:37-51` 的 `SFT_ARGS` 包含 4 个关键 flag：

```bash
--rollout-function-path slime.rollout.sft_rollout.generate_rollout
--loss-type sft_loss
--disable-compute-advantages-and-returns
--debug-train-only
```

这 4 行就完成了"RL → SFT"的全部切换。**没有 `train_sft.py`、没有 SFT 专用 placement、没有 SFT 专用 Megatron actor**。`--rollout-function-path` 换成 SFT 的 rollout 函数（实际上是个 dataset reader，没有 SGLang 调用）；`--debug-train-only` 让 SGLang 不被启动；`--loss-type sft_loss` 让 loss function 走 SFT 分支；`--disable-compute-advantages-and-returns` 让 actor 跳过 advantage 计算。整套 Ray 编排、placement、checkpoint、weight backup 全部复用。

这套设计赌的是：**SFT 是 RL 的一个退化形态**（采样函数变成 dataset 读取、loss 函数变成 cross-entropy、不需要 advantage），因此应该共享同一个执行框架。从用户视角，这意味着学会 RL 的启动脚本结构后，SFT 是"少写几个数组而已"，认知成本几乎为零。代价是 SFT 用户会被迫看到大量 RL 概念（`rollout_batch_size`、`n_samples_per_prompt=1`），这一点对纯 SFT 用户不友好——但 slime 不是 SFT 框架，是 RL 框架，这个取舍是合理的。

另外注意 SFT 脚本用的是 `train_async.py`（`run-qwen3-4B-base-sft.sh:117`），这进一步证明"训练入口和任务形态正交"：sync/async 由 `python3 train.py` vs `python3 train_async.py` 决定，task 由 hook 参数决定，两者自由组合。

### 6.5 `--megatron-config-path` YAML 用"按 role 覆盖" 而不是"按 role 重写整组配置"

actor / critic 都用 Megatron，但 critic 的并行度、学习率、batch size 经常和 actor 不同。slime 的解法（`arguments.py:1571-1646`）是：

- base CLI 是 actor 的；
- 如果给了 `--megatron-config-path`，则 YAML 里写一个 `megatron: [{role: critic, overrides: {lr: 5e-6, tensor_model_parallel_size: 4, ...}}]` 列表；
- `parse_megatron_role_args` 做 `deepcopy(actor_args)` 后**只覆盖 overrides 里的 key**。

`role == "critic"` 还会硬性写死几条："critic 不能 KL"（`kl_coef = 0`、`use_opd = False`、`custom_advantage_function_path = None`）、"critic 必须解耦 embedding / output layer"（`untie_embeddings_and_output_weights = True`）、"critic 必须开 CPU backup"（`disable_param_buffers_cpu_backup = False`）。这几条不是建议，是真的会被覆盖，**即使 YAML 里写了反向值也会被改回去**。

这是参数面少见的"硬性 policy"——大部分参数 slime 都让用户决定，但 critic 的这几个被作者认为没有合理的反向用法（如果 critic 也加 KL，KL 算的就是 value 函数和 reference value 函数的距离，无意义）。把这种 policy 写进 `_apply_megatron_role_overrides` 而不是放进文档里，是 slime 的一贯风格——能在代码里强制的事不靠人记。

---

## 附录

### 已通读文件
- `slime/utils/arguments.py`（2010）
- `slime/backends/megatron_utils/arguments.py`（199）
- `slime/backends/sglang_utils/arguments.py`（199）
- `train.py`（103）
- `train_async.py`（80）
- `tests/test_megatron_argument_validation.py`（385）
- `scripts/run-qwen3-4B.sh`（162）
- `scripts/run-glm4.7-30B-A3B.sh`（178）
- `scripts/run-qwen3-4B-base-sft.sh`（127）
- `scripts/run-qwen2.5-0.5B-reproducibility.sh`（138）
- `scripts/models/qwen3-4B.sh`（17）
- `examples/fully_async/README.md`、`examples/search-r1/README.md`、`examples/on_policy_distillation/README.md`（前 80 行）

### 扫读文件
- `slime/ray/placement_group.py`（grep `args.` 看参数消费点；`placement_group.py:101-117` 的 GPU count 计算和 `placement_group.py:154-200` 的 `parse_megatron_role_args` 调用）
- `slime/utils/eval_config.py`（263 行，看 `DATASET_RUNTIME_SPECS / DATASET_SAMPLE_SPECS` 的 fallback chain 设计）
- `slime/backends/sglang_utils/external.py`（`apply_external_engine_info_to_args` 在 `slime_validate_args:1884` 被调用，但具体实现是 backends 子系统的事）

### 边界外（不要在本章写）
- SGLang argparse 拦截的具体语义（`add_sglang_arguments` 的 wrapper 怎么处理 `dest=` 已经显式给出、`-f` 这种短 flag 怎么处理）→ 归 backends 子系统
- `megatron_validate_args` 里 `_hf_validate_args` 比对的 8 个字段背后为什么挑这些 → 归 backends 子系统的 "Megatron 配置一致性"小节
- 18 个 `*-path` hook 的签名、用什么形式被加载、契约测试如何守 → 归 customization 子系统
- `--megatron-config-path` YAML 的 role 覆盖被下游的 `MegatronTrainRayActor` 怎么消费 → 归 backends / training 子系统
- shell 脚本里那些 `RUNTIME_ENV_JSON` 的 `env_vars`（`CUDA_DEVICE_MAX_CONNECTIONS`、`NCCL_NVLS_ENABLE` 等）背后的取舍 → 归"部署 / 运行时环境"子系统

### 未读 / 推迟
- 其余 30+ 个 `scripts/run-*.sh`：本笔记抽样了 4 个（dense / MoE / SFT / reproducibility），多模型 recipe 的差异面归"模型适配"子系统
- `scripts/models/*.sh` 22 个：每个都是 17-30 行的 `MODEL_ARGS=(...)` 数组，结构高度同构
- 多数 `examples/*/run_*.sh`：本笔记只看了 3 个 README（fully_async / search-r1 / on_policy_distillation），其他 examples 的 launcher 模式应该接近
- `slime/utils/types.py`、`slime/utils/eval_config.py` 完整内容：归 customization 与 Data Buffer 子系统更合适
