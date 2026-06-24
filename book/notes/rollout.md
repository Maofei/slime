# rollout 子系统研究笔记

> 对应书中第 5 章 "rollout loop——SGLang 这一侧"

## TL;DR

rollout 子系统是 slime 的"样本生产线"，它把 prompt 喂给 SGLang，拿回带 reward 的 `Sample`，然后切成训练用的 micro-batch。核心入口是 `RolloutManager`（Ray remote actor，1485 行的 `slime/ray/rollout.py`），但真正"生成"那一步是被外部注入的——`--rollout-function-path` 指向一个自由函数，manager 只负责调用它、然后把返回的 `list[Sample]` 转成训练张量。SGLang 引擎本身被一个三层包装暴露：`SGLangEngine` Ray actor → `ServerGroup`（同构引擎组）→ `RolloutServer`（一个 router 后挂的一个模型）。slime 支持 6 种 rollout 模式（同步、流式、全异步、SFT、OPD、sleep），它们都只是一个用户层的函数实现，对 manager 透明。manager 自己只关心生命周期：起 router/engine、offload/onload 显存、健康检查、weight update 时取锁、把 sample 转成 `rollout_data_ref`。

## 1. 架构与模块边界

### 1.1 三层引擎抽象

`slime/ray/rollout.py` 把 SGLang 引擎拆成三层，自底向上：

```
RolloutManager (Ray actor, 1 个)
└── self.servers: dict[str, RolloutServer]
    └── server_groups: list[ServerGroup]
        └── all_engines: list[SGLangEngine actor]
```

- **`SGLangEngine`**（`slime/backends/sglang_utils/sglang_engine.py`，本子系统的"backends"边界）：一个 Ray actor，包一个真正的 SGLang HTTP server。
- **`ServerGroup`**（`rollout.py:105`）：一组**配置同构**的 engine（同 TP size、同 nodes_per_engine、同一个 placement group bundle）。一个 group 可以是 prefill、decode、encoder 或 regular——PD/EPD disaggregation 时一个模型会拆成多个 group。
- **`RolloutServer`**（`rollout.py:282`）：一个挂在**同一个 sgl-router 后面**的模型，可包含多个 group。`servers` 是 `dict[model_name, RolloutServer]`——`--sglang-config` 可以同时部署多模型（actor、reward model、reference model 并存），每个模型自带 router 端口。
- **`RolloutManager`**（`rollout.py:421`）：唯一对外的 Ray actor，是 driver 调用 `generate.remote(rollout_id)` 的入口。

### 1.2 6 种 rollout 模式的关系

manager 本身**不知道**有多少种 rollout 模式。所有模式都是一个普通 Python 函数，由 `--rollout-function-path` 注入：

| 模式 | 文件 | 何时用 |
|------|------|--------|
| 同步默认 | `slime/rollout/sglang_rollout.py:generate_rollout` | RL 主路径（GRPO/PPO） |
| 流式 | `slime/rollout/sglang_streaming_rollout.py:generate_streaming` | 不是顶层 rollout，是 `--custom-generate-function-path` 插件，让 abort 时已收到的 token 不丢 |
| 全异步 | `slime/rollout/fully_async_rollout.py:generate_rollout_fully_async` | 长尾——下一个 rollout 不用等本轮最慢的样本 |
| SFT | `slime/rollout/sft_rollout.py:generate_rollout` | 不调 SGLang，纯走 mask generator 算 loss_mask |
| OPD | `slime/rollout/on_policy_distillation.py:reward_func` + `post_process_rewards` | 不是 rollout，是 reward+post_process 钩子；让 teacher 通过 SGLang 算 logprob |
| sleep | `slime/rollout/sleep_rollout.py:sleep` | debug：让 rollout actor 永远睡，但 SGLang 还在跑（验证 backend 启动） |

边界提示：**SFT 和 OPD 严格说不属于"生成"**，但它们重用 `RolloutManager` 的样本-to-训练数据管线，所以走同一个槽位。

## 2. 关键抽象

### 2.1 `Sample`（`slime/utils/types.py:93-408`）

整个子系统的数据契约。一个 dataclass，跨越 prompt → response → reward → train data 整条流水线：

- 身份字段：`group_index`（同一 prompt 的 N 次采样共享）、`index`（全局序号）、`rollout_id`（用于"compact rollout"——一次 rollout 产出 N 个训练样本时所有兄弟共享同一 id，让 loss reducer 不重复计数；默认 None 时回退到 `index`）。
- 状态机：`Sample.Status` enum，5 个状态：`PENDING / COMPLETED / TRUNCATED / ABORTED / FAILED`。`FAILED` 是为 agent 场景留的"部分有效"语义。
- 关键方法 `append_response_tokens`（`types.py:253`）：流式/分段写 response 的统一入口，同时维护 `loss_mask`、`rollout_log_probs`、top-p replay 的 token ids/offsets（不可训练 token 会自动 mask=0、log_prob=0、top-p span 用空 padding）。
- 嵌套子结构 `SpecInfo`、`PrefixCacheInfo`——分别累计 speculative decoding 命中率和 prefix cache 命中率，每 chunk 都 `.add(meta_info)`。
- `to_dict / from_dict` 让 sample 可以 torch.save 进 debug dump（`--save-debug-rollout-data` / `--load-debug-rollout-data` 用这个 round-trip）。

### 2.2 RolloutFn 契约（`slime/rollout/base_types.py`）

```python
@dataclass
class RolloutFnTrainOutput: samples: list[list[Sample]]; metrics: dict[str, Any]
@dataclass
class RolloutFnEvalOutput:  data: dict[str, dict[str, Any]];  metrics: dict[str, Any]

def call_rollout_fn(fn, *args, evaluation, **kwargs):
    output = fn(*args, **kwargs, evaluation=evaluation)
    if not isinstance(output, (RolloutFnTrainOutput, RolloutFnEvalOutput)):
        output = RolloutFnEvalOutput(data=output) if evaluation else RolloutFnTrainOutput(samples=output)
    return output
```

一共 26 行——故意做得轻。**用户函数可以直接返回 `list[list[Sample]]`** 走 legacy 路径，框架会自动包装。这是 slime 给"customization 子系统"留的最小契约表面。

### 2.3 `RolloutManager.generate`（`rollout.py:546`）

整个子系统的 driver-facing 主入口，只有 14 行：

```python
def generate(self, rollout_id):
    self.health_monitoring_resume()
    if ci_test and use_fault_tolerance and rollout_id >= 2:
        self._try_ci_fault_injection()
    data, metrics = self._get_rollout_data(rollout_id=rollout_id)  # → call user fn
    self._save_debug_rollout_data(...)
    _log_rollout_data(...)
    if debug_rollout_only: return
    data = self._convert_samples_to_train_data(data)               # → rewards + masks
    return self._split_train_data_by_dp(data)                       # → 1 Box per DP rank
```

### 2.4 `_convert_samples_to_train_data`（`rollout.py:713-824`）

把 `list[Sample]` 摊平成训练用 dict。最关键的两步：
- `_post_process_rewards`（`rollout.py:686`）：对 GRPO/GSPO/CISPO/RPP+baseline 做 group-norm（reshape 成 `(rollout_batch_size, n_samples_per_prompt)` 后减均值、可选除标准差）。
- `rollout_mask_sums`（`rollout.py:773-778`）：每个 rollout 的 total mask sum 被**复制给该 rollout 里的每个 sample**，作为 per-mb reducer 的分母——micro-batch 切分可能把一个 rollout 的多个 sample 切到不同 mb 里，这个字段让 loss 仍然能按整个 rollout 求 token-加权均值。

### 2.5 `_split_train_data_by_dp`（`rollout.py:829-895`）

把 train_data 按 DP rank 切片，每个 rank 一个 `Box`（slime 自家的 ray reference wrapper）。schedule 由 `build_dp_schedule`（`slime/utils/dp_schedule.py`）独立计算，可以 unit test。支持 `--rollout-data-transport` = `object-store`（默认 ray.put）或 `nixl`（GPU-to-GPU 直传）。

### 2.6 `_validate_rollout_id_annotated`（`rollout.py:898-927`）

校验"compact rollout"契约。默认 shape 是 `list[list[Sample]]`（prompt × n_samples），叶子 list 在 depth=1，不校验。compact/subagent 会出现 `list[list[list[Sample]]]` 的 depth=2 叶子（一次 rollout 拆出 N 个训练 sample），此时强制要求：所有 sibling 必须有 `rollout_id`，且全部相同。

### 2.7 `GenerateState`（`sglang_rollout.py:84`）

SGLang 默认 rollout 的全局状态，单例（`SingletonMeta`）。三件事：
- `asyncio.Semaphore(sglang_server_concurrency * num_engines)`——背压。
- `dp_rank_context()`——每发起一个请求，从 `dp_counts` 选最空闲的 dp rank（least-loaded），用 `np.random.choice` 在所有最小者中抽一个。**这是 router 之外 slime 自己做的二次负载均衡**。
- `pendings: set[Task]`——所有在飞 task 的集合，用于 abort 时收尾。

## 3. 数据流

### 3.1 从 `generate.remote()` 到 `rollout_data_ref` 的完整链路

```
driver: ray.get(rollout_manager.generate.remote(rollout_id))    [train.py:67]
  └── RolloutManager.generate(rollout_id)                        [rollout.py:546]
        ├── health_monitoring_resume()
        ├── _get_rollout_data(rollout_id):                       [rollout.py:635]
        │     └── call_rollout_fn(self.generate_rollout, ...)    [base_types.py:19]
        │           └── (user fn, e.g.) generate_rollout_async   [sglang_rollout.py:375]
        │                 ├── while len(data) < target:
        │                 │     ├── samples = data_source(over_sampling_batch_size)
        │                 │     ├── state.submit_generate_tasks(samples)
        │                 │     │     └── asyncio.create_task(generate_and_rm_group(...))
        │                 │     │           └── for sample in group:
        │                 │     │                 └── asyncio.create_task(generate_and_rm)
        │                 │     │                       ├── async with state.semaphore:
        │                 │     │                       │     └── generate(): POST router/generate
        │                 │     │                       │           └── sample.append_response_tokens(...)
        │                 │     │                       └── sample.reward = await async_rm(args, sample)
        │                 │     └── await asyncio.wait(state.pendings, FIRST_COMPLETED)
        │                 │     └── dynamic_filter → may drop group
        │                 └── abort(): /abort_request → drain pending → 部分 sample 回写 buffer
        ├── _save_debug_rollout_data(...)
        ├── _convert_samples_to_train_data(data):                [rollout.py:713]
        │     ├── _post_process_rewards: group-norm rewards
        │     ├── loss_masks 整理
        │     └── rollout_mask_sums 广播
        └── _split_train_data_by_dp(data):                       [rollout.py:829]
              ├── build_dp_schedule(...): 计算 micro-batch 切分
              └── for r in range(dp_size): Box(ray.put(rollout_data))
        return list[Box]  ←  这就是 rollout_data_ref
```

### 3.2 Sample 状态机

```
data_source.get_samples() → Sample(status=PENDING, prompt, tokens=[])
        │
        ├─ generate() OK     → COMPLETED  (finish_reason=stop)
        ├─ hit max_new_tokens → TRUNCATED  (finish_reason=length)
        ├─ /abort_request    → ABORTED   (finish_reason=abort)
        └─ tool/api error    → FAILED    (agent-only, 部分有效)

ABORTED 样本路径分歧：
  - 非 partial_rollout：丢弃；下一轮重抽 prompt。
  - partial_rollout=True：abort() 把它们打包返回 → `data_source.add_samples` 回写 buffer，
    带上 metadata["start_rollout_id"]；下一轮重新调度时延续 `tokens` 已有内容（off-policy 部分可被 mask 掉）。
  - fully_async：worker 的 done_cb 直接 `add_samples` 回 buffer，不进 output_queue。
```

## 4. 设计模式

1. **三层引擎抽象 + property 投影**：`RolloutServer.engines`、`engine_gpu_counts`、`engine_gpu_offsets` 都是 property，按需把 group 嵌套结构摊平给 weight-sync 用——避免维护重复状态。
2. **Singleton 全局状态 + 上下文管理器**：`GenerateState` 是单例，但 `dp_rank_context()` 用 `@contextmanager` 保证计数器在 task 异常时也能减回去。
3. **Producer/Consumer 异步循环**：`generate_rollout_async` 的 while 循环既是 producer（不停往 pendings 塞）又是 consumer（FIRST_COMPLETED 取出处理）——经典 "持续 over-sample 直到 dynamic filter 攒够"。
4. **Background thread + asyncio loop**：`AsyncRolloutWorker._thread_main` 在专属线程里跑 `asyncio.run(self._loop())`，跨 rollout 边界保留 warm queue。
5. **Compact rollout 契约 = 形状约定 + 校验函数**：不引入新类型，靠 `list` 嵌套深度区分 default vs compact，配 `_validate_rollout_id_annotated` 静态保障。
6. **Ray Box 包装 + 可插拔 transport**：`Box(ray.put(data))` 默认走 object-store，`_tensor_transport="nixl"` 走 GPU-direct——透明切换不改 caller。
7. **External 锁 + 锁所有权外移**：`rollout_engine_lock` 由 manager 持有但**借给 train actor**（`get_updatable_engines_and_lock` 把 lock 作为返回值之一）；weight update 期间 train actor 取锁阻止 rollout 抢 GPU 内存。
8. **Health monitor pause/resume**：长操作（offload、recover）前后 `pause/resume`，避免心跳把正在迁移的 engine 误判为 dead。
9. **CI fault injection 内置**：`_try_ci_fault_injection`（`rollout.py:474`）在 generate 时主动 crash 一个 engine——把"recovery 路径正确性"做成 CI 一等公民。

## 5. 集成点

### 5.1 与 Data Buffer 的接口

只走两个方法（`DataSource` 抽象类，`data_source.py:17`）：
- 拉 prompt：`samples = data_source.get_samples(over_sampling_batch_size)` → `list[list[Sample]]`，外层是 group，内层是同 prompt 的 N 次采样。
- 回写：`data_source.add_samples(aborted_samples)`（partial_rollout / fully_async ABORTED 回流路径）。
- 持久化：`data_source.save(rollout_id) / load(rollout_id)`——manager 透传给 driver 调度的 checkpoint hook。

manager 在 `__init__` 时用 `load_function(args.data_source_path)` 加载 data source class，**永远不直接操作 buffer 内部结构**。

### 5.2 与 SGLang engine 的接口

只通过两条通道：
- **HTTP**（数据面）：rollout 函数自己拼 `http://{router_ip}:{router_port}/generate`，走 `slime.utils.http_utils` 的全局 httpx client。slime 不在 Python 层拿 engine handle 发请求——一律走 router。
- **Ray remote**（控制面）：`engine.release_memory_occupation / resume_memory_occupation / update_weights / check_weights / health_generate / simulate_crash`。所有 GPU 内存动作走 ray，所有推理动作走 HTTP。

router 启动在 `_start_router`（`rollout.py:1019`），用 `sglang_router.launch_router.RouterArgs.from_cli_args(args, use_router_prefix=True)`——所有 sgl-router CLI 参数被 slime 透明转发。

### 5.3 与 customization hook 的注入点

manager 在 `__init__` 加载 4 个用户函数（全部可选）：
- `args.rollout_function_path` → `generate_rollout`（必填）
- `args.eval_function_path` → `eval_generate_rollout`（必填）
- `args.custom_reward_post_process_path` → `custom_reward_post_process_func`（覆盖 `_post_process_rewards`）
- `args.custom_convert_samples_to_train_data_path` → 覆盖整个 `_convert_samples_to_train_data`

二级钩子在 `sglang_rollout.py` 里（被 default rollout 调用）：
- `args.custom_generate_function_path` / `Sample.generate_function_path`（per-sample 优先级更高）
- `args.dynamic_sampling_filter_path` → group 级别 keep/drop
- `args.rollout_sample_filter_path` → batch 完成后再过一遍
- `args.rollout_all_samples_process_path` → 拿到所有 oversample 样本（包括被 drop 的）做日志
- `args.custom_rollout_log_function_path` / `args.custom_eval_rollout_log_function_path` → 完全接管日志

eval 数据集级别钩子（`EvalDatasetConfig`）：每个数据集可单独指定 `custom_generate_function_path` / `custom_rm_path`，通过给 `Sample` 字段赋值传递给 `generate_and_rm`。

## 6. 出人意料的决策

### 6.1 manager 真正核心只有 ~200 行；1485 行的 80% 是 SGLang 启动 + metric 抽取

`RolloutManager` 类本体（line 421-633）约 213 行。其余分布：
- `ServerGroup` + `RolloutServer`：~315 行（多 group 调度、PG 编排）
- `start_rollout_servers` + `_start_router` + `_allocate_rollout_engine_addr_and_ports_normal` + `_resolve_sglang_config`：~340 行（**端口/host/PG bundle 的 bookkeeping 占了启动逻辑大头**——因为同节点上不同 server group 不能撞端口，所以 `port_cursors: dict[node_idx, port]` 要在 group 间串起来）
- metric 计算（`_compute_*_metrics`、`compute_metrics_from_samples`、`compute_perf_metrics_from_samples`）：~250 行——专门把 sglang trace span 的 attrs 抽出来做 prefill/decode 阶段的 P50/P99 统计

也就是说，**1485 行里"rollout 逻辑"非常少**，主要复杂度在引擎部署和性能可观测。

### 6.2 流式 rollout 是"反 abort"而生，不是"反延迟"

`sglang_streaming_rollout.py` 的注释说得很白：流式版本的目的**不是降低首 token 延迟**，而是 abort 时不依赖 `/abort_request` 的返回值——每个 SSE chunk 都把累计 state 写回 `sample.tokens / response / log_probs`，断流瞬间 sample 已经在最后一个 chunk 的状态。这是为了 partial_rollout 和 weight update 时部分样本可被无损"截图"。代码里还埋了一个 footgun 警告：sglang 默认是**累积式**流（每个 chunk 包含 from-start 的全量），如果哪天打开 `--incremental-streaming-output`，这里要改成 delta 拼接。

### 6.3 `compact rollout` 用列表嵌套深度区分形状

`_validate_rollout_id_annotated` 既不是泛型也不是新类型——它直接 walk 用户返回的嵌套 list，靠**叶子 list[Sample] 出现的 depth** 推断"这是默认 1-rollout-1-sample 还是 compact N-samples-per-rollout"。default 在 depth=1 跳过校验；compact 在 depth≥2 强制 sibling 共享 rollout_id。这避免了引入新 dataclass，也保留了 legacy `list[list[Sample]]` 兼容。`_fanout_test_helpers.py:42` 的 `compact_generate` 是参考实现（一次 sglang 调用，deepcopy N-1 次，全部赋同一个 `rollout_id`）。

### 6.4 `rollout_mask_sums` 是分布式 loss reducer 的"分母广播"

按理说一个 rollout 的 loss 应该 = `sum(per-token loss) / sum(mask)`。但 DP+micro-batch packing 会把一个 rollout 的多个 sample 切到不同 mb 里去——每个 mb 局部分母都不对。slime 在 manager 这一侧（**它能看到全量 sample**）把每个 rollout 的总 mask sum 算出来，复制给该 rollout 的每个 sample。训练侧的 reducer 用这个全局分母做加权求和，结果天然正确。这是个少见的"在数据组装阶段提前算 reducer 元数据"的做法——避免训练侧再做一次 all-reduce。

### 6.5 dp_rank 是 slime 在 router 之外的二次负载均衡

`GenerateState.dp_rank_context()` 维护 `dp_counts: list[int]`（长度 = `sglang_dp_size`），每次发请求挑当前最小的 dp rank。sgl-router 已经做了 worker 级别均衡，但**router 看不到 dp_attention 内部的 dp rank 占用**——slime 自己跟踪后通过 sampling_params 把 dp_rank 透传给 SGLang。这是 router 抽象的一个泄漏点。

### 6.6 fully_async 是个全局单例的后台线程，跨 rollout 调用复用 queue

`fully_async_rollout.py:49` 的 `_global_worker` 是模块级单例，`_get_global_worker` 用 `threading.Lock` 守护——`generate_rollout_fully_async` 每次被调用时**不创建新 worker**，而是从 worker 的 `output_queue` 里 drain 已完成的 group。这意味着 worker 上一轮 rollout 没收完的 in-flight 任务，下一轮调用还在跑。结合 `atexit.register(_stop_global_worker)` 优雅退出。

### 6.7 多模型部署：`servers` 是 dict，按名字找路由

`args.sglang_model_routers = {name: (ip, port)}` 在 `start_rollout_servers` 末尾设置，自定义 rollout 函数可以用 `get_model_url(args, "ref", "/generate")` 调用 reference model。`update_weights=True` 只有一个 server（默认第一个），其他都是 frozen。`_get_updatable_server` 在多 server 时返回首个 updatable——目前**多模型权重同步还没支持**（明确写在 docstring 里），这是已知 limitation。

### 6.8 PD/EPD 启动是两阶段：encoder 先起，URL 注入 LLM workers

`start_rollout_servers` 里 `has_epd` 分支（`rollout.py:1171-1205`）做两阶段：
1. 起所有 encoder group，**同步等 `ray.get(handles)`**，再 `ray.get([e.get_url.remote() for e in group.engines])` 收集 URL。
2. 起 prefill/regular group 时把 `encoder_urls` 通过 `sglang_overrides["encoder_urls"]` 注入。

这是为啥 EPD 路径不能像普通路径那样把所有 init 异步并发——encoder URL 是 LLM init 的必需参数，存在跨 group 强依赖。

### 6.9 `recover()` 重建死掉的 engine，并区分 updatable / non-updatable

`RolloutServer.recover()`（`rollout.py:341`）扫描 `all_engines` 中 `None` 的槽位，重新 `start_engines`，然后：
- 如果是 updatable model：触发 `release_memory_occupation` + `resume_memory_occupation(WEIGHTS)`——等 train actor 后续 `update_weights` 灌进来。
- 如果是 non-updatable model 且有 `model_path`：调 `update_weights_from_disk(model_path)` 自己从 checkpoint 恢复，避免占 CPU 内存做 weight backup。

`num_new_engines` 字段是给 train actor 看的——actor 知道有 N 个新 engine 后会跳过他们的 weight broadcast、改用 P2P 单独同步。

## 附录

### A. 扫读文件清单（已读但未深挖）

- `slime/rollout/_fanout_test_helpers.py`：compact rollout 的参考实现 + GRPO group_index 自定义归一化，用于 e2e test
- `slime/rollout/forge_load.py`：把 dump 出来的 sample 重新 load 进 rollout 管线，用于 memory test（保留完整 SGLang 启动）
- `slime/rollout/data_source.py`：只看了 DataSource 抽象类签名，详细归 Data Buffer 子系统

### B. 未读文件清单（边界外，归其他章节）

- `slime/rollout/filter_hub/` → customization 子系统
- `slime/rollout/rm_hub/` → customization 子系统
- `slime/backends/sglang_utils/` → backends 子系统（`SGLangEngine.__init__`、router 启动细节）
- `slime/agent/` → agent harness 独立子系统
- `slime/utils/dp_schedule.py:build_dp_schedule` → 训练侧打包逻辑，归 training 章节
- `slime/utils/health_monitor.py:RolloutHealthMonitor` → 工程基础设施

### C. 关键代码地址速查

| 抽象 | 位置 |
|------|------|
| `RolloutManager` | `slime/ray/rollout.py:421` |
| `RolloutServer` | `slime/ray/rollout.py:282` |
| `ServerGroup` | `slime/ray/rollout.py:105` |
| `generate` 入口 | `slime/ray/rollout.py:546` |
| `_convert_samples_to_train_data` | `slime/ray/rollout.py:713` |
| `_split_train_data_by_dp` | `slime/ray/rollout.py:829` |
| `_validate_rollout_id_annotated` | `slime/ray/rollout.py:898` |
| `start_rollout_servers` | `slime/ray/rollout.py:1089` |
| `_start_router` | `slime/ray/rollout.py:1019` |
| `Sample` | `slime/utils/types.py:93` |
| `Sample.append_response_tokens` | `slime/utils/types.py:253` |
| `RolloutFnTrainOutput / call_rollout_fn` | `slime/rollout/base_types.py:8 / 19` |
| `GenerateState` | `slime/rollout/sglang_rollout.py:84` |
| 默认 `generate_rollout_async` | `slime/rollout/sglang_rollout.py:375` |
| `generate_and_rm_group` | `slime/rollout/sglang_rollout.py:294` |
| `abort` | `slime/rollout/sglang_rollout.py:336` |
| 流式 `generate_streaming` | `slime/rollout/sglang_streaming_rollout.py:43` |
| `AsyncRolloutWorker` | `slime/rollout/fully_async_rollout.py:76` |
| SFT rollout | `slime/rollout/sft_rollout.py:17` |
| OPD reward+post_process | `slime/rollout/on_policy_distillation.py:8` |
| validation 校验 | `slime/ray/rollout_validation.py:1` |
