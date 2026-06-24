# 研究笔记：training 子系统（Megatron 侧训练循环）

## TL;DR

slime 的 training 子系统是一个**极薄的训练循环**——`train.py` 总共只有 103 行，主循环用一个朴素的 `for rollout_id in range(...)` 拉数据、训练、回写权重，把所有"重的事"（rollout 生成、weight sync、Megatron 模型构造、CP 切分）都委托给下游模块。它的核心抽象是：把"每个 GPU 上的 Megatron 训练 actor"封装为一个 Ray actor (`MegatronTrainRayActor`)，actor 内部直接调用 Megatron 的 `forward_backward_func`。设计上最非显然的地方是：(1) 训练侧的 loss 函数会按 micro-batch / global-batch / CP / DP 做一个**精心反向放缩**，使得最终梯度等价于"per-rollout-mean"——这套放缩在 `loss_function` + `cp_utils.reduce_train_step_metrics` + Megatron 自身的 mb 缩放 + DDP 的 grad average 中分四步组合而成；(2) ref/teacher/old_actor 这种"多份权重"不是开多个模型实例，而是**通过 CPU pinned-memory 的 `TensorBackuper` 在同一组 GPU 参数上切换**；(3) 数据进入 train 之前已经在 rollout 侧由 `dp_schedule.build_dp_schedule` 算好了 DP 分区和 mbs 调度，train 侧只是"按计划执行"。

---

## 1. 架构与模块边界

训练这一侧有四层，自上而下：

1. **主循环（`train.py`）**：一个进程，运行在驱动节点。它只有一个 for 循环，每个 `rollout_id` 做：拉数据 → actor.async_train → 周期性 save → update_weights → 周期性 eval。所有可选分支（offload、use_critic、check_weight_update_equal）都在这一层用 if 串起来。整个文件 103 行，逐行可读。
2. **Ray 层（`slime/ray/train_actor.py` + `slime/ray/actor_group.py`）**：`TrainRayActor` 是抽象基类，定义了 `init / train / save_model / update_weights / sleep / wake_up` 接口；`RayTrainGroup.async_train` 把对所有 worker 的 `train.remote` 收成一个 ref 列表返回。这一层不知道 Megatron，只知道"每个 GPU 一个 actor"。
3. **Backend actor（`slime/backends/megatron_utils/actor.py`，674 行）**：`MegatronTrainRayActor` 实现 `TrainRayActor`。这是子系统里最厚的一层，负责：模型/optimizer 初始化、CPU pinned 权重备份、ref/teacher/old_actor 多份权重切换、`train_actor` / `train_critic` 主流程、log_prob 预算、`compute_advantages_and_returns` 调用、调 `train()` 触发 Megatron forward-backward、`update_weights` 调度。
4. **Megatron 适配（`model.py` / `loss.py` / `data.py` / `cp_utils.py`）**：直接调用 Megatron 的 `get_forward_backward_func`、`DistributedDataParallel`、`get_megatron_optimizer`、`OptimizerParamScheduler`。slime 不重写训练循环本身，只是把 forward step 写成"喂 packed token + 在 loss 函数里做 PPO/GRPO/CISPO"。

边界很清楚：**train.py 不知道 Megatron**；**Ray actor 不知道 PPO**；**Megatron 适配层不知道 Ray**。

---

## 2. 关键抽象

| 抽象 | 路径 | 说明 |
| --- | --- | --- |
| `train(args)` | `train.py:9-98` | 全书主线，整本书都会反复回到这 90 行 |
| `MegatronTrainRayActor.init` | `slime/backends/megatron_utils/actor.py:47-186` | 一次性把 Megatron + 模型 + ref/teacher/old_actor 备份 + weight updater 装好；用 `@with_defer` 装饰，函数返回后 Timer 自动 start `train_wait` |
| `MegatronTrainRayActor.train_actor` | `slime/backends/megatron_utils/actor.py:428-553` | 单个 rollout 的训练编排：算 ref/teacher log_prob → 切回 actor → `compute_advantages_and_returns` → `train()` |
| `train_one_step` | `slime/backends/megatron_utils/model.py:509-696` | 真正包装 Megatron pipeline-parallel forward-backward 的地方，含 nan-grad 跳过、`opt_param_scheduler.step(increment=step_global_batch_size)` |
| `loss_function` | `slime/backends/megatron_utils/loss.py:1215-1315` | 4 路分发（policy/value/sft/custom），并执行**关键的 mb/gbs/CP 反向放缩** |
| `compute_advantages_and_returns` | `slime/backends/megatron_utils/loss.py:657-824` | 唯一的 advantage 入口，分发 GRPO/GSPO/CISPO/PPO/REINFORCE++；返回值通过 `rollout_data` dict 原地写回 |
| `get_sum_of_sample_mean` | `slime/backends/megatron_utils/cp_utils.py:47-124` | 返回 `closure(x) -> scalar`，对每个样本按 loss_mask 加权求平均；CP=1 和 CP>1 走两条分支 |
| `reduce_train_step_metrics` | `slime/backends/megatron_utils/cp_utils.py:127-168` | 训练 step 完成后，把 per-mb 的 metric 字典在 DP*CP 组里 all_reduce 并按 cp_factor 反向消除 CP 膨胀 |
| `build_dp_schedule` | `slime/utils/dp_schedule.py:82-209` | rollout 侧（不是 train 侧！）调用的纯 Python 调度器，先 pack 再 distribute，决定 partition + micro_batch_indices + num_microbatches |
| `TensorBackuper` | `slime/utils/tensor_backper.py:10-74` | "多模型"的实现：把当前 GPU 参数拷到 CPU pinned-memory，restore 时再拷回来；ref/teacher/old_actor 都共享同一组 GPU 权重 |
| `StatelessAdam` | `slime/backends/megatron_utils/stateless_adam.py:7-109` | 不保存 `exp_avg/exp_avg_sq` 的 Adam，每步等价于"先把 moment 清零再算 Adam"，给一步式 rollout 用 |

---

## 3. 数据流：从 `rollout_data_ref` 到一次反传

```
ray.get(rollout_manager.generate.remote(rollout_id))     # train.py:67
        │
        ▼   list[Box] 长度 = dp_size，每个 Box 包一个 ray ref
actor.async_train(rollout_id, rollout_data_ref)          # train.py:81
        │
        ▼   广播给所有 train worker
MegatronTrainRayActor.train(...)                         # actor.py:378
        │
        ▼
self._get_rollout_data(rollout_data_ref)                 # actor.py:220-274
    │  ├─ process_rollout_data(args, ref, dp_rank, dp_size)  # utils/data.py:292
    │  │     ↳ ray.get(ref[dp_rank].inner) 只取本 rank 那一份
    │  ├─ tokens / loss_masks → GPU
    │  └─ rollout_log_probs 用 slice_log_prob_with_cp 切成 zigzag CP layout
        ▼
get_data_iterator(rollout_data)                          # data.py:241
    ↳ 每个 VPP 阶段一个 DataIterator，按 rollout_data["micro_batch_indices"] 取
        │
        ▼  if "ref" in backup_tags → switch_model("ref") → compute_log_prob
        ▼  if "teacher" in backup_tags → switch_model("teacher") → compute_log_prob
        ▼  switch_model("actor") (or "old_actor")
        ▼  forward_only(get_log_probs_and_entropy, ...) → rollout_data["log_probs"]
        ▼
compute_advantages_and_returns(args, rollout_data)       # loss.py:657
        │
        ▼  rollout_data["advantages" / "returns"] 就地写回
train(rollout_id, model, optimizer, scheduler, ...)      # model.py:704
        │
        ▼  for step_id in num_steps_per_rollout:
        ▼    train_one_step(...)                         # model.py:509
        ▼      forward_backward_func(forward_step, ...)
        ▼        forward_step(...) -> (logits, partial(loss_function, ...))
        ▼          get_batch(data_iterator, keys, pad, allgather_cp)  # data.py:28
        ▼            ↳ slice_with_cp → cat → pad → PackedSeqParams (thd)
        ▼          loss_function(...)                    # loss.py:1215
        ▼            ↳ policy_loss_function / value_loss_function / sft_loss_function
        ▼            ↳ 返回 (loss, num_tokens_or_1, {"keys", "values"})
        ▼            ↳ loss 被乘以 num_mbs * dp_with_cp / step_gbs，反向消除 Megatron 自己的除法
```

要点：rollout 侧的 `dp_schedule` 已经在 `rollout_data["micro_batch_indices"]` 里记录了"哪些 sample 进哪个 mbs"，train 侧不再做任何重排，只是按表执行。

---

## 4. 设计模式

- **Ray actor + 抽象基类**：`TrainRayActor` 定 abstract 接口，`MegatronTrainRayActor` 实现。书里如果要写 FSDP backend，只要再实一个子类，主循环和 ray 层都不用动。
- **薄主循环**：`train.py` 用 `should_run_periodic_action` 这样的纯函数检查"是否到了 save/eval"，没有 plugin/hook 系统，所有控制流摆在一个文件里。这是反工程化的取向——书可以强调这一点。
- **数据计划与执行分离**：DP 分区、micro-batch 计划、global_batch_sizes 全部在 rollout 侧由 `build_dp_schedule`（纯 Python，`dp_schedule.py`）算好，train 侧只是"按 plan 取数据"。这让所有调度逻辑可以在 CPU-only CI 上单元测试（`tests/test_dp_schedule.py`）。
- **forward_step 闭包传递 loss 函数**：Megatron 的 pipeline-parallel 调度要求 `forward_step` 返回 `(output_tensor, loss_callable)`。slime 把 args/batch/num_mbs/step_gbs **全部 partial 闭包**进 `loss_function`（`model.py:638`），让 Megatron 自己决定何时调它。
- **Context parallel 双 chunk zigzag**：CP 不是简单切两半，而是 `[cp_rank, 2*cp_size - cp_rank - 1]` 两块拼起来（`cp_utils.py:9-44`），保证 RingAttention 的负载均衡。`allgather_cp` 是另一条更朴素的 path（直接 chunk + all-gather），由 `args.allgather_cp` 控制；两条 path 在 loss/data 中都各有一套分支。
- **CPU pinned 权重备份做多模型**：见后文"出人意料的决策"。
- **per-rollout-mean 反向放缩**：见后文"出人意料的决策"。
- **`with_defer` + `inverse_timer` 拼装 train_wait 时间**：`init` 用 `@with_defer(lambda: Timer().start("train_wait"))` 装饰（actor.py:47），返回时就开始计 "train_wait"；`train_actor` 里用 `with inverse_timer("train_wait"), timer("train")` 把 train_wait 停掉、开始计 train。这套 trick 让"等数据的时间"自动覆盖 init 完到第一次 train 之间、以及每两次 train 之间的全部空隙。

---

## 5. 集成点

- **与 rollout 的接口**：`actor.async_train(rollout_id, rollout_data_ref, external_data=None)`（`ray/actor_group.py:131`）。`rollout_data_ref` 是 `list[Box]`，长度等于 dp_size，每个 Box 包一个 ray object ref；train actor 用 `dp_rank` 取自己那一份。`external_data` 用于 critic→actor 传 values（见 `train.py:75-77`）。
- **与 weight sync 的接口**：`actor_model.update_weights()`（`train.py:26 / 89`）。`MegatronTrainRayActor.update_weights`（`actor.py:580-644`）会：(a) 从 rollout_manager 拿"哪些 engine 需要更新"，(b) 调 `weight_updater.connect_rollout_engines` + `weight_updater.update_weights()`，(c) 如果开了 `keep_old_actor` 还要做 "actor → rollout_actor → old_actor" 的队列式滚动。`update_weight_cls` 在 `actor.py:139-159` 根据 `colocate / update_weight_mode / update_weight_transport` 三个开关分发到 4 个实现（Tensor/Distributed/DistributedDelta/Disk）。
- **与 Megatron 的接口**：
  - **forward-backward**：调 `get_forward_backward_func()`（标准入口）。
  - **优化器**：用 `get_megatron_optimizer`，可以临时 patch `megatron.core.optimizer.Adam` 为 `StatelessAdam`（`model.py:248-267` 的 `_patch_megatron_adam` 上下文管理器）。
  - **DDP grad sync**：依赖默认行为，但是 patch 了 `_allreduce_non_tensor_model_parallel_grads`（`megatron_patch/megatron_chunked_grad_coalesce_patch.py`），把 TP 侧的 SP/qk_layernorm grad 的 all_reduce 改成按 1 GiB 分块——大模型下 `_flatten_dense_tensors(grads)` 会一次性申请巨型连续 buffer，容易在 allocator 碎片下 OOM。Patch 还兼容了 core_v0.13 和 v0.15rc7 之后两条 Megatron 分支的 API 差异（`use_custom_fsdp` vs `use_megatron_fsdp`，`_get_main_grad_attr` 参数数量），靠 `inspect.signature` 运行时检测，**不做版本条件 import**。
  - **checkpoint**：`load_checkpoint` 既支持 Megatron `iter_xxx` 目录，也支持 HF 目录（通过 `megatron.bridge.AutoBridge`，见 `checkpoint.py:97-152`）。`save_hf_model_to_path` 反过来把 Megatron 权重导出成 HF safetensors（含 sharding + 多节点写）。
  - **额外 patch**：`__init__.py` 还 patch 了 `deep_ep.Buffer.__init__` 让 torch_memory_saver 不把 deep_ep 的显存当成"interesting region"——否则 offload 会动到这些 buffer。

---

## 6. 出人意料的决策（最重要部分）

### 6.1 "多模型"不是多份 GPU 权重，是一个权重 + CPU pinned 备份字典

ref / teacher / old_actor 这些"另一份模型"在大多数 RL 框架里是单独的模型实例（占用独立显存）。slime 选择了完全不同的做法：所有这些角色共用一组 GPU 参数，靠 `TensorBackuper`（`slime/utils/tensor_backper.py:42-74`）把"另一份权重"放到 CPU pinned-memory 字典里。`_switch_model("ref")` 时调 `restore("ref")` 把 CPU 字典逐 tensor 拷到 GPU（actor.py:276-280）。代价是 PCIe 带宽 + GPU↔CPU 拷贝，收益是 **N 个角色 = N × CPU memory + 1 × GPU memory**，对 MoE 大模型是质的差别。

更细节的是 `keep_old_actor` 模式下的滚动：`actor → rollout_actor → old_actor` 三档队列式备份（`actor.py:630-639`），用 `copy(src_tag, dst_tag)` 在 CPU 内完成，避免每次都从 GPU 重抓。

`_TensorBackuperNoop`（`tensor_backper.py:77-102`）是一个"只算 hash 不真备份"的 stub——当只有一份模型（不需要 ref/teacher）时用它，省 CPU 内存；它在 backup/restore 时校验 hash，确保调用者没误用。

### 6.2 loss 的"per-rollout-mean"放缩：4 处放缩组合成一个干净的梯度

`loss_function`（`loss.py:1285-1293`）里有这两行神秘的代码：
```python
if not args.calculate_per_token_loss:
    loss = loss * num_microbatches / step_global_batch_size * mpu.get_data_parallel_world_size(with_context_parallel=True)
else:
    loss = loss * mpu.get_context_parallel_world_size()
```
这不是"梯度按 batch 缩放"——这是**反向消除下游会做的除法**。`tests/test_loss_cp_invariance.py:24-43` 的注释把链路写得很清楚：

1. slime 自己 pre-scale：`× num_mbs / step_gbs × (dp × cp)`；
2. Megatron 的 schedules.py 会 `/= clamp(num_tokens, 1)`（=1 时是 no-op），然后 `/= num_microbatches`；
3. 反传攒梯度；
4. Megatron DDP 在 DP×CP 组里平均：`× 1 / (dp × cp)`。

四步合起来正好是 `total_sum_of_rollout_means / step_global_batch_size`，即"每个 rollout 算一个均值再对所有 rollout 求平均"的真正梯度——并且**不依赖 CP**。整本书最值得讲的就是这件事：作者宁可写一个 CPU 多进程 spawn 测试（`test_loss_cp_invariance.py`）来守住"CP 改变不影响梯度范数"这个不变量，也不愿意把放缩散开放到几个不同的地方。

`reduce_train_step_metrics`（`cp_utils.py:127-168`）做同一件事的对偶版本：metric 报告时按 `cp_factor` 反向消除 CP 膨胀。per-token-loss 模式下 `cp_factor = cp_size`，per-rollout-mean 模式下 `cp_factor = 1`——因为 per-token 路径的 num_tokens 在每个 CP rank 上都重复算了一次。

### 6.3 调度在 rollout 侧、执行在 train 侧——但 train 一行调度代码都没有

`build_dp_schedule`（`slime/utils/dp_schedule.py`）是这个项目里测试最重的纯函数之一（`test_dp_schedule.py`），它在 **rollout** 侧被调用，吐出 `(partitions, micro_batch_indices, num_microbatches, global_batch_sizes)` 四元组，塞进 `rollout_data` 里序列化过 Ray 传给 train。train actor 在 `get_data_iterator`（`data.py:241`）里直接 `rollout_data["micro_batch_indices"]`，不再做任何分组/打包。

这背后的理由是：(a) 调度算法（first-fit-pack、Karmarkar-Karp 双路平衡、按 FLOPs 分配）需要看到全 batch 的 seq lengths，rollout 侧本来就有；(b) 把调度抽到 CPU-only 纯函数后，CI 可以无 GPU 跑 invariant 测试——`test_dp_schedule.py` 验证的不变量包括"每个 DP rank 跑相同的 num_microbatches"（PP 要求）、"每个 mbs ≤ max_tokens_per_gpu × cp_size"（OOM 守护）、"所有样本恰好被切一次"（正确性）。

副作用是 train 侧的 `train_one_step` 接受一个 `step_global_batch_size: int` 参数（model.py:518），用来给 `opt_param_scheduler.step(increment=...)` 喂值——LR scheduler 的"已消费样本数"必须跟实际 rollout 数对齐，而不是用启动时估算的 `args.global_batch_size`，否则在 dynamic batching 下会走偏。`model.py:204` 那条 `args.train_iters = ...` 的注释直接承认这是个估算，只用来 size 一下 cosine schedule。

### 6.4 一次 train step 内部"重新计算 log_prob"的复杂走线

`train_actor`（`actor.py:464-491`）里有一段相当克制的 if：当 `can_reuse_log_probs_in_loss` 为真时，**根本不调** `compute_log_prob`——loss 函数内部第一次算 log_prob 时顺便把 detach 之后的当成 old_log_prob 用（`loss.py:924-925`）。这个优化的条件极苛刻：单 step、policy_loss、kl_coef==0、不用 rollout_logprobs、不要 mismatch metrics、不用 critic、不开 keep_old_actor、不开 OPD、不开 routing replay、不是 GSPO。任何一条不满足都得真去前向一次再算 log_prob，多消耗一次 forward。

这条路径直接说明了 RLHF 训练循环的一个常被忽略的事实：**"什么时候 old_log_prob 等于当前 actor 的 log_prob"是个 N 元布尔表达式**，作者把它写出来当条件而不是省略。

### 6.5 chunked grad coalesce patch：上游 Megatron 直接 OOM 的修补

`megatron_patch/megatron_chunked_grad_coalesce_patch.py` 是整个子系统里"为什么这样设计"最有故事的一段。问题是 Megatron 原生的 `_allreduce_non_tensor_model_parallel_grads` 用 `_flatten_dense_tensors(grads)` 把所有 TP 侧梯度 flatten 成一块大连续内存做一次 all_reduce——对大模型这块 buffer 动辄几 GiB，allocator 碎片下直接 OOM。slime 把它改成按 `SLIME_GRAD_COALESCE_CHUNK_BYTES`（默认 1 GiB）分块；因为 SUM/AVG 是 element-wise，分块在数学上完全等价。

更聪明的是 patch 怎么"塞回去"：因为 `megatron.core.distributed` 这个父包重新 export 了同名函数遮蔽掉了子模块，直接 `module.func = new_func` 不生效；slime 走 `sys.modules["megatron.core.distributed.finalize_model_grads"]` 直接拿到子模块本体来 setattr（patch 文件第 129 行）。这种 monkey-patch 在书里值得讲，因为它揭示了一个真相：**slime 不 fork Megatron，但偶尔需要外科手术式地改它**。

---

## 附录

### 已通读文件
- `train.py`（103）
- `slime/ray/train_actor.py`（128）
- `slime/backends/megatron_utils/actor.py`（674）
- `slime/backends/megatron_utils/loss.py`（1315）
- `slime/backends/megatron_utils/data.py`（540）
- `slime/backends/megatron_utils/model.py`（1007）
- `slime/backends/megatron_utils/cp_utils.py`（344）
- `slime/backends/megatron_utils/model_provider.py`（286）
- `slime/backends/megatron_utils/checkpoint.py`（157）
- `slime/backends/megatron_utils/hf_checkpoint_saver.py`（388）
- `slime/backends/megatron_utils/initialize.py`（113）
- `slime/backends/megatron_utils/__init__.py`（44）
- `slime/backends/megatron_utils/megatron_patch/__init__.py`
- `slime/backends/megatron_utils/megatron_patch/megatron_chunked_grad_coalesce_patch.py`（147）
- `slime/backends/megatron_utils/stateless_adam.py`（109）
- `slime/backends/megatron_utils/misc_utils.py`（5）
- `slime/utils/dp_schedule.py`（209）
- `slime/utils/seqlen_balancing.py`（238）
- `slime/utils/tensor_backper.py`（前 110 行；后半是 hash util）
- `slime/utils/timer.py:55-103`（timer/inverse_timer/with_defer）
- `slime/utils/data.py:280-303`（process_rollout_data）
- `slime/ray/actor_group.py:110-170`（async_train）

### 扫读文件（仅看了关键测试断言）
- `tests/test_chunked_gae.py`：守护 `chunked_gae` 与 `vanilla_gae` 在多 (B, T, chunk_size) 下数值一致（atol 1e-5）。
- `tests/test_cispo_loss.py`：守护 CISPO loss 的闭式形式 + 梯度只通过 log_probs 流（IS ratio 必须 stop-grad）。
- `tests/test_cp_utils.py`：守护 `slice_with_cp` / `all_gather_with_cp` 等 CP 几何变换。
- `tests/test_loss_cp_invariance.py`：CPU 多进程 spawn，端到端校验"同样 sample 在不同 (dp, cp) 因子化下 grad_norm 严格相等"。
- `tests/test_dp_schedule.py`：守护 `build_dp_schedule` 的 4 条不变量（每 rank 同 num_mbs / 单 mbs ≤ token 上限 / 所有 sample 被覆盖 / 每 rank mbs 平铺）。

### 边界外（注意不要在本章写）
- `slime/backends/megatron_utils/update_weight/*`：weight sync 子系统。
- `slime/backends/megatron_utils/sglang.py`：rollout 引擎适配，归 backends 章。
- `slime/utils/ppo_utils.py`：advantage/loss 的纯数学实现，归"算法"章。
- `slime/utils/reloadable_process_group.py` + `torch_memory_saver`：offload 机制，归 deployment / topology 章。
- `slime/utils/routing_replay.py` + R3 相关 stage 切换：归 MoE / 算法章。

### 未读 / 推迟阅读
- `slime/backends/megatron_utils/ci_utils.py`（84 行）：CI 守护检查，应在"工程基础设施"章节顺便提一下。
- `slime/backends/megatron_utils/arguments.py`（199 行）：参数补丁，归"CLI / 参数"章。
- `train_async.py`：异步训练入口，本章只看了同步路径；如要写 async 模式应该和这本子系统一起讲。
