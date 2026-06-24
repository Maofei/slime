# Ray 编排层研究笔记

## TL;DR

slime 用 Ray 解决的核心问题是：**在一组共享的 GPU 上，同时调度"训练侧的 Megatron actor 进程组"与"推理侧的 SGLang engine actor 进程组"，并让它们能够交替使用 GPU 显存**。Ray 在 slime 里只承担三件事：(1) 用 placement group 把 GPU 切成确定性的逻辑顺序（按节点 IP + GPU id 排序），(2) 把每张 GPU 上的进程包装成 named actor，(3) 提供 `actor.method.remote()` 与 `ray.get()` 作为跨进程 RPC。真正的训练/推理业务逻辑都在 actor 内部，Ray 这一层很薄但非常关键——它定义了 placement 契约（test_placement_group 守护的不变量），定义了 actor 的生命周期接口（init / train / update_weights / sleep / wake_up / offload / onload），并通过一个特殊的 `RolloutManager` actor 作为训练循环与所有推理 engine 之间的代理。

---

## 1. 架构与模块边界

```
train.py / train_async.py
        │
        ▼
slime/ray/placement_group.py
    ├─ create_placement_groups(args)          ──► 决定总 GPU 数与 actor/rollout offset
    ├─ create_rollout_manager(args, pg)       ──► 生成 RolloutManager actor
    └─ create_training_models(args, pgs, ...) ──► 生成 actor_model / critic_model (RayTrainGroup)

slime/ray/actor_group.py
    └─ RayTrainGroup            ── world_size 个 TrainRayActor 的集合，封装 async_train / update_weights / sleep / wake_up

slime/ray/train_actor.py
    └─ TrainRayActor            ── 单个训练进程的抽象基类（每张 GPU 一个），子类是 MegatronTrainRayActor

slime/ray/rollout.py            ── RolloutManager + RolloutServer + ServerGroup（推理 actor 编排）
    ├─ RolloutManager           ── @ray.remote 单例，代理所有 SGLang engine 的 RPC，承载 rollout/eval/数据转换
    ├─ RolloutServer            ── 一个模型 + 一个 router + 多个 ServerGroup
    └─ ServerGroup              ── 一组同构的 SGLang engine（worker_type: regular/prefill/decode/encoder/placeholder）

slime/ray/ray_actor.py          ── RayActor 基类：只提供 _get_current_node_ip_and_free_port，所有 actor 共享
slime/ray/rollout_validation.py ── 一个纯函数，验证 server group 的 GPU 索引落在 placement group 内
slime/ray/utils.py              ── NOSET_VISIBLE_DEVICES 列表、RAY_USE_UVLOOP=0、Lock actor
```

关键边界：
- **训练业务逻辑（loss / optimizer / megatron pp/tp）在 `slime/backends/megatron_utils/actor.py:MegatronTrainRayActor` 里**，本笔记不展开。
- **rollout 业务逻辑（generate_rollout、sample 构造、KV/log_prob 采集）在 `slime/rollout/` 与 `slime/backends/sglang_utils/`**，本笔记不展开。
- **backend engine 初始化（SGLangEngine、Megatron 初始化）在 backends 子系统**，本笔记只看到调用点（`engine.init.remote(...)` / `actor.init.remote(args, role, ...)`）。

---

## 2. 关键抽象

### 2.1 placement 层

- `slime/ray/placement_group.py:16` `InfoActor`：一个 `num_gpus=1` 的临时 actor，专门用来回报 `(node_ip, gpu_id)`。用完即 `ray.kill`。
- `slime/ray/placement_group.py:21` `sort_key(x)`：先按节点 IP 数值排序，再按 GPU id；遇到无法解析 IP 的 hostname 退化为 ASCII 数值，但**保证排序稳定**。这是 placement 契约的核心——它决定了 rank 0 总是落在 IP 最小节点的 GPU 0。
- `slime/ray/placement_group.py:42` `_create_placement_group(num_gpus)`：用 `strategy="PACK"` 创建一个含 `num_gpus` 个 bundle 的 pg，每 bundle = 1 GPU + 1 CPU。返回 `(pg, reordered_bundle_indices, reordered_gpu_ids)`——后两者是**逻辑 rank → 物理 bundle/GPU 的查询表**。
- `slime/ray/placement_group.py:100` `_get_placement_group_layout(args)`：根据 `colocate / debug_train_only / debug_rollout_only / rollout_external` 五种模式返回 `(num_gpus, rollout_offset)`。这是整个系统的 GPU 拓扑决策点。
- `slime/ray/placement_group.py:120` `create_placement_groups(args)`：唯一对外入口，返回 `{"actor": (pg, indices, gpus), "rollout": (pg, indices, gpus), "critic": ...}`，三者**共享同一个 pg**，只在 indices/gpus 的偏移上不同。

### 2.2 actor 集合层

- `slime/ray/ray_actor.py:4` `RayActor`：所有 Ray actor 的基类，10 行，只提供 IP/free-port 探测。
- `slime/ray/actor_group.py:10` `RayTrainGroup`：world_size 个训练 actor 的集合。重要细节：
  - `slime/ray/actor_group.py:103` `TrainRayActor = ray.remote(**actor_options)(actor_impl)`——在这里把普通 Python 类 (`MegatronTrainRayActor` 或用户自定义 `actor_cls`) **动态包成 ray remote class**。
  - `slime/ray/actor_group.py:108` 循环 `for rank in range(world_size)` 按 `reordered_bundle_indices[rank]` 一对一占用 bundle。
  - `slime/ray/actor_group.py:118` rank 0 启动后立刻 `ray.get(actor.get_master_addr_and_port.remote())`，后续 rank 全部复用这个 master——这就是 torch.distributed 的 rendezvous。
  - `slime/ray/actor_group.py:121` `async_init`、`:131` `async_train` 返回 `list[ObjectRef]`（即 fire-and-forget），caller 决定何时 `ray.get`；`save_model` / `update_weights` / `onload` / `offload` 反之，内部直接 `ray.get`。前缀 `async_` 是这个边界的命名规则。
- `slime/ray/train_actor.py:28` `TrainRayActor`：单 actor 基类。在 `__init__` 里通过 rank 0 拿到 master_addr/master_port，并设置 `MASTER_ADDR / MASTER_PORT / WORLD_SIZE / RANK / LOCAL_RANK` 环境变量，让 actor 内部的 `dist.init_process_group()` 能在 `init()` 阶段直接成功。
- `slime/ray/train_actor.py:50` `init(args, role, with_ref, with_opd_teacher)`：建立 NCCL 进程组、绑定 NUMA 亲和（`pynvml.nvmlDeviceSetCpuAffinity`）。`@abc.abstractmethod` 的 `sleep / wake_up / train / save_model / update_weights / _get_parallel_config` 是 backend 必须实现的钩子。
- `slime/ray/train_actor.py:125` `set_rollout_manager(rollout_manager)`：训练 actor 拿到 rollout manager 的 handle 后，只有 rank 0 会反向把自己的 `train_parallel_config` 注册回 rollout manager（用于 dp 切分）。

### 2.3 rollout 编排层

- `slime/ray/rollout.py:106` `@dataclasses.dataclass ServerGroup`：一组同构 SGLang engine 的容器。字段记录 `pg / all_engines / num_gpus_per_engine / worker_type / rank_offset / gpu_offset / needs_offload / model_path / router_ip / router_port`。
  - `slime/ray/rollout.py:137` `start_engines(port_cursors)`：把每个 engine 包成 `ray.remote(SGLangEngine)`，按 `num_gpus=0.2` 占资源（**故意小于 1**，避免和训练 actor 抢同一个 bundle），用 `PlacementGroupSchedulingStrategy(bundle_index=reordered_bundle_indices[gpu_index])` 钉到具体 GPU。返回 `(init_handles, port_cursors)` 不阻塞。
  - `slime/ray/rollout.py:249/259/269` `offload / onload / onload_weights_from_disk`：以 sglang 的 `release_memory_occupation / resume_memory_occupation` 为基础，支持按 `[GPU_MEMORY_TYPE_WEIGHTS / KV_CACHE / CUDA_GRAPH]` 标签做精细化的显存释放。
- `slime/ray/rollout.py:283` `RolloutServer`：一个 router 后面挂 N 个 `ServerGroup`（典型场景：PD 解聚时 prefill TP=2、decode TP=4 是两个 group）。`recover()` 在 health monitor 检测到 engine 死亡后并发重启所有 group 的 dead engine。
- `slime/ray/rollout.py:421` `@ray.remote class RolloutManager`：整个 rollout 侧的代理。它**自己不占 GPU**（`num_cpus=1, num_gpus=0`），但持有所有 SGLang engine actor 的引用、router 进程、Lock actor、HealthMonitor，并以 `generate / eval / save / load / offload / onload / update_weights 相关方法` 作为对训练循环的统一接口。

### 2.4 跨层辅助

- `slime/ray/utils.py:46` `Lock` actor：一个 `@ray.remote` 的非重入互斥锁，10 行。被 `RolloutManager.get_updatable_engines_and_lock` 返回给训练侧，用来串行化 weight update（多个 dp 训练 rank 同时 push 时不能并发触碰同一个 sglang engine）。
- `slime/ray/utils.py:16` `NOSET_VISIBLE_DEVICES_ENV_VARS_LIST`：枚举所有 Ray 私有的 `RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES` 环境变量，统一设为 `"1"` 让 backend（megatron / sglang）自己管理可见设备。
- `slime/ray/utils.py:26` `RAY_DEFAULT_ENV_VARS = {"RAY_USE_UVLOOP": "0"}`：源码里的注释说 "Ray's uvloop integration has caused intermittent async actor issues."——这是一条踩过坑的默认值。

---

## 3. 数据流：从 `python train.py` 到所有 actor 就位

```
python train.py
  │
  ├─ parse_args()
  ├─ create_placement_groups(args)
  │     ├─ _get_placement_group_layout(args)          ──► (num_gpus, rollout_offset)
  │     ├─ ray.util.placement_group(bundles, PACK)
  │     ├─ wait pg.ready() （每 30s 打印一次 GPU 注册数，永不超时）
  │     ├─ 临时 InfoActor × num_bundles 探测 (ip, gpu_id)，然后 ray.kill
  │     ├─ sorted(bundle_infos, key=sort_key)         ──► reordered_bundle_indices, reordered_gpu_ids
  │     └─ return {"actor": (pg, idx, gpu), "rollout": (pg, idx[offset:], gpu[offset:]), "critic": ...}
  │
  ├─ create_rollout_manager(args, pgs["rollout"])
  │     ├─ RolloutManager.options(num_cpus=1, num_gpus=0).remote(args, pg)
  │     │     └─ RolloutManager.__init__
  │     │           ├─ init_http_client(args)
  │     │           ├─ start_rollout_servers(args, pg)
  │     │           │     ├─ _resolve_sglang_config(args)
  │     │           │     ├─ for each model:
  │     │           │     │     ├─ _start_router(args, ...)      ──► multiprocessing.Process(daemon=True), sleep(3)
  │     │           │     │     ├─ for each server_group:
  │     │           │     │     │     ├─ ServerGroup(...)
  │     │           │     │     │     ├─ group.start_engines(port_cursors)
  │     │           │     │     │     │     ├─ ray.remote(SGLangEngine).options(num_gpus=0.2, bundle_index=...).remote(...)
  │     │           │     │     │     │     ├─ _allocate_rollout_engine_addr_and_ports_normal(...) — 在每张 GPU 上分配 server/nccl/dist_init/dp 端口
  │     │           │     │     │     │     └─ engine.init.remote(host, port, nccl_port, dist_init_addr, router_ip, router_port)  ← 不阻塞
  │     │           │     │     │     └─ (EPD 模式下，encoder group 必须先 ray.get 完才能把 URL 注入 prefill group)
  │     │           │     │     └─ servers[model_name] = RolloutServer(...)
  │     │           │     └─ args.sglang_model_routers = {name: (ip, port) for ...}
  │     │           ├─ ray.get(rollout_init_handles)              ← 这里阻塞，等所有 engine.init 完成
  │     │           ├─ Lock.options(...).remote()                 ← 创建 weight update 互斥锁
  │     │           └─ for each server_group: RolloutHealthMonitor(group).start()
  │     ├─ rollout_manager.get_num_rollout_per_epoch.remote()    ── 若 num_rollout 未指定则按 epoch 推
  │     └─ if args.offload_rollout: rollout_manager.offload.remote()
  │
  ├─ create_training_models(args, pgs, rollout_manager)
  │     ├─ allocate_train_group(args, ..., pg=pgs["actor"], role="actor")
  │     │     └─ RayTrainGroup(...)
  │     │           └─ _allocate_gpus_for_actor:
  │     │                 ├─ TrainRayActor = ray.remote(num_gpus=1, ...)(actor_cls or MegatronTrainRayActor)
  │     │                 └─ for rank in range(world_size):
  │     │                       TrainRayActor.options(num_cpus=0.4, num_gpus=0.4, bundle_index=reordered_bundle_indices[rank])
  │     │                                    .remote(world_size, rank, master_addr, master_port)
  │     │                       if rank == 0: master_addr, master_port = ray.get(actor.get_master_addr_and_port.remote())
  │     ├─ ray.get(actor_model.async_init(args, role="actor", with_ref=..., ...))    ── 在 actor 内 init_process_group + 加载 megatron 模型
  │     ├─ actor_model.set_rollout_manager(rollout_manager)
  │     │     └─ rank 0 在内部 ray.get(rollout_manager.set_train_parallel_config.remote(...))
  │     └─ if args.rollout_global_dataset: ray.get(rollout_manager.load.remote(start_rollout_id - 1))
  │
  ├─ if args.offload_rollout: ray.get(rollout_manager.onload_weights.remote())
  ├─ actor_model.update_weights()        ── 初始权重从 megatron 推到 sglang（让推理与训练对齐）
  └─ for rollout_id in range(...):       ── 训练循环
        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))
        if args.offload_rollout: ray.get(rollout_manager.offload.remote())
        ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
        ...
        actor_model.update_weights()     ── 每步训完都 push 权重
        if args.offload_rollout: ray.get(rollout_manager.onload_kv.remote())
        ...
```

`train_async.py` 与上述同构，唯一差别是把 `generate.remote(rollout_id+1)` 提前一步发出去，下一轮 `ray.get(rollout_data_next_future)` 拿结果——本质上还是同一组 actor，**异步性来自不 `ray.get`，而不是来自不同的 actor 拓扑**。

---

## 4. 设计模式

### 4.1 placement 决策表（5 种模式 × 2 张表）

`_get_placement_group_layout` 是一张小决策表（`placement_group.py:100`）：

| 模式 | num_gpus | rollout_offset | 含义 |
|---|---|---|---|
| `debug_train_only` | `actor_gpus` | 0 | 不要 sglang |
| `rollout_external + debug_rollout_only` | 0 | 0 | 完全不开 pg |
| `rollout_external`（其他） | `actor_gpus` | `actor_gpus` | 只为 megatron 开 pg；sglang 在外部 |
| `debug_rollout_only` | `rollout_gpus` | 0 | 不要 megatron |
| `colocate` | `max(actor_gpus, rollout_gpus)` | 0 | actor 与 rollout **复用同一组物理 GPU**，依赖 offload 切换 |
| 默认 | `actor_gpus + rollout_gpus` | `actor_gpus` | 两侧物理隔离 |

`rollout_offset` 表示：rollout 的 reordered_indices 是从 actor 的同一张表里**截取后半段**。也就是 actor 和 rollout 共享一个 PG，但通过 list slicing 划分各自的 bundle 范围。

### 4.2 actor 集合的协调模式

- "fire-and-collect": `async_init / async_train` 返回 `list[ObjectRef]`，让 caller 决定阻塞时机。在 `train_async.py` 里这条规则被进一步利用：`generate.remote(rollout_id+1)` 在训练 `rollout_id` 期间已经开跑。
- "全 actor 同步操作": `save_model / update_weights / sleep / wake_up / clear_memory` 全部 `ray.get([... for actor])`。这些是**屏障语义**——必须所有 rank 都到位才能继续。
- "rank 0 唯一动作": `set_rollout_manager` 内部、`update_weights` 内部的 `recover_updatable_engines` 都只让 rank 0 与 rollout manager 通信，再用 `dist.barrier(group=get_gloo_group())` 等其他 rank。这避免了 dp_size × engine_count 次的 RPC 风暴。

### 4.3 offload 设计

slime 有两套独立但配合工作的 offload：

- `offload_train` / `wake_up`（在 train_actor.py:101-107 是 abstract 钩子）：megatron 侧用 `torch_memory_saver` 把权重/optimizer 暂存到 CPU，要求 `LD_PRELOAD` 一个 `.abi3.so` 动态库——这个 LD_PRELOAD 在 `actor_group.py:64-83` **必须在 `ray.remote` 创建时通过 `runtime_env.env_vars` 注入**，否则进程已经起来再设就无效。
- `offload_rollout` / `onload_weights` / `onload_kv`：sglang 侧的 `release_memory_occupation(tags=[...])`，可以按 weights / kv_cache / cuda_graph **分阶段** 释放与恢复。`ServerGroup.needs_offload` 字段记录"我这个 group 的 GPU 是否与 megatron 重叠"——只有重叠的才真正 offload，避免 colocate 之外的 group 做无谓工作（`rollout.py:1140`）。

### 4.4 角色（actor / rollout / ref / critic）的复用

值得注意：slime **没有给 ref / reward 单独建 actor group**。在 `create_training_models` 里只创建了 `actor_model` 与可选的 `critic_model`，`with_ref=actor_args.kl_coef != 0 or actor_args.use_kl_loss` 表示 ref 模型**和 actor 同住在 MegatronTrainRayActor 内部**。Critic 才独立成一个 `RayTrainGroup`，因为 critic 用的是不同的 megatron 配置（通过 `parse_megatron_role_args(args, megatron_config_path, role="critic")` 解析）。

Reward model 走的是另一条路：要么作为 sglang 上的 frozen `RolloutServer`（`update_weights=False`），要么作为外部 HTTP 服务，根本不进 Ray 编排。

### 4.5 PD/EPD 解聚的两阶段启动

当 `has_encoder_disaggregation` 时，`start_rollout_servers`（`rollout.py:1171-1205`）拆成两阶段：先 `ray.get` 等所有 encoder group 起来，拿到 `encoder_urls`，把 URL **注入到 prefill/regular group 的 sglang overrides（`language_only=True, encoder_urls=[...]`）**，再启动剩余 group。这是这一层里唯一被迫同步的地方——其他所有 group 都是 fire-and-collect。

---

## 5. 集成点

| 边界 | 形式 | 文件:行 |
|---|---|---|
| train.py → ray 层 | 三个工厂函数：`create_placement_groups`, `create_rollout_manager`, `create_training_models` | `placement_group.py:120/220/152` |
| ray 层 → backends（训练） | `RayTrainGroup.__init__` 中 `actor_cls or MegatronTrainRayActor`，再 `ray.remote(...)(actor_impl)` | `actor_group.py:90-103` |
| ray 层 → backends（推理） | `ray.remote(SGLangEngine)` 直接包，没有抽象基类 | `rollout.py:168` |
| 训练 actor → rollout manager | `actor.set_rollout_manager(rollout_manager)`；rank 0 在 `update_weights` 里调 `rollout_manager.get_updatable_engines_and_lock.remote()` 拿到 `(engines, lock, num_new, gpu_counts, gpu_offsets)` | `train_actor.py:125-128`, `megatron_utils/actor.py:587-615` |
| rollout manager → 训练 actor | 不直接调用——通过 `set_train_parallel_config` 把 dp_size 等信息回传给 rollout manager，用于 `_split_train_data_by_dp` | `rollout.py:826-895` |
| ray 层 → 用户自定义 rollout/eval/data_source | `load_function(args.rollout_function_path / eval_function_path / data_source_path / custom_*)` 在 RolloutManager init 时一次性 import | `rollout.py:438-451` |

---

## 6. 出人意料的决策

### 6.1 RolloutManager 为什么是 1485 行的"重 actor"

直觉上，"manager" 应该只编排（拉起 engine、收集结果、停掉）。但 `RolloutManager` 还承担：
- **数据转换**（`_convert_samples_to_train_data` 把 `list[Sample]` 转成训练数据 dict，约 110 行）
- **DP 切分**（`_split_train_data_by_dp` 按训练侧的 dp_size 把数据切成 dp_size 份，并打包成 ray Box 或 nixl，约 70 行）
- **奖励后处理**（`_post_process_rewards` 做 group norm / GRPO 归一化）
- **指标计算**（compute_*_metrics / log_eval_rollout_data，约 250 行）
- **健康监控**（启停 `RolloutHealthMonitor`、CI fault injection）

为什么这些不放到训练 actor 内？因为 **rollout manager 是 0 GPU 的 CPU actor**，做这些 CPU-bound 的转换既不会和训练抢 GPU，也避免了让 dp_size 个训练 actor 各自重复一遍同样的转换。它本质上是"训练循环的 sidecar"——把所有"需要看到所有 rollout 结果"的工作都收拢到这里完成。也正因如此，它持有 router 进程、Lock、HealthMonitor 等所有跨 actor 的共享状态。**book 里这一章应该单独画一张图说明"为什么 rollout manager 不是无状态 manager 而是状态收拢点"。**

### 6.2 placement strategy 选 PACK，不是 STRICT_SPREAD

`placement_group.py:48` 写死了 `strategy="PACK"`——尽可能把 bundle 塞到同一个节点。这是 RL 场景的合理选择（actor 与 rollout 需要 NCCL/RDMA 高带宽通信，跨节点会成为瓶颈），但**对 PG 调度的灵活性是有代价**的——它意味着如果集群里某个节点 8 GPU 里只有 7 个空闲，整个 PG 都会等。代码里用 `while not ray.wait([ready_ref], timeout=30)[0]` 做 30s 一次的等待日志（`placement_group.py:60-67`），故意不设超时，让 autoscaler 触发扩容时不会卡死——这是一个为大规模训练集群专门写的细节。

### 6.3 sglang engine 用 `num_gpus=0.2`，不是 1

`rollout.py:176` 写 `num_gpus = 0.2`。这不是占用 1/5 显存的意思，而是 Ray 的 fractional GPU 语义——它告诉 ray "我在逻辑上占 0.2 张 GPU"，让 ray 不会阻止同一 bundle 上再调度别的 actor。配合 `placement_group_capture_child_tasks=True` 与 `NOSET_*_VISIBLE_DEVICES=1`，sglang 子进程实际上**绕开 ray 的 visible-device 注入**自己去看物理 GPU。这是 colocate 场景能工作的关键——megatron actor 也占 0.4，两者加起来 < 1，Ray 允许它们落在同一 bundle 上。

### 6.4 RolloutManager 是 ray actor，但 router 是 multiprocessing.Process

`_start_router`（`rollout.py:1060-1068`）用 `multiprocessing.Process(daemon=True)` 而不是另一个 ray actor 启动 sglang router。原因猜测：router 进程需要绑定一个稳定可路由的 IP/port，且它要被很多外部 actor（甚至外部 client）连接——把它做成 daemon 进程比做成 ray actor 更容易在容器/K8s 网络里暴露端口；而且 router 死了我们希望整个 rollout manager 重启而不是 ray 重调度它。`time.sleep(3)` + `assert process.is_alive()` 是一个**故意非精致**的就绪检查。

### 6.5 `Lock` actor 而不是 `asyncio.Lock`

`utils.py:46-64` 实现了一个 ray actor 形式的 Lock，只有 `acquire / release`，且 `acquire` 是**非阻塞的轮询语义**："caller should retry until it returns True"。为什么不用 `asyncio.Lock` 或 `ray.util.Queue`？因为这个 Lock 要**跨 actor、跨节点**协调——dp_size 个 megatron actor 各自想 push 权重时，需要一个所有节点都能看到的全局互斥。让它是 actor 而非 asyncio 对象，就能用 `ray.get(lock.acquire.remote())` 直接调用。非阻塞 + 轮询是为了防止某个 caller 阻塞 actor 的 event loop（ray actor 默认单线程处理）。

### 6.6 `test_placement_group.py` 守护的不变量

`tests/test_placement_group.py:30-43` 用 10 种配置组合断言 `_get_placement_group_layout` 的返回值。值得 book 单独列表的不变量是：
- `colocate + rollout_num_gpus > actor_gpus`：返回 `(rollout_num_gpus, 0)`——意味着 colocate 模式下，**总 GPU 数由二者较大者决定**（actor 必须能"挤进"rollout 的 GPU 集合，反之亦然）。
- `rollout_external`：始终 `(actor_gpus, actor_gpus)`，**rollout_offset 等于 actor_gpus 但总 GPU 也是 actor_gpus**——这是一个看似矛盾的设置，意思是"rollout 的 indices 是 actor.indices[actor_gpus:] = []"。外部 rollout 不参与 PG。
- `debug_train_only`：rollout_offset=0，**rollout 与 actor 共享同一段 indices**（用于纯训练 debug 时让 rollout-side 调用立刻 no-op）。
这些行为很反直觉，**修改 `_get_placement_group_layout` 时不跑这套测试会直接打乱 GPU 拓扑**。

### 6.7 `ray.remote(SGLangEngine)` 不需要继承 `RayActor`

`rollout.py:168` 直接对 `SGLangEngine`（在 `slime/backends/sglang_utils/sglang_engine.py`）用 `ray.remote(...)`。看 `_get_current_node_ip_and_free_port` 的调用方（`rollout.py:972, 982`），这两个调用其实需要 `RayActor` 基类提供——所以 `SGLangEngine` 必须自己提供这两个方法（或继承 `RayActor`）。这是一个隐式契约：**任何想被 ray 层调度的 backend engine，都必须实现 `RayActor` 那个 10 行的接口**，但代码里没有用 abstract 来强制。

---

## 附录

### 已通读文件清单（带行数）
- `slime/ray/__init__.py` (0 行，空文件)
- `slime/ray/ray_actor.py` (10 行)
- `slime/ray/utils.py` (64 行)
- `slime/ray/rollout_validation.py` (32 行)
- `slime/ray/placement_group.py` (247 行)
- `slime/ray/actor_group.py` (169 行)
- `slime/ray/train_actor.py` (128 行)
- `slime/ray/rollout.py` (1485 行，重点看 1-247 的 ServerGroup/RolloutServer、421-633 的 RolloutManager、930-1228 的 start_rollout_servers/_allocate_ports/_start_router，跳过 686-897 的奖励/数据转换业务逻辑、1258-1485 的 metrics 计算)
- `tests/test_placement_group.py` (54 行)

### 扫读文件清单（边界确认）
- `train.py` (104 行，调用入口)
- `train_async.py` (81 行，异步训练入口)
- `slime/backends/megatron_utils/actor.py` 仅看 `update_weights` (580-628) 与 `sleep/wake_up` 签名 (189-208)

### 未读但相关的文件（归其他子系统）
- `slime/backends/sglang_utils/sglang_engine.py` —— SGLangEngine 实现 (backends 子系统)
- `slime/backends/megatron_utils/actor.py` 完整 (training 子系统)
- `slime/rollout/*` —— generate_rollout / Sample 类型 (rollout 子系统)
- `slime/utils/health_monitor.py` —— RolloutHealthMonitor (fault tolerance / 工程基础设施)
- `slime/utils/dp_schedule.py` —— build_dp_schedule (training 子系统)
- `slime/backends/sglang_utils/external.py` —— start_external_rollout_servers (deployment 子系统)
