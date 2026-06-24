# 部署拓扑 / SGLang config / PD 解耦 / 外部引擎 — 研究笔记

> 子系统范围：`slime/backends/sglang_utils/sglang_config.py`、`external.py`、`server_control.py`、`slime/utils/routing_replay.py`，以及 `_resolve_sglang_config` / `_start_router` / `start_rollout_servers` 这条编排链路。
> 边界：`sglang_engine.py` 本身归 backends；weight sync 的 disk 路径细节归 weight sync；单节点 GPU placement 归 Ray 编排。

## TL;DR

slime 的部署能力本质上是"对 SGLang 上游的薄编排层"——它不重新实现 router、不重新实现 PD、不重新实现 disk 加载，只做两件事：(1) 用一个 YAML schema (`--sglang-config`) 把上游 `ServerArgs` 提到**按 server group 维度可覆盖**的层级；(2) 把 router、引擎生命周期、weight sync 这三个独立轴解耦成可选积木——`--sglang-config` 让 slime 拥有引擎生命周期，`--rollout-external-engine-addrs` 让外部系统拥有，二者互斥但都接同一套 sgl-router 与 `update_weights_from_disk`。RoutingReplay (R3) 则是部署拓扑给训练循环开的一个"反向通道"：让 rollout 时被路由器决定的 expert 选择，回到训练时的 MoE 前向中重放。

## 1. 架构与模块边界

| 模块 | 角色 | 关键入口 |
| :--- | :--- | :--- |
| `sglang_config.py` | YAML schema 与 dataclass：`SglangConfig` → `ModelConfig` → `ServerGroupConfig`。负责解析、回退默认值、校验同模型 `model_path` 一致、自动推断 `update_weights`。 | `SglangConfig.from_yaml`、`ModelConfig.resolve` |
| `external.py` | 发现并接入外部预启动的 SGLang server：HTTP 拉 `/server_info`，推断 `worker_type`、`num_gpus`、`disaggregation_bootstrap_port`，构造一个轻量 `ExternalRolloutServer`（offload/onload/recover 全是 no-op）。 | `discover_external_engines`、`apply_external_engine_info_to_args`、`start_external_rollout_servers` |
| `server_control.py` | SGLang server 生命周期里**只属于 slime 的部分**：abort-until-idle 与负载读数（`/v1/loads?include=core`），用来在 weight sync / 切模型时让 server 干净进入空闲。 | `abort_servers_until_idle`、`num_requests_from_load` |
| `routing_replay.py` | 训练时重放 rollout 期路由决策的 hook：进程内单例 `ROUTING_REPLAY`，在 `compute_topk` 处拦截，按 `record/replay_forward/replay_backward/fallthrough` 切换。 | `RoutingReplay`、`get_routing_replay_compute_topk`、`register_routing_replay` |
| `slime/ray/rollout.py` 中的 `_resolve_sglang_config` / `_start_router` / `start_rollout_servers` | 整套部署的实际编排：选择配置来源、为每个 model 拉一个 router、按 `server_groups` 物化 `ServerGroup`、注入 EPD 的 `encoder_urls`、把每个模型的 router 写回 `args.sglang_model_routers`。 | `slime/ray/rollout.py:1019`、`:1089`、`:1231` |

这五个文件加在一起不到 600 行——剩下的活全在 SGLang 上游：sgl-router、`ServerArgs`、`update_weights_from_disk`、disaggregation runtime、mooncake transfer backend。slime 没有自己的 router 策略实现，只是把 `RouterArgs` 透传 (`add_sglang_router_arguments` 直接 `RouterArgs.add_cli_args(parser, use_router_prefix=True, exclude_host_port=True)`，见 `slime/backends/sglang_utils/arguments.py:31`)。

## 2. 关键抽象

### 2.1 YAML 三层结构

`sglang_config.py:115` 定义的 `SglangConfig` 是顶层容器，往下是 `ModelConfig`（`:44`）、`ServerGroupConfig`（`:11`）。三层各自承担不同的回退职责：

- **组级**：`worker_type` ∈ {regular, prefill, decode, placeholder, encoder}（`sglang_config.py:37`），`num_gpus` 强制 `>0`（`:41`），`num_gpus_per_engine` 可选，`overrides` 是任意 SGLang `ServerArgs` 字段名 → 值的 dict。
- **模型级**：`name`、`model_path`、`update_weights`、模型默认的 `num_gpus_per_engine`。`resolve(args)`（`:68`）把模型默认值下推到没设置的组里，把 `model_path` 注入到每组的 `overrides`（让下游 `_compute_server_args` 单点读取），并校验**同模型所有组共享同一个 `model_path`**（`:80-85`）。
- **全局级**：在 `_resolve_sglang_config`（`slime/ray/rollout.py:1231`）中校验总 GPU 数等于 `--rollout-num-gpus`。

`update_weights` 的推断逻辑很有意思（`sglang_config.py:91-100`）：如果有效 `model_path` 与 `args.hf_checkpoint` 一致就默认 `True`，否则默认 `False` 并打 warning。这等于说 reference / reward 模型只要写不同的路径就自动变成冻结。

### 2.2 worker_type 五元枚举

regular / prefill / decode / placeholder / encoder。其中：

- **placeholder**：占 GPU 槽位但不启 engine（`slime/ray/rollout.py:1155` 处用 `all_engines=[]` 实现）。文档里给的场景是 colocate 训练需要为训练预留 GPU。
- **encoder**：用于 EPD（Encoder-Prefill-Decode）解耦——`start_rollout_servers` 有一个**两阶段启动**（`:1171-1205`）：先把 encoder 组完全启起来、阻塞拿 `encoder_urls`，再把这些 URL 通过 `overrides_extra={"language_only": True, "encoder_urls": ...}` 注入到 prefill / regular 组的 `ServerArgs` 里。这是整个部署链路里唯一非异步的同步点。
- **prefill / decode 互斥校验**：FAQ 与代码都强调，同一个 model 不能混用 regular 和 prefill/decode（`pd-disaggregation.md:80`、`sglang-config.md:439`）。

### 2.3 三种"覆盖"的优先级

`_compute_server_args`（`slime/backends/sglang_utils/sglang_engine.py:546-648`）把 server args 按以下优先级合并：

1. 函数内显式 base kwargs（host/port/nnodes/base_gpu_id、`enable_memory_saver`、按 `worker_type` 派生的 `disaggregation_mode`/`load_balance_method`/`disaggregation_bootstrap_port` 等）；
2. 从 `args.sglang_*` 自动同步到所有 `ServerArgs` 字段（`:618`）——这是 `--sglang-*` CLI 的统一入口；
3. `sglang_overrides`（来自 YAML `overrides`）以最高优先级覆盖（`:624-640`），并把 `-` 自动转 `_`。

未被 SGLang 当前版本识别的 key 会被丢弃并打 info 日志（`:643`），这是一种"上游升级时配置不破"的兼容策略。

### 2.4 ExternalEngineInfo & ExternalRolloutServer

`external.py:14` 的 `ExternalEngineInfo` 是 frozen dataclass，承载从 `/server_info` 推断出的拓扑：worker_type 由 `_infer_worker_type`（`:70`）从 `disaggregation_mode` / `encoder_only` 字段映射，`num_gpus` 优先用 server_info 自报值，否则退化到 `tp_size * pp_size`（`:89`）。

`external.py:135` 的 `ExternalRolloutServer` 是一个**故意做扁的** RolloutServer：`offload/onload/onload_weights/onload_kv` 全部 `return []`，`recover` 只打 warning（`:152-165`）。它存在的唯一意义是给 `RolloutManager` 提供一个和 `RolloutServer` 同形的对象，避免在 manager 那一层 if/else。

### 2.5 RoutingReplay 的注入点

`routing_replay.py:13` 的 `RoutingReplay` 用类变量 `all_routing_replays` 做注册表，按 MoE 层创建。注入点是 SGLang/Megatron 的 `compute_topk` 函数：`get_routing_replay_compute_topk`（`:57`）返回一个包装版，根据环境变量 `ROUTING_REPLAY_STAGE` 切换四种行为：

- `fallthrough`：原样调用（用于 ref/teacher 前向，需要它们自己重新算 topk）。
- `record`：算 topk 同时把 `top_indices` offload 到 pinned CPU。
- `replay_forward` / `replay_backward`：从 list 里弹出之前记录的 `top_indices`，用 `scores.gather` 重算 probs。

环境变量切换由 `slime/backends/megatron_utils/actor.py:434-537` 在训练循环里精确调度：ref/teacher 设 `fallthrough`，本地 actor 前向设 `record` 或 `replay_forward`，反向设 `replay_backward`，每个 rollout 结束 `clear_all()`。

`use_rollout_routing_replay`（`slime/utils/arguments.py:1064`）走的是 SGLang 侧 `enable_return_routed_experts=True`（`slime/backends/sglang_utils/sglang_engine.py:606`），把 rollout 期 SGLang engine 实际选的 expert 通过 `rollout_routed_experts` 字段回传给 training，由 `fill_routing_replay`（`actor.py:282`）按 layer 灌进 `RoutingReplay`。这是部署拓扑与训练循环之间**真正的反向数据通道**。

## 3. 数据流

```
CLI/启动参数 (--sglang-config 或 --rollout-external-engine-addrs 或 --prefill-num-servers 或 无)
        │
        ▼
slime/utils/arguments.py:1881      args.rollout_external = ...
            │ rollout_external?
            ├─是→ apply_external_engine_info_to_args(args)
            │       └─ HTTP /server_info → ExternalEngineInfo[] → args.rollout_external_engine_infos
            │
            ▼
RolloutManager.__init__ → start_rollout_servers(args, pg)   # slime/ray/rollout.py:1089
        │
        ├─ rollout_external → start_external_rollout_servers
        │         └─ _start_router(has_pd=any prefill/decode)
        │            for info in infos: RolloutRayActor.remote(...).init.remote(**external_engine_init_kwargs(info))
        │
        └─ 否 → _resolve_sglang_config(args)                  # :1231
                 │  ├─ --sglang-config 路径 → SglangConfig.from_yaml + 总 GPU 校验
                 │  ├─ --prefill-num-servers 路径 → from_prefill_num_servers
                 │  └─ 默认 → 单 regular 组
                 │
                 ▼
                 for model_cfg in config.models:
                     model_cfg.resolve(args)
                     router_ip, router_port = _start_router(has_pd_disaggregation=model.has_pd,
                                                            force_new=(model_idx>0))   # 每个 model 独立 router
                     if has_encoder_disaggregation:
                         Phase 1: 启 encoder 组 → ray.get → 拿 URL
                         Phase 2: 启其余组（注入 encoder_urls）
                     else:
                         一次性启所有组
                     servers[name] = RolloutServer(server_groups=...)
                 args.sglang_model_routers = {name: (ip, port)}    # :1226

        ▼
请求路由：
  rollout 函数通过 get_model_url(args, "actor", "/generate") 拿到 router URL，
  附带 X-SMG-Routing-Key: <session_id>（当 router_policy=consistent_hashing 时）
  (slime/rollout/sglang_rollout.py:195-199, slime/utils/types.py:148)

        ▼
路由回传（仅 MoE + R3 启用时）：
  SGLang engine 因 enable_return_routed_experts=True，把每层选中的 expert
  附在 rollout 响应里 → rollout_data["rollout_routed_experts"]
        ▼
  actor.fill_routing_replay 把它按层灌进 RoutingReplay
        ▼
  training 前向时 ROUTING_REPLAY_STAGE=replay_forward → MoE topk 直接弹历史
```

## 4. 设计模式

### 4.1 YAML override 的"加层"模式
不替换上游 `ServerArgs`，而是在它之上加一层组级 override。`_compute_server_args` 始终先用 `--sglang-*` 全局值填充所有 ServerArgs 字段，再让 YAML 的 `overrides` 覆盖。结果是：

- 90% 的常规字段仍走 CLI 一处配；
- 仅当 prefill / decode 需要差异（chunked_prefill_size、deepep_mode、mem_fraction_static）时才在 YAML 里写覆盖；
- 上游字段更名 / 新增不会让旧 YAML 失效（unknown key 仅 log）。

### 4.2 每模型独立 router
`_start_router` 的 `force_new=(model_idx>0)`（`slime/ray/rollout.py:1121`）保证多模型场景下每个 model 都拿到一个新的 router 端口。这把"模型隔离"下沉到了路由层而不是 server 内部，意味着 actor/ref/reward 在路由层就分流，连负载均衡都是独立的。`args.sglang_model_routers` 是这个设计对外暴露的唯一 API。

### 4.3 External engine 的"训练任务外部管理"
`ExternalRolloutServer` 的所有生命周期方法都是 no-op，等于声明："这个 server slime 不管"。fault tolerance、offload/onload、recover 都被禁用——文档显式说明这是和 colocate / fault tolerance 不兼容的代价（`external-rollout-engines.md:111`）。它换来的是：rollout serving 可以使用与训练完全不同的 Python 环境、不同的容器、甚至不同型号或不同厂家的 GPU（前提是 disk transport 而非 NCCL）。

### 4.4 RoutingReplay 的"环境变量调度状态机"
没有把 stage 做成函数参数，而是用 `os.environ["ROUTING_REPLAY_STAGE"]` 作为全局信号——因为 `compute_topk` 是 SGLang/Megatron 深处的算子，重写它的所有调用路径不现实。环境变量是这个 hook 唯一能在不改上游签名的前提下被四种语义切换的方式。代价是它必须由 actor 训练循环手工"翻牌"。

### 4.5 兼容旧字段
`SglangConfig.from_yaml`（`:169`）同时接受 `server_groups` 和遗留的 `engine_groups`；`SglangConfig.from_prefill_num_servers`（`:183`）把老的 `--prefill-num-servers` 适配成 YAML 等价结构。两条路径都互斥，但通过把所有路径汇聚到同一个 dataclass，下游编排只看 `SglangConfig`。

## 5. 集成点

- **与 sgl-router**：`_start_router`（`slime/ray/rollout.py:1019`）以子进程方式启 `sglang_router.launch_router.run_router`；`RouterArgs.from_cli_args(args, use_router_prefix=True)` 透传所有 `--sglang-router-*` CLI。PD 场景额外设 `pd_disaggregation=True` 和 `disable_circuit_breaker=True`（注释解释：RDMA transfer timeout 是 PCIe 压力下的瞬时事件，不该把 decode worker 标死）。
- **与 backends（engine 启动）**：`ServerGroup.start_engines` → `SGLangEngine.init`（`slime/backends/sglang_utils/sglang_engine.py`）。external 路径走 `RolloutRayActor.options(num_cpus=0.2, num_gpus=0)`（`external.py:196-206`）——`num_gpus=0` 是关键，因为 GPU 已由外部系统占用。
- **与 weight sync**：`update_weights: false` 的模型从此完全跳过 weight sync（reference/reward）。disk transport 路径下，跨型号 GPU 部署成为可能；`--update-weight-disk-dir` 路径要求训练端与 SGLang server 都能看到。
- **与 training（R3）**：通过 `args.use_rollout_routing_replay` → SGLang `enable_return_routed_experts=True` → rollout 响应回带 `rollout_routed_experts` → `actor.fill_routing_replay` → `RoutingReplay.record`。Megatron 前向通过 `register_routing_replay` 在每个 MoE 层注册 hook，前向时 `compute_topk` 被替换。
- **与 server_control**：weight sync 前/切模型前调用 `abort_servers_until_idle`，循环 POST `/abort_request?abort_all=true` 并轮询 `/v1/loads`，直到 server 进入空闲。

## 6. 出人意料的决策

### 6.1 SGLang config 选 YAML 而不是 CLI
表面理由是"参数多"，但更深的原因是 **per-group override**——CLI 不能自然表达"prefill 用 `chunked_prefill_size=8192` 而 decode 用 `mem_fraction_static=0.88`"这种结构化覆盖。这也是为什么 YAML 的 `overrides` 直接以 `ServerArgs` 字段名作为 key 而非新发明一套命名——它声明性地说"按这个组应用这些字段"，逻辑就压缩成了 dict 合并。代价是 YAML 总 GPU 数必须手工等于 `--rollout-num-gpus`（`slime/ray/rollout.py:1238`），slime 不会自动推断。

### 6.2 PD 解耦的真正收益是 multi-turn 而不是 throughput
文档（`pd-disaggregation.md:64-73`）强调的 RL 场景特征——长 prompt（来自 tool history）、多轮交互、decode latency long tail、session-local prefix cache、不同模型不同资源——揭示了 PD 在 slime 的核心动机不是单 token 吞吐，而是**让 rollout topology 贴近真实 serving workload**。GLM-5.2 744B 的脚本（1 个 prefill engine + 3 个 decode engine，prefill 用 `deepep_mode=auto`、decode 用 `low_latency + deep_gemm`）是这个动机的极端体现。

更深一层：`consistent_hashing` 路由策略 + UUID `session_id` + `X-SMG-Routing-Key` header（`sglang_rollout.py:195`、`types.py:148`），让同一个 sample 的所有轮次去同一个 decode worker。在 agent 场景里这意味着 prefix cache 跨轮复用——这是 PD 真正给 RL 训练带来的算力节省。

### 6.3 External engine 跨厂家 GPU 不是 future work，是首要设计目标
`external-rollout-engines.md:51-56` 把它写得非常明确："训练可以在一组 GPU 上运行，rollout serving 可以放在另一组不同型号或不同厂家的 GPU 上"。这只在 disk transport 下成立，对应的内部代码是：`ExternalRolloutServer.offload/onload` 全部 no-op，意味着 slime 完全放弃了 colocate 假设。文档还引用 Cursor Composer 2 的 S3 + delta compression 作为"同类基础设施问题"，说明这条路径是面向跨数据中心训推解耦的生产形态，不是临时方案。

### 6.4 heterogeneous server group 的存在理由：EPD 与 placeholder
不是为了"不同 TP 用不同型号 GPU"——所有组仍跑在同一 placement group 里。它存在的真正理由是：
- **EPD**：encoder 必须先启好，让别的组拿到 URL（`encoder_urls` 注入），所以 `start_rollout_servers` 必须能区分 worker_type 并做两阶段编排（`slime/ray/rollout.py:1171-1205`）。
- **placeholder**：在 colocate 模式下，要在 rollout 端"宣告"某些 GPU 不会被 rollout 占——这些槽位会留给 training。如果没有 placeholder，rollout 会按 `rollout_num_gpus` 把整段 PG 全用掉。

### 6.5 RoutingReplay 用全局变量与环境变量而非 monkey patch 参数
`set_routing_replay(replay)` 设 module 全局变量，`ROUTING_REPLAY_STAGE` 用 `os.environ`——这套设计被四个不同位置使用（`compute_topk` 包装、`register_routing_replay` 的 `pre_forward_hook`、actor 训练循环的 stage 切换、Megatron `combined_1f1b` 前向的临时 stage 覆盖 `model.py:602-636`）。这种"全局信号"的选择对应一个事实：**MoE topk 调用点深埋在 SGLang/Megatron 内部，slime 无法通过显式参数传递控制它**。环境变量是唯一既不改上游签名又能精确切换语义的通道。

## 附录

### 已通读
- `slime/backends/sglang_utils/sglang_config.py`（208 行）
- `slime/backends/sglang_utils/external.py`（233 行）
- `slime/backends/sglang_utils/server_control.py`（68 行）
- `slime/utils/routing_replay.py`（93 行）
- `slime/backends/sglang_utils/arguments.py`（200 行，主要看 `add_sglang_arguments` 与互斥校验）
- `docs/zh/advanced/sglang-config.md` / `pd-disaggregation.md` / `external-rollout-engines.md`
- `docs/zh/examples/glm5.2-744B-A40B.md`
- `scripts/run-glm5.2-744B-A40B.sh`（SGLang config 段、SGLANG_ARGS）

### 扫读
- `slime/ray/rollout.py` 的 `_start_router` / `start_rollout_servers` / `_resolve_sglang_config` / `RolloutServer` dataclass
- `slime/backends/sglang_utils/sglang_engine.py:546-648` (`_compute_server_args`)
- `slime/backends/megatron_utils/actor.py:282-353, 428-537`（routing replay 集成）
- `slime/backends/megatron_utils/model.py:595-637`（前向时的 stage 临时覆盖）
- `tests/test_external_sglang_engines.py`（契约测试）
- `slime/rollout/sglang_rollout.py:195-312`（session_id 与 `X-SMG-Routing-Key` 注入）

### 未读 / 留给其他子系统
- `tests/test_qwen3_4B_external_pd.py` 完整端到端 PD 流程（与 ray 编排重叠，归 ray 子系统）
- `tests/test_glm4.7_30B_A3B_pd_mooncake.py`（同上，且涉及 mooncake backend 细节）
- `scripts/run-glm4.7-355B-A32B.sh` / `run-deepseek-r1.sh`（部署变体，可在第 8 章正文按需引用）
- `ServerGroup` / `RolloutServer` 完整生命周期（offload/onload/recover）→ 归 weight sync + ray 编排
- delta-weight-sync.md 详细 wire format → 归 weight sync 子系统
