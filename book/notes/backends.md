# 研究笔记：backends 适配层

> 子系统范围：`slime/backends/megatron_utils/`、`slime/backends/sglang_utils/`、
> `slime_plugins/mbridge/`、`slime_plugins/megatron_bridge/`、`slime_plugins/models/`。
> 对应书中第 3 章"运行时基础——Ray placement 与引擎初始化"的下半段。

## TL;DR

slime 的 backends 层只做一件事：**让 Megatron 和 SGLang 各自的"原生 CLI + 原生入口函数"在 Ray actor 进程里能直接跑起来**，
而不在两者之上叠抽象。Megatron 侧用 `megatron_parse_args` 透传上游 argparse、用
`init()` 重排 `mpu.initialize_model_parallel` 的并行轴顺序、用 `megatron_patch/`
对上游做最小化 monkey patch（目前只有一处：把 TP 梯度 all-reduce 拆 chunk，
避免大模型 OOM）；SGLang 侧用一个 argparse 包装器把
`ServerArgs.add_cli_args` 注入的所有参数加上 `--sglang-` 前缀重新挂到 slime 的
parser 上（`sglang_utils/arguments.py:65-115`），然后用 `_compute_server_args`
反向把它们解包成 `ServerArgs(**kwargs)` 喂回 SGLang 的 HTTP 服务进程。模型适配
分两套：`mbridge`（slime 自有，9 个具体模型 bridge）走 Megatron-LM 原生
checkpoint 格式；`megatron_bridge`（megatron-bridge 上游的 `AutoBridge`，
slime 只补 GLM-4.6V 一例）走 HF checkpoint 直接挂到 Megatron GPTModel。这层
"两套 bridge 并存"是全子系统最值得讲的设计。

---

## 1. 架构与模块边界

backends 适配层被切成三块互不引用的子区域，全部通过 actor 进程里的"参数 namespace"耦合：

```
┌─────────────────────────────────────────────────────────────┐
│ slime/backends/megatron_utils/                              │
│   __init__.py       deep_ep + Qwen3VL rotary 热补丁          │
│   initialize.py     init(args) → mpu.initialize_model_parallel │
│   arguments.py      megatron_parse_args / validate / hf 一致性 │
│   model_provider.py 三种模型构建路径（custom / bridge / raw）   │
│   model.py          setup_model_and_optimizer + train/forward │
│   megatron_patch/   对上游 Megatron 的最小化 monkey patch       │
│   server/           Megatron teacher logp HTTP server         │
│   sglang.py         单文件导入 SGLang 中给 Megatron 用的符号     │
│   {stateless_adam,misc_utils,ci_utils}.py  辅助工具            │
├─────────────────────────────────────────────────────────────┤
│ slime/backends/sglang_utils/                                │
│   arguments.py      add_sglang_arguments：参数前缀化注入        │
│   sglang_engine.py  SGLangEngine RayActor：launch_server_process │
│   (sglang_config.py、external.py、server_control.py 归 ch.8)    │
├─────────────────────────────────────────────────────────────┤
│ slime_plugins/                                              │
│   mbridge/          slime 自有 bridge，注册到 mbridge.core      │
│   megatron_bridge/  megatron-bridge AutoBridge 的扩展点         │
│   models/           HF attention/kernel 适配                  │
└─────────────────────────────────────────────────────────────┘
```

三块的耦合点：

- **`megatron_utils` ↔ `sglang_utils`**：物理上只有 `megatron_utils/sglang.py`
  这一个文件（44 行），把 SGLang 的 `FlattenedTensorBucket`、`MultiprocessingSerializer`、
  `DeltaSpec` 等少数符号集中 import + 兜异常版本兼容。weight sync 路径全部
  reference 这个文件，不直接 import sglang。
- **`megatron_utils` ↔ `slime_plugins`**：在 model_provider 和 checkpoint 两处
  通过 `import slime_plugins.megatron_bridge` 做"导入即注册"。`mbridge` 的导入
  发生在 `model_provider.py:90` 内部分支（`megatron_to_hf_mode == "bridge"` 时不会用），
  而是统一由 `slime_plugins/mbridge/__init__.py` 自动注册到 `mbridge.core.register_model`。
- **`sglang_utils` ↔ `slime_plugins`**：无。SGLang engine 进程里加载的是 HF 模型，
  不经过 slime 的 mbridge。

---

## 2. 关键抽象

### Megatron 侧

| 抽象 | 位置 | 作用 |
| --- | --- | --- |
| `megatron_parse_args(extra_args_provider, skip_hf_validate)` | `slime/backends/megatron_utils/arguments.py:184` | 调上游 `_megatron_parse_args(ignore_unknown_args=True)`，让 `--sglang-*` 默默通过；再做 `_hf_validate_args`（10 条 HF↔Megatron 配置等价检查） |
| `validate_args(args)` | `arguments.py:72` | 强制 `args.variable_seq_lengths = True`、把不支持 varlen 的 `moe_token_dispatcher_type="allgather"` 改成 `alltoall` |
| `set_default_megatron_args(args)` | `arguments.py:147` | 默认开 `use_distributed_optimizer`、设 `bf16=not fp16`、写 checkpoint I/O 默认值 |
| `init(args)` | `slime/backends/megatron_utils/initialize.py:56` | 顺序：`set_args` → `_initialize_distributed`（自己重排 `tp-cp-ep-dp-pp` 顺序）→ `_set_random_seed` → `_build_tokenizer` → `init_num_microbatches_calculator` → 可选 `_initialize_tp_communicators` → 末尾调 `custom_megatron_init_path` |
| `get_model_provider_func(args, role)` | `slime/backends/megatron_utils/model_provider.py:268` | 三选一：custom_model_provider_path → megatron-bridge AutoBridge → 内置 GPTModel；critic 额外 wrap `LinearForLastLayer` |
| `setup_model_and_optimizer(args, role)` | `slime/backends/megatron_utils/model.py:270` | `get_model` + `get_megatron_optimizer`；可选 `_patch_megatron_adam(StatelessAdam)` 上下文 |
| `initialize_model_and_optimizer(args, role)` | `model.py:968` | actor.py 的真正入口：构建模型 → load checkpoint → 返回 (model, optimizer, scheduler, iteration) |
| `megatron_patch.megatron_chunked_grad_coalesce_patch` | `slime/backends/megatron_utils/megatron_patch/megatron_chunked_grad_coalesce_patch.py:67` | 替换 `_allreduce_non_tensor_model_parallel_grads`，把 TP grad sync 切成 1 GiB chunk |

### SGLang 侧

| 抽象 | 位置 | 作用 |
| --- | --- | --- |
| `add_sglang_arguments(parser)` | `slime/backends/sglang_utils/arguments.py:35` | 拦截 `parser.add_argument`，把所有 SGLang ServerArgs 选项前缀成 `--sglang-xxx`，dest 改为 `sglang_xxx` |
| `sglang_parse_args()` | `arguments.py:176` | 独立 ArgumentParser + `parse_known_args()` → 只消费 sglang 相关 CLI |
| `SGLangEngine` (RayActor) | `slime/backends/sglang_utils/sglang_engine.py:101` | 单 engine 的 Ray actor；`init()` 内调 `launch_server_process` 起 HTTP server，再 `_register_to_router` 注册到 sgl-router |
| `_compute_server_args(args, rank, ...)` | `sglang_engine.py:546` | 把 `args.sglang_*` 反解回 `ServerArgs.fields` 字典，叠加 `--sglang-config` YAML 里的 per-group overrides |
| `launch_server_process(server_args)` | `sglang_engine.py:51` | `multiprocessing.Process(target=sglang.launch_server)` + `_wait_server_healthy` 阻塞拉起 |

### plugin 注册器

| 抽象 | 位置 | 作用 |
| --- | --- | --- |
| `mbridge.core.register_model(name)` 装饰器 | 用法在 `slime_plugins/mbridge/qwen3_next.py:8`、`glm4moe.py:7`、`deepseek_v32.py:7` | 把 `Qwen3NextBridge` 等子类注册到 mbridge 全局表；slime 内部按 HF arch 名字反查 |
| `slime_plugins/mbridge/__init__.py:1-9` | 入口 | 一次性 import 9 个具体 bridge → 触发 register_model 副作用 |
| `slime_plugins/megatron_bridge/__init__.py:1` | 入口 | 只导入 `glm4v_moe`，注册到 megatron-bridge 的 `AutoBridge` |
| `slime_plugins/megatron_bridge/glm4v_moe.py:1-32` | bridge 主体 | 通过 `MegatronModelBridge` + `MegatronMappingRegistry` 描述 HF 模块到 Megatron module 的映射 |

---

## 3. 数据流：启动序列

完整启动序列（一次 `python train.py ...`）。重点是 slime 把它折成三个阶段：

```
Phase A：CLI 解析（trainer 进程，未拉 Ray）
  parse_args()
    │  slime/utils/arguments.py:1525
    ├─ _pre_parse_mode()         读 --train-backend / --debug-*-only
    ├─ sglang_parse_args()       独立 parser、消费 --sglang-*
    │   └─ add_sglang_arguments()  ServerArgs.add_cli_args 注入并前缀化
    ├─ megatron_parse_args(extra=add_slime_arguments)
    │   └─ _megatron_parse_args(ignore_unknown_args=True)
    │   └─ _hf_validate_args     hidden_size、rope_theta 等 10 条 HF↔mcore 一致性
    │   └─ _set_default_megatron_args  use_distributed_optimizer / bf16
    ├─ 把 sglang_ns 的 attr 全部 setattr 进主 args
    └─ slime_validate_args + megatron_validate_args + sglang_validate_args

Phase B：Ray 拉起（trainer 进程仍是主导）
  create_placement_groups()    → 按 actor/rollout 角色切 PG
  RolloutGroup.create_rollout_engines()    slime/ray/rollout.py:168
    └─ RayActor = ray.remote(SGLangEngine)
    └─ for i: engine = RayActor.options(num_gpus=0.2, ...).remote(args, rank=i, ...)
    └─ engine.init.remote(dist_init_addr, port, nccl_port, ...)  ← Phase C-SGLang
  create_training_models()
    └─ for actor in actors: actor.init.remote(args, role)        ← Phase C-Megatron

Phase C-SGLang：SGLangEngine.init() 在 rollout actor 进程
  sglang_engine.py:118
  ├─ _compute_server_args  组装 ServerArgs kwargs
  │   ├─ tp_size = rollout_num_gpus_per_engine // sglang_pp_size
  │   ├─ base_gpu_id = _to_local_gpu_id(get_base_gpu_id(args, rank))
  │   ├─ for f in dataclasses.fields(ServerArgs):
  │   │      if hasattr(args, f"sglang_{f.name}"): 反向取出注入
  │   └─ 叠加 sglang_overrides（来自 --sglang-config YAML）
  ├─ rollout_external?
  │   ├─ True: _init_external → 拉远端 /get_server_info 做字段一致性 sanity check
  │   └─ False: _init_normal → launch_server_process(ServerArgs(**kwargs))
  │           └─ multiprocessing.Process(target=launch_server, args=(server_args,))
  │           └─ _wait_server_healthy 轮询 /health_generate
  └─ _register_to_router → POST router /workers

Phase C-Megatron：MegatronTrainRayActor.init() 在 train actor 进程
  slime/backends/megatron_utils/actor.py:48
  ├─ monkey_patch_torch_dist()
  ├─ super().init() = TrainRayActor.init（设 role、rank、device）
  ├─ initialize.init(args)                ← _initialize_distributed + tokenizer + seed
  │   └─ load_function(custom_megatron_init_path)(args)   ← 用户钩子
  ├─ 串行 read hf_config / tokenizer（每个本地 rank 排队，防并发写）
  ├─ initialize_model_and_optimizer(args, role)   model.py:968
  │   ├─ setup_model_and_optimizer
  │   │   ├─ get_model(get_model_provider_func(args, role))
  │   │   │   └─ 三选一：
  │   │   │     ├─ custom_model_provider_path
  │   │   │     ├─ megatron_to_hf_mode=="bridge"  →  AutoBridge.to_megatron_provider
  │   │   │     └─ raw：core_transformer_config_from_args + GPTModel
  │   │   ├─ get_megatron_optimizer（可选 _patch_megatron_adam(StatelessAdam)）
  │   │   └─ get_optimizer_param_scheduler
  │   └─ load_checkpoint
  │       ├─ _is_megatron_checkpoint? → 走 mcore dist-ckpt
  │       └─ else → AutoBridge.load_hf_weights（要求 megatron_to_hf_mode=="bridge"）
  ├─ TensorBackuper.backup("actor")  备份 actor weight（offload 用）
  ├─ with_ref / with_opd_teacher / keep_old_actor → load_other_checkpoint
  └─ 选择 weight_updater：colocate→FromTensor；delta→FromDistributedDelta；
       disk→FromDisk；else→FromDistributed

Phase D：第一次 generate
  train.py 调 rollout_manager.generate.remote(0)
  → SGLang HTTP server 已就位 → 接收第一个 prompt batch
```

`__init__.py` 在 import megatron_utils 的瞬间会执行两个"早绑定补丁"：
`slime/backends/megatron_utils/__init__.py:9-19` 给 `deep_ep.Buffer.__init__` 包一层
`torch_memory_saver` interest-region 切换；同文件 `:23-40` 给 Qwen3VL 的两个
`RotaryEmbedding.forward` 注入 `packed_seq_params=None` 兼容签名。

---

## 4. 设计模式

### 4.1 SGLang 参数透传：拦截 `parser.add_argument`

`sglang_utils/arguments.py:43-115` 是这套设计最精彩的一段：

```python
old_add_argument = parser.add_argument
def new_add_argument_wrapper(*name_or_flags, **kwargs):
    # 把 --foo-bar → --sglang-foo-bar，dest → sglang_foo_bar
    ...
parser.add_argument = new_add_argument_wrapper
ServerArgs.add_cli_args(parser)   # 上游不知道自己被前缀化了
parser.add_argument = old_add_argument
```

`ServerArgs.add_cli_args(parser)` 是 SGLang 上游公开的 API，slime 没改一个字母，
只是在调用前后把 `parser.add_argument` 临时换成一个"自动加前缀"的版本。
反向回填发生在 `sglang_engine.py:612-620`：

```python
for attr in dataclasses.fields(ServerArgs):
    if hasattr(args, f"sglang_{attr.name}") and attr.name not in kwargs:
        kwargs[attr.name] = getattr(args, f"sglang_{attr.name}")
```

收益是 **SGLang 增加新 ServerArgs 字段时 slime 零修改**（默认通过）。
代价是少数字段在 `skipped_args` 名单里被显式拦掉（`arguments.py:45-63`，比如
`model_path`、`tp_size`、`port` 这些 slime 想自己掌控的）。

### 4.2 Megatron 透传：`ignore_unknown_args=True` + 后置 dest 校验

Megatron 侧没法做"前缀化注入"，因为 `_megatron_parse_args` 是上游函数。
slime 的折衷：`arguments.py:186` 调用时传 `ignore_unknown_args=True`，让所有
`--sglang-*` 通过；然后 `_hf_validate_args`（`arguments.py:93`）补一层 HF↔mcore
配置一致性检查（10 条等式断言），把 Megatron 内置 validator 漏掉的"hidden_size
对不上 hf_config 但能跑"这种错误暴露出来。

### 4.3 megatron_patch：monkey-patch 而不是 fork

`megatron_patch/__init__.py` 只有一行：`from . import megatron_chunked_grad_coalesce_patch`。
后者直接通过 `sys.modules["megatron.core.distributed.finalize_model_grads"]` 拿到
模块对象，覆盖里面的 `_allreduce_non_tensor_model_parallel_grads`
（`megatron_chunked_grad_coalesce_patch.py:129-131`）。

这个 patch 解决的具体问题：上游用一个 `_flatten_dense_tensors(grads)`
做 TP 侧的 grad all-reduce，对大模型来说这块连续内存可能撑爆 allocator。
slime 把它切成 1 GiB（`SLIME_GRAD_COALESCE_CHUNK_BYTES` 默认）的 chunk
分批 all-reduce，元素级 SUM/AVG 数学上等价。

更妙的是 patch 同时兼容 Megatron-LM 两条主线：
core_v0.13.0（`use_custom_fsdp`、`_get_main_grad_attr(param, use_custom_fsdp)`）
和 post-0.15.0rc7 dev（`use_megatron_fsdp`、`_get_main_grad_attr(param)` 单参）。
用 `inspect.signature(_get_main_grad_attr).parameters` 在运行时探查（line 39），
不用 version-conditional import。

### 4.4 mbridge 注册：装饰器副作用 + import-as-registration

```python
@register_model("qwen3_next")
class Qwen3NextBridge(Qwen2MoEBridge):
    _ATTENTION_MAPPING = Qwen2MoEBridge._ATTENTION_MAPPING | {...}
    def _weight_name_mapping_mcore_to_hf(...): ...
    def _weight_to_mcore_format(...): ...
    def _build_config(...): ...
```

`@register_model` 来自上游 `mbridge.core`，slime 只负责 import 触发副作用：
`slime_plugins/mbridge/__init__.py:1-9` 一次性 import 9 个具体 bridge。

每个 bridge 给出三类信息：
1. **Weight name mapping**：`_ATTENTION_MAPPING`、`_MLP_MAPPING`、`_MTP_MAPPING`
   字典，告诉 mbridge "mcore 这个 weight 对应 HF 哪个/哪几个 weight"；
2. **格式转换**：`_weight_to_mcore_format` / `_weight_to_hf_format`
   做 qkv 拼接、rope 重排（DeepseekV32 swap 前后 64 维：`deepseek_v32.py:35-55`）；
3. **构建配置**：`_build_config` 调 `_build_base_config(...)` 给 mcore 的
   `TransformerConfig` 注入 MoE/QKLayerNorm/RotaryInterleaved 等 model-specific 默认值。

这套 bridge 在 RL post-training 里被两个路径用：
- weight sync：`update_weight/hf_weight_iterator_bridge.py:45` 也 `import slime_plugins.megatron_bridge` 触发注册；
- checkpoint：`backends/megatron_utils/checkpoint.py:133` 加载 HF checkpoint。

---

## 5. 集成点

### 与 Ray actor 的接口

- **train actor → Megatron**：`MegatronTrainRayActor.init()`（`actor.py:48`）是
  Ray actor 类的方法，调 `initialize.init(args)` → `initialize_model_and_optimizer`。
  这意味着 **Megatron 的所有全局状态都被绑定在某个 Ray actor 进程里**——这是 slime
  能用同一份 Megatron-LM 代码并行跑 actor/ref/critic 多角色的关键。
- **rollout actor → SGLang**：`SGLangEngine` 本身继承 `RayActor`
  （`sglang_engine.py:101`），但它再 fork 一个 `multiprocessing.Process` 跑
  `sglang.launch_server`——也就是说 **Ray actor 不直接持有 SGLang 模型**，
  只持有 HTTP server 进程的句柄。RayActor 与 server 之间通过 HTTP 走（rank 0 才
  发请求，`_make_request` 里有 `if self.node_rank != 0: return` 守卫）。

### 与 customization 的接口

backends 暴露了 3 个 Megatron 钩子：

| 钩子参数 | 触发位置 | 用途 |
| --- | --- | --- |
| `--custom-megatron-init-path` | `initialize.py:100` | `init()` 最后调用，可在分布式初始化完成后 patch Megatron |
| `--custom-megatron-before-log-prob-hook-path` | `model.py:451` | `forward_only` 切到 eval 前调，用于按 store_prefix 切换模型/router |
| `--custom-megatron-before-train-step-hook-path` | `model.py:554` | train step 前调 |
| `--custom-model-provider-path` | `model_provider.py:66` | 完全自定义 GPTModel 构造 |

参数定义集中在 `slime/utils/arguments.py:1417-1438`（plugin 那一组）和
`:215-226`（custom_model_provider_path 与 reward 那组并列）。

### 与 weight sync 的接口

SGLangEngine 暴露 5 个 weight 更新入口（`sglang_engine.py:264-470`）：

- `init_weights_update_group` / `destroy_weights_update_group`（distributed 通道用）
- `update_weights_from_tensor`（colocate 时直接传 GPU 张量句柄）
- `update_weights_from_distributed`（NCCL 直传，含 `delta` 字段）
- `update_weights_from_disk`（disk 传，支持 delta `load_format`）
- `release_memory_occupation` / `resume_memory_occupation`（offload 时挪 KV cache）
- `pause_generation` / `continue_generation`（更新期间冻请求）

`MegatronTrainRayActor.init` 根据 `args.colocate` / `args.update_weight_mode` /
`args.update_weight_transport` 三个参数挑 4 个 `UpdateWeight*` 类之一
（`actor.py:139-159`），全部都通过上面这些 SGLangEngine 方法对接，不直接访问 SGLang。

---

## 6. 出人意料的决策

### 6.1 两套 model bridge 共存（`mbridge` vs `megatron_bridge`）

最让人意外的设计：`slime_plugins/mbridge/`（9 个模型）和 `slime_plugins/megatron_bridge/`
（1 个模型 glm4v_moe）这两个 bridge 系统**同时存在但分工不同**：

- **`mbridge`** 走的是 slime 自有路径（基于上游 mbridge 项目）：
  `model_provider.py:158-184` 默认分支，自己拼 `transformer_layer_spec` 然后实例化 `GPTModel`，
  bridge 类只负责"weight name 映射 + 格式转换"。**主要用于 weight sync 路径**
  （`hf_weight_iterator_bridge.py`、`megatron_to_hf/`），把训练完的 Megatron 张量
  写回 HF 格式给 SGLang 用。
- **`megatron_bridge`** 走 nvidia 官方 `megatron-bridge.AutoBridge`：
  `model_provider.py:87-123`，bridge 直接产出 model_provider 函数，可以自动
  从 HF checkpoint **加载权重到 Megatron**。**主要用于 checkpoint 加载**
  （`checkpoint.py:129-152` 的 `_load_checkpoint_hf` 分支）。

由 `--megatron-to-hf-mode` 选择走哪条（`arguments.py` 默认值未在范围内文件里）：
`raw`（默认，全自己拼 GPTModel）、`bridge`（用 AutoBridge）。
**bridge 只为新模型快速适配 HF checkpoint 而生**，weight sync 始终用 mbridge 这一套。
"两个系统并存"的副作用是新模型作者要决定接到哪个系统，但好处是 weight sync 的
稳定 codepath 不被 AutoBridge 的演进影响。

### 6.2 SGLang 参数被 monkey-patch 进 argparse，但 Megatron 只能 `ignore_unknown_args`

两个 backend 的参数透传策略**不对称**：

- SGLang 那侧能拦截 `parser.add_argument` 因为 `ServerArgs.add_cli_args(parser)`
  接受外部传入的 parser；
- Megatron 那侧 `parse_args()` 是顶层入口、内部自己 new parser，slime 没法插手，
  只能 `ignore_unknown_args=True` 让 `--sglang-*` 通过。

结果：CLI 解析被切成两个 phase（`slime/utils/arguments.py:1530-1558`），
先用独立 ArgumentParser + `parse_known_args` 吃掉 sglang 部分，再让 Megatron 解析
剩下的。这种"phased argparse"的隐藏成本是：**两个 parser 都不知道对方的 help 信息**，
`python train.py --help` 出来的是 Megatron + slime 自有参数，看不到 SGLang 的。

### 6.3 只对 Megatron 打一个 patch

`megatron_patch/` 里**只有一个补丁**：grad coalesce chunking。这条边界守得极严。
其他能 monkey-patch 的需求（StatelessAdam 替换 `megatron_optimizer.Adam`）走
context manager `_patch_megatron_adam`（`model.py:248-267`），用完就还原；
deep_ep 与 Qwen3VL 的 forward 包装在 `megatron_utils/__init__.py` 里，
是 import-time side effect。**核心 megatron_patch 严格只解决"上游会 OOM 但不能修"
的硬伤**——chunking 是因为 `_flatten_dense_tensors` 在 mcore 里被多处复用，
fork 就要追同步成本，而这个改动数学上等价、行为 100% 兼容、还兼容 mcore 两个主版本
（运行时探签名分支，line 39）。

slime 没有 fork Megatron-LM，全部 patch 集中在两个地方：一个 monkey-patch（运行时
打到 sys.modules）、一组 import-time wrap（包装现有类的 `__init__` 或 `forward`）。
**这是"native engine 透传"五条赌注里最物理的一条**：上游升级时，slime 这边除了
重新检查 patch 适用性几乎不需要改动。

### 6.4 SGLangEngine 是 RayActor，但模型在 multiprocessing 子进程

`SGLangEngine(RayActor)`（`sglang_engine.py:101`）这个类本身**只是个 HTTP 客户端**——
它通过 `requests.post` 给 server 发 RPC。真正持模型的是 `launch_server_process`
（`sglang_engine.py:51`）fork 出去的 `multiprocessing.Process`，跑的是 SGLang
上游的 `launch_server`。

这导致一个反直觉的拓扑：
```
Ray actor (RayActor) ──HTTP→ multiprocessing.Process (SGLang HTTP server) ──TP→ GPU workers
```

为什么这样？因为 SGLang 自己的进程拓扑（TP workers + scheduler + tokenizer + HTTP）
**已经在 multiprocessing 里组好了**，slime 不重新发明。Ray 这层只负责 placement、
故障检测、shutdown。`SGLangEngine` 持有 server PID（`self.process.pid`），
shutdown 时 `kill_process_tree(self.process.pid)` 一次性杀掉整棵树。
不直接把 SGLang 模型塞进 Ray actor 的另一个意义是：**weight sync 的所有控制流
都走 HTTP**——CPU-light、跨语言、对 Ray 的 object store 没有依赖。

---

## 附录

### 通读过的文件（21 份）

- `slime/backends/megatron_utils/__init__.py`
- `slime/backends/megatron_utils/initialize.py`
- `slime/backends/megatron_utils/arguments.py`
- `slime/backends/megatron_utils/model.py`（仅读 1-318 + 700-790 + 968-1007）
- `slime/backends/megatron_utils/model_provider.py`
- `slime/backends/megatron_utils/actor.py`（仅读 1-200，init 部分）
- `slime/backends/megatron_utils/sglang.py`
- `slime/backends/megatron_utils/megatron_patch/__init__.py`
- `slime/backends/megatron_utils/megatron_patch/megatron_chunked_grad_coalesce_patch.py`
- `slime/backends/megatron_utils/server/__init__.py`
- `slime/backends/megatron_utils/server/megatron_server.py`
- `slime/backends/megatron_utils/server/arguments.py`
- `slime/backends/megatron_utils/server/logprob_utils.py`（仅读 1-100）
- `slime/backends/megatron_utils/stateless_adam.py`
- `slime/backends/megatron_utils/ci_utils.py`
- `slime/backends/megatron_utils/misc_utils.py`
- `slime/backends/megatron_utils/checkpoint.py`（仅读 100-160 的 bridge 分支）
- `slime/backends/sglang_utils/__init__.py`（空文件）
- `slime/backends/sglang_utils/arguments.py`
- `slime/backends/sglang_utils/sglang_engine.py`
- `slime_plugins/mbridge/__init__.py`
- `slime_plugins/mbridge/qwen3_next.py`
- `slime_plugins/mbridge/glm4moe.py`
- `slime_plugins/mbridge/deepseek_v32.py`
- `slime_plugins/megatron_bridge/__init__.py`
- `slime_plugins/megatron_bridge/glm4v_moe.py`（仅读 1-80）
- `slime_plugins/models/__init__.py`（空）
- `slime_plugins/models/qwen3_next.py`（仅读 1-80）
- `slime_plugins/models/glm4.py`
- `slime_plugins/models/glm5/glm5.py`（仅读 1-50）
- `slime/utils/megatron_bridge_utils.py`（仅读 1-80）
- `slime/utils/arguments.py`（仅读 1417-1568、1780-1820 段落，找 plugin 参数）
- `slime/ray/rollout.py`（仅读 140-260，找 SGLangEngine 实例化点）

### 未读但已索引的文件（归其他子系统）

- `slime/backends/megatron_utils/{actor,loss,data,cp_utils,checkpoint,hf_checkpoint_saver}.py` 200 行以下后续 → training 子系统
- `slime/backends/megatron_utils/update_weight/`、`megatron_to_hf/` → weight sync 子系统
- `slime/backends/sglang_utils/{sglang_config,external,server_control}.py` → deployment 子系统
- `slime/backends/megatron_utils/kernels/{fp8_kernel.py,int4_qat/}` → 性能子系统
- `slime_plugins/models/glm5/ops/{sparse_mla,tilelang_*}` → 性能子系统
- `slime/utils/arguments.py` 主体 2010 行 → CLI / 参数子系统

### 重要但归别处的隔壁文件（交叉引用线索）

- `slime/utils/arguments.py:1525` `parse_args()`：backends 透传 CLI 的两阶段实现
- `slime/ray/rollout.py:168`：`RolloutRayActor = ray.remote(SGLangEngine)`，
  SGLangEngine 真正变成 Ray actor 的地方
- `slime/ray/train_actor.py`：`MegatronTrainRayActor` 的父类
- `slime/backends/megatron_utils/update_weight/hf_weight_iterator_bridge.py:45`：
  mbridge 在 weight sync 路径上的注册触发点
