# 研究笔记：weight sync —— 四条传输路径

## TL;DR

slime 在 Megatron-LM 训练完一个 step 后，需要把更新过的参数推送给 SGLang rollout engine 接管下一轮采样。看起来"复制一次张量"足够，但实际有四种成本/能力组合：(1) **NCCL 全量广播** 是默认路径，最快但要求训推同集群可建 NCCL group；(2) **共享内存 / CUDA IPC（tensor 路径）** 在 colocate 模式下复用同一进程地址空间，权重换"句柄"而不是字节；(3) **共享文件系统 (disk full)** 把每次 sync 物化成完整 HF checkpoint，让 SGLang `update_weights_from_disk` 重读，给跨集群 / 不同 GPU 厂商兜底；(4) **delta sync**（NCCL 或 disk 两种载体）按位差比对、只发送变化的 (位置, 值)，密度 2-3% 时把 355B 模型的 wire 量压到约 5 GB，是跨数据中心训推解耦的支柱。四条路径共享一套 Megatron→HF 转换器 + processors（FP8 / compressed-tensors 量化、vocab padding 移除），只在最后的 carrier 上分叉。

## 1. 架构与模块边界

`slime/backends/megatron_utils/` 下两个子目录构成 weight sync 子系统：

- `update_weight/` —— 四个 carrier 类 + 公共抽象 + HF 迭代器。`__init__.py` 是空文件，外部统一通过 `from .update_weight.update_weight_from_X import ...` 引入。
- `megatron_to_hf/` —— Megatron 内部 param name / shape 到 HuggingFace 命名空间的转换，按模型架构分文件。处理子任务的小工具放在 `megatron_to_hf/processors/`（padding remover、FP8 量化、compressed-tensors 量化）。
- `sglang.py` —— 一个 44 行的"防火墙"模块，集中 import 所有 SGLang 类型（`DeltaEncoding`、`FlattenedTensorBucket`、`MultiprocessingSerializer` 等），对老 image 缺失的符号 try/except 设 None，让 weight sync 主代码不直接耦合 SGLang 版本。

**关键边界**：

- `update_weight/common.py`（237 行）提供两个核心算子：`all_gather_param`（同步 TP/EP 收集分片）+ `named_params_and_buffers`（产出跨 PP/EP 的全局参数名迭代器，重写 `decoder.layers.{idx}` 和 `mtp.layers.{idx}.experts.{idx}` 的索引偏移，让所有 rank 看到统一的参数命名空间）。这一层不感知 carrier。
- `megatron_to_hf/__init__.py:24` 的 `convert_to_hf(args, model_name, name, param, quantization_config)` 是所有 carrier 的统一入口：`remove_padding → 按模型名分发到具体 converter → quantize_params`。这是参数级的"管线"。
- `update_weight/hf_weight_iterator_base.py` 抽象出 chunk 级迭代器，让 colocate path 在 raw（手写 converter）和 bridge（megatron-bridge 库）两条路线之间切换。

四个 carrier 类全部由 `slime/backends/megatron_utils/actor.py:140-159` 根据 `--update-weight-mode`、`--update-weight-transport`、`--colocate` 三个参数选择实例化：

```python
if colocate:                             → UpdateWeightFromTensor   # IPC + 可选 distributed
elif mode == "delta":                    → UpdateWeightFromDistributedDelta
elif mode == "full" and transport=="disk": → UpdateWeightFromDisk
else (mode == "full" and transport=="nccl"): → UpdateWeightFromDistributed
```

## 2. 关键抽象

### Carrier 类入口

四个 `update_weights()` 方法对外签名完全一致（`self.weight_updater.update_weights()`），返回 None；版本号递增和 `pop_metrics()` 接口一致。

- `UpdateWeightFromDistributed.update_weights` — `slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py:102`
- `UpdateWeightFromDistributedDelta.update_weights` — `slime/backends/megatron_utils/update_weight/update_weight_from_distributed_delta.py:568`
- `UpdateWeightFromTensor.update_weights` — `slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py:147`
- `UpdateWeightFromDisk.update_weights` — `slime/backends/megatron_utils/update_weight/update_weight_from_disk.py:61`

`UpdateWeightFromDistributed._send_weights` (`update_weight_from_distributed.py:135`) 是 NCCL 全量路径的核心循环；`UpdateWeightFromDistributedDelta` 直接继承它并复用 `_iter_non_expert_chunks` / `_iter_expert_chunks`（这两个迭代器只关心"产出 HF chunk"，并不关心后续会被广播还是会被 diff/encode）。

公共底层函数：

- `connect_rollout_engines_from_distributed` — `update_weight_from_distributed.py:277`（创建跨训练-engine NCCL group，注意 `engine_gpu_counts` 支持异构 TP，比如 prefill TP=2 + decode TP=4，每个 engine 占用不同 rank 数）
- `update_weights_from_distributed` — `update_weight_from_distributed.py:335`（Ray RPC 推 metadata + NCCL broadcast 推张量）
- `post_process_weights` — `update_weight_from_distributed.py:370`（int4/fp4 quant 时的两阶段 hook，被 NCCL/tensor/delta 路径调用）

### HF 迭代器三种形态

`HfWeightIteratorBase.create` (`hf_weight_iterator_base.py:5`) 是工厂，按 `--megatron-to-hf-mode` 在 raw / bridge 两种实现间切换：

- `HfWeightIteratorDirect` (`hf_weight_iterator_direct.py:19`)：自家手写转换路径。先在 `_get_megatron_local_param_info_buckets` 把所有参数按 `--update-weight-buffer-size` 切桶（注意要乘以 TP 复制数才能反映 all-gather 后的真实占用），然后每桶执行 `_get_megatron_full_params`（先 PP broadcast → 后 EP broadcast → 最后 TP `all_gather_params_async`），再喂给 `megatron_to_hf` 转换器。
- `HfWeightIteratorBridge` (`hf_weight_iterator_bridge.py:39`)：用 `megatron-bridge` 库的 `AutoBridge.from_hf_pretrained(...).export_hf_weights(model)` 直接吐出 HF 格式 stream，slime 只接 streaming quantize + chunk。代码量从 ~210 行降到 ~110 行，代价是引入一个外部库依赖，且当前不支持量化（注释里 TODO）。

第三个隐藏形态：**`UpdateWeightFromDistributed` 不用 iterator 类**。它的 `_iter_non_expert_chunks` (`update_weight_from_distributed.py:152`) 和 `_iter_expert_chunks` (`update_weight_from_distributed.py:177`) 是直接长在 carrier 类里的生成器，每次 yield 时立刻发出 broadcast（不需要先在内存里凑齐整批）。这是为什么 NCCL 全量路径不走 HfWeightIterator 抽象——它要 carrier 和 chunk 边界同步，不能分两层抽象。

### Converter 注册

`megatron_to_hf/__init__.py:37` 的 `_convert_to_hf_core` 是一个**纯字符串匹配的注册表**：

```python
if "minimaxm2" in model_name: ...
elif "deepseekv3" in model_name or "glmmoedsa" in model_name: ...
elif "glm4moe" in model_name: ...
elif "qwen3moe" in model_name: ...
elif "qwen2" in model_name or "qwen3" in model_name: ...
elif "llama" in model_name: ...
```

没有 decorator，没有 `register_converter`，就是一串 if-elif。Model name 来自 `type(hf_config).__name__.lower()`（actor.py:164），所以注册的"key"实际上是 HF config 类名子串。新增模型需要在这里加分支，并在 `__init__.py` 头部 `from .X import convert_X_to_hf`。

## 3. 数据流

完整流程（以非 colocate、`mode=full transport=nccl` 为例）：

1. **训练完一个 step**（`slime/backends/megatron_utils/actor.py:581 update_weights`）：actor 等 rollout engine 全部停止（`pause_generation` + `flush_cache`），通过 Ray 拿到 `rollout_engines` 列表、`engine_gpu_counts/offsets`、`rollout_engine_lock`。
2. **首次或新增 engine** → `self.weight_updater.connect_rollout_engines(...)` 建立或重建 NCCL group。Group name 是 `f"slime-pp_{pp_rank}"`，每个 PP rank 一个 group，只有 DP=TP=0 的"PP 源 rank"参与广播。
3. **正式 sync** → `self.weight_updater.update_weights()`：
   - 走 `weights_backuper.get("actor")` 拿到 `weights_getter` 闭包返回的 megatron 本地权重 dict（在 `train_actor` 调用前已经 backup）。
   - 对每个参数：`all_gather_param` 把 TP 分片汇齐 → `convert_to_hf` 把名字 + shape 转成 HF 格式（含 GLU rechunk、QKV split、router 重命名等）→ 经过 `remove_padding`（vocab）→ 经过 `quantize_params`（如果配了 fp8 / int4） → 累积到 buffer 直到达到 `--update-weight-buffer-size`。
   - Bucket flush：`_update_bucket_weights_from_distributed` 拿到 lock，做 Ray RPC（告诉 engine 这一波 names/dtypes/shapes）+ NCCL `dist.broadcast(param, src=0, group)`（同时进行）。
4. **接收端**（SGLang engine）：在 `slime/backends/sglang_utils/sglang_engine.py:431 update_weights_from_distributed`（HTTP 转发到 SGLang 内部 endpoint），按 names/dtypes/shapes 顺序 `dist.broadcast` recv 张量，再调用 SGLang 自己的 `model.load_weights(...)` 写入。
5. **收尾** → 所有 chunk 发完 → `continue_generation` 让 engine 恢复采样。

**FP8 / quant 是在第 3 步管线里就完成的**：`quantize_params_fp8` (`quantizer_fp8.py:10`) 在 PP 源 rank 上把 bf16 权重做 blockwise FP8 cast（用 triton kernel 或 deepgemm 的 ue8m0 变体），再把 `(qweight, scale)` 两个张量加进 HF chunk。也就是说 **wire 上传的就是 FP8 + scale**，rollout engine 不需要再做量化。compressed-tensors (int4/fp4) 还要额外两次 RPC：在 broadcast 前 `restore_weights_before_load`（把 SGLang 侧老的 weight cache 复原），在 broadcast 后 `post_process_quantization`（让 SGLang 把刚收到的"伪量化"张量压回真正的 int4 packing）。

**Delta 路径的 4 阶段管线**（`update_weight_from_distributed_delta.py:625-704`）：

```
[PP broadcast → TP gather → HF convert]        ← 复用全量路径
       ↓
[prefetch snapshot H2D（侧 stream）]            ← 1-step lookahead，与下一 chunk 计算重叠
       ↓
[bytewise diff: current ≠ snapshot]            ← view as int_dtype，不做算术
       ↓
[encode_indices / encode_deltas]                ← 稀疏位置 + 紧凑 values
       ↓
[bucket.add 累积，达到阈值就 flush]
       ↓
[NCCL: broadcast(__positions__, __values__)
 OR disk: AsyncSafetensorsWriter.enqueue(...)] ← 两条 carrier 同 wire 格式
       ↓
[D2H 更新快照（侧 stream，与下一 chunk encode 重叠）]
```

**Disk 全量路径**（`update_weight_from_disk.py:61`）就简单得多：rank 0 调 `save_hf_model_to_path` 把完整 HF checkpoint 写到 `weight_v{version:06d}/`，然后 RPC 让所有 engine `update_weights_from_disk(model_path=...)`。SGLang 内部走 HF 标准的 safetensors 加载。结束后默认 `shutil.rmtree(version_dir)` 删除 checkpoint，除非 `--update-weight-disk-keep-files`。

## 4. 设计模式

### 四条路径的统一抽象

四个 carrier 类**没有公共基类**（除了 `UpdateWeightFromDistributedDelta` 继承自 `UpdateWeightFromDistributed`），只有**鸭子类型契约**：

- `__init__(args, model, weights_getter, *, model_name, quantization_config)`
- `connect_rollout_engines(rollout_engines, rollout_engine_lock, engine_gpu_counts=, engine_gpu_offsets=)`
- `disconnect_rollout_engines()`（disk 是 no-op）
- `update_weights()`
- `pop_metrics() -> dict[str, float]`
- `weight_version: int`（递增，CI 测试在 `actor.py:622` 拿来核对）

这种"四个类各写一遍"的方式而不是抽公共基类，理由从代码里能看出来：四条路径的"循环结构"差异巨大。NCCL 全量是 chunk-by-chunk 流式广播，没有内存里凑整批；Disk 全量是先落盘再 RPC，控制面单纯；Tensor (IPC) 是"先序列化整桶再走 gloo gather_object 跨进程交换 IPC handle"；Delta 是"先求差再发 sparse blob 再更新 snapshot"。强行抽公共基类只会逼大家走 template method + 一堆空 hook，反而不如平铺四个类清晰。

但 **Delta 是个例外** —— 它和 Full NCCL 共享 ~70% 的代码（PP/TP/EP gather、HF convert、bucket 累积），所以继承是合理的，并通过 `_on_chunk` (`update_weight_from_distributed.py:147`) 这一个 hook 注入 delta-specific 行为。

### 4 路径间共享层

```
┌─────────────────────────────────────────────────────────────┐
│  carrier 层（4 个独立实现）                                 │
│  Full-NCCL │ Full-Disk │ Delta-NCCL/Disk │ Tensor-IPC      │
└────────────┬────────────┬────────────────┬─────────────────┘
             ↓            ↓                ↓
        ┌─────────────────────────────────────┐
        │  HfWeightIterator (raw | bridge)    │  ← 只在 Tensor 路径用
        └─────────────────────────────────────┘
             ↓
        ┌─────────────────────────────────────┐
        │  common.all_gather_param            │
        │  common.named_params_and_buffers    │
        └─────────────────────────────────────┘
             ↓
        ┌─────────────────────────────────────┐
        │  megatron_to_hf.convert_to_hf       │
        │  ├ remove_padding (processor)       │
        │  ├ convert_{model}_to_hf (10+)      │
        │  └ quantize_params (processor)      │
        │     ├ fp8                            │
        │     └ compressed-tensors             │
        └─────────────────────────────────────┘
```

### 模型转换器的共同模式

10 个 converter 文件（llama / qwen2 / qwen3moe / deepseekv3 / glm4 / glm4moe / glm4 / gpt_oss / mimo / minimax_m2 / qwen3_5 / qwen3_next / qwen3_vl）共享 95% 的代码结构：

```python
def convert_X_to_hf(args, name, param):
    # 1) 三个根级特殊名直接映射
    if name == "module.module.embedding.word_embeddings.weight": return [(...)]
    if name == "module.module.output_layer.weight":              return [(...)]
    if name == "module.module.decoder.final_layernorm.weight":   return [(...)]

    # 2) 取 head_dim / value_num_per_group
    head_dim = args.kv_channels or args.hidden_size // args.num_attention_heads
    value_num_per_group = args.num_attention_heads // args.num_query_groups

    # 3) 正则提取 layer_idx + rest
    match = re.match(r"module\.module\.decoder\.layers\.(\d+)\.(.+)", name)
    if match:
        layer_idx, rest = match.groups()

        # 4) 对 MoE 模型先匹配 experts 模式
        # 5) 对 MoE 模型再匹配 shared_experts 模式
        # 6) attention：linear_qkv → q/k/v 切分（用 num_query_groups view + split）
        # 7) MLP：linear_fc1 → gate/up chunk(2,dim=0)（GLU 拆分）
        # 8) layernorm 重命名
        # 9) router / expert_bias 重命名
    raise ValueError(...)
```

差异点全部集中在 HF 侧的命名约定：
- minimax_m2 用 `block_sparse_moe.experts.{i}.w1/w2/w3`；其他用 `mlp.experts.{i}.gate_proj/down_proj/up_proj`
- qwen3moe 的 shared expert 是 `mlp.shared_expert.*`（无 s），deepseekv3/glm4moe 是 `mlp.shared_experts.*`（带 s）
- deepseekv3 多出 MLA（`linear_q_down_proj` → `q_a_proj`、`linear_kv_down_proj` → `kv_a_proj_with_mqa`、`q_a_layernorm`）以及 indexer 相关参数（`wq_b`、`wk`、`k_norm` 需要做 `cat([param[64:], param[:64]])` 半切换位）
- qwen3_next、mimo 有 MTP（Multi-Token Prediction）层，需要把 `mtp.layers.{idx}` 映射回 `decoder.layers.{idx+num_layers}` 后递归调用同一个 converter
- glm4moe 多了 `post_self_attn_layernorm` 和 `post_mlp_layernorm`
- gpt_oss 有可学习 softmax offset（`softmax_offset` → `self_attn.sinks`），并且 expert 有 bias

**最复杂的两个 converter**：`deepseekv3.py`（151 行，承担 MLA + indexer + MTP + shared experts 全套）和 `qwen3_next.py`（194 行，MTP 递归映射）。新增模型基本就是从一个相近 converter 复制一份，调整 HF 命名约定。

### Processors 链式处理

`megatron_to_hf/processors/__init__.py:8 quantize_params` 是分派器：根据 `quantization_config["quant_method"]` 路由到 `quantize_params_fp8`、`quantize_params_compressed_tensors` 或原样返回。没有量化时它就是恒等函数，所以 `convert_to_hf` 始终安全调用一次。

`padding_remover.remove_padding` 只对 `embedding.word_embeddings.weight` 和 `output_layer.weight` 两个名字生效，把 Megatron 内部对齐到 TP shard 的 vocab 切掉到真实 vocab size（GPT-OSS 这种 tokenizer vocab 小于 model vocab 的场景尤其重要，actor.py:133-137 解释了优先用 hf_config.vocab_size 的原因）。

## 5. 集成点

### 与 training 的接口

唯一入口是 `actor.update_weights()`（`actor.py:581`，被 `train_actor.update_weights` 通过 Ray RPC 转发）。Training 子系统不知道 carrier 类型——它只持有 `self.weight_updater` 这个不透明对象。

`weights_getter=lambda: self.weights_backuper.get("actor")` 这个闭包是与 training 的真正契约：weight sync 不直接读 Megatron 内部 state_dict，而是要求 training 先用 `TensorBackuper` 把当前 actor 参数 backup 到一个 tagged dict 里（默认 tag 是 `"actor"`）。这允许 training 在做 keep_old_actor / queue-style update 时保持权重历史。但 `UpdateWeightFromDistributed` 和 `UpdateWeightFromDistributedDelta` 实际上**没有用 `weights_getter`**——它们直接通过 `named_params_and_buffers(args, self.model)` 遍历活的 model 模块。只有 Tensor 和 Disk 路径会调用 `self.weights_getter()`。这是一个未对齐的接口，看起来是历史遗留。

### 与 backends (SGLang) 的接口

SGLang 侧三个对应的 receiver endpoint：

- `sglang_engine.py:264 update_weights_from_tensor` ← colocate IPC 路径
- `sglang_engine.py:382 update_weights_from_disk` ← disk full 路径 + delta disk 路径（用 `load_format="delta"` 参数区分）
- `sglang_engine.py:431 update_weights_from_distributed` ← NCCL full 路径 + delta NCCL 路径（用 `load_format="delta"` + `delta=DeltaSpec(...)` 参数区分）

注意 delta 不是单独的 endpoint：它复用 disk / distributed 两个 endpoint，加 `load_format="delta"` 让 SGLang 走 `_apply_delta_payload` 解码路径。这个设计的好处是新 carrier 不需要新 endpoint，但代价是 SGLang 必须知道 delta wire 格式（`DeltaSpec`、`DeltaParam`、`DeltaEncoding` 在 SGLang 侧定义，在 `slime/backends/megatron_utils/sglang.py` 重新 import，对老 image try/except None）。

### 与 deployment 的接口

`engine_gpu_counts` / `engine_gpu_offsets` 是 deployment（rollout_manager）通过 Ray 传给 `connect_rollout_engines` 的。这两个数组使三件事变可能：(1) 异构 TP 组（prefill engine TP=2 + decode engine TP=4），(2) **placeholder rank**（GPU 槽位被占但 engine 没启动，例如 fault-tolerance 期间），(3) external engine（slime 不启动而是连接已有 server）。

`UpdateWeightFromTensor.connect_rollout_engines` (`update_weight_from_tensor.py:62`) 用 `engine_gpu_offsets + engine_gpu_counts` 判断哪些 engine 是 colocate 的（它们的 GPU 范围落在 actor GPU 范围内），剩下的 engine 走 distributed broadcast；这是 colocate + distributed 混合模式的入口（`use_distribute=True`）。

External engine 的 update 走 disk 路径居多——这是 `docs/zh/advanced/external-rollout-engines.md` 推荐的兜底方案：不需要建 NCCL group，只要训练端和 engine 端能看到同一个 `--update-weight-disk-dir`。

## 6. 出人意料的决策

### 6.1 为什么需要 delta sync（不是单纯优化）

直觉上 delta sync 是"传输量更少、节省带宽"。但代码注释（`update_weight_from_distributed_delta.py:6-28` 和 `docs/zh/advanced/delta-weight-sync.md`）表明真正动机是**训推跨数据中心解耦**——训练集群和推理集群可能在不同 region，只能通过共享 S3 / NFS 通信，带宽在 100 MB/s 量级。对 355B 模型，全量 ~700 GB 完全不可行；2-3% 密度的 delta 是 ~5 GB，可行。NCCL transport 存在的唯一原因是"作为基线，在数据中心内验证 wire encoding 和 apply 逻辑没问题"——它不是 production path。这是一个明显被 Cursor + Fireworks AI 的 Composer 2 架构 + arXiv 2509.19128 启发的设计（README 里直接引用）。

**Delta apply 是 bit-exact 的**（`docs/zh/advanced/delta-weight-sync.md` 第 89 行）：接收端把变化位置直接覆写为训练端的精确字节，没有任何算术，所以没有数值漂移，不需要周期性 base re-sync。这一点比"每 N 步发一次 full sync 校正"的朴素 delta 设计强很多。

### 6.2 Disk 路径不是 fault recovery，而是异构 GPU + 外部 engine

`update_weight_from_disk.py` 是四条路径中代码最短的（97 行），但它解决一个 NCCL 无能为力的问题：**训练和推理用不同型号或不同厂商的 GPU**。`docs/zh/advanced/external-rollout-engines.md` 第 54-56 行说得明白："训练可以在一组 GPU 上运行，rollout serving 可以放在另一组不同型号或不同厂家的 GPU 上"——只要 SGLang 支持目标硬件就行。NCCL 在跨厂商场景下根本建不出 group。

不是 fault recovery（fault recovery 走 `recover_updatable_engines` + 重建 NCCL group 的路径，见 `actor.py:585-588`）。

### 6.3 HF weight iterator 三种形态的真实原因

`HfWeightIteratorBase` 只在 `UpdateWeightFromTensor` 里使用。`UpdateWeightFromDistributed` 和 `UpdateWeightFromDistributedDelta` 自己写迭代器。为什么不让 NCCL 路径也走 `HfWeightIteratorBase`？

读 `hf_weight_iterator_direct.py` 的实现：它在 `_get_megatron_full_params` 里先**集中做 PP + EP broadcast → 然后 TP all-gather → 一次性吐整批 HF tensor**。这种"先凑齐整桶"的模式适合 IPC 路径（把整桶塞进 `FlattenedTensorBucket` 走 CUDA IPC），但**不适合 NCCL 流式广播**——NCCL 路径希望 chunk 边界和 broadcast 边界对齐，每个 chunk 一旦凑够 `update_weight_buffer_size` 就立刻广播出去，不在 CPU 侧攒整批。所以 NCCL 路径自己写了 `_iter_non_expert_chunks` / `_iter_expert_chunks`，把"all_gather + convert + buffer + yield"全在一个生成器里串起来。

Bridge 形态（`HfWeightIteratorBridge`）是新引入的选项，让 slime 不必为每个新模型手写 converter，直接用 megatron-bridge 库的转换。代价是把控制流交给外部库，且**不支持量化**（代码里有 `# TODO support quantization`）。这是个还在路上的迁移。

### 6.4 FP8 在传输时的角色

第一反应可能是"FP8 量化是 inference engine 的事，训练 sync 时传 bf16，engine 接收后再量化"。但 slime 不是这样：`quantize_params_fp8` 在**发送前**就把 bf16 转成 FP8（`update_weight_from_distributed.py:166` 调用 `convert_to_hf` 时已经做完）。Wire 上传的就是 FP8 + scale。

理由有两个：(1) **省带宽**——FP8 wire 大小是 bf16 的一半。(2) **训练和推理使用一致的量化方案** ——具体来说 deepgemm 的 ue8m0 scale 变体（`sglang.py:3 quant_weight_ue8m0`）是 SGLang 内部使用的量化算子，slime 直接调用它确保比特级一致。这是为什么 `quantize_params_fp8` 要 import `sglang.srt.layers.quantization.fp8_utils` 的原因。

但 compressed-tensors (int4/fp4) 不一样：那条路径是发"伪量化"（bf16 占位）+ 两阶段 hook (`post_process_weights`)，让 SGLang 侧自己做 int4 packing。两个量化方案分叉的原因没有写出来，推测是 int4 packing 太复杂（涉及 `pack_to_int32` + `WQLinear_GEMM` 的 AWQ-style 重排），不适合在训练端做。

### 6.5 Delta 的"snapshot 在 pinned CPU"+ 双侧 stream pipeline

`DeltaState`（`update_weight_from_distributed_delta.py:259-329`）保留全部 HF 参数的副本在 pinned CPU memory。355B 模型的 snapshot 大约 700 GB——这是**训练节点的 host memory 需求**，是 delta sync 的硬约束。论文里通常不写这个数字。

snapshot 的更新和读取都在**侧 stream** 上做（`d2h_stream` 和 `h2d_stream`），与默认 stream 的 compute 重叠：当前 chunk 在 GPU 上 diff + encode 时，下一 chunk 的 snapshot 正在 H2D 拷贝，刚发完的 chunk 的新 snapshot 正在 D2H 拷贝。这种 "1-step lookahead pipeline" 是 `_pipeline_pass`（`update_weight_from_distributed_delta.py:661`）的核心，把 H2D / D2H 延迟从串行收益里挖出来。

### 6.6 Expert pass 切 4 个 sub-pass

`_EXPERT_SUBPASSES = 4`（`update_weight_from_distributed_delta.py:648`）把 MoE expert 参数切成 4 份依次处理，每份处理完做一次 `_publish_batch`。理由（注释里写明）是"receiver apply for an earlier batch overlaps with later expert encoding"——让 SGLang 端在 apply 第 1 个 sub-pass 时，训练端已经在 encode 第 2 个 sub-pass，把"end-of-sync 大 RPC 等待"摊平。Megatron 把 MoE 层均匀分到 PP rank，所以每个 rank 的 expert param 列表长度一样，切 4 份后每个 rank 的 publish 次数也一样（保持 barrier 不会 desync）。

非 expert pass 不切，因为非 expert 参数总量小、流式广播本身已经够流畅。

### 6.7 Disk delta 的 AsyncSafetensorsWriter 后台线程

`AsyncSafetensorsWriter` (`update_weight_from_distributed_delta.py:402`) 是单独的 daemon thread，从 Queue 拉 `(path, tensors, metadata)`，做 safetensors encode + 可选 zstd 压缩 + 原子 rename。这把磁盘 I/O 从 GPU 计算关键路径上摘除。配合 `commit_future`（自定义 pre-push hook，例如 S3 复制）和 `_rpc_executor`（单线程 ThreadPoolExecutor，让 RPC dispatch 也异步），整个 publish 流程变成：

```
GPU 算完一个 bucket → 后台写盘 + 后台 zstd 压缩 + 后台等 commit + 后台 fire RPC
                  ↑ 主线程不阻塞，继续 encode 下一个 bucket
```

这是为什么 disk delta 在 100 MB/s 共享 FS 上还能跑得动的关键。

### 6.8 Checksum 用 `torch.hash_tensor` 而不是 CRC

`_checksum`（`update_weight_from_distributed_delta.py:95`）用 `torch.hash_tensor` 对 positions 和 values 各做一次 XOR-reduce 算 hash，sender 编完算一次，receiver 收完算一次，不一致就说明 wire 传输出错。这是 GPU-resident 的 reduction，比把张量拉回 CPU 算 CRC 快很多。代价是 `torch.hash_tensor` 是新 API（不是所有 PyTorch 版本都有），且 hash 冲突理论上存在（但 XOR + 移位组合下实际可以忽略）。

## 附录

### 已通读文件清单（行号位置 = 关键抽象）

| 文件 | 角色 | 关键位置 |
|---|---|---|
| `slime/backends/megatron_utils/update_weight/__init__.py` | 空 (0 行) | — |
| `slime/backends/megatron_utils/update_weight/common.py` | TP gather + 全局 param name | `:15 all_gather_param`, `:53 all_gather_params_async`, `:118 named_params_and_buffers` |
| `slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py` | Full NCCL carrier | `:23 UpdateWeightFromDistributed`, `:102 update_weights`, `:135 _send_weights`, `:277 connect_rollout_engines_from_distributed`, `:335 update_weights_from_distributed` |
| `slime/backends/megatron_utils/update_weight/update_weight_from_distributed_delta.py` | Delta carrier (NCCL + disk) | `:259 DeltaState`, `:335 DeltaBucket`, `:402 AsyncSafetensorsWriter`, `:479 UpdateWeightFromDistributedDelta`, `:568 update_weights`, `:625 _send_weights`, `:705 _flush_bucket`, `:747 _publish_batch` |
| `slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py` | Colocate IPC carrier | `:24 UpdateWeightFromTensor`, `:62 connect_rollout_engines`, `:147 update_weights`, `:218 _send_to_colocated_engine` |
| `slime/backends/megatron_utils/update_weight/update_weight_from_disk.py` | Full Disk carrier | `:21 UpdateWeightFromDisk`, `:61 update_weights` |
| `slime/backends/megatron_utils/update_weight/hf_weight_iterator_base.py` | 迭代器工厂 | `:4 HfWeightIteratorBase`, `:5 create` |
| `slime/backends/megatron_utils/update_weight/hf_weight_iterator_direct.py` | 手写转换 + chunk 迭代 | `:19 HfWeightIteratorDirect`, `:108 _get_megatron_local_param_info_buckets`, `:138 _get_megatron_local_param_infos` |
| `slime/backends/megatron_utils/update_weight/hf_weight_iterator_bridge.py` | megatron-bridge 转换 | `:39 HfWeightIteratorBridge` |
| `slime/backends/megatron_utils/megatron_to_hf/__init__.py` | Converter 注册表 | `:17 postprocess_hf_param`, `:24 convert_to_hf`, `:37 _convert_to_hf_core` |
| `slime/backends/megatron_utils/megatron_to_hf/llama.py` | 最简 dense 模型 (53 行) | `:5 convert_llama_to_hf` |
| `slime/backends/megatron_utils/megatron_to_hf/qwen2.py` | qwen2 / qwen3 dense | `:5 convert_qwen2_to_hf` |
| `slime/backends/megatron_utils/megatron_to_hf/qwen3moe.py` | MoE 代表 (117 行) | `:6 convert_qwen3moe_to_hf` |
| `slime/backends/megatron_utils/megatron_to_hf/deepseekv3.py` | 最复杂 (MLA + indexer + MTP) | `:6 convert_deepseekv3_to_hf` |
| `slime/backends/megatron_utils/megatron_to_hf/glm4moe.py` | MoE + 多 layernorm | `:6 convert_glm4moe_to_hf` |
| `slime/backends/megatron_utils/megatron_to_hf/gpt_oss.py` | softmax sinks + expert bias | `:6 convert_gpt_oss_to_hf` |
| `slime/backends/megatron_utils/megatron_to_hf/minimax_m2.py` | `block_sparse_moe` 命名 | `:6 convert_minimax_m2_to_hf` |
| `slime/backends/megatron_utils/megatron_to_hf/qwen3_next.py` | MTP 递归映射 | `:6 _convert_mtp_layer`, `:44 convert_qwen3_next_to_hf` |
| `slime/backends/megatron_utils/megatron_to_hf/processors/__init__.py` | 量化分派器 | `:8 quantize_params` |
| `slime/backends/megatron_utils/megatron_to_hf/processors/padding_remover.py` | vocab padding 裁剪 (12 行) | `:6 remove_padding` |
| `slime/backends/megatron_utils/megatron_to_hf/processors/quantizer_fp8.py` | FP8 blockwise quant | `:10 quantize_params_fp8`, `:94 _quantize_param` |
| `slime/backends/megatron_utils/megatron_to_hf/processors/quantizer_compressed_tensors.py` | int4 AWQ-style packing | `:266 quantize_params_compressed_tensors` |
| `slime/backends/megatron_utils/sglang.py` | SGLang import 防火墙 (44 行) | `:17 DeltaEncoding/DeltaParam/DeltaSpec`, `:29 FlattenedTensorBucket` |

### 扫读文件清单

| 文件 | 用途 |
|---|---|
| `slime/backends/megatron_utils/actor.py:140-159` | Carrier 类选择逻辑 |
| `slime/backends/megatron_utils/actor.py:581-644` | `update_weights()` 调用入口 |
| `slime/backends/megatron_utils/hf_checkpoint_saver.py:22 save_hf_model_to_path` | `UpdateWeightFromDisk` 的实际落盘工具 |
| `slime/utils/arguments.py:129-214` | weight sync 相关 CLI 参数定义 |
| `slime/utils/arguments.py:1697-1745` | weight sync 参数校验 |
| `slime/backends/sglang_utils/sglang_engine.py:264,382,431` | SGLang 接收端的三个 endpoint |
| `tests/test_full_disk_weight_update.py` | E2E smoke test for full disk path |
| `examples/delta_weight_sync/README.md` + `run-glm4.7-355B-A32B-delta.sh` | 355B delta 启动脚本 + W&B 对比 |
| `docs/zh/advanced/delta-weight-sync.md` | Delta 设计动机 + 编码选择 |
| `docs/zh/advanced/external-rollout-engines.md` | External engine + disk sync 部署指南 |
| 其他几个 converter (`glm4.py`, `mimo.py`, `qwen3_5.py`, `qwen3_vl.py`) | 只看了入口签名，结构与 qwen2/qwen3moe 同构 |

### 未读 / 边界外文件

- `slime/backends/megatron_utils/kernels/fp8_kernel.py`（`blockwise_cast_to_fp8_triton` 的实现，是 triton kernel 细节，超出本子系统）
- SGLang 侧的 `_apply_delta_payload`、`load_weights` 等接收端实现（归 backends 子系统）
- `slime/ray/rollout.py` 中 `rollout_manager.get_updatable_engines_and_lock` 的实现（归 Ray 编排子系统）
- `train.py` 中调度 `actor.update_weights` 的循环（归 training 子系统）
