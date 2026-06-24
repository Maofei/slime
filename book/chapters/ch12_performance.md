# 第 12 章：性能与正确性——混合精度、CP 不变量与瓶颈分析

## 把设计赌注具象到具体数字

这是全书最后一章核心内容章节（下一章是结语）。前 11 章讲了 slime
的五条核心赌注——主循环作为契约、native engine 透传、单一 SGLang
backend、customization 取代 fork、explicit dataflow + CI 一等公民。
每一条都对应一组章节展开了它的具体技术。

这一章视角不同。它不引入新的设计赌注，而是把前面 11 章里散布的
**性能与正确性话题**集中到一处看：

- 第 4 章讲了 4 处梯度放缩组合出 per-rollout-mean，但没展开 CP
  不变量在 cp_utils.py 里具体怎么实现
- 第 7 章讲了 FP8 在 send-side 量化、int4 在 receive-side packing，
  但没展开整个 BF16 训练 + FP8 rollout + FP8 KV cache 这套"组合拳"
  在生产部署里怎么协调
- 第 11 章讲了 `test_loss_cp_invariance.py` 用 CPU spawn 守住
  grad norm，但没展开 cispo loss、chunked GAE 这些算法层的不变量

这些话题在前面章节里都是"被引用"的——但它们值得自己被讲清楚一次，
因为它们决定了 slime 实际能在生产里跑多快、跑多稳。**前面是"设计
赌注"，这一章是"具体数字"**。

读完这一章你应该能回答：slime 推荐的生产精度配置是什么？为什么这
样配？CP 不变量在代码里靠什么守住？cispo / chunked GAE 这些算法
不变量的具体边界在哪？RL 训练的瓶颈通常出在哪一环？

## 12.1 BF16 训练 + FP8 rollout + FP8 KV cache 的组合拳

slime 在 `docs/zh/advanced/low-precision.md` 里把生产推荐路径写得
非常明确：

> **Megatron BF16 训练 + SGLang FP8 rollout/inference**

这是 slime 大规模 MoE RL 的**默认推荐**。看起来简单，背后是个三层
精度的组合拳：

| 层 | 精度 | 理由 |
|---|---|---|
| **训练（Megatron）** | BF16 | 数值稳定，optimizer state 必须高精度 |
| **rollout（SGLang weights）** | FP8 blockwise | 显存减半，wire 带宽减半 |
| **KV cache** | FP8 e4m3（可选） | long-context / agentic workload 下提升有效 cache 容量 |

三层之间通过几个关键机制协调起来：

**训练 → rollout 的转换在 weight sync 阶段做**。第 7 章讲过：
slime 把 BF16 权重在**发送前**就量化成 FP8（用 deepgemm 的 ue8m0
变体，与 SGLang 内部使用的量化算子比特级一致），wire 上传的就是
FP8 + scale。这意味着 rollout 拿到的权重直接就是 FP8 格式，**不
需要二次转换**——避免了一次量化误差。

配套工具是 `tools/convert_hf_to_fp8.py`：

```bash
# 伪代码 —— illustrative
python tools/convert_hf_to_fp8.py \
    --model-dir $BF16_MODEL \
    --save-dir $FP8_MODEL \
    --strategy block --block-size 128 128 \
    --max-workers 4
```

它把 BF16 HF checkpoint 一次性转成 FP8 blockwise 格式，写到
`config.json` 的 `quantization_config` 字段里。slime 启动时
`--hf-checkpoint $FP8_MODEL` 一指，自动走 FP8 rollout 路径。

**KV cache FP8 是 rollout 侧独立配置**：

```bash
--sglang-kv-cache-dtype fp8_e4m3
```

这是第 2 章讲的 `--sglang-` 透传机制的直接受益——slime 不需要为
KV cache 精度写任何代码，SGLang 上游加什么字段 slime 当天就能用。
这个参数的实际收益是**提升 SGLang 的有效 KV cache 容量**——意味
着相同显存下可以跑更长 context 或更高并发。

为什么这套组合是"推荐的"而不是"必须的"？因为它有具体代价：

- BF16 训练比 FP8 训练慢（cuBLAS FP8 GEMM 比 BF16 GEMM 快约 1.5-2x）
- FP8 rollout 在某些 GPU 上的 GEMM 支持还在演进，性能可能不如理论值
- FP8 KV cache 在某些 SGLang 版本里有数值漂移的边界情况

slime 选择推荐 "BF16 训练 + FP8 rollout" 而不是 "FP8 训练 + FP8
rollout"，是因为**训练的数值稳定性比 rollout 重要**。rollout 是只
读路径（推理），就算有点精度损失也只是采样略不准；训练是写路径
（梯度反传），精度损失会累积。这套分层精度本质上是"在不影响训练
稳定性的前提下最大化 rollout 吞吐"。

## 12.2 INT4 与 FP8 训练：明确写出实验性边界

slime 还支持两条更激进的精度路径——INT4 QAT 训练和 FP8 训练。但
它们都被明确标注为 **beta 或 experimental**，且文档里**直接列出
已知冲突**。

**INT4 QAT**。`slime/backends/megatron_utils/kernels/int4_qat/` 目录
下有完整的 INT4 STE（Straight-Through Estimator）实现。但启用它的
方式很有意思——不是 CLI 参数，是**环境变量**：

```python
# 伪代码 —— illustrative，启动脚本里
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"OPEN_TRAINING_INT4_FAKE_QAT_FLAG\": \"1\",
    \"OPEN_TRAINING_INT4_GROUP_SIZE\": \"128\"
  }
}"
```

为什么用环境变量而不是 CLI？因为 INT4 QAT 是个**全局开关**——影响
Megatron 内部 Linear 层的 forward 行为，不是某个具体 hook 能控制
的。这与第 8 章讲过的 RoutingReplay 用环境变量驱动状态机是同一
个模式——**当 hook 点深埋在上游内部时，环境变量是合法的反向通道**。

**FP8 训练** 更激进——TransformerEngine 的 Linear 和 GroupLinear
层在 FP8 context 中构建，forward/backward GEMM 走 cuBLAS FP8 GEMM。
但它有一个**明确的已知冲突**：

> `--fp8-param-gather` 可以节省显存，但目前需要 TransformerEngine
> `FusedAdam`，这与大规模 Megatron-LM RL 中常用的 CPU Adam offload
> 路径冲突。

这种"我有这个功能，但它跟某个常用路径冲突"的诚实在 slime 里很常见。
slime 不会假装所有功能都互相兼容——这条冲突是真实的（FusedAdam
在 GPU 上跑、CPU Adam 在 CPU 上跑，两者无法共用 optimizer state），
slime 选择**直接写出来**而不是埋在 commit message 或 issue
tracker 里。

这种态度的好处是：用户在调研某个 feature 时一眼能看到它能不能用。
**坏的文档让用户踩坑 3 天后才发现"原来不能跟我现在的路径配合"——
诚实的文档把这 3 天还给用户**。第 8 章讲的 external engine
明确写出 fault tolerance 兼容性、第 11 章讲的 reproducibility
明确写出 5 个非确定性源头——这些都是同一种态度的体现。

## 12.3 CP 不变量在代码里靠什么守住

第 4 章讲过那 4 处梯度放缩组合出 per-rollout-mean，最后一步是
"DDP grad reduce × 1 / (dp × cp)" 让 CP 反向消除。第 11 章讲过
`test_loss_cp_invariance.py` 在 CPU 上跑真 backward 守住 grad norm
不变。这一节看 CP 不变量在 `cp_utils.py` 里**具体怎么实现**。

CP（context parallel）是 Megatron 的一种并行策略——把单条序列在
sequence 维度切分到多个 GPU 上算 attention。它和 DP / TP / PP 不同
之处在于：**CP 切分的是同一条序列**，每个 CP rank 看到的是这条
序列的一段。这意味着 loss 计算时，如果朴素地"每个 CP rank 算自己
那段的 loss 再 all_reduce"，会得到错误的结果——因为同一条序列的
loss mask sum 被算了 `cp_size` 次。

slime 的解决方法是 `get_sum_of_sample_mean` 这个函数（
`cp_utils.py:47`）。它**返回一个 closure**，对每个样本按 loss_mask
加权求平均：

```python
# 伪代码 —— illustrative
def get_sum_of_sample_mean(loss_masks):
    cp_size = mpu.get_context_parallel_world_size()
    if cp_size == 1:
        # 单 GPU 路径：直接算每个样本的 mean loss
        sample_masks = compute_per_sample_masks(loss_masks)
        def closure(values):
            return (values * sample_masks).sum()
        return closure
    else:
        # CP > 1 路径：需要跨 CP rank all_reduce 后再算
        cp_group = mpu.get_context_parallel_group()
        sample_masks = compute_per_sample_masks_with_cp(loss_masks, cp_size)
        def closure(values):
            local = (values * sample_masks).sum()
            dist.all_reduce(local, group=cp_group)
            return local / cp_size
        return closure
```

关键设计是 **CP=1 和 CP>1 走两条不同分支**——这不是为了优化，是为了
**让 CP=1 的代码路径完全不引入 all_reduce overhead**。slime 不强求
所有路径走同一套抽象，而是承认 CP=1 是常见配置（debug、单节点测试、
小模型训练），值得专门优化。

CP 几何变换也有讲究。slime 的 `slice_with_cp` 不是简单二分，而是
**zigzag 切分**：

```python
# 伪代码 —— illustrative
def slice_with_cp(tensor, cp_rank, cp_size):
    # cp_rank 拿第 cp_rank 块和第 (2*cp_size - cp_rank - 1) 块
    # 不是简单的 tensor.chunk(cp_size)[cp_rank]
    chunk_a = tensor[cp_rank * chunk_size : (cp_rank + 1) * chunk_size]
    chunk_b = tensor[(2 * cp_size - cp_rank - 1) * chunk_size :
                     (2 * cp_size - cp_rank) * chunk_size]
    return torch.cat([chunk_a, chunk_b], dim=0)
```

为什么 zigzag？因为 RingAttention 在 causal mask 下，**前面的位置
和后面的位置计算量不一样**——后面的位置要 attend 到更多前面的
positions。简单二分会让前一半的 GPU 闲、后一半的 GPU 忙；zigzag
把"前段 + 后段"配对，让每个 GPU 拿到等量的工作。

这种"几何细节决定性能"的设计被 `test_cp_utils.py` 守住——所有
`slice_with_cp` / `all_gather_with_cp` 的不变量必须严格满足，改
`cp_utils.py` 不跑这个测试，CP 路径直接乱套。

## 12.4 cispo loss：用 grad 流向作 stop-gradient 断言

slime 的 `loss.py` 实现了多种 RL loss——PPO、GRPO、GSPO、CISPO、
custom loss。这一节挑 **CISPO**（Clipped IS-weight Policy Gradient
with Off-policy，MiniMax-M1 提出，arXiv:2506.13585）讲一个具体的
正确性边界。

PPO 经典 loss 是：

```python
# 伪代码 —— illustrative
ratio = exp(log_prob_new - log_prob_old)  # IS ratio
clipped_ratio = clamp(ratio, 1 - eps_clip, 1 + eps_clip_high)
pg_loss = -min(ratio * advantage, clipped_ratio * advantage)
```

这里 `ratio` 参与反传——梯度会通过 `ratio` 流回 `log_prob_new`。
但 CISPO 改了一处：**clipped IS ratio 必须 stop-gradient**：

```python
# 伪代码 —— illustrative
def compute_cispo_loss(ppo_kl, log_probs, advantages, eps_clip, eps_clip_high):
    ratio = exp(-ppo_kl)
    clipped_ratio = clamp(ratio, 1 - eps_clip, 1 + eps_clip_high)
    clipped_ratio = clipped_ratio.detach()   # 关键：stop-gradient

    # 梯度只通过 log_probs，不通过 ratio
    pg_loss = -clipped_ratio * advantages * log_probs
    return pg_loss
```

`detach()` 让 `clipped_ratio` 在反传时被当作常数——梯度只通过
`log_probs` 流回模型，不通过 `ratio`。这是 CISPO 的核心数学性质，
论文用它来证明 off-policy correction 的收敛性。

这种"某个张量必须 stop-gradient"的约束怎么测？看 `test_cispo_loss.py`
的第二个测试函数——它用了一个很巧妙的设计：**通过梯度是否流过来
判定 stop-gradient 是否正确**。

```python
# 伪代码 —— illustrative，test_cispo_loss.py:34
def test_compute_cispo_loss_gradient_flows_only_through_log_probs(...):
    log_ratios = torch.tensor([math.log(r) for r in ratios], requires_grad=True)
    ppo_kl = -log_ratios
    log_probs = LOG_PROBS.clone().requires_grad_()

    pg_losses, _ = compute_cispo_loss(ppo_kl, log_probs, ...)
    pg_losses.sum().backward()

    # log_probs 必须有梯度（这是 CISPO 想要的）
    torch.testing.assert_close(log_probs.grad, -clamped * ADVANTAGES * ..., ...)

    # log_ratios 必须没梯度或者梯度全为 0（这是 CISPO 的 stop-gradient 约束）
    assert log_ratios.grad is None or torch.all(log_ratios.grad == 0), \
        f"CISPO must stop-gradient on the IS ratio; log_ratios.grad={log_ratios.grad}"
```

这个测试不测输出值——它测**梯度的传播路径**。如果 CISPO 实现失败
stop-gradient（比如忘了 `.detach()`），backward 会让 `log_ratios.grad`
非零，断言失败。如果 stop-gradient 正确，`log_ratios.grad` 要么是
`None`（PyTorch 没分配 grad tensor）要么全为 0（梯度流到这里就停了）。

这是个非常 slime 风格的测试设计——**用 PyTorch autograd 的实际
行为作为契约的最终裁判**，而不是依赖代码 review 检查"开发者有没有
写 `.detach()`"。哪天有人 refactor 把 `.detach()` 改掉了，测试
立刻失败。

第 10 章讲过 `runtime_marker` 用 grep 守住调用点不被 refactor 破坏——
这里的"用 grad 流向守住 stop-gradient"是同一种工程哲学：**用机器
化的检查替代人工 review**。

## 12.5 chunked GAE：并行 scan 与数值一致

GAE（Generalized Advantage Estimation）是 PPO 算 advantage 的标准
方法。朴素实现是**batch-serial**——对每个 sample 循环 T 步：

```python
# 伪代码 —— illustrative，vanilla_gae
def vanilla_gae(rewards, values, gamma, lam):
    B, T = rewards.shape
    advantages = torch.zeros_like(rewards)
    gae = 0
    for t in reversed(range(T)):
        next_value = values[:, t+1] if t < T-1 else 0
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        gae = delta + gamma * lam * gae
        advantages[:, t] = gae
    return advantages, advantages + values
```

这套实现 T=4096 时还快，T=128K（long-context agentic rollout）就
变慢了——T 维循环不能并行。

slime 的 `chunked_gae` 是个**并行 scan** 实现——把 T 维切成
`chunk_size` 块，块内并行算，块间串行组合：

```python
# 伪代码 —— illustrative，chunked_gae
def chunked_gae(rewards, values, gamma, lam, chunk_size):
    B, T = rewards.shape
    # 切成 chunk_size 大小的块
    chunks = T // chunk_size
    # 每块内部用 prefix scan 算 advantage（GPU 并行）
    # 块间用残差累加，O(log chunks) depth
    # ... 详细实现略
    return advantages, returns
```

`chunked_gae` 的关键约束是 **必须与 vanilla_gae 数值一致**——不是
"接近"，是 `atol=1e-5`。这是性能优化里的硬不变量：你优化得再快，
如果改变了 advantage 数字，整套 RL 训练就跑偏了。

`test_chunked_gae.py` 用 `(B, T, chunk_size)` 三个维度参数化测试：

```python
# (B, T): (16, 4096) / (32, 8192) / (256, 128*1024)
# chunk_size: 64 / 128 / 256
```

每个组合都跑 vanilla_gae 和 chunked_gae 对比，断言
`abs(adv_serial - adv_parallel).max() < 1e-5`。注意它还打印 speedup
ratio——这是 slime 关心"性能优化 + 数值正确"两件事的具体证据。

这种"性能优化必须有数值一致性测试守住"的工程文化在 slime 里反复
出现。`_apply_delta_payload`（第 7 章 weight sync delta）是
bit-exact、`reduce_train_step_metrics`（第 11 章）守住两条 metric
管线给同一数字、`chunked_gae` 守住与 vanilla_gae atol=1e-5——
slime 不接受"为了快牺牲正确"的 tradeoff，性能优化和正确性是
共存的硬约束。

## 12.6 rollout long-tail 是 RL 训练的最大瓶颈

最后看 RL 训练的**瓶颈分析**。这个话题 slime 自己没专门写文档，
但从代码设计倒推能看出团队的认知。

直觉上 RL 训练的瓶颈应该是 **GPU 训练 step**——backward + optimizer
update 是最重的计算。但实际生产中，**rollout 通常占 RL 训练时间
的 80-90%**——APRIL 项目（被 slime README 引用的外部参考）直接
点名："rollout generation 通常会消耗 RL 训练 90% 以上的时间"。

为什么？因为 rollout 是 **autoregressive 的——每个 token 必须等
上一个 token**。即使 SGLang 极致优化了 throughput，单条样本的
latency 还是被序列长度乘以单 token 延迟决定。一批 sample 里只要
有一条特别长（长尾），整个 rollout 就被它卡住。

slime 几个核心设计都是为了解决这个 long-tail 问题：

| 设计 | 解决什么 |
|---|---|
| **partial rollout**（第 5 章） | weight update 时主动 abort 长尾样本，下一轮接着算（流式 rollout 保证不浪费 token） |
| **fully_async rollout**（第 5 章） | 长尾样本不阻塞下一个 rollout 开始，跨 rollout 边界保留 in-flight |
| **abort + 流式**（第 5 章） | abort 时已生成 token 不丢，让中断成为"断点"而不是"损失" |
| **dynamic batching**（第 4 章） | 在训练侧做 dp_schedule，让长样本不让短样本等 |
| **multi-turn prefix cache**（第 8 章） | agent 跨轮复用 KV cache，少重新 prefill 长 history |

每一个都是针对 rollout long-tail 的具体应对。slime 不假设 "rollout
能跑得均匀"——它默认 rollout 是 long-tail 的，所有设计都围绕这个
现实展开。

**weight sync 的占比通常 < 10%**。第 7 章讲了四条传输路径，但实际
生产里 weight sync 不是大头——一次 sync 几秒到几十秒，相对几分钟
的 rollout 占比小。slime 在 weight sync 上花的精力主要是为了**解锁
部署形态**（跨厂商 GPU、跨数据中心），不是为了挤性能。

**CP overhead 看 batch 是否均匀**。CP 切分 + 各 rank all_reduce
有固定开销，但 dp_schedule（第 4 章）已经把 sequence length 均匀
分到每个 dp rank。只要 batch 内 sequence length 分布合理，CP
overhead 在 5% 以内。如果 batch 高度不均（一条 100K + 31 条 1K），
就算 dp_schedule 也救不了——这种场景下短样本的 GPU 在等长样本
完成，整体利用率低。

这种"瓶颈在 rollout 不在训练"的认知决定了 slime 的工程优先级——
它在 rollout manager（1485 行）上花的代码比训练 actor（128 行
`train_actor.py`）多 10 倍以上。

> **深入剖析：用 PyTorch autograd 的实际行为作为契约的最终裁判**
>
> `test_cispo_loss.py:46-48` 那行 assert 值得单独讲：
>
> ```python
> assert log_ratios.grad is None or torch.all(
>     log_ratios.grad == 0
> ), f"CISPO must stop-gradient on the IS ratio; log_ratios.grad={log_ratios.grad}"
> ```
>
> 这是个非常巧妙的测试设计——**测的不是输出值，是梯度的传播路径**。
> CISPO 的核心数学性质是 "clipped IS ratio 必须 stop-gradient"，
> 而 PyTorch autograd 是这个性质的最终裁判：
>
> - 如果实现正确（用了 `.detach()`），autograd 在反传时遇到 detach
>   就停，`log_ratios` 拿不到梯度
> - 如果实现失败（忘了 `.detach()`），autograd 会把梯度一路传过去，
>   `log_ratios.grad` 会有非零值
>
> 这种测试设计的核心洞察是：**stop-gradient 是个"看不见"的属性，
> 但 autograd 能"看见"它**。你不需要去 review 代码检查 `.detach()`
> 写了没——让 autograd 跑一遍，看梯度有没有泄漏到不该泄漏的位置。
>
> 这是个可以广泛迁移的测试模式。任何依赖 PyTorch autograd 行为的
> 约束（stop-gradient、no_grad context、register_hook、detach 等），
> 都可以用类似的"测梯度流向"方式守住。比单纯断言输出值更可靠，
> 因为输出值偶尔会因为浮点误差通过断言，但梯度路径要么有要么没有。

## Apply This

5 条可迁移到自己 RL 或大规模训练系统的设计模式：

**1. 精度配置按角色分层**

slime 的"BF16 训练 + FP8 rollout + FP8 KV cache"是三层独立配置——
训练要稳定所以 BF16，rollout 要省显存所以 FP8，KV cache 要扩容量
所以 FP8 e4m3。不要用一个总精度开关配置所有层，让每一层根据自己
的 workload 特征选合适的精度。

**怎么改造适配**：你的系统里有几种独立的"计算 workload"？（训练 /
rollout / eval / serving）每种的精度需求可能不一样——梯度敏感的
用高精度，吞吐敏感的用低精度。把精度从全局开关拆成 per-workload
配置。

**陷阱**：跨层精度转换要有明确的协议。slime 在 weight sync 时把
BF16 量化成 FP8 用 deepgemm 的 ue8m0 算子，与 SGLang 内部使用的
算子比特级一致——避免了二次量化误差。如果你的转换协议不够精确，
跨层精度的好处会被转换误差吃掉。

**2. 实验性功能明确写出已知冲突**

slime 的 `--fp8-param-gather` 文档直接写"与 CPU Adam offload 冲突"。
这种诚实让用户在调研 feature 时一眼能看到能不能用，避免踩坑 3 天
后才发现。

**怎么改造适配**：你的项目里有 beta / experimental 功能吗？写一段
"已知限制"明确列出它和哪些常用路径冲突——不要埋在 issue tracker
里。诚实的文档把"踩坑时间"还给用户。

**陷阱**：已知冲突要随版本更新——slime 的 `--fp8-param-gather`
现在与 CPU Adam 冲突，将来可能修复。冲突列表要有 owner 定期回顾，
避免文档过时。

**3. 性能优化必须有数值一致性测试守住**

slime 的 `chunked_gae` 必须与 `vanilla_gae` 数值一致（atol=1e-5），
`_apply_delta_payload` 是 bit-exact，`reduce_train_step_metrics`
两条管线给同一数字——slime 不接受"为了快牺牲正确"的 tradeoff。

**怎么改造适配**：你的系统里有"性能优化版"和"参考实现"两套代码
吗？写一个测试用参数化覆盖 `(B, T, chunk_size)` 等关键维度，断言
两套实现数值一致。优化版改了算法，测试自动验证它没改语义。

**陷阱**：数值一致性的 atol 要选对——太严格（atol=1e-10）会被浮点
误差搅黄，太松（atol=1e-2）守不住实际 bug。slime 的 `atol=1e-5`
是经验值，对应"float32 的有效精度大约 7 位"。

**4. 用 grad 流向作 stop-gradient 测试断言**

`test_cispo_loss.py` 用 `assert log_ratios.grad is None or torch.all(
log_ratios.grad == 0)` 测 stop-gradient——让 PyTorch autograd 跑
一遍，看梯度有没有泄漏到不该泄漏的位置。比 review 代码检查
`.detach()` 写了没更可靠。

**怎么改造适配**：你的代码里有 stop-gradient / no_grad / detach
这类"看不见但要守"的约束吗？写一个测试触发 backward，检查
`tensor.grad` 是 None 或 0——这是 autograd 给你的最直接的契约
检查。

**陷阱**：测试要确保 backward 真的能执行——`tensor.requires_grad_()`
要正确设置，loss 要能 `.sum().backward()`。如果 backward 失败，
`tensor.grad` 是 None 不是因为 stop-gradient，是因为根本没反传。

**5. 优化瓶颈环节而不是整体——rollout long-tail 是 RL 的瓶颈**

slime 在 rollout manager 上花的代码比训练 actor 多 10 倍以上，
因为它知道 rollout 占 RL 训练时间的 80-90%。一上来优化训练 step
是错的方向——优化 rollout long-tail 才有 ROI。

**怎么改造适配**：先 profile 你的系统，找出**最大单一时间消耗者**。
然后所有优化精力先投这一项。RL 训练里通常是 rollout long-tail，
其他系统可能是别的——但"先找瓶颈再优化"的原则普适。

**陷阱**：瓶颈会随负载变化——slime 的 long-tail 优化在 long-context
agentic rollout 下效果显著，在 short-prompt SFT 场景下反而是
weight sync 占大头。优化要针对具体 workload，不要在 benchmark 上
追求"通用最优"。

---

## 下一站

到这里 12 章核心内容章节都讲完了。这一章把前面 11 章里散布的性能
与正确性话题集中讲完——精度组合拳、CP 不变量、cispo / chunked
GAE 这类算法不变量、rollout long-tail 瓶颈。下一章是结语，把全书
12 章的核心洞察压缩成一组**可迁移的如果我要从零做 RL 框架，我会
做哪些和 slime 一样的决定、哪些不一样**的清单。
