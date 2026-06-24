# 第 10 章：扩展契约——18 个 hook 与契约测试

## slime 第四条赌注的最终落点

slime 的五条核心赌注里，第四条是 **"customization 取代 fork"**——
multi-turn、tool use、sandbox、verifier、multi-agent，这些"agentic
框架"通常会做的事，slime 不在自己内部实现，而是暴露**18 个独立的
hook 点**让用户在 rollout / training 各个阶段注入自己的逻辑。

这一章是这条赌注的最终落点。前面所有章节都在引用某个 hook——第 5
章讲 `--custom-generate-function-path` 让 6 种 rollout 形态成为
不同的用户函数、第 6 章讲 `--data-source-path` 让 Buffer 实现可替
换、第 9 章讲 agent harness 最终通过 `--custom-generate-function-path`
接入。但这些章节都没说清一件事：**slime 的 hook 机制本身长什么
样**？

答案出人意料地简单——**没有 plugin 系统、没有继承式 API、没有
decorator 注册表**，全部通过 18 个 `--xxx-path` CLI 参数指向字符串
路径，在运行时由一个 5 行的 `load_function` 通过 `importlib` 加载。
整个 customization 子系统的加载基础设施就是这 5 行。

这种"hook-by-path"极致简化的代价是：**契约只在运行时验证**——如
果你的函数签名写错了，要等到第一次 rollout step 才会崩。slime 用
一套 `tests/plugin_contracts/` 测试把"约定"机器化：通过环境变量
让**内置 hook 和外部用户 hook 走完全相同的断言**，把 hook-by-path
从"接口约定"升级成"可机器化校验的契约"。

这一章拆这个机制。重点不在"slime 提供了 18 个 hook"（那是
`customization.md` 的事），而在"slime 为什么选 hook-by-path、怎么
让它在生产里 scale"。

## 10.1 一个 5 行的加载器

整个 customization 子系统的加载入口是 `slime/utils/misc.py:37` 的
`load_function`：

```python
# 伪代码 —— illustrative，几乎是原文
def load_function(path):
    module_path, _, attr = path.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, attr)
```

整个 customization 系统的"插件机制"就是这 5 行。它支撑全部 21 个
path hook（`customization.md` 列了 18 个，加上 3 个 `custom-megatron-*`
等于 21 个）。**没有第二个加载入口**。

这种设计意味着用户写一个 hook 时：

```python
# 用户的 my_rewards.py 文件
async def my_custom_rm(args, sample):
    # 完全不需要 import slime 的任何东西
    return compute_my_reward(sample.response, sample.label)
```

然后在命令行加 `--custom-rm-path my_rewards.my_custom_rm` 就完成
接入。**用户的代码完全不需要 import slime 的任何 ABC 或 decorator
就能成立**。你想用 slime 训自己的 reward？不需要继承 `RewardModel`
基类，不需要 `@register_rm("my_rm")` 装饰器，不需要把代码打成 pip
可发布的包——写一个普通函数就行。

为什么 slime 选 hook-by-path 而不是常见的几种扩展模型？读源码能
看出这是个深思熟虑的工程权衡：

| 扩展模型 | 优势 | 代价 |
|---|---|---|
| **plugin 系统**（setuptools entry points） | 自动发现 | 要求用户把代码打成可发布的包，对 RL 用户太重 |
| **继承基类**（要求 subclass 一个 ABC） | 类型安全 | 基类一改用户就崩，slime 不想暴露稳定性义务 |
| **decorator 注册表**（`@register_xx("name")`） | 声明性 | 要求模块被显式 import 才会注册，需要 plugin discovery |
| **hook-by-path**（slime 的选择） | zero-import 接入 | 契约只在运行时验证 |

代价是真实的：用户的函数签名写错了，slime 启动时不报错，要等
`load_function` 被调用时才崩。slime 的解法是把"约定"机器化——见
10.4 节。

但还有个细节值得说：`load_function` **没有 try/except 兜底**。
`importlib.import_module` 失败、`getattr` 找不到属性都会直接抛
异常。这与 slime 整体"配置错误即崩"的哲学一致——宁可在初始化时
崩，也不要在跑了 8 小时之后某个 rollout step 才发现 hook 路径写
错了。第 2 章讲的 `slime_validate_args` 在解析期就把所有跨参数
约束 fail-fast，是同一种态度。

## 10.2 18 个 hook 按 rollout 阶段分 9 组

18 个 hook 不是平铺在一起的杂烩。按 rollout step → reward →
filter → training → log 的时间轴归类，它们自然分成 9 组：

| 组 | hook | 调用位置 |
|---|---|---|
| **(1) rollout 编排** | `rollout-function-path` / `eval-function-path` | rollout 进程入口 |
| **(2) 生成** | `custom-generate-function-path` | per-sample dispatch |
| **(3) 数据源** | `data-source-path` | rollout controller 初始化 |
| **(4) 奖励** | `custom-rm-path`（+ `--group-rm` 切批量） | reward 计算阶段 |
| **(5) 过滤** | `dynamic-sampling-filter-path`、`buffer-filter-path`、`rollout-sample-filter-path`、`rollout-all-samples-process-path` | rollout / buffer 各阶段 |
| **(6) 数据后处理** | `rollout-data-postprocess-path`、`custom-reward-post-process-path`、`custom-convert-samples-to-train-data-path` | rollout 完成后、训练开始前 |
| **(7) 训练 loss** | `custom-loss-function-path`、`custom-advantage-function-path`、`custom-tis-function-path`、`custom-pg-loss-reducer-function-path` | Megatron loss 计算阶段 |
| **(8) Megatron 训练 step** | `custom-megatron-init-path`、`custom-megatron-before-log-prob-hook-path`、`custom-megatron-before-train-step-hook-path` | Megatron 初始化 + 每个 train step 前 |
| **(9) 日志** | `custom-rollout-log-function-path`、`custom-eval-rollout-log-function-path` | metric 上报阶段 |

为什么这么分组？上半部分（组 1-5）由 **rollout 进程**调用，下半
部分（组 6-9）由 **trainer 进程**调用。同一个 `args` 对象通过 Ray
跨进程传递，让两边都能 `load_function(args.xxx_path)` 触发同一个
`importlib.import_module` 路径——**同一份用户代码在两个进程里被
独立加载**。

4 个 filter hook 看起来很多，但它们职责完全不同：

- `dynamic-sampling-filter-path` 决定一个 group 是否**进入训练
  batch**（DAPO 风格：全对或全错丢弃）
- `buffer-filter-path` 决定从 buffer 里**怎么 pop 样本**（默认
  `pop_first` 是 FIFO）
- `rollout-sample-filter-path` 通过修改 `sample.remove_sample`
  决定 sample **是否参与 loss**（但仍参与 advantage normalization）
- `rollout-all-samples-process-path` 是**只读副作用**，可以观察被
  丢弃的样本做日志/统计

这套粒度差异让 RL 训练里的"哪些样本进 batch 哪些不进"决策被精细
拆开——可以一边丢一边记，不互相耦合。

## 10.3 "core 4" 覆盖 80% 场景

18 个 hook 里官方明确推荐的有 4 个，对应 `customization.md` 的
"想做的事 → 应使用的接口"映射表：

| 想做的事 | 应使用的接口 |
|---|---|
| agentic / RAG / tool use | `--custom-generate-function-path` |
| verifier / rule-based / 外部 reward | `--custom-rm-path` |
| DAPO 风格丢 group | `--dynamic-sampling-filter-path` |
| 自定义任务采样 / buffer | `--data-source-path` |
| 替换整套 rollout（**仅在 per-sample 不够用时**） | `--rollout-function-path` |

注意最后一条加了强调——slime **不希望用户一上来就重写整个 rollout
函数**。重写 `rollout-function-path` 意味着失去内置的 partial
rollout、abort、dynamic filter 编排。这套"扩展层级建议"是 slime
给用户的一个梯度——从最局部（`custom_generate` 只改 per-sample
生成逻辑）到最全局（`rollout_function` 接管整个 rollout 编排），
**优先选最局部**。

这条原则在 `rm_hub` 的设计里有个范本实现。`async_rm` 函数（
`slime/rollout/rm_hub/__init__.py:55`）的优先级是：

```python
# 伪代码 —— illustrative
async def async_rm(args, sample):
    # 最高优先级：per-sample（eval 时不同 dataset 不同 reward）
    if hasattr(sample, "custom_rm_path") and sample.custom_rm_path:
        return await load_function(sample.custom_rm_path)(args, sample)

    # 次优先级：全局 CLI 注入
    if args.custom_rm_path:
        return await load_function(args.custom_rm_path)(args, sample)

    # fallback：根据 --rm-type 走内置实现
    rm_type = args.rm_type
    if rm_type == "deepscaler":
        return get_deepscaler_rule_based_reward(sample.response, sample.label)
    elif rm_type == "gpqa":
        return compute_gpqa_reward(sample.response, sample.label)
    # ... 7 种内置 fallback
```

这是教科书级的"内置默认 + 用户覆盖"实现——**内置 reward 实现的
形式是普通 Python 函数（不是类、不是子类）**，与用户自定义的
`custom_rm` 签名完全相同。`rm_hub/deepscaler.py` 的
`get_deepscaler_rule_based_reward(response, label)` 和
`rm_hub/gpqa.py` 的 `compute_gpqa_reward(response, label, metadata=None)`
都是 1-3 个原始入参，外面套一层 `async_rm` 做参数适配。

slime 的内置 reward 和用户 reward 是**同一个接口的不同实例**——
内置的不享有特权，用户的不被歧视。这种平等处理让"我先用内置的
跑通，再逐步替换成自己的"成为零摩擦的迁移路径。

## 10.4 契约测试：把"约定"机器化

hook-by-path 的代价是契约只在运行时验证。slime 用一套 4 个测试
文件解决这个问题（`tests/plugin_contracts/`）：

| 测试文件 | 覆盖 |
|---|---|
| `test_plugin_rollout_contracts.py` | 整个 rollout 函数（最重的一类） |
| `test_plugin_generate_contracts.py` | `custom-generate-function-path` |
| `test_plugin_path_loading_contracts.py` | 7 个"加载即可"的 hook |
| `test_plugin_runtime_hook_contracts.py` | 5 个 runtime hook |

测试通过 `inspect.signature` 断言函数签名、跑一遍返回结构校验。
但最有意思的是它的**双向校验设计**——slime 用环境变量
`SLIME_CONTRACT_*` 让外部用户的实现走完全相同的断言代码：

```python
# 伪代码 —— illustrative，_shared.py 的核心
def run_contract_test_for_file(file_path):
    # 1. 从 sys.argv 解析 --xxx-path 参数
    parser = ArgumentParser()
    parser.add_argument("--rollout-function-path")
    parser.add_argument("--custom-rm-path")
    # ... 其他 path
    args, remaining = parser.parse_known_args()

    # 2. 写到环境变量
    if args.rollout_function_path:
        os.environ["SLIME_CONTRACT_ROLLOUT_FUNCTION_PATH"] = args.rollout_function_path
    # ... 其他 path

    # 3. 把控制权交给 pytest
    raise SystemExit(pytest.main([file_path, *remaining]))


# 测试函数里
def test_rollout_signature():
    path = get_contract_path(
        "ROLLOUT_FUNCTION_PATH",
        default="slime.rollout.sglang_rollout.generate_rollout"  # 内置 fallback
    )
    fn = load_function(path)
    sig = inspect.signature(fn)
    assert list(sig.parameters) == ["args", "rollout_id", "data_source", "evaluation"]
```

这种设计的妙处是**让"测试内置 hook 是否符合契约"和"测试外部用户
hook 是否符合契约"是同一份测试代码**：

- CI 跑这套测试时不传 `--xxx-path`，环境变量为空，`get_contract_path`
  fallback 到内置默认——校验 slime 内置实现自己符合自己的契约
- 用户验证自己的实现时跑
  `python tests/plugin_contracts/test_plugin_rollout_contracts.py
  --rollout-function-path my.func`，环境变量指向用户实现——同一
  份测试代码校验用户实现符合契约

这是把 hook-by-path 从"接口约定"升级成"可机器化校验的契约"的
关键。用户不需要重写测试，slime 不需要为内置和外部实现写两套
测试，**契约只定义一次，验证两边**。

`test_plugin_runtime_hook_contracts.py` 还有一招更聪明的设计——
**调用点字符串作为"反 refactor 守卫"**。每个测试 case 都带一个
`runtime_marker`：

```python
# 伪代码 —— illustrative
HOOK_CASES = [
    HookCase(
        name="custom_reward_post_process",
        runtime_marker="self.custom_reward_post_process_func(self.args, samples)",
        source_file="slime/ray/rollout.py",
    ),
    # ... 其他 case
]

def test_call_site_exists(case):
    source = Path(case.source_file).read_text()
    assert case.runtime_marker in source, \
        f"call site for {case.name} was refactored away"
```

这相当于**用 grep 替代了完整的调用图分析**——非常便宜，但能在
有人把 `self.custom_reward_post_process_func(self.args, samples)`
这一行从 `slime/ray/rollout.py` 里改掉时立刻失败。这是个聪明
的"低成本守护"——与其写一个 AST 分析器，不如假设调用点写法稳定，
让 string match 来守护外部约定不被内部 refactor 破坏。

这种"用便宜技术解决重要问题"的态度在 slime 里反复出现。第 7 章
讲过 `torch.hash_tensor` 用 XOR-reduce 算 hash 替代 CRC——也是
"够用 + 便宜 + GPU-resident"。slime 反复证明"够用的解法不需要 fancy"。

## 10.5 sibling samples：一次 prompt 产出多个训练片段

`custom_generate` 这个 hook 有一个反直觉但非常重要的设计——它的
返回值**可以是 `list[Sample]`**。

为什么需要这个？因为 agentic RL 场景下，一次 prompt 自然会拆成
多个训练片段：

- **subagent 派发**：主 agent 派了一个子 agent 处理 sub-task，子
  agent 的轨迹和主 agent 的后续轨迹都需要参与训练
- **multi-agent**：多个 agent 协作完成同一任务，每个 agent 的
  trajectory 独立训练
- **context compact**：claude-code 等 agent 把上下文压缩成一条
  system 消息再继续，compact 前后的上下文被切成多个 segment

朴素方案是把这些 segment 都"扁平化"成一个长 sample——但这丢失
了 segment 边界，loss 计算会被破坏。slime 让 `custom_generate`
直接返回 `list[Sample]`，每个 segment 是独立的 Sample 对象：

```python
# 伪代码 —— illustrative
async def custom_generate(args, sample, sampling_params):
    # 跑一次 agent rollout，拆出多个训练片段
    segments = await run_agent_and_split_segments(args, sample, sampling_params)

    samples = []
    for segment in segments:
        s = copy.copy(sample)
        s.tokens = segment.tokens
        s.response = segment.response
        s.response_length = segment.response_length
        s.loss_mask = segment.loss_mask
        s.reward = segment.reward
        s.status = Sample.Status.COMPLETED
        # 关键约束：所有 sibling 共享同一个 rollout_id
        s.rollout_id = sample.rollout_id if sample.rollout_id is not None else sample.index
        samples.append(s)
    return samples
```

**关键约束**：这些 sibling samples 必须设置**相同的 `rollout_id`**。
slime 在训练切分和 loss 聚合时把它们当作**同一次 rollout**——避免
把"一次 prompt 产出 4 个 sample"重复计数为"4 次独立 rollout"。

这条约束是第 5 章讲的 `rollout_mask_sums` 设计的具体落点。第 5 章
讲 slime 在 rollout manager（能看到全量 sample 的位置）提前算好
每个 rollout 的 total mask sum，复制给该 rollout 的每个 sample——
这里"每个 rollout"靠的就是 `rollout_id` 来识别。sibling samples
共享 `rollout_id`，就被 manager 当成同一次 rollout 算分母。

测试 `test_plugin_generate_contracts.py` 有专门的
`test_generate_and_rm_group_rm_accepts_list_result_from_custom_generate`
（行 176-198）守住这条契约——sibling 返回 `list[Sample]` 且共享
`rollout_id` 必须正确工作，破坏这条约定的修改不能合入。

## 10.6 两个"非 -path 但概念上是 hook"的特殊成员

18 个 hook 里有两个不是 `--xxx-path` 形态，但概念上是扩展点：

**`--group-rm` 切换 reward 形态**。这是个开关而不是 path，但它让
`--custom-rm-path` 指向的函数可以是两种不同签名：

```python
# 单样本模式（默认）
async def custom_rm(args, sample) -> float:
    return reward_for_one_sample(sample)

# 批量模式（--group-rm 开启时）
async def batched_custom_rm(args, samples: list[Sample]) -> list[float]:
    return rewards_with_group_context(samples)
```

为什么需要批量模式？因为有一类 reward 本质上是 group 内对比——
"哪个回答比其他更好"的 pairwise 排序、或者群体一致性指标。这些
reward 单看一个 sample 算不出，必须看整个 group。`generate_and_rm_group`
会延迟到 group 集齐后再统一调用 `batched_async_rm`。

这是 18 个 hook 里唯一一对"signatures 不同但参数 path 共用"的
hook——靠 `--group-rm` 开关切换形态。这种"用开关切签名"的设计
不常见，但对这个特定场景是合理的妥协。

**MoE routing replay 是个"非 -path 但概念上是 hook"的特殊成员**。
它不通过 `--xxx-path` 注入，而是通过 `--use-routing-replay` /
`--use-rollout-routing-replay` 开关 + monkey-patch `compute_topk`
（`routing_replay.py:57-82`）。第 8 章讲过它用环境变量
`ROUTING_REPLAY_STAGE` 做四态状态机切换。

为什么这个 hook 是特殊形态？因为它需要在 **MoE expert routing 这种
深度内嵌的 forward 路径**上插桩——用户代码无论如何写不出来，必须
slime 自己提供。它属于"内置 hook 但通过 env var 与开关串联"的特
殊形态，是 slime 给"训练-推理 MoE routing 一致性"留的扩展位。

这两个特殊成员告诉我们：**"hook" 这个概念在 slime 里不是单一形态**。
绝大多数 hook 是 `--xxx-path`，但形态适配是被具体场景决定的——
`--group-rm` 是因为签名要切换、routing replay 是因为注入点在上游
深处。slime 没有强迫所有扩展点走同一形态，而是为每种场景挑最合适
的形态。

> **深入剖析：per-sample hook 只在 eval 上下文存在**
>
> 仔细读 `rm_hub/__init__.py:55` 的 `async_rm` 函数，会发现一个
> 微妙的设计——**reward 优先级最高的是 `sample.custom_rm_path`
> 这个 per-sample 属性**，比 `args.custom_rm_path` 这个全局 CLI
> 还高。但这个 per-sample 属性只在 eval 上下文里被设置。
>
> 看 `slime/utils/eval_config.py`：
>
> ```python
> # 伪代码 —— illustrative
> @dataclass
> class EvalDatasetConfig:
>     name: str
>     prompt_data: str
>     custom_rm_path: str | None = None              # per-dataset reward
>     custom_generate_function_path: str | None = None  # per-dataset generate
>     # ...
> ```
>
> 每个 eval dataset 可以单独指定 `custom_rm_path` 和
> `custom_generate_function_path`，这两个属性会被挂到该 dataset
> 的 sample 上，覆盖全局 `args.*_path`。
>
> 这是个非常实用的设计——eval 时不同 dataset 可以用不同的 verifier：
> 主任务用 GRPO 训练，eval 时跑 IFBench / GPQA / math 三个 dataset
> 各自的 verifier 算 reward。如果没有 per-sample 优先级，你只能
> 全局用一个 reward 函数算所有 dataset 的 metric。
>
> 为什么训练时不暴露 per-sample hook？因为训练时 reward 一致性
> 很重要——同一个 prompt 在不同 rollout step 用不同 reward 会让
> 训练信号混乱。eval 是只读操作，per-dataset 切换完全安全。
>
> 这是 slime "权限按层切开"的另一个例子（第 6 章讲过 `get_samples`
> 方法引用切分读写权限）——eval 是观察，可以灵活配置；training
> 是写入，必须保持一致。

## Apply This

5 条可迁移到自己框架扩展系统的设计模式：

**1. 用字符串路径替代 plugin 系统**

slime 的 `load_function` 5 行支撑全部 21 个 hook。用户代码完全不
需要 import 框架的任何东西就能接入——比 plugin discovery 简单得多，
比 ABC 继承灵活得多。代价是契约只在运行时验证，但这可以用契约测试
弥补。

**怎么改造适配**：你的框架有扩展点吗？看看能不能把它做成
`--xxx-path module.fn` 形式。让用户写普通函数 + CLI 参数指向它，
零 import 接入。

**陷阱**：字符串路径加载没有 IDE 类型检查。slime 接受这个代价——
目标用户是"知道自己在配什么"的 RL 工程师，不是需要 IDE 提示的
新手。如果你的用户群体不一样，可能需要 Protocol 或 ABC 补足。

**2. 契约测试用环境变量让内置与外部走同一断言**

slime 的 `SLIME_CONTRACT_*` 设计让"测试内置 hook"和"测试外部用户
hook"是同一份测试代码。CI 跑 fallback 到内置；用户跑指向自己实现。
这把 hook-by-path 从约定升级成机器化契约。

**怎么改造适配**：你的框架有用户可替换的接口吗？写一套断言测试，
让接口路径通过环境变量传入。CI 默认验证内置实现，用户用同一套
测试验证自己的实现。

**陷阱**：环境变量解析要早于 pytest 加载——slime 在自己的
`run_contract_test_for_file` 里手动 parse `argv` 后才 `pytest.main()`。
直接用 pytest fixture 不行，因为 fixture 是测试运行时才执行。

**3. 调用点字符串作为"反 refactor 守卫"**

slime 用 `runtime_marker` 字符串 + 源文件 `read_text` + `assert
marker in source` 来守护调用点。比写 AST 分析器便宜得多，但能在
有人 refactor 掉调用点时立刻失败。

**怎么改造适配**：你的框架有"必须从这个位置调用某个 hook"的约定
吗？写一个测试，用 grep 守住这个调用点。如果以后调用点变了，测试
失败提醒你"这是外部约定，不要随便改"。

**陷阱**：调用点字符串要选得精确——不能是
`self.custom_func(...)` 这种太宽泛的写法（容易撞到无关代码），
也不能太精确（一个空格变化就破）。slime 用完整的调用语句包括
参数名，平衡度合理。

**4. 提供"core 4"扩展层级建议而不是只 ship 接口**

slime 不只 ship 18 个 hook，还在文档里明确推荐"core 4"覆盖 80%
场景，并强调"重写 `rollout-function-path` 仅在 per-sample 不够用
时"。这避免了用户一上来就重写最重的接口、丢失内置编排逻辑。

**怎么改造适配**：你的框架有大量扩展点吗？画一张"从最局部到最全局"
的扩展层级图，标出"优先用哪个"。让用户先选最局部的扩展，避免他们
为了一个小定制重写整个流程。

**陷阱**：层级建议要根据真实使用场景演进。slime 的"core 4"是经过
社区反馈后才稳定下来的——最早可能不是这 4 个。你的层级建议也要
有版本化，每年回顾一次哪些 hook 实际被频繁使用、哪些是冷门。

**5. 让"读路径"灵活、"写路径"保持一致**

slime 在 eval 上下文暴露 per-sample `custom_rm_path` 让不同 dataset
用不同 verifier；但 training 上下文不暴露这个——训练时 reward
一致性比灵活性更重要。

**怎么改造适配**：你的系统里有"观察"和"修改"两类操作吗？观察类
操作可以接受灵活的 per-sample 配置；修改类操作应该限制成全局一致。
这种"权限按操作类型分层"的设计能避免一致性 bug。

**陷阱**：权限分层要明确告诉用户——slime 在 `EvalDatasetConfig`
的文档里说清"这些 path 只在 eval 时生效"。否则用户配了 per-sample
training reward 发现没生效，会很困惑。

---

## 下一站

到这里 slime 五条核心赌注的最后一条（customization 取代 fork）也
讲完了。从第 1 章导览开始，前 10 章覆盖了 slime 的核心抽象——主
循环、配置、运行时、训练 step、rollout、Buffer、weight sync、部署、
agent、扩展契约。下一章我们换个视角，从"工程基础设施"角度看 slime
怎么保证这套系统在生产里**真的能跑**——可观测、debug、CI、容错、
reproducibility。这一章对应 slime 第五条赌注："explicit dataflow +
CI 一等公民"。
