# Data Buffer 研究笔记

## TL;DR

slime 里没有一个独立的"Data Buffer"模块——这个词描述的是一组**契约**而不是一段代码：训练侧需要从某处拿到 `list[list[Sample]]`，rollout 侧需要把生成完的 sample 放进同一个地方，断点续训需要它能 save/load。slime 用 `DataSource` 抽象基类（`slime/rollout/data_source.py:17`）固定这份契约，用 `RolloutDataSourceWithBuffer`（同文件 `:168`）给出"dataset + 内存 list"的默认实现，把更复杂的"全异步 HTTP buffer"作为外置参考实现扔到 `slime_plugins/rollout_buffer/`。Dynamic filter 也不属于 buffer，而是 rollout 主循环里独立调用的 hook（`slime/rollout/sglang_rollout.py:441`），这让 buffer 保持薄并把"选哪些样本进 batch"的决策开放给用户。整体设计可以一句话概括为：**接口中心化、实现去中心化、过滤逻辑外置化**。

## 1. 架构与模块边界

### 1.1 为什么 Buffer 没有独立目录

直觉上"训练前的数据中转站"听起来应该是个一级子系统，理论上可以期待 `slime/data_buffer/` 这种目录。但事实上 slime 的 Buffer 概念分散在三个层次：

1. **契约层**：`slime/rollout/data_source.py:17` 的 `DataSource(abc.ABC)`，定义 `get_samples / add_samples / save / load / __len__` 五个方法。
2. **默认实现层**：同文件 `:50` 的 `RolloutDataSource`（只读 dataset）与 `:168` 的 `RolloutDataSourceWithBuffer`（dataset + 内存 buffer，"buffer"是个 Python `list[list[Sample]]`）。
3. **外置参考实现层**：`slime_plugins/rollout_buffer/buffer.py` 提供了一个独立 FastAPI server，演示"agent 异步生成 → HTTP 写 buffer → 训练侧 HTTP 拉取"。它并不是上面任何一个 `DataSource` 子类，而是通过 `slime_plugins/rollout_buffer/rollout_buffer_example.py` 里的 `generate_rollout` 函数与 `RolloutDataSourceWithBuffer` 接驳。

这种分层使得"buffer"既是 RL 论文上 replay buffer 的影子，又是 agent 系统里 trajectory queue 的影子；slime 通过把契约前置、实现下沉避开了"到底用哪种 buffer"的早期表决。

### 1.2 调用方与被调用方

唯一在 ray actor 内构造 data source 的位置是 `slime/ray/rollout.py:438`：

```python
data_source_cls = load_function(self.args.data_source_path)
self.data_source = data_source_cls(args)
```

`--data-source-path` 默认是 `"slime.rollout.data_source.RolloutDataSourceWithBuffer"`（`slime/utils/arguments.py:607`）。一旦构造，`self.data_source` 就被作为第三个位置参数透传给所有 `rollout_function_path` 入口（`slime/ray/rollout.py:651` 与 `:567`），由具体的 generate_rollout 函数自行决定如何 `get_samples / add_samples`。

## 2. 关键抽象

### 2.1 `DataSource`（`slime/rollout/data_source.py:17`）

抽象基类，只有五个方法：

| 方法 | 责任 |
|---|---|
| `get_samples(num_samples) -> list[list[Sample]]` | 返回 num_samples 个 prompt 组，每组 `n_samples_per_prompt` 个 sample（同一个 prompt 的若干 rollout） |
| `add_samples(samples)` | 把 rollout 生成完的 sample 写回（可选，read-only 实现会抛错） |
| `save(rollout_id)` / `load(rollout_id)` | 在 ckpt 路径下持久化游标，用来断点续训 |
| `__len__()` | "数据源还有多少"——返回值会随 get/add 变化 |

注意 `get_samples` 的返回类型是嵌套 list：**外层是 prompt 维度，内层是同 prompt 的 n 次采样**。这个 group 概念把 GRPO/DAPO 风格 reward 标准化天然嵌进 buffer 的最小单元。

### 2.2 `RolloutDataSource`（`:50`）

一次性 dataset 读取器。`__init__` 通过 `slime/utils/data.Dataset` 加载 jsonl prompt 数据（`:71`），维护 `sample_offset / epoch_id / sample_group_index / sample_index` 四个游标。`get_samples` 的核心是：

- 从 `self.dataset.samples` 里切 `num_samples` 个 prompt，跨 epoch 时自动 reshuffle（`:99-103`）。
- 对每个 prompt `deepcopy` 出 `n_samples_per_prompt` 个 Sample，统一打上 `group_index` 与 `index`（`:108-117`）。

`add_samples` 直接抛 `RuntimeError`：表明它是**只读源**——只生产 prompt、不接受写回。

### 2.3 `RolloutDataSourceWithBuffer`（`:168`）

在父类之上加了一个 Python list `self.buffer`（`:171`）和可插拔的 `self.buffer_filter`（`:172-175`）。读取顺序：

1. `_get_samples_from_buffer` 先用 `buffer_filter(args, None, buffer, num_samples)` 从 buffer 摘出尽可能多的 group（`:191-196`）。
2. 不够再调 `super().get_samples`，从底层 dataset 补足（`:188`）。

默认 `buffer_filter` 是 `pop_first`（`:225`），FIFO 行为；可以通过 `--buffer-filter-path` 注入策略（例如按 reward 优先级、按 rollout_id 老化等）。`add_samples` 接受嵌套 list，严格校验 `len(group) == n_samples_per_prompt`（`:207-209`），保证 buffer 永远是 group 整齐的。

### 2.4 `Sample` 的生命周期状态（`slime/utils/types.py:130`）

```python
class Status(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    TRUNCATED = "truncated"
    ABORTED = "aborted"
    FAILED = "failed"
```

Buffer 表面上不关心 Status，但状态决定了它会不会被回写：
- `ABORTED` 在 fully async 路径会回流到 buffer（`slime/rollout/fully_async_rollout.py:183-187`）；
- 在 partial rollout 路径，主循环 abort 时也会把没跑完的 group 收集后通过 `data_source.add_samples` 重新入队（`slime/rollout/sglang_rollout.py:651`）。

`Sample` 携带 `rollout_id`、`weight_versions`、`metadata`、`spec_info`、`prefix_cache_info` 等大量字段——buffer 全程对它们透明，是它在异步、partial rollout、speculative decoding 等场景下能被反复 round-trip 的关键。

### 2.5 `DynamicFilterOutput`（`slime/rollout/filter_hub/base_types.py:5`）

```python
@dataclass
class DynamicFilterOutput:
    keep: bool
    reason: str | None = None
```

加上 `call_dynamic_filter(fn, *args, **kwargs)` 兼容旧式只返回 bool 的过滤函数。`MetricGatherer.on_dynamic_filter_drop(reason)`（`:28`）把不 keep 的原因按字符串聚合，最终以 `rollout/dynamic_filter/drop_<reason>` 这种 metric key 上报。整个 filter_hub 模块小得令人惊讶——只有一种内置 filter `check_reward_nonzero_std`（`slime/rollout/filter_hub/dynamic_sampling_filters.py:9`），用于 DAPO 那种"全对或全错 group 不进 batch"的策略。

## 3. 数据流

以默认 sglang 路径 + `RolloutDataSourceWithBuffer` 为例，一个完整的数据流：

1. **构造期**：`RolloutGroup.__init__`（`slime/ray/rollout.py:438`）实例化 `data_source`。`RolloutDataSource.__init__` 立刻把 jsonl prompt 全部加载进内存的 `slime.utils.data.Dataset`，做一次 shuffle（`slime/rollout/data_source.py:85`）。此时 `self.buffer = []`。
2. **拉取 prompt**：`generate_rollout_async`（`slime/rollout/sglang_rollout.py:413`）调 `data_source(args.over_sampling_batch_size)`——注意它传的是 `data_source.get_samples` 这个**方法引用**而不是对象（`:649`），刻意把 buffer 概念藏在调用方背后。`get_samples` 先尝试从 buffer 摘，不够再补 dataset。
3. **生成**：sglang 后端跑完后，每个 group 经过 `dynamic_filter` 判定（`:441`）。被丢的 group 不进 `data`，但已经在 `state.remaining_batch_size` 里扣除了名额，循环会通过 `over_sampling_batch_size` 重新拉。
4. **abort / partial**：当 batch 满了或者上游打断时，`abort(args, rollout_id)` 把没跑完的 group 在 `partial_rollout=True` 下收集成 `aborted_samples`（`slime/rollout/sglang_rollout.py:357-372`）。
5. **回写**：`generate_rollout` 在拿到 `aborted_samples` 后调 `data_source.add_samples(aborted_samples)`（`:651`），下一个 rollout 启动时 buffer 里会先消费这批继续生成。
6. **训练**：`RolloutGroup.generate` 拿到 `RolloutFnTrainOutput.samples`，扁平化后送 `_convert_samples_to_train_data` → `_split_train_data_by_dp`（`slime/ray/rollout.py:558-559`）。buffer 不参与 train。
7. **持久化**：`save(rollout_id)` 把游标状态 `torch.save` 到 `<save>/rollout/global_dataset_state_dict_<rollout_id>.pt`（`slime/rollout/data_source.py:127-136`）。注意——**buffer 本身不会被序列化**（state_dict 只存 offset/index/metadata），重启后 buffer 是空的。

## 4. 设计模式

### 4.1 三层结构的意图

"`DataSource` 抽象 + `RolloutDataSourceWithBuffer` 默认实现 + `slime_plugins/rollout_buffer/` 外置参考"这三层不是过度设计，它分别对应三种 RL 训练拓扑：

- **同步 RL（默认）**：dataset + 薄 buffer 足够，partial rollout 偶尔回写。
- **异步 RL（fully async）**：仍然用同一个 `RolloutDataSourceWithBuffer`，但 `AsyncRolloutWorker._loop` 持续以 `get_samples(1)` 抽 group，并把 aborted group 通过 `add_samples` 回流（`slime/rollout/fully_async_rollout.py:137,185`）。
- **agent / long-tail trajectory**：rollout 由独立进程驱动、写入 HTTP buffer server，slime 训练侧只 GET 已经成型的 group——这就是 `slime_plugins/rollout_buffer/buffer.py` 的 `RolloutBuffer` 与 `BufferQueue`。这里的"group 是否 valid"逻辑（`is_valid_group_func`）下放到 generator 层，避免训练框架知道任何 agent 任务的细节。

三层共享同一组语义（拿 group / 放 group / 数 group），但实现可以是 list、可以是 HTTP server、可以是任意外部存储——这是把 `data_source` 当 plugin 注入而非硬编码的回报。

### 4.2 Dynamic filter 为什么不放进 buffer

一个朴素的设计是：buffer 内部维持 "valid group" 与 "invalid group" 两个 list，`get_samples` 永远只返回 valid 的。slime 没有这么做：dynamic_filter 在 `generate_rollout_async` 主循环里**消费完 group 之后立刻判定**（`slime/rollout/sglang_rollout.py:441-445`），不 keep 的 group 直接丢弃，不进 buffer、也不计数。

这背后有两层理由：

1. **解耦数据持有与采样策略**：buffer 不需要知道 reward、std、长度阈值这些训练侧的概念，filter 函数想看什么就传什么。
2. **配合 over-sampling**：DAPO 风格的 dynamic filter 必然丢一部分 group，所以主循环里有 `over_sampling_batch_size`（`slime/utils/arguments.py:408`）控制"一次多拉一些"。filter 与 over-sampling 协同必须在主循环里完成，而不是在 buffer 内部。

### 4.3 `slime_plugins/rollout_buffer/` 作为"展示什么是可能的"

`slime_plugins/rollout_buffer/buffer.py` 本质上是另一份 buffer 实现，但故意不进入 `slime/` 包：

- 它是 FastAPI server，端口 `8889`，独立部署；
- 它的"buffer"以 `instance_id` 做 key，`BufferQueue.append`/`get`（`buffer.py:145,184`）按 group 完整性发放数据；
- 它通过 `discover_generators()`（`:54`）自动扫 `generator/` 目录，每个 generator 模块声明 `TASK_TYPE` 与 `run_rollout`；
- slime 训练侧通过 `slime_plugins/rollout_buffer/rollout_buffer_example.py` 里的 `generate_rollout` 函数把 HTTP buffer 包装成 `RolloutDataSourceWithBuffer` 能消费的形态（`rollout_buffer_example.py:299-300`）。

这种"参考实现外置"暗示：**slime 不想钦定 agent trajectory 怎么收集**——math、code、SWE-agent 任务各自需要的 buffer 行为差别太大，把它放在主包反而会绑死用户。

## 5. 集成点

### 5.1 与 rollout 主循环

唯一稳定的契约就是 `data_source` 这个对象。三类 rollout 函数都符合 `def generate_rollout(args, rollout_id, data_source, evaluation=False)` 签名（参见 `tests/plugin_contracts/test_plugin_rollout_contracts.py:167` 断言的默认签名），通过：

- `data_source.get_samples(N)` / 直接 `data_source(N)`（sglang 路径传方法引用，`sglang_rollout.py:649`）
- `data_source.add_samples(groups)`（partial/aborted 回写）
- `data_source.get_metadata()` / `update_metadata()`（仅 plugin 路径用，`rollout_buffer_example.py:214,264`，注释里标了 `# TODO remove`）

### 5.2 与 dynamic filter hook

通过 `--dynamic-sampling-filter-path` 注入（`slime/utils/arguments.py:421`），在 `sglang_rollout.py:395` `load_function` 后传入 `call_dynamic_filter` 调用，签名约定为 `(args, samples: list[Sample], **kwargs) -> DynamicFilterOutput | bool`。除此还有 `--buffer-filter-path`（从 buffer 摘选 group 的策略，`data_source.py:172`）、`--rollout-sample-filter-path`（最终 batch 出去前过一遍，`sglang_rollout.py:470`）、`--rollout-all-samples-process-path`（处理所有样本含被过滤的，`:475`）四个独立 hook。它们之间互不感知，全靠 rollout 主循环串起来。

### 5.3 SFT / async / long-tail 行为差异

| 路径 | get_samples 调用方式 | add_samples 用法 | dynamic_filter |
|---|---|---|---|
| `slime/rollout/sft_rollout.py:42` | `data_buffer.get_samples(rollout_batch_size)` 一次拿满 | 不调用——SFT 没有生成阶段，prompt 即 trajectory | 不调用 |
| `slime/rollout/sglang_rollout.py` 默认 | `data_source(over_sampling_batch_size)` 循环拉 | 仅在 `partial_rollout` 下回写 aborted group | 必经 |
| `slime/rollout/fully_async_rollout.py` | `data_buffer.get_samples(1)` 持续滴灌 | ABORTED group 全部回流（`:185`） | 由 `generate_and_rm_group` 内部处理，不在 worker 主循环 |
| `slime_plugins/rollout_buffer/rollout_buffer_example.py` | 先 HTTP fetch → `data_buffer.add_samples`，再 `data_buffer.get_samples`（`:299-300`） | 用于把 HTTP 来的样本"喂"给 RolloutDataSourceWithBuffer | 不调用（filter 责任下放给 buffer server 的 `is_valid_group_func`） |

## 6. 出人意料的决策

### 6.1 Buffer 状态在 ckpt 里被丢弃

`RolloutDataSource.save`（`slime/rollout/data_source.py:127-136`）只保存 dataset 游标，**`self.buffer` 这个 list 完全不进 state_dict**。`RolloutDataSourceWithBuffer` 也没有覆盖 `save/load`。这意味着：partial rollout 攒了一堆没生成完的 group，节点挂了就全丢——这些样本的下一轮起点丢失，prompt 还会被重新当 fresh 处理（除非 dataset 游标恰好没走过）。

这是个非常刻意的取舍：buffer 序列化要面对 Sample 里 numpy/torch 字段的序列化复杂度、跨 rollout_id 的 weight_versions 一致性问题。slime 选择"buffer 是临时缓存，断点不保证连续"，反而让 DataSource 接口最干净。

### 6.2 Dataset 是单进程内存对象

`RolloutDataSource.__init__` 直接把整个 jsonl 加载进 `self.dataset.samples`，并且这个对象只存在于 `RolloutGroup` 这一个 ray actor 里（`slime/ray/rollout.py:438` 是单例式构造）。所有 sglang engine、所有 DP rank 都通过这一个 buffer actor 取 prompt。不需要 sharding 也不需要分布式 sampler——因为 rollout 本身就是按 batch 一次性拉给主循环再 fanout 的，不存在 "每个 worker 独立 sample" 的需求。

代价是 prompt 数据集必须能装进 driver 节点内存。这种"buffer 只在一处、prompt 也只在一处"的设计极大简化了"epoch / shuffle / 断点游标"的语义。

### 6.3 `get_samples` 被以方法引用形式传入

`sglang_rollout.py:649` 调用是：

```python
output, aborted_samples = run(generate_rollout_async(args, rollout_id, data_source.get_samples))
```

而 `generate_rollout_async` 的签名是 `data_source: Callable[[int], list[list[Sample]]]`——只接受一个**函数**，不接受 data_source 对象。这是有意的：内部生成循环只配拉数据、不应碰 `add_samples / save / load`。只有外层 `generate_rollout` 才有完整 `data_source`，决定要不要 add aborted_samples。这是个把"读"权限和"写"权限按层切开的小技巧。

### 6.4 `slime_plugins/rollout_buffer` 里的 Buffer 是"两份数据"

`BufferQueue` 同时维护 `self.data` 和 `self.temp_data`（`slime_plugins/rollout_buffer/buffer.py:134-136`）。`data` 是"会被消费、消费后清空"的工作队列；`temp_data` 是"自上次 get 以来"的累积视图，用于在 `/get_rollout_data` 接口返回 meta_info 时统计 avg_reward / total_samples，并在每次成功 read 后由调用方手动清空（见 `buffer.py:289` 的 `buffer.buffer.temp_data = {}`）。

这种"消费视图 vs 监控视图"的二元结构比较少见，背后的需求是：HTTP buffer 必须在外置的 wandb logging 里看到"原始未过滤分布"，但训练拿到的只是已 valid 的 group。

### 6.5 "buffer filter" 不是 buffer 的方法，而是注入函数

`RolloutDataSourceWithBuffer` 没有 `pop / filter / sort` 之类的方法，所有"怎么从 buffer 选样本"的策略都通过 `--buffer-filter-path` 注入一个 `(args, rollout_id, buffer, num_samples) -> list[list[Sample]]` 形式的函数（`data_source.py:195`）。默认 `pop_first` 干脆是模块级裸函数（`:225`），10 行代码。这与 dynamic_filter 的设计一脉相承：**buffer 只持有数据，所有决策都从外部注入**。

## 附录

### 通读文件清单

- `slime/rollout/data_source.py`（全文 229 行）
- `slime/rollout/filter_hub/__init__.py`（空）
- `slime/rollout/filter_hub/base_types.py`（37 行）
- `slime/rollout/filter_hub/dynamic_sampling_filters.py`（15 行）
- `slime/rollout/forge_load.py`（114 行）
- `slime_plugins/rollout_buffer/buffer.py`（340 行）
- `slime_plugins/rollout_buffer/generator/__init__.py`（7 行）
- `slime_plugins/rollout_buffer/generator/base_generator.py`（351 行）
- `slime_plugins/rollout_buffer/rollout_buffer_example.py`（307 行）

### 扫读文件清单（笔记里引用但未通读）

- `slime/utils/types.py:93-249`（`Sample` dataclass）
- `slime/rollout/base_types.py`（`RolloutFnTrainOutput / EvalOutput / call_rollout_fn`）
- `slime/ray/rollout.py:425-665`（`RolloutGroup` 中构造 data_source 与 `_get_rollout_data`）
- `slime/rollout/sglang_rollout.py:340-479`（abort / generate_rollout_async / generate_rollout）
- `slime/rollout/fully_async_rollout.py:53-256`（AsyncRolloutWorker）
- `slime/rollout/sft_rollout.py`（SFT 走 data_buffer 的极简路径）
- `slime/utils/arguments.py:395-630, 1356-1361`（buffer/filter/data_source 相关 CLI）
- `tests/plugin_contracts/test_plugin_rollout_contracts.py`（rollout 函数签名契约）
- `slime_plugins/rollout_buffer/README.md`（架构图）

### 未读但可能相关的文件

- `slime/utils/data.py`（`Dataset` 类，dataset 加载和 shuffle 的具体实现）
- `slime_plugins/rollout_buffer/rollout_buffer_example.sh`（启动脚本）
- `slime/rollout/sglang_streaming_rollout.py` / `sleep_rollout.py` / `on_policy_distillation.py`（其他 rollout 路径如何用 data_source）
- `tests/test_qwen3_4B_streaming_partial_rollout.py`（partial rollout 集成测试，应能体现 add_samples 回写路径）
