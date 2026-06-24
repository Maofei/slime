# 研究笔记：agent harness 子系统

> **本子系统在书中对应**：第 9 章 "agent harness——slime 自带的执行层"
> **范围**：`slime/agent/`（约 2.1k 行）+ `tests/test_agent/`（约 2.8k 行）
> **不归本章**：18 个 customization hook 的签名、rollout manager、sandbox 的部署/安装。

## TL;DR

slime 的核心叙事是"don't be an agent framework"——它把 agent 的语义抽出来交给社区（strands、tau-bench、retool 都自己造 agent）。但 `slime/agent/` 仍然存在，因为有一类训练目标社区框架解决不了：**让真实的、社区维护的 agent CLI（claude-code、codex）跑在沙盒里产生 trajectory，并且 token 级地训练那条 trajectory**。这就是 harness 子系统：它不是给用户写 agent 的脚手架，而是一组"把外部 agent CLI 接进 slime 训练循环"的胶水。整套设计的中心问题是 **token capture without TITO drift**——既要 CLI 把模型当成 Anthropic/OpenAI 来用，又要保证训练的 token id 与采样时一致。模块本身坚决保持小：4 个文件 ~1.4k 行（adapters 三层 + trajectory + parsing + harness），只对 coding agent 一种形态做了真实集成。

## 1. 架构与模块边界

```
                    ┌──────────────────────┐
                    │  Sandbox (E2BSandbox)│  ← 沙盒协议层
                    └──────────┬───────────┘
                               │ 安装 CLI + 写 config
                    ┌──────────┴───────────┐
                    │  Harness (CC / Codex)│  ← agent CLI 生命周期
                    └──────────┬───────────┘
                               │ 启动外部 CLI 进程
                               │      ↓
                               │  CLI 把自己当成 Anthropic/OpenAI
                               │      ↓ HTTP
┌─────────────────────────┐    │
│  aiohttp_threaded       │ ───┤  Adapter (HTTP server)
│  (后台线程跑 aiohttp)   │    │  ← 在主线程提供 web.Application
└─────────────────────────┘    │
                               │  _run_turn:
                               │    翻译 → 渲染 token →
                               │    POST sglang /generate →
                               │    parse 工具调用 →
                               │    record_turn
                               │      ↓
                    ┌──────────┴───────────┐
                    │  TrajectoryManager   │  ← per-sid 路由树
                    │  (message tree)      │
                    └──────────┬───────────┘
                               │ finish_session
                               ↓
                       list[Sample]（slime 训练样本）
```

**六个文件的分工**（`slime/agent/__init__.py:1` 只有一行 docstring）：

| 文件 | 职责 |
|---|---|
| `trajectory.py` | per-sid 消息树 + token 漂移分类 + Sample 装配 |
| `sandbox.py` | `Sandbox` Protocol + `E2BSandbox` 实现 |
| `parsing.py` | sglang 解析器适配 + XML tool_call fallback |
| `aiohttp_threaded.py` | 把 aiohttp 跑到 daemon 线程里 |
| `harness/*.py` | agent CLI 生命周期抽象 + 两个真实集成 |
| `adapters/*.py` | Anthropic/OpenAI 协议 → sglang `/generate` 的 HTTP 适配 |

**adapter 与 harness 是完全解耦的两层**。adapter 实现 HTTP 协议（Anthropic Messages 或 OpenAI Chat-Completions），harness 实现"安装 CLI + 启动它指向某个 URL"。`tests/test_agent/test_agent_rollout_cpu.py:256` 显式验证 `CodexHarness + OpenAIAdapter` 也能跑通，说明两层正交。

## 2. 关键抽象

### TurnRecord（一次 sglang 调用快照）
`slime/agent/trajectory.py:28`

```python
@dataclasses.dataclass(frozen=True)
class TurnRecord:
    prompt_ids: list[int]
    output_ids: list[int]
    finish_reason: str
    output_log_probs: list[float] = ...
```

"adapter 与 manager 之间的契约"。frozen 是关键——一旦交给 manager 就不能改。

### MessageNode + per-sid 路由树
`slime/agent/trajectory.py:45`

每个 sid 一棵树。一个节点要么是"路由专用"（`turn is None`，比如 user/system/tool 消息，或被 rewrite-merge 降级的 assistant），要么是"生成的"（`turn is not None`，带 TurnRecord）。

`response_trained: bool`（`trajectory.py:81`）是 **sibling 共享前缀去重**的关键：多个叶子如果共享同一个生成 turn，只有第一个 leaf 训练它，后面的当 context 重发。

### DriftKind（三档漂移分类）
`slime/agent/trajectory.py:129`

```python
CLEAN    = "clean"    # prompt_ids 是已持有 token 的精确前缀延伸
REALIGN  = "realign"  # 漂移落在最近一次响应跨度内，且短，可覆盖
FORK     = "fork"     # 其它都是 fork：关闭这个 builder，开新的
```

这是 TITO（token-in token-out）回放对不齐时的容错机制。`_common_prefix_len`（`trajectory.py:115`）4096-chunk 比较找出第一个分歧点。

### Sandbox Protocol
`slime/agent/sandbox.py:25`

```python
@runtime_checkable
class Sandbox(Protocol):
    sandbox_id: str
    async def __aenter__(self) -> Sandbox: ...
    async def __aexit__(...) -> None: ...
    async def exec(self, cmd, *, user, env, timeout, check) -> ExecResult: ...
    async def write_file(self, path, content, *, user) -> None: ...
    async def read_file(self, path, *, user) -> str: ...
```

刻意小——只有 4 个方法。`E2BSandbox`（`sandbox.py:66`）是唯一实装。它有显著的**瞬态错误自动重试**逻辑（`_is_transient_rpc_error` `sandbox.py:107`，覆盖 `ProtocolError/ConnectError/SSLError/PoolTimeout` 一堆 E2B 网关常见的抖动）。

### BaseHarness
`slime/agent/harness/common.py:56`

```python
class BaseHarness(ABC, metaclass=SingletonABCMeta):
    name: str = ""
    @abstractmethod async def install_cli(self, sb): ...
    @abstractmethod async def write_config(self, sb, ctx): ...
    @abstractmethod async def launch_and_wait(self, sb, ctx, prompt, time_budget_sec): ...

    async def run(self, sb, *, workdir, session_id, adapter_url, time_budget_sec, prompt):
        # ensure_agent_user -> write_config -> launch_and_wait
```

`SingletonABCMeta` 让每个 harness 类全局一个实例——它本身无状态，singleton 只是省构造开销。

`HarnessContext`（`harness/common.py:41`）是"无任务字段"的纯运行上下文：workdir、session_id、adapter_url、model_label，故意不带数据集或 prompt 字段。

### BaseAdapter
`slime/agent/adapters/common.py:126`

```python
class BaseAdapter:
    logger, log_prefix, max_token_keys, stop_keys = ...
    def _register_routes(self, app): ...   # 子类填路由
    def _session_id(self, request, body): ...
    def _translate(self, body): ...
    def _build_reply(self, parsed, raw_finish, translated, tools_schema): ...
    async def _respond(self, request, body, reply, in_tok, out_tok, stream): ...

    async def _run_turn(self, request):  # 共享的一次轮次管线
```

`_run_turn`（`adapters/common.py:302`）是核心：translate → render → sglang `/generate` → parse → record_turn → respond。子类只填协议特定的钩子。

`Session`（`adapters/common.py:32`）只装"采样默认值 + 上下文预算"，**trajectory 状态不在这里**——它在共享的 `TrajectoryManager` 里以 sid 为键。

## 3. 数据流：一次 agent rollout

以 `coding_agent_rl` 为例（`examples/coding_agent_rl/generate.py:165`）：

1. **进入 generate**：slime 的 customization 层调 `generate(args, base_sample, sampling_params)`。
2. **AdapterService 启动一次**：`_AdapterService`（`generate.py:120`）是 SingletonMeta，整个进程**一个 adapter 共享所有 sample**。它构造 `AnthropicAdapter` 并通过 `run_app_in_thread` 把 `adapter.app` 跑到后台线程。
3. **per-sample open_session**：`adapter.open_session(sid, sampling_defaults=..., max_context_tokens=...)`（`adapters/common.py:209`）。sid 是 `cagent-{instance_id}-{index}-{group_index}`。
4. **沙盒 boot**：`boot_agent_sandbox` 拿到 `E2BSandbox`，`ClaudeCodeHarness().install_cli` 上传 Node22+claude tarball。
5. **workspace prep**：`swe.prepare_workspace` 写 PROBLEM_STATEMENT.md（task layer，不在 harness 范围）。
6. **harness.run**：
   - `ensure_agent_user`（`sandbox.py:257`）建 unprivileged 用户。
   - `write_config`（`harness/claude_code.py:44`）写 `~/.claude/settings.json` 预 ack bypass-permissions。
   - `launch_and_wait` → `run_command`（`harness/common.py:106`）：写 `run.sh` launcher → `setsid` 启动到后台 → 每 5s polling `done` marker → 读 `PIPESTATUS[0]` 拿真正退出码。
7. **CLI 在沙盒内运行**：`claude -p "..." --permission-mode bypassPermissions --output-format stream-json ...`。环境变量 `ANTHROPIC_BASE_URL=adapter_url`、`ANTHROPIC_AUTH_TOKEN=session_id`。**sid 走 Bearer header**。
8. **每次 CLI → adapter HTTP 调用**就是一次 `_run_turn`：
   - `_session_id`（`adapters/anthropic.py:192`）从 Bearer 解 sid。
   - `_translate`（`adapters/anthropic.py:81`）把 Anthropic 消息列表翻译成 chat-template 格式。
   - `_render_token_ids`（`adapters/common.py:57`）调 tokenizer.apply_chat_template 拿 prompt_ids。
   - `call_sglang_generate`（`adapters/common.py:408`）POST sglang `/generate`，带 `X-SMG-Routing-Key: sid` 让 router 把同一 sid 始终路到同一个 worker（KV 复用）。
   - 拿到 output_ids → `tokenizer.decode` → `parse_model_output`（用 sglang 的 ReasoningParser+FunctionCallParser，外加 XML fallback）。
   - `_build_reply` 装回 Anthropic blocks。
   - `manager.record_turn(sid, turn, prompt_messages, response_message)` 把 turn 挂到 sid 的树上。
   - 返回 wire response（可流可非流，`anthropic.py:211` / `openai.py:324`）。
9. **CLI 退出后** `swe.git_diff` 抓 patch，`swe.evaluate` 在干净沙盒里 grade。
10. **finish_session**（`adapters/common.py:236`）：
    - `shutdown_session` 等待 inflight。
    - `manager.get_trajectory(sid)` 把树线性化：每个 routing leaf 走一遍 `path_from_root` → `_split_chain_into_builders` 把 turns 装进 `_SampleBuilder`（漂移分类决定是延伸还是 fork）→ 每个 builder 产 0/1 个 Sample。
    - reward 平分到 N 个 Sample（`trajectory.py:323`）。
    - adapter 用自己的 tokenizer 解码 `s.tokens[-rlen:]` 填 `s.response`（manager 是 tokenizer-free 的）。

## 4. 设计模式

### HTTP 并发模型：aiohttp_threaded
`slime/agent/aiohttp_threaded.py:1`

slime 的 rollout 函数（`generate`）是 async 的，自己有事件循环。但 adapter 也要监听 HTTP——而且要被沙盒里**所有 sample 的 CLI 同时调用**。如果 adapter 跑在同一个事件循环里，它就会和 rollout 互相 starvation。

解决方案：**aiohttp 跑在专门的 daemon 线程**，里面用自己的 `asyncio.new_event_loop()`（`aiohttp_threaded.py:66`）。`run_app_in_thread` 同步等到 socket 实际 listen 才返回（`started.wait`）。`AppHandle.stop` 用 `run_coroutine_threadsafe` 跨线程停止。

`FilteredAccessLogger`（`aiohttp_threaded.py:14`）只 log 慢请求（>120s）或非 200。一次 SWE rollout 几百次 turn，默认 access log 会爆。

### harness 与 adapter 的解耦
- `claude_code` → 标准跑 `ClaudeCodeHarness + AnthropicAdapter`。
- `codex` → 跑 `CodexHarness + OpenAIAdapter`。
- 但两者两两组合都成立（测试 `test_codex_openai_rollout_closes_loop` 验证）。

解耦的代价：harness 必须**把 sid 编码进 CLI 能接受的某个字段里**。Claude 用 `ANTHROPIC_AUTH_TOKEN`，Codex 用 `OPENAI_API_KEY`（`harness/claude_code.py:64`、`harness/codex.py:78`），都被 adapter 当 Bearer 解。这是个聪明的复用："API key" 字段在所有 LLM CLI 里都存在，正好拿来做 session 标识。

### trajectory branching：dict equality 路由
`record_turn` 的核心是 `_find_mount_point`（`trajectory.py:333`）——它走树，**用 dict 相等比较**决定每条消息挂在哪。这意味着：

- sub-agent 派发（assistant 启了一个子任务，produce 不同的 user 消息序列）→ 自然 fork。
- auto-compaction（claude-code 把历史压缩成一条 system 消息再继续）→ system 不匹配 → 在树根附近 fork。
- 同一条历史回放过来（claude-code 内部会重发上一次的 history）→ 完全匹配 → 不 fork。

但要让 dict 相等成立有几个**容易踩的不变量**（`adapters/common.py:109` 的 `tool_call_dict` 注释、`adapters/openai.py:107` 的 `_translate_messages` docstring）：
- `tool_calls[].function.arguments` 保持 dict（不是 JSON 字符串），否则 key order 一变就 diverge。
- 每次响应的 wire-only id（`call_xxx` / `toolu_xxx`）必须丢掉。
- assistant 同时有 text 和 tool_calls 时 content 设 `""`（OpenAI）或对应处理（Anthropic），因为客户端会拆。

`_try_merge_assistant_rewrite`（`trajectory.py:351`）是这个机制的**软化补丁**：claude-code 偶尔会在下一次 prompt 里把上一次的 assistant 微调（去掉尾空格之类），导致 dict 不等→ fork → 孤立的旧 leaf 还得训。短于 `fork_threshold` 的就强行覆盖原节点+demote 成 routing-only，不再训。

## 5. 集成点

### 与 customization 层的对接
harness 在最外层是被 `--custom-generate-function-path` 指向的 `generate` 函数调用的（`examples/coding_agent_rl/generate.py:165`）。slime 自己**不知道**有 harness 这回事——它只看到这个 generate 函数返回 `list[Sample]`。这是设计的关键：harness 不是 slime 的 plugin point，它是 plugin point 里的一种实现选择。

### 与 sglang 的对接
唯一接触点：`call_sglang_generate`（`adapters/common.py:408`）POST 到 `f"{sglang_url}/generate"`。带的 header：
- `X-SMG-Routing-Key: <sid>` —— 让 sglang router 把同一 sid 始终路到同一个 worker，复用 KV cache（巨大的性能影响）。

异常路径：客户端断开/超时时立刻 fire-and-forget 一个 `/abort_request`（`adapters/common.py:472`）释放 sglang 的 slot，否则孤立生成会一直占 KV 到 length cap。

### 与 LLM API 的对接
两条协议线，宿主端是 sglang，客户端是真实的 CLI：

| 维度 | AnthropicAdapter | OpenAIAdapter |
|---|---|---|
| 路由 | `POST /v1/messages` + `/v1/messages/count_tokens` | `POST /v1/chat/completions` |
| sid 来源 | `Authorization: Bearer` → `X-Api-Key` → "default" | `Bearer` → `metadata.session_id` → `user` → "default" |
| stream 格式 | Anthropic SSE 多事件（message_start/content_block_*/...）| OpenAI SSE 单类 chatcmpl chunk → `[DONE]` |
| max_tokens key | `max_tokens` | `max_completion_tokens` → `max_tokens` → `max_output_tokens` |
| 特殊处理 | `_fold_mid_list_system_into_user`：把非首位的 system 折成 `<system-reminder>` 包到 user 里 | `developer` role → `system`；tool_calls[].arguments 字符串 ↔ dict 双向转 |

`count_tokens` 端点（`anthropic.py:279`）直接返回 0——claude-code 用它做 hint 不是硬预算。

## 6. 出人意料的决策

### 决策 1：把 claude_code 和 codex 作为**真实集成**而不是只给抽象框架
设计上更"干净"的做法是 ship `BaseHarness` 让用户自己 subclass。但仓库里直接住了 71 行的 `ClaudeCodeHarness` 和 86 行的 `CodexHarness`——**而且只在测试里**会运行 codex 版（生产路径硬编码了 claude_code）。

为什么这么做：claude-code 和 codex 的**真正复杂度**不在 BaseHarness 上，而在那些"非显然的"细节里：
- `bypassPermissionsModeAccepted: true` 必须预先写到 settings.json，否则 claude-code 会卡在 onboarding（`claude_code.py:46`）。
- Codex 的 `base_url` **必须**写在 TOML 里 inline，因为它只对 "default OpenAI provider" 才认环境变量（`codex.py:36-46`）。
- Codex 配置文件要 base64 round-trip 进沙盒，否则 shell quoting 会爆（`codex.py:60`）。
- claude-code 必须传 `--output-format stream-json --include-partial-messages --include-hook-events --verbose`，否则训练数据缺关键信号（`claude_code.py:24`）。

这些是"花了半年才踩明白"的经验。如果只 ship 抽象，用户自己写 harness 一定会重新踩一遍。所以 slime 选择**把经验固化为代码**——这违背了一般的"框架不该 ship 用法"原则，但很符合 slime 这种"愿意把 hard-won knowledge 直接写进仓库"的风格。

### 决策 2：`aiohttp_threaded` 把 aiohttp 跑进**线程**而不是同一个事件循环
"async 同进程为什么不直接共用 loop？"——因为 slime 的 `generate` 是被 ray actor 跨 sample 并发驱动的，每个 sample 一个 task。adapter 同时要服务**所有 sample**的 HTTP 请求。如果共用 loop，任何一个 sample 的 sync 操作或长 await 都会卡住 adapter 响应给其它 sample 的 CLI。

更深的原因：adapter 内部还要 await sglang `/generate`（可能 30 秒+），如果共用 loop，这次 await 期间整个 rollout 的进度都得跟着它的 scheduler。**单独线程跑 aiohttp + 自己的 event loop**是最低耦合的解。

命名也很坦诚——不是 "AdapterServer" 或 "HTTPHost"，就叫 `aiohttp_threaded`，清楚说明这是"把 aiohttp 跑进线程"的工具。

### 决策 3：trajectory 用**消息树**而不是消息列表
最自然的设计是每个 sid 一个 turn 列表，append 进去就完事。但 claude-code 这种 agent 会做：
- **sub-agent 派发**：主 agent 派一个子 agent 处理 sub-task，子 agent 有自己的消息流。
- **auto-compaction**：上下文打到 80% 就把历史压缩成一条 system 消息。
- **rewrite**：偶尔会把上一次的 assistant 消息微调后重发。

如果用列表，这些场景要么扭曲成线性（丢掉子任务）要么需要特殊 case 处理。**树**让所有"我重新回到某个历史节点开了一条新路"的语义自然成立——任何 prompt 在某个深度不匹配就 fork。`get_trajectory` 把所有叶子各自线性化成 Sample，子 agent 自然就是一条独立 Sample。

而且 `response_trained` 的 sibling 去重（`trajectory.py:81`）保证共享前缀只训一次——子任务 fork 出去后，前面共同的 turns 在两条 Sample 里只有一边带 `loss_mask=1`，另一边当 context 重发。

### 决策 4：sid 编码进 LLM **API key 字段**
这是个非常 hacky 但极优雅的选择。每个 sample 一个 sid，CLI 要把它带回每次 HTTP 调用——但 claude-code/codex 的命令行界面**没有"session id"参数**。

解决方法：abuse "API key"。所有 LLM CLI 都接受一个 API key 环境变量（`ANTHROPIC_AUTH_TOKEN` / `OPENAI_API_KEY`），CLI 都会把它放进 `Authorization: Bearer` header。adapter 端从 Bearer 解出 sid 就行了——**zero-code-change** 复用了一个普适的字段（`harness/claude_code.py:64`、`harness/codex.py:78`、`adapters/common.py:362`）。

副作用：sandbox 内的 CLI 看到的 API key 其实是 sid，但因为 adapter 接受任何字符串，它就当成 key 用了。

### 决策 5：drift 容错的三档分类
最暴力的处理是"每次 prompt 不匹配就 fork"。但 TITO 回放在生产里**到处都会轻微不匹配**——chat template 重新渲染、UTF-8 边界、引号转义。每次都 fork 会让 trajectory 树爆炸。

`DriftKind.REALIGN`（`trajectory.py:188`）的精妙之处：只对**最后一次响应跨度内、且新响应短于 fork_threshold** 的漂移做覆盖。这个边界条件意味着：
- 老 turn 的 token 不动（已经训过）。
- 新 turn 因为太短信号也少，覆盖掉损失可控。
- 漂移如果落在更早的 turn 里，老的训练信号会被毁掉——所以必须 fork。

这是经验性的，不是理论上 optimal，但实测能把"无意义 fork"压到很低。

## 附录

### 通读文件清单
- `slime/agent/__init__.py` ✓
- `slime/agent/trajectory.py` ✓（477 行，整篇核心）
- `slime/agent/sandbox.py` ✓
- `slime/agent/parsing.py` ✓
- `slime/agent/aiohttp_threaded.py` ✓
- `slime/agent/harness/__init__.py` ✓
- `slime/agent/harness/common.py` ✓
- `slime/agent/harness/claude_code.py` ✓
- `slime/agent/harness/codex.py` ✓
- `slime/agent/adapters/__init__.py` ✓
- `slime/agent/adapters/common.py` ✓
- `slime/agent/adapters/anthropic.py` ✓
- `slime/agent/adapters/openai.py` ✓
- `tests/test_agent/test_harness.py` ✓
- `tests/test_agent/test_adapters.py` ✓
- `tests/test_agent/test_agent_rollout_cpu.py` ✓
- `tests/test_agent/test_trajectory_manager_branching.py` ✓（只读了 outline + 前 200 行）
- `examples/coding_agent_rl/generate.py` ✓（前 290 行）
- `examples/coding_agent_rl/README.md` ✓

### 扫读 / 未读
- `tests/test_agent/_fakes.py`（308 行 fake sandbox/tokenizer/sglang，没细读）
- `tests/test_agent/_dump_helpers.py`（85 行调试 dump）
- `examples/coding_agent_rl/swe.py`（任务层，归 examples 章节）
- `examples/coding_agent_rl/run_qwen36_35b_a3b_swe_8nodes.sh`（启动脚本，归 deployment）
- `examples/strands_sglang/`、`examples/tau-bench/`、`examples/retool/` —— **确认它们不用 `slime.agent.*`**，自己造 agent，归别处。
- `slime/utils/misc.py` 的 `SingletonMeta`（在多处被用到，归工程基础设施章节）

### 几个值得在正文里展开的代码片段
1. `trajectory.py:140-250` `_SampleBuilder` 全文 —— 训练 token 装配的核心逻辑。
2. `adapters/common.py:302-360` `_run_turn` —— 一次 agent turn 的标准流程。
3. `harness/common.py:106-156` `run_command` —— "detach + poll marker" 这个非显然的进程管理模式。
4. `harness/codex.py:30-50` `config_toml` —— 把"踩坑经验"固化为代码注释的典型样本。
5. `aiohttp_threaded.py:46-98` `run_app_in_thread` —— 跨线程启动 + 同步等待 listen 的全部 ~50 行。
