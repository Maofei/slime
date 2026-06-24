# 研究笔记：工程基础设施子系统（trace / metric / health / CI / 不变量 / reproducibility / fault tolerance）

## TL;DR

slime 把工程基础设施当作一等公民，根本原因是一句口号：**"RL 的 bug 不报错"**——loss 还在下降、grad_norm 看起来正常、wandb 曲线漂亮，但模型其实没学到任何东西。所以光有 log 不够，必须有 (1) **细粒度 trace**（每条 sample 一条 timeline，记录 generation/reward/agent step 的耗时和属性，离线可视化）；(2) **跨并行不变量测试**（同一份样本不管走 DP/CP/per-token-loss/per-rollout-mean 哪条路径，gradient norm 和上报指标必须完全相等）；(3) **rollout 健康监控**（长尾任务里 SGLang server 可能 hang，必须有 heartbeat + 自动重启）。这三件套加上 Jinja2 生成的 CI 矩阵（CPU unit + GPU e2e 双层覆盖 dense/MoE/PPO/MTP/OPD/async/ckpt/precision）共同守住 "脚本能跑不算 done" 这条底线。值得在书里讲的是：trace 系统用 `contextvars` 做 span 栈、容错地把所有异常吞掉只记 debug log（trace 自己绝不能拖垮训练），metric 系统在 train/rollout 两侧各自跑一遍同样的 reducer 并由 `test_metric_report_dist.py` 证明两条路径给出同一个数字，而 `tests/_cp_dist_helpers.py` 用 `mp.spawn` + gloo + Megatron mpu stub 在纯 CPU CI 上跑出真实的 4-rank 分布式行为——这件事是 slime 最反常识的 CI 投资。

---

## 1. 架构与模块边界

工程基础设施横跨 6 个区域，每个区域的边界都很清楚：

1. **可观测三件套**（都在 `slime/utils/` 下）
   - **Trace**：`trace_utils.py`（776 行），sample-scoped 的 span/event 记录器，唯一会被 user code 直接调用的可观测设施。
   - **Metric**：`metric_utils.py`（pass@k、compression_ratio、compute_statistics 等纯函数）+ `train_metric_utils.py`（`Timer` → `perf/*` 的聚合管线）+ `train_dump_utils.py`（rollout pt dump）。
   - **Health**：`health_monitor.py`（177 行），一个 daemon thread 监 SGLang 引擎健康状态。
2. **Timer 与分布式工具**：`timer.py` 提供 singleton `Timer`（with `@timer` decorator / `with timer("name")` 两用 + `inverse_timer` 反向计时 + `@with_defer` 装饰器）；`distributed_utils.py` 提供 `init_gloo_group` 和 `distributed_masked_whiten`（全局白化）。
3. **Profiling**：`profile_utils.py`（147 行），`TrainProfiler` 包装 `torch.profiler` + `torch.cuda.memory._record_memory_history` + memray，按 rollout step 调度 start/stop；面向 SGLang 侧的 GPU profiling 走 `tools/profile_rollout.py` HTTP 触发。
4. **CI 基础设施**：
   - `.github/workflows/pr-test.yml.j2`（Jinja2 模板，用 `<% %>` / `<< >>` 作为定界符以避开 GitHub Actions 的 `${{ }}`），由 `generate_github_workflows.py` 渲染成 `pr-test.yml`。
   - `tests/ci/gpu_lock_exec.py`：自托管 runner 上用 fcntl 文件锁串行化 GPU 占用，让一台 8 卡机器能同时跑多个 CI job 而不相互踩 GPU。
   - `tests/ci/github_runner/docker-compose.yml`：runner 自身的 Docker 化部署说明。
   - `slime/backends/megatron_utils/ci_utils.py`：训练循环内的 CI assertion 钩子（如 `check_mtp_only_grad` 验证 truncation 场景下只有 MTP 层有梯度）。
5. **不变量测试矩阵**（`tests/`）
   - **CP / loss 不变量**：`test_cp_utils.py`（per-rollout reducer 契约）、`test_loss_cp_invariance.py`（end-to-end backward grad norm CP 不变性）、`_cp_dist_helpers.py`（mp.spawn + gloo + megatron stub 的 fixture）。
   - **报告公式不变量**：`test_metric_report.py`（单进程）、`test_metric_report_dist.py`（多进程）保证 train 侧 `reduce_train_step_metrics` 和 rollout 侧 `rollout_log_metric_contribution` 在任意 DP/CP 切分下给出同一个数字。
   - **算法层不变量**：`test_chunked_gae.py`（并行 scan 与 serial GAE 数值一致 atol=1e-5）、`test_cispo_loss.py`。
   - **配置层不变量**：`test_placement_group.py`（colocate / debug_train_only / debug_rollout_only / external 10 种组合下 actor/rollout GPU 分配数字必须固定）、`test_dp_schedule.py`、`test_megatron_argument_validation.py`。
6. **文档化最佳实践**：`docs/zh/developer_guide/ci.md` / `debug.md` / `trace.md` / `profiling.md` 与 `docs/zh/advanced/reproducibility.md` / `fault-tolerance.md` / `observability.md`——这些不是事后的 README，而是和代码一起被维护的"使用契约"。

---

## 2. 关键抽象

| 抽象 | 路径 | 说明 |
| --- | --- | --- |
| `trace_span(target, name, attrs=...)` | `slime/utils/trace_utils.py:351` | sample-scoped 的 contextmanager span；用 `contextvars._TRACE_STACK` 维护 parent-child 关系；yield 一个 `TraceSpanContext` 允许 `span.update(attrs)` 补充结束属性 |
| `trace_event(target, name, attrs=...)` | `slime/utils/trace_utils.py:332` | 瞬时事件，没有 duration |
| `trace_function(name, target=...)` | `slime/utils/trace_utils.py:457` | decorator，同时支持 sync / async；通过 `inspect.signature` 解析 `target=` 参数名 |
| `TraceHandle` / `bind_trace(sample)` / `export_trace` / `import_trace` | `slime/utils/trace_utils.py:58, 244, 295, 317` | trace 的"carrier"——一个挂在 `sample.trace` 上的 dict；可以序列化跨进程，用于把 rollout 子任务的 trace 合并回 driver 端的 sample |
| `build_sglang_meta_trace_attrs(meta)` | `slime/utils/trace_utils.py:146` | 把 SGLang response 的 `meta_info` 转成 trace attrs；含 PD 拆解（prefill/decode 各自再拆 bootstrap/alloc_wait/forward/transfer 子 span） |
| `_log_trace_error(action, exc)` | `slime/utils/trace_utils.py:134` | trace 系统的核心防御性原则：所有异常都走 `logger.debug` 吞掉，绝不向上抛——trace 失败永远不能把训练带崩 |
| `Timer` (singleton) | `slime/utils/timer.py:15` | 进程内全局 timer 字典；`Timer().start("train_wait")` / `.end(...)` 累加；`log_dict()` 把累加结果 dump 到 metric 系统；可作 `@timer` decorator 或 `with timer("name"):` |
| `with_defer(deferred_func)` | `slime/utils/timer.py:92` | 装饰器，函数返回后调用 `deferred_func()`；典型用法 `@with_defer(lambda: Timer().start("train_wait"))` 让 init 完成自动启动等待计时 |
| `log_perf_data_raw` | `slime/utils/train_metric_utils.py:13` | `Timer().log_dict()` → `perf/*_time` → 自动计算 `perf/actor_train_tflops`、`perf/wait_time_ratio`、`perf/step_time` |
| `compute_perf_metrics_from_samples` | `slime/ray/rollout.py:1324` | rollout 侧的 perf 聚合；调 `_compute_sglang_request_perf_metrics` 把每条 sample trace 里的 SGLang timing attr 用 `compute_statistics` 聚合成 `perf/request/...`、`perf/prefill/...`、`perf/decode/...` |
| `RolloutHealthMonitor` | `slime/utils/health_monitor.py:10` | daemon thread；start/stop/pause/resume 四态；初始 `paused`，weight onload 后 resume；每次 resume 后等 `--rollout-health-check-first-wait` 秒（默认 300s，给 MoE kernel compile 留时间）再开始 heartbeat |
| `_kill_engine` | `slime/utils/health_monitor.py:160` | health check fail 后调 `ray.kill(engine)`；把 `all_engines[i] = None` 标记为死亡，由 rollout 侧在下一个 round 重启 |
| `TrainProfiler` | `slime/utils/profile_utils.py:13` | 包装 `torch.profiler.profile` + `_TorchMemoryProfiler` + `_MemrayMemoryProfiler`；profile_target 用 set-membership 控制 train_overall / train_actor / train_log_probs 三档 |
| `_TorchMemoryProfiler.oom_observer` | `slime/utils/profile_utils.py:114` | 注册到 `torch._C._cuda_attach_out_of_memory_observer`，OOM 时自动 dump snapshot——RL 长任务必备 |
| `init_gloo_group` | `slime/utils/distributed_utils.py:20` | 全局 gloo group，专门给 CPU 上的 metric all-reduce 用；NCCL 不参与 |
| `distributed_masked_whiten` | `slime/utils/distributed_utils.py:94` | 全 DP 范围内的 masked whitening，给 PPO advantage normalization 用——保证 whitening 是全局而非 per-rank 的 |
| `gpu_lock_exec.py` (FdLock) | `tests/ci/gpu_lock_exec.py:201` | fcntl LOCK_EX | LOCK_NB 文件锁，路径 `/dev/shm/custom_gpu_lock_{gpu_id}.lock`；`--count N` 任取 N 张，`--devices 0,1` 指定；通过 `CUDA_VISIBLE_DEVICES` 暴露给子进程 |
| `stub_megatron_in_worker(cp_size, cp_rank)` | `tests/_cp_dist_helpers.py:68` | 在 mp.spawn 出来的 child 里 mutate `megatron.core.mpu`（一个事先安装的 fake module）的 `get_context_parallel_*` 函数；这是让 CPU CI 跑 CP 测试的关键 trick |
| `check_mtp_only_grad` | `slime/backends/megatron_utils/ci_utils.py:11` | 训练循环里的 assertion 钩子：truncation 测试场景下只有 `.mtp.` 参数应有非零梯度；触发条件由 CI test (`test_mimo_7B_mtp_only_grad.py`) 设置 |
| `generate_github_workflows.py` | `.github/workflows/generate_github_workflows.py` | jinja2 模板渲染脚本；定界符设为 `<% %>` / `<< >>` 避免和 GitHub Actions 的 `${{ }}` 冲突 |

---

## 3. 数据流

### 3.1 Trace：从一条 sample 到 timeline

1. **创建**：rollout 侧 `generate_and_rm(sample)` 被 `@trace_function("generate_and_rm", target="sample")` 装饰，进入时 `bind_trace(sample)` 在 `sample.trace` 上创建 carrier dict `{version, trace_id, events: [], sample_id, group_id, attempt}`，并把 `(trace_id, span_id)` push 进 `_TRACE_STACK` (contextvar) 形成调用栈。
2. **嵌套**：`generate_and_rm` 内部 `with trace_span(sample, "sglang_generate"):` 创建子 span，自动通过 contextvar 栈找到 parent span_id；`span.update(build_sglang_meta_trace_attrs(meta_info))` 把 SGLang 返回的 timing 拆解（含 PD 子段）合并进 end_attrs。
3. **跨进程**：当一个 sample 进入 SGLang workers，主调用方用 `export_trace(handle)` 拿到 `{trace_id, sample_id, parent_span_id, ...}` 5 字段 payload；worker 端用 `import_trace(payload, carrier=...)` 重建 TraceHandle，记录的 events 会带正确的 parent_span_id；返回后 carrier dict 被 merge 回 sample.trace。
4. **落盘**：rollout 结束后由 `--save-debug-rollout-data /path/to/rollout_{rollout_id}.pt` 把整批 sample 序列化成 `.pt`（torch.save）；trace 是 sample 的一个 dict 字段，自然跟着走。
5. **可视化**：离线跑 `python tools/trace_timeline_viewer.py rollout_0.pt`，生成 `*.trace_timeline_cache.json` 和 `*.trace_timeline_viewer.html`；HTML 里每行一个 sample，span 为条形块，event 为点；遇到 PD 字段自动补出 `[P]` / `[D]` 虚拟 lane 拆开展示 prefill/decode。

### 3.2 Metric：分布式下如何汇总

slime 的 metric 走**两条并行管线**：

- **训练侧 (`perf/`)**：`Timer()` 在每个 rank 累加；`log_perf_data` (`slime/backends/megatron_utils/data.py:496`) 只在 `tp_rank==0 && pp_last_stage && dp_rank==0` 这个 primary rank 上把 `Timer().log_dict()` 转成 `perf/*_time`，并基于 `Timer().seq_lens` + `calculate_fwd_flops` 算 `perf/actor_train_tflops`、`perf/actor_train_tok_per_s`、`perf/wait_time_ratio`。
- **训练指标（loss/kl/entropy/grad_norm）**：在 micro-batch 内由 `get_sum_of_sample_mean` 返回 closure，每个 mb 调用产出标量 → `reduce_train_step_metrics` (`cp_utils.py:127`) 在 dp-with-cp group 内 `dist.all_reduce`，按 `cp_factor` 反向消除 CP 重复，再除 `step_global_batch_size`。
- **rollout 侧 (`rollout/` + `perf/`)**：`log_rollout_data` (`data.py:248`) 对每个字段用 `rollout_log_metric_contribution` 产出 `(sum, count)` 元组而非均值（这一点很关键，避免"每 rank 同 N 样本"假设把不均匀 DP 切分下的均值拖偏）；再由 `gather_and_reduce_log_dict` 用 `dist.gather_object` 汇总，对每个 key 应用对应 reduction op（sum、weighted mean、max 等）。
- **SGLang 请求级 perf**：`_compute_sglang_request_perf_metrics` (`ray/rollout.py:1358`) 直接遍历每条 sample 的 trace events 找 `sglang_generate` span 的 attrs，把 e2e_latency / queue_time / decode_throughput / pd_*duration 用 `compute_statistics` 聚合成 mean/median/max/min——**不上传 SGLang 的高频 Prometheus metrics 到 W&B**，那个交给旁路 Prometheus + TSDB。

### 3.3 Fault：从检测到恢复

1. **start**：rollout actor `init` 里如果 `--use-fault-tolerance` 就为每个 server group 创建一个 `RolloutHealthMonitor`，daemon thread `start()`，但初始 `paused`。
2. **resume**：每次 onload engine 后调 `monitor.resume()`，设置 `_need_first_wait=True`，等 `rollout_health_check_first_wait` 秒。
3. **check**：每 `rollout_health_check_interval` 秒（默认 10s），对每个 engine `ray.get(engine.health_generate.remote(timeout=check_timeout))`；timeout/异常 → `_kill_engine`：`ray.kill(engine)` + 把 `all_engines[i] = None`。
4. **pause**：weight offload 前 `monitor.pause()`（offload 时 engine 不能响应 health check）。
5. **rebuild**：下一个 rollout round 开始时由 rollout 侧检测到 `all_engines[i] is None` 重启 server，重启后由 weight updater 推一次最新权重，再 `resume()`。
6. **CI fault injection**：`_try_ci_fault_injection` (`ray/rollout.py:474`) 在 `--ci-test` 下故意调一次 `engine.simulate_crash.remote()` + sleep `interval+timeout+5s`，验证容灾路径——这是把"容灾代码本身也要被 CI 覆盖"做成一等公民的具体证据。

---

## 4. 设计模式

- **Trace 的低开销采样**：trace 默认对每条 sample 都开（没有采样率），但有三个降本设计：(1) 只在 sample 维度，不在 token / kernel 维度；(2) 所有 trace API 都用 `try/except + _log_trace_error` 兜底，失败不影响主路径；(3) carrier 是普通 dict，跟着 sample 走 Ray object store，不需要单独的传输管道；(4) `_TRACE_STACK` 用 `contextvars.ContextVar` 实现，asyncio 友好。
- **Metric 命名规范**：所有指标按 prefix 强分组——`perf/` 是性能（时间、tflops、tok/s）、`rollout/` 是 rollout 统计（response_len、reward 分布、truncated_ratio）、`train/` 是训练数值（loss/kl/entropy/grad_norm）、`eval/` 是评测；prefix 由 `dict_add_prefix` 统一加，避免散落各处。
- **Health monitor 的四态机**：`stopped → paused → checking → paused`，pause 而非 stop 是为了支持 weight sync 期间临时关闭检测（offload 时 engine 无法响应 health_generate）；用 `threading.Event` 而非锁，让主线程的 pause/resume 不被检测循环阻塞。
- **CI 矩阵的覆盖策略**：六个 GPU 维度同时被压（dense `test_qwen3_4B`、MoE `test_qwen3_30B_A3B`、PPO `test_qwen3_4B_ppo`、MTP `test_mimo_7B_mtp_only_grad`、async `test_qwen2.5_0.5B_fully_async_short`、OPD `test_qwen2.5_0.5B_opd_sglang`、checkpoint 4 种 save/load 组合 + async_save、PD/Mooncake `test_qwen3.6_35B_A3B_pd_mooncake`、debug rollout-then-train replay `test_qwen2.5_0.5B_debug_rollout_then_train`、parallel 精度 `test_qwen3_0.6B_parallel_check`、FP8 `use_fp8_rollout=1`、DeepEP `use_deepep=1`）——目标不是"覆盖率"，而是"任何一个轴向变化都至少有一个 e2e 覆盖到"。
- **CPU unit + GPU e2e 的分工**：CPU 上跑公式 / schedule / reward-grading / contract / placement 共 25+ 个 test（在 ubuntu-latest 上免 GPU），GPU 上跑 e2e（自托管 runner，要 label 才触发）。CPU 是"correctness 第一道防线"，GPU 是"集成验证"。这意味着 80% 的 review 拒绝在 ubuntu-latest 上 5 分钟内完成，不烧 GPU。
- **Jinja2 生成 workflow**：`.yml.j2` → `.yml` 的好处是把 28 个 megatron test 的 entry 集中维护，不用复制粘贴 28 段 docker run；定界符改成 `<% %>` 是因为 GitHub Actions 自己用了 `${{ }}`。
- **gpu_lock_exec 的设计**：用 fcntl LOCK_NB 文件锁而非 GPU runtime 查询——这样不需要 SDK，也不依赖 nvidia-smi 输出格式；按 GPU id 一个一个锁，遇到全部锁不到就退避重试（`SLEEP_BACKOFF * random.random()` 避免羊群效应）。
- **reproducibility 的"全栈拼装"**：bitwise reproducibility 不是单点技术，需要同时关掉 4 个非确定性源头——SGLang 用 `--sglang-enable-deterministic-inference --sglang-attention-backend flashinfer`（关 FA3）、Megatron 用 `--deterministic-mode`、NCCL 用 `NCCL_ALGO=Ring`、cuBLAS 用 `CUBLAS_WORKSPACE_CONFIG=:4096:8`、TE 用 `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`。docs 明说"in the docker `pip uninstall flash_attn_3 -y`"——FA3 在某些版本是非确定性的，必须卸载。
- **`@with_defer` 模式**：`MegatronTrainRayActor.init` 用 `@with_defer(lambda: Timer().start("train_wait"))` 让初始化结束后**自动**进入"等待训练"计时——避免人为忘记在 init 末尾 start timer 的常见 bug。

---

## 5. 集成点

- **训练侧 → metric**：`Timer()` 在 `MegatronTrainRayActor.init/train/log_probs` 等关键方法用 `with timer("..."):` 或 `@timer` 包裹，自动累加到 singleton；`actor.py:553` 一处 `log_perf_data(...)` 把 timer dump 出去；`log_rollout_data` 走 `cp_utils.rollout_log_metric_contribution`，与训练 loss 共享同一 reducer——保证报表数字跟梯度数字落在同一空间。
- **Rollout 侧 → trace**：`sglang_rollout.py` 和 `sglang_streaming_rollout.py` 是仅有的两个文件直接调 `trace_span` / `trace_function`（生产代码里 trace API 调用点不到 10 处）；`generate_and_rm` 按 sample 打点、`generate_and_rm_group` 按 group 打点；reward span 单独 `with trace_span(samples_need_reward, "reward_model"):`。
- **Rollout 侧 → health**：`slime/ray/rollout.py:465-471` 在 init 时为每个 server group 创建 monitor；`get_updatable_engines_and_lock` 之类的 weight sync 入口前后调 `pause/resume`。
- **Customization → trace**：docs 明确说 "custom rollout / reward 逻辑——包括 agentic workflow 里的 agent step、tool call、sandbox 执行、verifier 调用——可以直接复用 `slime.utils.trace_utils`"——这是 trace 设计成 user-facing API 的原因。
- **CI → 训练代码**：`ci_utils.check_mtp_only_grad` 由训练循环在 `args.ci_check_mtp_only_grad` 时调用，是少数从 prod 代码反向调 CI helper 的例子；`_try_ci_fault_injection` (`ray/rollout.py:474`) 在 `--ci-test` 下注入崩溃。
- **多节点 metric 汇总**：训练侧用 dp-with-cp group 的 `dist.all_reduce`（NCCL）；rollout 侧用 `gather_and_reduce_log_dict` 走 `dist.gather_object`（gloo via `init_gloo_group`），因为 rollout 上报需要在 CPU object（dict）上跑而非 tensor。两条管线物理分离。
- **observability 与 deployment 的边界**：slime 不内置 Prometheus，但 `docs/zh/advanced/observability.md` 详细说明怎么用旁路 Prometheus scrape SGLang `/engine_metrics`、TSDB 怎么持久化——把"高频 serving metrics"和"训练侧 W&B/TB metrics"显式分开，避免上传几十万个 sglang:* 点拖垮 wandb。

---

## 6. 出人意料的决策

### 6.1 "trace 比 log 更重要"——因为 RL 的 bug 不报错

slime 把 trace 系统做到 776 行（比许多 utils 加起来还长）有一个具体的痛点：RL 训练中"长尾 sample"和"特定 sample 卡住"是常见故障，但日志只会显示"rollout 慢"——你不知道是哪几条 sample 慢、慢在 generation 还是 reward call、reward call 是 sandbox 慢还是 verifier 慢。trace viewer 直接把每条 sample 摊开，PD 字段还自动拆 `[P]/[D]` 子 lane——这是 log 给不到的视角。代价是每条 sample 多挂一个 dict、Ray object store 多走几 KB，但 throughput 影响可忽略。**反过来设计的两个细节**：(1) `_log_trace_error` 把所有 trace 异常吞掉成 `logger.debug`——trace 失败永远不能阻塞训练；(2) trace 不强制采样，因为 RL 的 sample 数本来就远低于 SFT，全量采样依然便宜。

### 6.2 CPU CI 真敢跑 4-rank 分布式

`tests/_cp_dist_helpers.py` 用 `mp.spawn` + gloo backend + 一个手写的 `megatron.core.mpu` stub module（在导入 `cp_utils` **之前**写进 `sys.modules`），让纯 CPU 的 ubuntu-latest runner 也能跑"dp=2, cp=2"的真实分布式行为——`test_metric_report_dist.py` 会枚举 (dp_size, cp_size) 的多个组合，每个 worker 真实拿到自己那份样本切片、调真实的 `reduce_train_step_metrics` / `gather_and_reduce_log_dict`、写回 rank-0 的结果文件，主进程读出来比对。这件事的反常之处是：很多团队会说"分布式只能在 GPU 上测"，slime 偏要在 CPU 上跑真分布式，原因是这些公式（per-rollout-mean / per-token-loss / cp_size 反向消除）是 RL loss 的核心，一旦错了静默错误代价极大，等到 GPU e2e job 跑完才发现就太晚。`test_loss_cp_invariance.py` 更进一步：用一个 `nn.Linear` 在 CPU 上跑真 backward，验证 grad norm 在不同 (dp, cp) 切分下完全一致——这个测试的 docstring 甚至点名指出"在它存在之前，slime 只有 report-formula 检查，没有 backward 检查，sign 或 factor 错误会漏掉"。

### 6.3 GPU 文件锁——一台机器同时跑多个 CI job

`gpu_lock_exec.py` 不依赖 nvidia-smi 也不依赖任何 GPU runtime，纯 fcntl 文件锁（`/dev/shm/custom_gpu_lock_{gpu_id}.lock`）。原因是自托管 8 卡 runner 上 GitHub Actions 可能同时 dispatch 5 个 job（matrix strategy），每个 job 想要 4 张卡——必须有一个进程外的、不依赖 CUDA SDK 的协调机制。LOCK_EX | LOCK_NB + 退避重试是最简方案。`--print-only` 模式可以在不持锁的情况下探测空闲卡——这是给运维"看看现在哪些卡空着"的工具。

### 6.4 Health monitor 的 "first wait 300s" 不是 magic number

`--rollout-health-check-first-wait` 默认 300 秒，docs 写得很明确："大 MoE 模型首次运行可能需要 kernel compilation"。这是 slime 在生产 MoE 训练中踩过坑的产物——MoE kernel JIT compile 在 first generate 时可能要几分钟，如果 health check 立刻开火会把还在 warming up 的 server 杀掉。`_need_first_wait` 标志在每次 `resume()` 后重置，意味着每次 weight onload 都重新等一轮 first_wait——这是为 weight sync 后可能触发的 kernel re-compile 留的缓冲。

### 6.5 Reproducibility 的"打开方式"是手册而非自动化

slime 的 reproducibility 没有一个 `--enable-reproducibility` 总开关。原因是 bitwise reproducibility 需要同时关掉 5 个不同层的非确定性源头（SGLang FA backend、Megatron deterministic mode、TE non-deterministic algo、NCCL Ring、cuBLAS workspace）+ 卸载 flash_attn_3，每个都有性能代价。slime 选择把这些罗列在 `docs/zh/advanced/reproducibility.md` + 一个完整可跑的 `scripts/run-qwen2.5-0.5B-reproducibility.sh` 里——让用户**显式地**承担确定性的成本，而不是默认开启把人坑哭。这反映了一个理念："reproducibility 是一种调试模式，不是日常配置"。

### 6.6 Fault tolerance 只覆盖 rollout 侧，trainer crash 交给上层

`docs/zh/advanced/fault-tolerance.md` 直白写道："集群级抢占、trainer rank failure 和 full-job resume 仍应由集群调度器、Ray restart policy 和 slime checkpointing 共同处理。" slime 不试图做"trainer 容灾"——因为 trainer rank 死了，整个 NCCL ring 就崩了，最经济的做法是 fail-fast + 重启整个 job + 从 checkpoint resume。slime 只做 rollout 侧的容灾，原因是 SGLang server 是无状态的（除了 KV cache），可以原地重启 + 推一次权重恢复。这种"明确边界"的容灾策略比"什么都自己做"更可维护。

### 6.7 CI 矩阵的 `e2e-test-changed-detect`——为新增测试自动生成 job

workflow 里有一段 `e2e-test-changed`：用 `git diff origin/main...HEAD` 找新增/修改的 `tests/test_*.py`，从每个文件顶部读 `NUM_GPUS = N` 常量动态生成 matrix。这意味着新增一个测试**不需要修改 workflow 文件**就能在 `run-ci-changed` label 下被运行——降低了"写新测试"的摩擦。代价是约定大于配置（必须写 `NUM_GPUS = N`），缺省值 8 容易把 CPU 测试错跑到 GPU 上，所以 docs 反复强调 CPU 测试要写 `NUM_GPUS = 0`。

### 6.8 wandb_always_use_train_step——曲线对齐的微小执着

`metric_utils.compute_rollout_step` 有一个 `if args.wandb_always_use_train_step` 的分支，把 rollout id 换算成 `rollout_id * rollout_batch_size * n_samples_per_prompt // global_batch_size`。这是为了让 rollout 曲线和 train 曲线在 wandb 上能用同一个 step 轴对齐——否则 rollout id 是"第几轮 rollout"，train step 是"第几次 backward"，两条曲线时间轴对不上。这种为可观测性做的微调，是工程基础设施"细致"的另一面。

---

## 附录

### A. 必读文件清单

- `slime/utils/trace_utils.py`（776）：核心，带文件:行号在 §2 表格里
- `slime/utils/metric_utils.py`（123）：pass@k / compute_statistics / compression_ratio / has_repetition / compute_rollout_step
- `slime/utils/train_metric_utils.py`（54）：`log_perf_data_raw` 把 Timer 汇总成 perf/*
- `slime/utils/health_monitor.py`（177）：`RolloutHealthMonitor` 四态机
- `slime/utils/profile_utils.py`（147）：`TrainProfiler` + `_TorchMemoryProfiler` + `_MemrayMemoryProfiler`，OOM observer
- `slime/utils/timer.py`（103）：singleton Timer + `@timer` + `with_defer`
- `slime/utils/distributed_utils.py`（154）：`init_gloo_group` + `init_process_group`（PyTorch fork，支持多 main group）+ `distributed_masked_whiten`
- `slime/backends/megatron_utils/ci_utils.py`（85）：`check_mtp_only_grad` / `check_mtp_loss`
- `tests/ci/gpu_lock_exec.py`（246）：fcntl 文件锁 GPU 分配器
- `tests/ci/README.md`：runner 部署手册
- `.github/workflows/pr-test.yml.j2`：CI matrix 真正的源头
- `.github/workflows/generate_github_workflows.py`：模板渲染脚本
- `tests/_cp_dist_helpers.py`（167）：mp.spawn + gloo + megatron mpu stub
- `tests/test_metric_report.py` / `test_metric_report_dist.py`：报告公式不变量
- `tests/test_loss_cp_invariance.py`：CPU 上的真实 backward grad norm 不变性
- `tests/test_cp_utils.py`：per-rollout reducer 契约
- `tests/test_placement_group.py`：placement layout 10 种 case 表
- `tests/test_chunked_gae.py` / `tests/test_cispo_loss.py`：算法层不变量
- `docs/zh/developer_guide/ci.md` / `debug.md` / `trace.md` / `profiling.md`
- `docs/zh/advanced/reproducibility.md` / `fault-tolerance.md` / `observability.md`
- `scripts/run-qwen2.5-0.5B-reproducibility.sh`：bitwise reproducibility 的完整可跑示例
- `scripts/run-qwen2.5-0.5B-gb10-smoke.sh`：单卡 smoke test

### B. 已扫读但未细讲的文件

- `tests/test_dp_schedule.py`（326）：DP 调度的纯 Python 单测（属 training 子系统，CI 守 schedule 不变性）
- `tools/trace_timeline_viewer.py` / `tools/profile_rollout.py` / `tools/analyze_profile.py`：trace/profile 的可视化工具
- `slime/utils/tensorboard_utils.py` / `wandb_utils.py` / `logging_utils.py`：log 后端适配，三选一统一接口
- `slime/utils/train_dump_utils.py`：`--save-debug-rollout-data` 的实现
- `slime/utils/memory_utils.py`：`print_memory` 等 GPU memory helper（被 OOM observer 调用）
- `slime/backends/megatron_utils/data.py`（`log_rollout_data` / `log_perf_data` 在这）：跨在 backends 与工程基础设施之间，主体归 backends 子系统
- `tests/github_runner/docker-compose.yml`：runner 容器化配置
- `tests/ci/github_runner/.env.example`：runner 环境变量样板
- `.github/workflows/bot-slash-lint.yaml` / `conda-ci.yml` / `pre-commit.yml` / `release-docs.yaml`：辅助 workflow，非核心矩阵

### C. 明确不在本子系统的内容

- CP/loss 反向放缩的数学含义（mb/gbs/CP/DP 四步组合）→ training 子系统笔记 §3
- `rollout_log_metric_contribution` 的具体公式 → training / Data Buffer 子系统
- 18 个 customization hook 的契约 → customization 子系统
- `tests/plugin_contracts/` → customization 子系统
- weight sync 失败的具体重试策略 → weight sync 子系统
- placement group 的 GPU 分配策略 → deployment 子系统
