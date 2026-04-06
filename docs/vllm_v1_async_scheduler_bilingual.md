# vLLM V1 Async Scheduler / 异步调度器

> Based on the current workspace implementation on 2026-04-06.  
> 基于当前工作区在 2026-04-06 的实现。

## Overview / 概述

### 中文

vLLM V1 的 async scheduler 不是一个完全独立的新系统，而是在 V1 `Scheduler` 之上加出的异步执行路径。它的目标是让 CPU 继续准备后续调度和状态更新时，GPU 不必因为等待结果回收而空转。

核心思路有三层：

1. 配置层把 `async_scheduling=True` 映射到 `AsyncScheduler`。
2. `EngineCore` 和 executor 允许多个 batch 同时 in-flight。
3. scheduler 与 worker 通过 placeholder、异步输出包装和队列维持请求状态一致。

### English

The V1 async scheduler is not a separate scheduling subsystem. It is an asynchronous execution path layered on top of the V1 `Scheduler`, designed to keep GPU work overlapped with CPU-side scheduling and bookkeeping.

The implementation has three layers:

1. Config maps `async_scheduling=True` to `AsyncScheduler`.
2. `EngineCore` and the executor allow multiple in-flight batches.
3. Scheduler and worker keep request state coherent with placeholders, async output wrappers, and queues.

## Enablement / 启用方式

### 中文

入口是 `SchedulerConfig.async_scheduling`：

- `False`：显式关闭。
- `True`：显式开启，不兼容时直接报错。
- `None`：由 `VllmConfig` 自动判定，兼容时自动开启。

当前代码中：

1. `SchedulerConfig.get_scheduler_cls()` 在开启时返回 `vllm.v1.core.sched.async_scheduler.AsyncScheduler`。
2. `VllmConfig` 会检查 executor、speculative decoding 等兼容性。
3. CPU backend 会直接把 `scheduler_config.async_scheduling` 设为 `False`。

### English

The entry point is `SchedulerConfig.async_scheduling`:

- `False`: explicitly disabled.
- `True`: explicitly enabled, with fail-fast incompatibility checks.
- `None`: auto-detected by `VllmConfig`.

In the current code:

1. `SchedulerConfig.get_scheduler_cls()` returns `vllm.v1.core.sched.async_scheduler.AsyncScheduler` when enabled.
2. `VllmConfig` validates executor support and speculative-decoding compatibility.
3. The CPU backend forces `scheduler_config.async_scheduling=False`.

## Compatibility / 兼容性

### 中文

从代码可直接确认：

1. 基类 `Executor.supports_async_scheduling()` 返回 `False`。
2. `UniProcExecutor`、`ExecutorWithExternalLauncher`、`MultiprocExecutor` 显式支持。
3. speculative decoding 只支持当前白名单方法。
4. `disable_padded_drafter_batch=True` 与 async scheduling 不兼容。
5. async speculative decoding 会自动关闭 cascade attention。

### English

Directly visible in the code:

1. Base `Executor.supports_async_scheduling()` returns `False`.
2. `UniProcExecutor`, `ExecutorWithExternalLauncher`, and `MultiprocExecutor` explicitly support it.
3. Speculative decoding is limited to the current allowlist.
4. `disable_padded_drafter_batch=True` is incompatible with async scheduling.
5. Async speculative decoding auto-disables cascade attention.

## Runtime Flow / 运行流程

```text
Request
  -> Scheduler / AsyncScheduler
  -> SchedulerOutput
  -> Executor.execute_model(non_block=True)
  -> Worker / ModelRunner
  -> ModelRunnerOutput or AsyncModelRunnerOutput
  -> Scheduler.update_from_output(...)
  -> next schedule()
```

### 中文

`EngineCore` 是中枢。它选择 scheduler 类、读取 executor 的并发 batch 能力，并在 `step()` 或 `step_with_batch_queue()` 中把调度和结果回收错开。

### English

`EngineCore` is the coordinator. It selects the scheduler class, reads executor batch concurrency, and separates scheduling from result collection in `step()` or `step_with_batch_queue()`.

## Batch Queue / 批次队列

### 中文

异步模式下最重要的运行时变化是允许多个 batch 同时在飞：

1. `UniProcExecutor.max_concurrent_batches` 在 async 模式下通常为 `2`。
2. `MultiprocExecutor.max_concurrent_batches` 在无 PP 时通常也为 `2`。
3. 若启用 pipeline parallelism，则并发 batch 数变成 `pp_size`，因为填满流水线本身就需要多个 batch。

### English

The main runtime change is support for multiple in-flight batches:

1. `UniProcExecutor.max_concurrent_batches` is typically `2` in async mode.
2. `MultiprocExecutor.max_concurrent_batches` is also typically `2` without PP.
3. With pipeline parallelism, the concurrency becomes `pp_size` because filling the pipeline already requires multiple batches.

## Placeholder Model / 占位模型

### 中文

`AsyncScheduler` 相比普通 `Scheduler` 最关键的新增，是 `num_output_placeholders`。

当 request 已经被调度、但本轮真实输出 token 还没完全回到账本时，scheduler 会先把逻辑上的输出长度记上去。若本轮包含 speculative tokens，也会先占位。

这让“输出尚未完全回收”和“逻辑长度已经增长”可以同时成立。

### English

The key addition in `AsyncScheduler` is `num_output_placeholders`.

When a request has been scheduled but its real output tokens have not yet fully returned, the scheduler still advances logical output length by reserving placeholders. Speculative tokens are also counted as placeholders.

This is what allows “results not fully collected yet” and “logical length already advanced” to both be true.

## Request Bookkeeping / 请求记账

### 中文

理解 async scheduling 时，最关键的字段在 `vllm/v1/request.py`：

1. `num_output_placeholders`
2. `discard_latest_async_tokens`
3. `spec_token_ids`
4. `num_computed_tokens`

关键细节是：发送给 worker 的 `CachedRequestData.num_output_tokens` 使用的是 `req.num_output_tokens + req.num_output_placeholders`。

另外，在 `reset_prefix_cache(reset_running_requests=True)` 的强制 preempt 场景下，代码会设置：

1. `request.num_output_placeholders = 0`
2. `request.discard_latest_async_tokens = True`

后续 `AsyncScheduler._update_request_with_output()` 会丢弃那一个不该保留的异步 token，防止恢复后重复输出。

### English

The most important fields for understanding async scheduling live in `vllm/v1/request.py`:

1. `num_output_placeholders`
2. `discard_latest_async_tokens`
3. `spec_token_ids`
4. `num_computed_tokens`

One crucial detail is that worker-facing `CachedRequestData.num_output_tokens` uses `req.num_output_tokens + req.num_output_placeholders`.

Also, in forced preemption via `reset_prefix_cache(reset_running_requests=True)`, the code sets:

1. `request.num_output_placeholders = 0`
2. `request.discard_latest_async_tokens = True`

Then `AsyncScheduler._update_request_with_output()` discards the async token that should no longer be kept, preventing duplicated output after resumption.

## Structured Output And Spec Decode / 结构化输出与投机解码

### 中文

async scheduling 对这两类能力都做了专门适配：

1. speculative token 会先占位，稍后由 worker 补写真正 token id。
2. 如果 draft token 被拒绝，scheduler 会回退 `num_computed_tokens`，async 场景下也会回退 `num_output_placeholders`。
3. `SchedulerOutput.pending_structured_output_tokens=True` 时，`EngineCore.step_with_batch_queue()` 会延后采样，因为 grammar bitmask 还依赖前一轮真实 token。

### English

Async scheduling has explicit handling for both features:

1. Speculative tokens reserve placeholders first and get real token IDs later on the worker side.
2. If draft tokens are rejected, the scheduler rolls back `num_computed_tokens`, and in async mode also rolls back `num_output_placeholders`.
3. When `SchedulerOutput.pending_structured_output_tokens=True`, `EngineCore.step_with_batch_queue()` defers sampling because grammar-bitmask computation still depends on real tokens from the prior step.

## Worker Async Output Path / Worker 侧异步输出路径

### 中文

worker 端也有异步输出路径：

1. `AsyncModelRunnerOutput` 表示结果稍后才能真正取出，可能仍在做 host copy。
2. `UniProcExecutor` 会把 `get_output()` 丢给后台线程。
3. `MultiprocExecutor.WorkerProc` 在 async 模式下会启动 `async_output_copy_thread`，再把结果送入消息队列。

### English

Workers also have an async output path:

1. `AsyncModelRunnerOutput` means the final result may still need later host-side materialization.
2. `UniProcExecutor` runs `get_output()` in a background thread.
3. `MultiprocExecutor.WorkerProc` starts `async_output_copy_thread` in async mode before enqueueing the result.

## Pipeline Parallelism / 流水线并行

### 中文

`PP + async scheduling` 下，请求即使还带着 `num_output_placeholders`，下一步仍然可以再次被调度。测试已经覆盖了这一点。对应地，scheduler 在这一路径下也不再依赖非 async PP 的那套 token 回传方式。

### English

Under `PP + async scheduling`, a request can be scheduled again even while it still carries `num_output_placeholders`. Tests explicitly cover this behavior. The scheduler also stops relying on the same token-return payload used by non-async PP.

## What To Read / 推荐阅读源码

### 中文

建议按以下顺序阅读：

1. `vllm/config/scheduler.py`
2. `vllm/config/vllm.py`
3. `vllm/v1/core/sched/async_scheduler.py`
4. `vllm/v1/core/sched/scheduler.py`
5. `vllm/v1/request.py`
6. `vllm/v1/core/sched/output.py`
7. `vllm/v1/engine/core.py`
8. `vllm/v1/executor/uniproc_executor.py`
9. `vllm/v1/executor/multiproc_executor.py`
10. `tests/v1/core/test_async_scheduler.py`

### English

Suggested reading order:

1. `vllm/config/scheduler.py`
2. `vllm/config/vllm.py`
3. `vllm/v1/core/sched/async_scheduler.py`
4. `vllm/v1/core/sched/scheduler.py`
5. `vllm/v1/request.py`
6. `vllm/v1/core/sched/output.py`
7. `vllm/v1/engine/core.py`
8. `vllm/v1/executor/uniproc_executor.py`
9. `vllm/v1/executor/multiproc_executor.py`
10. `tests/v1/core/test_async_scheduler.py`

## Summary / 总结

### 中文

V1 async scheduler 的本质不是“把 scheduler 放到另一个线程”，而是通过多 in-flight batch、output placeholders、异步输出路径和特性兼容分支，让 CPU 调度与 GPU 执行真正重叠。

### English

The essence of the V1 async scheduler is not simply “put the scheduler on another thread”. It is the combination of multiple in-flight batches, output placeholders, async output handling, and feature-specific compatibility logic that lets CPU scheduling overlap with GPU execution.
