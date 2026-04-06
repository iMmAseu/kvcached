# vLLM V1 Async Scheduler

> Based on the current workspace implementation on 2026-04-06.

## Overview

The V1 async scheduler is not a separate scheduling subsystem. It is an asynchronous execution path layered on top of the V1 `Scheduler`, designed to keep GPU work overlapped with CPU-side scheduling and bookkeeping.

The implementation has three layers:

1. Config maps `async_scheduling=True` to `AsyncScheduler`.
2. `EngineCore` and the executor allow multiple in-flight batches.
3. Scheduler and worker keep request state coherent with placeholders, async output wrappers, and queues.

## Enablement

The entry point is `SchedulerConfig.async_scheduling`:

- `False`: explicitly disabled.
- `True`: explicitly enabled, with fail-fast incompatibility checks.
- `None`: auto-detected by `VllmConfig`.

In the current code:

1. `SchedulerConfig.get_scheduler_cls()` returns `vllm.v1.core.sched.async_scheduler.AsyncScheduler` when enabled.
2. `VllmConfig` validates executor support and speculative-decoding compatibility.
3. The CPU backend forces `scheduler_config.async_scheduling=False`.

## Compatibility

Directly visible in the code:

1. Base `Executor.supports_async_scheduling()` returns `False`.
2. `UniProcExecutor`, `ExecutorWithExternalLauncher`, and `MultiprocExecutor` explicitly support it.
3. Speculative decoding is limited to the current allowlist.
4. `disable_padded_drafter_batch=True` is incompatible with async scheduling.
5. Async speculative decoding auto-disables cascade attention.

## Runtime Flow

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

`EngineCore` is the coordinator. It selects the scheduler class, reads executor batch concurrency, and separates scheduling from result collection in `step()` or `step_with_batch_queue()`.

## Batch Queue

The main runtime change is support for multiple in-flight batches:

1. `UniProcExecutor.max_concurrent_batches` is typically `2` in async mode.
2. `MultiprocExecutor.max_concurrent_batches` is also typically `2` without PP.
3. With pipeline parallelism, the concurrency becomes `pp_size` because filling the pipeline already requires multiple batches.

## Placeholder Model

The key addition in `AsyncScheduler` is `num_output_placeholders`.

When a request has been scheduled but its real output tokens have not yet fully returned, the scheduler still advances logical output length by reserving placeholders. Speculative tokens are also counted as placeholders.

This is what allows “results not fully collected yet” and “logical length already advanced” to both be true.

## Request Bookkeeping

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

## Structured Output And Spec Decode

Async scheduling has explicit handling for both features:

1. Speculative tokens reserve placeholders first and get real token IDs later on the worker side.
2. If draft tokens are rejected, the scheduler rolls back `num_computed_tokens`, and in async mode also rolls back `num_output_placeholders`.
3. When `SchedulerOutput.pending_structured_output_tokens=True`, `EngineCore.step_with_batch_queue()` defers sampling because grammar-bitmask computation still depends on real tokens from the prior step.

## Worker Async Output Path

Workers also have an async output path:

1. `AsyncModelRunnerOutput` means the final result may still need later host-side materialization.
2. `UniProcExecutor` runs `get_output()` in a background thread.
3. `MultiprocExecutor.WorkerProc` starts `async_output_copy_thread` in async mode before enqueueing the result.

## Pipeline Parallelism

Under `PP + async scheduling`, a request can be scheduled again even while it still carries `num_output_placeholders`. Tests explicitly cover this behavior. The scheduler also stops relying on the same token-return payload used by non-async PP.

## Suggested Source Reading Order

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

## Summary

The essence of the V1 async scheduler is not simply “put the scheduler on another thread”. It is the combination of multiple in-flight batches, output placeholders, async output handling, and feature-specific compatibility logic that lets CPU scheduling overlap with GPU execution.
