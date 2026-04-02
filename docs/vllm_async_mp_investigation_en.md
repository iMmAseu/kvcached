# vLLM Async Scheduling + MP in kvcached

At first glance, this looked like a small integration detail: once `vLLM --async-scheduling` is enabled, does `kvcached` really follow the same path? But once the investigation moved to the `mp` backend, the core issue quickly stopped being "did one flag get forwarded?" and became "which process is actually allowed to touch KV tensors?"

This write-up keeps the conclusions and the key data, but trims the story down to three steps: establish the `uni` baseline, isolate the real root cause under `mp + tp=1`, and then validate the fixed behavior on both single-GPU and dual-GPU runs.

## The Short Version

At this point, the investigation and follow-up tests support five conclusions:

- `--async-scheduling` and `--distributed-executor-backend` are independent knobs. One changes the scheduler; the other changes the execution model.
- On the `uni` path, async scheduling itself already provides the main benefit. Forwarding `async_sched` into `kvcached` preserves that gain and adds a small extra improvement.
- On the `mp` path, the real bug was not the `async_sched` flag itself. The real issue was that some KV tensor checks and `map/unmap` operations were still being executed in the wrong process.
- After the fix, `mp + tp=1 + --async-scheduling` starts cleanly, logs clearly show async scheduling on both the vLLM side and the `kvcached` side, and the benchmark results are complete.
- The `mp + tp=2` functional path is already validated, but the current dual-GPU async benchmark file is not a valid performance-comparison sample because the client started benchmarking before the service finished startup, which caused many `ConnectionRefused` failures.

## Why the Real Issue Only Shows Up on `mp`

The most important boundary is to keep these two concepts separate:

- `--async-scheduling` only changes the scheduler.
- `--distributed-executor-backend mp` is what splits vLLM into `EngineCore` and `Worker` processes.

That immediately changes the debugging model:

- On the `uni` path, this is mostly a flag-propagation and performance-observation problem.
- On the `mp` path, this becomes a process-boundary problem: does the current process really own the KV tensors it is trying to inspect or mutate?

That is why `uni` is a useful control case, but `mp` is where the actual fix matters.

## The `uni` Baseline

### Environment Variables

```bash
export MODEL=/root/data/models/Qwen/Qwen3-0.6B
export ENABLE_KVCACHED=true
export KVCACHED_AUTOPATCH=1
export KVCACHED_LOG_LEVEL=INFO
export VLLM_ENABLE_V1_MULTIPROCESSING=0
```

### Server Commands

Async disabled:

```bash
vllm serve "$MODEL" \
  --distributed-executor-backend uni \
  --no-async-scheduling \
  --tensor-parallel-size 1 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.8 \
  --no-enable-prefix-caching \
  --disable-log-requests \
  --port 18000 2>&1 | tee uni_no_async.log
```

Async enabled, before `kvcached` forwarded `async_sched`:

```bash
vllm serve "$MODEL" \
  --distributed-executor-backend uni \
  --async-scheduling \
  --tensor-parallel-size 1 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.8 \
  --no-enable-prefix-caching \
  --disable-log-requests \
  --port 18000 2>&1 | tee old_async.log
```

Async enabled, with `kvcached` also enabling `async_sched`:

```bash
vllm serve "$MODEL" \
  --distributed-executor-backend uni \
  --async-scheduling \
  --tensor-parallel-size 1 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.8 \
  --no-enable-prefix-caching \
  --disable-log-requests \
  --port 18000 2>&1 | tee new_async.log
```

### Benchmark Command

```bash
vllm bench serve \
  --backend openai \
  --base-url http://127.0.0.1:18000 \
  --endpoint /v1/completions \
  --model "$MODEL" \
  --dataset-name random \
  --input-len 512 \
  --output-len 128 \
  --num-prompts 200 \
  --request-rate 8 \
  --max-concurrency 16
```

### Results

Result files:

- `results/bench_no_async.txt`
- `results/bench_old_async.txt`
- `results/bench_new_async.txt`

| Case | req/s | total tok/s | mean TTFT (ms) | p99 TTFT (ms) | mean TPOT (ms) | p99 TPOT (ms) | mean ITL (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no async | 7.880 | 5043.05 | 26.68 | 48.33 | 3.16 | 4.18 | 3.16 |
| old async | 7.923 | 5071.39 | 27.39 | 53.83 | 1.88 | 2.83 | 1.88 |
| new async | 7.927 | 5072.77 | 26.78 | 50.56 | 1.81 | 2.53 | 1.81 |

The main observation is simple:

- Throughput barely moves.
- The meaningful benefit appears in token-level latency.
- `TPOT/ITL` in `old async` improve by roughly 40% relative to `no async`.
- Forwarding `async_sched` into `kvcached` preserves that gain and adds a small extra improvement.

That baseline is important because it shows async scheduling itself is valid. The later `mp` issue is therefore not "async is broken" but "the multiprocessing path is using the wrong process semantics."

## The Real Root Cause Under `mp + tp=1`

### The Two Early Symptoms

The first visible error was a compatibility issue:

```text
Failed to patch kv_cache_coordinator: 'KVCacheCoordinatorNoPrefixCache' object has no attribute 'vllm_config'
```

That was fixed with:

```python
_should_enable_async_sched(getattr(self, "vllm_config", None))
```

and by defining `_should_enable_async_sched(None) -> False`.

The real root-cause signal was this later error:

```text
ERROR: /root/kvcached/csrc/allocator.cpp:153: try to map to KV tensors when KV tensors are not created
```

This already tells the whole story:

- `kvcached` inside `EngineCore` was active
- it tried to run `map_to_kv_tensors()` locally
- but the process that actually created the KV tensors was the `Worker`

So the problem was not "KV tensors do not exist at all."  
The problem was "this process is not the one that owns them."

### The Root Cause in One Sentence

The old logic implicitly assumed:

- `world_size == 1`
- therefore local KV tensor access is safe

But under `mp + tp=1`, that is false.  
`world_size == 1` only tells us the tensor-parallel size. It does not tell us whether the current process actually owns the FTensors / KV tensors.

## The Fix: Unify the Meaning of "This Must Use Worker IPC"

The real fix was not to add more random conditionals. The correct approach was to define one semantic:

- if the current process is not the worker, it should not assume local KV tensor ownership

That semantic is then reused both for readiness checks and for `map/unmap`.

### 1. Add a Shared Decision: `should_use_worker_ipc()`

File:

- `kvcached/integration/vllm/interfaces.py`

Added:

```python
def should_use_worker_ipc() -> bool:
    return _kvcached_initialized and not _is_worker
```

The important part here is the meaning of `_is_worker`:

- it indicates whether the current process is a vLLM worker process

So the real meaning of this helper is:

- `kvcached` is initialized
- but the current process is not a worker
- therefore KV tensor access should go through worker IPC instead of assuming local ownership

### 2. Why Repeated `init_kvcached()` Calls Need to Upgrade State

There is another important block in [interfaces.py](d:/proj/kvcached/kvcached_async/kvcached/integration/vllm/interfaces.py):

```python
if is_worker and not _is_worker:
    _is_worker = True
if async_sched and not _async_sched:
    _async_sched = True
    logger.info("kvcached async scheduler enabled")
```

This is not a second initialization. It is state correction on repeated init calls.

There are two kinds of names here:

- `is_worker` / `async_sched`
  the facts provided by this specific `init_kvcached()` call
- `_is_worker` / `_async_sched`
  the global state already remembered inside the current process

These updates are intentionally one-way:

- if a later call proves this process is actually a worker, `_is_worker` is upgraded from `False` to `True`
- if a later call proves async scheduling is enabled, `_async_sched` is upgraded from `False` to `True`

Without this, the process could remain stuck with incomplete earlier state, even after later calls provide the correct role information.

### 3. Why the "KV Tensors Created" Check Also Had to Change

This is exactly where the `FTensorAllocator.num_layers == 0` issue belongs.

The old problem was:

- `KVCacheManager._post_init()` waits until KV tensors are created
- if it runs `kv_tensors_created()` locally inside `EngineCore`
- it checks the allocator state inside the current process
- but under `mp`, `FTensorAllocator` is initialized inside the `Worker`, not inside `EngineCore`
- so the local state can still look like `num_layers == 0`
- and `kvcached` incorrectly concludes that KV tensors have not been created yet

That is why this logic was changed to:

```python
try:
    from kvcached.integration.vllm.interfaces import should_use_worker_ipc
    vllm_remote = should_use_worker_ipc()
except ImportError:
    vllm_remote = False

if self.world_size > 1 or vllm_remote:
    return broadcast_kv_tensors_created(...)
else:
    return kv_tensors_created(...)
```

The meaning is:

- in a normal local case, keep checking locally
- but in a non-worker vLLM process, stop checking local state
- instead, ask the worker over IPC

### 4. Why `broadcast_kv_tensors_created(...)` Can Query the Worker State

It does not "know" the state by itself. It sends an IPC request to the worker and lets the worker answer.

Inside `tp_ipc_util.py`, it sends:

```python
{"cmd": "kv_tensors_created", "group_id": group_id}
```

The worker receives that message and executes:

```python
created = kv_tensors_created(group_id=group_id)
```

The important part is where that call runs:

- not inside `EngineCore`
- inside the worker process

So the queried state is the worker's own `FTensorAllocator` / KV tensor state, not the empty local state inside `EngineCore`.

That is exactly why this fixes the case where the current process still sees `num_layers == 0` and would otherwise falsely conclude that KV tensors are missing.

This also works for `tp=1`:

- `tp_size = 1`
- the IPC request is sent only to `rank=0`

So it becomes "ask the only worker process directly."

### 5. Why the `map/unmap` Branch Is the Core Runtime Fix

File:

- `kvcached/page_allocator.py`

The key change is:

```python
if self.world_size > 1 or _should_use_worker_ipc():
    broadcast_map_to_kv_tensors(...)
else:
    map_to_kv_tensors(...)
```

and similarly for unmap:

```python
if self.world_size > 1 or _should_use_worker_ipc():
    broadcast_unmap_from_kv_tensors(...)
else:
    unmap_from_kv_tensors(...)
```

This is very close to the most important runtime behavior change in the whole fix.

Why:

- the old logic only checked `world_size > 1`
- under `mp + tp=1`, that caused the local `map_to_kv_tensors()` branch to run
- but the actual KV tensors belonged to the `Worker`, not the `EngineCore`

So `broadcast_map_to_kv_tensors(...)` is not solving an "offset conflict."  
It is solving a "the operation was sent to the wrong process" conflict.

### 6. Why `broadcast_map_to_kv_tensors(...)` Still Works in Single-GPU Mode

It does not operate using only `world_size / pp_rank / group_id`. The most important payload also includes:

- `offsets`

which are the page-memory offsets that must be mapped.

Internally it:

- constructs a message

```python
{"cmd": "map_to_kv_tensors", "offsets": offsets, "group_id": group_id}
```

- then sends that message to each rank:

```python
for rank in range(tp_size)
```

So when `tp=1`:

- `range(1)` means only rank 0
- the message is sent to the only worker

The worker then executes:

```python
map_to_kv_tensors(msg["offsets"], group_id=group_id)
```

So the real flow is:

- `EngineCore` decides which offsets need to be mapped
- but it no longer maps them locally
- it sends the command to the worker that actually owns the KV tensors

That is why `broadcast` here really means "use the worker-side IPC path." In the single-worker case, the broadcast simply degenerates into a single worker message.

## Validation After the Fix

### Unit Tests

```bash
PYTHONPATH=d:\proj\kvcached\kvcached_async pytest tests\test_vllm_async_sched.py -q
```

Result:

```text
3 passed
```

### `mp + tp=1` Startup and Benchmark

From `results/mp_new_async.log`, the following are now clearly visible:

- `world_size=1`
- `Worker 0 IPC listener started`
- `kvcached async scheduler enabled`
- `Init kvcached KV cache allocator: ... world_size=1, async_sched=True`
- `Application startup complete.`

The earlier failure is no longer present:

```text
try to map to KV tensors when KV tensors are not created
```

Relevant benchmark files:

- `results/mp_tp1_bench_no_async.txt`
- `results/mp_tp1_bench_new_async.txt`

| Case | Success / Fail | req/s | total tok/s | mean TTFT (ms) | p99 TTFT (ms) | mean TPOT (ms) | p99 TPOT (ms) | mean ITL (ms) | p99 ITL (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mp tp=1 no async | 200 / 0 | 7.87 | 5037.54 | 26.63 | 50.58 | 3.16 | 4.08 | 3.16 | 15.56 |
| mp tp=1 new async | 200 / 0 | 7.93 | 5072.18 | 25.78 | 46.45 | 1.90 | 2.63 | 1.90 | 13.80 |

The conclusion is straightforward:

- throughput barely changes
- the meaningful async gain still appears in token-level latency
- `TPOT/ITL` improve from `3.16ms` to `1.90ms`, again close to a 40% reduction

## Dual-GPU Follow-Up: `mp + tp=2`

### What Is Already Confirmed

Relevant files:

- `results/mp_tp2_no_async.log`
- `results/mp_tp2_new_async.log`

From `results/mp_tp2_new_async.log`, we can confirm:

- vLLM prints `Asynchronous scheduling is enabled.`
- `world_size=2`
- both worker listeners are started
- the allocator log shows `world_size=2, async_sched=True`
- the service eventually prints `Application startup complete.`

That is enough to say the dual-GPU async startup and service-registration path works.

### How to Interpret the Current Dual-GPU Benchmark

The clean baseline sample is:

- `results/mp_tp2_bench_no_async.txt`

| Case | Success / Fail | req/s | total tok/s | mean TTFT (ms) | p99 TTFT (ms) | mean TPOT (ms) | p99 TPOT (ms) | mean ITL (ms) | p99 ITL (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mp tp=2 no async | 200 / 0 | 7.89 | 5049.25 | 31.90 | 56.59 | 3.21 | 5.55 | 3.21 | 20.52 |

The current async sample:

- `results/mp_tp2_bench_new_async.txt`

should not be treated as a final performance-comparison sample:

| Case | Success / Fail | req/s | total tok/s | mean TTFT (ms) | p99 TTFT (ms) | mean TPOT (ms) | p99 TPOT (ms) | mean ITL (ms) | p99 ITL (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mp tp=2 new async | 45 / 155 | 1.78 | 1141.20 | 34.54 | 60.72 | 2.00 | 2.62 | 2.00 | 20.13 |

The reason is not simply that the numbers are worse. This sample contains many:

```text
ConnectionRefusedError: [Errno 111] Connect call failed ('127.0.0.1', 18000)
Cannot connect to host 127.0.0.1:18000
```

And the timestamps show:

- the benchmark started at 2026-04-02 14:05:19
- the service reached `Application startup complete.` only at 2026-04-02 14:05:39

So the more reasonable interpretation is:

- the client started too early
- the current dual-GPU async sample is an invalid performance sample
- it proves the service eventually came up and handled requests, but it does not support a formal async-vs-no-async performance conclusion

## Recommended Validation Workflow

### Log Checks

```bash
grep -n "kvcached async scheduler enabled" mp_async.log
grep -n "Init kvcached KV cache allocator" mp_async.log
grep -n "async_sched=True" mp_async.log
grep -n "Asynchronous scheduling is enabled" mp_async.log
grep -Ei "try to map to KV tensors|Failed to patch kv_cache_coordinator|ERROR|Traceback" mp_async.log
```

For `tp=2`, also verify:

```bash
grep -n "world_size=2" mp_tp2_new_async.log
grep -n "Worker 0 IPC listener started" mp_tp2_new_async.log
grep -n "Worker 1 IPC listener started" mp_tp2_new_async.log
```

### Benchmark Command

```bash
vllm bench serve \
  --backend openai \
  --base-url http://127.0.0.1:18000 \
  --endpoint /v1/completions \
  --model "$MODEL" \
  --dataset-name random \
  --input-len 512 \
  --output-len 128 \
  --num-prompts 200 \
  --request-rate 8 \
  --max-concurrency 16
```

If you want a valid dual-GPU async performance sample, the most important thing is not changing the benchmark parameters. The important thing is:

- wait until the log explicitly shows `Application startup complete.`
- then start the benchmark, or enable endpoint readiness checking

## What This Fix Actually Taught Us

On the surface, this looked like "async scheduling breaks `mp`." The deeper lesson is simpler:

- `world_size` only describes parallel scale
- it does not tell us whether the current process owns the KV tensors

For `kvcached`, the important distinction is:

- does this process really own the local KV tensors?
- or must it inspect and mutate them through worker IPC?

Once that semantic is made explicit, readiness checks, `map/unmap`, logging, and regression tests all line up naturally.

At this point, the correct status summary is:

- the `uni` path is healthy and does not reproduce the `mp` startup issue
- `mp + tp=1 + --async-scheduling` has been fixed
- the `mp + tp=1` benchmark shows the async gain mainly in token-level latency
- `mp + tp=2` has passed startup and functional validation
- dual-GPU async performance still needs one clean rerun after endpoint readiness

## Related Files

- `kvcached/integration/vllm/patches.py`
- `kvcached/integration/vllm/interfaces.py`
- `kvcached/kv_cache_manager.py`
- `kvcached/page_allocator.py`
- `tests/test_vllm_async_sched.py`
- `docs/vllm_async_mp_investigation_en.md`

## Related Artifacts

- `results/bench_no_async.txt`
- `results/bench_old_async.txt`
- `results/bench_new_async.txt`
- `results/mp_async.log`
- `results/mp_no_async.log`
- `results/mp_new_async.log`
- `results/mp_tp1_bench_no_async.txt`
- `results/mp_tp1_bench_new_async.txt`
- `results/mp_tp2_no_async.log`
- `results/mp_tp2_new_async.log`
- `results/mp_tp2_bench_no_async.txt`
- `results/mp_tp2_bench_new_async.txt`
- `results/error_info.png`
- `results/no_async_info.png`
- `results/old_version_info.png`
