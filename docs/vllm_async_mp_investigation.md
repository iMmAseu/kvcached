# vLLM Async Scheduling + MP in kvcached

这次排查最开始看起来只是一个集成细节：`vLLM --async-scheduling` 打开之后，`kvcached` 是否真的跟上了这条调度路径？但真正把问题放到 `mp` 后端上之后，核心矛盾很快从“参数有没有透传”变成了“当前这个进程到底能不能直接碰 KV tensors”。

本文保留完整结论和关键数据，但把主线尽量压缩成三件事：先用 `uni` 路径确认 async scheduling 的收益，再定位 `mp + tp=1` 的真实根因，最后给出修复后的单卡与双卡验证结果。

## 结论先行

这轮调研和补测可以归纳为五点：

- `--async-scheduling` 和 `--distributed-executor-backend` 是两个独立维度。前者控制调度模式，后者控制执行进程模型。
- `uni` 路径下，async scheduling 本身已经带来主要收益；`kvcached` 继续透传 `async_sched` 后，收益仍在，但更多是小幅增益。
- `mp` 路径下，真正的问题不在 `async_sched` 透传本身，而在于 `EngineCore` 和 `Worker` 分进程后，某些 KV tensor 检查和 `map/unmap` 操作仍然错误地在当前进程本地执行。
- 修复后，`mp + tp=1 + --async-scheduling` 已可以稳定启动，日志能确认 vLLM 和 `kvcached` 两侧的 async 调度都已生效，而且 benchmark 结果完整。
- `mp + tp=2` 的功能链路已经跑通，但当前仓库中的双卡 async benchmark 样本不是正式性能样本，因为客户端在服务完全 ready 前就开始压测，导致大量 `ConnectionRefused`。

## 为什么问题会出现在 `mp`

最重要的边界是这两个概念必须分开看：

- `--async-scheduling` 只是调度开关。
- `--distributed-executor-backend mp` 才会让 vLLM 的 `EngineCore` 和 `Worker` 分进程运行。

这意味着：

- `uni` 路径主要是参数透传和性能观察问题。
- `mp` 路径会额外引入“当前进程是否真的拥有 KV tensors”这个问题。

也正因为如此，`uni` 适合做对照实验，`mp` 才是这次修复真正要解决的路径。

## `uni` 路径的基线结论

### 测试环境变量

```bash
export MODEL=/root/data/models/Qwen/Qwen3-0.6B
export ENABLE_KVCACHED=true
export KVCACHED_AUTOPATCH=1
export KVCACHED_LOG_LEVEL=INFO
export VLLM_ENABLE_V1_MULTIPROCESSING=0
```

### 服务端命令

关闭 async：

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

开启 async，但当时 `kvcached` 还没有透传 `async_sched`：

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

开启 async，且 `kvcached` 也同步开启 `async_sched`：

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

### benchmark 命令

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

### 结果

结果文件：

- `results/bench_no_async.txt`
- `results/bench_old_async.txt`
- `results/bench_new_async.txt`

| Case | req/s | total tok/s | mean TTFT (ms) | p99 TTFT (ms) | mean TPOT (ms) | p99 TPOT (ms) | mean ITL (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no async | 7.880 | 5043.05 | 26.68 | 48.33 | 3.16 | 4.18 | 3.16 |
| old async | 7.923 | 5071.39 | 27.39 | 53.83 | 1.88 | 2.83 | 1.88 |
| new async | 7.927 | 5072.77 | 26.78 | 50.56 | 1.81 | 2.53 | 1.81 |

这里最重要的观察很简单：

- 吞吐变化不大。
- async 的主要收益体现在 token 级延迟。
- `TPOT/ITL` 在 `old async` 相比 `no async` 大约下降 40%。
- `kvcached` 继续透传 `async_sched` 后，收益仍然存在，但已经从“主要变化”变成了“小幅增益”。

这组结果给出的信号很明确：async scheduling 本身是有效的，问题不在 `uni`，而在 `mp` 的进程边界。

## `mp + tp=1` 的真实根因

### 早期暴露的两个现象

第一个报错是兼容性问题：

```text
Failed to patch kv_cache_coordinator: 'KVCacheCoordinatorNoPrefixCache' object has no attribute 'vllm_config'
```

这部分后来通过：

```python
_should_enable_async_sched(getattr(self, "vllm_config", None))
```

以及 `_should_enable_async_sched(None) -> False` 解决。

真正的核心错误是后面这条：

```text
ERROR: /root/kvcached/csrc/allocator.cpp:153: try to map to KV tensors when KV tensors are not created
```

这条日志的含义是：

- `EngineCore` 里的 `kvcached` allocator 已经启动了。
- 它尝试在当前进程里本地 `map_to_kv_tensors()`。
- 但真正创建 KV tensors 的进程其实是 `Worker`。

换句话说，问题不是 “KV tensors 根本没创建”，而是 “当前进程没有资格本地去碰它们”。

### 根因一句话版本

旧逻辑默认认为：

- `world_size == 1`
- 就等价于“当前进程可以直接本地访问 KV tensors”

但在 `mp + tp=1` 下，这个推理是错的。  
`world_size == 1` 只说明张量并行大小是 1，不说明当前进程就是持有 FTensors / KV tensors 的那个进程。

## 修复思路：统一“当前是否必须通过 worker IPC 访问 KV tensors”

真正的修复并不是简单多写几个 `if`，而是把判断标准统一成一句话：

- 当前进程如果不是 worker，就不要默认本地访问 KV tensors。
- 该检查的地方通过 worker IPC 检查。
- 该执行的地方通过 worker IPC 执行。

### 1. 新增统一判断：`should_use_worker_ipc()`

文件：

- `kvcached/integration/vllm/interfaces.py`

新增：

```python
def should_use_worker_ipc() -> bool:
    return _kvcached_initialized and not _is_worker
```

这里最重要的是 `_is_worker` 的语义：  
它表示“当前进程是不是 vLLM 的 worker 进程”。

所以这条判断的真实含义是：

- `kvcached` 已初始化
- 但当前进程不是 worker
- 那当前进程就不应该本地访问 KV tensors，而应该走 worker IPC

### 2. 为什么要允许重复 init 时“升级状态”

在 [interfaces.py](d:/proj/kvcached/kvcached_async/kvcached/integration/vllm/interfaces.py) 里还有一段很关键的逻辑：

```python
if is_worker and not _is_worker:
    _is_worker = True
if async_sched and not _async_sched:
    _async_sched = True
    logger.info("kvcached async scheduler enabled")
```

这段代码不是重新初始化，而是在“已经初始化过”的情况下，吸收后续调用带来的新事实。

这里有两组变量：

- `is_worker` / `async_sched`
  这次 `init_kvcached()` 调用传进来的信息
- `_is_worker` / `_async_sched`
  当前进程里已经记录下来的全局状态

这两段逻辑做的事情都是“只升级，不回退”：

- 如果这次调用明确告诉我们当前进程其实是 worker，那就把 `_is_worker` 从 `False` 升级成 `True`
- 如果这次调用明确告诉我们 async 已启用，那就把 `_async_sched` 从 `False` 升级成 `True`

如果不这样做，就可能出现第一次 init 时拿到的是不完整信息，后续真正进入 worker 路径时，全局状态却还停留在旧值。

### 3. 为什么 `KV tensors created` 检查也要改

你前面提到的 `FTensorAllocator.num_layers == 0` 问题，本质上正是这个原因。

旧逻辑的问题是：

- `KVCacheManager._post_init()` 会先等 “KV tensors 已创建” 再继续初始化
- 如果它在 `EngineCore` 当前进程本地调用 `kv_tensors_created()`
- 那查到的是当前这个进程里的 `FTensorAllocator` 状态
- 但在 `mp` 模式下，真正初始化 `FTensorAllocator` 的是 `Worker`
- 于是 `EngineCore` 本地看到的仍然可能是 `num_layers == 0`
- `kvcached` 就会误以为 KV tensors 没创建，最终导致服务启动失败

所以这里改成了：

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

这段修改的意义是：

- 如果当前是普通本地场景，就继续在本地查
- 但如果当前是 vLLM 的 `EngineCore` 非 worker 进程，就不要再查本地
- 改成通过 `broadcast_kv_tensors_created(...)` 去 worker 查询

### 4. `broadcast_kv_tensors_created(...)` 为什么能查到 worker 的状态

关键点在于，它不是自己“知道”状态，而是发 IPC 给 worker，让 worker 自己回答。

在 `tp_ipc_util.py` 里，`broadcast_kv_tensors_created(...)` 会发一条消息：

```python
{"cmd": "kv_tensors_created", "group_id": group_id}
```

worker 侧收到之后，真正执行的是：

```python
created = kv_tensors_created(group_id=group_id)
```

注意这次调用发生在 worker 进程里，而不是 `EngineCore` 里。  
所以它检查到的是 worker 自己那个进程中的 `FTensorAllocator` / KV tensor 状态。

这也是为什么它能解决“当前进程里 `num_layers == 0`，误判 KV tensor 未创建”的问题。

即使 `tp=1` 也成立，因为：

- `tp_size = 1`
- IPC 只会发给 `rank=0` 的那个唯一 worker

所以它本质上就是“去唯一那个 worker 进程里查一次”。

### 5. 为什么 `map/unmap` 这段才是运行期最核心的修复

文件：

- `kvcached/page_allocator.py`

关键改动是：

```python
if self.world_size > 1 or _should_use_worker_ipc():
    broadcast_map_to_kv_tensors(...)
else:
    map_to_kv_tensors(...)
```

以及：

```python
if self.world_size > 1 or _should_use_worker_ipc():
    broadcast_unmap_from_kv_tensors(...)
else:
    unmap_from_kv_tensors(...)
```

这几乎就是这次“启动期和运行期都真正修好”的核心行为修改。

原因很直接：

- 旧逻辑只看 `world_size > 1`
- 在 `mp + tp=1` 下会误走本地 `map_to_kv_tensors()`
- 但此时真正持有 KV tensors 的是 `Worker`，不是 `EngineCore`

所以 `broadcast_map_to_kv_tensors(...)` 解决的不是“offset 冲突”，而是“操作发到了错误进程”这个冲突。

### 6. `broadcast_map_to_kv_tensors(...)` 为什么在单卡也能工作

它并不是只靠 `world_size / pp_rank / group_id` 工作，它真正还会携带：

- `offsets`

也就是这次需要 map 的 page 对应的内存偏移。

它内部的逻辑是：

- 构造一条消息：

```python
{"cmd": "map_to_kv_tensors", "offsets": offsets, "group_id": group_id}
```

- 然后对每个 rank 发 IPC：

```python
for rank in range(tp_size)
```

所以在 `tp=1` 时：

- `range(1)` 只会发给 `rank=0`
- 也就是只发给那个唯一的 worker

worker 收到后，才在自己的进程里真正执行：

```python
map_to_kv_tensors(msg["offsets"], group_id=group_id)
```

换句话说：

- `EngineCore` 负责决定“哪些 offset 需要 map”
- 但不再自己去 map
- 它只把命令发给真正持有 KV tensors 的 worker

所以这里的 `broadcast` 可以理解成“统一走 worker-side IPC”，单卡时只是广播对象退化成唯一一个 worker。

## 修复后的验证结果

### 单测

```bash
PYTHONPATH=d:\proj\kvcached\kvcached_async pytest tests\test_vllm_async_sched.py -q
```

结果：

```text
3 passed
```

### `mp + tp=1` 启动与 benchmark

从 `results/mp_new_async.log` 可以确认：

- `world_size=1`
- `Worker 0 IPC listener started`
- `kvcached async scheduler enabled`
- `Init kvcached KV cache allocator: ... world_size=1, async_sched=True`
- `Application startup complete.`

同时，修复前的关键错误已不再出现：

```text
try to map to KV tensors when KV tensors are not created
```

对应 benchmark 文件：

- `results/mp_tp1_bench_no_async.txt`
- `results/mp_tp1_bench_new_async.txt`

| Case | Success / Fail | req/s | total tok/s | mean TTFT (ms) | p99 TTFT (ms) | mean TPOT (ms) | p99 TPOT (ms) | mean ITL (ms) | p99 ITL (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mp tp=1 no async | 200 / 0 | 7.87 | 5037.54 | 26.63 | 50.58 | 3.16 | 4.08 | 3.16 | 15.56 |
| mp tp=1 new async | 200 / 0 | 7.93 | 5072.18 | 25.78 | 46.45 | 1.90 | 2.63 | 1.90 | 13.80 |

结论很清楚：

- 单卡 `mp` 下，吞吐变化不大
- async 的主要收益仍然体现在 token 级延迟
- `TPOT/ITL` 从 `3.16ms` 降到 `1.90ms`，降幅接近 40%

## 双卡补充测试：`mp + tp=2`

### 当前已经确认的部分

结果文件：

- `results/mp_tp2_no_async.log`
- `results/mp_tp2_new_async.log`

从 `results/mp_tp2_new_async.log` 可以确认：

- vLLM 打印 `Asynchronous scheduling is enabled.`
- `world_size=2`
- 两个 worker listener 都已启动
- `page_allocator` 初始化日志显示 `world_size=2, async_sched=True`
- 服务最终打印 `Application startup complete.`

这说明双卡 async 路径的启动和服务注册已经跑通。

### 当前双卡 benchmark 如何解读

完整成功的基线样本是：

- `results/mp_tp2_bench_no_async.txt`

| Case | Success / Fail | req/s | total tok/s | mean TTFT (ms) | p99 TTFT (ms) | mean TPOT (ms) | p99 TPOT (ms) | mean ITL (ms) | p99 ITL (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mp tp=2 no async | 200 / 0 | 7.89 | 5049.25 | 31.90 | 56.59 | 3.21 | 5.55 | 3.21 | 20.52 |

而这次双卡 async 样本：

- `results/mp_tp2_bench_new_async.txt`

不能直接拿来做正式性能对比：

| Case | Success / Fail | req/s | total tok/s | mean TTFT (ms) | p99 TTFT (ms) | mean TPOT (ms) | p99 TPOT (ms) | mean ITL (ms) | p99 ITL (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mp tp=2 new async | 45 / 155 | 1.78 | 1141.20 | 34.54 | 60.72 | 2.00 | 2.62 | 2.00 | 20.13 |

原因不是“它数值更差”，而是这份样本里有大量：

```text
ConnectionRefusedError: [Errno 111] Connect call failed ('127.0.0.1', 18000)
Cannot connect to host 127.0.0.1:18000
```

而且时间戳显示：

- benchmark 在 2026-04-02 14:05:19 就开始了
- 服务到 2026-04-02 14:05:39 才 `Application startup complete.`

所以更合理的解释是：

- 客户端在服务完全 ready 之前就开始压测
- 当前这份双卡 async 样本应视为无效性能样本
- 它可以证明“服务最终起来了并处理过请求”，但不能用来下正式性能结论

## 当前推荐的验证方式

### 日志检查

```bash
grep -n "kvcached async scheduler enabled" mp_async.log
grep -n "Init kvcached KV cache allocator" mp_async.log
grep -n "async_sched=True" mp_async.log
grep -n "Asynchronous scheduling is enabled" mp_async.log
grep -Ei "try to map to KV tensors|Failed to patch kv_cache_coordinator|ERROR|Traceback" mp_async.log
```

对于 `tp=2`，建议再确认：

```bash
grep -n "world_size=2" mp_tp2_new_async.log
grep -n "Worker 0 IPC listener started" mp_tp2_new_async.log
grep -n "Worker 1 IPC listener started" mp_tp2_new_async.log
```

### benchmark 命令

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

如果需要双卡 async 的正式性能样本，最重要的不是换参数，而是：

- 先等到日志明确出现 `Application startup complete.`
- 再启动 benchmark，或者开启 endpoint ready check

## 这次修复真正说明了什么

这次问题表面上是 “async scheduling 打开后 `mp` 启不来”，但本质上的教训其实更简单：

- `world_size` 只能说明并行规模
- 它不能说明当前进程是不是持有 KV tensors 的那个进程

对 `kvcached` 来说，真正重要的判断不是“现在是不是多卡”，而是：

- 当前进程是否拥有本地 KV tensors
- 还是必须通过 worker IPC 去检查和操作它们

一旦把这层语义抽出来，ready-check、`map/unmap`、日志和回归测试就都能对齐。

截至当前，可以下的结论是：

- `uni` 路径可用，且不会出现这次 `mp` 的启动错误
- `mp + tp=1 + --async-scheduling` 的启动问题已修复
- `mp + tp=1` 的 benchmark 证明 async 的收益主要仍然体现在 token 级延迟
- `mp + tp=2` 已完成启动与功能验证
- 双卡 async 的正式性能对比还需要补一轮严格 ready 后的 benchmark

## 相关文件

- `kvcached/integration/vllm/patches.py`
- `kvcached/integration/vllm/interfaces.py`
- `kvcached/kv_cache_manager.py`
- `kvcached/page_allocator.py`
- `tests/test_vllm_async_sched.py`
- `docs/vllm_async_mp_investigation.md`

## 相关附件

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
