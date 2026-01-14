# Mini-SGLang Online Serving 请求调度流程
1. 请求进入 / 文本转 token
- 前端收到用户请求，把文本交给 tokenizer。
- tokenizer 把文本转成 input_ids，连同采样参数一起打包成内部请求消息，丢给调度器。
2. 调度器接收请求，放入 prefill 待处理队列
- 调度器读到新请求，把它记录成一个待 prefill 的请求（包含 input_ids、最大输出长度等）。
- 此时请求还没有被分配 KV cache 页面，只是排队。
3. prefill 批次选择 & 预算检查
- 调度器根据一个 prefill token 预算（防止一次性放太多 token）从 pending 列表里挑一批请求，组成 prefill 批次。
- 同时会看当前 decode 阶段“在路上”的 token 数，预留一部分 KV cache 空间给 decode，避免把显存吃满。
4. 尝试复用已有 KV cache（前缀匹配）
- 对每个要进 prefill 的请求，用它的前缀在 Radix Cache 里做最长前缀匹配：
    - 如果找到匹配前缀：认为这部分 KV 已经算过了，得到“已缓存长度”和对应的 KV page 索引。
    - 如果没有匹配：从头开始，没有已缓存长度。
- 匹配到的那部分 KV cache 会被“锁住”，在这次请求完成前不会被驱逐。
5. 为请求分配 page table 行 & 写入已缓存前缀
- 给每个请求分配一个 page table 行号，表示它在“序列表”中的位置。
- 如果该请求有已缓存前缀：把这段前缀的 token id 和对应的 KV page 索引，拷贝到它的 page table 行里，等价于“把已经算好的 KV 关联到这个请求”。
6. 为“新计算的 token”分配 KV cache 页面
- 计算每个请求还需要新算多少 token（extend_len），把这个 batch 里所有请求的 extend_len 加起来，得到需要的新页面总数。
- 先用空闲页面直接分配；不够的话，按策略从可驱逐的旧 KV 前缀里回收一部分页面，再分给本批次。
- 分配结果是一段 page 索引数组，写进当前 batch 的 out_loc，表示“这些新 token 的 KV 存在这些页面里”。
7. 准备 attention 元数据
- 根据每个请求的：
    - 已缓存长度
    - 当前序列长度（含新 token）
    - 在 page table 里对应的 page 索引
- 计算各种 seqlens / 前缀累积长度 / positions 等，打包成 attention metadata，给后面的 attention kernel 使用。
8. prefill 前向计算 & 写 KV cache
- 调度器把该 batch 的 token ids 搬上 GPU，交给模型执行前向。
- 每一层 attention 内部都会：
    - 先用 out_loc 里的 page 索引，把当前批次新 token 的 K/V 写入 KV cache 对应页面。
    - 再用 page table + KV cache（旧的+刚写入的）做 attention，得到输出。
- prefill 阶段结束后，每个请求的“已缓存长度”更新为当前序列长度，这些 KV 变成后续 decode 的上下文。
9. 把完成 prefill 且还能继续生成的请求纳入 decode 集合
- 对已经 prefill 完成的请求，如果还允许继续生成（remain_len > 0，且没遇到 EOS），就标记为“正在 decode 的请求”，进入 decode 集合。
- 这时它们已经有完整的 KV cache（上下文部分）可以复用。
10. decode 批次选择
- 调度器在没有合适 prefill 批次时，会从 decode 集合中选出一批请求，组成 decode 批次（通常每个请求只生成 1 个新 token）。
- decode 的批次调度和 prefill 共用同一套逻辑，只是每个请求的 extend_len 基本为 1。
11. 为 decode 新 token 分配 KV 页面 & 更新 page table
- 对每个 decode 请求，给它即将生成的那个新 token 分配一个 KV 页面。
- 在对应请求的 page table 行里，在“当前序列末尾位置”写入这个新分配的 KV page 索引。
- attention metadata 中的 seqlens 更新为“历史长度 + 新 token”。
12. decode 前向计算（读历史 KV + 写最后一个 token 的 KV）
- 对 decode 批次执行前向：
    - Query 只对应最后这个新 token。
    - Key/Value 通过 page table + KV cache 读出整条序列的 KV（历史 + 新 token）。
    - 同时把该新 token 的 K/V 写进刚分配的 KV 页面。
- 得到 logits 后，采样出下一个 token id。
13. 把新 token 写回请求 / 判断是否结束
- 把 decode 出来的 token 追加到每个请求的 input_ids（host 侧记录完整序列）。
- 更新请求的长度信息：
    - cached_len 挪到当前末尾
    - remain_len 减 1
- 如果：
    - 达到最大输出 token 数，或
    - 采样到 EOS（且没有忽略 EOS）
    → 认为该请求完成。
14. 把新 token 流给前端
- 对于每个 decode 步，每次新生成的 token 都被立即打包成增量回复，发回前端（流式输出）。
- 前端按顺序收到 token，拼成最终文本。
15. 请求结束时释放 & 归档 KV cache
- 对已经完成且不再参与 decode 的请求：
    - 释放它在 page table 中的行号，让位给新请求。
    - 把它的前缀（cached_len 部分）的 page 索引插入 Radix Cache，用于将来其他请求的前缀复用。
    - 同时把这部分 pages 的“多余部分”标记为空闲，方便后续分配。
- 最终达到“部分 KV 变成可复用前缀，部分 KV 页面彻底释放”的状态。
16. 循环执行 prefill / decode，直到没有请求
- 调度器在 prefill 和 decode 之间动态切换（优先 prefill，有预算限制）。
- 不断重复上述过程，直到没有待 prefill 或待 decode 的请求。

本文档详细说明 Mini-SGLang 中 online serving 的请求调度过程，从 prefill 到 decode，以及 KV cache 的占用和存取机制。

## 整体架构

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Frontend   │ --> │  Tokenizer   │ --> │  Scheduler  │
│   Server    │     │   Server     │     │             │
└─────────────┘     └──────────────┘     └─────────────┘
                                              │
                                              v
                                    ┌─────────────────┐
                                    │     Engine      │
                                    │  (Model + KV)   │
                                    └─────────────────┘
```

## 核心组件

### 1. Scheduler (`python/minisgl/scheduler/scheduler.py`)
- **职责**: 请求调度、批处理、协调 prefill 和 decode
- **关键方法**:
  - `overlap_loop()`: 主循环，重叠执行和调度
  - `_schedule_next_batch()`: 调度下一个 batch（prefill 或 decode）
  - `_prepare_batch()`: 准备 batch，分配 KV cache pages
  - `_forward()`: 执行 forward pass

### 2. PrefillManager (`python/minisgl/scheduler/prefill.py`)
- **职责**: 管理 prefill 阶段的请求
- **关键方法**:
  - `add_one_req()`: 添加新请求到待处理列表
  - `schedule_next_batch()`: 调度 prefill batch

### 3. DecodeManager (`python/minisgl/scheduler/decode.py`)
- **职责**: 管理 decode 阶段的请求
- **关键方法**:
  - `schedule_next_batch()`: 调度 decode batch
  - `filter_reqs()`: 过滤可继续 decode 的请求

### 4. CacheManager (`python/minisgl/scheduler/cache.py`)
- **职责**: 管理 KV cache pages 的分配和释放
- **关键方法**:
  - `allocate()`: 分配 KV cache pages
  - `match_req()`: 匹配请求前缀，复用已有 cache
  - `free_and_cache_finished_req()`: 释放并缓存已完成的请求

### 5. TableManager (`python/minisgl/scheduler/table.py`)
- **职责**: 管理 page table 索引分配
- **关键方法**:
  - `allocate()`: 分配 table index
  - `free()`: 释放 table index

## 详细流程

### 阶段 1: 请求接收与 Tokenization

**代码位置**: `python/minisgl/tokenizer/server.py`

```python
# 1. Frontend 发送请求到 Tokenizer Server
# 2. Tokenizer Server tokenize 请求
tokenize_manager.tokenize(tokenize_msg)
# 3. 发送 UserMsg 到 Scheduler
send_backend.put(UserMsg(uid, input_ids, sampling_params))
```

**流程**:
1. Frontend 接收用户请求
2. Tokenizer Server 将文本 tokenize 为 `input_ids`
3. 创建 `UserMsg` 并发送到 Scheduler

---

### 阶段 2: Prefill 阶段

#### 2.1 请求添加到 PrefillManager

**代码位置**: `python/minisgl/scheduler/scheduler.py:136`

```python
def _process_one_msg(self, msg: BaseBackendMsg) -> None:
    if isinstance(msg, UserMsg):
        self.prefill_manager.add_one_req(msg)  # 添加到待处理列表
```

#### 2.2 Prefill Batch 调度

**代码位置**: `python/minisgl/scheduler/prefill.py:124`

```python
def schedule_next_batch(self, prefill_budget: int) -> Batch | None:
    # 1. 创建 PrefillAdder，设置 token budget
    adder = PrefillAdder(
        token_budget=prefill_budget,
        reserved_size=self.decode_manager.inflight_tokens,  # 为 decode 预留空间
        cache_manager=self.cache_manager,
        table_manager=self.table_manager,
    )
    
    # 2. 尝试添加请求到 batch
    for pending_req in self.pending_list:
        if req := adder.try_add_one(pending_req):
            reqs.append(req)
        else:
            break  # 无法添加更多请求
    
    return Batch(reqs=reqs, phase="prefill")
```

#### 2.3 KV Cache 分配与前缀匹配

**代码位置**: `python/minisgl/scheduler/prefill.py:38`

```python
def _try_allocate_one(self, req: PendingReq) -> Tuple[BaseCacheHandle, int] | None:
    # 1. 检查 table 是否有可用空间
    if self.table_manager.available_size == 0:
        return None
    
    # 2. 匹配请求前缀，尝试复用已有 KV cache
    handle, match_indices = self.cache_manager.match_req(req)
    cached_len = handle.cached_len  # 已缓存的前缀长度
    
    # 3. 估算所需空间
    extend_len = req.input_len - cached_len
    estimated_len = extend_len + req.output_len
    
    # 4. 检查是否有足够空间
    if estimated_len + self.reserved_size > self.cache_manager.available_size:
        return None
    
    # 5. 锁定 handle，防止被 evict
    self.cache_manager.lock(handle)
    
    # 6. 分配 table index
    table_idx = self.table_manager.allocate()
    
    # 7. 如果有已缓存的前缀，复制到 page table
    if cached_len > 0:
        device_ids = self.table_manager.token_pool[table_idx][:cached_len]
        page_entry = self.table_manager.page_table[table_idx][:cached_len]
        device_ids.copy_(req.input_ids[:cached_len].pin_memory(), non_blocking=True)
        page_entry.copy_(match_indices)  # 复制已缓存的 page indices
    
    return handle, table_idx
```

**KV Cache 前缀匹配** (`python/minisgl/scheduler/cache.py:24`):
```python
def match_req(self, req: PendingReq):
    # 调用 CacheManager 匹配前缀
    # RadixCacheManager 使用 Radix Tree 查找最长匹配前缀
    return self.manager.match_prefix(req.input_ids[: input_len - 1])
```

#### 2.4 Batch 准备与 KV Cache Pages 分配

**代码位置**: `python/minisgl/scheduler/scheduler.py:141`

```python
def _prepare_batch(self, batch: Batch) -> ForwardInput:
    # 1. 计算需要的 KV cache pages 数量
    needed_size = sum(r.extend_len for r in batch.reqs)
    
    # 2. 分配 KV cache pages
    batch.out_loc = self.cache_manager.allocate(needed_size)
    # out_loc: 每个 token 对应的 KV cache page index
    
    # 3. 准备 2D indices 用于 token ids 加载和写入
    load_indices = self._make_2d_indices(
        [(r.table_idx, r.cached_len, r.device_len) for r in batch.padded_reqs]
    )
    write_indices = self._make_2d_indices([...])
    
    # 4. 将 out_loc 写入 page_table
    self.page_table.view(-1)[load_indices] = batch.out_loc
    
    # 5. 准备 attention metadata
    self.engine.attn_backend.prepare_metadata(batch)
    
    return ForwardInput(batch, sample_args, load_indices, write_indices)
```

**KV Cache Pages 分配** (`python/minisgl/scheduler/cache.py:39`):
```python
def allocate(self, needed_len: int) -> torch.Tensor:
    # 1. 如果空闲 pages 足够，直接分配
    if needed_len <= len(self._free_slots):
        allocated = self._free_slots[:needed_len]
        self._free_slots = self._free_slots[needed_len:]
        return allocated
    
    # 2. 否则需要 evict 一些 pages
    evicted = self.manager.evict(needed_len - free_len)
    merged = torch.cat([self._free_slots, evicted])
    allocated = merged[:needed_len]
    self._free_slots = merged[needed_len:]
    return allocated
```

#### 2.5 Forward Pass 与 KV Cache 存储

**代码位置**: `python/minisgl/scheduler/scheduler.py:218`

```python
def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
    # 1. 从 token_pool 加载 token ids
    self._load_token_ids(forward_input)
    
    # 2. 执行 forward pass
    forward_output = self.engine.forward_batch(batch, sample_args)
    
    # 3. 写入生成的 token 到 token_pool
    self._write_token_ids(forward_input, forward_output)
    
    return forward_output
```

**Engine Forward** (`python/minisgl/engine/engine.py:208`):
```python
def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
    with self.ctx.forward_batch(batch):
        # 执行模型 forward
        logits = self.model.forward()  # 或 graph_runner.replay(batch)
    
    # 采样
    next_tokens_gpu = self.sampler.sample(logits[: batch.size], args)
    return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)
```

**KV Cache 存储** (`python/minisgl/attention/fa.py:54`):
```python
def forward(self, q, k, v, layer_id: int, batch: Batch) -> torch.Tensor:
    # 1. 存储新计算的 K, V 到 KV cache
    self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
    # batch.out_loc 指定每个 token 存储到哪个 page
    
    # 2. 执行 attention，从 KV cache 读取
    return _fa_sgl_impl(
        q=q,
        k_cache=self.kvcache.k_cache(layer_id),
        v_cache=self.kvcache.v_cache(layer_id),
        page_table=metadata.page_table,
        ...
    )
```

**KV Cache 存储实现** (`python/minisgl/kvcache/mha_pool.py:56`):
```python
def store_kv(
    self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
) -> None:
    from minisgl.kernel import store_cache
    
    # 使用 CUDA kernel 将 K, V 存储到指定 pages
    store_cache(
        k_cache=self._k_buffer[layer_id].view(self._storage_shape),
        v_cache=self._v_buffer[layer_id].view(self._storage_shape),
        indices=out_loc,  # page indices
        k=k,
        v=v,
    )
```

#### 2.6 Prefill 完成，请求转入 Decode

**代码位置**: `python/minisgl/core.py:51`

```python
def complete_one(self) -> None:
    self.cached_len = self.device_len  # 更新已缓存长度
    self.device_len += 1  # 增加 device 长度（包含新生成的 token）
```

---

### 阶段 3: Decode 阶段

#### 3.1 Decode Batch 调度

**代码位置**: `python/minisgl/scheduler/decode.py:23`

```python
def schedule_next_batch(self) -> Batch | None:
    if not self.runnable:
        return None
    # 返回所有可继续 decode 的请求
    return Batch(reqs=list(self.running_reqs), phase="decode")
```

**请求过滤** (`python/minisgl/scheduler/decode.py:13`):
```python
def filter_reqs(self, reqs: Iterable[Req]) -> None:
    # 只保留可以继续 decode 的请求（remain_len > 0）
    self.running_reqs = {req for req in self.running_reqs.union(reqs) if req.can_decode()}
```

#### 3.2 Decode Batch 准备

**代码位置**: `python/minisgl/scheduler/scheduler.py:141`

Decode 阶段的 batch 准备与 prefill 类似，但有以下区别：
- `extend_len = 1` (每次只生成一个 token)
- `out_loc` 只分配 1 个 page per request
- `write_indices` 指向新 token 位置

#### 3.3 Decode Forward Pass

**代码位置**: `python/minisgl/attention/fa.py:49`

```python
def forward(self, q, k, v, layer_id: int, batch: Batch) -> torch.Tensor:
    # 1. 存储新 token 的 K, V
    self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
    
    # 2. 执行 attention
    # 对于 decode，query 只有 1 个 token，key/value 来自整个序列（包括 cache）
    return _fa_sgl_impl(
        q=q,  # shape: (batch_size, 1, num_heads, head_dim)
        k_cache=self.kvcache.k_cache(layer_id),  # 包含所有历史 tokens
        v_cache=self.kvcache.v_cache(layer_id),
        page_table=metadata.page_table,  # 映射请求到 pages
        cache_seqlens=metadata.cache_seqlens,  # 每个请求的序列长度
        ...
    )
```

**Attention Metadata 准备** (`python/minisgl/attention/fa.py:67`):
```python
def prepare_metadata(self, batch: Batch) -> None:
    reqs = batch.padded_reqs
    
    # 对于 decode:
    seqlens_q = [req.extend_len for req in reqs]  # 都是 1
    seqlens_k = [req.device_len for req in reqs]  # 当前序列长度（包括新 token）
    cached_lens = [req.cached_len for req in reqs]  # 已缓存长度
    
    # 构建 page_table: 每个请求的 pages
    new_page_table = torch.stack([
        page_table[req.table_idx, :max_seqlen_k] 
        for req in reqs
    ])
    
    batch.attn_metadata = FAMetadata(
        cu_seqlens_k=cu_seqlens_k,  # cumulative sequence lengths for keys
        cu_seqlens_q=cu_seqlens_q,  # cumulative sequence lengths for queries
        page_table=new_page_table,  # page indices for each request
        ...
    )
```

#### 3.4 采样与结果返回

**代码位置**: `python/minisgl/scheduler/scheduler.py:75`

```python
def _process_last_data(self, last_data: ForwardData | None, ...) -> None:
    batch, (_, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]
    copy_done.synchronize()
    
    for i, req in enumerate(batch.reqs):
        next_token_id = next_tokens_cpu[i]
        req.append_host(next_token_id.unsqueeze(0))  # 添加到 input_ids
        next_token = int(next_token_id.item())
        
        # 检查是否完成
        finished = not req.can_decode()
        if not req.sampling_params.ignore_eos:
            finished |= next_token == self.eos_token_id
        
        # 发送结果到 frontend
        reply.append(DetokenizeMsg(uid=req.uid, next_token=next_token, finished=finished))
        
        # 如果完成，释放资源
        if finished:
            self.finished_reqs.add(req)
            self.decode_manager.remove_req(req)
            self.table_manager.free(req.table_idx)
            self.cache_manager.free_and_cache_finished_req(
                req.cache_handle,
                req.input_ids[: req.cached_len],
                self.page_table[req.table_idx, : req.cached_len],
            )
```

---

### 阶段 4: KV Cache 管理

#### 4.1 Page Table 结构

**代码位置**: `python/minisgl/engine/engine.py:67`

```python
# Page Table: (max_running_req, max_seq_len)
# 每一行对应一个请求，存储该请求的 KV cache page indices
self.page_table = create_page_table(
    (config.max_running_req + 1, self.max_seq_len),
    device=self.device,
)

# Token Pool: 与 page_table 同形状，存储 token ids
self.token_pool = torch.zeros_like(page_table, dtype=torch.int32)
```

**Page Table 使用**:
- `page_table[req.table_idx, :req.device_len]`: 请求的 KV cache page indices
- `token_pool[req.table_idx, :req.device_len]`: 请求的 token ids

#### 4.2 KV Cache 存储结构

**代码位置**: `python/minisgl/kvcache/mha_pool.py:16`

```python
class MHAKVCache(BaseKVCache):
    def __init__(self, ...):
        # KV buffer: (2, num_layers, num_pages, local_kv_heads, head_dim)
        # 2 = K + V, num_pages = 总 pages 数
        kv_buffer = torch.empty(
            (2, num_layers, num_pages, local_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        self._k_buffer = self._kv_buffer[0]  # (num_layers, num_pages, ...)
        self._v_buffer = self._kv_buffer[1]
```

**存储布局**:
- **LayerFirst**: `(num_layers, num_pages, num_heads, head_dim)` - 按层组织
- **PageFirst**: `(num_pages, num_layers, num_heads, head_dim)` - 按 page 组织

#### 4.3 Radix Cache 前缀复用

**代码位置**: `python/minisgl/kvcache/radix_manager.py`

Radix Cache 使用 Radix Tree 存储共享前缀：

```python
def match_prefix(self, input_ids: torch.Tensor) -> Tuple[RadixCacheHandle, torch.Tensor]:
    # 1. 在 Radix Tree 中查找最长匹配前缀
    match_len = self.get_match_len(input_ids)
    
    # 2. 返回 handle 和匹配的 page indices
    if match_len > 0:
        node = self._find_node(input_ids[:match_len])
        return RadixCacheHandle(cached_len=match_len), node.page_indices[:match_len]
    else:
        return RadixCacheHandle(cached_len=0), torch.empty(0, dtype=torch.int32)
```

**前缀复用流程**:
1. 新请求到达，调用 `match_prefix()` 查找最长匹配前缀
2. 如果找到匹配，直接复用已有 KV cache pages
3. 只计算和存储新 tokens 的 KV cache

#### 4.4 KV Cache Eviction

**代码位置**: `python/minisgl/kvcache/radix_manager.py:166`

```python
def evict(self, size: int) -> torch.Tensor:
    # 1. 收集可 evict 的叶子节点（未被锁定的）
    evictable_nodes = self._collect_leave_nodes_for_evict()
    
    # 2. 按 LRU 或其他策略选择要 evict 的节点
    # 3. 释放对应的 pages
    # 4. 从 Radix Tree 中删除节点
    
    return evicted_indices
```

**Eviction 策略**:
- 只 evict 未被锁定的 cache（`lock_handle()` 保护的不会被 evict）
- 优先 evict 不常用的前缀

---

## 调度循环

### Overlap Loop (重叠调度)

**代码位置**: `python/minisgl/scheduler/scheduler.py:233`

```python
def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
    # 1. 接收新消息（非阻塞）
    for msg in self.receive_msg(blocking=blocking):
        self._process_one_msg(msg)
    
    # 2. 调度下一个 batch
    forward_input = self._schedule_next_batch()
    
    # 3. 执行当前 batch（与处理上一个 batch 的结果重叠）
    if forward_input is not None:
        with self.engine_stream_ctx:
            ongoing_data = (forward_input, self._forward(forward_input))
    
    # 4. 处理上一个 batch 的结果（与当前 batch 执行重叠）
    self._process_last_data(last_data, ongoing_data)
    
    return ongoing_data
```

**重叠执行**:
- GPU 执行当前 batch 的同时，CPU 处理上一个 batch 的结果
- 提高 GPU 利用率

---

## 关键数据结构

### Req (请求)

**代码位置**: `python/minisgl/core.py:28`

```python
@dataclass(eq=False)
class Req:
    input_ids: torch.Tensor  # CPU tensor，包含所有 tokens
    table_idx: int  # page table 中的索引
    cached_len: int  # 已缓存的长度（KV cache 中）
    output_len: int  # 最大输出长度
    uid: int  # 用户 ID
    sampling_params: SamplingParams
    cache_handle: BaseCacheHandle  # KV cache handle
    
    @property
    def device_len(self) -> int:  # 当前 device 上的长度
    @property
    def extend_len(self) -> int:  # 需要扩展的长度 (device_len - cached_len)
    @property
    def remain_len(self) -> int:  # 剩余可生成长度
```

### Batch (批次)

**代码位置**: `python/minisgl/core.py:70`

```python
@dataclass
class Batch:
    reqs: List[Req]
    phase: Literal["prefill", "decode"]
    input_ids: torch.Tensor  # GPU tensor，当前 batch 的 input ids
    out_loc: torch.Tensor  # KV cache page indices for new tokens
    padded_reqs: List[Req]  # 可能包含 dummy reqs for padding
    attn_metadata: BaseAttnMetadata  # attention metadata
```

---

## 总结

### Prefill 阶段流程
1. **请求接收** → Tokenizer → Scheduler
2. **前缀匹配** → CacheManager.match_req() → 复用已有 KV cache
3. **资源分配** → TableManager.allocate() + CacheManager.allocate()
4. **Batch 准备** → 准备 metadata，分配 pages
5. **Forward Pass** → Model.forward() → Attention → store_kv()
6. **完成** → Req.complete_one() → 转入 Decode

### Decode 阶段流程
1. **Batch 调度** → DecodeManager.schedule_next_batch()
2. **资源分配** → 分配 1 个 page per request
3. **Forward Pass** → 读取历史 KV cache + 存储新 token KV
4. **采样** → Sampler.sample() → 生成新 token
5. **结果返回** → 发送到 Frontend
6. **资源释放** → 请求完成后释放 pages 和 table index

### KV Cache 管理
- **分配**: CacheManager.allocate() → 从 free_slots 或 evict
- **存储**: kvcache.store_kv() → CUDA kernel 写入 pages
- **读取**: Attention backend 通过 page_table 读取
- **复用**: RadixCacheManager.match_prefix() → 前缀匹配
- **释放**: CacheManager.free_and_cache_finished_req() → 插入 Radix Tree

### 关键代码文件
- `python/minisgl/scheduler/scheduler.py`: 主调度器
- `python/minisgl/scheduler/prefill.py`: Prefill 管理
- `python/minisgl/scheduler/decode.py`: Decode 管理
- `python/minisgl/scheduler/cache.py`: KV cache 分配
- `python/minisgl/engine/engine.py`: Engine 执行
- `python/minisgl/attention/fa.py`: Flash Attention backend
- `python/minisgl/kvcache/mha_pool.py`: KV cache 存储
- `python/minisgl/kvcache/radix_manager.py`: Radix Cache 管理

