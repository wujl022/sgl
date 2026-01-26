"""
示例：模型切换时复用 KV Cache

场景：
- 模型A部署好了，执行 req0 到一半
- 卸载模型A，部署模型B（量化版本，权重更少）
- 保留复用：kv_buffer, page_table, radix_tree_node, _free_slots
- 因为模型B权重更少，num_pages 变多，扩展 kv_buffer 和 _free_slots
"""

import os
import time

import torch
from minisgl.scheduler import Scheduler, SchedulerConfig
from minisgl.distributed import DistributedInfo


def switch_model_a_to_b(
    scheduler_a: Scheduler,
    config_b: SchedulerConfig,
) -> Scheduler:
    """
    从模型A切换到模型B，复用 KV cache 资源
    
    Args:
        scheduler_a: 模型A的调度器
        config_b: 模型B的配置
    
    Returns:
        模型B的调度器
    """
    # ========== 步骤 1: 同步 GPU 操作 ==========
    torch.cuda.synchronize(scheduler_a.engine.device)
    from minisgl.utils import init_logger
    logger = init_logger(__name__)
    logger.info("GPU operations synchronized")
    
    # ========== 步骤 2: 保存正在执行的请求状态 ==========
    running_requests = scheduler_a.decode_manager.running_reqs.copy()
    pending_requests = scheduler_a.prefill_manager.pending_list.copy()
    finished_requests = scheduler_a.finished_reqs.copy()
    
    logger.info(f"Saved {len(running_requests)} running requests")
    logger.info(f"Saved {len(pending_requests)} pending requests")
    logger.info(f"Saved {len(finished_requests)} finished requests")
    
    # ========== 步骤 3: 保存资源 ==========
    existing_kv_cache = scheduler_a.engine.kv_cache
    existing_page_table = scheduler_a.engine.page_table
    existing_token_pool = scheduler_a.table_manager.token_pool
    existing_cache_manager = scheduler_a.cache_manager
    old_num_pages = scheduler_a.engine.num_pages
    
    logger.info(f"Saved resources: old_num_pages={old_num_pages}")
    
    # ========== 步骤 4: 卸载模型A ==========
    # 先 destroy CUDA graphs
    scheduler_a.engine.graph_runner.destroy_cuda_graphs()
    
    # 删除模型权重（但保留其他资源）
    del scheduler_a.engine.model
    
    # 清理 GPU 缓存
    torch.cuda.empty_cache()
    logger.info("Model A unloaded")
    
    # ========== 步骤 5: 加载模型B ==========
    # 模型B会在 Scheduler 初始化时自动加载
    
    # ========== 步骤 6-11: 创建模型B的调度器（自动处理扩展和恢复） ==========
    scheduler_b = Scheduler(
        config_b,
        existing_kv_cache=existing_kv_cache,
        existing_page_table=existing_page_table,
        existing_token_pool=existing_token_pool,
        existing_cache_manager=existing_cache_manager,
        old_num_pages=old_num_pages,
        running_requests=running_requests,
        pending_requests=pending_requests,
        finished_requests=finished_requests,
    )
    
    # ========== 步骤 12: 验证完整性 ==========
    new_num_pages = scheduler_b.engine.num_pages
    free_slots_len = len(scheduler_b.cache_manager._free_slots)
    radix_tree_size = scheduler_b.cache_manager.manager.size_info.total_size
    
    logger.info(f"New num_pages: {new_num_pages}")
    logger.info(f"Free slots: {free_slots_len}")
    logger.info(f"Radix Tree size: {radix_tree_size}")
    logger.info(f"Total: {free_slots_len + radix_tree_size} (should <= {new_num_pages})")
    
    assert (
        free_slots_len + radix_tree_size <= new_num_pages
    ), f"Integrity check failed: {free_slots_len} + {radix_tree_size} > {new_num_pages}"
    
    # 验证 dummy_page
    assert (
        scheduler_b.engine.dummy_page == new_num_pages
    ), f"dummy_page mismatch: {scheduler_b.engine.dummy_page} != {new_num_pages}"
    
    logger.info("Model switching completed successfully!")
    
    return scheduler_b


# ========== 使用示例 ==========
if __name__ == "__main__":
    from minisgl.utils import init_logger
    from minisgl.core import SamplingParams
    from minisgl.message import UserMsg
    from collections import deque
    
    logger = init_logger(__name__)

    if os.environ.get("MINISGL_MODE", "bench") == "bench":
        from collections import deque
        from random import randint, seed

        from minisgl.core import SamplingParams
        from minisgl.message import UserMsg

        seed(int(os.environ.get("MINISGL_SEED", "0")))
        attn = os.environ.get("MINISGL_ATTENTION_BACKEND", "fi")
        cache_type = os.environ.get("MINISGL_CACHE_TYPE", "naive")
        bs = int(os.environ.get("MINISGL_BS", "64"))
        input_len = int(os.environ.get("MINISGL_INPUT_LEN", "128"))
        output_len = int(os.environ.get("MINISGL_OUTPUT_LEN", "1024"))
        split_after = int(os.environ.get("MINISGL_SPLIT_AFTER", "512"))
        verify_uid = int(os.environ.get("MINISGL_VERIFY_UID", "0"))

        if split_after <= 0 or split_after >= output_len:
            raise ValueError(f"Invalid split_after={split_after}, expected 0 < split_after < output_len")

        config_a = SchedulerConfig(
            model_path="/home/wujl022/models/LLM-Research/Meta-Llama-3-8B-Instruct",
            tp_info=DistributedInfo(0, 1),
            dtype=torch.bfloat16,
            attention_backend=attn,
            max_running_req=max(256, bs),
            cache_type=cache_type,
            memory_ratio=0.9,
            cuda_graph_bs=[0],
            offline_mode=True,
        )
        scheduler_a = Scheduler(config_a)

        config_b = SchedulerConfig(
            model_path="/home/wujl022/models/Huggingface/Meta-Llama-3-8B-Instruct-GPTQ-4bit-gs128",
            tp_info=DistributedInfo(0, 1),
            dtype=torch.bfloat16,
            attention_backend=attn,
            max_running_req=max(256, bs),
            cache_type=cache_type,
            memory_ratio=0.9,
            cuda_graph_bs=[0],
            offline_mode=True,
        )
        scheduler_b = switch_model_a_to_b(scheduler_a, config_b)

        kv_cache = scheduler_b.engine.kv_cache
        if not hasattr(kv_cache, "segments"):
            raise RuntimeError("KV cache is not segmented after switching")
        seg2_start = int(kv_cache.segment_starts[1])

        prompt_token_ids = [[randint(0, 10000) for _ in range(input_len)] for _ in range(bs)]
        uids = list(range(0, bs))

        def reorder(mode: str) -> None:
            free = scheduler_b.cache_manager._free_slots
            mask = free >= seg2_start if mode == "seg2" else free < seg2_start
            scheduler_b.cache_manager._free_slots = torch.cat([free[mask], free[~mask]])

        def reset_state() -> None:
            torch.cuda.synchronize(scheduler_b.engine.device)
            scheduler_b.prefill_manager.pending_list.clear()
            scheduler_b.decode_manager.running_reqs.clear()
            scheduler_b.finished_reqs.clear()
            scheduler_b.table_manager._free_slots = list(range(scheduler_b.table_manager._max_running_reqs))
            scheduler_b.cache_manager.manager.reset()
            scheduler_b.cache_manager._free_slots = torch.arange(
                scheduler_b.engine.num_pages, dtype=torch.int32, device=scheduler_b.engine.device
            )
            scheduler_b.page_table.fill_(scheduler_b.engine.dummy_page)
            scheduler_b.token_pool.zero_()

        def _normal_loop() -> None:
            try:
                scheduler_b.normal_loop()
            except Exception as e:
                if "RequestAllFinished" in str(type(e).__name__):
                    return
                raise

        def run_case(
            name: str,
            *,
            flip_to_seg2: bool,
            output_len_override: int | None = None,
            split_after_override: int | None = None,
            measure: bool = True,
        ) -> None:
            prompt_len = input_len
            local_output_len = output_len if output_len_override is None else int(output_len_override)
            local_split_after = split_after if split_after_override is None else int(split_after_override)

            pending_msgs = deque(
                [
                    UserMsg(
                        uid=uid,
                        input_ids=torch.tensor(prompt_token_ids[i], dtype=torch.int32),
                        sampling_params=SamplingParams(
                            ignore_eos=True,
                            max_tokens=local_output_len,
                            temperature=0.7,
                            top_p=0.95,
                            stop_token_ids=[128001, 128009],
                        ),
                    )
                    for i, uid in enumerate(uids)
                ]
            )

            counts = {uid: 0 for uid in uids}
            table_idx0 = None

            def min_cached_out() -> int:
                vals = [
                    int(r.cached_len) - prompt_len
                    for r in scheduler_b.decode_manager.running_reqs
                    if r.uid in counts
                ]
                return min(vals) if vals else -1

            def recv(blocking: bool = False):
                _ = blocking
                if not pending_msgs:
                    return []
                msgs = list(pending_msgs)
                pending_msgs.clear()
                return msgs

            def send(reply):
                for m in reply:
                    if m.uid in counts:
                        counts[m.uid] += 1

            scheduler_b.receive_msg = recv
            scheduler_b.send_result = send

            reorder("seg1")
            it = 0
            while min_cached_out() < local_split_after and it < 200000:
                it += 1
                _normal_loop()
                if table_idx0 is None:
                    r0 = next(
                        (r for r in scheduler_b.decode_manager.running_reqs if r.uid == verify_uid),
                        None,
                    )
                    if r0 is not None:
                        table_idx0 = int(r0.table_idx)

            if flip_to_seg2:
                reorder("seg2")

            t0 = None
            if measure:
                torch.cuda.synchronize(scheduler_b.engine.device)
                t0 = time.perf_counter()

            while min(counts.values()) < local_output_len and it < 400000:
                it += 1
                _normal_loop()

            if measure:
                torch.cuda.synchronize(scheduler_b.engine.device)
                t1 = time.perf_counter()

            if min(counts.values()) < local_output_len:
                raise RuntimeError(
                    f"{name}: generation did not finish: min_count={min(counts.values())}"
                )

            if table_idx0 is not None:
                kv_pages = scheduler_b.page_table[
                    table_idx0, prompt_len : prompt_len + local_output_len - 1
                ].to("cpu")
                first = kv_pages[:local_split_after]
                last = kv_pages[local_split_after:]
                if flip_to_seg2:
                    assert torch.all(first < seg2_start)
                    assert torch.all(last >= seg2_start)
                else:
                    assert torch.all(kv_pages < seg2_start)

            if measure:
                assert t0 is not None
                elapsed_s = t1 - t0
                timed_tokens = bs * (local_output_len - local_split_after - 1)
                tok_s = timed_tokens / elapsed_s
                logger.info(
                    f"{name}: bs={bs} input_len={input_len} output_len={local_output_len} split_after={local_split_after} "
                    f"time_last={elapsed_s:.3f}s throughput_last={tok_s:.2f} tok/s"
                )

        reset_state()
        warmup_output_len = min(output_len, int(os.environ.get("MINISGL_WARMUP_OUTPUT_LEN", "128")))
        warmup_split_after = min(split_after, warmup_output_len // 2)
        run_case(
            "warmup_seg1_then_seg2",
            flip_to_seg2=True,
            output_len_override=warmup_output_len,
            split_after_override=warmup_split_after,
            measure=False,
        )

        reset_state()
        run_case("all_in_seg1", flip_to_seg2=False)

        reset_state()
        run_case("seg1_then_seg2", flip_to_seg2=True)
        raise SystemExit(0)

    # 配置模型A
    config_a = SchedulerConfig(
        model_path="/home/wujl022/models/LLM-Research/Meta-Llama-3-8B-Instruct",
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        attention_backend="fi",
        max_running_req=256,
        cache_type="radix",
        memory_ratio=0.9,
        cuda_graph_bs=[0],
        offline_mode=True,  # 使用离线模式
    )
    
    # 部署模型A
    scheduler_a = Scheduler(config_a)
    logger.info("Model A deployed")
    
    # ========== 发送请求 "what is ai?" ==========
    prompt = "what is ai?"
    tokenizer = scheduler_a.tokenizer
    input_ids = tokenizer.encode(prompt, return_tensors="pt").view(-1).to(torch.int32)
    print(f"Input IDs: {input_ids}")
    
    # 创建请求消息
    user_msg = UserMsg(
        uid=0,
        input_ids=input_ids,
        sampling_params=SamplingParams(temperature=0.7, top_p=0.95, ignore_eos=True, max_tokens=21, stop_token_ids=[128001, 128009]),
    )
    
    # 存储输出的 tokens
    output_tokens = []
    output_state = {"text": "", "count": 0}
    switch_after_tokens = 11 # 输出11个tokens后切换（保证前10个输出token的KV都已经写入段1）
    
    # 手动实现离线模式的请求处理
    pending_msgs = deque([user_msg])
    
    def offline_receive_msg(blocking=False):
        if pending_msgs:
            return [pending_msgs.popleft()]
        return []
    
    def offline_send_result(reply):
        for msg in reply:
            if msg.uid == 0:  # 我们的请求
                token = msg.next_token
                output_tokens.append(token)
                output_state["count"] += 1
                # 解码 token 到文本
                token_text = tokenizer.decode([token], skip_special_tokens=True)
                output_state["text"] += token_text
                logger.info(
                    f"Model A - Token {output_state['count']}: {token_text!r} "
                    f"(total: {output_state['text']!r})"
                )
    
    # 替换 scheduler 的 I/O 方法
    scheduler_a.receive_msg = offline_receive_msg
    scheduler_a.send_result = offline_send_result
    
    logger.info(f"Processing request: {prompt!r}")
    logger.info("Starting generation with Model A...")
    
    # 运行 scheduler 直到输出指定数量的 tokens
    max_iterations = 200
    iteration = 0
    
    while output_state["count"] < switch_after_tokens and iteration < max_iterations:
        iteration += 1
        try:
            # 运行一个调度循环
            scheduler_a.normal_loop()
        except Exception as e:
            if "RequestAllFinished" in str(type(e).__name__):
                break
            raise
    
    logger.info(f"Model A generated {output_state['count']} tokens: {output_state['text']!r}")
    logger.info("Switching to Model B...")
    
    # ========== 切换到模型B ==========
    config_b = SchedulerConfig(
        model_path="/home/wujl022/models/Huggingface/Meta-Llama-3-8B-Instruct-GPTQ-4bit-gs128",
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        attention_backend="fi",
        max_running_req=256,
        cache_type="radix",
        memory_ratio=0.9,
        cuda_graph_bs=[0],
        offline_mode=True,
    )
    
    scheduler_b = switch_model_a_to_b(scheduler_a, config_b)

    kv_cache = scheduler_b.engine.kv_cache
    if not hasattr(kv_cache, "segments"):
        raise RuntimeError(
            "KV cache is not segmented after switching. "
            "Make sure new num_pages > old num_pages so extension happens."
        )

    seg2_start = int(kv_cache.segment_starts[1])

    alloc_mode = os.environ.get("MINISGL_KV_ALLOC_MODE", "seg2")
    if alloc_mode not in {"seg1", "seg2"}:
        raise ValueError(f"Invalid MINISGL_KV_ALLOC_MODE={alloc_mode!r}, expected 'seg1' or 'seg2'")

    free = scheduler_b.cache_manager._free_slots
    if alloc_mode == "seg2":
        mask = free >= seg2_start
    else:
        mask = free < seg2_start
    scheduler_b.cache_manager._free_slots = torch.cat([free[mask], free[~mask]])
    logger.info(
        f"Reordered _free_slots: alloc_mode={alloc_mode}, seg2_start={seg2_start}, "
        f"free_slots={len(scheduler_b.cache_manager._free_slots)}"
    )

    req0 = next((r for r in scheduler_b.decode_manager.running_reqs if r.uid == 0), None)
    if req0 is None:
        raise RuntimeError("Cannot find req uid=0 right after switching")
    req0_table_idx = int(req0.table_idx)
    prompt_len = int(len(input_ids))

    # ========== 继续处理请求（使用模型B） ==========
    logger.info("Continuing generation with Model B...")
    
    # 更新输出处理函数（使用模型B的tokenizer）
    def offline_send_result_b(reply):
        for msg in reply:
            if msg.uid == 0:  # 我们的请求
                token = msg.next_token
                output_tokens.append(token)
                output_state["count"] += 1
                # 解码 token 到文本
                token_text = scheduler_b.tokenizer.decode([token], skip_special_tokens=True)
                output_state["text"] += token_text
                logger.info(
                    f"Model B - Token {output_state['count']}: {token_text!r} "
                    f"(total: {output_state['text']!r})"
                )
                if msg.finished:
                    logger.info("Request finished!")
    
    # 设置 I/O 方法
    scheduler_b.receive_msg = offline_receive_msg
    scheduler_b.send_result = offline_send_result_b
    
    torch.cuda.synchronize(scheduler_b.engine.device)
    t0 = time.perf_counter()

    max_iterations = 400
    iteration = 0

    while output_state["count"] < 21 and iteration < max_iterations:
        iteration += 1
        try:
            scheduler_b.normal_loop()
        except Exception as e:
            if "RequestAllFinished" in str(type(e).__name__):
                break
            raise

    torch.cuda.synchronize(scheduler_b.engine.device)
    t1 = time.perf_counter()
    elapsed_s = t1 - t0

    final_output = scheduler_b.tokenizer.decode(output_tokens, skip_special_tokens=True)
    logger.info(f"Final output ({len(output_tokens)} tokens): {final_output!r}")
    logger.info(f"Model B stage time (10 decode steps): {elapsed_s * 1000:.3f} ms")
    logger.info("Request completed!")

    kv_pages = scheduler_b.page_table[req0_table_idx, prompt_len : prompt_len + 20].to("cpu")
    first10 = kv_pages[:10]
    last10 = kv_pages[10:]

    logger.info(
        f"KV page indices for output tokens 1..20: first10(min={int(first10.min())}, max={int(first10.max())}), "
        f"last10(min={int(last10.min())}, max={int(last10.max())}), seg2_start={seg2_start}"
    )

    if alloc_mode == "seg2":
        assert torch.all(first10 < seg2_start), (
            f"Expected first 10 KV pages in segment1 (<{seg2_start}), got {first10.tolist()}"
        )
        assert torch.all(last10 >= seg2_start), (
            f"Expected last 10 KV pages in segment2 (>= {seg2_start}), got {last10.tolist()}"
        )
        logger.info("Segment KV buffer check passed: first10 in segment1, last10 in segment2")
    else:
        assert torch.all(kv_pages < seg2_start), (
            f"Expected KV pages all in segment1 (<{seg2_start}), got {kv_pages.tolist()}"
        )
        logger.info("Segment KV buffer check passed: all in segment1")

    logger.info("Running reference generation without switching...")

