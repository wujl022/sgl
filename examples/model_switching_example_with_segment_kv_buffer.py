"""
示例：模型切换时复用 KV Cache

场景：
- 模型A部署好了，执行 req0 到一半
- 卸载模型A，部署模型B（量化版本，权重更少）
- 保留复用：kv_buffer, page_table, radix_tree_node, _free_slots
- 因为模型B权重更少，num_pages 变多，扩展 kv_buffer 和 _free_slots
"""

import gc
import json
import os
import time
from dataclasses import replace

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

    kv_cache_id = id(existing_kv_cache)
    page_table_ptr = int(existing_page_table.data_ptr())
    token_pool_ptr = int(existing_token_pool.data_ptr())
    cache_mgr_id = id(existing_cache_manager.manager)

    k0_ptr = None
    v0_ptr = None
    if hasattr(existing_kv_cache, "k_cache") and hasattr(existing_kv_cache, "v_cache"):
        try:
            k0_ptr = int(existing_kv_cache.k_cache(0).data_ptr())
            v0_ptr = int(existing_kv_cache.v_cache(0).data_ptr())
        except Exception:
            k0_ptr = None
            v0_ptr = None

    logger.info(
        "Reuse baseline: "
        f"kv_cache_id={kv_cache_id} page_table_ptr={page_table_ptr} token_pool_ptr={token_pool_ptr} "
        f"cache_mgr_id={cache_mgr_id} k0_ptr={k0_ptr} v0_ptr={v0_ptr}"
    )
    
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
    
    config_b = replace(config_b, num_page_override=old_num_pages)

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

    logger.info(
        "Reuse checks: "
        f"kv_cache_is_same={scheduler_b.engine.kv_cache is existing_kv_cache} "
        f"page_table_is_same={scheduler_b.engine.page_table is existing_page_table} "
        f"token_pool_is_same={scheduler_b.token_pool is existing_token_pool} "
        f"cache_mgr_is_same={scheduler_b.cache_manager.manager is existing_cache_manager.manager}"
    )

    assert scheduler_b.engine.kv_cache is existing_kv_cache
    assert scheduler_b.engine.page_table is existing_page_table
    assert scheduler_b.token_pool is existing_token_pool
    assert scheduler_b.cache_manager.manager is existing_cache_manager.manager

    assert int(scheduler_b.engine.page_table.data_ptr()) == page_table_ptr
    assert int(scheduler_b.token_pool.data_ptr()) == token_pool_ptr

    if k0_ptr is not None and hasattr(scheduler_b.engine.kv_cache, "k_cache") and hasattr(scheduler_b.engine.kv_cache, "v_cache"):
        k0_ptr_b = int(scheduler_b.engine.kv_cache.k_cache(0).data_ptr())
        v0_ptr_b = int(scheduler_b.engine.kv_cache.v_cache(0).data_ptr())
        logger.info(f"KV ptr check: k0_ptr(before={k0_ptr}, after={k0_ptr_b}) v0_ptr(before={v0_ptr}, after={v0_ptr_b})")
        assert k0_ptr_b == k0_ptr
        assert v0_ptr_b == v0_ptr
    
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

def reset_scheduler_state(scheduler: Scheduler) -> None:
    torch.cuda.synchronize(scheduler.engine.device)
    scheduler.prefill_manager.pending_list.clear()
    scheduler.decode_manager.running_reqs.clear()
    scheduler.finished_reqs.clear()
    scheduler.table_manager._free_slots = list(range(scheduler.table_manager._max_running_reqs))
    scheduler.cache_manager.manager.reset()
    scheduler.cache_manager._free_slots = torch.arange(
        scheduler.engine.num_pages, dtype=torch.int32, device=scheduler.engine.device
    )
    scheduler.page_table.fill_(scheduler.engine.dummy_page)
    scheduler.token_pool.zero_()

def normal_loop_no_throw(scheduler: Scheduler) -> None:
    try:
        scheduler.normal_loop()
    except Exception as e:
        if "RequestAllFinished" in str(type(e).__name__):
            return
        raise

def run_opencompass_json() -> None:
    from collections import deque

    from minisgl.core import SamplingParams
    from minisgl.message import UserMsg
    from minisgl.utils import init_logger

    logger = init_logger(__name__)

    json_in = os.environ.get(
        "MINISGL_OC_JSON_IN",
        "/home/wujl022/opencompass/outputs/llama3-8b-instruct-gptq4bit/20260126_151217/predictions/llama3-8b-instruct-gptq4bit-hf/gsm8k.json",
    )
    json_out = os.environ.get("MINISGL_OC_JSON_OUT", "").strip() or json_in

    attn = os.environ.get("MINISGL_ATTENTION_BACKEND", "fi")
    cache_type = os.environ.get("MINISGL_CACHE_TYPE", "naive")
    bs = int(os.environ.get("MINISGL_BATCH_SIZE", os.environ.get("MINISGL_BS", "70")))
    max_tokens = int(os.environ.get("MINISGL_MAX_TOKENS", os.environ.get("MINISGL_OUTPUT_LEN", "650")))
    split_after_env = int(os.environ.get("MINISGL_SPLIT_AFTER", "138"))
    split_after = (max_tokens // 2) if split_after_env < 0 else split_after_env

    one_batch = os.environ.get("MINISGL_ONE_BATCH", "0").strip() == "1"
    batch_idx = int(os.environ.get("MINISGL_BATCH_IDX", "0"))
    max_seq_len = int(os.environ.get("MINISGL_MAX_SEQ_LEN", "2200"))
    temperature = float(os.environ.get("MINISGL_TEMPERATURE", "0.7"))
    top_p = float(os.environ.get("MINISGL_TOP_P", "0.95"))

    model_a_path = os.environ.get(
        "MINISGL_MODEL_A_PATH",
        "/home/wujl022/models/LLM-Research/Meta-Llama-3-8B-Instruct",
    )
    model_b_path = os.environ.get(
        "MINISGL_MODEL_B_PATH",
        "/home/wujl022/models/Huggingface/Meta-Llama-3-8B-Instruct-GPTQ-4bit-gs128",
    )

    if split_after <= 0 or split_after >= max_tokens:
        raise ValueError(
            f"Invalid MINISGL_SPLIT_AFTER={split_after}, expected 0 < split_after < max_tokens={max_tokens}"
        )

    with open(json_in, "r", encoding="utf-8") as f:
        data = json.load(f)

    def _key_to_int(k: str) -> int:
        try:
            return int(k)
        except Exception:
            return 10**18

    items = sorted(data.items(), key=lambda kv: _key_to_int(kv[0]))
    prompts: list[str] = []
    keys: list[str] = []
    for k, v in items:
        keys.append(k)
        if isinstance(v, dict) and "origin_prompt" in v:
            prompts.append(str(v["origin_prompt"]))
        else:
            raise KeyError(f"Missing origin_prompt for key={k} in {json_in}")

    config_a = SchedulerConfig(
        model_path=model_a_path,
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        attention_backend=attn,
        max_running_req=bs,
        cache_type=cache_type,
        memory_ratio=0.9,
        cuda_graph_bs=[0],
        offline_mode=True,
        max_seq_len_override=max_seq_len,
    )

    config_b = SchedulerConfig(
        model_path=model_b_path,
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        attention_backend=attn,
        max_running_req=bs,
        cache_type=cache_type,
        memory_ratio=0.9,
        cuda_graph_bs=[0],
        offline_mode=True,
        max_seq_len_override=max_seq_len,
    )

    scheduler_b = None
    cur_model = "a"

    def hard_cleanup_scheduler() -> None:
        nonlocal scheduler_b, cur_model
        if scheduler_b is None:
            return

        import minisgl.core as _mcore

        try:
            torch.cuda.synchronize(scheduler_b.engine.device)
        except Exception:
            pass

        try:
            scheduler_b.shutdown()
        except Exception:
            try:
                scheduler_b.engine.graph_runner.destroy_cuda_graphs()
            except Exception:
                pass

        tmp = scheduler_b
        scheduler_b = None
        cur_model = "a"
        del tmp

        _mcore._GLOBAL_CTX = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        if hasattr(torch._C, "_cuda_clearCublasWorkspaces"):
            torch._C._cuda_clearCublasWorkspaces()

    def run_one_batch(batch_prompts: list[str]) -> list[str]:
        nonlocal scheduler_b, cur_model

        uids = list(range(len(batch_prompts)))
        pending_msgs = deque(
            [
                UserMsg(
                    uid=uid,
                    input_ids=scheduler_b.tokenizer.encode(p, return_tensors="pt").view(-1).to(torch.int32),
                    sampling_params=SamplingParams(
                        ignore_eos=False,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                    ),
                )
                for uid, p in zip(uids, batch_prompts)
            ]
        )

        counts = {uid: 0 for uid in uids}
        finished = {uid: False for uid in uids}
        out_ids = {uid: [] for uid in uids}

        def recv(blocking: bool = False):
            _ = blocking
            if not pending_msgs:
                return []
            msgs = list(pending_msgs)
            pending_msgs.clear()
            return msgs

        def send(reply):
            eos_id = int(scheduler_b.eos_token_id)
            for m in reply:
                if m.uid not in counts:
                    continue
                if not (m.finished and int(m.next_token) == eos_id):
                    counts[m.uid] += 1
                    out_ids[m.uid].append(int(m.next_token))
                if m.finished:
                    finished[m.uid] = True

        scheduler_b.receive_msg = recv
        scheduler_b.send_result = send

        it = 0
        while (
            any((not finished[uid]) and counts[uid] < split_after for uid in uids)
            and it < 200000
        ):
            it += 1
            normal_loop_no_throw(scheduler_b)

        if any(not finished[uid] for uid in uids):
            scheduler_b = switch_model_a_to_b(scheduler_b, config_b)
            cur_model = "b"
            scheduler_b.receive_msg = recv
            scheduler_b.send_result = send

        while any(not finished[uid] for uid in uids) and it < 400000:
            it += 1
            normal_loop_no_throw(scheduler_b)

        if any(not finished[uid] for uid in uids):
            unfinished = {uid: counts[uid] for uid in uids if not finished[uid]}
            raise RuntimeError(f"generation did not finish: unfinished={unfinished}")

        return [scheduler_b.tokenizer.decode(out_ids[uid]) for uid in uids]

    if one_batch:
        starts = [batch_idx * bs]
    else:
        starts = list(range(0, len(prompts), bs))

    for start in starts:
        if start < 0 or start >= len(prompts):
            raise SystemExit(0)

        scheduler_b = Scheduler(config_a)
        cur_model = "a"
        try:
            reset_scheduler_state(scheduler_b)
            batch_prompts = prompts[start : start + bs]
            batch_keys = keys[start : start + bs]
            batch_out = run_one_batch(batch_prompts)
            for k, pred in zip(batch_keys, batch_out):
                data[k]["prediction"] = pred
        finally:
            hard_cleanup_scheduler()

        processed = min(start + bs, len(prompts))
        logger.info(f"processed {processed}/{len(prompts)}")

        if one_batch:
            with open(json_out, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            logger.info(f"updated predictions (one batch): {json_out}")
            raise SystemExit(0)

    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    logger.info(f"updated predictions: {json_out}")

def run_bench() -> None:
    from collections import deque
    from random import randint, seed

    from minisgl.core import SamplingParams
    from minisgl.message import UserMsg
    from minisgl.utils import init_logger

    logger = init_logger(__name__)

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
        model_path="/root/autodl-tmp/Meta-Llama-3-8B-Instruct",
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
        model_path="/root/autodl-tmp/Meta-Llama-3-8B-Instruct-GPTQ-4bit-gs128",
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        attention_backend=attn,
        max_running_req=max(256, bs),
        cache_type=cache_type,
        memory_ratio=0.9,
        cuda_graph_bs=[0],
        offline_mode=True,
    )

    scheduler_b = scheduler_a

    prompt_token_ids = [[randint(0, 10000) for _ in range(input_len)] for _ in range(bs)]
    uids = list(range(0, bs))

    def run_case(
        name: str,
        *,
        switch_to_b: bool,
        output_len_override: int | None = None,
        split_after_override: int | None = None,
        measure: bool = True,
    ) -> None:
        nonlocal scheduler_b
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

        it = 0
        while min_cached_out() < local_split_after and it < 200000:
            it += 1
            normal_loop_no_throw(scheduler_b)
            if table_idx0 is None:
                r0 = next(
                    (r for r in scheduler_b.decode_manager.running_reqs if r.uid == verify_uid),
                    None,
                )
                if r0 is not None:
                    table_idx0 = int(r0.table_idx)

        if switch_to_b:
            scheduler_b = switch_model_a_to_b(scheduler_b, config_b)
            scheduler_b.receive_msg = recv
            scheduler_b.send_result = send

        t0 = None
        if measure:
            torch.cuda.synchronize(scheduler_b.engine.device)
            t0 = time.perf_counter()

        while min(counts.values()) < local_output_len and it < 400000:
            it += 1
            normal_loop_no_throw(scheduler_b)

        if measure:
            torch.cuda.synchronize(scheduler_b.engine.device)
            t1 = time.perf_counter()

        if min(counts.values()) < local_output_len:
            raise RuntimeError(f"{name}: generation did not finish: min_count={min(counts.values())}")

        if table_idx0 is not None:
            kv_pages = scheduler_b.page_table[
                table_idx0, prompt_len : prompt_len + local_output_len - 1
            ].to("cpu")
            assert torch.all(kv_pages < scheduler_b.engine.num_pages)

        if measure:
            assert t0 is not None
            elapsed_s = t1 - t0
            timed_tokens = bs * (local_output_len - local_split_after - 1)
            tok_s = timed_tokens / elapsed_s
            logger.info(
                f"{name}: bs={bs} input_len={input_len} output_len={local_output_len} split_after={local_split_after} "
                f"time_last={elapsed_s:.3f}s throughput_last={tok_s:.2f} tok/s"
            )

    reset_scheduler_state(scheduler_b)
    run_case("a_then_b_same_kv_buffer1", switch_to_b=True)

# ========== 使用示例 ==========
if __name__ == "__main__":
    from minisgl.utils import init_logger

    logger = init_logger(__name__)

    mode = os.environ.get("MINISGL_MODE", "oc_json").strip().lower()
    if mode == "oc_json":
        run_opencompass_json()
        raise SystemExit(0)

    if mode == "bench":
        run_bench()
        raise SystemExit(0)
