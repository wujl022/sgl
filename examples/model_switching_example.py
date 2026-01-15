"""
示例：模型切换时复用 KV Cache

场景：
- 模型A部署好了，执行 req0 到一半
- 卸载模型A，部署模型B（量化版本，权重更少）
- 保留复用：kv_buffer, page_table, radix_tree_node, _free_slots
- 因为模型B权重更少，num_pages 变多，扩展 kv_buffer 和 _free_slots
"""

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
    logger.info(f"Total: {free_slots_len + radix_tree_size} (should equal {new_num_pages})")
    
    assert (
        free_slots_len + radix_tree_size == new_num_pages
    ), f"Integrity check failed: {free_slots_len} + {radix_tree_size} != {new_num_pages}"
    
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
    from minisgl.message import UserMsg, DetokenizeMsg
    from collections import deque
    
    logger = init_logger(__name__)
    
    # 配置模型A
    config_a = SchedulerConfig(
        model_path="/root/autodl-tmp/Meta-Llama-3-8B-Instruct",
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        max_running_req=256,
        cache_type="radix",
        memory_ratio=0.5,
        offline_mode=True,  # 使用离线模式
    )
    
    # 部署模型A
    scheduler_a = Scheduler(config_a)
    logger.info("Model A deployed")
    
    # ========== 发送请求 "what is ai?" ==========
    prompt = "what is ai?"
    tokenizer = scheduler_a.tokenizer
    input_ids = tokenizer.encode(prompt, return_tensors="pt").view(-1).to(torch.int32)
    
    # 创建请求消息
    user_msg = UserMsg(
        uid=0,
        input_ids=input_ids,
        sampling_params=SamplingParams(temperature=0.7, max_tokens=10),
    )
    
    # 存储输出的 tokens
    output_tokens = []
    output_state = {"text": "", "count": 0}
    switch_after_tokens = 5  # 输出5个tokens后切换
    
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
    max_iterations = 10
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
        model_path="/root/autodl-tmp/Llama-3-8B-Instruct-GPTQ-4-Bit",
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        max_running_req=256,
        cache_type="radix",
        memory_ratio=0.4,
        offline_mode=True,
    )
    
    scheduler_b = switch_model_a_to_b(scheduler_a, config_b)
    
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
    
    # 继续生成直到完成
    max_iterations = 1000
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        try:
            scheduler_b.normal_loop()
            # 检查请求是否完成
            if 0 in {req.uid for req in scheduler_b.finished_reqs}:
                break
        except Exception as e:
            if "RequestAllFinished" in str(type(e).__name__):
                break
            raise
    
    final_output = scheduler_b.tokenizer.decode(output_tokens, skip_special_tokens=True)
    logger.info(f"Final output ({len(output_tokens)} tokens): {final_output!r}")
    logger.info("Request completed!")

