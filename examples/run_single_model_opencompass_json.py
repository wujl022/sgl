import gc
import json
import os
from collections import deque

import torch
from minisgl.core import SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.message import UserMsg
from minisgl.scheduler import Scheduler, SchedulerConfig
from minisgl.utils import init_logger


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


def run_opencompass_json_single_model() -> None:
    logger = init_logger(__name__)

    json_in = os.environ.get(
        "MINISGL_OC_JSON_IN",
        "/home/wujl022/opencompass/outputs/llama3-8b-instruct-gptq4bit/20260126_151217/predictions/llama3-8b-instruct-gptq4bit-hf/gsm8k.json",
    )
    json_out = os.environ.get("MINISGL_OC_JSON_OUT", "").strip() or json_in

    model_path = os.environ.get(
        "MINISGL_MODEL_PATH",
        "/home/wujl022/models/Huggingface/Meta-Llama-3-8B-Instruct-GPTQ-4bit-gs128",
    )

    attn = os.environ.get("MINISGL_ATTENTION_BACKEND", "fi")
    cache_type = os.environ.get("MINISGL_CACHE_TYPE", "naive")
    bs = int(os.environ.get("MINISGL_BATCH_SIZE", os.environ.get("MINISGL_BS", "70")))
    max_tokens = int(os.environ.get("MINISGL_MAX_TOKENS", os.environ.get("MINISGL_OUTPUT_LEN", "650")))
    max_seq_len = int(os.environ.get("MINISGL_MAX_SEQ_LEN", "2200"))
    temperature = float(os.environ.get("MINISGL_TEMPERATURE", "0.7"))
    top_p = float(os.environ.get("MINISGL_TOP_P", "0.95"))

    one_batch = os.environ.get("MINISGL_ONE_BATCH", "0").strip() == "1"
    batch_idx = int(os.environ.get("MINISGL_BATCH_IDX", "0"))

    tp_rank = int(os.environ.get("MINISGL_TP_RANK", os.environ.get("MINISGL_CUDA_ID", "0")))
    max_steps = int(os.environ.get("MINISGL_MAX_STEPS", "400000"))

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

    config = SchedulerConfig(
        model_path=model_path,
        tp_info=DistributedInfo(tp_rank, 1),
        dtype=torch.bfloat16,
        attention_backend=attn,
        max_running_req=bs,
        cache_type=cache_type,
        memory_ratio=0.9,
        cuda_graph_bs=[0],
        offline_mode=True,
        max_seq_len_override=max_seq_len,
    )

    scheduler: Scheduler | None = None

    def hard_cleanup_scheduler() -> None:
        nonlocal scheduler
        if scheduler is None:
            return

        import minisgl.core as _mcore

        try:
            torch.cuda.synchronize(scheduler.engine.device)
        except Exception:
            pass

        try:
            scheduler.shutdown()
        except Exception:
            try:
                scheduler.engine.graph_runner.destroy_cuda_graphs()
            except Exception:
                pass

        tmp = scheduler
        scheduler = None
        del tmp

        _mcore._GLOBAL_CTX = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        if hasattr(torch._C, "_cuda_clearCublasWorkspaces"):
            torch._C._cuda_clearCublasWorkspaces()

    def run_one_batch(batch_prompts: list[str]) -> list[str]:
        assert scheduler is not None

        uids = list(range(len(batch_prompts)))
        pending_msgs = deque(
            [
                UserMsg(
                    uid=uid,
                    input_ids=scheduler.tokenizer.encode(p, return_tensors="pt")
                    .view(-1)
                    .to(torch.int32),
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
            eos_id = int(scheduler.eos_token_id)
            for m in reply:
                if m.uid not in counts:
                    continue
                if not (m.finished and int(m.next_token) == eos_id):
                    counts[m.uid] += 1
                    out_ids[m.uid].append(int(m.next_token))
                if m.finished:
                    finished[m.uid] = True

        scheduler.receive_msg = recv
        scheduler.send_result = send

        it = 0
        while any(not finished[uid] for uid in uids) and it < max_steps:
            it += 1
            normal_loop_no_throw(scheduler)

        if any(not finished[uid] for uid in uids):
            unfinished = {uid: counts[uid] for uid in uids if not finished[uid]}
            raise RuntimeError(f"generation did not finish: unfinished={unfinished}")

        return [scheduler.tokenizer.decode(out_ids[uid]) for uid in uids]

    if one_batch:
        starts = [batch_idx * bs]
    else:
        starts = list(range(0, len(prompts), bs))

    for start in starts:
        if start < 0 or start >= len(prompts):
            raise SystemExit(0)

        scheduler = Scheduler(config)
        try:
            reset_scheduler_state(scheduler)
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


if __name__ == "__main__":
    run_opencompass_json_single_model()