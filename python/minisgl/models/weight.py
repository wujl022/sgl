from __future__ import annotations

import glob
import os
from typing import Any, Dict

import safetensors
import torch
from huggingface_hub import snapshot_download
from minisgl.distributed import get_tp_info
from minisgl.utils import divide_up
from tqdm.asyncio import tqdm


class DisabledTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, disable=True)


def _shard_state_dict(
    state_dict: Dict[str, torch.Tensor],
    quantize_config: Dict[str, Any] | None,
) -> Dict[str, torch.Tensor]:
    shard_state_dict: Dict[str, torch.Tensor] = {}
    tp_info = get_tp_info()
    r = tp_info.rank
    n = tp_info.size

    qcfg = quantize_config or {}
    quant_method = str(qcfg.get("quant_method", "")).lower()
    is_gptq = quant_method == "gptq" or any(
        k.endswith((".qweight", ".qzeros", ".scales", ".g_idx")) for k in state_dict.keys()
    )
    bits = int(qcfg.get("bits", 0) or 0)
    pack_factor = 32 // bits if bits in (4, 8) else None
    group_size = int(qcfg.get("group_size", -1) or -1)
    desc_act = bool(qcfg.get("desc_act", False))

    SPLIT_DIM_0_LIST = [
        ".q_proj",
        ".k_proj",
        ".v_proj",
        ".gate_proj",
        ".up_proj",
    ]
    SPLIT_DIM_1_LIST = [
        ".o_proj",
        ".down_proj",
    ]

    def _should_shard_scales_and_zeros_for_row_parallel() -> bool:
        return is_gptq and (group_size != -1) and (not desc_act)

    for key, value in state_dict.items():
        if any(key.count(sub) for sub in SPLIT_DIM_0_LIST):
            if is_gptq and pack_factor is not None:
                if key.endswith(".qweight"):
                    shard_state_dict[key] = value.chunk(n, dim=1)[r]
                elif key.endswith(".qzeros"):
                    shard_state_dict[key] = value.chunk(n, dim=1)[r]
                elif key.endswith(".scales"):
                    shard_state_dict[key] = value.chunk(n, dim=1)[r]
                elif key.endswith(".bias"):
                    shard_state_dict[key] = value.chunk(n, dim=0)[r]
                else:
                    shard_state_dict[key] = value
            else:
                shard_state_dict[key] = value.chunk(n, dim=0)[r]

        elif any(key.count(sub) for sub in SPLIT_DIM_1_LIST):
            if is_gptq and pack_factor is not None:
                if key.endswith(".qweight"):
                    shard_state_dict[key] = value.chunk(n, dim=0)[r]
                elif key.endswith(".g_idx"):
                    shard_state_dict[key] = value.chunk(n, dim=0)[r]
                elif key.endswith(".qzeros"):
                    if _should_shard_scales_and_zeros_for_row_parallel():
                        shard_state_dict[key] = value.chunk(n, dim=0)[r]
                    else:
                        shard_state_dict[key] = value
                elif key.endswith(".scales"):
                    if _should_shard_scales_and_zeros_for_row_parallel():
                        shard_state_dict[key] = value.chunk(n, dim=0)[r]
                    else:
                        shard_state_dict[key] = value
                else:
                    shard_state_dict[key] = value
            else:
                shard_state_dict[key] = value.chunk(n, dim=1)[r]

        elif key.count("lm_head") or key.count("embed_tokens"):
            num_embeddings = value.shape[0]
            num_embeddings_per_partition = divide_up(num_embeddings, n)
            vocab_start_idx = r * num_embeddings_per_partition
            vocab_end_idx = min((r + 1) * num_embeddings_per_partition, num_embeddings)
            shard_state_dict[key] = value[vocab_start_idx:vocab_end_idx, :]
        else:
            shard_state_dict[key] = value

    return shard_state_dict


def _merge_state_dict(
    state_dict: Dict[str, torch.Tensor],
    quantize_config: Dict[str, Any] | None,
) -> Dict[str, torch.Tensor]:
    filtered_state_dict: Dict[str, torch.Tensor] = {}
    qcfg = quantize_config or {}
    bits = int(qcfg.get("bits", 0) or 0)
    pack_factor = 32 // bits if bits in (4, 8) else None

    def _merge_three(prefix_key: str, src: str, dst: str, cat_dim: int) -> None:
        q = state_dict[prefix_key]
        k = state_dict[prefix_key.replace(src, ".k_proj")]
        v = state_dict[prefix_key.replace(src, ".v_proj")]
        new_key = prefix_key.replace(src, dst)
        filtered_state_dict[new_key] = torch.cat([q, k, v], dim=cat_dim)
        del state_dict[prefix_key]
        del state_dict[prefix_key.replace(src, ".k_proj")]
        del state_dict[prefix_key.replace(src, ".v_proj")]

    def _merge_two(prefix_key: str, src: str, other: str, dst: str, cat_dim: int) -> None:
        a = state_dict[prefix_key]
        b = state_dict[prefix_key.replace(src, other)]
        new_key = prefix_key.replace(src, dst)
        filtered_state_dict[new_key] = torch.cat([a, b], dim=cat_dim)
        del state_dict[prefix_key]
        del state_dict[prefix_key.replace(src, other)]

    keys = list(state_dict.keys())
    for key in keys:
        if key not in state_dict:
            continue

        if ".q_proj" in key:
            if key.endswith(".qweight") or key.endswith(".qzeros") or key.endswith(".scales"):
                cat_dim = 1
            elif key.endswith(".g_idx"):
                q = state_dict[key]
                k = state_dict[key.replace(".q_proj", ".k_proj")]
                v = state_dict[key.replace(".q_proj", ".v_proj")]
                if not (torch.equal(q, k) and torch.equal(q, v)):
                    raise ValueError(f"g_idx mismatch when merging qkv: {key}")
                filtered_state_dict[key.replace(".q_proj", ".qkv_proj")] = q
                del state_dict[key]
                del state_dict[key.replace(".q_proj", ".k_proj")]
                del state_dict[key.replace(".q_proj", ".v_proj")]
                continue
            else:
                cat_dim = 0

            _merge_three(key, ".q_proj", ".qkv_proj", cat_dim)

        elif ".gate_proj" in key:
            if key.endswith(".qweight") or key.endswith(".qzeros") or key.endswith(".scales"):
                cat_dim = 1
            elif key.endswith(".g_idx"):
                gate = state_dict[key]
                up = state_dict[key.replace(".gate_proj", ".up_proj")]
                if not torch.equal(gate, up):
                    raise ValueError(f"g_idx mismatch when merging gate/up: {key}")
                filtered_state_dict[key.replace(".gate_proj", ".gate_up_proj")] = gate
                del state_dict[key]
                del state_dict[key.replace(".gate_proj", ".up_proj")]
                continue
            else:
                cat_dim = 0

            _merge_two(key, ".gate_proj", ".up_proj", ".gate_up_proj", cat_dim)

        elif ".k_proj" in key or ".v_proj" in key or ".up_proj" in key:
            continue
        else:
            filtered_state_dict[key] = state_dict[key]

    return filtered_state_dict


def load_hf_weight(
    model_path: str,
    device: torch.device,
    *,
    quantize_config: Dict[str, Any] | None = None,
) -> Dict[str, torch.Tensor]:
    if os.path.isdir(model_path):
        hf_folder = model_path
    else:
        try:
            hf_folder = snapshot_download(
                model_path,
                allow_patterns=["*.safetensors"],
                tqdm_class=DisabledTqdm,
            )
        except Exception:
            raise ValueError(
                f"Model path '{model_path}' is neither a local directory nor a valid HuggingFace repository ID"
            )

    # find the all *.pt files in the hf_folder
    files = glob.glob(f"{hf_folder}/*.safetensors")
    state_dict: Dict[str, torch.Tensor] = {}
    for file in sorted(files):
        with safetensors.safe_open(file, framework="pt", device="cpu") as f:
            for name in f.keys():
                state_dict[name] = f.get_tensor(name)

    if get_tp_info().size > 1:
        state_dict = _shard_state_dict(state_dict, quantize_config)

    state_dict = {k: v.to(device) for k, v in state_dict.items()}
    return _merge_state_dict(state_dict, quantize_config)
