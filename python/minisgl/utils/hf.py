from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from transformers import LlamaConfig


@lru_cache()
def _load_config(model_path: str) -> Any:
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(model_path)


def cached_load_hf_config(model_path: str) -> LlamaConfig:
    config = _load_config(model_path)
    return type(config)(**config.to_dict())


@lru_cache()
def _resolve_hf_folder(model_path: str) -> str:
    if os.path.isdir(model_path):
        return model_path

    from huggingface_hub import snapshot_download

    return snapshot_download(
        model_path,
        allow_patterns=["quantize_config.json", "quantization_config.json"],
    )


@lru_cache()
def cached_load_hf_quantize_config(model_path: str) -> Dict[str, Any] | None:
    hf_folder = _resolve_hf_folder(model_path)
    candidates = [
        os.path.join(hf_folder, "quantize_config.json"),
        os.path.join(hf_folder, "quantization_config.json"),
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        return None

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data if isinstance(data, dict) else None
