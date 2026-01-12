from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.utils import divide_even

from .base import BaseOP


@dataclass(frozen=True)
class GPTQQuantConfig:
    bits: int
    group_size: int
    desc_act: bool
    sym: bool
    is_marlin_format: bool


def _parse_gptq_quantize_config(cfg: Dict[str, Any] | None) -> GPTQQuantConfig | None:
    if not cfg:
        return None
    quant_method = str(cfg.get("quant_method", "")).lower()
    if quant_method != "gptq":
        return None
    bits = int(cfg.get("bits", 0) or 0)
    group_size = int(cfg.get("group_size", -1) or -1)
    desc_act = bool(cfg.get("desc_act", False))
    sym = bool(cfg.get("sym", True))
    is_marlin_format = (
        str(cfg.get("checkpoint_format", "")).lower() == "marlin" or bool(cfg.get("is_marlin_format", False))
    )
    return GPTQQuantConfig(
        bits=bits,
        group_size=group_size,
        desc_act=desc_act,
        sym=sym,
        is_marlin_format=is_marlin_format,
    )


def _marlin_make_workspace(device: torch.device, max_blocks_per_sm: int = 1) -> torch.Tensor:
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    return torch.zeros(sms * max_blocks_per_sm, dtype=torch.int, device=device)


def _marlin_is_k_full(act_order: bool, is_row_parallel: bool) -> bool:
    return (not act_order) or (act_order and not is_row_parallel)


def _marlin_make_empty_int(device: torch.device) -> torch.Tensor:
    return torch.empty((0,), dtype=torch.int32, device=device)


def _marlin_sort_g_idx(g_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.argsort(g_idx).to(torch.int32)
    return g_idx[idx], idx


def _marlin_get_scale_perms() -> tuple[list[int], list[int]]:
    scale_perm: list[int] = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single: list[int] = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return scale_perm, scale_perm_single


def _marlin_permute_scales(s: torch.Tensor, *, size_k: int, size_n: int, group_size: int) -> torch.Tensor:
    scale_perm, scale_perm_single = _marlin_get_scale_perms()
    if group_size < size_k and group_size != -1:
        s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
    else:
        s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    return s.reshape((-1, size_n)).contiguous()


def _can_use_gptq_marlin(q: GPTQQuantConfig, *, in_features: int, out_features: int, device: torch.device) -> bool:
    if q.is_marlin_format:
        return True
    if device.type != "cuda":
        return False
    major, _minor = torch.cuda.get_device_capability(device)
    if major < 8:
        return False
    if not (q.sym and q.bits in (4, 8)):
        return False
    if q.group_size not in (-1, 32, 64, 128):
        return False
    if out_features % 128 != 0:
        return False
    if in_features % 64 != 0:
        return False
    if q.group_size != -1 and (in_features % q.group_size != 0):
        return False
    return True


class _LinearTPImpl(BaseOP):
    def __init__(
        self,
        full_isize: int,
        full_osize: int,
        local_isize: int,
        local_osize: int,
        has_bias: bool,
        *,
        quantize_config: Dict[str, Any] | None,
    ):
        self.full_input_size = full_isize
        self.full_output_size = full_osize
        self.local_input_size = local_isize
        self.local_output_size = local_osize
        self._is_row_parallel = full_isize != local_isize

        self._gptq = _parse_gptq_quantize_config(quantize_config)
        self._gptq_use_shuffle = True
        self._use_marlin = False
        self._marlin_wtype = None
        self._marlin_is_k_full = True
        self.g_idx_sort_indices = None
        self.workspace = None

        if self._gptq is None:
            self.weight = torch.empty(local_osize, local_isize)
            self.bias = torch.empty(local_osize) if has_bias else None
            self.qweight = None
            self.qzeros = None
            self.scales = None
            self.g_idx = None
            return

        bits = self._gptq.bits
        if bits not in (4, 8):
            raise ValueError(f"Unsupported GPTQ bits: {bits}")
        pack_factor = 32 // bits

        group_size = self._gptq.group_size
        if group_size == -1:
            group_size_eff = full_isize
        else:
            group_size_eff = group_size

        shard_scales = self._is_row_parallel and (group_size != -1) and (not self._gptq.desc_act)
        if shard_scales:
            scales_k = local_isize // group_size_eff
        else:
            scales_k = full_isize // group_size_eff

        self.weight = None
        self.bias = torch.empty(local_osize) if has_bias else None
        self.qweight = torch.empty(local_isize // pack_factor, local_osize, dtype=torch.int32)
        self.qzeros = torch.empty(scales_k, local_osize // pack_factor, dtype=torch.int32)
        self.scales = torch.empty(scales_k, local_osize)
        self.g_idx = torch.empty(local_isize if self._is_row_parallel else full_isize, dtype=torch.int32)

    def process_weights_after_loading(self) -> None:
        if self._gptq is None:
            return

        try:
            from sgl_kernel import gptq_marlin_repack, gptq_shuffle
            from sgl_kernel.scalar_type import scalar_types
        except Exception:
            return

        device = self.qweight.device

        if self._use_marlin:
            return

        if _can_use_gptq_marlin(
            self._gptq,
            in_features=self.local_input_size,
            out_features=self.local_output_size,
            device=device,
        ):
            if self._gptq.bits == 4:
                wtype = scalar_types.uint4b8
            else:
                wtype = scalar_types.uint8b128

            self.workspace = _marlin_make_workspace(device)
            self._marlin_is_k_full = _marlin_is_k_full(self._gptq.desc_act, self._is_row_parallel)
            self._marlin_wtype = wtype

            if not self._gptq.is_marlin_format:
                if self._gptq.desc_act:
                    g_idx_sorted, g_idx_sort_indices = _marlin_sort_g_idx(self.g_idx)
                    self.g_idx = g_idx_sorted
                    self.g_idx_sort_indices = g_idx_sort_indices
                else:
                    self.g_idx = _marlin_make_empty_int(device)
                    self.g_idx_sort_indices = _marlin_make_empty_int(device)

                self.qweight = gptq_marlin_repack(
                    self.qweight.contiguous(),
                    perm=self.g_idx_sort_indices,
                    size_k=self.local_input_size,
                    size_n=self.local_output_size,
                    num_bits=self._gptq.bits,
                )
                self.scales = _marlin_permute_scales(
                    self.scales.contiguous(),
                    size_k=self.local_input_size,
                    size_n=self.local_output_size,
                    group_size=self._gptq.group_size,
                )
                self.qzeros = _marlin_make_empty_int(device)
            else:
                if self._gptq.desc_act:
                    self.g_idx, self.g_idx_sort_indices = _marlin_sort_g_idx(self.g_idx)
                else:
                    self.g_idx = _marlin_make_empty_int(device)
                    self.g_idx_sort_indices = _marlin_make_empty_int(device)

            self._use_marlin = True
            return

        if self._is_row_parallel and (self._gptq.group_size != -1) and self._gptq.desc_act:
            self._gptq_use_shuffle = False

        if self._gptq_use_shuffle:
            if self._gptq.desc_act:
                self.g_idx = torch.argsort(self.g_idx).to(torch.int32)
            else:
                self.g_idx = _marlin_make_empty_int(device)
            gptq_shuffle(self.qweight, self.g_idx, self._gptq.bits)

    def _apply_linear(self, x: torch.Tensor) -> torch.Tensor:
        if self._gptq is None:
            return F.linear(x, self.weight, self.bias)

        out_shape = x.shape[:-1] + (self.local_output_size,)
        reshaped_x = x.reshape(-1, x.shape[-1])

        if self._use_marlin:
            from sgl_kernel import gptq_marlin_gemm

            y = gptq_marlin_gemm(
                reshaped_x,
                None,
                self.qweight,
                self.scales,
                None,
                None,
                self.g_idx,
                self.g_idx_sort_indices,
                self.workspace,
                self._marlin_wtype,
                size_m=reshaped_x.shape[0],
                size_n=self.local_output_size,
                size_k=self.local_input_size,
                is_k_full=self._marlin_is_k_full,
                use_atomic_add=False,
                use_fp32_reduce=True,
                is_zp_float=False,
            )
            if self.bias is not None:
                y.add_(self.bias)
            return y.reshape(out_shape)

        from sgl_kernel import gptq_gemm

        y = gptq_gemm(
            reshaped_x,
            self.qweight,
            self.qzeros,
            self.scales,
            self.g_idx,
            self._gptq_use_shuffle,
            self._gptq.bits,
        )
        if self.bias is not None:
            y.add_(self.bias)
        return y.reshape(out_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._apply_linear(x)


class LinearColParallelMerged(_LinearTPImpl):
    def __init__(
        self,
        input_size: int,
        output_sizes: List[int],
        has_bias: bool,
        *,
        quantize_config: Dict[str, Any] | None = None,
    ):
        tp_info = get_tp_info()
        tp_output_sizes = [divide_even(size, tp_info.size) for size in output_sizes]
        output_size = sum(output_sizes)
        tp_output_size = sum(tp_output_sizes)
        super().__init__(
            input_size,
            output_size,
            input_size,
            tp_output_size,
            has_bias,
            quantize_config=quantize_config,
        )


class LinearQKVMerged(_LinearTPImpl):
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_qo_heads: int,
        num_kv_heads: int,
        has_bias: bool,
        *,
        quantize_config: Dict[str, Any] | None = None,
    ):
        tp_info = get_tp_info()

        GQA_ratio = divide_even(num_qo_heads, num_kv_heads)
        local_num_kv = divide_even(num_kv_heads, tp_info.size)
        full_isize = hidden_size
        full_osize = (GQA_ratio + 2) * num_kv_heads * head_dim
        local_isize = hidden_size
        local_osize = (GQA_ratio + 2) * local_num_kv * head_dim
        super().__init__(
            full_isize,
            full_osize,
            local_isize,
            local_osize,
            has_bias,
            quantize_config=quantize_config,
        )


class LinearOProj(_LinearTPImpl):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool,
        *,
        quantize_config: Dict[str, Any] | None = None,
    ):
        tp_info = get_tp_info()
        full_isize = input_size
        full_osize = output_size
        local_isize = divide_even(input_size, tp_info.size)
        local_osize = output_size
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size
        super().__init__(
            full_isize,
            full_osize,
            local_isize,
            local_osize,
            has_bias,
            quantize_config=quantize_config,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self._apply_linear(x)
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return y


class LinearRowParallel(_LinearTPImpl):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool,
        *,
        quantize_config: Dict[str, Any] | None = None,
    ):
        tp_info = get_tp_info()
        local_input_size = divide_even(input_size, tp_info.size)
        local_output_size = output_size
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size
        super().__init__(
            input_size,
            output_size,
            local_input_size,
            local_output_size,
            has_bias,
            quantize_config=quantize_config,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self._apply_linear(x)
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return y
