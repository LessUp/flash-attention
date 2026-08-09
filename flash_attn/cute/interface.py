# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# [2025-07-04] Cute-DSL 版本，面向 Hopper 和 Blackwell。需要安装 nvidia-cutlass-dsl==4.2.0。

import os
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple, Callable

import torch



import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32
from quack.compile_utils import make_fake_tensor as fake_tensor
from flash_attn.cute.cache_utils import get_jit_cache
from flash_attn.cute.testing import is_fake_mode


if os.environ.get("CUTE_DSL_PTXAS_PATH", None) is not None:
    from flash_attn.cute import cute_dsl_ptxas  # noqa: F401

    # 打补丁：导出 PTX，然后用系统 ptxas 编译成 cubin
    cute_dsl_ptxas.patch()


from flash_attn.cute import utils
from flash_attn.cute import fa_logging
from flash_attn.cute.cute_dsl_utils import (
    get_aux_tensor_metadata,
    get_broadcast_dims,
    get_num_sms_for_selection,
    maybe_contiguous,
    to_cute_aux_tensor,
    to_cute_tensor,
    validate_output_layout,
)
from flash_attn.cute.flash_fwd import FlashAttentionForwardSm80
from flash_attn.cute.flash_fwd_sm90 import FlashAttentionForwardSm90
from flash_attn.cute.flash_fwd_sm100 import FlashAttentionForwardSm100, DescaleTensors
from flash_attn.cute.flash_fwd_sm120 import FlashAttentionForwardSm120
from flash_attn.cute.flash_bwd_preprocess import FlashAttentionBackwardPreprocess
from flash_attn.cute.flash_bwd import FlashAttentionBackwardSm80
from flash_attn.cute.flash_bwd_sm90 import FlashAttentionBackwardSm90
from flash_attn.cute.flash_bwd_sm100 import FlashAttentionBackwardSm100
from flash_attn.cute.flash_bwd_sm120 import FlashAttentionBackwardSm120
from flash_attn.cute.flash_bwd_postprocess import (
    FlashAttentionBackwardPostprocess,
    LearnableSinkBwdTensors,
)
from flash_attn.cute.flash_fwd_combine import FlashAttentionForwardCombine
from flash_attn.cute.flash_fwd_mla_sm100 import FlashAttentionMLAForwardSm100
from flash_attn.cute.prepare_scheduler import FlashPrepareScheduler, SchedulerMetadataTensorsTorch
from flash_attn.cute.cu_blocks_kernel import CuSeqlensToBlocksKernel, CuBlocksToBatchKernel
from flash_attn.cute.flash_bwd_mla_sm100 import FlashAttentionSparseMLABackwardSm100
from flash_attn.cute.flash_bwd_mla_dq_dqv_sm100 import dQdQvGemmKernel
from flash_attn.cute.flash_bwd_mla_dk_sm100 import dKGemmKernel

# SM100 head_dim=256 2CTA kernel 导入
from flash_attn.cute.sm100_hd256_2cta_fmha_forward import BlackwellFusedMultiHeadAttentionForward
from flash_attn.cute.sm100_hd256_2cta_fmha_backward import BlackwellFusedMultiHeadAttentionBackward

from flash_attn.cute.utils import AuxData
from flash_attn.cute.block_sparsity import (
    BlockSparseTensorsTorch,
    block_sparse_bwd_supports_2cta,
    get_kv_subtile_factor,
    get_sparse_q_block_size,
    to_cute_block_sparse_tensors,
    normalize_block_sparse_config,
    normalize_block_sparse_config_bwd,
)

BIN_BATCH_SEARCH_THRESH = 256  # 超过此 batch 大小时，SingleTileVarlenScheduler 会启用 batch 查找辅助
# 在 cu hint 生效的地方，用 O(1) 的 flat-block -> batch 查找代替二分查找。
USE_BLOCKS_TO_BATCH: bool = True


def _parse_arch_str(arch_str):
    """解析架构字符串（例如 'sm_80'、'sm_90a'、'80'、'100'）为整数（例如 80、90、100）。"""
    import re
    match = re.match(r"^(?:sm_?|SM_?)?(\d+)(\d)([af]?)$", arch_str)
    if not match:
        raise ValueError(f"Invalid arch format: {arch_str}")
    major, minor, _ = match.groups()
    return int(major) * 10 + int(minor)


@lru_cache(maxsize=None)
def _get_device_arch():
    """带缓存的设备架构检查。

    可通过 FLASH_ATTENTION_ARCH（例如 'sm_80' 或 '80'）覆盖默认值，从而独立于
    编译目标（CUTE_DSL_ARCH）选择要走哪条 kernel 路径（SM80/SM90/SM100/SM120）。

    仅在 CPU 上编译（没有 GPU）时，请设置：
      FLASH_ATTENTION_ARCH=sm_80  （kernel 选择）
      CUTE_DSL_ARCH=sm_80         （编译目标）
      FLASH_ATTENTION_NUM_SMS=132 （目标 SKU 选择器元数据）
    """
    arch_override = os.environ.get("FLASH_ATTENTION_ARCH", None)
    if arch_override is not None:
        return _parse_arch_str(arch_override)
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + int(minor)


@lru_cache(maxsize=None)
def _validate_head_dims(head_dim: int, head_dim_v: int, compute_capability: int, alignment: int) -> None:
    """根据计算能力（compute capability）校验 head_dim 的约束条件。"""
    is_deepseek_shape = head_dim == 192 and head_dim_v == 128
    is_deepseek_mla_absorbed_shape = (head_dim == 64 or head_dim == head_dim_v) and head_dim_v == 512
    is_dedicate_kernel_shape = head_dim == 256 and head_dim_v == 256
    is_standard_range = 8 <= head_dim <= 128 and 8 <= head_dim_v <= 128

    is_sm90_range = 8 <= head_dim <= 256 and 8 <= head_dim_v <= 256
    if compute_capability == 9:
        assert is_sm90_range and head_dim % alignment == 0 and head_dim_v % alignment == 0, (
            f"(head_dim, head_dim_v)=({head_dim}, {head_dim_v}) is not supported on SM90. "
            f"head_dim and head_dim_v must be between 8 and 256 and divisible by {alignment}."
        )
    elif compute_capability in [10, 11]:
        assert (is_standard_range or is_deepseek_shape or is_deepseek_mla_absorbed_shape or is_dedicate_kernel_shape) and head_dim % alignment == 0 and head_dim_v % alignment == 0, (
            f"(head_dim, head_dim_v)=({head_dim}, {head_dim_v}) is not supported on SM100/SM110. "
            f"head_dim and head_dim_v must be between 8 and 128 and divisible by {alignment}, or (192, 128) for DeepSeek, or (256, 256) for hd256."
        )


@dataclass(frozen=True)
class FwdConfig:
    m_block_size: int
    n_block_size: int
    mma_pv_is_rs: bool
    intra_wg_overlap: bool
    q_stage: int = 1
    num_splits: int = 1


def _tile_size_fwd_sm90(head_dim, head_dim_v, is_causal, is_local, sparse_block_size_q=None):
    """返回 SM90 前向的 FwdConfig。

    tile 大小与标志位参考 hopper/tile_size.h 中的 tile_size_fwd_sm90，并根据 Python
    kernel 在寄存器/共享内存（smem）上的不同取舍做了调整（在 H100 SXM 上基准测试过）。

    设置了 sparse_block_size_q 时，tile_m 必须能整除它。对于 head_dim <= 96，当兼容时
    使用最优的 tile_m=192，否则回退到 128。
    """
    if head_dim <= 64:
        # C++：192×192 非 causal，192×128 causal/local。
        # Python：192×128 RS+OL（寄存器共享 + 输出重叠）在各种 seqlen 下都稳定最优。
        if sparse_block_size_q is not None and sparse_block_size_q % 192 != 0:
            return FwdConfig(128, 128, True, True)
        return FwdConfig(192, 128, True, True)
    elif head_dim <= 96:
        # C++：所有情况下都用 192×144 noRS+OL。
        # Python：RS 与 192× tile 组合是灾难性的（约 300 对 600 TFLOPS）。
        # 始终要求 noRS+OL。causal 时短 seqlen 下 192×128 略优。
        if sparse_block_size_q is not None and sparse_block_size_q % 192 != 0:
            return FwdConfig(128, 128, False, True)
        if is_causal or is_local:
            return FwdConfig(192, 128, False, True)
        else:
            return FwdConfig(192, 144, False, True)
    elif head_dim <= 128:
        return FwdConfig(128, 128, True, True)
    elif head_dim <= 192:
        tile_n = 96 if is_local else (128 if head_dim_v <= 128 else 112)
        return FwdConfig(128, tile_n, True, True)
    else:  # hdim 256
        tile_n = 64 if is_local else 80
        return FwdConfig(128, tile_n, True, True)

@dataclass(frozen=True)
class BwdConfig:
    m_block_size: int
    n_block_size: int
    num_stages_Q: int
    num_stages_dO: int
    num_stages_PdS: int
    SdP_swapAB: bool
    dKV_swapAB: bool
    dQ_swapAB: bool
    AtomLayoutMSdP: int
    AtomLayoutNdKV: int
    AtomLayoutMdQ: int
    num_wg: int = 2  # MMA warp group 数量（总线程数 = (num_wg + 1) * 128）
    dQ_single_wg: bool = False


def _tile_size_bwd_sm90(head_dim, head_dim_v, causal, local, sparse_block_size_q=None):
    """返回 SM90 反向的 BwdConfig。

    配置参考 C++ FA3 的 hopper/flash_bwd_launch_template.h，
    在 H100 SXM 上基准测试过。
    """
    if head_dim <= 64:
        # C++ FA3：128, 128, 64, ..., 2, 2, true, false, false, 2, 1, 2, 2
        return BwdConfig(
            m_block_size=128, n_block_size=128,
            num_stages_Q=2, num_stages_dO=2, num_stages_PdS=2,
            SdP_swapAB=True, dKV_swapAB=False, dQ_swapAB=False,
            AtomLayoutMSdP=1, AtomLayoutNdKV=2, AtomLayoutMdQ=2,
        )
    elif head_dim <= 96:
        # C++ FA3：64, 128, 96, dQ_swapAB=False
        return BwdConfig(
            m_block_size=64, n_block_size=128,
            num_stages_Q=2, num_stages_dO=2, num_stages_PdS=2,
            SdP_swapAB=True, dKV_swapAB=False, dQ_swapAB=False,
            AtomLayoutMSdP=1, AtomLayoutNdKV=2, AtomLayoutMdQ=1,
            dQ_single_wg=True,
        )
    elif head_dim <= 128:
        # C++ FA3：causal/local 用 64, 128；non-causal 用 80, 128 且带 dQ_swapAB
        is_causal_or_local = causal or local
        m_block_size = 64 if is_causal_or_local else 80
        if sparse_block_size_q is not None and sparse_block_size_q % m_block_size != 0:
            m_block_size = 64
        return BwdConfig(
            m_block_size=m_block_size,
            n_block_size=128,
            num_stages_Q=2, num_stages_dO=2, num_stages_PdS=2,
            SdP_swapAB=True, dKV_swapAB=False,
            dQ_swapAB=m_block_size % 64 != 0,
            AtomLayoutMSdP=1, AtomLayoutNdKV=2, AtomLayoutMdQ=1,
        )
    elif head_dim <= 192:
        hdimv128 = head_dim_v <= 128
        if hdimv128:
            return BwdConfig(
                m_block_size=64, n_block_size=96,
                num_stages_Q=2, num_stages_dO=2, num_stages_PdS=1,
                SdP_swapAB=False, dKV_swapAB=True, dQ_swapAB=False,
                AtomLayoutMSdP=1, AtomLayoutNdKV=2, AtomLayoutMdQ=1,
                num_wg=2,
            )
        else:
            return BwdConfig(
                m_block_size=64, n_block_size=96,
                num_stages_Q=2, num_stages_dO=1, num_stages_PdS=1,
                SdP_swapAB=False, dKV_swapAB=True, dQ_swapAB=False,
                AtomLayoutMSdP=1, AtomLayoutNdKV=2, AtomLayoutMdQ=1,
                num_wg=2,
            )
    else:
        # hdim 256
        return BwdConfig(
            m_block_size=64, n_block_size=64,
            num_stages_Q=1, num_stages_dO=1, num_stages_PdS=1,
            SdP_swapAB=False, dKV_swapAB=False, dQ_swapAB=False,
            AtomLayoutMSdP=1, AtomLayoutNdKV=1, AtomLayoutMdQ=1,
        )



def _validate_tensor(t, name, expected_shape, expected_dtype, expected_device):
    assert t.shape == expected_shape, f"{name} shape {t.shape} != expected {expected_shape}"
    assert t.dtype == expected_dtype, f"{name} dtype {t.dtype} != expected {expected_dtype}"
    assert t.device == expected_device, f"{name} device {t.device} != expected {expected_device}"
    assert t.is_cuda, f"{name} must be on CUDA"

torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
    torch.float8_e4m3fn: cutlass.Float8E4M3FN,
    torch.float8_e5m2: cutlass.Float8E5M2,
}

_LEARNABLE_SINK_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def num_splits_heuristic(total_mblocks, num_SMs, num_n_blocks, max_splits):
    # 如果 num_n_blocks 太小，就只用 1 个 split。例如 hdim=128 且 seqlen_k=512 时我们从不 split。
    if num_n_blocks <= 4:
        return 1
    # 当 batch_size 或 seqlen_q 为 0 时避免 ZeroDivisionError。_flash_attn_fwd 中的
    # 空 Q 提前退出负责处理这些形状的正确性；这里的防护只是让启发式逻辑在其它
    # 调用场景下也保持安全。
    if total_mblocks == 0:
        return 1

    # 注意：等 split KV 支持持久化（persistence）之后，这个启发式需要重新审视。
    # 有时为了更好的效率，多调度一些 split 反而是理想的。
    return min(num_SMs // total_mblocks, max_splits, num_n_blocks)


def _get_fwd_config(
    *,
    arch: int,
    head_dim: int,
    head_dim_v: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    num_head_kv: int,
    qhead_per_kvhead: int,
    pack_gqa: bool,
    batch_size: int,
    causal: bool,
    local: bool,
    window_size_left: Optional[int],
    window_size_right: Optional[int],
    num_splits: int,
    device,
    seqlen_q: Optional[int] = None,
    tile_mn: Optional[Tuple[int, int]] = None,
    block_sparse_tensors: Optional[BlockSparseTensorsTorch] = None,
    mma_pv_is_rs: Optional[bool] = None,
    intra_wg_overlap: Optional[bool] = None,
) -> FwdConfig:
    if seqlen_q is None:
        seqlen_q = max_seqlen_q

    # 基础 tile 大小与标志位：优先显式覆盖，否则按架构启发式决定。
    cfg = FwdConfig(128, 128, True, True)
    if tile_mn is None:
        if arch // 10 == 12:
            # 针对 99 KB SMEM 容量调优的 SM120 tile 大小：
            # D<=64：128x128 → 48 KB（占用率好）
            # D>64： 128x64  → 64 KB（128x128 会用掉 96 KB，损害占用率）
            if head_dim > 64:
                cfg = FwdConfig(128, 64, True, True)
        elif arch // 10 == 8:
            cfg = FwdConfig(128, 64, True, True)  # SM80，需要调优
        elif arch // 10 == 9:
            sparse_q = get_sparse_q_block_size(block_sparse_tensors, seqlen_q)
            cfg = _tile_size_fwd_sm90(
                head_dim, head_dim_v, causal, local, sparse_block_size_q=sparse_q
            )
    else:
        cfg = FwdConfig(tile_mn[0], tile_mn[1], cfg.mma_pv_is_rs, cfg.intra_wg_overlap)

    tile_m, tile_n = cfg.m_block_size, cfg.n_block_size
    if mma_pv_is_rs is None:
        mma_pv_is_rs = cfg.mma_pv_is_rs
    if intra_wg_overlap is None:
        intra_wg_overlap = cfg.intra_wg_overlap

    seqlen_q_packgqa = max_seqlen_q * (qhead_per_kvhead if pack_gqa else 1)
    if arch // 10 in [10, 11]:
        q_stage = 2 if seqlen_q_packgqa > tile_m else 1
    else:
        q_stage = 1

    m_block_size_effective = q_stage * tile_m
    seqlen_k_loaded = (
        max_seqlen_k
        if not local
        else max(
            0,
            min(
                max_seqlen_k,
                (window_size_right or max_seqlen_k)
                + (window_size_left or max_seqlen_k)
                + 1
                + tile_m,
            ),
        )
    )
    num_m_blocks = (seqlen_q_packgqa + m_block_size_effective - 1) // m_block_size_effective
    total_mblocks = batch_size * num_head_kv * num_m_blocks
    num_n_blocks = (seqlen_k_loaded + tile_n - 1) // tile_n
    num_SMs = None
    if arch // 10 == 12:
        assert num_splits == 1, "SM120 forward only supports num_splits=1"
    elif num_splits < 1:
        num_SMs = get_num_sms_for_selection(device.index, arch)
        num_splits = num_splits_heuristic(total_mblocks, num_SMs, num_n_blocks, 128)

    # SplitKV 使用 float32 的部分输出，会使共享内存中 O 缓冲区的体积翻倍，
    # 对于不同 head 维（diff-headdim，192, 128）会导致 OOM
    if arch // 10 in [10, 11] and head_dim != head_dim_v and num_splits > 1:
        if num_n_blocks >= 64 and head_dim_v != 512:
            tile_n = 64
            num_n_blocks = (seqlen_k_loaded + tile_n - 1) // tile_n
            if num_SMs is None:
                num_SMs = get_num_sms_for_selection(device.index, arch)
            num_splits = num_splits_heuristic(total_mblocks, num_SMs, num_n_blocks, 128)
        else:
            num_splits = 1

    return FwdConfig(tile_m, tile_n, mma_pv_is_rs, intra_wg_overlap, q_stage, num_splits)


def _resolve_causal_local_window(causal, window_size_left, window_size_right, mask_mod=None):
    """把 causal/local/window 设置解析为规范形式。

    返回 (causal, local, window_size_left, window_size_right)。
    """
    if mask_mod is not None:
        return False, False, window_size_left, window_size_right
    if causal:
        window_size_right = 0
    if window_size_left is not None and window_size_right is not None and window_size_left + window_size_right < 0:
        window_size_left = None
        window_size_right = None
    if window_size_left is not None or window_size_right is not None:
        if window_size_left is None and window_size_right == 0:
            causal, local = True, False
            window_size_right = None
        else:
            causal, local = False, True
    else:
        local = False
    return causal, local, window_size_left, window_size_right


def _compute_tile_cumsum(
    *,
    num_m_blocks: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    num_splits_dynamic: Optional[torch.Tensor] = None,
    virtual_batch_idx: Optional[torch.Tensor] = None,
    tile_size: int = 1,
    q_stage: int = 1,
    cluster_shape_m: int = 1,
    qhead_per_kvhead: int = 1,
    pack_gqa: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """返回 (cu_total_m_blocks, cu_total_splits_m_blocks)，均为 int32，形状 (num_batch + 1,)。

    当 num_splits_dynamic 为 None 时，cu_total_splits_m_blocks 为 None。
    """
    assert num_m_blocks is not None or cu_seqlens is not None or seqused is not None, (
        "_compute_tile_cumsum requires num_m_blocks, cu_seqlens, or seqused"
    )
    if num_m_blocks is not None:
        # num_m_blocks 已经是以 tile_size 为单位，直接放进 seqused 槽位传递。
        seqused = num_m_blocks
        tile = q_stage * cluster_shape_m
        seqlen_q_multiplier = 1
    else:
        tile = tile_size * q_stage * cluster_shape_m
        seqlen_q_multiplier = qhead_per_kvhead if pack_gqa and qhead_per_kvhead > 1 else 1
    batch_size = seqused.shape[0] if seqused is not None else cu_seqlens.shape[0] - 1
    device = seqused.device if seqused is not None else cu_seqlens.device
    cu_total_m_blocks = torch.empty(batch_size + 1, dtype=torch.int32, device=device)
    cu_total_splits_m_blocks = (
        torch.empty(batch_size + 1, dtype=torch.int32, device=device)
        if num_splits_dynamic is not None
        else None
    )
    compile_key = (
        tile,
        seqlen_q_multiplier,
        cu_seqlens is not None,
        seqused is not None,
        num_splits_dynamic is not None,
        virtual_batch_idx is not None,
    )
    if compile_key not in _compute_tile_cumsum.compile_cache:
        cute_tensors = [
            to_cute_tensor(t, assumed_align=4, leading_dim=0) if t is not None else None
            for t in (
                cu_total_m_blocks,
                cu_total_splits_m_blocks,
                cu_seqlens,
                seqused,
                num_splits_dynamic,
                virtual_batch_idx,
            )
        ]
        _compute_tile_cumsum.compile_cache[compile_key] = cute.compile(
            CuSeqlensToBlocksKernel(tile=tile, seqlen_q_multiplier=seqlen_q_multiplier),
            *cute_tensors,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    if not is_fake_mode():
        _compute_tile_cumsum.compile_cache[compile_key](
            cu_total_m_blocks,
            cu_total_splits_m_blocks,
            cu_seqlens,
            seqused,
            num_splits_dynamic,
            virtual_batch_idx,
        )
    return cu_total_m_blocks, cu_total_splits_m_blocks


_compute_tile_cumsum.compile_cache = get_jit_cache("tile_cumsum")


def _blocks_to_batch_size(total_q, num_batch, tile_m, qhead_per_kvhead, pack_gqa):
    """给定 varlen 调用中 m_blocks 数量的上界"""
    seqlen_mult = qhead_per_kvhead if pack_gqa and qhead_per_kvhead > 1 else 1
    return (total_q * seqlen_mult + num_batch * (tile_m - 1)) // tile_m + 1


def _compute_blocks_to_batch(cu_total_blocks, num_blocks, device):
    """_compute_tile_cumsum 的反向索引：扁平 scheduler block -> batch，int32，形状 (num_blocks,)。

    超出最后一个 batch 范围的 block 映射到 batch_size（视为无效）。
    """
    blocks_to_batch = torch.empty(num_blocks, dtype=torch.int32, device=device)
    compile_key = ()
    if compile_key not in _compute_blocks_to_batch.compile_cache:
        cu_total_blocks_tensor, blocks_to_batch_tensor = [
            to_cute_tensor(t, assumed_align=4, leading_dim=0)
            for t in (cu_total_blocks, blocks_to_batch)
        ]
        _compute_blocks_to_batch.compile_cache[compile_key] = cute.compile(
            CuBlocksToBatchKernel(),
            cu_total_blocks_tensor,
            blocks_to_batch_tensor,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    if not is_fake_mode():
        _compute_blocks_to_batch.compile_cache[compile_key](cu_total_blocks, blocks_to_batch)
    return blocks_to_batch


_compute_blocks_to_batch.compile_cache = get_jit_cache("blocks_to_batch")


def _flash_attn_fwd(
    q: Optional[torch.Tensor],
    k: Optional[torch.Tensor],
    v: torch.Tensor,
    qv: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    min_seqlen_k: Optional[int] = None,
    page_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    softcap: Optional[float] = None,
    window_size_left: Optional[int] = None,
    window_size_right: Optional[int] = None,
    learnable_sink: Optional[torch.Tensor] = None,
    tile_mn: Optional[Tuple[int, int]] = None,
    mma_pv_is_rs: Optional[bool] = None,
    intra_wg_overlap: Optional[bool] = None,
    num_threads: int = 384,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    _arch: Optional[int] = None,
    score_mod: Optional[Callable] = None,
    mask_mod: Optional[Callable] = None,
    block_sparse_tensors: Optional[BlockSparseTensorsTorch] = None,
    return_lse: bool = False,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    aux_tensors: Optional[list[torch.Tensor]] = None,
    aux_scalars: Optional[tuple] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    gather_kv_indices: Optional[torch.Tensor] = None,
    scheduler_metadata: Optional[SchedulerMetadataTensorsTorch] = None,
    seqlen_k_per_split: Optional[int] = None,
    disable_scheduler_metadata: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """FlashAttention 前向计算。

    这是标准（非变长）注意力前向的核心实现。它接收 Q/K/V（以及可选的 qv），
    在 GPU 上执行精确（exact）注意力，返回输出 out 与 log-sum-exp（lse）。

    Args:
        ...
        score_mod: 一个可调用对象，接收注意力分数并对其施加修改（例如 softcap、
            衰减等）。它在 kernel 编译期被哈希进 compile key。
        mask_mod: 一个可调用对象，接收 token 位置信息并有选择地做 mask
            （例如自定义的 causal 或局部窗口之外的屏蔽）。
        block_sparse_tensors: 用于 block sparsity（块稀疏）的一组张量。
            它允许跳过注意力矩阵中已知为空的块，从而在长序列上节省算力。
        return_lse: 是否返回注意力分数的 log softmax。设为 True 时总是会计算 LSE，
            且返回的 LSE 支持求梯度（可用于后续对 LSE 的反向传播）。
        out: 可选，预分配的输出张量。为 None 时在内部自动分配。
        lse: 可选，预分配的 log-sum-exp 张量。为 None 时在需要时才分配。
        aux_tensors: 某些 score_mod 需要从全局 aux_tensors 中读取数据，
            这是把它们透传到内部 kernel 的机制。
        aux_scalars: 供 score_mod 或 mask_mod 使用的运行时标量捕获。
        注：省略号 ... 表示这里只列出部分关键参数，完整参数列表见对外接口
        flash_attn_func / flash_attn_varlen_func 的签名与 docstring。
    """
    aux_scalars = tuple(aux_scalars) if aux_scalars else None
    requires_grad = any(
        t is not None and t.requires_grad for t in (q, k, v, qv, learnable_sink)
    )
    fake_mode = is_fake_mode()
    q, k, v, qv = [maybe_contiguous(t) for t in (q, k, v, qv)]
    assert q is not None or qv is not None
    assert v is not None
    q_descale, k_descale, v_descale = [
        maybe_contiguous(t, align_bytes=4) for t in (q_descale, k_descale, v_descale)
    ]
    page_table = maybe_contiguous(page_table, align_bytes=4)
    learnable_sink = maybe_contiguous(learnable_sink, align_bytes=4)
    gather_kv_indices = maybe_contiguous(gather_kv_indices, align_bytes=16)
    q_shape = q.shape if q is not None else qv.shape
    num_head, head_dim = q_shape[-2:]
    if cu_seqlens_q is None:
        batch_size, seqlen_q = q_shape[:2]
        total_q = batch_size * seqlen_q
    else:
        batch_size = cu_seqlens_q.shape[0] - 1
        seqlen_q = None
        total_q = q_shape[0]
    if page_table is not None:
        assert cu_seqlens_k is None, "page_table is not supported with cu_seqlens_k"
        assert page_table.dtype == torch.int32, "page_table must be int32"
        assert page_table.stride(-1) == 1, "page_table must be contiguous in the last dimension"
        max_num_pages_per_seq = page_table.shape[1]
        assert page_table.shape == (batch_size, max_num_pages_per_seq)
        num_pages, page_size = v.shape[:2]
        seqlen_k = num_pages * page_size
    else:
        num_pages, page_size = None, None
        seqlen_k = v.shape[-3]
    num_head_kv = v.shape[-2]
    head_dim_v = v.shape[-1]
    if cu_seqlens_k is None:
        if page_table is None:
            assert k is None or k.shape == (batch_size, seqlen_k, num_head_kv, head_dim)
            assert v.shape == (batch_size, seqlen_k, num_head_kv, head_dim_v)
        else:
            assert k is None or k.shape == (num_pages, page_size, num_head_kv, head_dim)
            assert v.shape == (num_pages, page_size, num_head_kv, head_dim_v)
    else:
        assert k is None or k.shape == (seqlen_k, num_head_kv, head_dim)
        assert v.shape == (seqlen_k, num_head_kv, head_dim_v)
        assert cu_seqlens_k.shape == (batch_size + 1,), (
            "cu_seqlens_k must have shape (batch_size + 1,)"
        )

    if cu_seqlens_q is not None:
        assert cu_seqlens_q.shape == (batch_size + 1,), (
            "cu_seqlens_q must have shape (batch_size + 1,)"
        )
    assert seqused_q is None or seqused_q.shape == (batch_size,), (
        "seqused_q must have shape (batch_size,)"
    )
    assert seqused_k is None or seqused_k.shape == (batch_size,), (
        "seqused_k must have shape (batch_size,)"
    )
    assert v.dtype in [torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2], (
        "inputs must be float16, bfloat16, fp8 e4m3fn, or fp8 e5m2"
    )
    
    assert all(t is None or t.dtype == v.dtype for t in (q, k, qv)), (
        "q, k, v, and qv must have the same dtype"
    )

    q_dtype = q.dtype if q is not None else qv.dtype

    for t in [cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k]:
        if t is not None:
            assert t.dtype == torch.int32, (
                "cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k must be int32"
            )
            assert t.stride(0) == 1, (
                "cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k must be contiguous"
            )
    if learnable_sink is not None:
        assert learnable_sink.shape == (num_head,)
        assert learnable_sink.dtype in _LEARNABLE_SINK_DTYPES, (
            "learnable_sink must be float16, bfloat16, or float32"
        )

    if not fake_mode:
        assert all(
            t is None or t.is_cuda
            for t in (
                q,
                k,
                v,
                qv,
                q_descale,
                k_descale,
                v_descale,
                cu_seqlens_q,
                cu_seqlens_k,
                seqused_q,
                seqused_k,
                page_table,
                learnable_sink,
            )
        ), "inputs must be on CUDA device"
    arch = _get_device_arch() if _arch is None else _arch
    assert arch // 10 in [8, 9, 10, 11, 12], "Unsupported compute capability. Supported: 8.x, 9.x, 10.x, 11.x, 12.x"
    assert num_head % num_head_kv == 0, "num_head must be divisible by num_head_kv"
    alignment = 16 // v.element_size()
    if arch // 10 not in [8, 12]:
        _validate_head_dims(head_dim, head_dim_v, arch // 10, alignment)
    if softmax_scale is None:
        softmax_scale = (
            1.0 / math.sqrt(head_dim) if qv is None or q is None
            else 1.0 / math.sqrt(head_dim + head_dim_v)
        )
    if softcap == 0.0:
        softcap = None
    qhead_per_kvhead = num_head // num_head_kv
    if pack_gqa is None:
        pack_gqa = qhead_per_kvhead > 1

    is_fp8 = v.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    if is_fp8 and requires_grad:
        raise NotImplementedError("FA4 CuTe FP8 backward is not supported yet (forward-only).")
    out_torch_dtype = torch.bfloat16 if is_fp8 else q_dtype
    device = v.device
    q_batch_seqlen_shape = (batch_size, seqlen_q) if cu_seqlens_q is None else (total_q,)

    if qv is None:
        lse_shape = (batch_size, num_head, seqlen_q) if cu_seqlens_q is None else (num_head, total_q)
    else:
        # 对 MLA absorbed 中的 MQA 场景，让 num_head 连续更有利
        lse_shape = (batch_size, seqlen_q, num_head) if cu_seqlens_q is None else (total_q, num_head)

    if out is None:
        out = torch.empty(
            *q_batch_seqlen_shape, num_head, head_dim_v, dtype=out_torch_dtype, device=device
        )
    else:
        _validate_tensor(
            out,
            "out",
            (*q_batch_seqlen_shape, num_head, head_dim_v),
            out_torch_dtype,
            device,
        )
        validate_output_layout(out, "out", align_bytes=16)

    if lse is None:
        lse = (
            torch.empty(lse_shape, dtype=torch.float32, device=device)
            if requires_grad or return_lse
            else None
        )
    elif lse is not None:
        _validate_tensor(lse, "lse", lse_shape, torch.float32, device)
        validate_output_layout(lse, "lse", align_bytes=4)

    if seqlen_k == 0 or total_q == 0:
        out.zero_()
        if lse is not None:
            if learnable_sink is None:
                lse.fill_(float("-inf"))
            else:
                assert qv is None
                lse.copy_(
                    learnable_sink[None, :, None]
                    if cu_seqlens_q is None
                    else learnable_sink[:, None]
                )
        return out, lse, None, None

    if is_fp8:
        for t, name in ((q_descale, "q_descale"), (k_descale, "k_descale"), (v_descale, "v_descale")):
            if t is not None:
                _validate_tensor(
                    t,
                    name,
                    (batch_size, num_head_kv),
                    torch.float32,
                    device,
                )
    else:
        assert q_descale is None and k_descale is None and v_descale is None, (
            "q_descale/k_descale/v_descale are only supported for FP8 inputs"
        )

    dtype = torch2cute_dtype_map[q_dtype]
    if is_fp8:
        assert arch // 10 == 10, "FP8 is only supported on SM100 (compute capability 10.x) for FA4 CuTe."
    use_block_sparsity = block_sparse_tensors is not None

    causal, local, window_size_left, window_size_right = _resolve_causal_local_window(
        causal, window_size_left, window_size_right, mask_mod
    )

    requested_use_clc_scheduler = utils._get_use_clc_scheduler_default()
    requested_disable_2cta = utils._get_disable_2cta_default(is_fwd=True)

    # SM80/SM120：使用 SM80 MMA，128 线程（4 warps）
    if arch // 10 in [8, 12]:
        num_threads = 128

    if max_seqlen_q is None:
        max_seqlen_q = seqlen_q if cu_seqlens_q is None else total_q
    if max_seqlen_k is None:
        max_seqlen_k = seqlen_k
    if cu_seqlens_k is None and seqused_k is None:
        min_seqlen_k = seqlen_k

    fwd_cfg = _get_fwd_config(
        arch=arch,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        causal=causal,
        local=local,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        qhead_per_kvhead=qhead_per_kvhead,
        pack_gqa=pack_gqa,
        batch_size=batch_size,
        num_head_kv=num_head_kv,
        num_splits=num_splits,
        device=device,
        seqlen_q=seqlen_q,
        tile_mn=tile_mn,
        block_sparse_tensors=block_sparse_tensors,
        mma_pv_is_rs=mma_pv_is_rs,
        intra_wg_overlap=intra_wg_overlap,
    )
    tile_m, tile_n = fwd_cfg.m_block_size, fwd_cfg.n_block_size
    q_stage = fwd_cfg.q_stage
    num_splits = fwd_cfg.num_splits
    mma_pv_is_rs = fwd_cfg.mma_pv_is_rs
    intra_wg_overlap = fwd_cfg.intra_wg_overlap

    seqlen_q_packgqa = max_seqlen_q * (qhead_per_kvhead if pack_gqa else 1)
    max_m_blocks_leq_one = seqlen_q_packgqa <= q_stage * tile_m

    is_split_kv = num_splits > 1
    if is_split_kv:
        out_partial = torch.empty(num_splits, *q_batch_seqlen_shape, num_head, head_dim_v, dtype=torch.float32, device=device)
        lse_partial = torch.empty(num_splits, *lse_shape, dtype=torch.float32, device=device)

    use_2cta_instrs = (
        arch // 10 in [10, 11]
        and not requested_disable_2cta
        and not causal
        and not local
        and not is_split_kv
        and cu_seqlens_q is None
        and seqused_q is None
        and not use_block_sparsity
        and page_size in [None, 128]
        and int(math.ceil(head_dim / 16) * 16) in [128, 192]
        and int(math.ceil(head_dim_v / 16) * 16) == 128
        and seqlen_q_packgqa > 2 * tile_m
        and (tile_m % qhead_per_kvhead == 0 or not pack_gqa)
    )

    # hd=256 2CTA 前向使用专用 kernel（Blackwell 家族）
    use_dedicated_hd256_kernel = arch // 10 in [10, 11] and head_dim == 256 and head_dim_v == 256
    use_2cta_instrs = use_2cta_instrs or use_dedicated_hd256_kernel

    if softcap is not None:
        assert score_mod is None, "softcap and score_mod cannot be used together"
        score_mod = utils.create_softcap_scoremod(softcap)
    elif score_mod is not None:
        if arch // 10 == 8:
            raise NotImplementedError("Custom user-provided score_mod is not supported on SM8x architectures.")
        
    # 对 score_mod 和 mask_mod 求哈希，用于 compile cache
    score_mod_hash = utils.hash_callable(score_mod) if score_mod is not None else False
    mask_mod_hash = utils.hash_callable(mask_mod) if mask_mod is not None else False

    is_varlen = (
        cu_seqlens_q is not None
        or cu_seqlens_k is not None
        or seqused_q is not None
        or seqused_k is not None
    )

    # CLC（Cluster Launch Control）调度在 varlen MHA 和稠密非 causal 场景下反而退化。
    # 不均衡的 varlen 形状会让更多 K/V block 同时在飞，损害 L2 命中；稠密非 causal
    # 基本只是在为 work-stealing 的开销买单。
    is_varlen_mha = is_varlen and qhead_per_kvhead == 1
    is_dense_noncausal = not is_varlen and not causal and not local
    use_clc_scheduler = requested_use_clc_scheduler and not is_varlen_mha and not is_dense_noncausal

    if use_block_sparsity:
        # 注意：pack_gqa 要求 block sparse 的 head 维 == 1（广播）
        head_dim_idx = 0 if block_sparse_tensors.mask_block_cnt.ndim == 2 else 1
        if pack_gqa and block_sparse_tensors.mask_block_cnt.shape[head_dim_idx] != 1:
            pack_gqa = False
        if cu_seqlens_q is not None:
            assert block_sparse_tensors.cu_total_m_blocks is not None, (
                "Varlen block sparsity requires block_sparse_tensors.cu_total_m_blocks."
            )
            if (
                block_sparse_tensors.cu_block_idx_offsets is None
                and (cu_seqlens_k is not None or seqused_k is not None)
            ):
                raise ValueError(
                    "Varlen block sparsity with cu_seqlens_k or seqused_k requires "
                    "block_sparse_tensors.cu_block_idx_offsets."
                )

    # 为什么 compile key 里需要这个，参见 get_broadcast_dims
    block_sparse_broadcast_pattern = None
    normalized_block_sparse_tensors = None
    q_subtile_factor = 1
    kv_subtile_factor = 1
    if block_sparse_tensors is not None:
        block_sparse_config = normalize_block_sparse_config(
            block_sparse_tensors,
            batch_size=batch_size,
            num_head=num_head,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            block_size=(tile_m, tile_n),
            q_stage=q_stage,
            allow_kv_subtile=arch // 10 in [10, 11],
        )
        normalized_block_sparse_tensors = block_sparse_config.tensors
        block_sparse_broadcast_pattern = block_sparse_config.broadcast_pattern
        q_subtile_factor = block_sparse_config.q_subtile_factor
        kv_subtile_factor = block_sparse_config.kv_subtile_factor
    if aux_tensors is not None:
        aux_tensor_metadata = get_aux_tensor_metadata(aux_tensors)
    else:
        aux_tensor_metadata = None
    aux_scalar_metadata = tuple(type(s) for s in aux_scalars) if aux_scalars is not None else None

    if qv is not None:
        assert arch // 10 in [10, 11], "only support Blackwell arch with qv"
        assert q is None or qv.shape[:-1] == q.shape[:-1]
        assert qv.shape[-1] == head_dim_v
        assert head_dim_v == 512
        assert q is None or head_dim == 64
        assert not local, "local not yet supported with qv"
        assert q_descale is None and k_descale is None and v_descale is None, (
            "q_descale/k_descale/v_descale are not yet supported with qv"
        )
        assert tile_n == 128

        assert not is_split_kv, "split kv not supported with qv"
        assert learnable_sink is None
        assert softcap is None
        assert score_mod is None
        assert mask_mod is None

        if page_table is not None:
            assert gather_kv_indices is None, "paged KV + topk sparsity not yet supported together"
        
        qv = maybe_contiguous(qv)

        gather_kv_length = 2048  # 占位值
        sparse_kv = gather_kv_indices is not None
        # 默认总是使用 kv bitmask（用于处理 -1 哨兵值）
        disable_sparse_kv_bitmask = False
        if sparse_kv:
            assert gather_kv_indices.shape[:-1] == qv.shape[:-2]
            gather_kv_length = gather_kv_indices.shape[-1]
            assert gather_kv_length % 128 == 0
            # if min_seqlen_k is None or causal:
            #     disable_sparse_kv_bitmask = False
            # else:
            #     # seqlen_k_boundary = min_seqlen_k - max_seqlen_q + 1 if causal else min_seqlen_k
            #     seqlen_k_boundary = min_seqlen_k
            #     disable_sparse_kv_bitmask = seqlen_k_boundary >= gather_kv_length
        
        if requires_grad and sparse_kv:
            if cu_seqlens_q is None:
                p = torch.empty(batch_size, seqlen_q, num_head, gather_kv_length, dtype=q_dtype, device=device)
                row_max = torch.empty(batch_size, seqlen_q, gather_kv_length//128, num_head, dtype=torch.float32, device=device)
            else:
                p = torch.empty(total_q, num_head, gather_kv_length, dtype=q_dtype, device=device)
                row_max = torch.empty(total_q, gather_kv_length//128, num_head, dtype=torch.float32, device=device)
        else:
            p = row_max = None
    else:
        assert gather_kv_indices is None, "gather_kv_indices is only supported with qv"
        gather_kv_length = None
        sparse_kv = None
        disable_sparse_kv_bitmask = None
        p = row_max = None


    reuse_scheduler_metadata = scheduler_metadata is not None
    is_varlen_q = cu_seqlens_q is not None or seqused_q is not None
    cluster_shape_m = 2 if use_2cta_instrs else 1
    if use_dedicated_hd256_kernel:
        # hd=256 2CTA 前向 kernel 不支持动态持久化调度器。
        scheduler_metadata = None
        reuse_scheduler_metadata = False
    if (
        is_split_kv
        and is_varlen_q
        and scheduler_metadata is None
        and not disable_scheduler_metadata
        and not use_dedicated_hd256_kernel
    ):
        scheduler_metadata = _get_scheduler_metadata(
            num_batch=batch_size,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            nheads=num_head,
            nheads_kv=num_head_kv,
            headdim=head_dim,
            num_splits=num_splits,
            tile_m=tile_m,
            tile_n=tile_n,
            headdim_v=head_dim_v,
            pack_gqa=pack_gqa,
            causal=causal,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            seqlen_k_per_split=seqlen_k_per_split,
            q_stage=q_stage,
            cluster_shape_m=cluster_shape_m,
            total_q=total_q if cu_seqlens_q is not None else None,
            use_clc_scheduler=use_clc_scheduler,
        )

    has_scheduler_metadata = scheduler_metadata is not None and not disable_scheduler_metadata
    if has_scheduler_metadata:
        num_m_blocks = scheduler_metadata.num_m_blocks_ptr
        num_splits_dynamic = scheduler_metadata.num_splits_dynamic_ptr
        virtual_batch_idx = scheduler_metadata.virtual_batch_idx_ptr
        num_nheads_in_l2 = scheduler_metadata.num_nheads_in_l2_ptr
        tile_count_semaphore = scheduler_metadata.tile_count_semaphore
        assert all(
            t is None or t.is_cuda
            for t in scheduler_metadata
        ), "scheduler metadata must be on CUDA device"
        assert all(
            t is None or t.shape == (batch_size,)
            for t in (
                num_m_blocks,
                num_splits_dynamic,
                virtual_batch_idx,
                num_nheads_in_l2,
            )
        ), "these scheduler metadata tensors must have shape (batch_size,)"
        if tile_count_semaphore is not None:
            assert tile_count_semaphore.shape == (1,), "semaphore must have size 1"
    else:
        num_m_blocks = None
        num_splits_dynamic = None
        virtual_batch_idx = None
        num_nheads_in_l2 = None
        tile_count_semaphore = None

    # 在 SingleTileVarlenScheduler 中使用二分 batch 查找，避免 O(N^2) 的查找；
    # 实测只在 batch_size > BIN_BATCH_SEARCH_THRESH 时才更快；该阈值可调
    cu_total_m_blocks = None
    cu_total_splits_m_blocks = None
    blocks_to_batch_idx = None
    use_single_tile_varlen_scheduler = tile_count_semaphore is None
    use_cu_hint = (
        is_varlen_q
        and use_single_tile_varlen_scheduler
        and batch_size > BIN_BATCH_SEARCH_THRESH
        and not use_dedicated_hd256_kernel
    )
    if (
        use_cu_hint
        and has_scheduler_metadata
        and scheduler_metadata.cu_total_m_blocks is not None
    ):
        cu_total_m_blocks = scheduler_metadata.cu_total_m_blocks
        cu_total_splits_m_blocks = scheduler_metadata.cu_total_splits_m_blocks
        blocks_to_batch_idx = scheduler_metadata.blocks_to_batch_idx
    elif use_cu_hint:
        cu_total_m_blocks, cu_total_splits_m_blocks = _compute_tile_cumsum(
            num_m_blocks=num_m_blocks,
            cu_seqlens=cu_seqlens_q,
            seqused=seqused_q,
            num_splits_dynamic=num_splits_dynamic,
            virtual_batch_idx=virtual_batch_idx,
            tile_size=tile_m,
            q_stage=q_stage,
            cluster_shape_m=cluster_shape_m,
            qhead_per_kvhead=qhead_per_kvhead,
            pack_gqa=pack_gqa,
        )
    if blocks_to_batch_idx is None and USE_BLOCKS_TO_BATCH and cu_total_m_blocks is not None:
        blocks_to_batch_idx = _compute_blocks_to_batch(
            cu_total_m_blocks,
            _blocks_to_batch_size(total_q, batch_size, tile_m, qhead_per_kvhead, pack_gqa),
            cu_total_m_blocks.device,
        )

    # 张量形式的 max_seqlen 值（例如 HF varlen）绝不能泄漏进 compile key：
    # 张量身份每次调用都变，会直接击穿 JIT cache。
    is_static_persistent = (
        not causal
        and not local
        and cu_seqlens_q is None
        and seqused_q is None
        and not is_split_kv
    ) or (
        not torch.is_tensor(max_m_blocks_leq_one)
        and max_m_blocks_leq_one
        and not is_split_kv
    )

    # 在把 layout 标记为动态时，CuTe 会保持 stride-zero 的 mode 为静态。
    tensor_broadcast_patterns = tuple(
        get_broadcast_dims(tensor) if tensor is not None else None
        for tensor in (
            q,
            k,
            v,
            qv,
            page_table,
            q_descale,
            k_descale,
            v_descale,
            gather_kv_indices,
        )
    )

    compile_key = (
        dtype,
        head_dim,
        head_dim_v,
        qhead_per_kvhead,
        causal,
        score_mod_hash,
        mask_mod_hash,
        use_block_sparsity,
        block_sparse_broadcast_pattern,
        tensor_broadcast_patterns,
        aux_tensor_metadata,
        aux_scalar_metadata,
        lse is None,
        cu_seqlens_q is None,
        cu_seqlens_k is None,
        seqused_q is None,
        seqused_k is None,
        page_table is not None,
        window_size_left is not None,
        window_size_right is not None,
        (
            torch2cute_dtype_map[learnable_sink.dtype]
            if learnable_sink is not None
            else None
        ),
        q_descale is not None,
        k_descale is not None,
        v_descale is not None,
        block_sparse_tensors is None or block_sparse_tensors.cu_total_m_blocks is None,
        block_sparse_tensors is None or block_sparse_tensors.cu_block_idx_offsets is None,
        tile_m,
        tile_n,
        q_stage,
        num_threads,
        is_split_kv,
        pack_gqa,
        arch,
        page_size not in [None, tile_n],  # paged KV 走非 TMA 路径
        use_2cta_instrs,
        q_subtile_factor,
        kv_subtile_factor,
        mma_pv_is_rs,
        intra_wg_overlap,
        use_clc_scheduler,
        num_splits_dynamic is not None,
        virtual_batch_idx is not None,
        num_nheads_in_l2 is not None,
        tile_count_semaphore is not None,
        cu_total_m_blocks is not None,
        cu_total_splits_m_blocks is not None,
        blocks_to_batch_idx is not None,
        seqlen_k_per_split,
        is_static_persistent,
        q is not None,
        qv is not None,
        p is not None,
        row_max is not None,
        gather_kv_length,
        sparse_kv,
        disable_sparse_kv_bitmask,
        fa_logging.get_fa_log_level(),
    )

    if compile_key not in _flash_attn_fwd.compile_cache:
        current_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        (
            cu_seqlens_q_tensor,
            cu_seqlens_k_tensor,
            seqused_q_tensor,
            seqused_k_tensor,
            learnable_sink_tensor,
        ) = [
            to_cute_tensor(t, assumed_align=4, leading_dim=0)
            if t is not None
            else None
            for t in (cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, learnable_sink)
        ]
        page_table_tensor = (
            to_cute_tensor(page_table, assumed_align=4, leading_dim=1)
            if page_table is not None
            else None
        )
        q_tensor, k_tensor, v_tensor, o_tensor = [
            to_cute_tensor(t) for t in (q, k, v, out if not is_split_kv else out_partial)
        ]
        if is_split_kv:
            lse_tensor = to_cute_tensor(lse_partial, assumed_align=4)
        else:
            lse_tensor = to_cute_tensor(lse, assumed_align=4)

        q_descale_tensor, k_descale_tensor, v_descale_tensor = (
            to_cute_tensor(t, assumed_align=4, leading_dim=1)
            for t in (q_descale, k_descale, v_descale)
        )
        descale_tensors_tensor = (
            DescaleTensors(
                q_descale=q_descale_tensor,
                k_descale=k_descale_tensor,
                v_descale=v_descale_tensor,
            )
            if q_descale_tensor is not None
            or k_descale_tensor is not None
            or v_descale_tensor is not None
            else None
        )

        sparse_tensors = None
        if normalized_block_sparse_tensors is not None:
            sparse_tensors = to_cute_block_sparse_tensors(normalized_block_sparse_tensors)

        cute_aux_tensors = None
        aux_tensor_metadata = None
        if aux_tensors is not None:
            cute_aux_tensors = [to_cute_aux_tensor(buf) for buf in aux_tensors]

        (
            num_splits_dynamic_tensor,
            tile_count_semaphore_tensor,
            virtual_batch_idx_tensor,
            num_nheads_in_l2_tensor,
            cu_total_m_blocks_tensor,
            cu_total_splits_m_blocks_tensor,
            blocks_to_batch_idx_tensor,
        ) = [
            to_cute_tensor(t, assumed_align=4, leading_dim=0)
            for t in (
                num_splits_dynamic,
                tile_count_semaphore,
                virtual_batch_idx,
                num_nheads_in_l2,
                cu_total_m_blocks,
                cu_total_splits_m_blocks,
                blocks_to_batch_idx,
            )
        ]

        qv_tensor = to_cute_tensor(qv)
        gather_kv_indices_tensor = to_cute_tensor(gather_kv_indices)
        p_tensor = to_cute_tensor(p)
        row_max_tensor = to_cute_tensor(row_max)

        if arch // 10 == 8:
            assert page_table is None, "paged KV not supported on SM 8.0"
            assert not is_split_kv, "SplitKV not supported on SM 8.0"
            fa_fwd = FlashAttentionForwardSm80(
                dtype,
                head_dim,
                head_dim_v,
                qhead_per_kvhead,
                is_causal=causal,
                is_local=local,
                pack_gqa=pack_gqa,
                tile_m=tile_m,
                tile_n=tile_n,
                num_stages=1,
                num_threads=num_threads,
                Q_in_regs=False,
                score_mod=score_mod,
                mask_mod=mask_mod,
                has_aux_tensors=aux_tensors is not None,
            )
        elif arch // 10 == 9:
            assert not is_split_kv, "SplitKV not supported on SM 9.0"
            fa_fwd = FlashAttentionForwardSm90(
                dtype,
                head_dim,
                head_dim_v,
                qhead_per_kvhead,
                is_causal=causal,
                is_local=local,
                pack_gqa=pack_gqa,
                tile_m=tile_m,
                tile_n=tile_n,
                # num_stages=1,
                num_stages=2,
                num_threads=num_threads,
                Q_in_regs=False,
                intra_wg_overlap=intra_wg_overlap,
                mma_pv_is_rs=mma_pv_is_rs,
                mask_mod=mask_mod,
                score_mod=score_mod,
                has_aux_tensors=aux_tensors is not None,
                q_subtile_factor=q_subtile_factor,
                paged_kv_non_tma=page_size not in [None, tile_n],
            )
        elif arch // 10 in [10, 11]:
            if qv is not None:
                paged_kv_cpasync = page_table is not None and page_size != tile_n
                has_qk = q is not None
                fa_fwd = FlashAttentionMLAForwardSm100(
                    is_causal=causal,
                    use_cpasync_load_KV=sparse_kv or paged_kv_cpasync,
                    topk_length=gather_kv_length,
                    is_topk_gather=sparse_kv,
                    pack_gqa=pack_gqa,
                    qhead_per_kvhead=qhead_per_kvhead,
                    nheads_kv=num_head_kv,
                    has_seqused_q=seqused_q is not None,
                    has_cu_seqlens_q=cu_seqlens_q is not None,
                    disable_bitmask=disable_sparse_kv_bitmask,
                    has_qk=has_qk,
                )
            else:
                if use_dedicated_hd256_kernel:
                    # hd=256 2CTA 前向：检查当前不支持的特性
                    assert softcap is None, "SM100 forward with head_dim=256 does not support softcap"
                    assert not use_block_sparsity, \
                        "SM100 forward with head_dim=256 does not support block sparsity"
                    assert learnable_sink is None, \
                        "SM100 forward with head_dim=256 does not support learnable_sink"
                    assert seqused_q is None and seqused_k is None, \
                        "SM100 forward with head_dim=256 does not support seqused_q/seqused_k"
                    if page_table is not None:
                        assert max_seqlen_k % page_size == 0, (
                            f"SM100 hd256 2CTA paged KV requires max_seqlen_k divisible by "
                            f"page_size ({page_size}), got max_seqlen_k={max_seqlen_k}"
                        )
                        assert page_table.shape[1] == max_seqlen_k // page_size, (
                            f"SM100 hd256 2CTA paged KV requires page_table.shape[1] == "
                            f"max_seqlen_k // page_size ({max_seqlen_k} // {page_size} = "
                            f"{max_seqlen_k // page_size}), got {page_table.shape[1]}; "
                            f"pass page_table[:, :{max_seqlen_k // page_size}] to slice to "
                            f"the actual sequence length"
                        )
                    # pack_gqa 是自动选择的优化；对 hd256 kernel 需要禁用它
                    pack_gqa = False

                flash_fwd_obj_cls = (
                    BlackwellFusedMultiHeadAttentionForward
                    if use_dedicated_hd256_kernel
                    else FlashAttentionForwardSm100
                )

                fa_fwd_kwargs = dict(
                    qhead_per_kvhead=qhead_per_kvhead,
                    is_causal=causal,
                    is_local=local,
                    is_split_kv=is_split_kv,
                    pack_gqa=pack_gqa,
                    m_block_size=tile_m,
                    n_block_size=tile_n,
                    q_stage=q_stage,
                    is_static_persistent=is_static_persistent,
                    score_mod=score_mod,
                    mask_mod=mask_mod,
                    has_aux_tensors=aux_tensors is not None,
                    paged_kv_non_tma=page_size not in [None, tile_n],
                    is_varlen_q=cu_seqlens_q is not None or seqused_q is not None,
                    q_subtile_factor=q_subtile_factor,
                    kv_subtile_factor=kv_subtile_factor,
                    use_2cta_instrs=use_2cta_instrs,
                    use_clc_scheduler=use_clc_scheduler,
                    seqlen_k_per_split=seqlen_k_per_split,
                )
                if not use_dedicated_hd256_kernel:
                    fa_fwd_kwargs["has_tile_count_semaphore"] = tile_count_semaphore is not None
                fa_fwd = flash_fwd_obj_cls(head_dim, head_dim_v, **fa_fwd_kwargs)
        elif arch // 10 == 12:
            # SM120（Blackwell GeForce / DGX Spark）：使用 SM80 MMA，拥有 SM120 的 SMEM 容量
            assert not use_block_sparsity, "Block sparsity not supported on SM 12.0"
            assert page_table is None, "Paged KV not supported on SM 12.0 in this PR"
            assert not is_split_kv, "SplitKV not supported on SM 12.0 in this PR"
            fa_fwd = FlashAttentionForwardSm120(
                dtype,
                head_dim,
                head_dim_v,
                qhead_per_kvhead,
                is_causal=causal,
                is_local=local,
                pack_gqa=pack_gqa,
                tile_m=tile_m,
                tile_n=tile_n,
                num_stages=1,
                num_threads=num_threads,
                Q_in_regs=False,
                score_mod=score_mod,
                mask_mod=mask_mod,
                has_aux_tensors=aux_tensors is not None,
            )
        else:
            raise ValueError(
                f"Unsupported compute capability: {arch}. Supported: 8.x, 9.x, 10.x, 11.x, 12.x"
            )
        # TODO: 检查 @can_implement
        if qv is not None:
            _flash_attn_fwd.compile_cache[compile_key] = cute.compile(
                fa_fwd,
                q_tensor,
                qv_tensor,
                k_tensor,
                v_tensor,
                o_tensor,
                lse_tensor,
                softmax_scale,
                p_tensor,
                row_max_tensor,
                cu_seqlens_q_tensor,
                cu_seqlens_k_tensor,
                seqused_q_tensor,
                seqused_k_tensor,
                gather_kv_indices_tensor,
                page_table_tensor,
                window_size_left,
                window_size_right,
                current_stream,
                options="--enable-tvm-ffi",
            )
        else:
            compile_args = [
                fa_fwd,
                q_tensor,
                k_tensor,
                v_tensor,
                o_tensor,
                lse_tensor,
                softmax_scale,
                cu_seqlens_q_tensor,
                cu_seqlens_k_tensor,
                seqused_q_tensor,
                seqused_k_tensor,
                page_table_tensor,
                window_size_left,
                window_size_right,
                learnable_sink_tensor,
            ]
            if arch // 10 in [10, 11]:
                compile_args.append(descale_tensors_tensor)
            compile_args.extend([
                sparse_tensors,
                AuxData(cute_aux_tensors, aux_scalars),
            ])
            if arch // 10 in [10, 11] and not use_dedicated_hd256_kernel:
                compile_args.extend([
                    num_splits_dynamic_tensor,
                    tile_count_semaphore_tensor,
                    virtual_batch_idx_tensor,
                    num_nheads_in_l2_tensor,
                    cu_total_m_blocks_tensor,
                    cu_total_splits_m_blocks_tensor,
                    blocks_to_batch_idx_tensor,
                    max_seqlen_q,
                ])
            elif arch // 10 in [8, 9, 12]:
                compile_args.extend([
                    cu_total_m_blocks_tensor,
                    cu_total_splits_m_blocks_tensor,
                ])
            compile_args.append(current_stream)
            _flash_attn_fwd.compile_cache[compile_key] = cute.compile(*compile_args, options="--enable-tvm-ffi")

    if not fake_mode:
        q_call, k_call, v_call, qv_call = [
            t.detach() if t is not None else None
            for t in (q, k, v, qv)
        ]
        if is_fp8:
            # 在固定到支持 fp8 导出的 torch >= 2.11.0 之前，需要用 uint8 变通
            q_call, k_call, v_call, qv_call = [
                t.view(torch.uint8) if t is not None else None
                for t in (q_call, k_call, v_call, qv_call)
            ]
        descale_tensors = (
            DescaleTensors(q_descale=q_descale, k_descale=k_descale, v_descale=v_descale)
            if q_descale is not None or k_descale is not None or v_descale is not None
            else None
        )
        if qv is not None:
            _flash_attn_fwd.compile_cache[compile_key](
                q_call,
                qv_call,
                k_call,
                v_call,
                out.detach(),
                lse,
                softmax_scale,
                p,
                row_max,
                cu_seqlens_q,
                cu_seqlens_k,
                seqused_q,
                seqused_k,
                gather_kv_indices,
                page_table,
                window_size_left,
                window_size_right,
            )
        else:
            call_args = [
                q_call,
                k_call,
                v_call,
                out.detach() if not is_split_kv else out_partial,
                lse_partial if is_split_kv else lse,
                softmax_scale,
                cu_seqlens_q,
                cu_seqlens_k,
                seqused_q,
                seqused_k,
                page_table,
                window_size_left,
                window_size_right,
                learnable_sink,
            ]
            if arch // 10 in [10, 11]:
                call_args.append(descale_tensors)
            call_args.extend([
                (
                    normalized_block_sparse_tensors.mask_block_cnt,
                    normalized_block_sparse_tensors.mask_block_idx,
                    normalized_block_sparse_tensors.full_block_cnt,
                    normalized_block_sparse_tensors.full_block_idx,
                    normalized_block_sparse_tensors.cu_total_m_blocks,
                    normalized_block_sparse_tensors.cu_block_idx_offsets,
                    normalized_block_sparse_tensors.dq_write_order,
                    normalized_block_sparse_tensors.dq_write_order_full,
                )
                if normalized_block_sparse_tensors is not None
                else None,
                AuxData(aux_tensors, aux_scalars),
            ])
            if arch // 10 in [10, 11] and not use_dedicated_hd256_kernel:
                call_args.extend([
                    num_splits_dynamic,
                    tile_count_semaphore,
                    virtual_batch_idx,
                    num_nheads_in_l2,
                    cu_total_m_blocks,
                    cu_total_splits_m_blocks,
                    blocks_to_batch_idx,
                    max_seqlen_q,
                ])
            elif arch // 10 in [8, 9, 12]:
                call_args.extend([
                    cu_total_m_blocks,
                    cu_total_splits_m_blocks,
                ])
            _flash_attn_fwd.compile_cache[compile_key](*call_args)
    if is_split_kv:
        _flash_attn_fwd_combine(
            out_partial,
            lse_partial.transpose(-1, -2),
            out,
            lse.transpose(-1, -2) if lse is not None else None,
            cu_seqlens_q,
            seqused_q,
            num_splits_dynamic_ptr=num_splits_dynamic if has_scheduler_metadata else None,
            virtual_batch_idx=virtual_batch_idx if has_scheduler_metadata else None,
            _arch=arch,
        )
    if reuse_scheduler_metadata and tile_count_semaphore is not None:
        # TODO: 把 tile_count_semaphore 传给 combine kernel 并在其中清零（is_split_kv 时
        # 用 CTA 0，因为后面的 CTA 可能提前退出），这样主机侧的清零只在
        # is_split_kv=False 时才需要。
        tile_count_semaphore.zero_()
    return out, lse, p, row_max


_flash_attn_fwd.compile_cache = get_jit_cache("fwd")


def make_fake_bwd_tensors(dtype, has_gqa, varlen_q, varlen_k, nheads_major=False):
    sym = cute.sym_int
    # 元素个数形式的整除性：assumed_align_bytes = divisibility * dtype.width // 8
    # 对于 16 字节对齐：fp16/bf16 → divisibility=8，float32 → divisibility=4
    div = 128 // dtype.width  # fp16/bf16 时为 8
    # 为跨张量必须一致的维度创建共享的 sym_ints
    b, seqlen_q, seqlen_k, h_q, d, d_v = sym(), sym(), sym(), sym(), sym(), sym()
    topk = sym()
    h_kv = h_q if not has_gqa else sym()
    seqlen_q_rounded, seqlen_k_rounded = sym(), sym()
    seqlen_q_d_rounded, seqlen_k_d_rounded, seqlen_k_dv_rounded = sym(), sym(), sym()
    total_q, total_k, total_q_rounded, total_k_rounded = sym(), sym(), sym(), sym()
    total_q_d_rounded, total_k_d_rounded, total_k_dv_rounded = sym(), sym(), sym()
    b_seqlenq = (b, seqlen_q) if not varlen_q else (total_q,)
    b_seqlenk = (b, seqlen_k) if not varlen_k else (total_k,)
    mQ = fake_tensor(dtype, (*b_seqlenq, h_q, d), divisibility=div)
    mO = fake_tensor(dtype, (*b_seqlenq, h_q, d_v), divisibility=div)
    mdO = fake_tensor(dtype, (*b_seqlenq, h_q, d_v), divisibility=div)
    mK = fake_tensor(dtype, (*b_seqlenk, h_kv, d), divisibility=div)
    mV = fake_tensor(dtype, (*b_seqlenk, h_kv, d_v), divisibility=div)
    mdQ = fake_tensor(dtype, (*b_seqlenq, h_q, d), divisibility=div)
    mdK = fake_tensor(dtype, (*b_seqlenk, h_kv, d), divisibility=div)
    mdV = fake_tensor(dtype, (*b_seqlenk, h_kv, d_v), divisibility=div)

    sq    = seqlen_q         if not varlen_q else total_q
    sq_r  = seqlen_q_rounded if not varlen_q else total_q_rounded
    sq_dr = seqlen_q_d_rounded if not varlen_q else total_q_d_rounded

    def shape(*dims):
        batch = (b,) if not varlen_q else ()
        return (*batch, h_q, *dims) if not nheads_major else (*batch, *dims, h_q)

    mLSE     = fake_tensor(Float32, shape(sq),       divisibility=1)
    mLSElog2 = fake_tensor(Float32, shape(sq_r),     divisibility=4)
    mPdPsum  = fake_tensor(Float32, shape(sq_r),     divisibility=4)
    dQaccum  = fake_tensor(Float32, shape(sq_dr),    divisibility=4)
    mScaleP  = fake_tensor(Float32, shape(sq, topk), divisibility=4)

    if not has_gqa:
        mdKaccum, mdVaccum = None, None
    else:
        if not varlen_k:
            mdKaccum = fake_tensor(Float32, (b, h_kv, seqlen_k_rounded), divisibility=4)
            mdVaccum = fake_tensor(Float32, (b, h_kv, seqlen_k_dv_rounded), divisibility=4)
        else:
            mdKaccum = fake_tensor(Float32, (h_kv, total_k_rounded), divisibility=4)
            mdVaccum = fake_tensor(Float32, (h_kv, total_k_dv_rounded), divisibility=4)
    return mQ, mK, mV, mO, mdO, mdQ, mdK, mdV, mLSE, mLSElog2, mPdPsum, dQaccum, mdKaccum, mdVaccum, mScaleP


def _compile_bwd_preprocess(
    dtype,
    head_dim,
    head_dim_v,
    m_block_size,
    has_cuseqlens_q,
    has_seqused_q,
    has_dlse,
    has_dq_accum,
    has_scaleP,
    use_padded_offsets,
    nheads_major,
    pack_gqa,
    qhead_per_kvhead,
    nheads_kv,
    has_cu_total_m_blocks,
):
    """使用 cute fake tensors 编译反向预处理 kernel（无需真实 GPU 张量）。"""
    mQ, mK, mV, mO, mdO, mdQ, mdK, mdV, mLSE, mLSElog2, mPdPsum, mdQaccum, mdKaccum, mdVaccum, mScaleP = make_fake_bwd_tensors(
        dtype, has_gqa=True, varlen_q=has_cuseqlens_q, varlen_k=False, nheads_major=nheads_major,
    )
    batch = mQ.shape[0] if not has_cuseqlens_q else cute.sym_int()
    batchp1 = cute.sym_int()
    mCuSeqlensQ = fake_tensor(Int32, (batchp1,), divisibility=1) if has_cuseqlens_q else None
    mSequsedQ = fake_tensor(Int32, (batch,), divisibility=1) if has_seqused_q else None
    mdLSE = fake_tensor(Float32, mLSE.shape, divisibility=1) if has_dlse else None
    mLSElog2 = None if has_scaleP else mLSElog2
    mdQaccum = mdQaccum if has_dq_accum else None
    mRowMax = fake_tensor(Float32, mScaleP.shape, divisibility=1) if has_scaleP else None
    mScaleP = fake_tensor(Float32, mScaleP.shape, divisibility=1) if has_scaleP else None
    softmax_scale = Float32(1.0)
    mCuTotalMBlocks = fake_tensor(Int32, (batchp1,), divisibility=1) if has_cu_total_m_blocks else None
    fa_bwd_pre = FlashAttentionBackwardPreprocess(
        dtype, head_dim, head_dim_v, m_block_size,
        use_padded_offsets=use_padded_offsets,
        nheads_major=nheads_major,
        pack_gqa=pack_gqa,
        qhead_per_kvhead=qhead_per_kvhead,
        nheads_kv=nheads_kv,
    )
    return cute.compile(
        fa_bwd_pre, mO, mdO, mPdPsum, mLSE, mLSElog2, mdQaccum, mCuSeqlensQ, mSequsedQ, mdLSE,
        mRowMax, mScaleP, softmax_scale, mCuTotalMBlocks,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _bwd_preprocess(
    out, dout, dpsum, lse, lse_log2, dq_accum,
    cu_seqlens_q, seqused_q, dlse,
    dtype, head_dim, head_dim_v, m_block_size,
    row_max=None,
    scale_p=None,
    use_padded_offsets=True,
    nheads_major=False,
    pack_gqa=False,
    qhead_per_kvhead=1,  # 仅 pack_gqa 时使用
    nheads_kv=1,         # 仅 pack_gqa 时使用
    softmax_scale=1.0,   # 仅 scale_p 时使用
    cu_total_m_blocks=None,
    *,
    fake_mode,
):
    """反向预处理：计算 (o * dout).sum(dim=-1) - dLSE、lse * log2_e，并将 dq_accum 清零。"""
    if row_max is not None:
        assert scale_p is not None
    is_varlen = cu_seqlens_q is not None or seqused_q is not None
    if is_varlen:
        batch_size = (cu_seqlens_q.shape[0] - 1) if cu_seqlens_q is not None else seqused_q.shape[0]
    else:
        batch_size = 0
    if cu_total_m_blocks is None and is_varlen and batch_size > BIN_BATCH_SEARCH_THRESH:
        cu_total_m_blocks, _ = _compute_tile_cumsum(
            cu_seqlens=cu_seqlens_q,
            seqused=seqused_q,
            tile_size=m_block_size,
            qhead_per_kvhead=qhead_per_kvhead,
            pack_gqa=pack_gqa,
        )
    compile_key = (
        dtype, head_dim, head_dim_v, m_block_size,
        cu_seqlens_q is not None,
        seqused_q is not None,
        dlse is not None,
        dq_accum is not None,
        row_max is not None,
        use_padded_offsets,
        nheads_major,
        pack_gqa,
        qhead_per_kvhead,
        nheads_kv,
        cu_total_m_blocks is not None,
    )
    if compile_key not in _bwd_preprocess.compile_cache:
        _bwd_preprocess.compile_cache[compile_key] = _compile_bwd_preprocess(*compile_key)
    if not fake_mode:
        _bwd_preprocess.compile_cache[compile_key](
            out, dout, dpsum, lse, lse_log2, dq_accum, cu_seqlens_q, seqused_q, dlse,
            row_max, scale_p, softmax_scale, cu_total_m_blocks,
        )


_bwd_preprocess.compile_cache = get_jit_cache("bwd_pre")


def _compile_bwd_postprocess(
    dtype, hdim, block_size, num_threads, atom_layout, swap_ab,
    has_cuseqlens_q, has_seqused_q,
    use_2cta_instrs, cluster_size, arch,
    has_cu_total_m_blocks,
    learnable_sink_dtype,
):
    """使用 cute fake tensors 编译反向后处理 kernel。"""
    mQ, mK, mV, mO, mdO, mdQ, mdK, mdV, mLSE, mLSElog2, mPdPsum, mdQaccum, mdKaccum, mdVaccum, mScaleP = make_fake_bwd_tensors(
        dtype, has_gqa=True, varlen_q=has_cuseqlens_q, varlen_k=False
    )
    batch = mQ.shape[0] if not has_cuseqlens_q else cute.sym_int()
    batchp1 = cute.sym_int()
    mCuSeqlensQ = fake_tensor(Int32, (batchp1,), divisibility=1) if has_cuseqlens_q else None
    mSeqUsedQ = fake_tensor(Int32, (batch,), divisibility=1) if has_seqused_q else None
    mCuTotalMBlocks = fake_tensor(Int32, (batchp1,), divisibility=1) if has_cu_total_m_blocks else None
    sink_tensors = (
        LearnableSinkBwdTensors(
            mPdPsum,
            mLSE,
            fake_tensor(learnable_sink_dtype, (mQ.shape[-2],), divisibility=1),
            fake_tensor(learnable_sink_dtype, (mQ.shape[-2],), divisibility=1),
        )
        if learnable_sink_dtype is not None
        else None
    )
    fa_bwd_post = FlashAttentionBackwardPostprocess(
        dtype, hdim, arch, block_size, num_threads, atom_layout, swap_ab,
        use_2cta_instrs=use_2cta_instrs,
        cluster_size=cluster_size,
    )
    return cute.compile(
        fa_bwd_post, mdQaccum, mdQ, Float32(0.0), mCuSeqlensQ, mSeqUsedQ,
        sink_tensors,
        mCuTotalMBlocks,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _bwd_postprocess_convert(
    accum, output, scale,
    cu_seqlens, seqused,
    arch, dtype, hdim, block_size, num_threads,
    atom_layout, swap_ab,
    use_2cta_instrs=False, cluster_size=1,
    cu_total_m_blocks=None,
    sink_tensors=None,
    *,
    fake_mode,
):
    """反向后处理：把 float32 累加器转换为 bf16/fp16 输出。"""
    is_varlen = cu_seqlens is not None or seqused is not None
    if is_varlen:
        batch_size = (cu_seqlens.shape[0] - 1) if cu_seqlens is not None else seqused.shape[0]
    else:
        batch_size = 0
    if cu_total_m_blocks is None and is_varlen and batch_size > BIN_BATCH_SEARCH_THRESH:
        cu_total_m_blocks, _ = _compute_tile_cumsum(
            cu_seqlens=cu_seqlens,
            seqused=seqused,
            tile_size=block_size,
        )
    compile_key = (
        dtype, hdim, block_size, num_threads, atom_layout, swap_ab,
        cu_seqlens is not None, seqused is not None,
        use_2cta_instrs, cluster_size, arch,
        cu_total_m_blocks is not None,
        (
            torch2cute_dtype_map[sink_tensors.sink.dtype]
            if sink_tensors is not None
            else None
        ),
    )
    if compile_key not in _bwd_postprocess_convert.compile_cache:
        _bwd_postprocess_convert.compile_cache[compile_key] = _compile_bwd_postprocess(*compile_key)
    if not fake_mode:
        _bwd_postprocess_convert.compile_cache[compile_key](
            accum, output, scale, cu_seqlens, seqused,
            sink_tensors,
            cu_total_m_blocks,
        )


_bwd_postprocess_convert.compile_cache = get_jit_cache("bwd_post")


def _flash_attn_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    softcap: float = 0.0,
    window_size_left: Optional[int] = None,
    window_size_right: Optional[int] = None,
    m_block_size: int = 64,
    n_block_size: int = 128,
    num_threads: int = 256,
    pack_gqa: bool = False,
    num_stages_Q: int = 2,
    num_stages_dO: int = 2,
    SdP_swapAB: bool = False,
    dKV_swapAB: bool = False,
    dQ_swapAB: bool = False,
    AtomLayoutMSdP: int = 2,
    AtomLayoutNdKV: int = 2,
    AtomLayoutMdQ: int = 2,
    V_in_regs: bool = False,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    deterministic: bool = False,
    dq: Optional[torch.Tensor] = None,
    dk: Optional[torch.Tensor] = None,
    dv: Optional[torch.Tensor] = None,
    score_mod: Optional[Callable] = None,
    score_mod_bwd: Optional[Callable] = None,
    mask_mod: Optional[Callable] = None,
    aux_tensors: Optional[list[torch.Tensor]] = None,
    aux_scalars: Optional[tuple] = None,
    block_sparse_tensors: Optional[BlockSparseTensorsTorch] = None,
    dlse: Optional[torch.Tensor] = None,
    learnable_sink: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, ...]:
    aux_scalars = tuple(aux_scalars) if aux_scalars else None
    fake_mode = is_fake_mode()
    arch = _get_device_arch()
    assert arch // 10 in [9, 10, 11, 12], "Unsupported compute capability. Supported: 9.x, 10.x, 11.x, 12.x"
    if block_sparse_tensors is not None:
        assert (
            cu_seqlens_q is None
            and cu_seqlens_k is None
            and seqused_q is None
            and seqused_k is None
        ), "Varlen backward with block sparsity is not yet supported"
    if learnable_sink is not None:
        assert arch // 10 in [9, 10, 11], "Learnable sink backward is supported on SM90 and SM100/SM110"
        assert lse is not None, "learnable_sink backward requires LSE"
        if q.numel() == 0 or k.numel() == 0:
            dq = torch.zeros_like(q) if dq is None else dq.zero_()
            dk = torch.zeros_like(k) if dk is None else dk.zero_()
            dv = torch.zeros_like(v) if dv is None else dv.zero_()
            dsink = (
                dlse.sum(dim=(0, 2) if dlse.ndim == 3 else 1).to(learnable_sink.dtype)
                if dlse is not None
                else torch.zeros_like(learnable_sink)
            )
            return dq, dk, dv, dsink
    sparse_q = None
    kv_subtile_factor = 1
    if block_sparse_tensors is not None:
        if block_sparse_tensors.block_size is not None:
            sparse_q = block_sparse_tensors.block_size[0]
        elif arch // 10 == 9:
            sparse_q = 128

    num_head, head_dim = q.shape[-2:]
    head_dim_v = v.shape[-1]

    window_size = [window_size_left, window_size_right]
    causal, local, window_size_left, window_size_right = _resolve_causal_local_window(
        causal, window_size_left, window_size_right
    )

    if arch // 10 == 12:
        # SM120：使用 SM80 MMA，99 KB SMEM，128 线程（4 warps）。
        m_block_size = 64
        n_block_size = 64
        if head_dim <= 64:
            num_stages_Q = 2
            num_stages_dO = 2
        else:
            num_stages_Q = 1
            num_stages_dO = 1
        SdP_swapAB = False
        dKV_swapAB = False
        dQ_swapAB = False
        AtomLayoutMSdP = 4
        AtomLayoutNdKV = 4
        AtomLayoutMdQ = 4
        V_in_regs = False
        dQ_single_wg = False
        cluster_size = 1
        use_2cta_instrs = False
        num_threads = 128
        assert not (block_sparse_tensors is not None), "Block sparsity backward not supported on SM 12.0"
        assert score_mod is None and score_mod_bwd is None, "score_mod backward not supported on SM 12.0"
        assert mask_mod is None, "mask_mod backward not supported on SM 12.0"
        assert deterministic is False, "deterministic backward not supported on SM 12.0"
    elif arch // 10 == 9:
        cfg = _tile_size_bwd_sm90(
            head_dim,
            head_dim_v,
            causal,
            local,
            sparse_block_size_q=sparse_q,
        )
        m_block_size = cfg.m_block_size
        n_block_size = cfg.n_block_size
        num_stages_Q = cfg.num_stages_Q
        num_stages_dO = cfg.num_stages_dO
        num_stages_PdS = cfg.num_stages_PdS
        SdP_swapAB = cfg.SdP_swapAB
        dKV_swapAB = cfg.dKV_swapAB
        dQ_swapAB = cfg.dQ_swapAB
        AtomLayoutMSdP = cfg.AtomLayoutMSdP
        AtomLayoutNdKV = cfg.AtomLayoutNdKV
        AtomLayoutMdQ = cfg.AtomLayoutMdQ
        num_threads = (cfg.num_wg + 1) * 128
        dQ_single_wg = cfg.dQ_single_wg
        cluster_size = 1
        use_2cta_instrs = False
    else:
        m_block_size = 128
        n_block_size = 128
        dQ_swapAB = False
        dKV_swapAB = False
        AtomLayoutMdQ = 1
        AtomLayoutNdKV = 1
        requested_disable_2cta = utils._get_disable_2cta_default()
        kv_subtile_factor = get_kv_subtile_factor(block_sparse_tensors, n_block_size)
        use_2cta_instrs = (
            head_dim >= 128
            and not requested_disable_2cta
            and block_sparse_bwd_supports_2cta(block_sparse_tensors, n_block_size)
        )
        if block_sparse_tensors is not None and head_dim == 192 and not use_2cta_instrs:
            reason = (
                "2CTA was disabled by request"
                if requested_disable_2cta
                else (
                    f"sparse_block_size[1] must cover an even number of tile_n={n_block_size} "
                    f"tiles; got factor {kv_subtile_factor}"
                )
            )
            raise ValueError(
                f"SM100 block-sparse backward with head_dim=192 requires 2CTA; {reason}."
            )
        cluster_size = 2 if use_2cta_instrs else 1

    use_dedicated_hd256_kernel = arch // 10 in [10, 11] and head_dim == 256 and head_dim_v == 256
    if use_dedicated_hd256_kernel:
        assert learnable_sink is None, (
            "SM100 backward with head_dim=256 does not support learnable_sink"
        )
    use_2cta_instrs = use_2cta_instrs or use_dedicated_hd256_kernel
    is_varlen = (
        cu_seqlens_q is not None
        or cu_seqlens_k is not None
        or seqused_q is not None
        or seqused_k is not None
    )

    q, k, v, out, dout, lse, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, learnable_sink = [
        maybe_contiguous(t)
        for t in (
            q,
            k,
            v,
            out,
            dout,
            lse,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            learnable_sink,
        )
    ]
    if cu_seqlens_q is None:
        batch_size, seqlen_q = q.shape[:2]
        total_q = batch_size * seqlen_q
    else:
        batch_size = cu_seqlens_q.shape[0] - 1
        total_q = q.shape[0]
        seqlen_q = max_seqlen_q if max_seqlen_q is not None else total_q

    if cu_seqlens_k is None:
        batch_size, seqlen_k = k.shape[:2]
        total_k = batch_size * seqlen_k
    else:
        batch_size = cu_seqlens_k.shape[0] - 1
        total_k = k.shape[0]
        seqlen_k = max_seqlen_k if max_seqlen_k is not None else total_k

    num_head_kv = k.shape[-2]

    use_block_sparsity = block_sparse_tensors is not None
    if sparse_q is not None and (sparse_q <= 0 or sparse_q % m_block_size != 0):
        raise ValueError(
            "Block sparsity requires sparse_block_size[0] to be a multiple of "
            f"tile_m={m_block_size}; got {sparse_q}."
        )
    q_subtile_factor = sparse_q // m_block_size if sparse_q is not None else 2
    seqlen_q_rounded = (seqlen_q + m_block_size - 1) // m_block_size * m_block_size
    seqlen_k_rounded = (seqlen_k + n_block_size - 1) // n_block_size * n_block_size
    num_n_blocks = seqlen_k_rounded // n_block_size
    if cluster_size == 2 and num_n_blocks % cluster_size != 0:
        seqlen_k_rounded = seqlen_k_rounded + n_block_size

    # 下面的单 block 特化只是为了防御 TVM stride 污染（stride poisoning）——
    # 它是选择 kernel 变体的主机侧分支谓词。当 max_seqlen 以张量传入（例如 HF/TE varlen）时，
    # seqlen_*_rounded 也是张量，此时 `seqlen_*_rounded // block == 1` 会把张量泄漏进
    # compile key。张量的 pickle 哈希每次调用都不同，会导致每步都重新编译。
    # 因此只有当 seqlen 已经是主机侧标量时才特化；张量调用方回退到多 block 默认路径，
    # 从而在无需设备同步的情况下保持 key 稳定。
    single_q_block = (not torch.is_tensor(seqlen_q_rounded)) and (seqlen_q_rounded // m_block_size == 1)
    single_k_block = (not torch.is_tensor(seqlen_k_rounded)) and (seqlen_k_rounded // n_block_size == 1)

    if cu_seqlens_k is None:
        assert k.shape == (batch_size, seqlen_k, num_head_kv, head_dim)
        assert v.shape == (batch_size, seqlen_k, num_head_kv, head_dim_v)
    else:
        assert k.shape == (total_k, num_head_kv, head_dim)
        assert v.shape == (total_k, num_head_kv, head_dim_v)
        assert cu_seqlens_k.shape == (batch_size + 1,), (
            "cu_seqlens_k must have shape (batch_size + 1,)"
        )

    if cu_seqlens_q is not None:
        assert cu_seqlens_q.shape == (batch_size + 1,), (
            "cu_seqlens_q must have shape (batch_size + 1,)"
        )

        assert out.shape == (total_q, num_head, head_dim_v)
        assert dout.shape == (total_q, num_head, head_dim_v)
        assert lse.shape == (num_head, total_q), "lse must have shape (num_head, total_q)"
    else:
        assert out.shape == (batch_size, seqlen_q, num_head, head_dim_v)
        assert dout.shape == (batch_size, seqlen_q, num_head, head_dim_v)
        assert lse.shape == (batch_size, num_head, seqlen_q), (
            "lse must have shape (batch_size, num_head, seqlen_q)"
        )

    assert q.dtype in [torch.float16, torch.bfloat16], "inputs must be float16 or bfloat16"
    assert q.dtype == k.dtype == v.dtype == out.dtype == dout.dtype, (
        "inputs must have the same dtype"
    )
    for t in [cu_seqlens_q, cu_seqlens_k]:
        if t is not None:
            assert t.dtype == torch.int32, "cu_seqlens_q, cu_seqlens_k must be int32"
    assert lse.dtype == torch.float32, "lse must be float32"
    if dlse is not None:
        dlse = maybe_contiguous(dlse)
    if learnable_sink is not None:
        assert learnable_sink.shape == (num_head,)
        assert learnable_sink.dtype in _LEARNABLE_SINK_DTYPES, (
            "learnable_sink must be float16, bfloat16, or float32"
        )
    if not fake_mode:
        assert all(
            t is None or t.is_cuda
            for t in (q, k, v, out, dout, lse, cu_seqlens_q, cu_seqlens_k, learnable_sink)
        ), "inputs must be on CUDA device"
    assert num_head % num_head_kv == 0, "num_head must be divisible by num_head_kv"
    alignment = 16 // q.element_size()
    if arch // 10 != 12:
        _validate_head_dims(head_dim, head_dim_v, arch // 10, alignment)
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    qhead_per_kvhead = num_head // num_head_kv
    if pack_gqa is None:
        pack_gqa = qhead_per_kvhead > 1
    # pack_gqa 反向（bwd）尚未支持
    pack_gqa = False
    
    if softcap != 0.0:
        assert score_mod is None and score_mod_bwd is None, (
            "softcap and score_mod/score_mod_bwd cannot be used together"
        )
        score_mod = utils.create_softcap_scoremod(softcap)
        score_mod_bwd = utils.create_softcap_scoremod_bwd(softcap)
    if score_mod is not None:
        assert score_mod_bwd is not None, "score_mod_bwd is required when score_mod is provided"
        if arch // 10 == 8:
            raise NotImplementedError("Custom user-provided score_mod is not supported on SM8x architectures.")

    device = q.device
    out_torch_dtype = q.dtype

    if dq is None:
        dq = torch.empty_like(q)
    else:
        _validate_tensor(dq, "dq", q.shape, out_torch_dtype, device)

    if dk is None:
        dk = torch.empty_like(k)
    else:
        _validate_tensor(dk, "dk", k.shape, out_torch_dtype, device)

    if dv is None:
        dv = torch.empty_like(v)
    else:
        _validate_tensor(dv, "dv", v.shape, out_torch_dtype, device)

    head_dim_rounded = (head_dim + 32 - 1) // 32 * 32

    if cu_seqlens_q is None:
        dq_accum = (
            None
            if use_dedicated_hd256_kernel
            else torch.empty(
                batch_size,
                num_head,
                seqlen_q_rounded * head_dim_rounded,
                dtype=torch.float32,
                device=device,
            )
        )
        dpsum = torch.empty(
            batch_size, num_head, seqlen_q_rounded, dtype=torch.float32, device=device
        )
        lse_log2 = torch.empty(
            batch_size, num_head, seqlen_q_rounded, dtype=torch.float32, device=device
        )
    else:
        total_q_rounded_padded = (
            (total_q + cu_seqlens_q.shape[0] * m_block_size - 1) // m_block_size * m_block_size
        )
        dq_accum = (
            None
            if use_dedicated_hd256_kernel
            else torch.empty(
                num_head, total_q_rounded_padded * head_dim_rounded, dtype=torch.float32, device=device
            )
        )
        dpsum = torch.empty(num_head, total_q_rounded_padded, dtype=torch.float32, device=device)
        lse_log2 = torch.empty(num_head, total_q_rounded_padded, dtype=torch.float32, device=device)

    # GQA（qhead_per_kvhead > 1）需要 dK/dV 的累加+后处理，因为多个 Q head
    # 会累加到同一个 dK/dV。SM90 上 qhead_per_kvhead==1 的 varlen_k 现在用
    # ragged TMA 张量直接存储，不再需要累加+后处理。
    # hd=256 2CTA 反向对 dK/dV 有自己内部的后处理。
    dKV_postprocess = qhead_per_kvhead > 1 and not use_dedicated_hd256_kernel
    if dKV_postprocess:
        head_dim_v_rounded = (head_dim_v + 32 - 1) // 32 * 32
        if cu_seqlens_k is None:
            dk_accum = torch.zeros(
                batch_size,
                num_head_kv,
                seqlen_k_rounded * head_dim_rounded,
                dtype=torch.float32,
                device=device,
            )
            dv_accum = torch.zeros(
                batch_size,
                num_head_kv,
                seqlen_k_rounded * head_dim_v_rounded,
                dtype=torch.float32,
                device=device,
            )
        else:
            cluster_tile_n = cluster_size * n_block_size
            total_k_rounded_padded = (
                (total_k + cu_seqlens_k.shape[0] * cluster_tile_n - 1) // cluster_tile_n * cluster_tile_n
            )
            dk_accum = torch.zeros(
                num_head_kv,
                total_k_rounded_padded * head_dim_rounded,
                dtype=torch.float32,
                device=device,
            )
            dv_accum = torch.zeros(
                num_head_kv,
                total_k_rounded_padded * head_dim_v_rounded,
                dtype=torch.float32,
                device=device,
            )

    dtype = torch2cute_dtype_map[q.dtype]

    if deterministic:
        dQ_semaphore = torch.zeros(batch_size, num_head, seqlen_q_rounded // m_block_size, cluster_size, dtype=torch.int32, device=device)
    else:
        dQ_semaphore = None

    if deterministic and qhead_per_kvhead > 1:
        dK_semaphore = torch.zeros(batch_size, num_head_kv, seqlen_k_rounded // n_block_size, 2, dtype=torch.int32, device=device)
        dV_semaphore = torch.zeros(batch_size, num_head_kv, seqlen_k_rounded // n_block_size, 2, dtype=torch.int32, device=device)
    else:
        dK_semaphore = None
        dV_semaphore = None

    # SingleTileVarlenScheduler 的 batch 查找辅助（batch_size 高于 BIN_BATCH_SEARCH_THRESH 时）；
    # 在预处理、主反向以及三次后处理调用之间共享。
    cu_total_m_blocks_q = None
    cu_total_m_blocks_k = None
    if is_varlen and batch_size > BIN_BATCH_SEARCH_THRESH and not use_dedicated_hd256_kernel:
        cu_total_m_blocks_q, _ = _compute_tile_cumsum(
            cu_seqlens=cu_seqlens_q,
            seqused=seqused_q,
            tile_size=m_block_size,
        )
        cu_total_m_blocks_k, _ = _compute_tile_cumsum(
            cu_seqlens=cu_seqlens_k,
            seqused=seqused_k,
            tile_size=n_block_size,
            cluster_shape_m=cluster_size,
        )

    dsink = torch.empty_like(learnable_sink) if learnable_sink is not None else None

    # 预处理 kernel：计算 (o * dout).sum(dim=-1) - dLSE、lse * log2_e，并把 dq_accum 清零。
    # 对 hd=256 专用路径，dq_accum 为 None，预处理只填充 dpsum/lse_log2。
    _bwd_preprocess(
        out, dout, dpsum, lse, lse_log2, dq_accum,
        cu_seqlens_q, seqused_q, dlse,
        dtype, head_dim, head_dim_v, m_block_size,
        cu_total_m_blocks=cu_total_m_blocks_q,
        fake_mode=fake_mode,
    )
    # num_threads：SM90 由 BwdConfig.num_wg 推导，SM120 上面已设为 128，
    # SM100/SM110 使用函数签名里的默认值（384）。
    if arch // 10 not in [9, 12]:
        num_threads = 384

    # 反向 kernel：计算 dk、dv、dq_accum。
    score_mod_hash = utils.hash_callable(score_mod) if score_mod else False
    score_mod_bwd_hash = utils.hash_callable(score_mod_bwd) if score_mod_bwd else False
    mask_mod_hash = utils.hash_callable(mask_mod) if mask_mod else False
    num_aux_tensors = len(aux_tensors) if aux_tensors else 0
    aux_tensor_metadata = get_aux_tensor_metadata(aux_tensors) if aux_tensors is not None else None
    aux_scalar_metadata = tuple(type(s) for s in aux_scalars) if aux_scalars is not None else None

    block_sparse_broadcast_pattern = None
    normalized_block_sparse_tensors = None
    if use_block_sparsity:
        (
            normalized_block_sparse_tensors,
            block_sparse_broadcast_pattern,
        ) = normalize_block_sparse_config_bwd(
            block_sparse_tensors,
            batch_size=batch_size,
            num_head=num_head,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            block_size=(m_block_size, n_block_size),
            q_subtile_factor=q_subtile_factor,
            kv_subtile_factor=kv_subtile_factor,
        )
        if deterministic:
            if normalized_block_sparse_tensors.dq_write_order is None:
                raise ValueError(
                    "deterministic block-sparse backward requires dq_write_order in block_sparse_tensors"
                )
            if (
                normalized_block_sparse_tensors.full_block_cnt is not None
                and normalized_block_sparse_tensors.dq_write_order_full is None
            ):
                raise ValueError(
                    "deterministic block-sparse backward requires dq_write_order_full when full blocks are present"
                )
            if normalized_block_sparse_tensors.spt is None:
                raise ValueError(
                    "deterministic block-sparse backward requires block_sparse_tensors.spt "
                    "to match dq_write_order direction"
                )
    if (
        normalized_block_sparse_tensors is not None
        and normalized_block_sparse_tensors.spt is not None
    ):
        spt = normalized_block_sparse_tensors.spt and deterministic
    else:
        spt = (causal or local) and deterministic

    if arch // 10 in [8, 9, 12]:
        compile_key = (
            arch,
            dtype,
            head_dim,
            head_dim_v,
            qhead_per_kvhead,
            causal,
            window_size_left is not None,
            window_size_right is not None,
            m_block_size,
            n_block_size,
            num_threads,
            pack_gqa,
            num_stages_Q,
            num_stages_dO,
            SdP_swapAB,
            dKV_swapAB,
            dQ_swapAB,
            AtomLayoutMSdP,
            AtomLayoutNdKV,
            AtomLayoutMdQ,
            V_in_regs,
            dQ_single_wg,
            deterministic,
            cu_seqlens_q is None,
            cu_seqlens_k is None,
            seqused_q is None,
            seqused_k is None,
            score_mod_hash,
            score_mod_bwd_hash,
            mask_mod_hash,
            num_aux_tensors,
            aux_tensor_metadata,
            aux_scalar_metadata,
            use_block_sparsity,
            q_subtile_factor,
            block_sparse_broadcast_pattern,
            get_broadcast_dims(q),
            get_broadcast_dims(k),
            get_broadcast_dims(v),
            get_broadcast_dims(dout),
            # 只有一个 block 时，防止 TVM stride 污染。
            single_q_block,
            single_k_block,
            cu_total_m_blocks_k is not None,
        )
    else:
        compile_key = (
            arch,
            dtype,
            head_dim,
            head_dim_v,
            qhead_per_kvhead,
            causal,
            window_size_left is not None,
            window_size_right is not None,
            m_block_size,
            n_block_size,
            num_threads,
            pack_gqa,
            cluster_size,
            use_2cta_instrs,
            q_subtile_factor,
            kv_subtile_factor,
            deterministic,
            spt,
            score_mod_hash,
            score_mod_bwd_hash,
            mask_mod_hash,
            num_aux_tensors,
            aux_tensor_metadata,
            aux_scalar_metadata,
            use_block_sparsity,
            block_sparse_broadcast_pattern,
            cu_seqlens_q is None,
            cu_seqlens_k is None,
            seqused_q is None,
            seqused_k is None,
            get_broadcast_dims(q),
            get_broadcast_dims(k),
            get_broadcast_dims(v),
            get_broadcast_dims(dout),
            # 只有一个 block 时，防止 TVM stride 污染。
            single_q_block,
            single_k_block,
            cu_total_m_blocks_k is not None,
        )

    if compile_key not in _flash_attn_bwd.compile_cache:
        current_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        cute_aux_tensors = (
            [to_cute_aux_tensor(buf) for buf in aux_tensors]
            if aux_tensors is not None
            else None
        )
        q_tensor, k_tensor, v_tensor, do_tensor, dq_tensor, dk_tensor, dv_tensor = [
            to_cute_tensor(t) for t in (q, k, v, dout, dq, dk, dv)
        ]
        lse_log2_tensor, dpsum_tensor = [to_cute_tensor(t) for t in (lse_log2, dpsum)]
        dq_accum_tensor = to_cute_tensor(dq_accum) if dq_accum is not None else None
        if dKV_postprocess:
            dk_accum_tensor, dv_accum_tensor = [
                to_cute_tensor(t) for t in (dk_accum, dv_accum)
            ]
        cu_seqlens_q_tensor, cu_seqlens_k_tensor, seqused_q_tensor, seqused_k_tensor, cu_total_m_blocks_k_tensor = [
            to_cute_tensor(t, assumed_align=4) if t is not None else None
            for t in (cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, cu_total_m_blocks_k)
        ]
        dQ_semaphore_tensor, dK_semaphore_tensor, dV_semaphore_tensor = [
            utils.convert_from_dlpack_leading_static(t.detach(), leading_dim=3, alignment=4, stride_order=t.dim_order())
            if t is not None else None
            for t in (dQ_semaphore, dK_semaphore, dV_semaphore)
        ]
        if arch // 10 in [8, 12]:
            flash_bwd_obj_cls = FlashAttentionBackwardSm120 if arch // 10 == 12 else FlashAttentionBackwardSm80
            fa_bwd_obj = flash_bwd_obj_cls(
                dtype,
                head_dim,
                head_dim_v,
                qhead_per_kvhead,
                m_block_size,
                n_block_size,
                num_stages_Q,
                num_stages_dO,
                num_threads,
                pack_gqa,
                causal,
                local,
                SdP_swapAB,
                dKV_swapAB,
                dQ_swapAB,
                AtomLayoutMSdP,
                AtomLayoutNdKV,
                AtomLayoutMdQ,
                V_in_regs=V_in_regs,
                score_mod=score_mod,
                score_mod_bwd=score_mod_bwd,
            )
        elif arch // 10 == 9:
            fa_bwd_obj = FlashAttentionBackwardSm90(
                dtype,
                head_dim,
                head_dim_v,
                qhead_per_kvhead,
                causal,
                is_local=local,
                deterministic=deterministic,
                tile_m=m_block_size,
                tile_n=n_block_size,
                Q_stage=num_stages_Q,
                dO_stage=num_stages_dO,
                PdS_stage=num_stages_PdS,
                SdP_swapAB=SdP_swapAB,
                dKV_swapAB=dKV_swapAB,
                dQ_swapAB=dQ_swapAB,
                AtomLayoutMSdP=AtomLayoutMSdP,
                AtomLayoutNdKV=AtomLayoutNdKV,
                AtomLayoutMdQ=AtomLayoutMdQ,
                num_threads=num_threads,
                V_in_regs=V_in_regs,
                score_mod=score_mod,
                score_mod_bwd=score_mod_bwd,
                mask_mod=mask_mod,
                has_aux_tensors=aux_tensors is not None,
                q_subtile_factor=q_subtile_factor,
                dQ_single_wg=dQ_single_wg,
            )
        else:
            if use_dedicated_hd256_kernel:
                assert softcap == 0.0, "SM100 backward with head_dim=256 does not support softcap"
                assert block_sparse_tensors is None, \
                    "SM100 backward with head_dim=256 does not support block sparsity"
                assert dlse is None, \
                    "SM100 backward with head_dim=256 does not support dlse"
                assert seqused_q is None and seqused_k is None, \
                    "SM100 backward with head_dim=256 does not support seqused_q/seqused_k"

                dq_tile_mn = (128, 128)
                dkdv_tile_mn = (128, 64)
                fa_bwd_obj = BlackwellFusedMultiHeadAttentionBackward(
                    head_dim,
                    head_dim_v,
                    is_causal=causal,
                    is_local=local,
                    qhead_per_kvhead=qhead_per_kvhead,
                    is_persistent=False,
                    deterministic=deterministic,
                    cluster_size=cluster_size,
                    use_2cta_instrs=use_2cta_instrs,
                    score_mod=score_mod,
                    score_mod_bwd=score_mod_bwd,
                    mask_mod=mask_mod,
                    has_aux_tensors=aux_tensors is not None,
                    q_subtile_factor=q_subtile_factor,
                    tile_m_dq=dq_tile_mn[0],
                    tile_n_dq=dq_tile_mn[1],
                    tile_m_dkdv=dkdv_tile_mn[0],
                    tile_n_dkdv=dkdv_tile_mn[1],
                )
            else:
                fa_bwd_obj = FlashAttentionBackwardSm100(
                    head_dim,
                    head_dim_v,
                    is_causal=causal,
                    is_local=local,
                    qhead_per_kvhead=qhead_per_kvhead,
                    tile_m=m_block_size,
                    tile_n=n_block_size,
                    cluster_size=cluster_size,
                    use_2cta_instrs=use_2cta_instrs,
                    deterministic=deterministic,
                    spt=spt,
                    score_mod=score_mod,
                    score_mod_bwd=score_mod_bwd,
                    mask_mod=mask_mod,
                    has_aux_tensors=aux_tensors is not None,
                    q_subtile_factor=q_subtile_factor,
                    kv_subtile_factor=kv_subtile_factor,
                )

        # 反向的 block sparse 张量使用 Q 方向的索引（相对前向做了转置）。
        sparse_tensors_compile = None
        if normalized_block_sparse_tensors is not None:
            sparse_tensors_compile = to_cute_block_sparse_tensors(normalized_block_sparse_tensors)
        dq_accum_tensor = dq_tensor if use_dedicated_hd256_kernel else dq_accum_tensor

        compile_args = [
            fa_bwd_obj,
            q_tensor,
            k_tensor,
            v_tensor,
            do_tensor,
            lse_log2_tensor,
            dpsum_tensor,
            dq_accum_tensor,
            dk_tensor if not dKV_postprocess else dk_accum_tensor,
            dv_tensor if not dKV_postprocess else dv_accum_tensor,
            softmax_scale,
            cu_seqlens_q_tensor,
            cu_seqlens_k_tensor,
            seqused_q_tensor,
            seqused_k_tensor,
            window_size_left,
            window_size_right,
            dQ_semaphore_tensor,
            dK_semaphore_tensor,
            dV_semaphore_tensor,
            AuxData(cute_aux_tensors, aux_scalars),
            sparse_tensors_compile,
        ]
        if not use_dedicated_hd256_kernel:
            compile_args.append(cu_total_m_blocks_k_tensor)
        compile_args.append(current_stream)

        # TODO: 检查 @can_implement
        _flash_attn_bwd.compile_cache[compile_key] = cute.compile(
            *compile_args, options="--enable-tvm-ffi"
        )
    if not fake_mode:
        dq_accum = dq if use_dedicated_hd256_kernel else dq_accum
        call_args = [
            q.detach(),
            k.detach(),
            v.detach(),
            dout,
            lse_log2,
            dpsum,
            dq_accum,
            dk if not dKV_postprocess else dk_accum,
            dv if not dKV_postprocess else dv_accum,
            softmax_scale,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            window_size_left,
            window_size_right,
            dQ_semaphore,
            dK_semaphore,
            dV_semaphore,
            AuxData(aux_tensors, aux_scalars),
            (
                normalized_block_sparse_tensors.mask_block_cnt,
                normalized_block_sparse_tensors.mask_block_idx,
                normalized_block_sparse_tensors.full_block_cnt,
                normalized_block_sparse_tensors.full_block_idx,
                normalized_block_sparse_tensors.cu_total_m_blocks,
                normalized_block_sparse_tensors.cu_block_idx_offsets,
                normalized_block_sparse_tensors.dq_write_order,
                normalized_block_sparse_tensors.dq_write_order_full,
            )
            if normalized_block_sparse_tensors is not None
            else None,
        ]
        if not use_dedicated_hd256_kernel:
            call_args.append(cu_total_m_blocks_k)
        _flash_attn_bwd.compile_cache[compile_key](*call_args)
    # 后处理：把 dq_accum 从 float32 转换为 bf16/fp16 的 dq
    # hd=256 2CTA 反向有内部的后处理，这里跳过。
    if not use_dedicated_hd256_kernel:
        if arch // 10 == 9:
            # dQ 后处理：与主 kernel 的 MMA warp group 数量保持一致，除非 dQ_single_wg
            num_threads_post_dQ = 128 if dQ_single_wg else cfg.num_wg * 128
            num_threads_post_dKV = cfg.num_wg * 128
        else:
            num_threads_post_dQ = 128
            num_threads_post_dKV = 128

        _bwd_postprocess_convert(
            dq_accum, dq, softmax_scale,
            cu_seqlens_q, seqused_q,
            arch, dtype, head_dim, m_block_size, num_threads_post_dQ,
            AtomLayoutMdQ, dQ_swapAB,
            use_2cta_instrs=use_2cta_instrs, cluster_size=1,
            cu_total_m_blocks=cu_total_m_blocks_q,
            sink_tensors=(
                LearnableSinkBwdTensors(dpsum, lse, learnable_sink, dsink)
                if learnable_sink is not None
                else None
            ),
            fake_mode=fake_mode,
        )

        if dKV_postprocess:
            # 后处理：把 dk_accum 从 float32 转换为 bf16/fp16 的 dk
            _bwd_postprocess_convert(
                dk_accum, dk, softmax_scale,
                cu_seqlens_k, seqused_k,
                arch, dtype, head_dim, n_block_size, num_threads_post_dKV,
                AtomLayoutNdKV, dKV_swapAB,
                cluster_size=cluster_size,
                cu_total_m_blocks=cu_total_m_blocks_k if cluster_size == 1 else None,
                fake_mode=fake_mode,
            )
            # 后处理：把 dv_accum 从 float32 转换为 bf16/fp16 的 dv
            _bwd_postprocess_convert(
                dv_accum, dv, 1.0,
                cu_seqlens_k, seqused_k,
                arch, dtype, head_dim_v, n_block_size, num_threads_post_dKV,
                AtomLayoutNdKV, dKV_swapAB,
                cluster_size=cluster_size,
                cu_total_m_blocks=cu_total_m_blocks_k if cluster_size == 1 else None,
                fake_mode=fake_mode,
            )

    return (dq, dk, dv) if learnable_sink is None else (dq, dk, dv, dsink)


_flash_attn_bwd.compile_cache = get_jit_cache("bwd")


def _flash_attn_bwd_sparse_mla(
    q: Optional[torch.Tensor],
    k: Optional[torch.Tensor],
    v: torch.Tensor,
    qv: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    p: torch.Tensor,
    row_max: torch.Tensor,
    gather_kv_indices: torch.Tensor,
    learnable_sink: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    m_block_size: int = 128,
    n_block_size: int = 64,
    num_threads: int = 256,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    min_seqlen_k: Optional[int] = None,
    deterministic: bool = False,
    dq: Optional[torch.Tensor] = None,
    dk: Optional[torch.Tensor] = None,
    dv: Optional[torch.Tensor] = None,
    dqv: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    fake_mode = is_fake_mode()
    arch = _get_device_arch()
    assert arch // 10 in [10, 11], "Unsupported compute capability. Supported: 10.x, 11.x"
    assert gather_kv_indices is not None, "require gather kv indices for backward"

    q_shape = q.shape if q is not None else qv.shape
    nheads, head_dim = q_shape[-2:]
    nheads_kv, head_dim_v = v.shape[-2:]
    qhead_per_kvhead = nheads // nheads_kv
    gather_kv_length = gather_kv_indices.shape[-1]
    assert nheads_kv == 1 and qhead_per_kvhead == 128, f"sparse MLA bwd: only MQA 128 supported for now"
    assert gather_kv_length % 128 == 0, f"sparse MLA bwd: {gather_kv_length=} must be divisible by 128"
    assert deterministic is False, "sparse MLA bwd: deterministic mode not yet supported"
    assert learnable_sink is None, "sparse MLA bwd: learnable sink not yet supported"
    assert seqused_q is None and seqused_k is None, "sparse MLA bwd: seqused_q,k not yet supported"

    if softmax_scale is None:
        softmax_scale = (
            1.0 / math.sqrt(head_dim) if qv is None or q is None
            else 1.0 / math.sqrt(head_dim + head_dim_v)
        )

    q, k, v, qv, out, dout, lse, p, row_max = [
        maybe_contiguous(t)
        for t in (q, k, v, qv, out, dout, lse, p, row_max)
    ]
    gather_kv_indices, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, learnable_sink = [
        maybe_contiguous(t)
        for t in (gather_kv_indices, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, learnable_sink)
    ]
    device = v.device

    varlen_q = cu_seqlens_q is not None or seqused_q is not None
    if cu_seqlens_q is None:
        batch_size, seqlen_q = q_shape[:2]
        total_q = batch_size * seqlen_q
        p_shape = (batch_size, seqlen_q, nheads, gather_kv_length)
    else:
        batch_size = cu_seqlens_q.shape[0] - 1
        total_q = q_shape[0]
        seqlen_q = max_seqlen_q if max_seqlen_q is not None else total_q
        p_shape = (total_q, nheads, gather_kv_length)

    varlen_k = cu_seqlens_k is not None or seqused_k is not None
    if cu_seqlens_k is None:
        batch_size, seqlen_k = v.shape[:2]
        total_k = batch_size * seqlen_k
    else:
        batch_size = cu_seqlens_k.shape[0] - 1
        total_k = v.shape[0]
        seqlen_k = max_seqlen_k if max_seqlen_k is not None else total_k
    if not varlen_k:
        min_seqlen_k = seqlen_k 

    assert varlen_q == varlen_k, "sparse MLA bwd: either q and k are both varlen or not"

    # 默认总是使用 kv bitmask（用于处理 -1 哨兵值）
    disable_sparse_kv_bitmask = False
    # if min_seqlen_k is None or causal:
    #     disable_sparse_kv_bitmask = False
    # else:
    #     disable_sparse_kv_bitmask = min_seqlen_k >= gather_kv_length

    prealloc_dq = dq is not None
    prealloc_dk = dk is not None
    prealloc_dqv = dqv is not None
    prealloc_dv = dv is not None
    if not prealloc_dq and q is not None:
        dq = torch.empty_like(q)
    if not prealloc_dk and k is not None:
        dk = torch.zeros_like(k, dtype=torch.float32)
    if not prealloc_dv:
        dv = torch.zeros_like(v, dtype=torch.float32)
    if not prealloc_dqv:
        dqv = torch.empty_like(qv)
    ds = torch.empty_like(p)

    device = v.device
    dtype = v.dtype
    if q is not None:
        _validate_tensor(dq, "dq", q.shape, dtype, device)
    if k is not None:
        _validate_tensor(dk, "dk", k.shape, torch.float32, device)
    _validate_tensor(dv, "dv", v.shape, torch.float32, device)
    _validate_tensor(dqv, "dqv", qv.shape, dtype, device)
    _validate_tensor(p, "p", p_shape, dtype, device)

    if cu_seqlens_q is None:
        dpsum = torch.empty(batch_size, seqlen_q, nheads, dtype=torch.float32, device=device)
    else:
        dpsum = torch.empty(total_q, nheads, dtype=torch.float32, device=device)
    scale_p = torch.empty_like(row_max)

    dtype = torch2cute_dtype_map[dout.dtype]

    # 预处理 kernel：计算 (o * dout).sum(dim=-1) 和 scale_p。
    _bwd_preprocess(
        out, dout, dpsum, lse, None, None,
        cu_seqlens_q, seqused_q, None,
        dtype, head_dim, head_dim_v, m_block_size,
        row_max=row_max,
        scale_p=scale_p,
        use_padded_offsets=False,
        nheads_major=True,
        pack_gqa=True,
        qhead_per_kvhead=qhead_per_kvhead,
        nheads_kv=nheads_kv,
        softmax_scale=softmax_scale,
        fake_mode=fake_mode,
    )

    compile_key = (
        dtype,
        head_dim,
        head_dim_v,
        qhead_per_kvhead,
        causal,
        cu_seqlens_q is None,
        cu_seqlens_k is None,
        seqused_q is None,
        seqused_k is None,
        q is not None,
        gather_kv_length,
        learnable_sink is not None,
        disable_sparse_kv_bitmask,
    )

    if compile_key not in _flash_attn_bwd_sparse_mla.compile_cache:
        current_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        (
            cu_seqlens_q_tensor,
            cu_seqlens_k_tensor,
            seqused_q_tensor,
            seqused_k_tensor,
            learnable_sink_tensor,
        ) = [
            to_cute_tensor(t, assumed_align=4, leading_dim=0)
            for t in (cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, learnable_sink)
        ]
        (
            v_tensor,
            qv_tensor,
            do_tensor,
            p_tensor,
            scale_p_tensor,
            dpsum_tensor,
            ds_tensor,
            dv_tensor,
            gather_kv_indices_tensor,
         ) = [
            to_cute_tensor(t) for t in (v, qv, dout, p, scale_p, dpsum, ds, dv, gather_kv_indices)
        ]

        fa_bwd_obj = FlashAttentionSparseMLABackwardSm100(
            is_causal=causal,
            topk_length=gather_kv_length,
            qhead_per_kvhead=qhead_per_kvhead,
            nheads_kv=nheads_kv,
            has_seqused_q=seqused_q is not None,
            disable_bitmask=disable_sparse_kv_bitmask,
        )
        fa_bwd_kernel = cute.compile(
            fa_bwd_obj,
            do_tensor,
            v_tensor,
            qv_tensor,
            p_tensor,
            dv_tensor,
            ds_tensor,
            gather_kv_indices_tensor,
            softmax_scale,
            scale_p_tensor,
            dpsum_tensor,
            cu_seqlens_q_tensor,
            cu_seqlens_k_tensor,
            seqused_q_tensor,
            seqused_k_tensor,
            current_stream,
            options="--enable-tvm-ffi",
        )
        _flash_attn_bwd_sparse_mla.compile_cache[compile_key] = fa_bwd_kernel

    if not fake_mode:
        _flash_attn_bwd_sparse_mla.compile_cache[compile_key](
            dout,
            v,
            qv,
            p,
            dv,
            ds,
            gather_kv_indices,
            softmax_scale,
            scale_p,
            dpsum,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
        )

    v = v.squeeze(-2)
    if k is not None:
        k = k.squeeze(-2)
    
    _sparse_mla_dq_dqv(
        ds, k, v, dq, dqv, gather_kv_indices, cu_seqlens_q, cu_seqlens_k,
    )

    if k is not None:
        dk = dk.squeeze(-2)
        _sparse_mla_dk(ds, gather_kv_indices, q, dk, cu_seqlens_q, cu_seqlens_k)
        dk = dk.unsqueeze(-2)
    
    # 以 float32 返回 dk、dv：跨序列并行 rank 的 all-reduce 必须发生在降精度之前，
    # 以避免 rank 间梯度累加时的舍入误差
    return dq, dk, dv, dqv

_flash_attn_bwd_sparse_mla.compile_cache = get_jit_cache("bwd_dsa")


def _compile_sparse_mla_dq_dqv(
    dtype, nheads, head_dim, head_dim_v, top_k, varlen_q, varlen_k, compute_dq,
):
    sym = cute.sym_int 
    b, b_plus_1, seqlen_q, seqlen_k = sym(), sym(), sym(), sym()
    total_q, total_k = sym(), sym()
    b_seqlenq = (b, seqlen_q) if not varlen_q else (total_q,)
    b_seqlenk = (b, seqlen_k) if not varlen_k else (total_k,)
    
    div = 128 // dtype.width  # fp16/bf16 时为 8
    
    mdS = fake_tensor(dtype, (*b_seqlenq, nheads, top_k), divisibility=div)
    mK = fake_tensor(dtype, (*b_seqlenk, head_dim), divisibility=div)
    mV = fake_tensor(dtype, (*b_seqlenk, head_dim_v), divisibility=div)
    mdQ = fake_tensor(dtype, (*b_seqlenq, nheads, head_dim), divisibility=div)
    mdQv = fake_tensor(dtype, (*b_seqlenq, nheads, head_dim_v), divisibility=div)
    mIdxTopK = fake_tensor(Int32, (*b_seqlenq, top_k), divisibility=div)
    
    mCuSeqlensQ = fake_tensor(Int32, (b_plus_1,), divisibility=1) if varlen_q else None 
    mCuSeqlensK = fake_tensor(Int32, (b_plus_1,), divisibility=1) if varlen_k else None 
    
    dq_dqv_gemm = dQdQvGemmKernel(
        acc_dtype=Float32,
        nheads=nheads,
        head_dim_k=head_dim,
        head_dim_v=head_dim_v,
        top_k=top_k,
    )
    
    return cute.compile(
        dq_dqv_gemm,
        mdS,
        mK if compute_dq else None,
        mV,
        mdQ if compute_dq else None,
        mdQv,
        mIdxTopK,
        mCuSeqlensQ,
        mCuSeqlensK,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _sparse_mla_dq_dqv(
    ds, k, v, dq, dqv, gather_kv_indices, cu_seqlens_q, cu_seqlens_k,
):
    """计算 dQ = dS @ K 以及 dQv = dS @ V"""
    *_, nheads, gather_kv_length = ds.shape
    
    head_dim_v = v.shape[-1]
    head_dim = k.shape[-1] if k is not None else 0
    
    dtype = ds.dtype
    dtype_cute = torch2cute_dtype_map[dtype]
    
    varlen_q = cu_seqlens_q is not None
    varlen_k = cu_seqlens_k is not None
    
    compile_key = (
        dtype_cute, nheads, head_dim, head_dim_v, gather_kv_length, varlen_q, varlen_k, k is not None,
    )
    if compile_key not in _sparse_mla_dq_dqv.compile_cache:
        _sparse_mla_dq_dqv.compile_cache[compile_key] = _compile_sparse_mla_dq_dqv(
            *compile_key
        )
    if not is_fake_mode():
        _sparse_mla_dq_dqv.compile_cache[compile_key](
            ds, k, v, dq, dqv, gather_kv_indices, cu_seqlens_q, cu_seqlens_k
        )

_sparse_mla_dq_dqv.compile_cache = get_jit_cache("dq_dqv_gemm")


def _compile_sparse_mla_dk(
    dtype,
    dtype_acc,
    nheads: int,
    head_dim: int,
    topk: int,
    varlen: bool,
):
    kernel = dKGemmKernel(
        topk,
        nheads,
        head_dim,
        varlen,
    )
    # 检查该配置能否被实现
    kernel.check_can_implement()

    div = 128 // dtype.width

    sym = cute.sym_int
    batch_fake = sym()
    batchp1_fake = sym()
    seqlen_q_fake = sym()
    seqlen_k_fake = sym()
    total_q_fake = (batch_fake, seqlen_q_fake) if not varlen else (sym(),)
    total_k_fake = (batch_fake, seqlen_k_fake) if not varlen else (sym(),)

    mdS = fake_tensor(dtype, (*total_q_fake, nheads, topk), divisibility=div)
    mI = fake_tensor(Int32, (*total_q_fake, topk), divisibility=div)
    mQ = fake_tensor(dtype, (*total_q_fake, nheads, head_dim), divisibility=div)
    mdK = fake_tensor(dtype_acc, (*total_k_fake, head_dim), divisibility=div)
    mCuSeqlensQ = fake_tensor(Int32, (batchp1_fake,), divisibility=1) if varlen else None
    mCuSeqlensK = fake_tensor(Int32, (batchp1_fake,), divisibility=1) if varlen else None
    
    return cute.compile(
        kernel,
        mdS,
        mI,
        mQ,
        mdK,
        mCuSeqlensQ,
        mCuSeqlensK,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _sparse_mla_dk(
    dS: torch.Tensor,
    index_topk: torch.Tensor,
    q: torch.Tensor,
    dk: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
):
    """计算 dKaccum = scatter(dS'^T @ Q, I)（按 topk 索引把梯度分散累加到 dK）。

    Args:
      dS:          (*total_q, heads, topk), bf16
      index_topk:  (*total_q, topk), int32
      Q:           (*total_q, heads, dim), bf16
      dK:          (*total_q, dim), fp32
      cuSeqlensQ:  (batch + 1,), int32, 非 varlen 时省略
      cuSeqlensK:  (batch + 1,), int32, 非 varlen 时省略

    在 dK 原有内容上原地累加。

    对 varlen：total_q 和 total_k 是一维的，每个 batch 的 seqlen 索引由
    cuSeqlensQ 和 cuSeqlensK 张量确定。
    对非 varlen：total_q、total_k 分别是 (batch, seqlen_q) 与 (batch, seqlen_k)。
    """
    dtype = dS.dtype
    dtype_cute = torch2cute_dtype_map[dtype]
    dtype_acc = dk.dtype
    dtype_acc_cute = torch2cute_dtype_map[dtype_acc]

    varlen = cu_seqlens_q is not None
    nheads, topk = dS.shape[-2], dS.shape[-1]
    head_dim = q.shape[-1] if q is not None else 0

    compile_key = (
        dtype_cute, dtype_acc_cute, nheads, head_dim, topk, varlen,
    )

    if compile_key not in _sparse_mla_dk.compile_cache:
        _sparse_mla_dk.compile_cache[compile_key] = _compile_sparse_mla_dk(*compile_key)

    if not is_fake_mode():
        _sparse_mla_dk.compile_cache[compile_key](dS, index_topk, q, dk, cu_seqlens_q, cu_seqlens_k)
    
_sparse_mla_dk.compile_cache = get_jit_cache("dk_gemm")


class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        qv: Optional[torch.Tensor] = None,
        gather_kv_indices: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        window_size: Tuple[Optional[int], Optional[int]] = (None, None),
        learnable_sink: Optional[torch.Tensor] = None,
        softcap: float = 0.0,
        num_splits: int = 1,
        pack_gqa: Optional[bool] = None,
        deterministic: bool = False,
        score_mod: Optional[Callable] = None,
        score_mod_bwd: Optional[Callable] = None,
        mask_mod: Optional[Callable] = None,
        aux_tensors: Optional[list] = None,
        aux_scalars: Optional[tuple] = None,
        block_sparse_tensors: Optional[BlockSparseTensorsTorch] = None,
        block_sparse_tensors_bwd: Optional[BlockSparseTensorsTorch] = None,
        return_lse: bool = False,
    ):
        aux_scalars = tuple(aux_scalars) if aux_scalars else None
        shared_kv = k is v
        if shared_kv and v.shape[-1] == 512:
            # 特化 MLA 注意力公式
            # O = softmax(Q @ K.T + Qv @ V.T) @ V
            # 通过把 q、k 设为 None 来实现
            qv = q if qv is None else qv
            q = k = None
        out, lse, p, row_max = _flash_attn_fwd(
            q,
            k,
            v,
            qv=qv,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            learnable_sink=learnable_sink,
            softcap=softcap,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            score_mod=score_mod,
            mask_mod=mask_mod,
            aux_tensors=aux_tensors,
            aux_scalars=aux_scalars,
            block_sparse_tensors=block_sparse_tensors,
            return_lse=return_lse,
            gather_kv_indices=gather_kv_indices,
        )
        ctx.save_for_backward(q, k, v, qv, out, lse, p, row_max, gather_kv_indices, learnable_sink, *(aux_tensors or ()))
        ctx.shared_kv = shared_kv
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        ctx.return_lse = return_lse
        ctx.score_mod = score_mod 
        ctx.score_mod_bwd = score_mod_bwd 
        ctx.mask_mod = mask_mod
        ctx.aux_scalars = aux_scalars
        ctx.block_sparse_tensors_bwd = block_sparse_tensors_bwd
        ctx.set_materialize_grads(False)
        return out, lse

    @staticmethod
    def backward(ctx, dout, dlse):
        q, k, v, qv, out, lse, p, row_max, gather_kv_indices, learnable_sink, *aux = ctx.saved_tensors
        aux_tensors = aux if aux else None
        if not ctx.return_lse:
            dlse = None
        if dout is None:
            dout = torch.zeros_like(out)
        if qv is not None:
            dq, dk, dv, dqv = _flash_attn_bwd_sparse_mla(
                q,
                k,
                v,
                qv,
                out,
                dout,
                lse,
                p,
                row_max,
                gather_kv_indices,
                softmax_scale=ctx.softmax_scale,
                causal=ctx.causal,
            )
            if ctx.shared_kv:
                return dqv, dv, None, None, *((None,) * 30)
            else:
                return dq, dk, dv, dqv, *((None,) * 30)
        else:
            bwd_result = _flash_attn_bwd(
                q,
                k,
                v,
                out,
                dout,
                lse,
                ctx.softmax_scale,
                ctx.causal,
                ctx.softcap,
                window_size_left=ctx.window_size[0],
                window_size_right=ctx.window_size[1],
                deterministic=ctx.deterministic,
                score_mod=ctx.score_mod,
                score_mod_bwd=ctx.score_mod_bwd,
                mask_mod=ctx.mask_mod,
                aux_tensors=aux_tensors,
                aux_scalars=ctx.aux_scalars,
                block_sparse_tensors=ctx.block_sparse_tensors_bwd,
                dlse=dlse,
                learnable_sink=learnable_sink,
            )
            if learnable_sink is None:
                dq, dk, dv = bwd_result
                dsink = None
            else:
                dq, dk, dv, dsink = bwd_result
            return dq, dk, dv, None, None, None, None, None, dsink, *((None,) * 12)


class FlashAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: Optional[torch.Tensor],
        k: Optional[torch.Tensor],
        v: torch.Tensor,
        qv: Optional[torch.Tensor] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        seqused_q: Optional[torch.Tensor] = None,
        seqused_k: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_k: Optional[int] = None,
        min_seqlen_k: Optional[int] = None,
        gather_kv_indices: Optional[torch.Tensor] = None,
        page_table: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        window_size: Tuple[Optional[int], Optional[int]] = (None, None),
        learnable_sink: Optional[torch.Tensor] = None,
        softcap: float = 0.0,
        num_splits: int = 1,
        pack_gqa: Optional[bool] = None,
        deterministic: bool = False,
        score_mod: Optional[Callable] = None,
        score_mod_bwd: Optional[Callable] = None,
        mask_mod: Optional[Callable] = None,
        block_sparse_tensors: Optional[list] = None,
        aux_tensors: Optional[list] = None,
        aux_scalars: Optional[tuple] = None,
        return_lse: bool = False,
        scheduler_metadata: Optional["SchedulerMetadataTensorsTorch"] = None,
        seqlen_k_per_split: Optional[int] = None,
        disable_scheduler_metadata: bool = False,
    ):
        aux_scalars = tuple(aux_scalars) if aux_scalars else None
        shared_kv = k is v
        if shared_kv and v.shape[-1] == 512:
            # 特化 MLA 注意力公式
            # O = softmax(Q @ K.T + Qv @ V.T) @ V
            # 通过把 q、k 设为 None 来实现
            qv = q if qv is None else qv
            q = k = None
        out, lse, p, row_max = _flash_attn_fwd(
            q,
            k,
            v,
            qv=qv,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            min_seqlen_k=min_seqlen_k,
            page_table=page_table,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            learnable_sink=learnable_sink,
            softcap=softcap,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            score_mod=score_mod,
            mask_mod=mask_mod,
            block_sparse_tensors=block_sparse_tensors,
            aux_tensors=aux_tensors,
            aux_scalars=aux_scalars,
            return_lse=return_lse,
            gather_kv_indices=gather_kv_indices,
            scheduler_metadata=scheduler_metadata,
            seqlen_k_per_split=seqlen_k_per_split,
            disable_scheduler_metadata=disable_scheduler_metadata,
        )
        ctx.save_for_backward(
            q,
            k,
            v,
            qv,
            out,
            lse,
            p,
            row_max,
            gather_kv_indices,
            learnable_sink,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            *(aux_tensors or ()),
        )
        ctx.shared_kv = shared_kv
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.min_seqlen_k = min_seqlen_k
        ctx.return_lse = return_lse
        ctx.score_mod = score_mod
        ctx.score_mod_bwd = score_mod_bwd
        ctx.mask_mod = mask_mod
        ctx.aux_scalars = aux_scalars
        ctx.set_materialize_grads(False)
        return out, lse

    @staticmethod
    def backward(ctx, dout, dlse):
        q, k, v, qv, out, lse, p, row_max, gather_kv_indices, learnable_sink, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, *aux = ctx.saved_tensors
        aux_tensors = aux if aux else None
        if not ctx.return_lse:
            dlse = None
        if dout is None:
            dout = torch.zeros_like(out)
        if qv is not None:
            dq, dk, dv, dqv = _flash_attn_bwd_sparse_mla(
                q,
                k,
                v,
                qv,
                out,
                dout,
                lse,
                p,
                row_max,
                gather_kv_indices,
                softmax_scale=ctx.softmax_scale,
                causal=ctx.causal,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                seqused_q=seqused_q,
                seqused_k=seqused_k,
                max_seqlen_q=ctx.max_seqlen_q,
                max_seqlen_k=ctx.max_seqlen_k,
                min_seqlen_k=ctx.min_seqlen_k,
            )
            if ctx.shared_kv:
                return dqv, dv, None, None, *((None,) * 31)
            else:
                return dq, dk, dv, dqv, *((None,) * 31)
        else:
            bwd_result = _flash_attn_bwd(
                q,
                k,
                v,
                out,
                dout,
                lse,
                ctx.softmax_scale,
                ctx.causal,
                ctx.softcap,
                window_size_left=ctx.window_size[0],
                window_size_right=ctx.window_size[1],
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                seqused_q=seqused_q,
                seqused_k=seqused_k,
                max_seqlen_q=ctx.max_seqlen_q,
                max_seqlen_k=ctx.max_seqlen_k,
                deterministic=ctx.deterministic,
                score_mod=ctx.score_mod,
                score_mod_bwd=ctx.score_mod_bwd,
                aux_tensors=aux_tensors,
                aux_scalars=ctx.aux_scalars,
                mask_mod=ctx.mask_mod,
                dlse=dlse,
                learnable_sink=learnable_sink,
            )
            if learnable_sink is None:
                dq, dk, dv = bwd_result
                dsink = None
            else:
                dq, dk, dv, dsink = bwd_result
            return dq, dk, dv, None, *((None,) * 12), dsink, *((None,) * 14)


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qv: Optional[torch.Tensor] = None,
    gather_kv_indices: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    learnable_sink: Optional[torch.Tensor] = None,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    deterministic: bool = False,
    score_mod: Optional[Callable] = None,
    score_mod_bwd: Optional[Callable] = None,
    mask_mod: Optional[Callable] = None,
    aux_tensors: Optional[list] = None,
    aux_scalars: Optional[tuple] = None,
    block_sparse_tensors: Optional[BlockSparseTensorsTorch] = None,
    block_sparse_tensors_bwd: Optional[BlockSparseTensorsTorch] = None,
    return_lse: bool = False,
):
    return FlashAttnFunc.apply(
        q,
        k,
        v,
        qv,
        gather_kv_indices,
        softmax_scale,
        causal,
        window_size,
        learnable_sink,
        softcap,
        num_splits,
        pack_gqa,
        deterministic,
        score_mod,
        score_mod_bwd,
        mask_mod,
        aux_tensors,
        aux_scalars,
        block_sparse_tensors,
        block_sparse_tensors_bwd,
        return_lse,
    )


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qv: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    min_seqlen_k: Optional[int] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    gather_kv_indices: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    learnable_sink: Optional[torch.Tensor] = None,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    deterministic: bool = False,
    score_mod: Optional[Callable] = None,
    score_mod_bwd: Optional[Callable] = None,
    mask_mod: Optional[Callable] = None,
    block_sparse_tensors: Optional[BlockSparseTensorsTorch] = None,
    aux_tensors: Optional[list] = None,
    aux_scalars: Optional[tuple] = None,
    return_lse: bool = False,
    scheduler_metadata: Optional[SchedulerMetadataTensorsTorch] = None,
    seqlen_k_per_split: Optional[int] = None,
    disable_scheduler_metadata: bool = False,
):
    """
    变长（varlen）FlashAttention 的对外接口。

    varlen 与 flash_attn_func 的区别：batch 内各条序列的长度可以不同，Q/K/V 被
    「拼接」成一维的 (total_q, ...) / (total_k, ...)，再用 cu_seqlens_q / cu_seqlens_k
    记录每个 batch 的累积起始位置（cu_seqlens[i] 到 cu_seqlens[i+1] 即第 i 条序列的
    长度区间）。max_seqlen_q / max_seqlen_k 给出最大的可能长度，用于 kernel 端
    预分配与 tile 规划。

    Tensor arguments:
        q:  (total_q, nheads,   hdim)   or (batch, seqlen_q, nheads,   hdim)
        k:  (total_k, nheads_k, hdim)   or (batch, seqlen_k, nheads_k, hdim)
        v:  (total_k, nheads_k, hdim_v) or (batch, seqlen_k, nheads_k, hdim_v)
        qv: (total_q, nheads,   hdim_v) or (batch, seqlen_q, nheads,   hdim_v)
        cu_seqlens_q: (batch + 1)       or seqused_q: (batch)
        cu_seqlens_k: (batch + 1)       or seqused_k: (batch)
        gather_kv_indices: (total_q, gather_kv_length) or
                           (batch, seqlen_q, gather_kv_length)
        page_table: (batch, max_num_pages_per_seq)
        说明：q/k/v 传一维形式时是 varlen；传 (batch, seqlen, ...) 形式时是标准 batched。
        seqused_q/seqused_k 与 cu_seqlens 二选一，用于表示每个 batch 实际参与计算的长度。

    Return:
       out: (total_q, nheads, hdim) or (batch, seqlen_q, nheads, hdim)
       lse: (nheads, total_q)       or (batch, nheads, seqlen_q) if not has_qv (standard)
            (total_q, nheads)       or (batch, seqlen_q, nheads) if has_qv

    部分可选参数与设计取舍的说明：

    qv: 我们把 MLA（Multi-head Latent Attention）权重吸收公式写成
        O = softmax(scale * (Q @ K.T + Qv @ V.T)) @ V
        其中 Q = q_pe, Qv = q_nope, K = pe_cache, V = kv_cache。
        qv 是 MLA 专用的「第二组 query」，用于把权重吸收进 V 侧，从而避免在
        KV cache 中存完整的 K。

    lse 返回形状：有 qv 时典型是 MQA 且 nheads 至少被 4 整除，因此我们让 nheads
        作为连续维以利于向量化。

    gather_kv_indices: 与 MLA absorption kernel 配合，用于 topk 稀疏：只对每个 query
        预先选定的 gather_kv_length 个 KV 做注意力，索引就存在这里。

    min_seqlen_k: 对 varlen 而言，指定任意 batch 的最小 kv 序列长度。
        与 gather_kv_indices 一起用于判断是否需要 oob（越界）mask。

    scheduler_metadata: 供某些 tile scheduler 使用的可选张量，用于优化与功能扩展，
        由 get_scheduler_metadata 计算。

    seqlen_k_per_split: 使用动态（按 batch）num_splits 时，可固定每个 split 覆盖的
        seqlen_k，以实现前向/反向的逐位可复现（bitwise reproducibility）。

    disable_scheduler_metadata: 为 True 时忽略传入的 scheduler_metadata，
        并跳过重新计算元数据的步骤。
    """
    return FlashAttnVarlenFunc.apply(
        q,
        k,
        v,
        qv,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        min_seqlen_k,
        gather_kv_indices,
        page_table,
        softmax_scale,
        causal,
        window_size,
        learnable_sink,
        softcap,
        num_splits,
        pack_gqa,
        deterministic,
        score_mod,
        score_mod_bwd,
        mask_mod,
        block_sparse_tensors,
        aux_tensors,
        aux_scalars,
        return_lse,
        scheduler_metadata,
        seqlen_k_per_split,
        disable_scheduler_metadata,
    )


def _compile_fwd_combine(
    _arch, dtype, dtype_partial, head_dim, num_head, tile_m, k_block_size, log_max_splits,
    has_cu_seqlens, has_seqused, has_lse, has_virtual_batch_idx,
    has_num_splits_dynamic, has_semaphore_to_reset,
):
    """使用 cute fake tensors 编译前向 combine kernel（无需真实 GPU 张量）。"""
    sym = cute.sym_int
    div = 128 // dtype_partial.width  # 元素个数意义上的 16 字节对齐

    fa_combine = FlashAttentionForwardCombine(
        dtype=dtype,
        dtype_partial=dtype_partial,
        head_dim=head_dim,
        num_head=num_head,
        tile_m=tile_m,
        k_block_size=k_block_size,
        log_max_splits=log_max_splits,
    )
    if not fa_combine.can_implement(
        dtype, dtype_partial, head_dim, tile_m, k_block_size, log_max_splits,
        num_threads=256,
    ):
        raise RuntimeError(
            "FlashAttention combine kernel cannot be implemented with given parameters"
        )

    if has_cu_seqlens:
        # 变长：(num_splits, total_q, nheads, headdim)
        num_splits, total_q, nheads = sym(), sym(), sym()
        mO_partial = fake_tensor(dtype_partial, (num_splits, total_q, nheads, head_dim), divisibility=div)
        mLSE_partial = fake_tensor(Float32, (num_splits, total_q, nheads), divisibility=1, leading_dim=1)
        mO = fake_tensor(dtype, (total_q, nheads, head_dim), divisibility=div)
        mLSE = fake_tensor(Float32, (total_q, nheads), divisibility=1, leading_dim=0) if has_lse else None
    else:
        # 批处理：(num_splits, batch, seqlen, nheads, headdim)
        num_splits, batch, seqlen, nheads = sym(), sym(), sym(), sym()
        mO_partial = fake_tensor(dtype_partial, (num_splits, batch, seqlen, nheads, head_dim), divisibility=div)
        mLSE_partial = fake_tensor(Float32, (num_splits, batch, seqlen, nheads), divisibility=1, leading_dim=2)
        mO = fake_tensor(dtype, (batch, seqlen, nheads, head_dim), divisibility=div)
        mLSE = fake_tensor(Float32, (batch, seqlen, nheads), divisibility=1, leading_dim=1) if has_lse else None
        batch = mO_partial.shape[1]

    batch_for_1d = batch if not has_cu_seqlens else sym()
    batchp1 = sym()
    mCuSeqlens = fake_tensor(Int32, (batchp1,), divisibility=1) if has_cu_seqlens else None
    mSeqused = fake_tensor(Int32, (batch_for_1d,), divisibility=1) if has_seqused else None
    mNumSplitsDynamic = fake_tensor(Int32, (batch_for_1d,), divisibility=1) if has_num_splits_dynamic else None
    mVirtualBatchIdx = fake_tensor(Int32, (batch_for_1d,), divisibility=1) if has_virtual_batch_idx else None
    mSemaphore = fake_tensor(Int32, (1,), divisibility=1) if has_semaphore_to_reset else None

    return cute.compile(
        fa_combine,
        mO_partial, mLSE_partial, mO, mLSE,
        mCuSeqlens, mSeqused, mNumSplitsDynamic, mVirtualBatchIdx, mSemaphore,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _flash_attn_fwd_combine(
    out_partial: torch.Tensor,
    lse_partial: torch.Tensor,
    out: torch.Tensor,
    lse: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    num_splits_dynamic_ptr: Optional[torch.Tensor] = None,
    virtual_batch_idx: Optional[torch.Tensor] = None,
    semaphore_to_reset: Optional[torch.Tensor] = None,
    *,
    _arch: Optional[int] = None,
) -> None:
    """Split attention 的前向 combine kernel。

    把注意力计算多个 split（num_splits > 1 的 SplitKV）产生的部分输出与
    log-sum-exp 值合并为最终输出。每个 split 各自维护局部的行最大值与行求和，
    combine 阶段用 LSE 重新归一化，保证结果与不 split 时数值一致。

    Args:
        out_partial: 部分输出张量，形状 (num_splits, batch, seqlen, nheads, headdim)；
            有 cu_seqlens 时为 (num_splits, total_q, nheads, headdim)
        lse_partial: 部分 LSE 张量，形状 (num_splits, batch, seqlen, nheads)；
            有 cu_seqlens 时为 (num_splits, total_q, nheads)
        out: 输出张量，形状 (batch, seqlen, nheads, headdim)；
            有 cu_seqlens 时为 (total_q, nheads, headdim)
        lse: 输出的 LSE 张量，形状 (batch, seqlen, nheads)；
            有 cu_seqlens 时为 (total_q, nheads)
        cu_seqlens: 变长序列的累积序列长度
        seqused: 每个 batch 实际使用的序列长度
        num_splits_dynamic_ptr: 每个 batch 的动态 split 数量
        semaphore_to_reset: 用于同步的信号量
        k_block_size: head 维度方向上的块大小

    Returns:
        None
    """
    fake_mode = is_fake_mode()
    assert out_partial.dtype in [torch.float16, torch.bfloat16, torch.float32], (
        "out_partial must be fp16, bf16, or fp32"
    )
    if not fake_mode:
        assert out_partial.is_cuda and lse_partial.is_cuda, "tensors must be on CUDA device"
    for tensor, name in (
        (cu_seqlens, "cu_seqlens"),
        (seqused, "seqused"),
        (num_splits_dynamic_ptr, "num_splits_dynamic_ptr"),
        (virtual_batch_idx, "virtual_batch_idx"),
        (semaphore_to_reset, "semaphore_to_reset"),
    ):
        if tensor is not None:
            if not fake_mode:
                assert tensor.is_cuda, f"{name} must be on CUDA device"
            assert tensor.is_contiguous(), f"{name} must be contiguous"
    head_dim = out_partial.shape[-1]
    num_head = out_partial.shape[-2]
    num_splits = out_partial.shape[0]
    assert num_splits <= 256
    # 如果 hdim 是 96 或 192，把它们分别向上取整到 128 或 256 会更快，
    # 因为这样 kBlockM 更小，我们能获得更多并行度。
    k_block_size = 64 if head_dim <= 64 else 128
    # 我们希望 kBlockM 尽可能小以最大化并行度。
    # 例如 hdim 为 64 时，kBlockM 取 16，这样能用 256 个线程，每个线程读 4 个元素（float）。
    tile_m = 8 if k_block_size % 128 == 0 else (16 if k_block_size % 64 == 0 else 32)
    log_max_splits = max(math.ceil(math.log2(num_splits)), 4)
    if tile_m == 8:
        # 如果 kBlockM == 8，那么最小的 split 数量是 32。
        # TODO: 我们可以改用 128 线程来解决这个问题
        log_max_splits = max(log_max_splits, 5)

    # 创建 combine kernel 配置
    dtype = torch2cute_dtype_map[out.dtype]
    dtype_partial = torch2cute_dtype_map[out_partial.dtype]
    compile_key = (
        _get_device_arch() if _arch is None else _arch,
        dtype,
        dtype_partial,
        head_dim,
        num_head,
        tile_m,
        k_block_size,
        log_max_splits,
        cu_seqlens is not None,
        seqused is not None,
        lse is not None,
        virtual_batch_idx is not None,
        num_splits_dynamic_ptr is not None,
        semaphore_to_reset is not None,
    )
    if compile_key not in _flash_attn_fwd_combine.compile_cache:
        _flash_attn_fwd_combine.compile_cache[compile_key] = _compile_fwd_combine(
            *compile_key
        )
    if not fake_mode:
        _flash_attn_fwd_combine.compile_cache[compile_key](
            out_partial, lse_partial, out, lse,
            cu_seqlens, seqused, num_splits_dynamic_ptr, virtual_batch_idx,
            semaphore_to_reset,
        )


_flash_attn_fwd_combine.compile_cache = get_jit_cache("fwd_combine")


def flash_attn_combine(
    out_partial: torch.Tensor,
    lse_partial: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    virtual_batch_idx: Optional[torch.Tensor] = None,
    return_lse: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Split attention 的 FlashAttention combine 函数（面向用户的入口）。

    把注意力计算多个 split 产生的部分输出与 log-sum-exp 值合并为最终输出。
    这是 combine kernel 的主要用户接口。通常不需要手动调用——在 num_splits > 1 时
    flash_attn_func / flash_attn_varlen_func 内部会自动完成 combine；此函数用于
    拿到 partial 结果后需要自行合并的场景。

    Args:
        out_partial: 部分输出张量，形状：
            - (num_splits, batch_size, seqlen, num_heads, head_size) 普通 batched 输入
            - (num_splits, total_q, num_heads, head_size) 变长输入
        lse_partial: 部分 LSE 张量，形状：
            - (num_splits, batch_size, seqlen, num_heads) 普通 batched 输入
            - (num_splits, total_q, num_heads) 变长输入
        out: 可选输出张量。为 None 时自动创建。
        out_dtype: 可选输出 dtype。为 None 时根据输入使用 fp16/bf16。
        cu_seqlens: 变长序列的累积序列长度
        seqused: 每个 batch 实际使用的序列长度
        virtual_batch_idx: 可选的「虚拟 batch 索引 -> 真实 batch 索引」映射
            （int32 张量，形状 (batch_size,)）。持久化 tile scheduler 为做负载均衡
            重排 batch 处理顺序时会用到。
        return_lse: 是否返回合并后的 LSE 张量。默认 True。

    Returns:
        (out, lse) 元组，其中：
        - out: 合并后的输出张量，形状 (batch_size, seqlen, num_heads, head_size)
              或 varlen 下的 (total_q, num_heads, head_size)
        - lse: 合并后的 log-sum-exp 张量，形状 (batch_size, seqlen, num_heads)
              或 varlen 下的 (total_q, num_heads)。return_lse=False 时为 None

    Note:
        本函数期望输入张量是 split 注意力计算产生的格式，即第一维是 num_splits。
        从用户格式到 kernel 格式的转置现在在 kernel 内部完成。
    """
    # 输入校验
    assert out_partial.dim() in [4, 5], "out_partial must have 4 or 5 dimensions"
    # 根据维度判断是否为变长
    is_varlen = out_partial.dim() == 4
    if is_varlen:
        # 变长：(num_splits, total_q, num_heads, head_size)
        num_splits, total_q, num_heads, head_size = out_partial.shape
        batch_size = 1  # varlen 场景下视为单个 batch
        seqlen = total_q
    else:
        # 常规批处理：(num_splits, batch_size, seqlen, num_heads, head_size)
        num_splits, batch_size, seqlen, num_heads, head_size = out_partial.shape
    # 确定输出 dtype
    if out_dtype is None:
        out_dtype = out_partial.dtype
    # 未提供时创建输出
    device = out_partial.device
    if out is None:
        if is_varlen:
            out = torch.empty(total_q, num_heads, head_size, dtype=out_dtype, device=device)
        else:
            out = torch.empty(
                batch_size, seqlen, num_heads, head_size, dtype=out_dtype, device=device
            )
    # 仅在被要求时才创建 lse 输出
    if return_lse:
        if is_varlen:
            lse = torch.empty(num_heads, total_q, dtype=torch.float32, device=device)
        else:
            lse = torch.empty(batch_size, num_heads, seqlen, dtype=torch.float32, device=device)
        lse = lse.transpose(-1, -2)
    else:
        lse = None
    _flash_attn_fwd_combine(
        out_partial,
        lse_partial,
        out,
        lse,
        cu_seqlens,
        seqused,
        virtual_batch_idx=virtual_batch_idx,
    )
    return out, lse


def _get_scheduler_metadata(
    num_batch: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    nheads: int,
    nheads_kv: int,
    headdim: int,
    num_splits: int,
    tile_m: int,
    tile_n: int,
    headdim_v: Optional[int] = None,
    pack_gqa: Optional[bool] = False,
    q_stage: int = 1,
    cluster_shape_m: int = 1,
    causal: bool = False,
    enable_pdl: bool = False,
    sort: bool = False,
    seqlen_k_new: int = 0,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    cu_seqlens_k_new: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    leftpad_k: Optional[torch.Tensor] = None,
    seqlen_k_per_split: Optional[int] = None,
    zfill_padded_output: bool = True,
    total_q: Optional[int] = None,
    use_clc_scheduler: bool = False,
) -> SchedulerMetadataTensorsTorch:
    device = None
    for t in [cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k]:
        if t is not None:
            device = t.device
            break
    if device is None:
        raise ValueError(
            "At least one of cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k must be provided on device"
        )
    if headdim_v is None:
        headdim_v = headdim

    # 覆盖 enable_pdl（暂不支持）
    enable_pdl = False

    assert not sort, "LPT batch sort not yet implemented"

    if seqlen_k_per_split is not None:
        assert seqlen_k_per_split % tile_n == 0, "seqlen per split must be divisible by tile_n"
        n_blocks_per_split = seqlen_k_per_split // tile_n
        n_blocks_total = (max_seqlen_k + seqlen_k_new + tile_n - 1) // tile_n
        splits_needed = (n_blocks_total + n_blocks_per_split - 1) // n_blocks_per_split
        assert num_splits >= splits_needed, (
            f"seqlen_k_per_split={seqlen_k_per_split} needs num_splits>={splits_needed}, "
            f"got {num_splits}"
        )
    else:
        n_blocks_per_split = None

    is_split_kv = num_splits > 1
    needs_prepare_kernel = is_split_kv or causal or sort

    if needs_prepare_kernel:
        num_m_blocks = torch.empty(num_batch, dtype=torch.int32, device=device)
        num_splits_dynamic = torch.empty(num_batch, dtype=torch.int32, device=device)
        virtual_batch_idx = (
            torch.empty(num_batch, dtype=torch.int32, device=device) if sort else None
        )
        num_nheads_in_l2 = (
            torch.empty(num_batch, dtype=torch.int32, device=device) if causal else None
        )
        tile_count_semaphore = (
            torch.empty(1, dtype=torch.int32, device=device) if not use_clc_scheduler else None
        )

        num_warps = min((num_batch + 30) // 31, 32)
        num_warps = 1 << (num_warps - 1).bit_length()

        cache_key = (
            num_warps,
            tile_m,
            tile_n,
            nheads,
            nheads_kv,
            headdim,
            headdim_v,
            causal,
            pack_gqa,
            enable_pdl,
            sort,
            cu_seqlens_q is not None,
            cu_seqlens_k is not None,
            cu_seqlens_k_new is not None,
            seqused_q is not None,
            seqused_k is not None,
            leftpad_k is not None,
            num_m_blocks is not None,
            num_splits_dynamic is not None,
            virtual_batch_idx is not None,
            num_nheads_in_l2 is not None,
            tile_count_semaphore is not None,
            n_blocks_per_split is not None,
            zfill_padded_output,
        )

        if cache_key not in _get_scheduler_metadata.compile_cache:
            (
                num_m_blocks_cute,
                num_splits_dynamic_cute,
                virtual_batch_idx_cute,
                num_nheads_in_l2_cute,
                tile_count_semaphore_cute,
                cu_seqlens_q_cute,
                cu_seqlens_k_cute,
                cu_seqlens_k_new_cute,
                seqused_q_cute,
                seqused_k_cute,
                leftpad_k_cute,
            ) = [
                to_cute_tensor(t, assumed_align=4) if t is not None else None
                for t in (
                    num_m_blocks,
                    num_splits_dynamic,
                    virtual_batch_idx,
                    num_nheads_in_l2,
                    tile_count_semaphore,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    cu_seqlens_k_new,
                    seqused_q,
                    seqused_k,
                    leftpad_k,
                )
            ]
            scheduler = FlashPrepareScheduler(
                num_warps,
                tile_m,
                tile_n,
                nheads,
                nheads_kv,
                headdim,
                headdim_v,
                causal,
                packgqa=pack_gqa,
                sort=sort,
                zfill_padded_output=zfill_padded_output,
            )
            _get_scheduler_metadata.compile_cache[cache_key] = cute.compile(
                scheduler,
                max_seqlen_q,
                max_seqlen_k,
                seqlen_k_new,
                cu_seqlens_q_cute,
                cu_seqlens_k_cute,
                cu_seqlens_k_new_cute,
                seqused_q_cute,
                seqused_k_cute,
                leftpad_k_cute,
                num_batch,
                num_splits,
                tile_count_semaphore_cute,
                num_m_blocks_cute,
                num_splits_dynamic_cute,
                virtual_batch_idx_cute,
                num_nheads_in_l2_cute,
                n_blocks_per_split,
                cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                options="--enable-tvm-ffi",
            )

        if not is_fake_mode():
            _get_scheduler_metadata.compile_cache[cache_key](
                max_seqlen_q,
                max_seqlen_k,
                seqlen_k_new,
                cu_seqlens_q,
                cu_seqlens_k,
                cu_seqlens_k_new,
                seqused_q,
                seqused_k,
                leftpad_k,
                num_batch,
                num_splits,
                tile_count_semaphore,
                num_m_blocks,
                num_splits_dynamic,
                virtual_batch_idx,
                num_nheads_in_l2,
                n_blocks_per_split,
            )
    else:
        num_m_blocks = None
        num_splits_dynamic = None
        virtual_batch_idx = None
        num_nheads_in_l2 = None
        tile_count_semaphore = None

    qhead_per_kvhead = nheads // nheads_kv
    # 二分查找提示；仅当 batch 大小超过阈值时才被 single-tile scheduler 使用
    has_varlen_info = (
        cu_seqlens_q is not None or seqused_q is not None
    )
    needs_compute_tile_cumsum = (
        has_varlen_info
        and num_batch > BIN_BATCH_SEARCH_THRESH
        and tile_count_semaphore is None
    )
    if needs_compute_tile_cumsum:
        cu_total_m_blocks, cu_total_splits_m_blocks = _compute_tile_cumsum(
            num_m_blocks=num_m_blocks,
            cu_seqlens=cu_seqlens_q,
            seqused=seqused_q,
            num_splits_dynamic=num_splits_dynamic,
            virtual_batch_idx=virtual_batch_idx,
            tile_size=tile_m,
            q_stage=q_stage,
            cluster_shape_m=cluster_shape_m,
            qhead_per_kvhead=qhead_per_kvhead,
            pack_gqa=bool(pack_gqa),
        )
    else:
        cu_total_m_blocks, cu_total_splits_m_blocks = None, None

    blocks_to_batch_idx = None
    if USE_BLOCKS_TO_BATCH and cu_total_m_blocks is not None:
        blocks_to_batch_idx = _compute_blocks_to_batch(
            cu_total_m_blocks,
            _blocks_to_batch_size(
                total_q if total_q is not None else num_batch * max_seqlen_q,
                num_batch, tile_m, qhead_per_kvhead, pack_gqa,
            ),
            cu_total_m_blocks.device,
        )

    return SchedulerMetadataTensorsTorch(
        num_m_blocks_ptr=num_m_blocks,
        num_splits_dynamic_ptr=num_splits_dynamic,
        virtual_batch_idx_ptr=virtual_batch_idx,
        num_nheads_in_l2_ptr=num_nheads_in_l2,
        tile_count_semaphore=tile_count_semaphore,
        cu_total_m_blocks=cu_total_m_blocks,
        cu_total_splits_m_blocks=cu_total_splits_m_blocks,
        blocks_to_batch_idx=blocks_to_batch_idx,
    )


_get_scheduler_metadata.compile_cache = get_jit_cache("scheduler_metadata")


def get_scheduler_metadata(
    max_seqlen_q: int,
    max_seqlen_k: int,
    nheads: int,
    nheads_kv: int,
    headdim: int,
    num_splits: int,
    headdim_v: Optional[int] = None,
    pack_gqa: Optional[int] = None,
    causal: bool = False,
    window_size_left: Optional[int] = None,
    window_size_right: Optional[int] = None,
    seqlen_k_new: int = 0,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    cu_seqlens_k_new: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    leftpad_k: Optional[torch.Tensor] = None,
    seqlen_k_per_split: Optional[int] = None,
    _arch: Optional[int] = None,
) -> SchedulerMetadataTensorsTorch:
    """准备 varlen tile scheduler（SingleTileVarlenScheduler 与
    DynamicPersistentVarlenScheduler）所需的元数据张量。

    选中的参数说明：
        num_splits: prepare kernel 对每个 batch 条目最多可发出的 split 数量
        seqlen_k_per_split: 为在正向/反向之间实现逐位可复现，可固定每个 split 覆盖的
            精确 seqlen_k；num_splits 会据此推算。

    Returns
        SchedulerMetadataTensorsTorch，一个 named tuple，包含：
        - num_splits_dynamic_ptr: 每个 batch 的 num_splits
        - num_nheads_in_l2_ptr: 用于 head swizzle（打乱 head 顺序）以避免 L2 cache 抖动
        - tile_count_semaphore: DynamicPersistentVarlenScheduler 做原子自增用的全局信号量
        - cu_total_m_blocks: 统计总 m_blocks 数的 cumsum 张量，用于大批量下的二分 batch 查找
        - cu_total_splits_m_blocks: 与之互补的 cumsum 张量，用于二分 batch 查找，
            以及在没有 num_splits_dynamic_ptr 时提取动态 split 数量
    """
    arch = _get_device_arch() if _arch is None else _arch
    if headdim_v is None:
        headdim_v = headdim

    batch_sizes = {}
    if cu_seqlens_q is not None:
        batch_sizes["cu_seqlens_q"] = cu_seqlens_q.shape[0] - 1
    if cu_seqlens_k is not None:
        batch_sizes["cu_seqlens_k"] = cu_seqlens_k.shape[0] - 1
    if seqused_q is not None:
        batch_sizes["seqused_q"] = seqused_q.shape[0]
    if seqused_k is not None:
        batch_sizes["seqused_k"] = seqused_k.shape[0]
    assert batch_sizes, (
        "get_scheduler_metadata requires at least one of "
        "cu_seqlens_q/cu_seqlens_k/seqused_q/seqused_k"
    )
    num_batch = next(iter(batch_sizes.values()))
    assert all(b == num_batch for b in batch_sizes.values()), (
        f"inconsistent batch size across inputs: {batch_sizes}"
    )
    device = next(
        t.device for t in (cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k) if t is not None
    )

    causal, local, window_size_left, window_size_right = _resolve_causal_local_window(
        causal, window_size_left, window_size_right
    )

    qhead_per_kvhead = nheads // nheads_kv
    if pack_gqa is None:
        pack_gqa = qhead_per_kvhead > 1

    fwd_cfg = _get_fwd_config(
        arch=arch,
        head_dim=headdim,
        head_dim_v=headdim_v,
        causal=causal,
        local=local,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        qhead_per_kvhead=qhead_per_kvhead,
        pack_gqa=pack_gqa,
        batch_size=num_batch,
        num_head_kv=nheads_kv,
        num_splits=num_splits,
        device=device,
    )
    tile_m, tile_n = fwd_cfg.m_block_size, fwd_cfg.n_block_size
    q_stage = fwd_cfg.q_stage
    num_splits = fwd_cfg.num_splits

    return _get_scheduler_metadata(
        num_batch,
        max_seqlen_q,
        max_seqlen_k,
        nheads,
        nheads_kv,
        headdim,
        num_splits,
        tile_m,
        tile_n,
        headdim_v=headdim_v,
        pack_gqa=pack_gqa,
        q_stage=q_stage,
        causal=causal,
        enable_pdl=False,  # pdl 尚未启用
        sort=False,  # LPT batch 排序尚未启用
        seqlen_k_new=seqlen_k_new,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        cu_seqlens_k_new=cu_seqlens_k_new,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        leftpad_k=leftpad_k,
        seqlen_k_per_split=seqlen_k_per_split,
        zfill_padded_output=True,
        use_clc_scheduler=utils._get_use_clc_scheduler_default(),
    )
