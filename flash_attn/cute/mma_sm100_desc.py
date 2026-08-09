# Copyright (c) 2025, Tri Dao.
# 从 C++ 移植到 Python 的 Cutlass 代码：
# https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/mma_sm100_desc.hpp
# https://github.com/NVIDIA/cutlass/blob/main/include/cute/atom/mma_traits_sm100.hpp
# https://github.com/NVIDIA/cutlass/blob/main/include/cute/atom/mma_traits_sm100.hpp

from enum import IntEnum

import cutlass
import cutlass.cute as cute

# ---------------------------------------------------------------------------
# 与硬件编码一致的枚举（值必须保持完全一致）
# ---------------------------------------------------------------------------


class Major(IntEnum):  # 在 ISA 文档中称为矩阵 "layout"
    K = 0
    MN = 1


class ScaleIn(IntEnum):  # 取负标志
    One = 0
    Neg = 1


class Saturate(IntEnum):
    False_ = 0
    True_ = 1


class CFormat(IntEnum):  # 2 位字段（第 4-5 位）
    F16 = 0
    F32 = 1
    S32 = 2


class F16F32Format(IntEnum):  # 3 位字段（A/B 元素类型）
    F16 = 0
    BF16 = 1
    TF32 = 2


class S8Format(IntEnum):
    UINT8 = 0
    INT8 = 1


class MXF8F6F4Format(IntEnum):
    E4M3 = 0
    E5M2 = 1
    E2M3 = 3
    E3M2 = 4
    E2M1 = 5


class MaxShift(IntEnum):
    NoShift = 0
    MaxShift8 = 1
    MaxShift16 = 2
    MaxShift32 = 3


# ---------------------------------------------------------------------------
# CUTLASS 类型 → 编码辅助函数
# ---------------------------------------------------------------------------


def to_UMMA_format(cutlass_type) -> int:
    """
    把 CUTLASS 标量类映射到 Matrix A/B 的 3 位编码。
    """
    if cutlass_type is cutlass.Int8:
        return S8Format.INT8
    # 无符号 8 位（若你的 CUTLASS 构建支持）
    if cutlass_type is cutlass.Uint8:
        return S8Format.UINT8
    # FP-16 / BF-16
    if cutlass_type is cutlass.Float16:
        return F16F32Format.F16
    if cutlass_type is cutlass.BFloat16:
        return F16F32Format.BF16
    # TensorFloat-32（8 位指数、10 位尾数，打包在 19 位中）
    if cutlass_type is cutlass.TFloat32:
        return F16F32Format.TF32
    # Float-8 / Float-6 / Float-4 —— CUTLASS 暴露时再补充
    if cutlass_type is cutlass.Float8E4M3FN:
        return MXF8F6F4Format.E4M3
    if cutlass_type is cutlass.Float8E5M2:
        return MXF8F6F4Format.E5M2
    raise TypeError(f"Unsupported CUTLASS scalar type for A/B: {cutlass_type!r}")


def to_C_format(cutlass_type) -> int:
    """
    把 CUTLASS 标量类映射到 2 位累加器编码。
    """
    if cutlass_type is cutlass.Float16:
        return CFormat.F16
    if cutlass_type is cutlass.Float32:
        return CFormat.F32
    if cutlass_type is cutlass.Int32:
        return CFormat.S32
    raise TypeError(f"Unsupported CUTLASS scalar type for accumulator: {cutlass_type!r}")


# ---------------------------------------------------------------------------
# 构造函数 —— 只接受 CUTLASS 标量类
# ---------------------------------------------------------------------------


def make_instr_desc(
    a_type,  # CUTLASS 标量类，例如 cutlass.Int8
    b_type,
    c_type,
    M: int,  # 64、128 或 256
    N: int,  # 8 … 256（8 的倍数）
    a_major: Major,
    b_major: Major,
    a_neg: ScaleIn = ScaleIn.One,
    b_neg: ScaleIn = ScaleIn.One,
    c_sat: Saturate = Saturate.False_,
    is_sparse: bool = False,
    max_shift: MaxShift = MaxShift.NoShift,
) -> int:
    """
    构建 Blackwell MMA 的 32 位指令描述符。
    所有矩阵/累加器 **类型必须是 CUTLASS 标量类** ——
    禁止直接传整数。
    """
    # 讲解：MMA 指令描述符把 A/B/C 的元素格式、维度、布局（行主/列主）、
    # 饱和/取负等选项编码进一个 32 位字，UMMA 指令据此直接执行，
    # 无需每次 GEMM 都重新配置 Tensor Core。
    # --- 编码元素格式 -----------------------------------------------------------------
    a_fmt = int(to_UMMA_format(a_type))
    b_fmt = int(to_UMMA_format(b_type))
    c_fmt = int(to_C_format(c_type))

    # --- 对 M/N 做范围检查 -----------------------------------------------------------
    if M not in (64, 128, 256):
        raise ValueError("M must be 64, 128 or 256")
    if N < 8 or N > 256 or (N & 7):
        raise ValueError("N must be a multiple of 8 in the range 8…256")

    m_dim = M >> 4  # 5 位字段
    n_dim = N >> 3  # 6 位字段

    # fmt: off
    # --- pack the bit-fields -----------------------------------------------------
    desc = 0
    desc |= (0                 & 0x3) << 0        # sparse_id2（此处恒为 0）
    desc |= (int(is_sparse)    & 0x1) << 2        # sparse_flag
    desc |= (int(c_sat)        & 0x1) << 3        # saturate（饱和）
    desc |= (c_fmt             & 0x3) << 4        # c_format
    desc |= (a_fmt             & 0x7) << 7        # a_format
    desc |= (b_fmt             & 0x7) << 10       # b_format
    desc |= (int(a_neg)        & 0x1) << 13       # a_negate
    desc |= (int(b_neg)        & 0x1) << 14       # b_negate
    desc |= (int(a_major)      & 0x1) << 15       # a_major
    desc |= (int(b_major)      & 0x1) << 16       # b_major
    desc |= (n_dim             & 0x3F) << 17      # n_dim（6 位）
    desc |= (m_dim             & 0x1F) << 24      # m_dim（5 位）
    desc |= (int(max_shift)    & 0x3) << 30       # max_shift（2 位）
    # fmt: on

    return desc & 0xFFFF_FFFF  # 确保 32 位结果


def mma_op_to_idesc(op: cute.nvgpu.tcgen05.mma.MmaOp):
    return make_instr_desc(
        op.a_dtype,
        op.b_dtype,
        op.acc_dtype,
        op.shape_mnk[0],
        op.shape_mnk[1],
        Major.K if op.a_major_mode == cute.nvgpu.tcgen05.mma.OperandMajorMode.K else Major.MN,
        Major.K if op.b_major_mode == cute.nvgpu.tcgen05.mma.OperandMajorMode.K else Major.MN,
    )


class LayoutType(IntEnum):  # 占据最高的 3 位 [61:64)
    SWIZZLE_NONE = 0  # （旧文档中也称为 "INTERLEAVE"）
    SWIZZLE_128B_BASE32B = 1
    SWIZZLE_128B = 2
    SWIZZLE_64B = 4
    SWIZZLE_32B = 6
    # 值 3、5、7 对 UMMA 而言是保留/非法


# ---------------------------------------------------------------------------
#  辅助函数 —— 从张量布局确定 SWIZZLE_* 家族
# ---------------------------------------------------------------------------


def _layout_type(swizzle: cute.Swizzle) -> LayoutType:
    B, M, S = swizzle.num_bits, swizzle.num_base, swizzle.num_shift

    if M == 4:  # Swizzle<*,4,3>
        if S != 3:
            raise ValueError("Unexpected swizzle shift – want S==3 for M==4")
        return {
            0: LayoutType.SWIZZLE_NONE,
            1: LayoutType.SWIZZLE_32B,
            2: LayoutType.SWIZZLE_64B,
            3: LayoutType.SWIZZLE_128B,
        }[B]  # KeyError ⇒ B 非法 → 抛出异常
    if M == 5:  # Swizzle<2,5,2>（M==5 唯一合法的三元组）
        if (B, S) != (2, 2):
            raise ValueError("Only Swizzle<2,5,2> supported for 128B_BASE32B")
        return LayoutType.SWIZZLE_128B_BASE32B

    # 任何其它 (M,B,S) 三元组都不是 UMMA 合法的共享内存布局
    raise ValueError("Unsupported swizzle triple for UMMA smem descriptor")


def make_smem_desc_base(layout: cute.Layout, swizzle: cute.Swizzle, major: Major) -> int:
    """
    把 2-D *共享内存* Cute 布局转换为 Blackwell 64 位 smem 描述符，
    不含 smem 起始地址。
    layout 必须对应一个 uint128 张量的布局。
    """
    # ------------------------------------------------------------------ meta
    layout_type = _layout_type(swizzle)  # 解析 SWIZZLE_* 家族

    VERSION = 1  # 第 46-47 位
    LBO_MODE = 0  # 第 52 位
    BASE_OFFSET = 0  # 第 49-51 位（CUTLASS 恒为 0）

    # ---------------------------------------------------------- 步长（单位：uint128_t = 16 B）
    swizzle_atom_mn_size = {
        LayoutType.SWIZZLE_NONE: 1,
        LayoutType.SWIZZLE_32B: 2,
        LayoutType.SWIZZLE_64B: 4,
        LayoutType.SWIZZLE_128B: 8,
        LayoutType.SWIZZLE_128B_BASE32B: 8,
    }[layout_type]

    if major is Major.MN:
        swizzle_atom_k_size = 4 if layout_type is LayoutType.SWIZZLE_128B_BASE32B else 8
        canonical_layout = cute.logical_divide(layout, (swizzle_atom_mn_size, swizzle_atom_k_size))
        if not cute.is_congruent(canonical_layout, ((1, 1), (1, 1))):
            raise ValueError("Not a canonical UMMA_MN Layout: Expected profile failure.")
        stride_00 = canonical_layout.stride[0][0]
        if layout_type is not LayoutType.SWIZZLE_NONE and stride_00 != 1:
            raise ValueError("Not a canonical UMMA_MN Layout: Expected stride failure.")
        stride_10 = canonical_layout.stride[1][0]
        if stride_10 != swizzle_atom_mn_size:
            raise ValueError("Not a canonical UMMA_MN Layout: Expected stride failure.")
        stride_01, stride_11 = canonical_layout.stride[0][1], canonical_layout.stride[1][1]
        if layout_type is LayoutType.SWIZZLE_NONE:
            stride_byte_offset, leading_byte_offset = stride_01, stride_11
        else:
            stride_byte_offset, leading_byte_offset = stride_11, stride_01
    else:
        if layout_type == LayoutType.SWIZZLE_128B_BASE32B:
            raise ValueError("SWIZZLE_128B_BASE32B is invalid for Major-K")
        if not cute.size(layout.shape[0]) % 8 == 0:
            raise ValueError("Not a canonical UMMA_K Layout: Expected MN-size multiple of 8.")
        canonical_layout = cute.logical_divide(layout, (8, 2))
        if not cute.is_congruent(canonical_layout, ((1, 1), (1, 1))):
            raise ValueError("Not a canonical UMMA_K Layout: Expected profile failure.")
        stride_00 = canonical_layout.stride[0][0]
        if stride_00 != swizzle_atom_mn_size:
            raise ValueError("Not a canonical UMMA_K Layout: Expected stride failure.")
        stride_10 = canonical_layout.stride[1][0]
        if layout_type is not LayoutType.SWIZZLE_NONE and stride_10 != 1:
            raise ValueError("Not a canonical UMMA_K Layout: Expected stride failure.")
        stride_01 = canonical_layout.stride[0][1]
        stride_byte_offset, leading_byte_offset = stride_01, stride_10

    # ------------------------------------------------------------------ pack
    desc = 0
    # leading_byte_offset_  [16:30)
    desc |= (leading_byte_offset & 0x3FFF) << 16
    # stride_byte_offset_   [32:46)
    desc |= (stride_byte_offset & 0x3FFF) << 32
    # version_             [46:48)
    desc |= (VERSION & 0x3) << 46
    # base_offset_         [49:52)
    desc |= (BASE_OFFSET & 0x7) << 49
    # lbo_mode_            [52:53)
    desc |= (LBO_MODE & 0x1) << 52
    # layout_type_         [61:64)
    desc |= (int(layout_type) & 0x7) << 61

    return desc & 0xFFFF_FFFF_FFFF_FFFF  # 强制 64 位宽度


def make_smem_desc_start_addr(start_addr: cute.Pointer) -> cutlass.Int32:
    # 14 位，去掉 4 个最低有效位（desc 中的第 0-13 位）
    return (start_addr.toint() & 0x3FFFF) >> 4


def smem_desc_base_from_tensor(sA: cute.Tensor, major: Major) -> int:
    sA_swizzle = sA.iterator.type.swizzle_type
    return make_smem_desc_base(
        cute.recast_layout(128, sA.element_type.width, sA.layout[0]),
        sA_swizzle,
        major,
    )
