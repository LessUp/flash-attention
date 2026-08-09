# Copyright (c) 2025, Tri Dao.

import math
import hashlib
import inspect
import os
from functools import partial
from typing import Type, Callable, Optional, Tuple, overload, NamedTuple

import cutlass
import cutlass.cute as cute

from cutlass import Float32, Int32, const_expr
from cutlass.cute import FastDivmodDivisor
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import nvvm, llvm
from cutlass.cute.runtime import from_dlpack


import quack.activation

_MIXER_ATTRS = ("__vec_size__",)


class AuxData(NamedTuple):
    tensors: tuple | list | None = None
    scalars: tuple | None = None


# Obtained from sollya:
# fpminimax(exp(x * log(2.0)), 1, [|1,24...|],[0;1],relative);
POLY_EX2 = {
    0: (1.0),
    1: (
        1.0,
        0.922497093677520751953125,
    ),
    2: (
        1.0,
        0.6657850742340087890625,
        0.330107033252716064453125,
    ),
    3: (
        1.0,
        0.695146143436431884765625,
        0.227564394474029541015625,
        0.077119089663028717041015625,
    ),
    4: (
        1.0,
        0.693042695522308349609375,
        0.2412912547588348388671875,
        5.2225358784198760986328125e-2,
        1.3434938155114650726318359375e-2,
    ),
    5: (
        1.0,
        0.693151414394378662109375,
        0.24016360938549041748046875,
        5.5802188813686370849609375e-2,
        9.01452265679836273193359375e-3,
        1.86810153536498546600341796875e-3,
    ),
}

_fa_clc_enabled: bool = os.environ.get("FA_CLC", "0") == "1"
_fa_disable_2cta_enabled: bool = os.environ.get("FA_DISABLE_2CTA", "0") == "1"


def _is_cuda_12() -> bool:
    """检查 CUDA 工具包版本是否为 12.x。
    
    2CTA 前向非因果模式在 CUDA 12 上有代码生成回归，导致比 1CTA
    慢约 18%。该问题在 CUDA 13.x 中已修复。
    """
    try:
        import torch

        cuda_version = torch.version.cuda
        if cuda_version is not None:
            major = cuda_version.split(".")[0]
            return int(major) == 12
    except Exception:
        pass
    return False


_fa_disable_2cta_cuda12: bool = _is_cuda_12()


def _get_use_clc_scheduler_default() -> bool:
    return _fa_clc_enabled


def _get_disable_2cta_default(is_fwd: bool = False) -> bool:
    if is_fwd:
        return _fa_disable_2cta_enabled or _fa_disable_2cta_cuda12
    else:
        return _fa_disable_2cta_enabled


def _compute_base_hash(func: Callable) -> str:
    """根据源码或字节码以及闭包值计算哈希。"""
    try:
        data = inspect.getsource(func).encode()
    except (OSError, TypeError):
        if hasattr(func, "__code__") and func.__code__ is not None:
            data = func.__code__.co_code
        else:
            data = repr(func).encode()

    hasher = hashlib.sha256(data)

    if hasattr(func, "__closure__") and func.__closure__ is not None:
        for cell in func.__closure__:
            hasher.update(repr(cell.cell_contents).encode())

    return hasher.hexdigest()


def hash_callable(
    func: Callable, mixer_attrs: Tuple[str] = _MIXER_ATTRS, set_cute_hash: bool = True
) -> str:
    """基于源码或字节码以及闭包值对可调用对象做哈希。
    快速路径：若可调用对象（或其 __wrapped__ 基础）带有 ``__cute_hash__``
    属性，则直接返回该值作为基础哈希，再混入元数据 dunder 属性
    生成最终的字典键哈希。
    set_cute_hash: 是否设置 func.__cute_hash__
    """
    # 解析基础哈希
    if hasattr(func, "__cute_hash__"):
        base_hash = func.__cute_hash__
    else:
        # 解开装饰函数（例如 cute.jit 包装器）。
        base_func = getattr(func, "__wrapped__", func)

        if hasattr(base_func, "__cute_hash__"):
            base_hash = base_func.__cute_hash__
        else:
            base_hash = _compute_base_hash(base_func)

            if set_cute_hash:
                base_func.__cute_hash__ = base_hash

    # 混入可变元数据 dunder 属性
    mixer_values = tuple(getattr(func, attr, None) for attr in mixer_attrs)

    if all(v is None for v in mixer_values):
        return base_hash

    hasher = hashlib.sha256(base_hash.encode())

    for attr, val in zip(mixer_attrs, mixer_values):
        hasher.update(f"{attr}={val!r}".encode())

    return hasher.hexdigest()


def create_softcap_scoremod(softcap_val):
    @cute.jit
    def scoremod_premask_fn(
        acc_S_SSA, batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors
    ):
        scores = acc_S_SSA / softcap_val
        return softcap_val * cute.math.tanh(scores, fastmath=True)

    return scoremod_premask_fn


def create_softcap_scoremod_bwd(softcap_val):
    @cute.jit
    def scoremod_bwd_fn(
        grad_out_SSA, score_SSA, batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors
    ):
        scores = score_SSA / softcap_val
        tanh_scores = cute.math.tanh(scores, fastmath=True)
        return grad_out_SSA * (1.0 - tanh_scores * tanh_scores)

    return scoremod_bwd_fn


LOG2_E = math.log2(math.e)


def compute_softmax_scale_log2(softmax_scale, score_mod):
    """计算 softmax_scale_log2 和调整后的 softmax_scale，取决于是否使用 score_mod。
    
    当 score_mod 为 None 时，把 log2(e) 因子折进 softmax_scale_log2，并把
    softmax_scale 设为 None。当存在 score_mod 时，保持 softmax_scale 独立，
    以便在 score_mod 之前应用，并把 softmax_scale_log2 只设为换底常数。
    
    Returns (softmax_scale_log2, softmax_scale).
    """
    if const_expr(score_mod is None):
        return softmax_scale * LOG2_E, None
    else:
        return LOG2_E, softmax_scale


def compute_fastdiv_mods(mQ, mK, qhead_per_kvhead, pack_gqa, aux_tensors, mPageTable=None):
    """计算用于 aux_tensors 索引计算的 FastDivmodDivisor 对。
    
    Returns 一个 (seqlen_q_divmod, seqlen_k_divmod) 元组；若 aux_tensors 为 None 则返回 None。
    """
    if const_expr(aux_tensors is None):
        return None
    seqlen_q = cute.size(mQ.shape[0]) // (qhead_per_kvhead if const_expr(pack_gqa) else 1)
    seqlen_k = (
        cute.size(mK.shape[0])
        if const_expr(mPageTable is None)
        else mK.shape[0] * mPageTable.shape[1]
    )
    return (FastDivmodDivisor(seqlen_q), FastDivmodDivisor(seqlen_k))


def convert_from_dlpack(x, leading_dim, alignment=16, divisibility=1) -> cute.Tensor:
    return (
        from_dlpack(x, assumed_align=alignment)
        .mark_layout_dynamic(leading_dim=leading_dim)
        .mark_compact_shape_dynamic(
            mode=leading_dim, stride_order=x.dim_order(), divisibility=divisibility
        )
    )


def convert_from_dlpack_compact_dynamic(
    x,
    *,
    dynamic_modes: tuple[int, ...],
    alignment: int = 16,
    stride_order=None,
    divisibility: int = 1,
    enable_tvm_ffi: bool = False,
) -> cute.Tensor:
    """通过 DLPack 转换，并把选中的紧凑维标记为动态。"""
    if isinstance(dynamic_modes, int):
        dynamic_modes = (dynamic_modes,)
    if stride_order is None:
        stride_order = x.dim_order()
    t = (
        from_dlpack(x, assumed_align=alignment, enable_tvm_ffi=True)
        if enable_tvm_ffi
        else from_dlpack(x, assumed_align=alignment)
    )
    for mode in dynamic_modes:
        t = t.mark_compact_shape_dynamic(
            mode=mode,
            stride_order=stride_order,
            divisibility=divisibility,
        )
    return t


def convert_from_dlpack_leading_static(
    x, leading_dim, alignment=16, static_modes=None, stride_order=None
) -> cute.Tensor:
    if stride_order is None:
        stride_order = x.dim_order()
    x_ = from_dlpack(x, assumed_align=alignment)
    for i in range(x.ndim):
        if i != leading_dim and (static_modes is None or i not in static_modes):
            x_ = x_.mark_compact_shape_dynamic(mode=i, stride_order=stride_order)
    return x_


def make_tiled_copy_A(
    copy_atom: cute.CopyAtom, tiled_mma: cute.TiledMma, swapAB: cutlass.Constexpr[bool] = False
) -> cute.TiledCopy:
    if const_expr(swapAB):
        return cute.make_tiled_copy_B(copy_atom, tiled_mma)
    else:
        return cute.make_tiled_copy_A(copy_atom, tiled_mma)


def make_tiled_copy_B(
    copy_atom: cute.CopyAtom, tiled_mma: cute.TiledMma, swapAB: cutlass.Constexpr[bool] = False
) -> cute.TiledCopy:
    if const_expr(swapAB):
        return cute.make_tiled_copy_A(copy_atom, tiled_mma)
    else:
        return cute.make_tiled_copy_B(copy_atom, tiled_mma)


def mma_make_fragment_A(
    smem: cute.Tensor, thr_mma: cute.ThrMma, swapAB: cutlass.Constexpr[bool] = False
) -> cute.Tensor:
    if const_expr(swapAB):
        return mma_make_fragment_B(smem, thr_mma)
    else:
        return thr_mma.make_fragment_A(thr_mma.partition_A(smem))


def mma_make_fragment_B(
    smem: cute.Tensor, thr_mma: cute.ThrMma, swapAB: cutlass.Constexpr[bool] = False
) -> cute.Tensor:
    if const_expr(swapAB):
        return mma_make_fragment_A(smem, thr_mma)
    else:
        return thr_mma.make_fragment_B(thr_mma.partition_B(smem))


def get_smem_store_atom(
    arch: cutlass.Constexpr[int], element_type: Type[cute.Numeric], transpose: bool = False
) -> cute.CopyAtom:
    if const_expr(arch < 90 or element_type.width != 16):
        return cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            element_type,
            num_bits_per_copy=2 * element_type.width,
        )
    else:
        return cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=transpose, num_matrices=4),
            element_type,
        )


@cute.jit
def warp_reduce(
    val: cute.TensorSSA | cute.Numeric,
    op: Callable,
    width: cutlass.Constexpr[int] = cute.arch.WARP_SIZE,
) -> cute.TensorSSA | cute.Numeric:
    if const_expr(isinstance(val, cute.TensorSSA)):
        res = cute.make_rmem_tensor(val.shape, val.dtype)
        res.store(val)
        for i in cutlass.range_constexpr(cute.size(val.shape)):
            res[i] = warp_reduce(res[i], op, width)
        return res.load()
    else:
        for i in cutlass.range_constexpr(int(math.log2(width))):
            val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


@dsl_user_op
def smid(*, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [],
            "mov.u32 $0, %smid;",
            "=r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def fmax(
    a: float | Float32, b: float | Float32, c: float | Float32 | None = None, *, loc=None, ip=None
) -> Float32:
    return Float32(
        nvvm.fmax(
            Float32(a).ir_value(loc=loc, ip=ip),
            Float32(b).ir_value(loc=loc, ip=ip),
            c=Float32(c).ir_value(loc=loc, ip=ip) if c is not None else None,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def fmax_reduce(
    x: cute.TensorSSA, init_val: float | Float32 | None = None, arch: cutlass.Constexpr[int] = 80
) -> Float32:
    if const_expr(arch < 100 or cute.size(x.shape) % 8 != 0):
        # if const_expr(init_val is None):
        #     init_val = -cutlass.Float32.if
        # return x.reduce(cute.ReductionOp.MAX, init_val, 0)
        res = cute.make_rmem_tensor(x.shape, Float32)
        res.store(x)
        # local_max = [res[0], res[1]]
        # for i in cutlass.range_constexpr(2, cute.size(x.shape), 2):
        #     local_max[0] = fmax(local_max[0], res[i + 0])
        #     local_max[1] = fmax(local_max[1], res[i + 1])
        # local_max[0] = fmax(local_max[0], local_max[1])
        # return local_max[0] if const_expr(init_val is None) else fmax(local_max[0], init_val)
        local_max = [res[0], res[1], res[2], res[3]]
        for i in cutlass.range_constexpr(4, cute.size(x.shape), 4):
            local_max[0] = fmax(local_max[0], res[i + 0])
            local_max[1] = fmax(local_max[1], res[i + 1])
            local_max[2] = fmax(local_max[2], res[i + 2])
            local_max[3] = fmax(local_max[3], res[i + 3])
        local_max[0] = fmax(local_max[0], local_max[1])
        local_max[2] = fmax(local_max[2], local_max[3])
        local_max[0] = fmax(local_max[0], local_max[2])
        return local_max[0] if const_expr(init_val is None) else fmax(local_max[0], init_val)
    else:
        # [2025-06-15] x.reduce 似乎只用了 50% 的三输入 max 和 50% 的二输入 max
        # 这里改为强制使用三输入 max。
        res = cute.make_rmem_tensor(x.shape, Float32)
        res.store(x)
        local_max_0 = (
            fmax(init_val, res[0], res[1])
            if const_expr(init_val is not None)
            else fmax(res[0], res[1])
        )
        local_max = [
            local_max_0,
            fmax(res[2], res[3]),
            fmax(res[4], res[5]),
            fmax(res[6], res[7]),
        ]
        for i in cutlass.range_constexpr(8, cute.size(x.shape), 8):
            local_max[0] = fmax(local_max[0], res[i], res[i + 1])
            local_max[1] = fmax(local_max[1], res[i + 2], res[i + 3])
            local_max[2] = fmax(local_max[2], res[i + 4], res[i + 5])
            local_max[3] = fmax(local_max[3], res[i + 6], res[i + 7])
        local_max[0] = fmax(local_max[0], local_max[1])
        return fmax(local_max[0], local_max[2], local_max[3])


@cute.jit
def fadd_reduce(
    x: cute.TensorSSA, init_val: float | Float32 | None = None, arch: cutlass.Constexpr[int] = 80
) -> Float32:
    if const_expr(arch < 100 or cute.size(x.shape) % 8 != 0):
        if const_expr(init_val is None):
            init_val = Float32.zero
        return x.reduce(cute.ReductionOp.ADD, init_val, 0)
        # res = cute.make_rmem_tensor(x.shape, Float32)
        # res.store(x)
        # local_sum = [res[0], res[1], res[2], res[3]]
        # for i in cutlass.range_constexpr(4, cute.size(x.shape), 4):
        #     local_sum[0] += res[i + 0]
        #     local_sum[1] += res[i + 1]
        #     local_sum[2] += res[i + 2]
        #     local_sum[3] += res[i + 3]
        # local_sum[0] += local_sum[1]
        # local_sum[2] += local_sum[3]
        # local_sum[0] += local_sum[2]
        # return local_sum[0] if const_expr(init_val is None) else local_sum[0] + init_val
    else:
        res = cute.make_rmem_tensor(x.shape, Float32)
        res.store(x)
        local_sum_0 = (
            cute.arch.add_packed_f32x2((init_val, 0.0), (res[0], res[1]))
            # cute.arch.add_packed_f32x2((init_val / 2, init_val / 2), (res[0], res[1]))
            if const_expr(init_val is not None)
            else (res[0], res[1])
        )
        local_sum = [local_sum_0, (res[2], res[3]), (res[4], res[5]), (res[6], res[7])]
        for i in cutlass.range_constexpr(8, cute.size(x.shape), 8):
            local_sum[0] = cute.arch.add_packed_f32x2(local_sum[0], (res[i + 0], res[i + 1]))
            local_sum[1] = cute.arch.add_packed_f32x2(local_sum[1], (res[i + 2], res[i + 3]))
            local_sum[2] = cute.arch.add_packed_f32x2(local_sum[2], (res[i + 4], res[i + 5]))
            local_sum[3] = cute.arch.add_packed_f32x2(local_sum[3], (res[i + 6], res[i + 7]))
        local_sum[0] = cute.arch.add_packed_f32x2(local_sum[0], local_sum[1])
        local_sum[2] = cute.arch.add_packed_f32x2(local_sum[2], local_sum[3])
        local_sum[0] = cute.arch.add_packed_f32x2(local_sum[0], local_sum[2])
        return local_sum[0][0] + local_sum[0][1]


@dsl_user_op
def atomic_add_i32(a: int | Int32, ptr: cute.Pointer, *, loc=None, ip=None) -> Int32:
    return Int32(
        nvvm.atomicrmw(
            op=nvvm.AtomicOpKind.ADD,
            ptr=ptr.llvm_ptr,
            a=Int32(a).ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def atomic_add_fp32(a: float | Float32, gmem_ptr: cute.Pointer, *, loc=None, ip=None) -> None:
    # gmem_ptr_i64 = gmem_ptr.toint(loc=loc, ip=ip).ir_value()
    # # cache_hint = cutlass.Int64(0x12F0000000000000)
    # llvm.inline_asm(
    #     None,
    #     [gmem_ptr_i64, Float32(a).ir_value(loc=loc, ip=ip)],
    #     # [gmem_ptr_i64, Float32(a).ir_value(loc=loc, ip=ip), cache_hint.ir_value()],
    #     "red.global.add.f32 [$0], $1;",
    #     # "red.global.add.L2::cache_hint.f32 [$0], $1, 0x12F0000000000000;",
    #     # "red.global.add.L2::cache_hint.f32 [$0], $1, $2;",
    #     "l,f",
    #     # "l,f,l",
    #     has_side_effects=True,
    #     is_align_stack=False,
    #     asm_dialect=llvm.AsmDialect.AD_ATT,
    # )
    nvvm.atomicrmw(op=nvvm.AtomicOpKind.FADD, ptr=gmem_ptr.llvm_ptr, a=Float32(a).ir_value())


@dsl_user_op
def elem_pointer(x: cute.Tensor, coord: cute.Coord, *, loc=None, ip=None) -> cute.Pointer:
    return x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)


@cute.jit
def predicate_k(tAcA: cute.Tensor, limit: cutlass.Int32) -> cute.Tensor:
    # 只为 "k" 维计算谓词（predicate）；对 mn 维将使用 "if"
    tApA = cute.make_rmem_tensor(
        cute.make_layout(
            (cute.size(tAcA, mode=[0, 1]), cute.size(tAcA, mode=[1]), cute.size(tAcA, mode=[2])),
            stride=(cute.size(tAcA, mode=[2]), 0, 1),
        ),
        cutlass.Boolean,
    )
    for rest_v in cutlass.range_constexpr(tApA.shape[0]):
        for rest_k in cutlass.range_constexpr(tApA.shape[2]):
            tApA[rest_v, 0, rest_k] = cute.elem_less(tAcA[(0, rest_v), 0, rest_k][1], limit)
    return tApA


def canonical_warp_group_idx(sync: bool = True) -> cutlass.Int32:
    warp_group_idx = cute.arch.thread_idx()[0] // 128
    if const_expr(sync):
        warp_group_idx = cute.arch.make_warp_uniform(warp_group_idx)
    return warp_group_idx


# @dsl_user_op
# def warp_vote_any_lt(a: float | Float32, b: float | Float32, *, loc=None, ip=None) -> cutlass.Boolean:
#     mask = cutlass.Int32(-1)
#     return cutlass.Boolean(
#         llvm.inline_asm(
#             T.i32(),
#             [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip), mask.ir_value(loc=loc, ip=ip)],
#             ".pred p1, p2;\n"
#             "setp.lt.f32 p1, $1, $2;\n"
#             "vote.sync.any.pred p2, p1, $3;\n"
#             "selp.u32 $0, 1, 0, p2;",
#             # "selp.u32 $0, 1, 0, p1;",
#             "=r,f,f,r",
#             has_side_effects=False,
#             is_align_stack=False,
#             asm_dialect=llvm.AsmDialect.AD_ATT,
#         )
#     )


@cute.jit
def shuffle_sync(
    value: cute.Numeric,
    offset: cute.typing.Int,
    width: cutlass.Constexpr[int] = cute.arch.WARP_SIZE,
) -> cute.Numeric:
    assert value.width % 32 == 0, "value type must be a multiple of 32 bits"
    # 1 -> 0b11111, 2 -> 0b11110, 4 -> 0b11100, 8 -> 0b11000, 16 -> 0b10000, 32 -> 0b00000
    mask = cute.arch.WARP_SIZE - width
    clamp = cute.arch.WARP_SIZE - 1
    mask_and_clamp = mask << 8 | clamp
    # 重要：recast_tensor 要正常工作，步长需要是 1 而不是 0
    val = cute.make_rmem_tensor(cute.make_layout((1,), stride=(1,)), type(value))
    val[0] = value
    val_i32 = cute.recast_tensor(val, cutlass.Int32)
    for i in cutlass.range_constexpr(cute.size(val_i32)):
        val_i32[i] = cute.arch.shuffle_sync(val_i32[i], offset, mask_and_clamp=mask_and_clamp)
    return val[0]


@dsl_user_op
def shl_u32(val: cutlass.Uint32, shift: cutlass.Uint32, *, loc=None, ip=None) -> cutlass.Uint32:
    """
    用 PTX shl.b32（与符号无关）把 val 左移 shift 位。
    
    命名为 ``shl_u32``（而非 ``shl_b32``），因为 Python 类型标注区分有符号/无符号。
    
    PTX 语义（§9.7.8.8）："大于寄存器宽度 N 的移位量被钳制到 N。"
    因此 ``shl.b32 d, a, 32`` 是有明确定义的，结果为 0。
    
    这与 C/C++ 和 LLVM IR 不同：在它们中，移位量 >= 类型宽度是未定义行为。
    CuTeDSL 通过 MLIR -> LLVM IR 编译，因此普通的 Python 级
    ``Uint32(x) << Uint32(n)`` 继承了 LLVM 的 UB：优化器可能把结果当作
    poison 值并消除相关代码。内联 PTX 完全绕开 LLVM IR 的移位 ——
    指令原样发射到 PTX 中，其中的钳制对所有移位量都是安全的。
    """
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                cutlass.Uint32(val).ir_value(loc=loc, ip=ip),
                cutlass.Uint32(shift).ir_value(loc=loc, ip=ip),
            ],
            "shl.b32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def shr_u32(val: cutlass.Uint32, shift: cutlass.Uint32, *, loc=None, ip=None) -> cutlass.Uint32:
    """
    用 PTX shr.u32（零填充）把 val 无符号右移 shift 位。
    
    为什么用内联 PTX 而非普通 CuTeDSL 移位运算符，参见 ``shl_u32`` 的
    docstring（LLVM 按类型位宽移位的 UB 问题）。
    """
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                cutlass.Uint32(val).ir_value(loc=loc, ip=ip),
                cutlass.Uint32(shift).ir_value(loc=loc, ip=ip),
            ],
            "shr.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def warp_prefix_sum(val: cutlass.Int32, lane: Optional[cutlass.Int32] = None) -> cutlass.Int32:
    if const_expr(lane is None):
        lane = cute.arch.lane_idx()
    # if cute.arch.thread_idx()[0] >= 128 and cute.arch.thread_idx()[0] < 128 + 32 and cute.arch.block_idx()[0] == 0: cute.printf("tidx = %d, val = %d", cute.arch.thread_idx()[0] % 32, val)
    for i in cutlass.range_constexpr(int(math.log2(cute.arch.WARP_SIZE))):
        offset = 1 << i
        # 非常重要的是把 mask_and_clamp 设为 0
        partial_sum = cute.arch.shuffle_sync_up(val, offset=offset, mask_and_clamp=0)
        if lane >= offset:
            val += partial_sum
        # if cute.arch.thread_idx()[0] >= 128 and cute.arch.thread_idx()[0] < 128 + 32 and cute.arch.block_idx()[0] == 0: cute.printf("tidx = %d, partial_sum = %d, val = %d", cute.arch.thread_idx()[0] % 32, partial_sum, val)
    return val


@dsl_user_op
def cvt_f16x2_f32(
    a: float | Float32, b: float | Float32, to_dtype: Type, *, loc=None, ip=None
) -> cutlass.Int32:
    assert to_dtype in [cutlass.BFloat16, cutlass.Float16], "to_dtype must be BFloat16 or Float16"
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            f"cvt.rn.{'bf16x2' if to_dtype is cutlass.BFloat16 else 'f16x2'}.f32 $0, $2, $1;",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@overload
def cvt_f16(src: cute.Tensor, dst: cute.Tensor) -> None: ...


@overload
def cvt_f16(src: cute.Tensor, dtype: Type[cute.Numeric]) -> cute.Tensor: ...


@cute.jit
def cvt_f16(src: cute.Tensor, dst_or_dtype):
    """把 Float32 张量转换为 Float16/BFloat16。
    
    Args:
        src: 元素类型为 Float32 的源张量
        dst_or_dtype: 目标张量或目标 dtype（Float16/BFloat16）
    
    Returns:
        若 dst 是张量则返回 None；若提供 dtype 则返回新张量
    """
    if const_expr(isinstance(dst_or_dtype, type)):
        # dtype 变体：创建新张量并调用张量变体
        dtype = dst_or_dtype
        dst = cute.make_rmem_tensor(src.shape, dtype)
        cvt_f16(src, dst)
        return dst
    else:
        # 张量变体：写入 dst
        dst = dst_or_dtype
        assert cute.size(dst.shape) == cute.size(src.shape), "dst and src must have the same size"
        assert cute.size(src.shape) % 2 == 0, "src must have an even number of elements"
        assert dst.element_type in [cutlass.BFloat16, cutlass.Float16], (
            "dst must be BFloat16 or Float16"
        )
        assert src.element_type is Float32, "src must be Float32"
        dst_i32 = cute.recast_tensor(dst, cutlass.Int32)
        assert cute.size(dst_i32.shape) * 2 == cute.size(src.shape)
        for i in cutlass.range_constexpr(cute.size(dst_i32)):
            dst_i32[i] = cvt_f16x2_f32(src[2 * i], src[2 * i + 1], dst.element_type)


@dsl_user_op
@cute.jit
def evaluate_polynomial(x: Float32, poly: Tuple[Float32, ...], *, loc=None, ip=None) -> Float32:
    deg = len(poly) - 1
    out = poly[deg]
    for i in cutlass.range_constexpr(deg - 1, -1, -1):
        out = out * x + poly[i]
    return out


@dsl_user_op
@cute.jit
def evaluate_polynomial_2(
    x: Float32, y: Float32, poly: Tuple[Float32, ...], *, loc=None, ip=None
) -> Tuple[Float32, Float32]:
    deg = len(poly) - 1
    out = (poly[deg], poly[deg])
    for i in cutlass.range_constexpr(deg - 1, -1, -1):
        out = cute.arch.fma_packed_f32x2(out, (x, y), (poly[i], poly[i]))
    return out


@dsl_user_op
def add_round_down(x: float | Float32, y: float | Float32, *, loc=None, ip=None) -> Float32:
    # 可能可以调用 llvm 或 nvvm 来做这件事，而不是用 ptx
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(x).ir_value(loc=loc, ip=ip), Float32(y).ir_value(loc=loc, ip=ip)],
            "add.rm.ftz.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def combine_int_frac_ex2(x_rounded: Float32, frac_ex2: Float32, *, loc=None, ip=None) -> Float32:
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Float32(x_rounded).ir_value(loc=loc, ip=ip),
                Float32(frac_ex2).ir_value(loc=loc, ip=ip),
            ],
            "{\n\t"
            ".reg .s32 x_rounded_i, frac_ex_i, x_rounded_e, out_i;\n\t"
            "mov.b32 x_rounded_i, $1;\n\t"
            "mov.b32 frac_ex_i, $2;\n\t"
            "shl.b32 x_rounded_e, x_rounded_i, 23;\n\t"
            # add.u32 生成 IMAD 指令，add.s32 生成 LEA 指令
            # 据我所知，IMAD 走 FMA 流水线，LEA 走 ALU 流水线
            "add.s32 out_i, x_rounded_e, frac_ex_i;\n\t"
            "mov.b32 $0, out_i;\n\t"
            "}\n",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def ex2_emulation(x: Float32, *, poly_degree: int = 3, loc=None, ip=None) -> Float32:
    assert poly_degree in POLY_EX2, f"Polynomial degree {poly_degree} not supported"
    # 假设 x <= 127.0
    fp32_round_int = float(2**23 + 2**22)
    x_clamped = cute.arch.fmax(x, -127.0)
    # 这里要向下取整，使小数部分落在 [0, 1)
    x_rounded = add_round_down(x_clamped, fp32_round_int, loc=loc, ip=ip)
    # x 的整数下取整现在位于 x_rounded 的低 8 位
    # 假设接下来的 2 个操作四舍五入到最近的偶数。舍入模式很重要。
    x_rounded_back = x_rounded - fp32_round_int
    x_frac = x_clamped - x_rounded_back
    x_frac_ex2 = evaluate_polynomial(x_frac, POLY_EX2[poly_degree], loc=loc, ip=ip)
    return combine_int_frac_ex2(x_rounded, x_frac_ex2, loc=loc, ip=ip)


# TODO: 检查 ex2_emulation_2 生成的 SASS 是否与 ptx 版本相同
@dsl_user_op
def ex2_emulation_2(
    x: Float32, y: Float32, *, poly_degree: int = 3, loc=None, ip=None
) -> Tuple[Float32, Float32]:
    # 假设 x <= 127.0 且 y <= 127.0
    fp32_round_int = float(2**23 + 2**22)
    xy_clamped = (cute.arch.fmax(x, -127.0), cute.arch.fmax(y, -127.0))
    # 这里要向下取整，使小数部分落在 [0, 1)
    xy_rounded = cute.arch.add_packed_f32x2(xy_clamped, (fp32_round_int, fp32_round_int), rnd="rm")
    # x 和 y 的整数下取整现在位于 xy_rounded 的低 8 位
    # 希望接下来的 2 个操作四舍五入到最近的偶数。舍入模式很重要。
    xy_rounded_back = quack.activation.sub_packed_f32x2(
        xy_rounded, (fp32_round_int, fp32_round_int)
    )
    xy_frac = quack.activation.sub_packed_f32x2(xy_clamped, xy_rounded_back)
    xy_frac_ex2 = evaluate_polynomial_2(*xy_frac, POLY_EX2[poly_degree], loc=loc, ip=ip)
    x_out = combine_int_frac_ex2(xy_rounded[0], xy_frac_ex2[0], loc=loc, ip=ip)
    y_out = combine_int_frac_ex2(xy_rounded[1], xy_frac_ex2[1], loc=loc, ip=ip)
    return x_out, y_out


@dsl_user_op
def e2e_asm2(x: Float32, y: Float32, *, loc=None, ip=None) -> Tuple[Float32, Float32]:
    out_f32x2 = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Float32(x).ir_value(loc=loc, ip=ip), Float32(y, loc=loc, ip=ip).ir_value()],
        "{\n\t"
        ".reg .f32 f1, f2, f3, f4, f5, f6, f7;\n\t"
        ".reg .b64 l1, l2, l3, l4, l5, l6, l7, l8, l9, l10;\n\t"
        ".reg .s32 r1, r2, r3, r4, r5, r6, r7, r8;\n\t"
        "max.ftz.f32 f1, $2, 0fC2FE0000;\n\t"
        "max.ftz.f32 f2, $3, 0fC2FE0000;\n\t"
        "mov.b64 l1, {f1, f2};\n\t"
        "mov.f32 f3, 0f4B400000;\n\t"
        "mov.b64 l2, {f3, f3};\n\t"
        "add.rm.ftz.f32x2 l7, l1, l2;\n\t"
        "sub.rn.ftz.f32x2 l8, l7, l2;\n\t"
        "sub.rn.ftz.f32x2 l9, l1, l8;\n\t"
        "mov.f32 f7, 0f3D9DF09D;\n\t"
        "mov.b64 l6, {f7, f7};\n\t"
        "mov.f32 f6, 0f3E6906A4;\n\t"
        "mov.b64 l5, {f6, f6};\n\t"
        "mov.f32 f5, 0f3F31F519;\n\t"
        "mov.b64 l4, {f5, f5};\n\t"
        "mov.f32 f4, 0f3F800000;\n\t"
        "mov.b64 l3, {f4, f4};\n\t"
        "fma.rn.ftz.f32x2 l10, l9, l6, l5;\n\t"
        "fma.rn.ftz.f32x2 l10, l10, l9, l4;\n\t"
        "fma.rn.ftz.f32x2 l10, l10, l9, l3;\n\t"
        "mov.b64 {r1, r2}, l7;\n\t"
        "mov.b64 {r3, r4}, l10;\n\t"
        "shl.b32 r5, r1, 23;\n\t"
        "add.s32 r7, r5, r3;\n\t"
        "shl.b32 r6, r2, 23;\n\t"
        "add.s32 r8, r6, r4;\n\t"
        "mov.b32 $0, r7;\n\t"
        "mov.b32 $1, r8;\n\t"
        "}\n",
        "=r,=r,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    out0 = Float32(llvm.extractvalue(T.f32(), out_f32x2, [0], loc=loc, ip=ip))
    out1 = Float32(llvm.extractvalue(T.f32(), out_f32x2, [1], loc=loc, ip=ip))
    return out0, out1


@dsl_user_op
def domain_offset_aligned(
    coord: cute.Coord, tensor: cute.Tensor, *, loc=None, ip=None
) -> cute.Tensor:
    assert isinstance(tensor.iterator, cute.Pointer)
    # 假设施加偏移不会改变指针的对齐
    new_ptr = cute.make_ptr(
        tensor.element_type,
        elem_pointer(tensor, coord).toint(),
        tensor.memspace,
        assumed_align=tensor.iterator.alignment,
    )
    return cute.make_tensor(new_ptr, tensor.layout)


@dsl_user_op
def warp_reduction(
    val: cute.Numeric, op: Callable, *, threads_in_group: int = 32, loc=None, ip=None
) -> cute.Numeric:
    """自定义二元运算的 warp 级归约辅助函数。"""
    offset = threads_in_group // 2
    while offset > 0:
        val = op(
            val,
            cute.arch.shuffle_sync_bfly(
                val, offset=offset, mask=-1, mask_and_clamp=31, loc=loc, ip=ip
            ),
        )
        offset //= 2
    return val


warp_reduction_max = partial(
    warp_reduction, op=lambda x, y: fmax(x, y) if isinstance(x, Float32) else max(x, y)
)
warp_reduction_sum = partial(warp_reduction, op=lambda x, y: x + y)  # noqa: FURB118


@dsl_user_op
def make_cotiled_copy(
    atom: cute.CopyAtom, atom_layout_tv: cute.Layout, data_layout: cute.Layout, *, loc=None, ip=None
) -> cute.TiledCopy:
    """已废弃 CuTeDSL `make_cotiled_copy` 的兼容包装。"""
    assert cute.is_static(atom_layout_tv.type), "atom_layout_tv must be static"
    assert cute.is_static(data_layout.type), "data_layout must be static"

    inv_layout_ = cute.left_inverse(data_layout, loc=loc, ip=ip)
    inv_data_layout = cute.make_layout(
        (inv_layout_.shape, (1)), stride=(inv_layout_.stride, (0)), loc=loc, ip=ip
    )
    layout_tv_data = cute.composition(inv_data_layout, atom_layout_tv, loc=loc, ip=ip)

    atom_layout_v_to_check = cute.coalesce(
        cute.make_layout(atom_layout_tv.shape[1], stride=atom_layout_tv.stride[1], loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )
    data_layout_v_to_check = cute.coalesce(
        cute.composition(
            data_layout,
            cute.make_layout(
                layout_tv_data.shape[1], stride=layout_tv_data.stride[1], loc=loc, ip=ip
            ),
            loc=loc,
            ip=ip,
        ),
        loc=loc,
        ip=ip,
    )
    assert data_layout_v_to_check == atom_layout_v_to_check, (
        "the memory pointed to by atom_layout_tv does not exist in the data_layout."
    )

    flat_data_shape = cute.product_each(data_layout.shape, loc=loc, ip=ip)
    tiler = tuple(
        cute.filter(
            cute.composition(
                cute.make_layout(
                    flat_data_shape,
                    stride=tuple(0 if j != i else 1 for j in range(cute.rank(flat_data_shape))),
                    loc=loc,
                    ip=ip,
                ),
                layout_tv_data,
                loc=loc,
                ip=ip,
            ),
            loc=loc,
            ip=ip,
        )
        for i in range(cute.rank(flat_data_shape))
    )
    tile2data = cute.composition(
        cute.make_layout(flat_data_shape, loc=loc, ip=ip), tiler, loc=loc, ip=ip
    )
    layout_tv = cute.composition(
        cute.left_inverse(tile2data, loc=loc, ip=ip), layout_tv_data, loc=loc, ip=ip
    )
    return cute.make_tiled_copy(atom, layout_tv, tiler, loc=loc, ip=ip)


@cute.jit
def scalar_to_ssa(a: cute.Numeric, dtype) -> cute.TensorSSA:
    """把标量转换为形状为 (1,) 且给定 dtype 的 cute TensorSSA"""
    vec = cute.make_rmem_tensor(1, dtype)
    vec[0] = a
    return vec.load()


def ssa_to_scalar(val):
    """可以内联，但保留以镜像上面的 API"""
    return val[0]


@cute.jit
def get_batch_from_cu_tensor(idx: Int32, cu_tensor: cute.Tensor) -> Int32:
    """在累积张量中用二分查找从打包索引确定 batch"""
    batch_size = cute.size(cu_tensor) - 1
    lo = Int32(0)
    hi = batch_size

    while lo < hi:
        mid = (lo + hi) // 2
        if cu_tensor[mid + 1] <= idx:
            lo = mid + 1
        else:
            hi = mid

    return lo


def as_bshkrd_tensor(
    tensor: cute.Tensor,
    h_k: Int32,
    h_r: Int32,
    varlen: bool,
) -> cute.Tensor:
    """把 (B,S,H,D)/(S,H,D) 张量归一化为 (B,S,H_k,H_r,D) 视图。"""
    if cutlass.const_expr(cute.rank(tensor.layout) == 5):
        return tensor
    if cutlass.const_expr(cute.rank(tensor.layout) == 4):
        return cute.make_tensor(
            tensor.iterator,
            cute.make_layout(
                (tensor.shape[0], tensor.shape[1], h_k, h_r, tensor.shape[3]),
                stride=(
                    tensor.stride[0],
                    tensor.stride[1],
                    tensor.stride[2] * h_r,
                    tensor.stride[2],
                    tensor.stride[3],
                ),
            ),
        )
    assert cutlass.const_expr(cute.rank(tensor.layout) == 3), "Expected rank-3 varlen tensor"
    assert cutlass.const_expr(varlen), "Rank-3 input is only valid for varlen"
    return cute.make_tensor(
        tensor.iterator,
        cute.make_layout(
            (1, tensor.shape[0], h_k, h_r, tensor.shape[2]),
            stride=(
                0,
                tensor.stride[0],
                tensor.stride[1] * h_r,
                tensor.stride[1],
                tensor.stride[2],
            ),
        ),
    )


def as_shhb_tensor(
    tensor: cute.Tensor,
    h_k: Int32,
    h_r: Int32,
    b: Int32,
    varlen: bool,
) -> cute.Tensor:
    """把 (B,H,S)/(H,S) 张量归一化为 (S, ((H_r, H_k), B)) 视图。"""
    if cutlass.const_expr(cute.rank(tensor.layout) == 3):
        return cute.make_tensor(
            tensor.iterator,
            cute.make_layout(
                (tensor.shape[2], ((h_r, h_k), tensor.shape[0])),
                stride=(
                    tensor.stride[2],
                    ((tensor.stride[1], tensor.stride[1] * h_r), tensor.stride[0]),
                ),
            ),
        )
    assert cutlass.const_expr(cute.rank(tensor.layout) == 2), "Expected rank-2 varlen tensor"
    assert cutlass.const_expr(varlen), "Rank-2 input is only valid for varlen"
    return cute.make_tensor(
        tensor.iterator,
        cute.make_layout(
            (tensor.shape[1], ((h_r, h_k), b)),
            stride=(
                tensor.stride[1],
                ((tensor.stride[0], tensor.stride[0] * h_r), 0),
            ),
        ),
    )
