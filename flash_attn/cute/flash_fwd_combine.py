# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# A reimplementation of https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_fwd_combine_kernel.h
# 从 Cutlass C++ 移植为 Cute-DSL 版本。
import math
from typing import Callable, Type, Optional
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync
from cutlass import Float32, Int32, Boolean, const_expr

from quack.cute_dsl_utils import ParamsBase

from flash_attn.cute import utils
from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned
from flash_attn.cute.seqlen_info import SeqlenInfo
from flash_attn.cute.tile_scheduler import (
    SingleTileScheduler,
    SingleTileVarlenScheduler,
    TileSchedulerArguments,
)
from cutlass.cute import FastDivmodDivisor


class FlashAttentionForwardCombine:
    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        dtype_partial: Type[cutlass.Numeric],
        head_dim: int,
        num_head: int,
        tile_m: int = 8,
        k_block_size: int = 64,
        log_max_splits: int = 4,
        num_threads: int = 256,
        stages: int = 4,
    ):
        """
        SplitKV 前向合并（combine）内核：把 SplitKV 各 split 的部分结果合并为最终输出。

        :param dtype: 输出数据类型
        :param dtype_partial: 部分累加（partial）结果的数据类型
        :param head_dim: 注意力头维度（head_dim）
        :param num_head: 头数
        :param tile_m: M 方向（query 行）的块大小
        :param k_block_size: K 方向（head_dim）的块大小
        :param log_max_splits: 最大 split 数的 log2 值
        :param num_threads: 线程数
        :param varlen: 是否使用变长序列
        :param stages: 流水线（pipeline）阶段数
        """
        self.dtype = dtype
        self.dtype_partial = dtype_partial
        self.head_dim = head_dim
        self.num_head = num_head
        self.tile_m = tile_m
        self.k_block_size = k_block_size
        self.max_splits = 1 << log_max_splits
        self.num_threads = num_threads
        self.is_even_k = head_dim % k_block_size == 0
        self.stages = stages

    @staticmethod
    def can_implement(
        dtype,
        dtype_partial,
        head_dim,
        tile_m,
        k_block_size,
        log_max_splits,
        num_threads,
    ) -> bool:
        """检查能否用给定参数实现该内核。"""
        if dtype not in [cutlass.Float16, cutlass.BFloat16, cutlass.Float32]:
            return False
        if dtype_partial not in [cutlass.Float16, cutlass.BFloat16, Float32]:
            return False
        if head_dim % 8 != 0:
            return False
        if num_threads % 32 != 0:
            return False
        if tile_m % 8 != 0:
            return False
        max_splits = 1 << log_max_splits
        if max_splits > 256:
            return False
        if (tile_m * max_splits) % num_threads != 0:
            return False
        return True

    def _setup_attributes(self):
        # O_partial 的全局内存（gmem）拷贝设置
        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // self.dtype_partial.width
        assert self.k_block_size % async_copy_elems == 0

        k_block_gmem = (
            128 if self.k_block_size % 128 == 0 else (64 if self.k_block_size % 64 == 0 else 32)
        )
        gmem_threads_per_row = k_block_gmem // async_copy_elems
        assert self.num_threads % gmem_threads_per_row == 0

        # O_partial 加载用的异步拷贝原子（async copy atom）
        atom_async_copy_partial = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self.dtype_partial,
            num_bits_per_copy=universal_copy_bits,
        )
        tOpartial_layout = cute.make_ordered_layout(
            (self.num_threads // gmem_threads_per_row, gmem_threads_per_row),
            order=(1, 0),
        )
        vOpartial_layout = cute.make_layout((1, async_copy_elems))  # 每次加载 4 个值
        self.gmem_tiled_copy_O_partial = cute.make_tiled_copy_tv(
            atom_async_copy_partial, tOpartial_layout, vOpartial_layout
        )

        # 最终 O 的 gmem 拷贝设置（store 使用通用拷贝 universal copy）
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dtype,
            num_bits_per_copy=async_copy_elems * self.dtype.width,
        )
        self.gmem_tiled_copy_O = cute.make_tiled_copy_tv(
            atom_universal_copy,
            tOpartial_layout,
            vOpartial_layout,  # 每次存储 4 个值
        )

        # LSE 的拷贝设置（使用异步拷贝，alignment = 1）
        lse_copy_bits = Float32.width  # 每次拷贝 1 个元素，这里的 width 以比特（bit）为单位
        m_block_smem = (
            128
            if self.tile_m % 128 == 0
            else (
                64
                if self.tile_m % 64 == 0
                else (32 if self.tile_m % 32 == 0 else (16 if self.tile_m % 16 == 0 else 8))
            )
        )
        gmem_threads_per_row_lse = m_block_smem
        assert self.num_threads % gmem_threads_per_row_lse == 0

        # LSE 加载用的异步拷贝原子
        atom_async_copy_lse = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.ALWAYS),
            Float32,
            num_bits_per_copy=lse_copy_bits,
        )
        tLSE_layout = cute.make_ordered_layout(
            (self.num_threads // gmem_threads_per_row_lse, gmem_threads_per_row_lse),
            order=(1, 0),
        )
        vLSE_layout = cute.make_layout(1)
        self.gmem_tiled_copy_LSE = cute.make_tiled_copy_tv(
            atom_async_copy_lse, tLSE_layout, vLSE_layout
        )

        # ///////////////////////////////////////////////////////////////////////////////
        # 共享内存（shared memory）
        # ///////////////////////////////////////////////////////////////////////////////

        # LSE 从共享内存到寄存器的拷贝
        self.smem_threads_per_col_lse = self.num_threads // m_block_smem
        assert 32 % self.smem_threads_per_col_lse == 0  # 必须整除 warp 大小

        s2r_layout_atom_lse = cute.make_ordered_layout(
            (self.smem_threads_per_col_lse, self.num_threads // self.smem_threads_per_col_lse),
            order=(0, 1),
        )
        self.s2r_tiled_copy_LSE = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32),
            s2r_layout_atom_lse,
            cute.make_layout(1),
        )

        # LSE 的共享内存布局，使用 swizzle 以避免 bank 冲突
        # 该 swizzle 对 kBlockMSmem = 8/16/32/64/128 均无 bank 冲突
        if const_expr(m_block_smem == 8):
            smem_lse_swizzle = cute.make_swizzle(5, 0, 5)
        elif const_expr(m_block_smem == 16):
            smem_lse_swizzle = cute.make_swizzle(4, 0, 4)
        else:
            smem_lse_swizzle = cute.make_swizzle(3, 2, 3)
        smem_layout_atom_lse = cute.make_composed_layout(
            smem_lse_swizzle, 0, cute.make_ordered_layout((8, m_block_smem), order=(1, 0))
        )
        self.smem_layout_lse = cute.tile_to_shape(
            smem_layout_atom_lse, (self.max_splits, self.tile_m), (0, 1)
        )

        # O_partial 的共享内存布局（为流水线阶段准备的简单布局）
        self.smem_layout_o = cute.make_ordered_layout(
            (self.tile_m, self.k_block_size, self.stages), order=(1, 0, 2)
        )

    @cute.jit
    def __call__(
        self,
        mO_partial: cute.Tensor,
        mLSE_partial: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor] = None,
        cu_seqlens: Optional[cute.Tensor] = None,
        seqused: Optional[cute.Tensor] = None,
        num_splits_dynamic_ptr: Optional[cute.Tensor] = None,
        virtual_batch_idx: Optional[cute.Tensor] = None,
        semaphore_to_reset: Optional[cute.Tensor] = None,
        # 务必把 stream 放在最后一个参数（EnvStream：通过 TVM FFI 隐式获得）。
        stream: cuda.CUstream = None,
    ):
        # 类型检查
        if const_expr(not (mO_partial.element_type == self.dtype_partial)):
            raise TypeError("O partial tensor must match dtype_partial")
        if const_expr(not (mO.element_type == self.dtype)):
            raise TypeError("O tensor must match dtype")
        if const_expr(mLSE_partial.element_type not in [Float32]):
            raise TypeError("LSE partial tensor must be Float32")
        if const_expr(mLSE is not None and mLSE.element_type not in [Float32]):
            raise TypeError("LSE tensor must be Float32")

        # 形状校验——输入张量是用户格式，需转换成内核格式
        if const_expr(len(mO_partial.shape) not in [4, 5]):
            raise ValueError(
                "O partial tensor must have 4 or 5 dimensions: (num_splits, batch, seqlen, nheads, headdim) or (num_splits, total_q, nheads, headdim)"
            )
        if const_expr(len(mLSE_partial.shape) not in [3, 4]):
            raise ValueError(
                "LSE partial tensor must have 3 or 4 dimensions: (num_splits, batch, seqlen, nheads) or (num_splits, total_q, nheads)"
            )
        if const_expr(len(mO.shape) not in [3, 4]):
            raise ValueError(
                "O tensor must have 3 or 4 dimensions: (batch, seqlen, nheads, headdim) or (total_q, nheads, headdim)"
            )
        if const_expr(mLSE is not None and len(mLSE.shape) not in [2, 3]):
            raise ValueError(
                "LSE tensor must have 2 or 3 dimensions: (batch, seqlen, nheads) or (total_q, nheads)"
            )

        mO_partial, mO = [assume_tensor_aligned(t) for t in (mO_partial, mO)]
        # O_partial 布局转置（用户格式 -> 内核格式）：
        # (num_splits, b, seqlen, h, d) -> (seqlen, d, num_splits, h, b)
        # 或 (num_splits, total_q, h, d) -> (total_q, d, num_splits, h)
        O_partial_layout_transpose = (
            [2, 4, 0, 3, 1] if const_expr(cu_seqlens is None) else [1, 3, 0, 2]
        )
        # O 布局转置：(b, seqlen, h, d) -> (seqlen, d, h, b)，或 (total_q, h, d) -> (total_q, d, h)
        mO_partial = cute.make_tensor(
            mO_partial.iterator, cute.select(mO_partial.layout, mode=O_partial_layout_transpose)
        )
        O_layout_transpose = [1, 3, 2, 0] if const_expr(cu_seqlens is None) else [0, 2, 1]
        mO = cute.make_tensor(mO.iterator, cute.select(mO.layout, mode=O_layout_transpose))
        # LSE_partial 布局转置：
        # (num_splits, b, seqlen, h) -> (seqlen, num_splits, h, b)
        # 或 (num_splits, total_q, h) -> (total_q, num_splits, h)
        LSE_partial_layout_transpose = [2, 0, 3, 1] if const_expr(cu_seqlens is None) else [1, 0, 2]
        mLSE_partial = cute.make_tensor(
            mLSE_partial.iterator,
            cute.select(mLSE_partial.layout, mode=LSE_partial_layout_transpose),
        )
        # LSE 布局转置：(b, seqlen, h) -> (seqlen, h, b)，或 (total_q, h) -> (total_q, h)
        LSE_layout_transpose = [1, 2, 0] if const_expr(cu_seqlens is None) else [0, 1]
        mLSE = (
            cute.make_tensor(mLSE.iterator, cute.select(mLSE.layout, mode=LSE_layout_transpose))
            if mLSE is not None
            else None
        )

        # 判断是否为变长序列（varlen）场景
        varlen = const_expr(cu_seqlens is not None or seqused is not None)

        self._setup_attributes()

        @cute.struct
        class SharedStorage:
            sLSE: cute.struct.Align[
                cute.struct.MemRange[Float32, cute.cosize(self.smem_layout_lse)], 128
            ]
            sMaxValidSplit: cute.struct.Align[cute.struct.MemRange[Int32, self.tile_m], 128]
            sO: cute.struct.Align[
                cute.struct.MemRange[self.dtype_partial, cute.cosize(self.smem_layout_o)], 128
            ]

        smem_size = SharedStorage.size_in_bytes()

        # 网格（grid）维度：(ceil_div(seqlen, m_block), ceil_div(head_dim, k_block), num_head * batch)
        seqlen = mO_partial.shape[0]
        num_head = mO_partial.shape[3]
        batch_size = (
            mO_partial.shape[4]
            if const_expr(cu_seqlens is None)
            else Int32(cu_seqlens.shape[0] - 1)
        )

        # 创建 FastDivmodDivisor 对象，用于高效的除法/取模运算
        seqlen_divmod = FastDivmodDivisor(seqlen)
        head_divmod = FastDivmodDivisor(num_head)

        if const_expr(varlen):
            TileScheduler = SingleTileVarlenScheduler
        else:
            TileScheduler = SingleTileScheduler
        tile_sched_args = TileSchedulerArguments(
            num_block=cute.ceil_div(seqlen * num_head, self.tile_m),
            num_head=cute.ceil_div(self.head_dim, self.k_block_size),
            num_batch=batch_size,
            num_splits=1,
            seqlen_k=1,
            headdim=1,
            headdim_v=1,
            total_q=mO_partial.shape[0] * num_head
            if const_expr(cu_seqlens is not None)
            else seqlen * batch_size * num_head,
            tile_shape_mn=(self.tile_m, self.tile_m),
            mCuSeqlensQ=cu_seqlens,
            mSeqUsedQ=seqused,
            qhead_per_kvhead_packgqa=self.num_head,
            virtual_batch_idx_ptr=virtual_batch_idx,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        self.kernel(
            mO_partial,
            mLSE_partial,
            mO,
            mLSE,
            cu_seqlens,
            seqused,
            num_splits_dynamic_ptr,
            virtual_batch_idx,
            semaphore_to_reset,
            SharedStorage,
            self.smem_layout_lse,
            self.smem_layout_o,
            self.gmem_tiled_copy_O_partial,
            self.gmem_tiled_copy_O,
            self.gmem_tiled_copy_LSE,
            self.s2r_tiled_copy_LSE,
            seqlen_divmod,
            head_divmod,
            varlen,
            tile_sched_params,
            TileScheduler,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            smem=smem_size,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mO_partial: cute.Tensor,
        mLSE_partial: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        cu_seqlens: Optional[cute.Tensor],
        seqused: Optional[cute.Tensor],
        num_splits_dynamic_ptr: Optional[cute.Tensor],
        virtual_batch_idx: Optional[cute.Tensor],
        semaphore_to_reset: Optional[cute.Tensor],
        SharedStorage: cutlass.Constexpr,
        smem_layout_lse: cute.Layout | cute.ComposedLayout,
        smem_layout_o: cute.Layout,
        gmem_tiled_copy_O_partial: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        gmem_tiled_copy_LSE: cute.TiledCopy,
        s2r_tiled_copy_LSE: cute.TiledCopy,
        seqlen_divmod: FastDivmodDivisor,
        head_divmod: FastDivmodDivisor,
        varlen: cutlass.Constexpr[bool],
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
    ):
        # 线程与块（block）索引
        tidx, _, _ = cute.arch.thread_idx()
        tile_scheduler = TileScheduler.create(tile_sched_params)
        work_tile = tile_scheduler.initial_work_tile_info()
        m_block, k_block, maybe_virtual_batch, _ = work_tile.tile_idx

        # 将虚拟 batch 索引映射为真实 batch 索引（用于 persistent tile scheduler）
        batch_idx = (
            virtual_batch_idx[maybe_virtual_batch]
            if const_expr(virtual_batch_idx is not None and not varlen)
            else maybe_virtual_batch
        )

        # ///////////////////////////////////////////////////////////////////////////////
        # 获取共享内存缓冲区
        # ///////////////////////////////////////////////////////////////////////////////
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sLSE = storage.sLSE.get_tensor(smem_layout_lse)
        sMaxValidSplit = storage.sMaxValidSplit.get_tensor((self.tile_m,))
        sO = storage.sO.get_tensor(smem_layout_o)

        # 处理信号量（semaphore）复位——先等待依赖的 grid 完成
        if const_expr(semaphore_to_reset is not None):
            bidx, bidy, bidz = cute.arch.block_idx()
            if (
                tidx == 0
                and bidx == cute.arch.grid_dim()[0] - 1
                and bidy == cute.arch.grid_dim()[1] - 1
                and bidz == cute.arch.grid_dim()[2] - 1
            ):
                cute.arch.griddepcontrol_wait()
                semaphore_to_reset[0] = 0

        if work_tile.is_valid_tile:
            # 获取 split 数量（使用 maybe_virtual_batch 按 batch 槽位取 split 数）
            num_splits = (
                num_splits_dynamic_ptr[maybe_virtual_batch]
                if const_expr(num_splits_dynamic_ptr is not None)
                else mLSE_partial.shape[1]
            )
            # 用 SeqlenInfo 处理变长序列
            seqlen_info = SeqlenInfo.create(
                batch_idx=batch_idx,
                seqlen_static=mO_partial.shape[0],
                cu_seqlens=cu_seqlens,
                seqused=seqused,
                # 不需要传入 tile 大小，因为这里不使用 offset_padded
            )
            seqlen, offset = seqlen_info.seqlen, seqlen_info.offset

            # 取出头数（head 索引将在运行期动态确定）
            num_head = mO_partial.shape[3]
            max_idx = seqlen * num_head

            # TODO: 若 split 数为动态，单 split 时本可提前退出——目前总是执行合并，
            # 以便 num_splits_dynamic == 1 时仍能从 mO_partial[0] 写出 mO。
            if (const_expr(num_splits_dynamic_ptr is None) or num_splits > 0) and (
                const_expr(not varlen) or m_block * self.tile_m < max_idx
            ):
                # 等待依赖的 grid（例如产生 O_partial/LSE_partial 的主注意力内核）完成
                cute.arch.griddepcontrol_wait()

                # ===============================
                # Step 1：把 LSE_partial 从 gmem 加载到共享内存
                # 讲解：合并内核第 1 步把各 split 的 LSE（每行 softmax 分母的对数）读入共享内存，
                # 后续需要用它们计算各 split 的加权系数。
                # ===============================

                mLSE_partial_cur = seqlen_info.offset_batch(mLSE_partial, batch_idx, dim=3)
                mLSE_partial_copy = cute.tiled_divide(mLSE_partial_cur, (1,))
                gmem_thr_copy_LSE = gmem_tiled_copy_LSE.get_slice(tidx)
                tLSEsLSE = gmem_thr_copy_LSE.partition_D(sLSE)
                # 创建恒等张量（identity tensor）用于坐标跟踪
                cLSE = cute.make_identity_tensor((self.max_splits, self.tile_m))
                tLSEcLSE = gmem_thr_copy_LSE.partition_S(cLSE)

                # 加载 LSE 的部分值
                for m in cutlass.range(cute.size(tLSEcLSE, mode=[2]), unroll_full=True):
                    mi = tLSEcLSE[0, 0, m][1]  # 取 m 坐标
                    idx = m_block * self.tile_m + mi
                    if idx < max_idx:
                        # 用 FastDivmodDivisor 计算真实的序列位置与 head 索引
                        if const_expr(not varlen):
                            head_idx, m_idx = divmod(idx, seqlen_divmod)
                        else:
                            head_idx = idx // seqlen
                            m_idx = idx - head_idx * seqlen
                        mLSE_partial_cur_copy = mLSE_partial_copy[None, m_idx, None, head_idx]
                        for s in cutlass.range(cute.size(tLSEcLSE, mode=[1]), unroll_full=True):
                            si = tLSEcLSE[0, s, 0][0]  # 取 split 坐标
                            if si < num_splits:
                                cute.copy(
                                    gmem_thr_copy_LSE,
                                    mLSE_partial_cur_copy[None, si],
                                    tLSEsLSE[None, s, m],
                                )
                            else:
                                tLSEsLSE[None, s, m].fill(-Float32.inf)
                    # 不需要把其余 LSE 清零，因为不会把这些位置写回 gmem
                cute.arch.cp_async_commit_group()

                # ===============================
                # Step 2：为流水线各阶段加载 O_partial
                # 讲解：第 2 步用 cp.async 预取 O_partial。合并涉及多个 split，这里用多级流水线缓冲，
                # 预先加载前 stages-1 个 split，让后续主循环能一边计算一边预取。
                # ===============================

                gmem_thr_copy_O_partial = gmem_tiled_copy_O_partial.get_slice(tidx)
                cO = cute.make_identity_tensor((self.tile_m, self.k_block_size))
                tOcO = gmem_thr_copy_O_partial.partition_D(cO)
                tOsO_partial = gmem_thr_copy_O_partial.partition_D(sO)
                mO_partial_cur = seqlen_info.offset_batch(mO_partial, batch_idx, dim=4)

                # 提前算好这些值，避免在循环里重复计算
                num_rows = const_expr(cute.size(tOcO, mode=[1]))
                tOmidx = cute.make_rmem_tensor(num_rows, cutlass.Int32)
                tOhidx = cute.make_rmem_tensor(num_rows, cutlass.Int32)
                tOrOptr = cute.make_rmem_tensor(num_rows, cutlass.Int64)
                for m in cutlass.range(num_rows, unroll_full=True):
                    mi = tOcO[0, m, 0][0]  # m 坐标
                    idx = m_block * self.tile_m + mi
                    if const_expr(not varlen):
                        tOhidx[m], tOmidx[m] = divmod(idx, seqlen_divmod)
                    else:
                        tOhidx[m] = idx // seqlen
                        tOmidx[m] = idx - tOhidx[m] * seqlen
                    tOrOptr[m] = utils.elem_pointer(
                        mO_partial_cur, (tOmidx[m], k_block * self.k_block_size, 0, tOhidx[m])
                    ).toint()
                    if idx >= max_idx:
                        tOhidx[m] = -1

                tOpO = None
                if const_expr(not self.is_even_k):
                    tOpO = cute.make_rmem_tensor(cute.size(tOcO, mode=[2]), Boolean)
                    for k in cutlass.range(cute.size(tOpO), unroll_full=True):
                        tOpO[k] = (
                            tOcO[0, 0, k][1] < mO_partial.shape[1] - k_block * self.k_block_size
                        )
                    # if cute.arch.thread_idx()[0] == 0 and k_block == 1: cute.print_tensor(tOpO)

                load_O_partial = partial(
                    self.load_O_partial,
                    gmem_tiled_copy_O_partial,
                    tOrOptr,
                    tOsO_partial,
                    tOhidx,
                    tOpO,
                    tOcO,
                    mO_partial_cur.layout,
                )

                # 预取前几个阶段的 O_partial
                for stage in cutlass.range(self.stages - 1, unroll_full=True):
                    if stage < num_splits:
                        load_O_partial(stage, stage)
                    cute.arch.cp_async_commit_group()

                # ===============================
                # Step 3：把 LSE 从 smem 加载到寄存器并做转置
                # ===============================

                # 等待 LSE 与初始 O_partial 各阶段加载完成
                cute.arch.cp_async_wait_group(self.stages - 1)
                cute.arch.sync_threads()
                # if cute.arch.thread_idx()[0] == 0:
                #     # cute.print_tensor(sLSE)
                #     for i in range(64):
                #         cute.printf("sLSE[%d, 0] = %f", i, sLSE[i, 0])
                # cute.arch.sync_threads()

                s2r_thr_copy_LSE = s2r_tiled_copy_LSE.get_slice(tidx)
                ts2rsLSE = s2r_thr_copy_LSE.partition_S(sLSE)
                ts2rrLSE = cute.make_rmem_tensor_like(ts2rsLSE)
                cute.copy(s2r_tiled_copy_LSE, ts2rsLSE, ts2rrLSE)

                # ===============================
                # Step 4：沿 split 维度计算最终 LSE
                # 讲解：第 4 步做数值稳定的 softmax 合并——先跨 split 取 LSE 最大值，再计算
                # exp(lse - lse_max) 作为权重并归一化，得到每个 split 的加权系数（数值稳定技巧）。
                # ===============================

                lse_sum = cute.make_rmem_tensor(cute.size(ts2rrLSE, mode=[2]), Float32)
                ts2rcLSE = s2r_thr_copy_LSE.partition_D(cLSE)
                # 为每一行计算"最大有效 split"索引，以便后续提前短路（跳过无效 split）
                max_valid_split = cute.make_rmem_tensor(cute.size(ts2rrLSE, mode=[2]), Int32)
                assert cute.size(ts2rrLSE, mode=[0]) == 1
                # 对每一行计算最大值、缩放系数与最终 LSE
                for m in cutlass.range(cute.size(ts2rrLSE, mode=[2]), unroll_full=True):
                    # 在所有 split 中找 LSE 最大值
                    threads_per_col = const_expr(self.smem_threads_per_col_lse)
                    lse_max = cute.arch.warp_reduction_max(
                        ts2rrLSE[None, None, m]
                        .load()
                        .reduce(cute.ReductionOp.MAX, init_val=-Float32.inf, reduction_profile=0),
                        threads_in_group=threads_per_col,
                    )
                    # if cute.arch.thread_idx()[0] == 0: cute.printf(lse_max)
                    # 找最大有效 split 索引
                    max_valid_idx = -1
                    for s in cutlass.range(cute.size(ts2rrLSE, mode=[1]), unroll_full=True):
                        if ts2rrLSE[0, s, m] != -Float32.inf:
                            max_valid_idx = ts2rcLSE[0, s, 0][0]  # 取 split 坐标
                    # if cute.arch.thread_idx()[0] < 32: cute.printf(max_valid_idx)
                    max_valid_split[m] = cute.arch.warp_reduction_max(
                        max_valid_idx, threads_in_group=threads_per_col
                    )
                    # 计算 exp 缩放系数并求和
                    lse_max_cur = (
                        0.0 if lse_max == -Float32.inf else lse_max
                    )  # 防止所有局部 LSE 都是 -inf
                    LOG2_E = math.log2(math.e)
                    lse_sum_cur = 0.0
                    for s in cutlass.range(cute.size(ts2rrLSE, mode=[1]), unroll_full=True):
                        scale = cute.math.exp2(
                            ts2rrLSE[0, s, m] * LOG2_E - (lse_max_cur * LOG2_E), fastmath=True
                        )
                        lse_sum_cur += scale
                        ts2rrLSE[0, s, m] = scale  # 暂存缩放系数，供后续使用
                    lse_sum_cur = cute.arch.warp_reduction_sum(
                        lse_sum_cur, threads_in_group=threads_per_col
                    )
                    lse_sum[m] = cute.math.log(lse_sum_cur, fastmath=True) + lse_max
                    # 归一化缩放系数
                    inv_sum = (
                        0.0
                        if (lse_sum_cur == 0.0 or lse_sum_cur != lse_sum_cur)
                        else 1.0 / lse_sum_cur
                    )
                    ts2rrLSE[None, None, m].store(ts2rrLSE[None, None, m].load() * inv_sum)
                # 把归一化后的权重 exp(lse - lse_logsum) 存回 smem
                cute.copy(s2r_tiled_copy_LSE, ts2rrLSE, ts2rsLSE)

                # 把最大有效 split 索引存入 smem
                for m in cutlass.range(cute.size(ts2rrLSE, mode=[2]), unroll_full=True):
                    if ts2rcLSE[0, 0, m][0] == 0:  # 只有负责 s=0 的线程写入
                        mi = ts2rcLSE[0, 0, m][1]
                        if mi < self.tile_m:
                            sMaxValidSplit[mi] = max_valid_split[m]

                # ===============================
                # Step 5：把最终 LSE 写回 gmem
                # ===============================

                if const_expr(mLSE is not None):
                    if const_expr(cu_seqlens is None):
                        mLSE_cur = mLSE[None, None, batch_idx]
                    else:
                        mLSE_cur = cute.domain_offset((offset, 0), mLSE)
                    if k_block == 0:  # 只有第一个 k_block 写入 LSE（当提供了 mLSE 时）
                        for m in cutlass.range(cute.size(ts2rrLSE, mode=[2]), unroll_full=True):
                            if ts2rcLSE[0, 0, m][0] == 0:  # 只有负责 s=0 的线程写入
                                mi = ts2rcLSE[0, 0, m][1]
                                idx = m_block * self.tile_m + mi
                                if idx < max_idx:
                                    if const_expr(not varlen):
                                        head_idx, m_idx = divmod(idx, seqlen_divmod)
                                    else:
                                        head_idx = idx // seqlen
                                        m_idx = idx - head_idx * seqlen
                                    mLSE_cur[m_idx, head_idx] = lse_sum[m]

                # ===============================
                # Step 6：读取 O_partial 并累加得到最终 O
                # 讲解：第 6 步是合并内核的主循环：按 split 累加 rO += scale_s * O_partial_s。
                # 它用两套 stage 索引做流水线——stage_load 负责 cp.async 预取后续 split，
                # stage_compute 负责读取当前 split，二者错开以隐藏访存延迟。
                # ===============================

                cute.arch.sync_threads()

                # 获取本线程负责行的最大有效 split
                thr_max_valid_split = sMaxValidSplit[tOcO[0, 0, 0][0]]
                for m in cutlass.range(1, cute.size(tOcO, mode=[1]), unroll_full=True):
                    thr_max_valid_split = max(thr_max_valid_split, sMaxValidSplit[tOcO[0, m, 0][0]])

                tOrO_partial = cute.make_rmem_tensor_like(tOsO_partial[None, None, None, 0])
                tOrO = cute.make_rmem_tensor_like(tOrO_partial, Float32)
                tOrO.fill(0.0)

                stage_load = self.stages - 1
                stage_compute = 0

                # 主累加循环
                for s in cutlass.range(thr_max_valid_split + 1, unroll=4):
                    # 取当前 split 的缩放系数
                    scale = cute.make_rmem_tensor(num_rows, Float32)
                    for m in cutlass.range(num_rows, unroll_full=True):
                        scale[m] = sLSE[s, tOcO[0, m, 0][0]]  # 从 smem 取缩放系数

                    # 必要时预取下一个阶段（stage）的数据
                    split_to_load = s + self.stages - 1
                    if split_to_load <= thr_max_valid_split:
                        load_O_partial(split_to_load, stage_load)
                    cute.arch.cp_async_commit_group()
                    stage_load = 0 if stage_load == self.stages - 1 else stage_load + 1

                    # 等待当前阶段的数据就绪
                    cute.arch.cp_async_wait_group(self.stages - 1)
                    # 这里不需要 __syncthreads()，因为每个线程只读自己那份 smem 数据
                    # 从 smem 拷贝到寄存器
                    cute.autovec_copy(tOsO_partial[None, None, None, stage_compute], tOrO_partial)
                    stage_compute = 0 if stage_compute == self.stages - 1 else stage_compute + 1

                    # 累加按比例缩放的部分结果
                    for m in cutlass.range(num_rows, unroll_full=True):
                        if tOhidx[m] >= 0 and scale[m] > 0.0:
                            tOrO[None, m, None].store(
                                tOrO[None, m, None].load()
                                + scale[m] * tOrO_partial[None, m, None].load().to(Float32)
                            )

                # ===============================
                # Step 7：把最终 O 写回 gmem
                # ===============================

                rO = cute.make_rmem_tensor_like(tOrO, self.dtype)
                rO.store(tOrO.load().to(self.dtype))
                if const_expr(cu_seqlens is None):
                    mO_cur = mO[None, None, None, batch_idx]
                else:
                    mO_cur = cute.domain_offset((offset, 0, 0), mO)
                mO_cur = utils.domain_offset_aligned((0, k_block * self.k_block_size, 0), mO_cur)
                elems_per_store = const_expr(cute.size(gmem_tiled_copy_O.layout_tv_tiled[1]))
                # mO_cur_copy = cute.tiled_divide(mO_cur, (1, elems_per_store,))
                gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
                # 写出最终结果
                for m in cutlass.range(num_rows, unroll_full=True):
                    if tOhidx[m] >= 0:
                        mO_cur_copy = cute.tiled_divide(
                            mO_cur[tOmidx[m], None, tOhidx[m]], (elems_per_store,)
                        )
                        for k in cutlass.range(cute.size(tOcO, mode=[2]), unroll_full=True):
                            k_idx = tOcO[0, 0, k][1] // elems_per_store
                            if const_expr(self.is_even_k) or tOpO[k]:
                                cute.copy(gmem_thr_copy_O, rO[None, m, k], mO_cur_copy[None, k_idx])

        # 讲解：load_O_partial 是 O_partial 的流水线预取辅助函数：把第 split 个分块的数据用
        # cp.async 拷入指定的 stage 缓冲（tOsO_partial[..., stage]），实现异步流水线加载。
    @cute.jit
    def load_O_partial(
        self,
        gmem_tiled_copy_O_partial: cute.TiledCopy,
        tOrOptr: cute.Tensor,
        tOsO_partial: cute.Tensor,
        tOhidx: cute.Tensor,
        tOpO: Optional[cute.Tensor],
        tOcO: cute.Tensor,
        mO_cur_partial_layout: cute.Layout,
        split: Int32,
        stage: Int32,
    ) -> None:
        elems_per_load = const_expr(cute.size(gmem_tiled_copy_O_partial.layout_tv_tiled[1]))
        tOsO_partial_cur = tOsO_partial[None, None, None, stage]
        for m in cutlass.range(cute.size(tOcO, [1]), unroll_full=True):
            if tOhidx[m] >= 0:
                o_gmem_ptr = cute.make_ptr(
                    tOsO_partial.element_type, tOrOptr[m], cute.AddressSpace.gmem, assumed_align=16
                )
                mO_partial_cur = cute.make_tensor(
                    o_gmem_ptr, cute.slice_(mO_cur_partial_layout, (0, None, None, 0))
                )
                mO_partial_cur_copy = cute.tiled_divide(mO_partial_cur, (elems_per_load,))
                for k in cutlass.range(cute.size(tOcO, mode=[2]), unroll_full=True):
                    k_idx = tOcO[0, 0, k][1] // elems_per_load
                    if const_expr(tOpO is None) or tOpO[k]:
                        cute.copy(
                            gmem_tiled_copy_O_partial,
                            mO_partial_cur_copy[None, k_idx, split],
                            tOsO_partial_cur[None, m, k],
                        )
