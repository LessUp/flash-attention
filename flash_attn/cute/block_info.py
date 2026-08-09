# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
from typing import Tuple, Optional
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr

from flash_attn.cute.seqlen_info import SeqlenInfoQK, SeqlenInfoQKNewK


@dataclass(frozen=True)
class BlockInfo:
    tile_m: cutlass.Constexpr[int]
    tile_n: cutlass.Constexpr[int]
    is_causal: cutlass.Constexpr[bool]
    is_local: cutlass.Constexpr[bool] = False
    is_split_kv: cutlass.Constexpr[bool] = False
    window_size_left: Optional[Int32] = None
    window_size_right: Optional[Int32] = None
    qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1
    num_splits: Int32 = 1
    # 若为 True，调度器将 num_splits 打包进 split_idx 的高 16 位
    pack_split_idx: cutlass.Constexpr[bool] = False
    num_n_blocks_per_split: Optional[cutlass.Constexpr[Int32]] = None

    @cute.jit
    def get_n_block_min_max(
        self,
        seqlen_info: SeqlenInfoQK,
        m_block: Int32,
        split_idx: Int32 = 0,
        num_splits: Int32 = 1,
    ) -> Tuple[Int32, Int32]:
        n_block_max = cute.ceil_div(seqlen_info.seqlen_k, self.tile_n)
        # 讲解：causal 或局部（local）窗口把每个 M 块允许的 N 块范围限制在上界内，
        # 被掩码覆盖的 KV 块无需参与 GEMM，从而大幅节省计算量。
        if const_expr(self.is_causal or (self.is_local and self.window_size_right is not None)):
            m_idx_max = (m_block + 1) * self.tile_m
            if const_expr(self.qhead_per_kvhead_packgqa > 1):
                m_idx_max = cute.ceil_div(m_idx_max, self.qhead_per_kvhead_packgqa)
            n_idx = m_idx_max + seqlen_info.seqlen_k - seqlen_info.seqlen_q
            n_idx_right = n_idx if const_expr(self.is_causal) else n_idx + self.window_size_right
            n_block_max = min(n_block_max, cute.ceil_div(n_idx_right, self.tile_n))
        n_block_min = 0
        if const_expr(self.is_local and self.window_size_left is not None):
            m_idx_min = m_block * self.tile_m
            if const_expr(self.qhead_per_kvhead_packgqa > 1):
                m_idx_min = m_idx_min // self.qhead_per_kvhead_packgqa
            n_idx = m_idx_min + seqlen_info.seqlen_k - seqlen_info.seqlen_q
            n_idx_left = n_idx - self.window_size_left
            n_block_min = cutlass.max(n_idx_left // self.tile_n, 0)
        if cutlass.const_expr(self.is_split_kv):
            if const_expr(self.pack_split_idx):
                # 从 split_idx 高 16 位解包 num_splits（由调度器打包）
                num_splits = split_idx >> 16
                split_idx = split_idx & 0xFFFF
            else:
                num_splits = self.num_splits
            if const_expr(self.num_n_blocks_per_split is not None):
                num_n_blocks_per_split = self.num_n_blocks_per_split
            else:
                num_n_blocks_per_split = (
                    Int32(0)
                    if n_block_max <= n_block_min
                    else (n_block_max - n_block_min + num_splits - 1) // num_splits
                )
            n_block_min = n_block_min + split_idx * num_n_blocks_per_split
            n_block_max = cutlass.min(n_block_min + num_n_blocks_per_split, n_block_max)
        return n_block_min, n_block_max

    @cute.jit
    def get_m_block_min_max(self, seqlen_info: SeqlenInfoQK, n_block: Int32) -> Tuple[Int32, Int32]:
        m_block_max = cute.ceil_div(seqlen_info.seqlen_q, self.tile_m)
        m_block_min = 0
        if const_expr(self.is_causal or (self.is_local and self.window_size_right is not None)):
            n_idx_min = n_block * self.tile_n
            m_idx = n_idx_min + seqlen_info.seqlen_q - seqlen_info.seqlen_k
            m_idx_right = m_idx if const_expr(self.is_causal) else m_idx - self.window_size_right
            m_block_min = max(m_block_min, m_idx_right // self.tile_m)
        if const_expr(self.is_local and self.window_size_left is not None):
            n_idx_max = (n_block + 1) * self.tile_n
            m_idx = n_idx_max + seqlen_info.seqlen_q - seqlen_info.seqlen_k
            m_idx_left = m_idx + self.window_size_left
            m_block_max = min(m_block_max, cute.ceil_div(m_idx_left, self.tile_m))
        return m_block_min, m_block_max

    @cute.jit
    def get_n_block_k_new_min_max(
        self,
        seqlen_info: SeqlenInfoQKNewK,
        m_block: Int32,
        split_idx: Int32 = 0,
        num_splits: Int32 = 1,
    ) -> Tuple[Int32, Int32]:
        """获取新增 K token（追加 KV）的块范围。

        先用 get_n_block_min_max 计算完整 n_block 范围，再通过减去
        seqlen_k_og 将这些块映射到新增 K 的索引空间。
        """
        n_block_min, n_block_max = self.get_n_block_min_max(
            seqlen_info,
            m_block,
            split_idx,
            num_splits,
        )
        idx_k_new_min = cutlass.max(n_block_min * self.tile_n - seqlen_info.seqlen_k_og, 0)
        idx_k_new_max = cutlass.min(
            n_block_max * self.tile_n - seqlen_info.seqlen_k_og, seqlen_info.seqlen_k_new
        )
        n_block_new_min = idx_k_new_min // self.tile_n
        n_block_new_max = (
            cute.ceil_div(idx_k_new_max, self.tile_n)
            if idx_k_new_max > idx_k_new_min
            else n_block_new_min
        )
        return n_block_new_min, n_block_new_max

    @cute.jit
    def get_n_block_min_causal_local_mask(
        self,
        seqlen_info: SeqlenInfoQK,
        m_block: Int32,
        n_block_min: Int32,
    ) -> Int32:
        """如果开头有使用 causal 或 local 掩码的独立迭代，那么这些迭代应在哪里停止"""
        m_idx_min = m_block * self.tile_m
        if const_expr(self.qhead_per_kvhead_packgqa > 1):
            m_idx_min = m_idx_min // self.qhead_per_kvhead_packgqa
        n_idx = m_idx_min + seqlen_info.seqlen_k - seqlen_info.seqlen_q
        n_idx_right = (
            n_idx
            if const_expr(not self.is_local or self.window_size_right is None)
            else n_idx + self.window_size_right
        )
        return cutlass.max(n_block_min, n_idx_right // self.tile_n)

    @cute.jit
    def get_n_block_min_before_local_mask(
        self,
        seqlen_info: SeqlenInfoQK,
        m_block: Int32,
        n_block_min: Int32,
    ) -> Int32:
        """如果结尾有使用 local 掩码的独立迭代，那么非掩码迭代应在哪里停止"""
        if const_expr(not self.is_local or self.window_size_left is None):
            return n_block_min
        else:
            m_idx_max = (m_block + 1) * self.tile_m
            if const_expr(self.qhead_per_kvhead_packgqa > 1):
                m_idx_max = cute.ceil_div(m_idx_max, self.qhead_per_kvhead_packgqa)
            n_idx = m_idx_max + seqlen_info.seqlen_k - seqlen_info.seqlen_q
            n_idx_left = n_idx - self.window_size_left
            return cutlass.max(n_block_min, cute.ceil_div(n_idx_left, self.tile_n))

    @cute.jit
    def get_n_block_max_for_m_block(
        self,
        seqlen_info: SeqlenInfoQK,
        m_block: Int32,
    ) -> Int32:
        n_block_max = cute.ceil_div(seqlen_info.seqlen_k, self.tile_n)
        if const_expr(self.is_causal or self.window_size_right is not None):
            m_idx_max = (m_block + 1) * self.tile_m
            if const_expr(self.qhead_per_kvhead_packgqa > 1):
                m_idx_max = cute.ceil_div(m_idx_max, self.qhead_per_kvhead_packgqa)
            n_idx_right = m_idx_max + seqlen_info.seqlen_k - seqlen_info.seqlen_q
            if const_expr(self.window_size_right is not None):
                n_idx_right += self.window_size_right
            n_block_max = min(n_block_max, cute.ceil_div(n_idx_right, self.tile_n))
        return n_block_max
