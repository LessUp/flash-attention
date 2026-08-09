# Copyright (c) 2025, Tri Dao.

from typing import Optional, Callable, TypeAlias, Tuple
from dataclasses import dataclass
import enum

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Uint32, const_expr
from cutlass.cutlass_dsl import min as dsl_min

from quack import layout_utils
import flash_attn.cute.utils as utils
from flash_attn.cute.block_info import BlockInfo
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.utils import AuxData

MaskGenFn: TypeAlias = Callable[[int], Uint32]
MASK_R2P_CHUNK_SIZE: int = 32


@cute.jit
def call_mask_mod(
    mask_mod: cutlass.Constexpr,
    batch_idx,
    head_idx,
    q_idx,
    kv_idx,
    seqlen_info,
    aux_data: AuxData,
):
    # 兼容层：为引入 aux_scalars 之前的旧版 mask_mod 可调用对象提供兼容。
    if const_expr(aux_data.scalars is not None):
        return mask_mod(
            batch_idx,
            head_idx,
            q_idx,
            kv_idx,
            seqlen_info,
            aux_data.tensors,
            aux_data.scalars,
        )
    return mask_mod(
        batch_idx,
        head_idx,
        q_idx,
        kv_idx,
        seqlen_info,
        aux_data.tensors,
    )


@cute.jit
def r2p_bitmask_below(limit: Int32, s: int) -> Uint32:
    """32 位 R2P 位掩码，保留位置 < limit 的元素（上界不包含 limit）。

    块 `s` 中位置 0..limit-1 对应的位为 1（保留），其余位为 0（掩码）。
    使用内联 PTX，以避免按类型位宽移位带来的未定义行为（UB）。
    """
    m = max((s + 1) * MASK_R2P_CHUNK_SIZE - limit, 0)
    return utils.shr_u32(Uint32(0xFFFFFFFF), Uint32(m))


@cute.jit
def r2p_bitmask_above(limit: Int32, s: int) -> Uint32:
    """32 位 R2P 位掩码，保留位置 >= limit 的元素（下界包含 limit）。

    块 `s` 中位置 limit..31 对应的位为 1（保留），其余位为 0（掩码）。
    使用内联 PTX，以避免按类型位宽移位带来的未定义行为（UB）。
    """
    n = max(limit - s * MASK_R2P_CHUNK_SIZE, 0)
    return utils.shl_u32(Uint32(0xFFFFFFFF), Uint32(n))


@cute.jit
def mask_r2p_lambda(
    X: cute.Tensor,
    mask_gen_fn: cutlass.Constexpr[MaskGenFn],
    rank1: bool = False,
) -> None:
    """用自定义位掩码生成器应用 R2P 掩码。

    mask_gen_fn(chunk_idx: constexpr int) -> Uint32:
        返回该块的 32 位掩码。第 i 位为 1 表示列 chunk_idx * chunk_size + i 被保留；
        第 i 位为 0 表示被掩码为 -inf。
    """
    # 讲解：R2P（寄存器转谓词）掩码技巧 —— 一条 32 位掩码一次处理 32 列，
    # 编译器把位测试降级为 R2P 谓词指令而非逐元素分支，显著降低掩码开销。
    ncol = const_expr(cute.size(X.shape[cute.rank(X) - 1]) if not rank1 else cute.size(X.shape))
    # 每 32 列一块。mask_gen_fn 返回 Uint32 位掩码（1=保留）。
    CHUNK_SIZE = MASK_R2P_CHUNK_SIZE
    for s in cutlass.range_constexpr(cute.ceil_div(ncol, CHUNK_SIZE)):
        mask = mask_gen_fn(s)
        # 这里必须用 range_constexpr，否则编译器无法生成 R2P 指令
        for i in cutlass.range_constexpr(min(CHUNK_SIZE, ncol - s * CHUNK_SIZE)):
            in_bound = cutlass.Boolean(mask & (Uint32(1) << i))
            c = s * CHUNK_SIZE + i
            if const_expr(rank1):
                X[c] = X[c] if in_bound else -Float32.inf
            else:
                for r in cutlass.range_constexpr(cute.size(X.shape[0])):
                    X[r, c] = X[r, c] if in_bound else -Float32.inf


@cute.jit
def sm90_col_to_r2p_idx(col_limit: Int32) -> Int32:
    """把 SM90 MMA 的列坐标转换为 R2P 元素索引。

    SM90 MMA 累加器的列索引不连续：0, 1, 8, 9, 16, 17, ...
    元素索引是连续的：0, 1, 2, 3, 4, 5, ...
    本函数把列空间的阈值转换为元素空间的阈值，供 r2p_bitmask_below/above 使用。
    """
    return col_limit // 8 * 2 + min(col_limit % 8, 2)


@cute.jit
def row_to_r2p_idx(x: Int32, num_rep: int, num_wg: int) -> Int32:
    """把行坐标转换为 warp-group 交错布局中的 R2P 元素索引。

    在 SM100 反向传播中，2 个 warp group 共享 TMEM。TMEM 加载原子指令按交错
    模式分布行：元素 0..num_rep-1 映射到行 0..num_rep-1（warp group 0），
    元素 num_rep..2*num_rep-1 映射到行 num_rep*num_wg..num_rep*num_wg+num_rep-1
    （warp group 1），以此类推。行坐标阈值（causal 上限、窗口边界、uih_len）
    必须先转换为元素索引，才能用于 r2p_bitmask_above/below。

    本线程不拥有的行（位于 warp group 之间的空隙）被钳制（clamp）到边界元素
    索引，这是安全的，因为 R2P 阈值是单调的。

    num_rep=16、num_wg=2 的示例：
        row  0 -> elem  0,  row 15 -> elem 15,
        row 16 -> elem 16 (clamped), row 31 -> elem 16 (clamped),
        row 32 -> elem 16, row 33 -> elem 17, row 47 -> elem 31.
    """
    return x // (num_rep * num_wg) * num_rep + min(x % (num_rep * num_wg), num_rep)


@cute.jit
def apply_packed_mask_chunk(
    X: cute.Tensor,
    chunk_idx: cutlass.Constexpr[int],
    mask: Uint32,
) -> None:
    """把一条 32 位保留掩码应用到一块 32 列的 chunk 上。

    单次迭代的 chunk 循环保持了与 mask_r2p_lambda 相同的低层化（lowering）模式。
    """
    ncol = const_expr(cute.size(X.shape))
    col_base = chunk_idx * MASK_R2P_CHUNK_SIZE
    for s in cutlass.range_constexpr(1):
        for i in cutlass.range_constexpr(
            min(MASK_R2P_CHUNK_SIZE, ncol - col_base - s * MASK_R2P_CHUNK_SIZE)
        ):
            in_bound = cutlass.Boolean(mask & (Uint32(1) << i))
            c = col_base + s * MASK_R2P_CHUNK_SIZE + i
            X[c] = X[c] if in_bound else -Float32.inf


@dataclass(frozen=True)
class AttentionMask:
    tile_m: cutlass.Constexpr[int]
    tile_n: cutlass.Constexpr[int]
    seqlen_info: SeqlenInfoQK
    window_size_left: Optional[Int32] = None
    window_size_right: Optional[Int32] = None
    qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1  # 仅在使用 PackGQA 时才传入
    swap_AB: cutlass.Constexpr[bool] = False

    @property
    def seqlen_q(self) -> Int32:
        return self.seqlen_info.seqlen_q

    @property
    def seqlen_k(self) -> Int32:
        return self.seqlen_info.seqlen_k

    @cute.jit
    def apply_mask(
        self,
        acc_S: cute.Tensor,
        batch_idx: cutlass.Int32,
        head_idx: cutlass.Int32,
        m_block: cutlass.Int32,
        n_block: cutlass.Int32,
        thr_mma: cute.TiledMma,
        mask_seqlen: cutlass.Constexpr[bool],
        mask_causal: cutlass.Constexpr[bool],
        mask_local: cutlass.Constexpr[bool] = False,
        mask_mod: cutlass.Constexpr[Optional[Callable]] = None,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=(None, None),
    ) -> None:
        assert not (mask_causal and mask_local), "mask_causal and mask_local cannot be both True"
        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S, transpose=self.swap_AB)
        acc_shape = (self.tile_m, self.tile_n)
        cS = cute.make_identity_tensor(acc_shape if not self.swap_AB else acc_shape[::-1])
        tScS_mn = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(cS), transpose=self.swap_AB)
        # 这里使用 t0ScS，因为这些索引在编译期已知；随后必须用线程列偏移减去列限制。
        t0ScS_mn = layout_utils.reshape_acc_to_mn(
            thr_mma.get_slice(0).partition_C(cS), transpose=self.swap_AB
        )
        ROW = 0 if const_expr(not self.swap_AB) else 1
        COL = 1 if const_expr(not self.swap_AB) else 0
        thr_col_offset = tScS_mn[0][COL]
        # 处理 n_block_max = 0 时整行被完全掩码的边界情况：
        # 把负数 n_block 当作第 0 个 n_block 处理
        # TODO: 寻找更透明的方案
        if n_block < 0:
            n_block = 0
        seqlenk_col_limit = self.seqlen_k - n_block * self.tile_n - thr_col_offset
        if const_expr(not mask_causal and not mask_local and mask_mod is None):
            if const_expr(mask_seqlen):
                r2p = const_expr(not self.swap_AB)
                if const_expr(not r2p):
                            # 遍历列索引。
                    for c in cutlass.range(cute.size(tScS_mn.shape[1]), unroll_full=True):
                        oob = t0ScS_mn[0, c][COL] >= seqlenk_col_limit
                        for r in cutlass.range(cute.size(tScS_mn.shape[0]), unroll_full=True):
                            acc_S_mn[r, c] = -Float32.inf if oob else acc_S_mn[r, c]
                else:
                    seqlenk_col_limit_r2p = sm90_col_to_r2p_idx(seqlenk_col_limit)
                    mask_r2p_lambda(acc_S_mn, lambda s: r2p_bitmask_below(seqlenk_col_limit_r2p, s))

        elif const_expr(
            not mask_causal and not mask_local and mask_mod is not None
        ):  # FlexAttention 掩码修改器
            nrow = const_expr(cute.size(tScS_mn.shape[0]))
            ncol = const_expr(cute.size(tScS_mn.shape[1]))
            has_fastdiv = const_expr(
                fastdiv_mods is not None
                and fastdiv_mods[0] is not None
                and fastdiv_mods[1] is not None
            )
            wrap_aux_indices = const_expr(
                has_fastdiv and mask_seqlen and const_expr(aux_data.tensors is not None)
            )

            for r in cutlass.range_constexpr(nrow):
                # 尊重 swap_AB：ROW/COL 决定哪个坐标分量对应 Q/KV。
                local_row = tScS_mn[r, 0][ROW]
                global_row_idx = local_row + m_block * self.tile_m
                row_for_mod = global_row_idx
                head_idx_for_mod = head_idx
                if const_expr(self.qhead_per_kvhead_packgqa != 1):
                    head_offset = global_row_idx % self.qhead_per_kvhead_packgqa
                    head_idx_for_mod = head_idx * self.qhead_per_kvhead_packgqa + head_offset
                    row_for_mod = global_row_idx // self.qhead_per_kvhead_packgqa
                row_for_seqlen = row_for_mod
                if const_expr(wrap_aux_indices):
                    _, row_for_mod = divmod(row_for_mod, fastdiv_mods[0])

                for col in cutlass.range_constexpr(ncol):
                    col_idx_local = t0ScS_mn[0, col][COL]
                    # 转换为绝对列索引
                    global_col_idx = thr_col_offset + col_idx_local + n_block * self.tile_n
                    col_for_mod = global_col_idx
                    if const_expr(wrap_aux_indices):
                        _, col_for_mod = divmod(global_col_idx, fastdiv_mods[1])

                    batch_idx_ssa = utils.scalar_to_ssa(batch_idx, cutlass.Int32)
                    head_idx_ssa = utils.scalar_to_ssa(head_idx_for_mod, cutlass.Int32)
                    q_idx_ssa = utils.scalar_to_ssa(row_for_mod, cutlass.Int32)
                    kv_idx_ssa = utils.scalar_to_ssa(col_for_mod, cutlass.Int32)
                    mask_value = call_mask_mod(
                        mask_mod,
                        batch_idx_ssa,
                        head_idx_ssa,
                        q_idx_ssa,
                        kv_idx_ssa,
                        self.seqlen_info,
                        aux_data,
                    )
                    cond = cutlass.Boolean(utils.ssa_to_scalar(mask_value))
                    if const_expr(mask_seqlen):
                        out_of_bounds = (row_for_seqlen >= self.seqlen_q) or (
                            global_col_idx >= self.seqlen_k
                        )
                        if out_of_bounds:
                            acc_S_mn[r, col] = -cutlass.Float32.inf
                        else:
                            acc_S_mn[r, col] = acc_S_mn[r, col] if cond else -cutlass.Float32.inf
                    else:
                        acc_S_mn[r, col] = acc_S_mn[r, col] if cond else -cutlass.Float32.inf

        else:  # Causal 或 local
            if const_expr(not self.swap_AB):
                # 若使用 PackGQA，把 divmod 的计算分摊给同一行内的多个线程
                threads_per_row = thr_mma.tv_layout_C.shape[0][0]
                mma_m_idx = None
                if const_expr(self.qhead_per_kvhead_packgqa != 1):
                    assert not self.swap_AB, "swap_AB with PackGQA not supported yet"
                    assert cute.arch.WARP_SIZE % threads_per_row == 0, (
                        "threads_per_row must divide WARP_SIZE"
                    )
                    assert cute.size(acc_S_mn.shape[0]) <= threads_per_row
                    tidx = thr_mma.thr_idx
                    mma_m_idx = (
                        m_block * self.tile_m + tScS_mn[tidx % threads_per_row, 0][0]
                    ) // self.qhead_per_kvhead_packgqa
                causal_row_offset = (
                    1 + self.seqlen_k - n_block * self.tile_n - self.seqlen_q - thr_col_offset
                )
                if const_expr(mask_causal):
                    r2p = const_expr(not self.swap_AB)  # R2P 技巧，参见 apply_mask_sm100
                    for r in cutlass.range(cute.size(tScS_mn.shape[0]), unroll_full=True):
                        # 根据当前行计算列索引上限；只考虑行索引，因此列索引设为 0。
                        if const_expr(self.qhead_per_kvhead_packgqa == 1):
                            row_idx = tScS_mn[r, 0][0] + m_block * self.tile_m
                        else:
                            row_idx = utils.shuffle_sync(
                                mma_m_idx, r % threads_per_row, width=threads_per_row
                            )
                        col_limit_right = row_idx + causal_row_offset
                        if const_expr(mask_seqlen):
                            col_limit_right = cutlass.min(col_limit_right, seqlenk_col_limit)
                        if const_expr(not r2p):
                            # 遍历列索引。
                            for c in cutlass.range(cute.size(tScS_mn.shape[1]), unroll_full=True):
                                acc_S_mn[r, c] = (
                                    -Float32.inf
                                    if t0ScS_mn[0, c][1] >= col_limit_right
                                    else acc_S_mn[r, c]
                                )
                        else:
                            col_limit_r2p = sm90_col_to_r2p_idx(col_limit_right)
                            mask_r2p_lambda(
                                acc_S_mn[r, None],
                                lambda s: r2p_bitmask_below(col_limit_r2p, s),
                                rank1=True,
                            )
                else:  # Local
                    local_row_offset_right = (
                        causal_row_offset + self.window_size_right
                        if const_expr(self.window_size_right is not None)
                        else None
                    )
                    local_row_offset_left = (
                        causal_row_offset - 1 - self.window_size_left
                        if const_expr(self.window_size_left is not None)
                        else None
                    )
                    r2p_local = const_expr(not self.swap_AB)
                    for r in cutlass.range(cute.size(tScS_mn.shape[0]), unroll_full=True):
                        if const_expr(self.qhead_per_kvhead_packgqa == 1):
                            row_idx = tScS_mn[r, 0][0] + m_block * self.tile_m
                        else:
                            row_idx = utils.shuffle_sync(
                                mma_m_idx, r % threads_per_row, width=threads_per_row
                            )
                        if const_expr(self.window_size_right is not None):
                            col_limit_right = row_idx + local_row_offset_right
                        else:
                            col_limit_right = self.tile_n
                        if const_expr(mask_seqlen):
                            col_limit_right = cutlass.min(col_limit_right, seqlenk_col_limit)
                        col_limit_left = (
                            row_idx + local_row_offset_left
                            if const_expr(self.window_size_left is not None)
                            else 0
                        )
                        if const_expr(not r2p_local):
                            # 遍历列索引。
                            for c in cutlass.range(cute.size(tScS_mn.shape[1]), unroll_full=True):
                                col_idx = t0ScS_mn[0, c][1]
                                if col_idx >= col_limit_right or col_idx < col_limit_left:
                                    acc_S_mn[r, c] = -Float32.inf
                        else:
                            col_limit_right_r2p = sm90_col_to_r2p_idx(col_limit_right)
                            col_limit_left_r2p = sm90_col_to_r2p_idx(col_limit_left)

                            def mask_gen_fn(s: int) -> Uint32:
                                return r2p_bitmask_below(
                                    col_limit_right_r2p, s
                                ) & r2p_bitmask_above(col_limit_left_r2p, s)

                            mask_r2p_lambda(acc_S_mn[r, None], mask_gen_fn, rank1=True)
            else:  # swap_AB 情况
                assert self.qhead_per_kvhead_packgqa == 1
                thr_row_offset = tScS_mn[0][ROW]
                causal_row_offset = (
                    seqlenk_col_limit - self.seqlen_q + m_block * self.tile_m + thr_row_offset
                )
                if const_expr(mask_causal):
                    for c in cutlass.range(cute.size(tScS_mn.shape[1]), unroll_full=True):
                        col0 = t0ScS_mn[0, c][COL]
                        # 若 col0 超出列限制，则通过把行上限设为 self.tile_m 来掩掉整列
                        #（即整列置 -inf）
                        row_limit_top = (
                            self.tile_m
                            if col0 >= seqlenk_col_limit and mask_seqlen
                            else col0 - causal_row_offset
                        )
                        for r in cutlass.range(cute.size(tScS_mn.shape[0]), unroll_full=True):
                            acc_S_mn[r, c] = (
                                -Float32.inf
                                if t0ScS_mn[r, 0][ROW] < row_limit_top
                                else acc_S_mn[r, c]
                            )
                else:
                    for c in cutlass.range(cute.size(tScS_mn.shape[1]), unroll_full=True):
                        col0 = t0ScS_mn[0, c][COL]
                        # 若 col0 超出列限制，则通过把行上限设为 self.tile_m 来掩掉整列
                        #（即整列置 -inf）
                        row_limit_top = (
                            self.tile_m
                            if col0 >= seqlenk_col_limit and mask_seqlen
                            else (
                                col0 - causal_row_offset - self.window_size_right
                                if const_expr(self.window_size_right is not None)
                                else 0
                            )
                        )
                        row_limit_bot = (
                            col0 - causal_row_offset + self.window_size_left
                            if const_expr(self.window_size_left is not None)
                            else self.tile_m
                        )
                        for r in cutlass.range(cute.size(tScS_mn.shape[0]), unroll_full=True):
                            row_idx = t0ScS_mn[r, 0][ROW]
                            acc_S_mn[r, c] = (
                                -Float32.inf
                                if row_idx < row_limit_top or row_idx > row_limit_bot
                                else acc_S_mn[r, c]
                            )

    @cute.jit
    def apply_mask_mod_sm100_scalar(
        self,
        acc_S: cute.Tensor,
        tScS_t2r: cute.Tensor,
        m_block: Int32,
        n_block: Int32,
        mask_seqlen: cutlass.Constexpr[bool],
        mask_mod: cutlass.Constexpr[Callable],
        batch_idx: Int32,
        head_idx: Int32,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=(None, None),
        head_divmod=None,
        check_q_boundary: bool = False,
    ) -> None:
        """把标量 FlexAttention mask_mod 应用到 SM100 累加器 fragment 上。

        每个累加器通道用逻辑 (batch, head, q, kv) 索引调用一次 mask_mod。
        Pack-GQA 行在调用前会被转换回逻辑 q/head 索引。存在 aux 张量时，
        用 fastdiv 对索引做环绕，使 mask_mod 永远不会越界读取逐样本的辅助存储。
        """
        has_fastdiv = const_expr(
            fastdiv_mods is not None and fastdiv_mods[0] is not None and fastdiv_mods[1] is not None
        )
        batch_idx_ssa = utils.scalar_to_ssa(batch_idx, cutlass.Int32)
        ncol = const_expr(cute.size(tScS_t2r.shape))

        for i in cutlass.range_constexpr(ncol):
            row_coord = tScS_t2r[i][0] if not self.swap_AB else tScS_t2r[i][1]
            col_coord = tScS_t2r[i][1] if not self.swap_AB else tScS_t2r[i][0]
            global_row = row_coord + m_block * self.tile_m
            global_col = col_coord + n_block * self.tile_n

            if const_expr(self.qhead_per_kvhead_packgqa != 1):
                assert head_divmod is not None
                mask_row, head_offset = divmod(global_row, head_divmod)
                head_idx_for_mod = head_idx * self.qhead_per_kvhead_packgqa + head_offset
            else:
                head_idx_for_mod = head_idx
                mask_row = global_row

            mask_row_for_mod = mask_row
            if const_expr(has_fastdiv and aux_data.tensors is not None):
                if check_q_boundary:
                    _, mask_row_for_mod = divmod(mask_row, fastdiv_mods[0])
            global_col_for_mod = global_col
            if const_expr(has_fastdiv and mask_seqlen and aux_data.tensors is not None):
                _, global_col_for_mod = divmod(global_col, fastdiv_mods[1])

            head_idx_ssa = utils.scalar_to_ssa(head_idx_for_mod, cutlass.Int32)
            mask_row_ssa = utils.scalar_to_ssa(mask_row_for_mod, cutlass.Int32)
            kv_idx_ssa = utils.scalar_to_ssa(global_col_for_mod, cutlass.Int32)
            mask_value = call_mask_mod(
                mask_mod,
                batch_idx_ssa,
                head_idx_ssa,
                mask_row_ssa,
                kv_idx_ssa,
                self.seqlen_info,
                aux_data,
            )
            cond = cutlass.Boolean(utils.ssa_to_scalar(mask_value))
            acc_S[i] = acc_S[i] if cond else -Float32.inf
            if const_expr(mask_seqlen):
                acc_S[i] = -Float32.inf if global_col >= self.seqlen_k else acc_S[i]
            if check_q_boundary:
                acc_S[i] = -Float32.inf if mask_row >= self.seqlen_q else acc_S[i]

    @cute.jit
    def apply_mask_mod_sm100_vector(
        self,
        acc_S: cute.Tensor,
        tScS_t2r: cute.Tensor,
        m_block: Int32,
        n_block: Int32,
        mask_seqlen: cutlass.Constexpr[bool],
        mask_mod: cutlass.Constexpr[Callable],
        batch_idx: Int32,
        head_idx: Int32,
        vec_size: cutlass.Constexpr[int],
        aux_data: AuxData = AuxData(),
        fastdiv_mods=(None, None),
        head_divmod=None,
        check_q_boundary: bool = False,
    ) -> None:
        """把向量化的 FlexAttention mask_mod 应用到 SM100 fragment 上。

        mask_mod 接收一个逻辑 q 行的 vec_size 个相邻 KV 索引，返回按位打包的
        Uint32 保留掩码。低位对应较小的 KV 索引。打包后的掩码与序列边界检查
        相结合，再按 32 列 chunk 应用，使最终掩码低层化为 R2P。
        """
        has_fastdiv = const_expr(
            fastdiv_mods is not None and fastdiv_mods[0] is not None and fastdiv_mods[1] is not None
        )
        batch_idx_ssa = utils.scalar_to_ssa(batch_idx, cutlass.Int32)
        ncol = const_expr(cute.size(tScS_t2r.shape))
        mask_vals_per_apply = const_expr(max(1, vec_size // 32))
        calls_per_apply = const_expr(max(1, 32 // vec_size))
        n_calls = const_expr(cute.ceil_div(ncol, vec_size))
        mask_vals = cute.make_rmem_tensor(mask_vals_per_apply, dtype=cutlass.Uint32)

        # 累积足够的向量化 mask_mod 调用，以产生可被 apply_packed_mask_chunk
        # 低层化为 R2P 的 32 位 chunk。
        for s in cutlass.range_constexpr(n_calls):
            if const_expr(s % calls_per_apply == 0):
                for c in cutlass.range_constexpr(mask_vals_per_apply):
                    mask_vals[c] = cutlass.Uint32(0)
            i = s * vec_size
            row_coord = tScS_t2r[i][0] if not self.swap_AB else tScS_t2r[i][1]
            col_coord = tScS_t2r[i][1] if not self.swap_AB else tScS_t2r[i][0]
            global_row = row_coord + m_block * self.tile_m
            global_col = col_coord + n_block * self.tile_n
            if const_expr(self.qhead_per_kvhead_packgqa != 1):
                assert head_divmod is not None
                mask_row, head_offset = divmod(global_row, head_divmod)
                head_idx_for_mod = head_idx * self.qhead_per_kvhead_packgqa + head_offset
            else:
                head_idx_for_mod = head_idx
                mask_row = global_row
            mask_row_for_mod = mask_row
            if const_expr(has_fastdiv and aux_data.tensors is not None):
                if check_q_boundary:
                    _, mask_row_for_mod = divmod(mask_row, fastdiv_mods[0])

            head_idx_ssa = utils.scalar_to_ssa(head_idx_for_mod, cutlass.Int32).broadcast_to(
                (vec_size,)
            )
            mask_row_ssa = utils.scalar_to_ssa(mask_row_for_mod, cutlass.Int32).broadcast_to(
                (vec_size,)
            )
            batch_idx_ssa_call = batch_idx_ssa.broadcast_to((vec_size,))
            kv_idx_vec = cute.make_rmem_tensor(vec_size, cutlass.Int32)

            # 为这次向量化 mask_mod 调用构造每个通道的 KV 索引。
            for j in cutlass.range_constexpr(min(vec_size, ncol - i)):
                col_j_coord = tScS_t2r[i + j][1] if not self.swap_AB else tScS_t2r[i + j][0]
                col_j_global = col_j_coord + n_block * self.tile_n
                col_j_for_mod = col_j_global
                if const_expr(has_fastdiv and mask_seqlen and aux_data.tensors is not None):
                    _, col_j_for_mod = divmod(col_j_global, fastdiv_mods[1])
                kv_idx_vec[j] = col_j_for_mod
            kv_idx_ssa = kv_idx_vec.load()

            # mask_value 已由向量化 mask_mod 按位打包。
            mask_value = call_mask_mod(
                mask_mod,
                batch_idx_ssa_call,
                head_idx_ssa,
                mask_row_ssa,
                kv_idx_ssa,
                self.seqlen_info,
                aux_data,
            )

            # 当 vec_size < 32 时，多次 mask_mod 调用填满一个 R2P chunk。
            bit_offset = const_expr((s % calls_per_apply) * vec_size)
            seqlen_thresh_call = (
                self.seqlen_k - global_col if const_expr(mask_seqlen) else cutlass.Int32(0)
            )
            q_in_bounds = mask_row < self.seqlen_q if check_q_boundary else cutlass.Boolean(True)
            for c in cutlass.range_constexpr(mask_vals_per_apply):
                mask_val = mask_value[c]
                if const_expr(vec_size < 32):
                    lane_keep = utils.shr_u32(
                        cutlass.Uint32(0xFFFFFFFF),
                        cutlass.Uint32(32 - vec_size),
                    )
                    mask_val = mask_val & lane_keep
                if const_expr(mask_seqlen):
                    mask_val = mask_val & r2p_bitmask_below(seqlen_thresh_call, c)
                if check_q_boundary:
                    mask_val = mask_val if q_in_bounds else cutlass.Uint32(0)
                mask_vals[c] = mask_vals[c] | (mask_val << bit_offset)

            # 仅在 32 位 chunk 填满或到达 tile 尾部时才应用掩码。
            is_last_in_apply = const_expr(s % calls_per_apply == calls_per_apply - 1)
            is_last_overall = const_expr(s == n_calls - 1)
            if const_expr(is_last_in_apply or is_last_overall):
                apply_idx = s // calls_per_apply
                for c in cutlass.range_constexpr(mask_vals_per_apply):
                    chunk_idx = apply_idx * mask_vals_per_apply + c
                    # 跳过起始位置超出累加器 fragment 的打包 chunk。
                    if const_expr(chunk_idx * 32 < ncol):
                        apply_packed_mask_chunk(acc_S, chunk_idx, mask_vals[c])

    @cute.jit
    def apply_mask_sm100(
        self,
        acc_S: cute.Tensor,
        m_block: Int32,
        n_block: Int32,
        thr_mma: cute.TiledMma,
        thr_tmem_load: cute.TiledCopy,
        mask_seqlen: cutlass.Constexpr[bool],
        mask_causal: cutlass.Constexpr[bool],
        mask_local: cutlass.Constexpr[bool] = False,
        mask_mod: cutlass.Constexpr[Optional[Callable]] = None,
        batch_idx: Int32 = None,
        head_idx: Int32 = None,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=(None, None),
        head_divmod=None,
        vec_size: cutlass.Constexpr[int] = 1,
        check_q_boundary: bool = False,
        r2p: bool = True,
        rBitmask: Optional[cute.Tensor] = None,
    ) -> None:
        assert not (mask_causal and mask_local), "mask_causal and mask_local cannot be both True"
        acc_shape = (self.tile_m, self.tile_n)
        cS = cute.make_identity_tensor(acc_shape if not self.swap_AB else acc_shape[::-1])
        tScS = thr_mma.partition_C(cS)
        tScS = tScS[(None, None), 0, 0]
        tScS_t2r = thr_tmem_load.partition_D(tScS)
        # 处理 n_block_max = 0 时整行被完全掩码的边界情况：
        # 把负数 n_block 当作第 0 个 n_block 处理
        # TODO: 寻找更透明的方案
        if n_block < 0:
            n_block = 0
        seqlenk_col_limit = self.seqlen_k - n_block * self.tile_n

        if const_expr(rBitmask is not None):
            ncol_packed = const_expr(cute.size(rBitmask.shape[0]))
            for i in cutlass.range_constexpr(ncol_packed):
                col_start = 32 * i  # 掩码按位打包进 uint32
                curr_mask_val = rBitmask[i]
                for j in cutlass.range_constexpr(32):
                    curr_col = col_start + j
                    mask = (curr_mask_val >> j) & 1
                    acc_S[curr_col] = acc_S[curr_col] if cutlass.Boolean(mask) else -Float32.inf

        elif const_expr(not mask_causal and not mask_local and mask_mod is None):
            if const_expr(mask_seqlen):
                if const_expr(not r2p):
                    for i in cutlass.range(cute.size(tScS_t2r.shape), unroll_full=True):
                        # if tScS_t2r[i][1] >= seqlenk_col_limit:
                        #     acc_S[i] = -Float32.inf
                        # 由于某种原因，上面两行会生成非常糟糕的 SASS
                        acc_S[i] = -Float32.inf if tScS_t2r[i][1] >= seqlenk_col_limit else acc_S[i]
                else:
                    mask_r2p_lambda(
                        acc_S,
                        lambda s: r2p_bitmask_below(seqlenk_col_limit, s),
                        rank1=True,
                    )

        elif const_expr(not mask_causal and not mask_local and mask_mod is not None):
            # FlexAttention mask_mod 的向量化由 `mask_mod.__vec_size__` 控制。
            # vec_size == 1 时返回标量 Boolean；vec_size > 1 时返回打包的
            # Uint32 掩码 fragment：每 32 个被求值的列对应一个 word。
            assert vec_size % 32 == 0 or 32 % vec_size == 0, (
                "vec_size must divide 32 or be a multiple of 32"
            )
            if const_expr(vec_size == 1):
                self.apply_mask_mod_sm100_scalar(
                    acc_S,
                    tScS_t2r,
                    m_block,
                    n_block,
                    mask_seqlen,
                    mask_mod,
                    batch_idx,
                    head_idx,
                    aux_data,
                    fastdiv_mods,
                    head_divmod,
                    check_q_boundary,
                )
            else:
                self.apply_mask_mod_sm100_vector(
                    acc_S,
                    tScS_t2r,
                    m_block,
                    n_block,
                    mask_seqlen,
                    mask_mod,
                    batch_idx,
                    head_idx,
                    vec_size,
                    aux_data,
                    fastdiv_mods,
                    head_divmod,
                    check_q_boundary,
                )

        else:  # Causal 或 local
            causal_row_offset = self.seqlen_k - n_block * self.tile_n - self.seqlen_q
            row_idx = tScS_t2r[0][0] + m_block * self.tile_m
            if const_expr(self.qhead_per_kvhead_packgqa != 1):
                row_idx = row_idx // self.qhead_per_kvhead_packgqa
            if const_expr(mask_causal):
                # 讲解：causal 掩码在 kernel 内退化为"每行只保留对角线以左的列"：
                # col_limit_right = row_idx + causal_row_offset，超过该列限的
                # 元素直接置 -inf，无需额外的掩码矩阵。
                col_limit_right = row_idx + causal_row_offset + 1
                if const_expr(mask_seqlen):
                    col_limit_right = cutlass.min(col_limit_right, seqlenk_col_limit)
                # if cute.arch.thread_idx()[0] % 32 == 0:
                #     cute.printf("tidx = %d, tidx tmem = %d, row_idx = %d, col_limit_right = %d, causal_row_offset = %d\n", cute.arch.thread_idx()[0], thr_tmem_load.thr_idx, row_idx, col_limit_right, causal_row_offset)
                ncol = const_expr(cute.size(tScS_t2r.shape))
                if const_expr(not r2p):
                    for i in cutlass.range(ncol, unroll_full=True):
                        acc_S[i] = -Float32.inf if tScS_t2r[i][1] >= col_limit_right else acc_S[i]
                else:
                    mask_r2p_lambda(
                        acc_S,
                        lambda s: r2p_bitmask_below(col_limit_right, s),
                        rank1=True,
                    )
            else:
                local_row_offset_right = (
                    causal_row_offset + 1 + self.window_size_right
                    if const_expr(self.window_size_right is not None)
                    else None
                )
                local_row_offset_left = (
                    causal_row_offset - self.window_size_left
                    if const_expr(self.window_size_left is not None)
                    else None
                )
                if const_expr(self.window_size_right is not None):
                    col_limit_right = row_idx + local_row_offset_right
                else:
                    col_limit_right = self.tile_n
                if const_expr(mask_seqlen):
                    col_limit_right = cutlass.min(col_limit_right, seqlenk_col_limit)
                col_limit_left = (
                    row_idx + local_row_offset_left
                    if const_expr(self.window_size_left is not None)
                    else 0
                )
                if const_expr(not r2p):
                    # if cute.arch.thread_idx()[0] == 0 or cute.arch.thread_idx()[0] == 128: cute.printf("m_block = {}, n_block = {}, row_idx = {}, causal_row_offset = {}, col_limit_right = {}, col_limit_left = {}", m_block, n_block, row_idx, causal_row_offset, col_limit_right, col_limit_left)
                    for i in cutlass.range(cute.size(tScS_t2r.shape), unroll_full=True):
                        col_idx = tScS_t2r[i][1]
                        acc_S[i] = (
                            -Float32.inf
                            if col_idx >= col_limit_right or col_idx < col_limit_left
                            else acc_S[i]
                        )
                else:
                    # SM100 的双边界 R2P 掩码。
                    # 掩码掉满足以下条件的元素：NOT (col_limit_left <= col < col_limit_right)

                    def mask_gen_fn(s: int) -> Uint32:
                        return r2p_bitmask_below(col_limit_right, s) & r2p_bitmask_above(
                            col_limit_left, s
                        )

                    mask_r2p_lambda(acc_S, mask_gen_fn, rank1=True)

    @cute.jit
    def apply_mask_sm100_transposed(
        self,
        acc_S: cute.Tensor,
        tScS_t2r: cute.Tensor,
        t0ScS_t2r: cute.Tensor,
        m_block: cutlass.Int32,
        n_block: cutlass.Int32,
        mask_seqlen: cutlass.Constexpr,
        mask_causal: cutlass.Constexpr,
        mask_local: cutlass.Constexpr,
        mask_mod: cutlass.Constexpr[Optional[Callable]] = None,
        batch_idx: Int32 = None,
        head_idx: Int32 = None,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=(None, None),
        is_full_block: bool = False,
        check_m_boundary: bool = True,
    ) -> None:
        """
        反向传播：掩码 S = K @ Q.T，其中 n_block 分块 seqlen_k，m_block 分块 seqlen_q。

        坐标约定：
        - ROW 对应 Q（m_block）
        - COL 对应 KV（n_block）

        is_full_block: 若为 True，跳过 mask_mod（所有元素有效），只应用 seqlen 掩码。
        check_m_boundary: 若为 False，跳过 seqlen_q 边界检查（针对非边界 m_block 的优化）。
                          按正序迭代 m_block 时，只有最后一个 m_block 可能是部分块。
        """
        assert not (mask_causal and mask_local), "mask_causal and mask_local cannot be both True"
        ROW = 0 if const_expr(not self.swap_AB) else 1
        COL = 1 if const_expr(not self.swap_AB) else 0
        # assert t0ScS_t2r[0][COL] == 0, "col0 == 0" # 2-cta bwd 的临时注释
        thr_col_offset = tScS_t2r[0][COL]
        seqlenk_col_limit = self.seqlen_k - n_block * self.tile_n - thr_col_offset

        if const_expr(not mask_causal and not mask_local and mask_mod is not None):
            # 带 mask_mod 的块稀疏场景（反向传播）
            #
            # 坐标约定：ROW → Q（m_block），COL → KV（n_block）。
            # 这些已经考虑了 swap_AB。
            #
            # FULL 块：mask_mod 对所有元素返回 True，因此跳过它。
            #   仍需 seqlen 边界检查（最后一个 m_block 的元素可能越界）。
            # PARTIAL 块：逐元素应用 mask_mod，再做 seqlen 边界检查。
            if is_full_block:
                if const_expr(mask_seqlen):
                    if seqlenk_col_limit <= 0:
                        # 整个 tile 对 K 来说都已越界（OOB）
                        for i in cutlass.range(cute.size(acc_S.shape), unroll_full=True):
                            acc_S[i] = -cutlass.Float32.inf
                    elif check_m_boundary:
                        # 最后一个 m_block：检查 Q 和 K 的边界
                        ncol = const_expr(cute.size(tScS_t2r.shape))
                        for i in cutlass.range_constexpr(ncol):
                            row_coord = tScS_t2r[i][ROW]
                            col_coord = tScS_t2r[i][COL]
                            global_q = row_coord + m_block * self.tile_m
                            global_kv = col_coord + n_block * self.tile_n
                            q_out_of_bounds = global_q >= self.seqlen_q
                            kv_out_of_bounds = global_kv >= self.seqlen_k
                            out_of_bounds = q_out_of_bounds or kv_out_of_bounds
                            acc_S[i] = -cutlass.Float32.inf if out_of_bounds else acc_S[i]
            else:
                # 部分块
                has_fastdiv = const_expr(
                    fastdiv_mods is not None
                    and fastdiv_mods[0] is not None
                    and fastdiv_mods[1] is not None
                )
                wrap_aux_indices = const_expr(
                    has_fastdiv and mask_seqlen and const_expr(aux_data.tensors is not None)
                )
                batch_idx_ssa = utils.scalar_to_ssa(batch_idx, cutlass.Int32)
                head_idx_ssa = utils.scalar_to_ssa(head_idx, cutlass.Int32)

                ncol = const_expr(cute.size(tScS_t2r.shape))
                for i in cutlass.range_constexpr(ncol):
                    row_coord = tScS_t2r[i][ROW]
                    col_coord = tScS_t2r[i][COL]
                    global_q = row_coord + m_block * self.tile_m
                    global_kv = col_coord + n_block * self.tile_n

                    q_idx_for_mod = global_q
                    kv_idx_for_mod = global_kv
                    if const_expr(wrap_aux_indices):
                        _, q_idx_for_mod = divmod(global_q, fastdiv_mods[0])
                        _, kv_idx_for_mod = divmod(global_kv, fastdiv_mods[1])

                    q_idx_ssa = utils.scalar_to_ssa(q_idx_for_mod, cutlass.Int32)
                    kv_idx_ssa = utils.scalar_to_ssa(kv_idx_for_mod, cutlass.Int32)

                    mask_value = call_mask_mod(
                        mask_mod,
                        batch_idx_ssa,
                        head_idx_ssa,
                        q_idx_ssa,
                        kv_idx_ssa,
                        self.seqlen_info,
                        aux_data,
                    )
                    cond = cutlass.Boolean(utils.ssa_to_scalar(mask_value))
                    acc_S[i] = acc_S[i] if cond else -cutlass.Float32.inf

                    if const_expr(mask_seqlen):
                        # check_m_boundary=False 时跳过非边界 m_block 的 q 检查
                        q_out_of_bounds = check_m_boundary and (global_q >= self.seqlen_q)
                        kv_out_of_bounds = global_kv >= self.seqlen_k
                        out_of_bounds = q_out_of_bounds or kv_out_of_bounds
                        acc_S[i] = -cutlass.Float32.inf if out_of_bounds else acc_S[i]

        elif const_expr(not mask_causal and not mask_local):
            if const_expr(mask_seqlen):
                if seqlenk_col_limit <= 0:
                    for i in cutlass.range(cute.size(acc_S.shape), unroll_full=True):
                        acc_S[i] = -cutlass.Float32.inf
        else:  # Causal 或 local
            thr_row_offset = tScS_t2r[0][ROW]
            seqlenq_row_limit = self.seqlen_q - m_block * self.tile_m - thr_row_offset
            causal_offset = seqlenq_row_limit - seqlenk_col_limit
            if const_expr(mask_causal):
                # tidx = cute.arch.thread_idx()[0] % 256
                # if tidx < 32:
                #     cute.printf("tidx = {}, {} {}, {} {}", tidx, tScS_t2r[0][0], tScS_t2r[0][1], tScS_t2r[1][0], tScS_t2r[1][1])
                row_limit_top = causal_offset
                if const_expr(mask_seqlen):
                    # 若 col 超出列限制，则通过把行上限设为 self.tile_m 来掩掉整列
                    #（即整列置 -inf）
                    if seqlenk_col_limit <= 0:
                        row_limit_top = self.tile_m
                r2p = True
                if const_expr(not r2p):
                    for i in cutlass.range(cute.size(acc_S.shape), unroll_full=True):
                        acc_S[i] = (
                            -cutlass.Float32.inf if t0ScS_t2r[i][ROW] < row_limit_top else acc_S[i]
                        )
                else:
                    num_rep = cute.size(tScS_t2r, mode=[0])  # 16 或 32
                    num_wg = 2
                    row_limit = row_to_r2p_idx(row_limit_top, num_rep, num_wg)
                    mask_r2p_lambda(
                        acc_S,
                        lambda s: r2p_bitmask_above(row_limit, s),
                        rank1=True,
                    )
            else:
                if const_expr(self.window_size_right is not None):
                    row_limit_top = causal_offset - self.window_size_right
                else:
                    row_limit_top = 0
                if const_expr(self.window_size_left is not None):
                    row_limit_bot = causal_offset + self.window_size_left
                if const_expr(mask_seqlen):
                    if seqlenk_col_limit <= 0:
                        row_limit_top = self.tile_m
                r2p = True
                if const_expr(not r2p):
                    for i in cutlass.range(cute.size(acc_S.shape), unroll_full=True):
                        row_idx = t0ScS_t2r[i][ROW]
                        local_mask = row_idx < row_limit_top
                        if const_expr(self.window_size_left is not None):
                            local_mask |= row_idx > row_limit_bot
                        acc_S[i] = -cutlass.Float32.inf if local_mask else acc_S[i]
                else:

                    def mask_gen_fn(s: int) -> Uint32:
                        num_rep = cute.size(tScS_t2r, mode=[0])
                        num_wg = 2

                        row_limit = row_to_r2p_idx(row_limit_top, num_rep, num_wg)
                        mask = r2p_bitmask_above(row_limit, s)

                        if const_expr(self.window_size_left is not None):
                            row_limit_bottom = row_to_r2p_idx(row_limit_bot + 1, num_rep, num_wg)
                            mask = mask & r2p_bitmask_below(row_limit_bottom, s)

                        return mask

                    mask_r2p_lambda(
                        acc_S,
                        mask_gen_fn,
                        rank1=True,
                    )


# -----------------------------------------------------------------------------
# SM100 FMHA 融合掩码策略层（独立于通用掩码原语）。
# -----------------------------------------------------------------------------


class Sm100MaskEnum(enum.Enum):
    """FMHA 操作使用的掩码类型枚举。

    - RESIDUAL_MASK: 用于处理变长序列的残差掩码
    - WINDOW_MASK: 注意力窗口掩码，也涵盖 causal 和无掩码情况
    - WINDOW_MASK_INFERENCE: 与窗口掩码相同，但限定 q 的末尾与 k 的末尾对齐
    - WINDOW_MASK_BWD: 反向传播的窗口掩码
    - WINDOW_MASK_BWD_INFERENCE: 反向传播窗口掩码的推理版本，限定 q 与 k 的末尾对齐
    """

    NO_MASK = enum.auto()
    RESIDUAL_MASK = enum.auto()
    CAUSAL_MASK = enum.auto()
    WINDOW_MASK = enum.auto()
    WINDOW_MASK_INFERENCE = enum.auto()
    # 以下类型已弃用
    WINDOW_MASK_BWD = enum.auto()
    WINDOW_MASK_BWD_INFERENCE = enum.auto()
    RESIDUAL_MASK_BWD = enum.auto()


class Sm100FusedMask:
    """FMHA 操作的融合掩码实现。

    本类处理不同类型的注意力掩码，包括无掩码、用于变长序列的残差掩码、
    以及用于自回归注意力模式的 causal 掩码。

    本类提供的方法：
    - 计算不同掩码类型的迭代次数（trip count）
    - 对注意力分数应用掩码
    - 处理掩码与非掩码迭代次数的计算
    """

    def get_trip_count(
        mask_type: Sm100MaskEnum,
        blk_coord: cute.Coord,
        tile_shape: cute.Shape,
        seqlen_q: Int32,
        seqlen_k: Int32,
        window_size_left: Optional[Int32] = None,
        window_size_right: Optional[Int32] = None,
    ) -> Int32:
        """
        计算当前块所需的迭代次数（trip count）。

        迭代次数取决于掩码类型和块的坐标。对于 causal 掩码，
        需要考虑自回归约束。

        :param mask_type: 要使用的掩码类型
        :type mask_type: utils.Sm100MaskEnum
        :param blk_coord: 块坐标。
        :type blk_coord: cute.Coord
        :param tile_shape: tile 的形状。
        :type tile_shape: cute.Shape
        :param seqlen_q: 注意力计算中 query 的序列长度。
        :type seqlen_q: Int32
        :param seqlen_k: 注意力计算中 key 的序列长度。
        :type seqlen_k: Int32
        :param window_size_left: 注意力掩码的左侧滑动窗口大小。
        :type window_size_left: Optional[Int32]
        :param window_size_right: 注意力掩码的右侧滑动窗口大小。
        :type window_size_right: Optional[Int32]

        :return: 所需的迭代次数。
        :rtype: Int32
        """
        result = 0
        offset = 0
        if cutlass.const_expr(mask_type is Sm100MaskEnum.WINDOW_MASK_INFERENCE):
            offset = seqlen_k - seqlen_q
        if cutlass.const_expr(mask_type is Sm100MaskEnum.WINDOW_MASK_BWD_INFERENCE):
            offset = seqlen_q - seqlen_k
        if cutlass.const_expr(mask_type == Sm100MaskEnum.RESIDUAL_MASK):
            result = cute.ceil_div(seqlen_k, tile_shape[1])
        if cutlass.const_expr(mask_type is Sm100MaskEnum.RESIDUAL_MASK_BWD):
            result = cute.ceil_div(seqlen_q, tile_shape[0])
        if cutlass.const_expr(
            mask_type == Sm100MaskEnum.WINDOW_MASK
            or mask_type == Sm100MaskEnum.WINDOW_MASK_INFERENCE
        ):
            if cutlass.const_expr(window_size_right is None):
                result = cute.ceil_div(seqlen_k, tile_shape[1])
            else:
                max_idx_q = (blk_coord[0] + 1) * tile_shape[0]
                idx_k = max_idx_q + offset + window_size_right
                tmp_blocks_k = cute.ceil_div(idx_k, tile_shape[1])
                max_blocks_k = cute.ceil_div(seqlen_k, tile_shape[1])
                result = dsl_min(max_blocks_k, tmp_blocks_k)
        if cutlass.const_expr(
            mask_type == Sm100MaskEnum.WINDOW_MASK_BWD
            or mask_type == Sm100MaskEnum.WINDOW_MASK_BWD_INFERENCE
        ):
            if cutlass.const_expr(window_size_left is None):
                result = cute.ceil_div(seqlen_q, tile_shape[0])
            else:
                max_idx_k = (blk_coord[1] + 1) * tile_shape[1]
                idx_k = max_idx_k + offset + window_size_left
                tmp_blocks_q = cute.ceil_div(idx_k, tile_shape[0])
                max_blocks_q = cute.ceil_div(seqlen_q, tile_shape[0])
                result = dsl_min(max_blocks_q, tmp_blocks_q)
        start_block = Sm100FusedMask.get_trip_start(
            mask_type,
            blk_coord,
            tile_shape,
            seqlen_q,
            seqlen_k,
            window_size_left,
            window_size_right,
        )
        result = result - start_block
        return result

    @cute.jit
    def get_trip_start_count_via_block_info(
        blk_coord: cute.Coord,
        tile_shape: cute.Shape,
        seqlen_q: Int32,
        seqlen_k: Int32,
        is_causal: cutlass.Constexpr[bool] = False,
        is_local: cutlass.Constexpr[bool] = False,
        window_size_left: Optional[Int32] = None,
        window_size_right: Optional[Int32] = None,
    ) -> Tuple[Int32, Int32]:
        block_info = BlockInfo(
            tile_m=tile_shape[0],
            tile_n=tile_shape[1],
            is_causal=is_causal,
            is_local=is_local and not is_causal,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
        )

        seqlen_info = SeqlenInfoQK(
            offset_q=Int32(0),
            offset_k=Int32(0),
            padded_offset_q=Int32(0),
            padded_offset_k=Int32(0),
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            m_block_offset=Int32(0),
            block_idx_offset=Int32(0),
            num_n_blocks=cute.ceil_div(seqlen_k, tile_shape[1]),
            has_cu_seqlens_q=False,
            has_cu_seqlens_k=False,
            has_seqused_q=False,
            has_seqused_k=False,
            has_cu_block_idx_offsets=False,
        )
        n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen_info, blk_coord[0])
        return n_block_min, n_block_max - n_block_min

    @cute.jit
    def get_trip_mask_bounds_via_block_info(
        blk_coord: cute.Coord,
        tile_shape: cute.Shape,
        seqlen_q: Int32,
        seqlen_k: Int32,
        is_causal: cutlass.Constexpr[bool] = False,
        is_local: cutlass.Constexpr[bool] = False,
        window_size_left: Optional[Int32] = None,
        window_size_right: Optional[Int32] = None,
    ) -> Tuple[Int32, Int32]:
        """返回用于稠密迭代的 SM100 风格掩码边界。

        Returns:
          - n_block_min_causal_local_mask: 右侧掩码区域的起点
          - n_block_min_before_local_mask: 完全无掩码中间区域的起点
        """
        block_info = BlockInfo(
            tile_m=tile_shape[0],
            tile_n=tile_shape[1],
            is_causal=is_causal,
            is_local=is_local and not is_causal,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
        )
        seqlen_info = SeqlenInfoQK(
            offset_q=Int32(0),
            offset_k=Int32(0),
            padded_offset_q=Int32(0),
            padded_offset_k=Int32(0),
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            m_block_offset=Int32(0),
            block_idx_offset=Int32(0),
            num_n_blocks=cute.ceil_div(seqlen_k, tile_shape[1]),
            has_cu_seqlens_q=False,
            has_cu_seqlens_k=False,
            has_seqused_q=False,
            has_seqused_k=False,
            has_cu_block_idx_offsets=False,
        )
        n_block_min, _ = block_info.get_n_block_min_max(seqlen_info, blk_coord[0])
        n_block_min_causal_local_mask = block_info.get_n_block_min_causal_local_mask(
            seqlen_info, blk_coord[0], n_block_min
        )
        n_block_min_before_local_mask = block_info.get_n_block_min_before_local_mask(
            seqlen_info, blk_coord[0], n_block_min
        )
        return n_block_min_causal_local_mask, n_block_min_before_local_mask

    @cute.jit
    def get_trip_start(
        mask_type: Sm100MaskEnum,
        blk_coord: cute.Coord,
        tile_shape: cute.Shape,
        seqlen_q: Int32,
        seqlen_k: Int32,
        window_size_left: Optional[Int32] = None,
        window_size_right: Optional[Int32] = None,
    ) -> Int32:
        """
        获取当前块的迭代起点（trip start）。

        :param mask_type: 要使用的掩码类型
        :type mask_type: utils.Sm100MaskEnum
        :param blk_coord: 块坐标。
        :type blk_coord: cute.Coord
        :param tile_shape: tile 的形状。
        :type tile_shape: cute.Shape
        :param seqlen_q: 注意力计算中 query 的序列长度。
        :type seqlen_q: Int32
        :param seqlen_k: 注意力计算中 key 的序列长度。
        :type seqlen_k: Int32
        :param window_size_left: 注意力掩码的左侧滑动窗口大小。
        :type window_size_left: Optional[Int32]
        :param window_size_right: 注意力掩码的右侧滑动窗口大小。
        :type window_size_right: Optional[Int32]
        """
        result = 0
        offset = 0
        if cutlass.const_expr(mask_type is Sm100MaskEnum.WINDOW_MASK_INFERENCE):
            offset = seqlen_k - seqlen_q
        if cutlass.const_expr(mask_type is Sm100MaskEnum.WINDOW_MASK_BWD_INFERENCE):
            offset = seqlen_q - seqlen_k
        if cutlass.const_expr(
            mask_type is Sm100MaskEnum.WINDOW_MASK
            or mask_type is Sm100MaskEnum.WINDOW_MASK_INFERENCE
        ):
            if cutlass.const_expr(window_size_left is not None):
                min_idx_q = blk_coord[0] * tile_shape[0]
                idx_k = min_idx_q + offset - window_size_left
                tmp_blocks_k = idx_k // tile_shape[1]
                result = max(tmp_blocks_k, result)
        if cutlass.const_expr(
            mask_type is Sm100MaskEnum.WINDOW_MASK_BWD
            or mask_type is Sm100MaskEnum.WINDOW_MASK_BWD_INFERENCE
        ):
            if cutlass.const_expr(window_size_right is not None):
                min_idx_k = blk_coord[1] * tile_shape[1]
                idx_q = min_idx_k + offset - window_size_right
                tmp_blocks_q = idx_q // tile_shape[0]
                result = max(tmp_blocks_q, result)
        return result

    @cute.jit
    def get_leading_mask_id(
        mask_type: Sm100MaskEnum,
        blk_coord: cute.Coord,
        tile_shape: cute.Shape,
        seqlen_q: Int32,
        seqlen_k: Int32,
        window_size_left: Optional[Int32] = None,
        window_size_right: Optional[Int32] = None,
    ) -> Tuple[Int32, Int32]:
        """
        获取 leading 掩码的起始与结束 tile 索引。

        :param mask_type: 要使用的掩码类型
        :type mask_type: utils.Sm100MaskEnum
        :param blk_coord: 块坐标。
        :type blk_coord: cute.Coord
        :param tile_shape: tile 的形状。
        :type tile_shape: cute.Shape
        :param seqlen_q: 注意力计算中 query 的序列长度。
        :type seqlen_q: Int32
        :param seqlen_k: 注意力计算中 key 的序列长度。
        :type seqlen_k: Int32
        :param window_size_left: 注意力掩码的左侧滑动窗口大小。
        :type window_size_left: Optional[Int32]
        :param window_size_right: 注意力掩码的右侧滑动窗口大小。
        :type window_size_right: Optional[Int32]

        :return: leading 掩码的 (起始, 结束) tile 索引。
        :rtype: Tuple[Int32, Int32]
        """
        offset = 0
        if cutlass.const_expr(mask_type is Sm100MaskEnum.WINDOW_MASK_INFERENCE):
            offset = seqlen_k - seqlen_q
        if cutlass.const_expr(mask_type is Sm100MaskEnum.WINDOW_MASK_BWD_INFERENCE):
            offset = seqlen_q - seqlen_k
        leading_mask_begin = Sm100FusedMask.get_trip_start(
            mask_type,
            blk_coord,
            tile_shape,
            seqlen_q,
            seqlen_k,
            window_size_left,
            window_size_right,
        )
        trip_count = Sm100FusedMask.get_trip_count(
            mask_type,
            blk_coord,
            tile_shape,
            seqlen_q,
            seqlen_k,
            window_size_left,
            window_size_right,
        )

        leading_mask_end = leading_mask_begin
        if cutlass.const_expr(
            mask_type is Sm100MaskEnum.WINDOW_MASK
            or mask_type is Sm100MaskEnum.WINDOW_MASK_INFERENCE
        ):
            if cutlass.const_expr(window_size_left is not None):
                min_idx_q = (blk_coord[0] + 1) * tile_shape[0] + offset - window_size_left
                leading_mask_end = dsl_min(
                    cute.ceil_div(min_idx_q, tile_shape[1]) - 1,
                    trip_count + leading_mask_begin - 1,
                )
            else:
                leading_mask_end = leading_mask_begin - 1
        elif cutlass.const_expr(
            mask_type is Sm100MaskEnum.WINDOW_MASK_BWD
            or mask_type is Sm100MaskEnum.WINDOW_MASK_BWD_INFERENCE
        ):
            if cutlass.const_expr(window_size_right is not None):
                min_idx_k = (blk_coord[1] + 1) * tile_shape[1] + offset - window_size_right
                leading_mask_end = cute.ceil_div(min_idx_k, tile_shape[0]) - 1
            else:
                leading_mask_end = leading_mask_begin - 1
        return leading_mask_begin, leading_mask_end

    @cute.jit
    def get_trailing_mask_id(
        mask_type: Sm100MaskEnum,
        blk_coord: cute.Coord,
        tile_shape: cute.Shape,
        seqlen_q: Int32,
        seqlen_k: Int32,
        window_size_left: Optional[Int32] = None,
        window_size_right: Optional[Int32] = None,
    ) -> Tuple[Optional[Int32], Optional[Int32]]:
        """
        获取 trailing 掩码的起始与结束 tile 索引。

        :param mask_type: 要使用的掩码类型
        :type mask_type: utils.Sm100MaskEnum
        :param blk_coord: 块坐标。
        :type blk_coord: cute.Coord
        :param tile_shape: tile 的形状。
        :type tile_shape: cute.Shape
        :param seqlen_q: 注意力计算中 query 的序列长度。
        :type seqlen_q: Int32
        :param seqlen_k: 注意力计算中 key 的序列长度。
        :type seqlen_k: Int32
        :param window_size_left: 注意力掩码的左侧滑动窗口大小。
        :type window_size_left: Optional[Int32]
        :param window_size_right: 注意力掩码的右侧滑动窗口大小。
        :type window_size_right: Optional[Int32]

        :return: trailing 掩码的 (起始, 结束) tile 索引。
        :rtype: Tuple[Int32, Int32]
        """
        offset = 0
        if cutlass.const_expr(mask_type is Sm100MaskEnum.WINDOW_MASK_INFERENCE):
            offset = seqlen_k - seqlen_q
        if cutlass.const_expr(mask_type is Sm100MaskEnum.WINDOW_MASK_BWD_INFERENCE):
            offset = seqlen_q - seqlen_k
        trip_start = Sm100FusedMask.get_trip_start(
            mask_type,
            blk_coord,
            tile_shape,
            seqlen_q,
            seqlen_k,
            window_size_left,
            window_size_right,
        )
        trip_count = Sm100FusedMask.get_trip_count(
            mask_type,
            blk_coord,
            tile_shape,
            seqlen_q,
            seqlen_k,
            window_size_left,
            window_size_right,
        )

        trailing_mask_begin, trailing_mask_end = None, None
        if cutlass.const_expr(
            mask_type is Sm100MaskEnum.WINDOW_MASK
            or mask_type is Sm100MaskEnum.WINDOW_MASK_INFERENCE
        ):
            if cutlass.const_expr(window_size_right is not None):
                min_idx_q = blk_coord[0] * tile_shape[0] + offset + window_size_right
                trailing_mask_begin = dsl_min(
                    min_idx_q // tile_shape[1], trip_count + trip_start - 1
                )
                trailing_mask_end = trip_count + trip_start - 1
            else:
                # 最后一个 tile：无论是否为残差 tile，总是对其应用掩码
                trailing_mask_begin = trip_count + trip_start - 1
                trailing_mask_end = trip_count + trip_start - 1
        else:
            if cutlass.const_expr(window_size_left is not None):
                min_idx_k = blk_coord[1] * tile_shape[1] + offset + window_size_left + 1
                max_idx_k = (blk_coord[1] + 1) * tile_shape[1] + offset + window_size_left
                trailing_mask_begin = dsl_min(
                    cute.ceil_div(min_idx_k, tile_shape[0]) - 1,
                    trip_count + trip_start - 1,
                )
                trailing_mask_end = dsl_min(
                    cute.ceil_div(max_idx_k, tile_shape[0]) - 1,
                    trip_count + trip_start - 1,
                )
            else:
                # 最后一个 tile：无论是否为残差 tile，总是对其应用掩码
                trailing_mask_begin = trip_count + trip_start - 1
                trailing_mask_end = trip_count + trip_start - 1

        return trailing_mask_begin, trailing_mask_end

    @cute.jit
    def get_masked_leading_count(
        mask_type: Sm100MaskEnum,
        blk_coord: cute.Coord,
        tile_shape: cute.Shape,
        seqlen_q: Int32,
        seqlen_k: Int32,
        window_size_left: Optional[Int32] = None,
        window_size_right: Optional[Int32] = None,
    ) -> Int32:
        """
        计算 leading 掩码中被掩码的迭代次数。

        用于因掩码而需要特殊处理的块。

        :param mask_type: 要使用的掩码类型
        :type mask_type: utils.Sm100MaskEnum
        :param blk_coord: 块坐标。
        :type blk_coord: cute.Coord
        :param tile_shape: tile 的形状。
        :type tile_shape: cute.Shape
        :param seqlen_q: 注意力计算中 query 的序列长度。
        :type seqlen_q: Int32
        :param seqlen_k: 注意力计算中 key 的序列长度。
        :type seqlen_k: Int32
        :param window_size_left: 注意力掩码的左侧滑动窗口大小。
        :type window_size_left: Optional[Int32]
        :param window_size_right: 注意力掩码的右侧滑动窗口大小。
        :type window_size_right: Optional[Int32]

        :return: 被掩码的迭代次数。
        :rtype: Int32
        """
        result = 0
        if cutlass.const_expr(
            mask_type is not Sm100MaskEnum.RESIDUAL_MASK
            and mask_type is not Sm100MaskEnum.RESIDUAL_MASK_BWD
        ):
            if cutlass.const_expr(window_size_left is not None or window_size_right is not None):
                leading_mask_begin, leading_mask_end = Sm100FusedMask.get_leading_mask_id(
                    mask_type,
                    blk_coord,
                    tile_shape,
                    seqlen_q,
                    seqlen_k,
                    window_size_left,
                    window_size_right,
                )
                result = max(leading_mask_end - leading_mask_begin + 1, 0)

        return result

    @cute.jit
    def get_masked_trailing_count(
        mask_type: Sm100MaskEnum,
        blk_coord: cute.Coord,
        tile_shape: cute.Shape,
        seqlen_q: Int32,
        seqlen_k: Int32,
        window_size_left: Optional[Int32] = None,
        window_size_right: Optional[Int32] = None,
        rem_count: Optional[Int32] = 0,
    ) -> Int32:
        """
        计算 trailing 掩码中被掩码的迭代次数。

        用于因掩码而需要特殊处理的块。

        :param mask_type: 要使用的掩码类型
        :type mask_type: utils.Sm100MaskEnum
        :param blk_coord: 块坐标。
        :type blk_coord: cute.Coord
        :param tile_shape: tile 的形状。
        :type tile_shape: cute.Shape
        :param seqlen_q: 注意力计算中 query 的序列长度。
        :type seqlen_q: Int32
        :param seqlen_k: 注意力计算中 key 的序列长度。
        :type seqlen_k: Int32
        :param window_size_left: 注意力掩码的左侧滑动窗口大小。
        :type window_size_left: Optional[Int32]
        :param window_size_right: 注意力掩码的右侧滑动窗口大小。
        :type window_size_right: Optional[Int32]
        :param rem_count: 前序计算剩余的迭代数。
        :type rem_count: Int32

        :return: 被掩码的迭代次数。
        :rtype: Int32
        """
        result = 0

        if cutlass.const_expr(
            mask_type is not Sm100MaskEnum.RESIDUAL_MASK
            and mask_type is not Sm100MaskEnum.RESIDUAL_MASK_BWD
        ):
            if cutlass.const_expr(window_size_left is not None or window_size_right is not None):
                trailing_mask_begin, trailing_mask_end = Sm100FusedMask.get_trailing_mask_id(
                    mask_type,
                    blk_coord,
                    tile_shape,
                    seqlen_q,
                    seqlen_k,
                    window_size_left,
                    window_size_right,
                )
                leading_mask_begin, leading_mask_end = Sm100FusedMask.get_leading_mask_id(
                    mask_type,
                    blk_coord,
                    tile_shape,
                    seqlen_q,
                    seqlen_k,
                    window_size_left,
                    window_size_right,
                )
                if cutlass.const_expr(
                    trailing_mask_begin is not None and trailing_mask_end is not None
                ):
                    if trailing_mask_begin <= leading_mask_end:
                        result = max(trailing_mask_end - leading_mask_end, 0)
                    else:
                        result = max(trailing_mask_end - trailing_mask_begin + 1, 0)
        else:
            if seqlen_k % tile_shape[1] != 0:
                result = 1
            else:
                result = 0

        return result + rem_count

    @cute.jit
    def get_unmasked_trip_count(
        mask_type: Sm100MaskEnum,
        blk_coord: cute.Coord,
        tile_shape: cute.Shape,
        seqlen_q: Int32,
        seqlen_k: Int32,
        window_size_left: Optional[Int32] = None,
        window_size_right: Optional[Int32] = None,
    ) -> Int32:
        """
        计算当前块中未掩码的迭代次数。

        表示不需要特殊掩码处理的迭代次数。

        :param mask_type: 要使用的掩码类型
        :type mask_type: utils.Sm100MaskEnum
        :param blk_coord: 块坐标。
        :type blk_coord: cute.Coord
        :param tile_shape: tile 的形状。
        :type tile_shape: cute.Shape
        :param seqlen_q: 注意力计算中 query 的序列长度。
        :type seqlen_q: Int32
        :param seqlen_k: 注意力计算中 key 的序列长度。
        :type seqlen_k: Int32
        :param window_size_left: 注意力掩码的左侧滑动窗口大小。
        :type window_size_left: Optional[Int32]
        :param window_size_right: 注意力掩码的右侧滑动窗口大小。
        :type window_size_right: Optional[Int32]

        :return: 未掩码的迭代次数。
        :rtype: Int32
        """
        result = (
            Sm100FusedMask.get_trip_count(
                mask_type,
                blk_coord,
                tile_shape,
                seqlen_q,
                seqlen_k,
                window_size_left,
                window_size_right,
            )
            - Sm100FusedMask.get_masked_leading_count(
                mask_type,
                blk_coord,
                tile_shape,
                seqlen_q,
                seqlen_k,
                window_size_left,
                window_size_right,
            )
            - Sm100FusedMask.get_masked_trailing_count(
                mask_type,
                blk_coord,
                tile_shape,
                seqlen_q,
                seqlen_k,
                window_size_left,
                window_size_right,
                0,
            )
        )
        return result

    @cute.jit
    def apply_mask(
        mask_type: Sm100MaskEnum,
        acc_qk: cute.Tensor,
        index_qk: cute.Tensor,
        seqlen_q: Int32,
        seqlen_k: Int32,
        window_size_left: Optional[int] = None,
        window_size_right: Optional[int] = None,
        index_transform: cutlass.Constexpr = lambda index_q, index_k: (
            index_q,
            index_k,
        ),
    ):
        """
        对注意力分数应用合适的掩码。

        本方法根据掩码类型和索引张量中的位置修改注意力分数（acc_qk）。

        :param mask_type: 要使用的掩码类型
        :type mask_type: utils.Sm100MaskEnum
        :param acc_qk: QK 注意力分数的累加张量。
        :type acc_qk: cute.Tensor
        :param index_qk: 包含位置信息的索引张量。
        :type index_qk: cute.Tensor
        :param seqlen_k: 注意力计算中 key 的序列长度。
        :type seqlen_k: Int32
        :param seqlen_q: 注意力计算中 query 的序列长度。
        :type seqlen_q: Optional[int]
        :param window_size_left: 注意力掩码的左侧滑动窗口大小。
        :type window_size_left: Optional[int]
        :param window_size_right: 注意力掩码的右侧滑动窗口大小。
        :type window_size_right: Optional[int]
        """
        offset = 0
        # 注意：本仓库的 causal 掩码在 seqlen_k != seqlen_q 时，把 Q 的*末尾*与 K 的*末尾*对齐
        #（与测试/参考实现一致）：
        #   k_index <= q_index + (seqlen_k - seqlen_q) + window_right
        # 在我们的 kernel 中，causal 由 (window_left is None, window_right is not None) 表示。
        if cutlass.const_expr(window_size_left is None and window_size_right is not None):
            offset = seqlen_k - seqlen_q
        elif cutlass.const_expr(
            mask_type is Sm100MaskEnum.WINDOW_MASK_INFERENCE
            or mask_type is Sm100MaskEnum.WINDOW_MASK_BWD_INFERENCE
        ):
            offset = seqlen_k - seqlen_q
        for i in cutlass.range_constexpr(cute.size(acc_qk), unroll_full=True):
            index_q, index_k = index_transform(*index_qk[i])
            if cutlass.const_expr(window_size_left is not None or window_size_right is not None):
                if cutlass.const_expr(window_size_left is None):
                    if index_q + offset + window_size_right < index_k:
                        acc_qk[i] = -Float32.inf
                    if index_k >= seqlen_k or index_q >= seqlen_q:  # 残差掩码
                        acc_qk[i] = -Float32.inf
                elif cutlass.const_expr(window_size_right is None):
                    if index_q + offset - window_size_left > index_k:
                        acc_qk[i] = -Float32.inf
                    if index_k >= seqlen_k or index_q >= seqlen_q:  # 残差掩码
                        acc_qk[i] = -Float32.inf
                else:
                    max_K_index = dsl_min(index_q + offset + window_size_right, seqlen_k)
                    min_K_index = max(0, index_q + offset - window_size_left)
                    if index_k > max_K_index or index_k < min_K_index:
                        acc_qk[i] = -Float32.inf
                    if index_k >= seqlen_k or index_q >= seqlen_q:  # 残差掩码
                        acc_qk[i] = -Float32.inf

            if cutlass.const_expr(
                mask_type == Sm100MaskEnum.RESIDUAL_MASK
                or mask_type == Sm100MaskEnum.RESIDUAL_MASK_BWD
            ):
                if index_k >= seqlen_k or index_q >= seqlen_q:
                    acc_qk[i] = -Float32.inf

    @cute.jit
    def apply_mask_via_causal_local(
        acc_qk: cute.Tensor,
        index_qk: cute.Tensor,
        seqlen_q: Int32,
        seqlen_k: Int32,
        apply_semantic_window: cutlass.Constexpr[bool] = True,
        is_causal: cutlass.Constexpr[bool] = False,
        is_local: cutlass.Constexpr[bool] = False,
        window_size_left: Optional[int] = None,
        window_size_right: Optional[int] = None,
        index_transform: cutlass.Constexpr = lambda index_q, index_k: (
            index_q,
            index_k,
        ),
    ):
        """在不用 mask_type 的情况下应用前向掩码。

        - 若 apply_semantic_window=True，应用 causal/local 窗口约束。
        - 始终应用残差 OOB 掩码（index_k>=seqlen_k 或 index_q>=seqlen_q）。
        """
        offset = 0
        if cutlass.const_expr(apply_semantic_window):
            # 匹配 WINDOW_MASK_INFERENCE 语义：长度不同时把 Q/K 末尾对齐。
            offset = seqlen_k - seqlen_q
        for i in cutlass.range_constexpr(cute.size(acc_qk), unroll_full=True):
            index_q, index_k = index_transform(*index_qk[i])
            if cutlass.const_expr(apply_semantic_window):
                if cutlass.const_expr(is_causal and not is_local):
                    # 纯 causal；兼容两种外部形式：
                    # - (None, None) 来自 interface
                    # - (None, 0) 来自融合掩码风格的调用方
                    right = 0 if const_expr(window_size_right is None) else window_size_right
                    if index_q + offset + right < index_k:
                        acc_qk[i] = -Float32.inf
                elif cutlass.const_expr(
                    is_local or window_size_left is not None or window_size_right is not None
                ):
                    if cutlass.const_expr(window_size_left is None):
                        if index_q + offset + window_size_right < index_k:
                            acc_qk[i] = -Float32.inf
                    elif cutlass.const_expr(window_size_right is None):
                        if index_q + offset - window_size_left > index_k:
                            acc_qk[i] = -Float32.inf
                    else:
                        max_K_index = dsl_min(index_q + offset + window_size_right, seqlen_k)
                        min_K_index = max(0, index_q + offset - window_size_left)
                        if index_k > max_K_index or index_k < min_K_index:
                            acc_qk[i] = -Float32.inf
            # 边界保护始终需要残差掩码。
            if index_k >= seqlen_k or index_q >= seqlen_q:
                acc_qk[i] = -Float32.inf
