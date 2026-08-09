# Copyright (c) 2025, Tri Dao, Siyu Wang, Shengbin Di, Yuxi Chi, Johnsonms, Linfeng Zheng, Haoyan Huang, Lanbo Li, Yun Zhong, Man Yuan, Minmin Sun, Yong Li, Wei Lin.

from enum import IntEnum, auto
from typing import Optional, Tuple, Protocol, runtime_checkable
from dataclasses import dataclass

try:
    from typing import override
except ImportError:  # Python < 3.12
    from typing_extensions import override

import cutlass
from cutlass.pipeline import PipelineClcFetchAsync, PipelineState
from cutlass._mlir import ir
import cutlass.cute as cute
from cutlass import Int32, const_expr
from cutlass.cute import FastDivmodDivisor
from cutlass.utils import ClcDynamicPersistentTileScheduler, ClcDynamicPersistentTileSchedulerParams
from cutlass.cute.typing import Boolean
from cutlass.cutlass_dsl import (
    min as dsl_min,
    extract_mlir_values,
    new_from_mlir_values,
)
from cutlass.utils.hardware_info import HardwareInfo

from quack.cute_dsl_utils import ParamsBase

import flash_attn.cute.utils as utils
from flash_attn.cute.fast_math import clz


class SchedulingMode(IntEnum):
    NONE = auto()
    STATIC = auto()
    DYNAMIC = auto()
    CLC = auto()


@dataclass
class SchedulerState(ParamsBase):
    """CLC 与动态持久化 tile 调度器共享的运行时状态：
    异步流水线及其生产者/消费者状态。

    主 kernel 通过 `create_clc` / `create_dynamic_persistent` 构造本对象，
    它们返回相应的具体状态（`ClcSchedulerState` 或
    `DynamicPersistentSchedulerState`）。调度器通过其 `__init__(...)` 的
    `ctx: SchedulerState | None` 参数消费它。
    """

    _pipeline: cutlass.pipeline.PipelineAsync
    _consumer_state: PipelineState
    _producer_state: PipelineState

    @staticmethod
    def create_clc(
        *,
        hw_scheduler: ClcDynamicPersistentTileScheduler,
        pipeline: PipelineClcFetchAsync,
        consumer_state: PipelineState,
        producer_state: PipelineState,
    ) -> "ClcSchedulerState":
        return ClcSchedulerState(pipeline, consumer_state, producer_state, hw_scheduler)

    @staticmethod
    def create_dynamic_persistent(
        *,
        work_info: cute.Tensor,
        pipeline: cutlass.pipeline.PipelineAsync,
        consumer_state: PipelineState,
        producer_state: PipelineState,
    ) -> "DynamicPersistentSchedulerState":
        return DynamicPersistentSchedulerState(pipeline, consumer_state, producer_state, work_info)

    def consumer_wait(self, *, loc=None, ip=None):
        self._pipeline.consumer_wait(self._consumer_state, loc=loc, ip=ip)

    def consumer_release(self, *, loc=None, ip=None):
        self._pipeline.consumer_release(self._consumer_state, loc=loc, ip=ip)
        self._consumer_state.advance(loc=loc, ip=ip)

    def advance_consumer_state(self, *, loc=None, ip=None):
        self._consumer_state.advance(loc=loc, ip=ip)

    def producer_tail(self, *, loc=None, ip=None):
        self._pipeline.producer_tail(self._producer_state, loc=loc, ip=ip)


@dataclass
class ClcSchedulerState(SchedulerState):
    """持有支持 CLC 的 tile 调度器共享的运行时状态。

    `FlashAttentionForwardSm100` 构造该状态，因为它拥有初始化硬件调度器和
    异步流水线所需的 CLC 响应缓冲区、mbarrier 存储和启动几何信息。各个 tile
    调度器随后消费该状态，并把硬件返回的工作 tile 映射到它们自己的逻辑
    `WorkTileInfo` 坐标。

    要给调度器添加 CLC 支持：
    - 实现 `clc_problem_shape(params)`，让 kernel 能创建硬件调度器
    - 把 `ctx.initial_work_tile_info()` 和 `ctx.get_current_work()` 映射到调度器坐标
    """

    _hw_scheduler: ClcDynamicPersistentTileScheduler

    def initial_work_tile_info(self):
        return self._hw_scheduler.initial_work_tile_info()

    def get_current_work(self):
        return self._hw_scheduler.get_current_work()

    def prefetch_next_work(self, *, loc=None, ip=None):
        self._pipeline.producer_acquire(self._producer_state, loc=loc, ip=ip)
        mbarrier_addr = self._pipeline.producer_get_barrier(self._producer_state, loc=loc, ip=ip)
        self._hw_scheduler.advance_to_next_work(mbarrier_addr, loc=loc, ip=ip)
        self._producer_state.advance(loc=loc, ip=ip)


@dataclass
class DynamicPersistentSchedulerState(SchedulerState):
    """基于信号量：调度器类驱动 atomicAdd + warp 前缀和（warp-prefix-sum），
    并通过 `write_work_info` 写出解析后的工作 tile。"""

    _work_info: cute.Tensor

    def producer_acquire(self, *, loc=None, ip=None):
        self._pipeline.producer_acquire(self._producer_state, loc=loc, ip=ip)

    def producer_commit(self, *, loc=None, ip=None):
        self._pipeline.producer_commit(self._producer_state, loc=loc, ip=ip)

    def advance_producer_state(self, *, loc=None, ip=None):
        self._producer_state.advance(loc=loc, ip=ip)

    def write_work_info(self, block: Int32, head: Int32, batch: Int32, split: Int32):
        self._work_info[0] = block
        self._work_info[1] = head
        self._work_info[2] = batch
        self._work_info[3] = split


class WorkTileInfo(cutlass.utils.WorkTileInfo):
    """包含四个轴的扩展版 WorkTileInfo：(block, head, batch, split)"""

    @override
    def __new_from_mlir_values__(self, values: list[ir.Value]) -> "WorkTileInfo":
        assert len(values) == 5
        new_tile_idx = cutlass.new_from_mlir_values(self._tile_idx, values[:-1])
        new_is_valid_tile = cutlass.new_from_mlir_values(self._is_valid_tile, [values[-1]])
        return WorkTileInfo(new_tile_idx, new_is_valid_tile)


@runtime_checkable
class TileSchedulerProtocol(Protocol):
    """定义所有 tile 调度器必须实现的接口协议。

    调度器负责：
    1. 坐标映射：线性 tile 索引 -> (m_block, head, batch, split)
    2. 工作分发：如何获取下一个 tile（静态 grid-stride 还是动态）
    """

    def initial_work_tile_info(self) -> WorkTileInfo:
        """获取该 CTA 的初始工作 tile。"""
        ...

    def advance_to_next_work(self, *, loc=None, ip=None):
        """消费者侧推进：移动到下一个 tile 并返回它。

        静态调度器：grid-stride 递增 + get_current_work。
        动态调度器：consumer wait + get_current_work + consumer release + 状态推进。
        """
        ...

    def prefetch_next_work(self, *, loc=None, ip=None) -> None:
        """生产者侧预取下一个工作 tile（静态调度器为 no-op）。

        动态调度器：producer acquire（+ 发起 CLC 查询）+ 生产者状态推进。
        仅由调度 warp 调用。
        """
        ...

    def producer_tail(self, *, loc=None, ip=None) -> None:
        """最后一个 tile 之后的生产者侧清理。

        静态调度器为 no-op。动态调度器：流水线 producer_tail。
        """
        ...


@dataclass
class TileSchedulerArguments(ParamsBase):
    num_block: Int32
    num_head: Int32
    num_batch: Int32
    num_splits: Int32
    seqlen_k: Int32
    headdim: Int32
    headdim_v: Int32
    total_q: Int32
    tile_shape_mn: cutlass.Constexpr[Tuple[int, int]]
    cluster_shape_mn: cutlass.Constexpr[Tuple[int, int]] = (1, 1)
    mCuSeqlensQ: Optional[cute.Tensor] = None
    mSeqUsedQ: Optional[cute.Tensor] = None
    qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1
    element_size: cutlass.Constexpr[int] = 2
    is_persistent: cutlass.Constexpr[bool] = False
    lpt: cutlass.Constexpr[bool] = False
    is_split_kv: cutlass.Constexpr[bool] = False
    head_swizzle: cutlass.Constexpr[bool] = False
    use_cluster_idx: cutlass.Constexpr[bool] = False
    num_splits_dynamic_ptr: Optional[cute.Tensor] = None
    num_m_blocks_ptr: Optional[cute.Tensor] = None
    virtual_batch_idx_ptr: Optional[cute.Tensor] = None
    num_nheads_in_l2_ptr: Optional[cute.Tensor] = None
    cu_total_m_blocks_ptr: Optional[cute.Tensor] = None
    cu_total_splits_m_blocks_ptr: Optional[cute.Tensor] = None
    blocks_to_batch_idx_ptr: Optional[cute.Tensor] = None
    tile_count_semaphore: Optional[cute.Pointer] = None
    persistent_cta_multiplier: cutlass.Constexpr[int] = 1


class SingleTileScheduler:
    @dataclass
    class Params(ParamsBase):
        num_block: Int32
        num_head: Int32
        num_batch: Int32
        num_splits: Int32
        num_splits_divmod: FastDivmodDivisor
        is_split_kv: cutlass.Constexpr[bool] = False
        cluster_shape_mn: cutlass.Constexpr[Tuple[int, int]] = (1, 1)
        use_cluster_idx: cutlass.Constexpr[bool] = False
        num_splits_dynamic_ptr: Optional[cute.Tensor] = None

        @staticmethod
        def create(
            args: TileSchedulerArguments, *, loc=None, ip=None
        ) -> "SingleTileScheduler.Params":
            return SingleTileScheduler.Params(
                args.num_block,
                args.num_head,
                args.num_batch,
                args.num_splits,
                FastDivmodDivisor(args.num_splits),
                args.is_split_kv,
                args.cluster_shape_mn,
                args.use_cluster_idx,
                args.num_splits_dynamic_ptr,
            )

    def __init__(self, params: Params, blk_coord: cute.Coord, *, loc=None, ip=None):
        self.params = params
        self._blk_coord = blk_coord
        self._is_first_block = True
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(
        args: TileSchedulerArguments,
        *,
        scheduling_mode: SchedulingMode = SchedulingMode.STATIC,
        loc=None,
        ip=None,
    ) -> Params:
        assert scheduling_mode == SchedulingMode.STATIC, (
            f"SingleTileScheduler only supports STATIC, got {scheduling_mode!r}"
        )
        return SingleTileScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    def create(
        params: Params, ctx: SchedulerState | None = None, *, loc=None, ip=None
    ) -> "SingleTileScheduler":
        if const_expr(cute.size(params.cluster_shape_mn) == 1 or not params.use_cluster_idx):
            blk_coord = cute.arch.block_idx()
        else:
            blk_coord = cute.arch.cluster_idx()
        return SingleTileScheduler(params, blk_coord, loc=loc, ip=ip)

    # 由主机（host）调用
    @staticmethod
    def get_grid_shape(
        params: Params,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        # TODO: 这里硬编码了只使用 cluster = (1, 1) 或 (2, 1) 的事实
        assert params.cluster_shape_mn[1] == 1, "Only cluster_shape_mn[1] == 1 is supported"
        if const_expr(params.use_cluster_idx):
            # 网格必须有 num_block * cluster_m 个物理块，才能形成 num_block 个簇
            grid_x = params.num_block * params.cluster_shape_mn[0]
        else:
            grid_x = cute.round_up(params.num_block, params.cluster_shape_mn[0])
        return (
            grid_x,
            params.num_head * params.num_splits,
            params.num_batch,
        )

    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        block_idx, head_idx, batch_idx = self._blk_coord
        is_valid = self._is_first_block
        if const_expr(self.params.is_split_kv):
            head_idx, split_idx = divmod(head_idx, self.params.num_splits_divmod)
        else:
            split_idx = Int32(0)
        # 把动态的逐 batch num_splits 打包进 split_idx 的高 16 位
        if const_expr(self.params.is_split_kv and self.params.num_splits_dynamic_ptr is not None):
            if is_valid:
                num_splits = Int32(self.params.num_splits_dynamic_ptr[batch_idx])
                split_idx = split_idx | (num_splits << 16)
        return WorkTileInfo(
            (block_idx, head_idx, batch_idx, split_idx),
            is_valid,
        )

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    def prefetch_next_work(self, *, loc=None, ip=None):
        pass

    def advance_to_next_work(self, *, loc=None, ip=None):
        self._is_first_block = False
        return self.get_current_work()

    def producer_tail(self, *, loc=None, ip=None):
        pass

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.params, self._blk_coord]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip([self.params, self._blk_coord], self._values_pos):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        scheduler = SingleTileScheduler(*(tuple(obj_list)), loc=self._loc)
        # 注意：_is_first_block 是仅存在于 Python 侧的属性，不包含在 MLIR 值中，
        # 因此重建后必须显式恢复它。
        scheduler._is_first_block = self._is_first_block
        return scheduler


class StaticPersistentTileScheduler:
    @dataclass
    class Params(ParamsBase):
        num_block_cluster_divmod: FastDivmodDivisor
        num_head_divmod: FastDivmodDivisor
        total_blocks_cluster: Int32
        cluster_shape_m: cutlass.Constexpr[int] = 1

        @staticmethod
        def create(
            args: TileSchedulerArguments, *, loc=None, ip=None
        ) -> "StaticPersistentTileScheduler.Params":
            num_block_cluster = cute.ceil_div(args.num_block, cute.size(args.cluster_shape_mn))
            total_blocks_cluster = num_block_cluster * args.num_head * args.num_batch
            return StaticPersistentTileScheduler.Params(
                FastDivmodDivisor(num_block_cluster),
                FastDivmodDivisor(args.num_head),
                total_blocks_cluster,
                cluster_shape_m=args.cluster_shape_mn[0],
            )

    def __init__(self, params: Params, tile_idx: Int32, *, loc=None, ip=None):
        self.params = params
        self._tile_idx = tile_idx
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(
        args: TileSchedulerArguments,
        *,
        scheduling_mode: SchedulingMode = SchedulingMode.STATIC,
        loc=None,
        ip=None,
    ) -> Params:
        assert scheduling_mode == SchedulingMode.STATIC, (
            f"StaticPersistentTileScheduler only supports STATIC, got {scheduling_mode!r}"
        )
        return StaticPersistentTileScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    def create(
        params: Params, ctx: SchedulerState | None = None, *, loc=None, ip=None
    ) -> "StaticPersistentTileScheduler":
        if const_expr(cute.size(params.cluster_shape_m) == 1):
            tile_idx = cute.arch.block_idx()[0]
        else:
            tile_idx = cute.arch.cluster_idx()[0]
        return StaticPersistentTileScheduler(params, tile_idx, loc=loc, ip=ip)

    @staticmethod
    def get_grid_shape(
        params: Params,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        hardware_info = cutlass.utils.HardwareInfo()
        sm_count = hardware_info.get_device_multiprocessor_count()
        max_ctas = (sm_count // params.cluster_shape_m) * params.cluster_shape_m
        grid_x = cutlass.min(max_ctas, params.total_blocks_cluster * params.cluster_shape_m)
        return (grid_x, Int32(1), Int32(1))

    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        hn_idx, block_idx = divmod(self._tile_idx, self.params.num_block_cluster_divmod)
        batch_idx, head_idx = divmod(hn_idx, self.params.num_head_divmod)
        is_valid = self._tile_idx < self.params.total_blocks_cluster
        return WorkTileInfo(
            (Int32(block_idx), Int32(head_idx), Int32(batch_idx), Int32(0)), is_valid
        )

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    def prefetch_next_work(self, *, loc=None, ip=None):
        pass

    def advance_to_next_work(self, *, loc=None, ip=None):
        if const_expr(self.params.cluster_shape_m == 1):
            self._tile_idx += cute.arch.grid_dim()[0]
        else:
            self._tile_idx += cute.arch.cluster_dim()[0]
        return self.get_current_work()

    def producer_tail(self, *, loc=None, ip=None):
        pass

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.params, self._tile_idx]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip(
            [self.params, self._tile_idx],
            self._values_pos,
        ):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return StaticPersistentTileScheduler(*(tuple(obj_list)), loc=self._loc)


class SingleTileLPTScheduler:
    @dataclass
    class Params(ParamsBase):
        total_blocks: Int32
        num_splits: Int32
        num_block: Int32
        num_head: Int32
        num_batch: Int32
        l2_minor: Int32
        num_head_divmod: FastDivmodDivisor
        l2_minor_divmod: FastDivmodDivisor
        l2_major_divmod: FastDivmodDivisor
        l2_minor_residual_divmod: FastDivmodDivisor
        num_hb_quotient: Int32
        num_splits_divmod: FastDivmodDivisor
        is_split_kv: cutlass.Constexpr[bool] = False
        cluster_shape_m: cutlass.Constexpr[int] = 1
        scheduling_mode: cutlass.Constexpr[SchedulingMode] = SchedulingMode.STATIC
        lpt: cutlass.Constexpr[bool] = True
        use_cluster_idx: cutlass.Constexpr[bool] = True
        num_splits_dynamic_ptr: Optional[cute.Tensor] = None

        @staticmethod
        @cute.jit
        def create(
            args: TileSchedulerArguments,
            *,
            scheduling_mode: SchedulingMode = SchedulingMode.STATIC,
            loc=None,
            ip=None,
        ) -> "SingleTileLPTScheduler.Params":
            assert scheduling_mode in (SchedulingMode.STATIC, SchedulingMode.CLC), (
                f"Only STATIC and CLC are supported, got {scheduling_mode!r}"
            )
            # int64：一旦 seqlen_k * (headdim + headdim_v) * element_size > 2**31
            #（hdim-128 bf16 时 seqlen_k > ~4M），该乘积会溢出 int32。
            size_one_kv_head = (
                cutlass.Int64(args.seqlen_k) * (args.headdim + args.headdim_v) * args.element_size
            )
            size_one_head = size_one_kv_head
            size_l2 = 50 * 1024 * 1024  # 40 MB 用于 K 和 V
            # swizzle 是每个"分区"（section）的大小。把 swizzle 圆整为 2 的幂
            # 需要注意只有一个 head 能放下的情况
            # swizzle 是 L2 中能放下的 head 数量
            # swizzle 是 2 的幂时似乎更快
            log2_floor = lambda n: 31 - clz(n)
            swizzle = (
                1 if size_l2 < size_one_head else (1 << log2_floor(Int32(size_l2 // size_one_head)))
            )
            # 若处于最后一个分区（称为残差分区），不应除以 swizzle，
            # 而应除以余数（remainder）。
            num_hb_quotient = (args.num_head * args.num_batch) // swizzle
            num_hb_remainder = (args.num_head * args.num_batch) % swizzle
            return SingleTileLPTScheduler.Params(
                total_blocks=args.num_block * args.num_head * args.num_batch,
                num_block=args.num_block,
                num_head=args.num_head,
                num_batch=args.num_batch,
                l2_minor=Int32(swizzle),
                num_head_divmod=FastDivmodDivisor(args.num_head),
                l2_minor_divmod=FastDivmodDivisor(swizzle),
                l2_major_divmod=FastDivmodDivisor(swizzle * args.num_block),
                l2_minor_residual_divmod=FastDivmodDivisor(max(num_hb_remainder, 1)),
                num_hb_quotient=Int32(num_hb_quotient),
                num_splits=args.num_splits,
                num_splits_divmod=FastDivmodDivisor(args.num_splits),
                is_split_kv=args.is_split_kv,
                cluster_shape_m=args.cluster_shape_mn[0],
                scheduling_mode=scheduling_mode,
                lpt=args.lpt,
                use_cluster_idx=args.use_cluster_idx,
                num_splits_dynamic_ptr=args.num_splits_dynamic_ptr,
            )

    def __init__(
        self,
        params: Params,
        tile_idx: Int32,
        split_idx: Int32,
        ctx: SchedulerState | None = None,
        *,
        loc=None,
        ip=None,
    ):
        self.params = params
        self._tile_idx = tile_idx
        self._split_idx = split_idx
        self._ctx = ctx
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(
        args: TileSchedulerArguments,
        *,
        scheduling_mode: SchedulingMode = SchedulingMode.STATIC,
        loc=None,
        ip=None,
    ) -> Params:
        return SingleTileLPTScheduler.Params.create(
            args, scheduling_mode=scheduling_mode, loc=loc, ip=ip
        )

    @staticmethod
    def _clc_grid_shape(params: Params):
        num_batch_splits = (
            params.num_batch * params.num_splits
            if const_expr(params.is_split_kv)
            else params.num_batch
        )
        if const_expr(params.use_cluster_idx):
            # 网格必须有 num_block * cluster_m 个物理块，才能形成 num_block 个簇
            grid_x = params.num_block * params.cluster_shape_m
        else:
            grid_x = cute.round_up(params.num_block, params.cluster_shape_m)
        return (
            grid_x,
            params.num_head,
            num_batch_splits,
        )

    @staticmethod
    @cute.jit
    def clc_problem_shape(params: Params):
        return ClcDynamicPersistentTileSchedulerParams(
            problem_shape_ntile_mnl=SingleTileLPTScheduler._clc_grid_shape(params),
            cluster_shape_mnk=(params.cluster_shape_m, 1, 1),
        )

    @staticmethod
    @cute.jit
    def create(
        params: Params, ctx: SchedulerState | None = None, *, loc=None, ip=None
    ) -> "SingleTileLPTScheduler":
        if const_expr(params.scheduling_mode == SchedulingMode.CLC):
            return SingleTileLPTScheduler(
                params, cute.arch.block_idx()[0], Int32(0), ctx, loc=loc, ip=ip
            )
        tile_idx, split_idx, _ = cute.arch.block_idx()
        return SingleTileLPTScheduler(params, tile_idx, split_idx, loc=loc, ip=ip)

    @staticmethod
    def get_grid_shape(
        params: Params,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        if const_expr(params.scheduling_mode == SchedulingMode.CLC):
            return SingleTileLPTScheduler._clc_grid_shape(params)
        return (params.total_blocks, params.num_splits, Int32(1))

    @cute.jit
    def clc_work_to_coords(self, work) -> WorkTileInfo:
        """把 CLC 响应 (block, head, batch_split) 转换为 WorkTileInfo。

        CLC 返回原始网格坐标 —— 无 L2 swizzle（顺序由硬件决定）。
        我们只应用簇（cluster）划分、可选的 LPT 块反转和 split_kv 解包。
        """
        block_idx = work.tile_idx[0]
        if const_expr(self.params.cluster_shape_m > 1):
            block_idx = block_idx // self.params.cluster_shape_m
        if const_expr(self.params.lpt):
            # 最长处理时间优先（LPT）：反转块顺序
            if const_expr(self.params.cluster_shape_m > 1 and not self.params.use_cluster_idx):
                num_block = self.params.num_block // self.params.cluster_shape_m
            else:
                num_block = self.params.num_block
            block_idx = num_block - 1 - block_idx
        split_idx = Int32(0)
        if const_expr(self.params.is_split_kv):
            batch_idx, split_idx = divmod(work.tile_idx[2], self.params.num_splits_divmod)
        else:
            batch_idx = work.tile_idx[2]
        if const_expr(self.params.cluster_shape_m > 1 and not self.params.use_cluster_idx):
            bidx_in_cluster = cute.arch.block_in_cluster_idx()
            block_idx = block_idx * self.params.cluster_shape_m + bidx_in_cluster[0]
        # 把动态的逐 batch num_splits 打包进 split_idx 的高 16 位
        if const_expr(self.params.is_split_kv and self.params.num_splits_dynamic_ptr is not None):
            if work.is_valid_tile:
                num_splits = Int32(self.params.num_splits_dynamic_ptr[batch_idx])
                split_idx = split_idx | (num_splits << 16)
        return WorkTileInfo(
            (Int32(block_idx), Int32(work.tile_idx[1]), Int32(batch_idx), Int32(split_idx)),
            work.is_valid_tile,
        )

    @cute.jit
    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            work = self._ctx.get_current_work()
            self._tile_idx = work.tile_idx[0]
            return self.clc_work_to_coords(work)
        # 静态路径：L2 swizzle 坐标映射
        params = self.params
        # 实现 LPT 调度坐标计算
        bidhb, l2_mod = divmod(self._tile_idx, params.l2_major_divmod)
        # 若处于最后一个分区（称为残差分区），不应除以 swizzle，
        # 而应除以余数（remainder）。
        block, bidhb_residual = 0, 0
        if bidhb < params.num_hb_quotient:
            block, bidhb_residual = divmod(l2_mod, params.l2_minor_divmod)
        else:
            block, bidhb_residual = divmod(l2_mod, params.l2_minor_residual_divmod)
        bidhb_actual = bidhb * params.l2_minor + bidhb_residual
        batch_idx, head_idx = divmod(bidhb_actual, params.num_head_divmod)
        # 最长处理时间优先（LPT）
        if const_expr(params.lpt):
            block = params.num_block - 1 - block
        is_valid = self._tile_idx < params.total_blocks
        split_idx = self._split_idx
        # 把动态的逐 batch num_splits 打包进 split_idx 的高 16 位
        if const_expr(params.is_split_kv and params.num_splits_dynamic_ptr is not None):
            if is_valid:
                num_splits = Int32(params.num_splits_dynamic_ptr[batch_idx])
                split_idx = split_idx | (num_splits << 16)
        return WorkTileInfo(
            (Int32(block), Int32(head_idx), Int32(batch_idx), Int32(split_idx)), is_valid
        )

    @cute.jit
    def initial_work_tile_info(self, *, loc=None, ip=None):
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            work = self._ctx.initial_work_tile_info()
            self._tile_idx = work.tile_idx[0]
            return self.clc_work_to_coords(work)
        return self.get_current_work(loc=loc, ip=ip)

    def prefetch_next_work(self, *, loc=None, ip=None):
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            self._ctx.prefetch_next_work(loc=loc, ip=ip)

    def advance_to_next_work(self, *, loc=None, ip=None):
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            self._ctx.consumer_wait(loc=loc, ip=ip)
            work = self.get_current_work()
            self._ctx.consumer_release(loc=loc, ip=ip)
            return work
        # 单 tile 调度器 —— 设为无效 tile_idx 以表示没有更多工作
        self._tile_idx = self.params.total_blocks
        return self.get_current_work()

    def producer_tail(self, *, loc=None, ip=None):
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            self._ctx.producer_tail(loc=loc, ip=ip)

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        objs = [self.params, self._tile_idx, self._split_idx]
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            objs += [self._ctx]
        for obj in objs:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        objs = [self.params, self._tile_idx, self._split_idx]
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            objs += [self._ctx]
        for obj, n_items in zip(objs, self._values_pos):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return self.__class__(*obj_list, loc=self._loc)


class SingleTileLPTBwdScheduler:
    @dataclass
    class Params(ParamsBase):
        total_blocks: Int32
        num_block: Int32
        l2_minor: Int32
        num_head_divmod: FastDivmodDivisor
        l2_minor_divmod: FastDivmodDivisor
        l2_major_divmod: FastDivmodDivisor
        l2_minor_residual_divmod: FastDivmodDivisor
        num_hb_quotient: Int32
        cluster_shape_mn: cutlass.Constexpr[Tuple[int, int]] = (1, 1)
        spt: cutlass.Constexpr[bool] = True

        @staticmethod
        @cute.jit
        def create(
            args: TileSchedulerArguments, *, loc=None, ip=None
        ) -> "SingleTileLPTBwdScheduler.Params":
            size_l2 = 50 * 1024 * 1024
            # int64：这些乘积在大 seqlen_k 时会溢出 int32（hdim-128 bf16 时
            # > ~4M；dqaccum *4 项甚至更早发生回绕）。
            size_one_qdo_head = (
                cutlass.Int64(args.seqlen_k) * (args.headdim + args.headdim_v) * args.element_size
            )
            size_one_dqaccum_head = cutlass.Int64(args.seqlen_k) * (args.headdim) * 4
            # size_one_dqaccum_head = 0
            size_one_head = size_one_qdo_head + size_one_dqaccum_head
            log2_floor = lambda n: 31 - clz(n)
            swizzle = (
                1 if size_l2 < size_one_head else (1 << log2_floor(Int32(size_l2 // size_one_head)))
            )
            # swizzle = 8
            # 若处于最后一个分区（称为残差分区），不应除以 swizzle，
            # 而应除以余数（remainder）。
            num_hb_quotient = (args.num_head * args.num_batch) // swizzle
            num_hb_remainder = (args.num_head * args.num_batch) % swizzle
            num_block = cute.ceil_div(args.num_block, args.cluster_shape_mn[0])
            return SingleTileLPTBwdScheduler.Params(
                total_blocks=(num_block * args.cluster_shape_mn[0])
                * args.num_head
                * args.num_batch,
                num_block=num_block,
                l2_minor=Int32(swizzle),
                num_head_divmod=FastDivmodDivisor(args.num_head),
                l2_minor_divmod=FastDivmodDivisor(swizzle),
                l2_major_divmod=FastDivmodDivisor(swizzle * num_block),
                l2_minor_residual_divmod=FastDivmodDivisor(
                    max(num_hb_remainder, 1)
                ),  # 避免除以 0
                num_hb_quotient=Int32(num_hb_quotient),
                cluster_shape_mn=args.cluster_shape_mn,
                spt=args.lpt,
            )

    def __init__(self, params: Params, tile_idx: Int32, *, loc=None, ip=None):
        self.params = params
        self._tile_idx = tile_idx
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(
        args: TileSchedulerArguments,
        *,
        scheduling_mode: SchedulingMode = SchedulingMode.STATIC,
        loc=None,
        ip=None,
    ) -> Params:
        assert scheduling_mode == SchedulingMode.STATIC, (
            f"SingleTileLPTBwdScheduler only supports STATIC, got {scheduling_mode!r}"
        )
        return SingleTileLPTBwdScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    @cute.jit
    def create(params: Params, *, loc=None, ip=None) -> "SingleTileLPTBwdScheduler":
        tile_idx = cute.arch.block_idx()[0]
        return SingleTileLPTBwdScheduler(params, tile_idx, loc=loc, ip=ip)

    # 由主机（host）调用
    @staticmethod
    def get_grid_shape(
        params: Params,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        return (params.total_blocks, Int32(1), Int32(1))

    @cute.jit
    def get_current_work(self, *, loc=None, ip=None) -> cutlass.utils.WorkTileInfo:
        cluster_idx = self._tile_idx // self.params.cluster_shape_mn[0]
        params = self.params
        # 实现 LPT 调度坐标计算
        bidhb, l2_mod = divmod(cluster_idx, params.l2_major_divmod)
        # 若处于最后一个分区（称为残差分区），不应除以 swizzle，
        # 而应除以余数（remainder）。
        block, bidhb_residual = 0, 0
        if bidhb < params.num_hb_quotient:
            block, bidhb_residual = divmod(l2_mod, params.l2_minor_divmod)
        else:
            block, bidhb_residual = divmod(l2_mod, params.l2_minor_residual_divmod)
        bidhb_actual = bidhb * params.l2_minor + bidhb_residual
        batch_idx, head_idx = divmod(bidhb_actual, params.num_head_divmod)
        if cutlass.const_expr(params.spt):
            block = params.num_block - 1 - block
        if cutlass.const_expr(params.cluster_shape_mn[0] > 1):
            bidx_in_cluster = cute.arch.block_in_cluster_idx()
            block = block * params.cluster_shape_mn[0] + bidx_in_cluster[0]
        is_valid = self._tile_idx < params.total_blocks
        return WorkTileInfo((Int32(block), Int32(head_idx), Int32(batch_idx), Int32(0)), is_valid)

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    def prefetch_next_work(self, *, loc=None, ip=None):
        pass

    def advance_to_next_work(self, *, loc=None, ip=None):
        # 单 tile 调度器 —— 设为无效 tile_idx 以表示没有更多工作
        self._tile_idx = self.params.total_blocks
        return self.get_current_work()

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.params, self._tile_idx]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip([self.params, self._tile_idx], self._values_pos):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return self.__class__(*(tuple(obj_list)), loc=self._loc)


@dataclass
class VarlenDecoder(ParamsBase):
    """逐 batch 的 m-block 查找 + 用 warp 前缀和搜索并解码 varlen 工作 tile。
    组合进 `SingleTileVarlenScheduler.Params` 和
    `DynamicPersistentVarlenScheduler.Params` 两者。

    `fold_splits_into_scan` 控制前缀和扫描是把逐 batch 的 `num_splits`
    折叠进逐 batch 的 tile 计数（DynamicPersistent），还是始终只统计
    m_blocks（SingleTileVarlen，其 splits 在网格层分发、扫描后解析）。
    """

    num_head: Int32
    num_batch: Int32
    num_splits: Int32
    max_kvblock_in_l2: Int32
    tile_shape_mn: cutlass.Constexpr[Tuple[int, int]]
    qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1
    is_split_kv: cutlass.Constexpr[bool] = False
    lpt: cutlass.Constexpr[bool] = False
    head_swizzle: cutlass.Constexpr[bool] = False
    cluster_shape_m: cutlass.Constexpr[int] = 1
    use_cluster_idx: cutlass.Constexpr[bool] = False
    fold_splits_into_scan: cutlass.Constexpr[bool] = False
    scheduling_mode: cutlass.Constexpr[SchedulingMode] = SchedulingMode.STATIC
    mCuSeqlensQ: Optional[cute.Tensor] = None
    mSeqUsedQ: Optional[cute.Tensor] = None
    num_m_blocks_ptr: Optional[cute.Tensor] = None
    num_splits_dynamic_ptr: Optional[cute.Tensor] = None
    virtual_batch_idx_ptr: Optional[cute.Tensor] = None
    num_nheads_in_l2_ptr: Optional[cute.Tensor] = None
    cu_total_m_blocks_ptr: Optional[cute.Tensor] = None
    cu_total_splits_m_blocks_ptr: Optional[cute.Tensor] = None
    blocks_to_batch_idx_ptr: Optional[cute.Tensor] = None

    @staticmethod
    @cute.jit
    def create(
        args: TileSchedulerArguments,
        *,
        fold_splits_into_scan: bool,
        head_swizzle: bool = False,
        cluster_shape_m: int = 1,
        scheduling_mode: SchedulingMode = SchedulingMode.STATIC,
        loc=None,
        ip=None,
    ) -> "VarlenDecoder":
        size_l2 = 50 * 1024 * 1024  # 50 MB 用于 K 和 V
        # 若是反向传播，这是 qdo 块大小
        kv_block_size = (args.headdim + args.headdim_v) * args.element_size * args.tile_shape_mn[1]
        # 若是反向传播，加上 dqaccum 块大小来计算 swizzle
        if head_swizzle:
            kv_block_size += args.headdim * 4 * args.tile_shape_mn[1]
        max_kvblock_in_l2 = size_l2 // kv_block_size
        return VarlenDecoder(
            num_head=args.num_head,
            num_batch=args.num_batch,
            num_splits=args.num_splits,
            max_kvblock_in_l2=max_kvblock_in_l2,
            tile_shape_mn=args.tile_shape_mn,
            qhead_per_kvhead_packgqa=args.qhead_per_kvhead_packgqa,
            is_split_kv=args.is_split_kv,
            lpt=args.lpt,
            head_swizzle=head_swizzle,
            cluster_shape_m=cluster_shape_m,
            use_cluster_idx=args.use_cluster_idx,
            fold_splits_into_scan=fold_splits_into_scan,
            scheduling_mode=scheduling_mode,
            mCuSeqlensQ=args.mCuSeqlensQ,
            mSeqUsedQ=args.mSeqUsedQ,
            num_m_blocks_ptr=args.num_m_blocks_ptr,
            num_splits_dynamic_ptr=args.num_splits_dynamic_ptr,
            virtual_batch_idx_ptr=args.virtual_batch_idx_ptr,
            num_nheads_in_l2_ptr=args.num_nheads_in_l2_ptr,
            cu_total_m_blocks_ptr=args.cu_total_m_blocks_ptr,
            cu_total_splits_m_blocks_ptr=args.cu_total_splits_m_blocks_ptr,
            blocks_to_batch_idx_ptr=args.blocks_to_batch_idx_ptr,
        )

    @cute.jit
    def _num_m_blocks(self, lane: Int32, bidb_start: Int32) -> Int32:
        """逐 batch 的 m-block 计数"""
        batch_idx = lane + bidb_start
        is_valid = batch_idx < self.num_batch and lane < cute.arch.WARP_SIZE - 1
        if cutlass.const_expr(self.num_m_blocks_ptr is not None):
            num_m_blocks_raw = Int32(0)
            if is_valid:
                if cutlass.const_expr(self.virtual_batch_idx_ptr is not None):
                    real_batch_idx = self.virtual_batch_idx_ptr[batch_idx]
                else:
                    real_batch_idx = batch_idx
                num_m_blocks_raw = Int32(self.num_m_blocks_ptr[real_batch_idx])
            return cute.ceil_div(num_m_blocks_raw, self.cluster_shape_m) if is_valid else Int32(0)
        if cutlass.const_expr(self.virtual_batch_idx_ptr is not None):
            seqlen = Int32(0)
            if is_valid:
                real_batch_idx = self.virtual_batch_idx_ptr[batch_idx]
                if cutlass.const_expr(self.mSeqUsedQ is not None):
                    seqlen = self.mSeqUsedQ[real_batch_idx]
                else:
                    seqlen = self.mCuSeqlensQ[real_batch_idx + 1] - self.mCuSeqlensQ[real_batch_idx]
            if cutlass.const_expr(self.qhead_per_kvhead_packgqa > 1):
                seqlen *= self.qhead_per_kvhead_packgqa
            return (
                cute.ceil_div(cute.ceil_div(seqlen, self.tile_shape_mn[0]), self.cluster_shape_m)
                if is_valid
                else Int32(0)
            )
        if cutlass.const_expr(self.mSeqUsedQ is not None):
            seqlen = Int32(0)
            if batch_idx < self.num_batch:
                seqlen = self.mSeqUsedQ[batch_idx]
        else:
            assert self.mCuSeqlensQ is not None
            cur_cu_seqlen = Int32(0)
            if batch_idx <= self.num_batch:
                cur_cu_seqlen = self.mCuSeqlensQ[batch_idx]
            next_cu_seqlen = cute.arch.shuffle_sync_down(cur_cu_seqlen, offset=1)
            seqlen = next_cu_seqlen - cur_cu_seqlen
        if cutlass.const_expr(self.qhead_per_kvhead_packgqa > 1):
            seqlen *= self.qhead_per_kvhead_packgqa
        return (
            cute.ceil_div(cute.ceil_div(seqlen, self.tile_shape_mn[0]), self.cluster_shape_m)
            if is_valid
            else Int32(0)
        )

    @cute.jit
    def _num_splits(self, lane: Int32, bidb_start: Int32) -> Int32:
        if cutlass.const_expr(not self.fold_splits_into_scan):
            return Int32(1)
        batch_idx = lane + bidb_start
        is_valid = batch_idx < self.num_batch and lane < cute.arch.WARP_SIZE - 1
        if cutlass.const_expr(not self.is_split_kv):
            return Int32(1)
        elif cutlass.const_expr(self.num_splits_dynamic_ptr is not None):
            num_splits = Int32(0)
            if is_valid:
                if cutlass.const_expr(self.virtual_batch_idx_ptr is not None):
                    batch_idx = self.virtual_batch_idx_ptr[batch_idx]
                num_splits = self.num_splits_dynamic_ptr[batch_idx]
            return num_splits
        else:
            return Int32(0) if not is_valid else self.num_splits

    @cute.jit
    def decode(
        self,
        next_tile_idx: Int32,
        bidb_start: Int32,
        group_start_tile: Int32,
    ) -> Tuple[Int32, Int32, Int32, Int32, Int32, Int32, Boolean]:
        """通过 warp 级前缀和搜索 varlen batch，并解码工作 tile。

        Returns
            - block
            - head_idx
            - batch_idx
            - split_idx
            - num_splits
            - group_start_tile
            - is_valid
        """
        # 扫描默认只统计 m_blocks，除非 splits 被折叠进扫描，因此用作提示的 cumsum
        # 必须与之匹配：cu_total_splits_m_blocks 只适用于折叠（持久化）布局，
        # 在那种布局里 SingleTileVarlen 把 splits 放在独立的网格维。
        if const_expr(self.fold_splits_into_scan):
            cu_hint_ptr = self.cu_total_splits_m_blocks_ptr
        else:
            cu_hint_ptr = self.cu_total_m_blocks_ptr
        # 同时适用于 SingleTileVarlen 的 STATIC 和 CLC；不适用于 DynamicPersistent
        #（其 warp-scan 的 _bidb_start 续扫已经分摊了每次调用的开销）。
        hint_mode_ok = const_expr(
            cu_hint_ptr is not None
            and (
                self.scheduling_mode == SchedulingMode.STATIC
                or self.scheduling_mode == SchedulingMode.CLC
            )
        )
        # O(1) 的倒排索引（扁平 block -> batch）直接取代 warp 扫描。它需要
        # 展开布局：该布局中一个 batch 拥有由 cumsum 给定的连续 tile 区间。
        use_blocks_to_batch = const_expr(
            hint_mode_ok
            and self.blocks_to_batch_idx_ptr is not None
            and not self.fold_splits_into_scan
        )
        use_cumsum_hint = const_expr(hint_mode_ok and not use_blocks_to_batch)

        lane_idx = cute.arch.lane_idx()
        if const_expr(use_blocks_to_batch):
            batch_idx = Int32(self.blocks_to_batch_idx_ptr[next_tile_idx // self.num_head])
            num_m_blocks, num_splits = Int32(0), Int32(1)
            is_valid = batch_idx < self.num_batch
            if is_valid:
                cu_lo = cu_hint_ptr[batch_idx]
                num_m_blocks = cu_hint_ptr[batch_idx + 1] - cu_lo
                group_start_tile = cu_lo * self.num_head
            else:
                batch_idx = Int32(self.num_batch)
        else:
            if const_expr(use_cumsum_hint):
                target = next_tile_idx // self.num_head
                lo = utils.get_batch_from_cu_tensor(target, cu_hint_ptr)
                group_size = Int32(cute.arch.WARP_SIZE - 1)
                bidb_start = (lo // group_size) * group_size
                group_start_tile = cu_hint_ptr[bidb_start] * self.num_head

            num_m_blocks = self._num_m_blocks(lane_idx, bidb_start=bidb_start)
            num_splits = self._num_splits(lane_idx, bidb_start=bidb_start)
            per_batch = num_m_blocks * num_splits if const_expr(self.is_split_kv) else num_m_blocks
            cumulative = utils.warp_prefix_sum(per_batch, lane_idx)
            m_blocks_in_group = cute.arch.shuffle_sync(cumulative, cute.arch.WARP_SIZE - 1)
            group_end_tile = m_blocks_in_group * self.num_head + group_start_tile

            batch_idx = bidb_start
            while group_end_tile <= next_tile_idx:
                batch_idx += cute.arch.WARP_SIZE - 1
                if batch_idx >= self.num_batch:
                    batch_idx = Int32(self.num_batch)
                    group_end_tile = next_tile_idx + 1
                else:
                    num_m_blocks = self._num_m_blocks(lane_idx, bidb_start=batch_idx)
                    num_splits = self._num_splits(lane_idx, bidb_start=batch_idx)
                    per_batch = (
                        num_m_blocks * num_splits if const_expr(self.is_split_kv) else num_m_blocks
                    )
                    cumulative = utils.warp_prefix_sum(per_batch, lane_idx)
                    m_blocks_in_group = cute.arch.shuffle_sync(cumulative, cute.arch.WARP_SIZE - 1)
                    group_end_tile += m_blocks_in_group * self.num_head

            is_valid = batch_idx < self.num_batch
            if is_valid:
                group_start_tile = group_end_tile - m_blocks_in_group * self.num_head
                batch_idx_in_group = cute.arch.popc(
                    cute.arch.vote_ballot_sync(
                        group_start_tile + cumulative * self.num_head <= next_tile_idx
                    )
                )
                batch_idx += batch_idx_in_group
                num_m_blocks_prev_lane = (
                    Int32(0)
                    if batch_idx_in_group == 0
                    else cute.arch.shuffle_sync(cumulative, batch_idx_in_group - 1)
                )
                group_start_tile += num_m_blocks_prev_lane * self.num_head
                num_m_blocks = cute.arch.shuffle_sync(num_m_blocks, batch_idx_in_group)
                if const_expr(self.is_split_kv):
                    num_splits = cute.arch.shuffle_sync(num_splits, batch_idx_in_group)

        block, head_idx, split_idx = Int32(0), Int32(0), Int32(0)
        if is_valid:
            mh_block = next_tile_idx - group_start_tile

            if const_expr(self.lpt or self.head_swizzle):
                # 这是 SingleTileLPTScheduler 的一个变体，复杂之处在于每个 batch 的
                # seqlen 可能不同。
                # TODO: 是否存在 num_m_blocks 为 0 的情况？
                if const_expr(not self.is_split_kv) or num_splits == 1:
                    if const_expr(self.num_nheads_in_l2_ptr is not None):
                        if const_expr(self.virtual_batch_idx_ptr is not None):
                            nheads_in_l2 = Int32(
                                self.num_nheads_in_l2_ptr[self.virtual_batch_idx_ptr[batch_idx]]
                            )
                        else:
                            nheads_in_l2 = Int32(self.num_nheads_in_l2_ptr[batch_idx])
                    else:
                        # TODO: 严格来说应读取 seqlen_kv，但这里假设 seqlen_q == seqlen_k
                        num_n_blocks = (
                            num_m_blocks
                            * self.tile_shape_mn[0]
                            * self.cluster_shape_m
                            // self.qhead_per_kvhead_packgqa
                            // self.tile_shape_mn[1]
                        )
                        # 让 nheads_in_l2 是 2 的幂似乎更快
                        nheads_in_l2 = (
                            16
                            if num_n_blocks * 16 <= self.max_kvblock_in_l2
                            else (
                                8
                                if num_n_blocks * 8 <= self.max_kvblock_in_l2
                                else (
                                    4
                                    if num_n_blocks * 4 <= self.max_kvblock_in_l2
                                    else (2 if num_n_blocks * 2 <= self.max_kvblock_in_l2 else 1)
                                )
                            )
                        )
                        nheads_in_l2 = min(nheads_in_l2, self.num_head)
                    mh_in_l2 = nheads_in_l2 * num_m_blocks
                    section_idx = mh_block // mh_in_l2
                    l2_mod = mh_block - section_idx * mh_in_l2
                    nheads_in_this_section = (
                        nheads_in_l2
                        if nheads_in_l2 * (section_idx + 1) <= self.num_head
                        else self.num_head - section_idx * nheads_in_l2
                    )
                    block = l2_mod // nheads_in_this_section
                    head_idx_residual = l2_mod - block * nheads_in_this_section
                    head_idx = section_idx * nheads_in_l2 + head_idx_residual
                else:
                    head_split_idx = mh_block // num_m_blocks
                    block = mh_block - head_split_idx * num_m_blocks
                    head_idx = head_split_idx // num_splits
                    split_idx = head_split_idx - head_idx * num_splits
                if const_expr(self.lpt):
                    block = num_m_blocks - 1 - block
            else:
                head_split_idx = mh_block // num_m_blocks
                block = mh_block - head_split_idx * num_m_blocks
                if const_expr(self.is_split_kv):
                    head_idx = head_split_idx // num_splits
                    split_idx = head_split_idx - head_idx * num_splits
                else:
                    head_idx = head_split_idx

            if const_expr(self.cluster_shape_m > 1 and not self.use_cluster_idx):
                bidx_in_cluster = cute.arch.block_in_cluster_idx()
                block = block * self.cluster_shape_m + bidx_in_cluster[0]

        return block, head_idx, batch_idx, split_idx, num_splits, group_start_tile, is_valid


class SingleTileVarlenScheduler:
    @dataclass
    class Params(ParamsBase):
        total_q: Int32
        scheduling_mode: cutlass.Constexpr[SchedulingMode]
        decoder: VarlenDecoder

        @staticmethod
        @cute.jit
        def create(
            args: TileSchedulerArguments,
            *,
            scheduling_mode: SchedulingMode = SchedulingMode.STATIC,
            loc=None,
            ip=None,
        ) -> "SingleTileVarlenScheduler.Params":
            assert scheduling_mode in (SchedulingMode.STATIC, SchedulingMode.CLC), (
                f"Only STATIC and CLC are supported, got {scheduling_mode!r}"
            )
            assert args.mCuSeqlensQ is not None or args.mSeqUsedQ is not None, (
                "At least one of mCuSeqlensQ or mSeqUsedQ must be provided"
            )
            assert args.cluster_shape_mn[1] == 1, "Only cluster_shape_mn[1] == 1 is supported"
            decoder = VarlenDecoder.create(
                args,
                fold_splits_into_scan=False,
                head_swizzle=args.head_swizzle,
                cluster_shape_m=args.cluster_shape_mn[0],
                scheduling_mode=scheduling_mode,
                loc=loc,
                ip=ip,
            )
            return SingleTileVarlenScheduler.Params(
                total_q=args.total_q,
                scheduling_mode=scheduling_mode,
                decoder=decoder,
            )

    def __init__(
        self,
        params: Params,
        tile_idx: Int32,
        split_idx: Int32,
        ctx: SchedulerState | None = None,
        *,
        loc=None,
        ip=None,
    ):
        self.params = params
        self._tile_idx = tile_idx
        self._split_idx = split_idx
        self._is_first_block = True
        self._ctx = ctx
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(
        args: TileSchedulerArguments,
        *,
        scheduling_mode: SchedulingMode = SchedulingMode.STATIC,
        loc=None,
        ip=None,
    ) -> Params:
        return SingleTileVarlenScheduler.Params.create(
            args, scheduling_mode=scheduling_mode, loc=loc, ip=ip
        )

    @staticmethod
    @cute.jit
    def clc_problem_shape(params: Params):
        return ClcDynamicPersistentTileSchedulerParams(
            problem_shape_ntile_mnl=SingleTileVarlenScheduler.get_grid_shape(params),
            cluster_shape_mnk=(1, 1, 1),
        )

    @staticmethod
    @cute.jit
    def create(
        params: Params, ctx: SchedulerState | None = None, *, loc=None, ip=None
    ) -> "SingleTileVarlenScheduler":
        if const_expr(params.scheduling_mode == SchedulingMode.CLC):
            block_idx = cute.arch.block_idx()
            split_idx = Int32(0)
            if const_expr(params.decoder.is_split_kv):
                split_idx = block_idx[1]
            return SingleTileVarlenScheduler(
                params,
                block_idx[0],
                split_idx,
                ctx,
                loc=loc,
                ip=ip,
            )
        tile_idx, split_idx, _ = cute.arch.block_idx()
        return SingleTileVarlenScheduler(params, tile_idx, split_idx, loc=loc, ip=ip)

    # 由主机（host）调用
    @staticmethod
    def get_grid_shape(
        params: Params,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        d = params.decoder
        total_blocks_max = (
            params.total_q + d.num_batch * (d.cluster_shape_m * d.tile_shape_mn[0] - 1)
        ) // d.tile_shape_mn[0]
        # 向下取整到最近的 cluster 倍数，因为多出的奇数部分总是填充。
        total_blocks_max = total_blocks_max // d.cluster_shape_m * d.cluster_shape_m
        return (total_blocks_max * d.num_head, d.num_splits, Int32(1))

    @cute.jit
    def _decode_work_tile(self) -> WorkTileInfo:
        """通过 warp 级前缀和把 self._tile_idx 映射到 (block, head, batch, split)。"""
        d = self.params.decoder
        next_tile_idx = self._tile_idx // d.cluster_shape_m
        block, head_idx, batch_idx, _, _, _, is_valid = d.decode(next_tile_idx, Int32(0), Int32(0))
        is_valid = is_valid and self._is_first_block
        split_idx = self._split_idx if const_expr(d.is_split_kv) else Int32(0)
        if const_expr(d.virtual_batch_idx_ptr is not None):
            if is_valid:
                batch_idx = d.virtual_batch_idx_ptr[batch_idx]
        # 把动态的逐 batch num_splits 打包进 split_idx 的高 16 位
        if const_expr(d.is_split_kv and d.num_splits_dynamic_ptr is not None):
            if is_valid:
                num_splits = Int32(d.num_splits_dynamic_ptr[batch_idx])
                split_idx = split_idx | (num_splits << 16)
        return WorkTileInfo(
            (Int32(block), Int32(head_idx), Int32(batch_idx), Int32(split_idx)),
            is_valid,
        )

    @cute.jit
    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            clc_work = self._ctx.get_current_work()
            # 默认取 grid_dim（最后一个有效扁平索引之后的一个位置），使 CLC 耗尽时
            # _decode_work_tile 返回 is_valid=False。CLC 无效时的 tile_idx 是垃圾值，
            # 不能信任。先算到局部变量再赋值，可避免运行时 if 中 self 的
            # CuTe DSL 结构不匹配。
            new_tile_idx = cute.arch.grid_dim()[0]
            new_split_idx = Int32(0)
            if clc_work.is_valid_tile:
                new_tile_idx = clc_work.tile_idx[0]
                if const_expr(self.params.decoder.is_split_kv):
                    new_split_idx = clc_work.tile_idx[1]
            self._tile_idx = new_tile_idx
            self._split_idx = new_split_idx
        return self._decode_work_tile()

    @cute.jit
    def initial_work_tile_info(self, *, loc=None, ip=None):
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            clc_work = self._ctx.initial_work_tile_info()
            # 为什么用 grid_dim 和"先局部后赋值"，参见 get_current_work。
            new_tile_idx = cute.arch.grid_dim()[0]
            new_split_idx = Int32(0)
            if clc_work.is_valid_tile:
                new_tile_idx = clc_work.tile_idx[0]
                if const_expr(self.params.decoder.is_split_kv):
                    new_split_idx = clc_work.tile_idx[1]
            self._tile_idx = new_tile_idx
            self._split_idx = new_split_idx
        return self._decode_work_tile()

    def prefetch_next_work(self, *, loc=None, ip=None):
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            self._ctx.prefetch_next_work(loc=loc, ip=ip)

    def advance_to_next_work(self, *, loc=None, ip=None):
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            self._ctx.consumer_wait(loc=loc, ip=ip)
            work = self.get_current_work()
            self._ctx.consumer_release(loc=loc, ip=ip)
            return work
        self._is_first_block = False
        return self.get_current_work()

    def producer_tail(self, *, loc=None, ip=None):
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            self._ctx.producer_tail(loc=loc, ip=ip)

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        objs = [self.params, self._tile_idx, self._split_idx]
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            objs += [self._ctx]
        for obj in objs:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        objs = [self.params, self._tile_idx, self._split_idx]
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            objs += [self._ctx]
        for obj, n_items in zip(objs, self._values_pos):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        scheduler = self.__class__(*obj_list, loc=self._loc)
        # 参见 SingleTileScheduler 中关于 Python 专属属性的说明。
        scheduler._is_first_block = self._is_first_block
        return scheduler


class DynamicPersistentVarlenScheduler:
    @dataclass
    class Params(ParamsBase):
        total_q: Int32
        decoder: VarlenDecoder
        tile_count_semaphore: Optional[cute.Pointer] = None
        persistent_cta_multiplier: cutlass.Constexpr[int] = 1

        @staticmethod
        @cute.jit
        def create(
            args: TileSchedulerArguments, *, loc=None, ip=None
        ) -> "DynamicPersistentVarlenScheduler.Params":
            assert args.mCuSeqlensQ is not None or args.mSeqUsedQ is not None, (
                "At least one of mCuSeqlensQ or mSeqUsedQ must be provided"
            )
            # TODO: 在后续 PR 中支持非平凡簇形状
            assert args.cluster_shape_mn[0] == 1 and args.cluster_shape_mn[1] == 1, (
                "DynamicPersistentVarlenScheduler currently requires cluster_shape_mn == (1, 1)"
            )
            decoder = VarlenDecoder.create(
                args,
                fold_splits_into_scan=True,
                scheduling_mode=SchedulingMode.DYNAMIC,
                loc=loc,
                ip=ip,
            )
            return DynamicPersistentVarlenScheduler.Params(
                total_q=args.total_q,
                decoder=decoder,
                tile_count_semaphore=args.tile_count_semaphore,
                persistent_cta_multiplier=args.persistent_cta_multiplier,
            )

    def __init__(
        self,
        params: Params,
        ctx: SchedulerState,
        bidb_start: Int32,
        group_start_tile: Int32,
        *,
        loc=None,
        ip=None,
    ):
        self.params = params
        self._ctx = ctx
        self._bidb_start = bidb_start
        self._group_start_tile = group_start_tile
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(
        args: TileSchedulerArguments,
        *,
        scheduling_mode: SchedulingMode = SchedulingMode.DYNAMIC,
        loc=None,
        ip=None,
    ) -> Params:
        assert scheduling_mode == SchedulingMode.DYNAMIC, (
            f"DynamicPersistentVarlenScheduler only supports DYNAMIC, got {scheduling_mode!r}"
        )
        return DynamicPersistentVarlenScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    @cute.jit
    def create(
        params: Params,
        ctx: SchedulerState,
        *,
        loc=None,
        ip=None,
    ) -> "DynamicPersistentVarlenScheduler":
        return DynamicPersistentVarlenScheduler(params, ctx, Int32(0), Int32(0), loc=loc, ip=ip)

    # 由主机（host）调用
    @staticmethod
    def get_grid_shape(
        params: Params,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        d = params.decoder
        total_blocks_max = (
            params.total_q + d.num_batch * (d.tile_shape_mn[0] - 1)
        ) // d.tile_shape_mn[0]
        total_blocks = total_blocks_max * d.num_head * d.num_splits
        hardware_info = HardwareInfo()
        sm_count = (
            hardware_info.get_device_multiprocessor_count() * params.persistent_cta_multiplier
        )
        return (cutlass.min(sm_count, total_blocks), Int32(1), Int32(1))

    @cute.jit
    def get_current_work(
        self,
        next_tile_idx: Int32,
        bidb_start: Int32,
        group_start_tile: Int32,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[WorkTileInfo, Int32]:
        d = self.params.decoder
        block, head_idx, batch_idx, split_idx, num_splits, group_start_tile, is_valid = d.decode(
            next_tile_idx, bidb_start, group_start_tile
        )
        if const_expr(d.is_split_kv and d.num_splits_dynamic_ptr is not None):
            if is_valid:
                split_idx = split_idx | (num_splits << 16)
        if const_expr(d.virtual_batch_idx_ptr is not None):
            if is_valid:
                batch_idx = d.virtual_batch_idx_ptr[batch_idx]
        return (
            WorkTileInfo(
                (Int32(block), Int32(head_idx), Int32(batch_idx), Int32(split_idx)),
                is_valid,
            ),
            group_start_tile,
        )

    @cute.jit
    def prefetch_next_work(self, *, loc=None, ip=None):
        ctx = self._ctx
        next_tile_idx = Int32(0)
        if cute.arch.lane_idx() == 0:
            next_tile_idx = cute.arch.grid_dim()[0] + utils.atomic_add_i32(
                1,
                self.params.tile_count_semaphore,
            )
        next_tile_idx = cute.arch.shuffle_sync(next_tile_idx, 0)
        work_info, new_group_start_tile = self.get_current_work(
            next_tile_idx, self._bidb_start, self._group_start_tile
        )
        # 推进扫描状态，使下一次预取从该 tile 所属的 batch 组继续，
        # 而不是从 batch 0 重新开始。
        self._bidb_start = Int32(work_info.tile_idx[2])
        self._group_start_tile = new_group_start_tile
        ctx.producer_acquire()
        with cute.arch.elect_one():
            block, head_idx, batch_idx, split_idx = work_info.tile_idx
            ctx.write_work_info(block, head_idx, batch_idx, split_idx)
            ctx.producer_commit()
        ctx.advance_producer_state()

    @cute.jit
    def advance_to_next_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        ctx = self._ctx
        ctx.consumer_wait()
        block = ctx._work_info[0]
        head_idx = ctx._work_info[1]
        batch_idx = ctx._work_info[2]
        split_idx = ctx._work_info[3]
        is_valid = batch_idx < self.params.decoder.num_batch
        work_info = WorkTileInfo((block, head_idx, batch_idx, split_idx), is_valid)
        ctx.consumer_release()
        return work_info

    @cute.jit
    def initial_work_tile_info(self, *, loc=None, ip=None) -> WorkTileInfo:
        cta_tile_idx, _, _ = cute.arch.block_idx()
        work_info, new_group_start_tile = self.get_current_work(cta_tile_idx, Int32(0), Int32(0))
        self._bidb_start = Int32(work_info.tile_idx[2])
        self._group_start_tile = new_group_start_tile
        return work_info

    def producer_tail(self, *, loc=None, ip=None):
        self._ctx.producer_tail(loc=loc, ip=ip)

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.params, self._ctx, self._bidb_start, self._group_start_tile]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip(
            [self.params, self._ctx, self._bidb_start, self._group_start_tile],
            self._values_pos,
        ):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return self.__class__(*obj_list, loc=self._loc)


# -----------------------------------------------------------------------------
# SM100 FMHA 专用调度器（与通用调度器分开维护）。
# -----------------------------------------------------------------------------


class Sm100FmhaStaticTileSchedulerParams:
    """表示 FMHA（融合多头注意力）静态 tile 调度器参数的类。

    本类持有初始化与配置 FMHA 操作 tile 调度器所需的配置参数。

    :ivar is_persistent: 是否使用持久化 kernel 模式。
    :type is_persistent: bool
    :ivar problem_shape_mbh: (M, B, H) 格式的问题形状。
    :type problem_shape_mbh: cute.Shape
    """

    def __init__(
        self,
        is_persistent: bool,
        problem_shape_mbh: cute.Shape,
        *,
        loc=None,
        ip=None,
    ):
        """
        用给定的参数初始化 Sm100FmhaStaticTileSchedulerParams。

        :param is_persistent: 是否使用持久化 kernel 模式。
        :type is_persistent: bool
        :param problem_shape_mbh: (M, B, H) 格式的问题形状。
        :type problem_shape_mbh: cute.Shape
        """
        self.is_persistent = is_persistent
        self.problem_shape_mbh = problem_shape_mbh
        self._loc = loc
        self._ip = ip

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.problem_shape_mbh]:
            obj_values = extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip([self.problem_shape_mbh], self._values_pos):
            obj_list.append(new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return Sm100FmhaStaticTileSchedulerParams(
            self.is_persistent, *(tuple(obj_list)), loc=self._loc
        )


class Sm100FmhaStaticTileScheduler:
    """FMHA（融合多头注意力）操作的静态 tile 调度器。

    本类管理 FMHA kernel 的工作 tile 调度，同时支持持久化与非持久化
    kernel 模式。它跟踪当前工作位置，并在问题空间上高效推进。

    :ivar _params: 调度器参数。
    :type _params: Sm100FmhaStaticTileSchedulerParams
    :ivar _blk_coord: 块坐标。
    :type _blk_coord: cute.Coord
    :ivar _grid_shape: kernel 的网格形状。
    :type _grid_shape: cute.Shape
    :ivar _is_persistent: 是否使用持久化 kernel 模式。
    :type _is_persistent: bool
    :ivar _current_work_linear_idx: 当前线性工作索引。
    :type _current_work_linear_idx: Int32
    :ivar _problem_shape_mbh: (M, B, H) 格式的问题形状。
    :type _problem_shape_mbh: cute.Layout
    :ivar _num_blocks: 问题中的块数。
    :type _num_blocks: Int32
    :ivar _is_first_block: 是否为第一个块。
    :type _is_first_block: bool
    :ivar num_persistent_sm: 持久化 SM 的数量。
    :type num_persistent_sm: Int32
    """

    def __init__(
        self,
        params: Sm100FmhaStaticTileSchedulerParams,
        current_work_linear_idx: Int32,
        blk_coord: cute.Coord,
        grid_shape: cute.Shape,
        *,
        loc=None,
        ip=None,
    ):
        """
        用给定的参数初始化 Sm100FmhaStaticTileScheduler。

        :param params: 调度器参数。
        :type params: Sm100FmhaStaticTileSchedulerParams
        :param current_work_linear_idx: 当前线性工作索引。
        :type current_work_linear_idx: Int32
        :param blk_coord: 块坐标。
        :type blk_coord: cute.Coord
        :param grid_shape: kernel 的网格形状。
        :type grid_shape: cute.Shape
        """
        self._params = params
        self._blk_coord = blk_coord
        self._grid_shape = grid_shape
        self._is_persistent = params.is_persistent
        self._current_work_linear_idx = current_work_linear_idx
        self._problem_shape_mbh = cute.make_layout(params.problem_shape_mbh, loc=loc, ip=ip)
        self._num_blocks = cute.size(self._problem_shape_mbh, loc=loc, ip=ip)
        self._is_first_block = True
        self.num_persistent_sm = cute.size(grid_shape, loc=loc, ip=ip)
        self._loc = loc
        self._ip = ip

    # 由主机（host）调用
    @staticmethod
    def get_grid_shape(
        params: Sm100FmhaStaticTileSchedulerParams,
        *,
        loc=None,
        ip=None,
    ) -> cute.Shape:
        """
        确定 FMHA kernel 的网格形状。

        对持久化 kernel，网格形状受设备可用 SM（流多处理器）数量限制；
        对非持久化 kernel，网格形状与问题形状一致。

        :param params: 调度器参数。
        :type params: Sm100FmhaStaticTileSchedulerParams

        :return: (M, B, H) 元组形式的网格形状。
        :rtype: cute.Shape
        """
        if params.is_persistent:
            hardware_info = HardwareInfo()
            sm_count = hardware_info.get_device_multiprocessor_count()
            return (
                dsl_min(sm_count, cute.size(params.problem_shape_mbh, loc=loc, ip=ip)),
                1,
                1,
            )
        else:
            return params.problem_shape_mbh

    @staticmethod
    def check_valid_work_for_seqlen_q(
        q_tiler: int,
        current_idx: Int32,
        seqlen_q: Int32,
    ) -> Boolean:
        """
        检查当前工作索引对给定 query 序列长度是否有效。

        本方法验证当前工作 tile 索引乘以 query tiler 大小后是否落在
        query 序列长度范围内。

        :param q_tiler: query tiler 大小。
        :type q_tiler: int
        :param current_idx: 当前工作索引。
        :type current_idx: Int32
        :param seqlen_q: query 序列长度。
        :type seqlen_q: Int32

        :return: 工作有效返回 True，否则返回 False。
        :rtype: Boolean
        """
        return current_idx * q_tiler < seqlen_q

    def get_current_work(self, *, loc=None, ip=None) -> cutlass.utils.WorkTileInfo:
        """
        获取当前工作 tile 的信息。

        判断当前工作是否有效，并根据 kernel 是持久化还是非持久化
        计算 tile 坐标。

        :return: 包含 tile 坐标与有效性标志的 WorkTileInfo。
        :rtype: WorkTileInfo
        """
        is_valid = (
            self._current_work_linear_idx < self._num_blocks
            if self._is_persistent
            else self._is_first_block
        )

        blk_coord = (0, 0, 0)
        if self._is_persistent:
            blk_coord = self._problem_shape_mbh.get_hier_coord(
                self._current_work_linear_idx, loc=loc, ip=ip
            )
        else:
            blk_coord = self._blk_coord

        # cur_tile_coord 是 (mid, 0, (bid, hid))
        cur_tile_coord = (
            blk_coord[0],
            0,
            (blk_coord[1], blk_coord[2]),
        )

        return cutlass.utils.WorkTileInfo(cur_tile_coord, is_valid)

    def initial_work_tile_info(self, *, loc=None, ip=None):
        """
        获取初始工作 tile 信息。

        :return: 初始 WorkTileInfo。
        :rtype: WorkTileInfo
        """
        return self.get_current_work(loc=loc, ip=ip)

    def advance_to_next_work(self, *, advance_count=1, loc=None, ip=None):
        """
        推进到下一个工作 tile 并返回它。

        持久化 kernel 按持久化 SM 数量推进。
        非持久化 kernel 则标记第一个块已处理。
        """
        if self._is_persistent:
            self._current_work_linear_idx += advance_count * self.num_persistent_sm
        self._is_first_block = False
        return self.get_current_work()

    def prefetch_next_work(self, *, loc=None, ip=None):
        """静态调度器为 no-op。"""
        pass

    def producer_tail(self, *, loc=None, ip=None):
        """静态调度器为 no-op。"""
        pass

    def __extract_mlir_values__(self):
        values = extract_mlir_values(self._params)
        values.extend(extract_mlir_values(self._current_work_linear_idx))
        values.extend(extract_mlir_values(self._blk_coord))
        values.extend(extract_mlir_values(self._grid_shape))
        return values

    def __new_from_mlir_values__(self, values):
        assert len(values) == 10
        new_params = new_from_mlir_values(self._params, values[0:3])
        new_current_work_linear_idx = new_from_mlir_values(
            self._current_work_linear_idx, [values[3]]
        )
        new_blk_coord = new_from_mlir_values(self._blk_coord, values[4:7])
        new_grid_shape = new_from_mlir_values(self._grid_shape, values[7:])
        scheduler = Sm100FmhaStaticTileScheduler(
            new_params, new_current_work_linear_idx, new_blk_coord, new_grid_shape
        )
        # 参见 SingleTileScheduler 中关于 Python 专属属性的说明。
        scheduler._is_first_block = self._is_first_block
        return scheduler


def compute_sm100_fmha_grid(
    o_shape: cute.Shape,
    cta_tiler: Tuple[int, int, int],
    is_persistent: bool,
) -> Tuple[Sm100FmhaStaticTileSchedulerParams, Tuple[int, int, int]]:
    """计算 FMHA（静态调度器）的网格参数。

    输出张量 o 的形状为 (s, d, ((h_r, h_k), b))。
    """
    tile_sched_params = Sm100FmhaStaticTileSchedulerParams(
        is_persistent,
        (
            cute.ceil_div(cute.size(o_shape[0]), cta_tiler[0]),
            cute.size(o_shape[2][0]),
            cute.size(o_shape[2][1]),
        ),
    )
    grid = Sm100FmhaStaticTileScheduler.get_grid_shape(tile_sched_params)
    return tile_sched_params, grid


##############################################################################
# FMHA CLC 动态 tile 调度器
##############################################################################


class Sm100FmhaClcDynamicTileSchedulerParams:
    """FMHA CLC 动态持久化 tile 调度器的参数。

    本类管理基于 CLC（Cluster Launch Control，簇启动控制）动态调度的
    tile 布局，适配 FMHA 的 (M, B, H) 问题形状。

    :ivar problem_shape_mbh: (M, B, H) 格式的问题形状。
    :type problem_shape_mbh: cute.Shape
    :ivar cluster_shape_mnk: (M, N, K) 格式的簇形状。
    :type cluster_shape_mnk: cute.Shape
    """

    def __init__(
        self,
        problem_shape_mbh: cute.Shape,
        cluster_shape_mnk: cute.Shape,
        *,
        loc=None,
        ip=None,
    ):
        self.problem_shape_mbh = problem_shape_mbh
        self._cluster_shape_mnk = cluster_shape_mnk
        self.cluster_shape_mn = cluster_shape_mnk[:2]
        self._loc = loc
        self._ip = ip

        # FMHA 使用 (M, B, H) 上的线性索引，转换为 (M, N, L) 风格
        # FMHA：M 维是沿序列的 tile 数，N=1，L=(B*H)
        self.problem_shape_ntile_mnl = (
            problem_shape_mbh[0],  # M tile 数
            1,  # N tile 数（FMHA 恒为 1）
            problem_shape_mbh[1] * problem_shape_mbh[2],  # L = B * H
        )

        # 创建簇到 tile 的映射布局
        self.problem_layout_ncluster_mnl = cute.make_layout(
            cute.ceil_div(self.problem_shape_ntile_mnl, cluster_shape_mnk[:2]),
            loc=loc,
            ip=ip,
        )

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [
            self.problem_shape_mbh,
            self._cluster_shape_mnk,
        ]:
            obj_values = extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        values_copy = list(values)
        for obj, n_items in zip(
            [self.problem_shape_mbh, self._cluster_shape_mnk],
            self._values_pos,
        ):
            obj_list.append(new_from_mlir_values(obj, values_copy[:n_items]))
            values_copy = values_copy[n_items:]
        return Sm100FmhaClcDynamicTileSchedulerParams(*(tuple(obj_list)), loc=self._loc)

    def get_grid_shape(self, *, loc=None, ip=None) -> Tuple[int, int, int]:
        """计算与簇形状对齐的网格形状。"""
        return cute.round_up(self.problem_shape_ntile_mnl, self._cluster_shape_mnk)

    def clc_hw_params(self) -> ClcDynamicPersistentTileSchedulerParams:
        """返回上游 CLC 硬件调度器的参数。"""
        return ClcDynamicPersistentTileSchedulerParams(
            problem_shape_ntile_mnl=self.problem_shape_ntile_mnl,
            cluster_shape_mnk=self._cluster_shape_mnk,
        )


class Sm100FmhaClcDynamicTileScheduler:
    """FMHA 的 CLC 动态持久化 tile 调度器。

    该调度器使用 Blackwell 的 Cluster Launch Control 硬件机制进行动态
    tile 分发，提供自动负载均衡。适配 FMHA 的 (M, B, H) 问题形状。
    """
    # 讲解：CLC（Cluster Launch Control）是 Blackwell 的硬件级动态调度：
    # 硬件按需把工作 tile 分配给空闲的 CTA/簇，避免软件原子计数器的争用
    # 与调度偏差，实现接近最优的负载均衡。

    def __init__(
        self,
        params: Sm100FmhaClcDynamicTileSchedulerParams,
        cta_id_in_cluster: cute.Coord,
        num_tiles_executed: Int32,
        clc_response_ptr: cute.Pointer,
        block_idx: Tuple,
        clc: ClcSchedulerState = None,
        *,
        loc=None,
        ip=None,
    ):
        self.params = params
        self.cta_id_in_cluster = cta_id_in_cluster
        self._num_tiles_executed = num_tiles_executed
        self._clc_response_ptr = clc_response_ptr
        self._block_idx = block_idx
        self.clc = clc
        self._loc = loc
        self._ip = ip

    def __extract_mlir_values__(self):
        values = extract_mlir_values(self.cta_id_in_cluster)
        values.extend(extract_mlir_values(self._num_tiles_executed))
        values.extend(extract_mlir_values(self._clc_response_ptr))
        values.extend(extract_mlir_values(self._block_idx))
        if self.clc is not None:
            values.extend(extract_mlir_values(self.clc))
        return values

    def __new_from_mlir_values__(self, values):
        new_cta_id_in_cluster = new_from_mlir_values(self.cta_id_in_cluster, values[0:3])
        new_num_tiles_executed = new_from_mlir_values(self._num_tiles_executed, [values[3]])
        new_clc_response_ptr = new_from_mlir_values(self._clc_response_ptr, [values[4]])
        new_block_idx = new_from_mlir_values(self._block_idx, values[5:8])
        new_clc = None
        if self.clc is not None:
            new_clc = new_from_mlir_values(self.clc, values[8:])
        return Sm100FmhaClcDynamicTileScheduler(
            self.params,
            new_cta_id_in_cluster,
            new_num_tiles_executed,
            new_clc_response_ptr,
            new_block_idx,
            new_clc,
        )

    @staticmethod
    def create(
        params: Sm100FmhaClcDynamicTileSchedulerParams,
        block_idx: Tuple,
        grid_dim: Tuple,
        clc_response_ptr: cute.Pointer,
        clc: ClcSchedulerState = None,
        *,
        loc=None,
        ip=None,
    ):
        """创建 CLC 动态 tile 调度器实例。"""
        bidx, bidy, bidz = block_idx

        # 簇内的 CTA id
        cta_id_in_cluster = (
            Int32(bidx % params.cluster_shape_mn[0]),
            Int32(bidy % params.cluster_shape_mn[1]),
            Int32(0),
        )

        num_tiles_executed = Int32(0)

        return Sm100FmhaClcDynamicTileScheduler(
            params,
            cta_id_in_cluster,
            num_tiles_executed,
            clc_response_ptr,
            block_idx,
            clc,
        )

    @staticmethod
    def get_grid_shape(
        params: Sm100FmhaClcDynamicTileSchedulerParams,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[int, int, int]:
        """获取 kernel 启动用的网格形状。"""
        return params.get_grid_shape(loc=loc, ip=ip)

    def work_tile_info_from_clc_response(self, result_addr: cute.Pointer, *, loc=None, ip=None):
        """解析 CLC 响应并转换为 FMHA tile 坐标。"""
        m_idx, n_idx, l_idx, vld = cute.arch.clc_response(result_addr, loc=loc, ip=ip)
        cute.arch.fence_proxy("async.shared", space="cta")

        # CLC 返回第一个 CTA 的坐标：m_idx=x, l_idx=z
        # l_idx 是 L（batch）维；解码为 (bid, hid)
        hid = l_idx % self.params.problem_shape_mbh[2]
        bid = l_idx // self.params.problem_shape_mbh[2]

        cta_idx_in_cluster, cta_idy_in_cluster, _ = self.cta_id_in_cluster
        cur_tile_coord = (
            m_idx + cta_idx_in_cluster,  # M 维
            0,  # FMHA 的 N 恒为 0
            (bid, hid),  # (B, H) 打包
        )

        return cutlass.utils.WorkTileInfo(cur_tile_coord, vld)

    def get_current_work(self, *, loc=None, ip=None):
        """从 CLC 响应获取当前工作 tile。"""
        return self.work_tile_info_from_clc_response(self._clc_response_ptr, loc=loc, ip=ip)

    def initial_work_tile_info(self, *, loc=None, ip=None):
        """根据块索引获取初始工作 tile。"""
        bidx, bidy, bidz = self._block_idx
        # bidz 是 L（batch）维；解码为 (bid, hid)
        hid = bidz % self.params.problem_shape_mbh[2]
        bid = bidz // self.params.problem_shape_mbh[2]
        return cutlass.utils.WorkTileInfo((bidx, 0, (bid, hid)), True)

    def advance_to_next_work(self, *, loc=None, ip=None):
        """消费者侧推进：等待下一个 tile、读取坐标、释放。"""
        self.clc.consumer_wait(loc=loc, ip=ip)
        work = self.get_current_work(loc=loc, ip=ip)
        self.clc.consumer_release(loc=loc, ip=ip)
        self._num_tiles_executed += Int32(1)
        return work

    def prefetch_next_work(self, *, loc=None, ip=None):
        """生产者侧：为下一个 tile 发起 CLC 查询。"""
        self.clc.prefetch_next_work(loc=loc, ip=ip)

    def producer_tail(self, *, loc=None, ip=None):
        """最后一个 tile 之后的生产者侧清理。"""
        self.clc.producer_tail(loc=loc, ip=ip)

    @property
    def num_tiles_executed(self) -> Int32:
        return self._num_tiles_executed


def compute_sm100_fmha_grid_clc(
    o_shape: cute.Shape,
    cta_tiler: Tuple[int, int, int],
    cluster_shape_mnk: Tuple[int, int, int],
) -> Tuple[Sm100FmhaClcDynamicTileSchedulerParams, Tuple[int, int, int]]:
    """计算带 CLC 动态调度的 FMHA 网格参数。"""
    problem_shape_mbh = (
        cute.ceil_div(cute.size(o_shape[0]), cta_tiler[0]),
        cute.size(o_shape[2][0]),
        cute.size(o_shape[2][1]),
    )
    tile_sched_params = Sm100FmhaClcDynamicTileSchedulerParams(problem_shape_mbh, cluster_shape_mnk)
    grid = Sm100FmhaClcDynamicTileScheduler.get_grid_shape(tile_sched_params)
    return tile_sched_params, grid


##############################################################################
# 融合掩码（Fused Mask）
##############################################################################


def make_sm100_thread_cooperative_group(size: int):
    return cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, size)


SM100_TMEM_CAPACITY_COLUMNS = 512
