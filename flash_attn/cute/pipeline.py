# Copyright (c) 2025, Tri Dao.

from typing import Optional
from dataclasses import dataclass

import cutlass.cute as cute
from cutlass import Boolean, Int32, const_expr
from cutlass.cutlass_dsl import if_generate, dsl_user_op
from cutlass.pipeline import PipelineState
from cutlass.pipeline import PipelineUserType
from cutlass.pipeline import NamedBarrier as NamedBarrierOg
from cutlass.pipeline import PipelineAsync as PipelineAsyncOg
from cutlass.pipeline import PipelineCpAsync as PipelineCpAsyncOg
from cutlass.pipeline import PipelineTmaAsync as PipelineTmaAsyncOg
from cutlass.pipeline import PipelineTmaUmma as PipelineTmaUmmaOg
from cutlass.pipeline import PipelineUmmaAsync as PipelineUmmaAsyncOg
from cutlass.pipeline import PipelineAsyncUmma as PipelineAsyncUmmaOg


def _override_create(parent_cls, child_cls):
    """创建静态工厂：先构造 parent_cls，再将其重新归类（re-class）为 child_cls。"""

    @staticmethod
    def create(*args, **kwargs):
        obj = parent_cls.create(*args, **kwargs)
        # 由于 dataclass 是 frozen 的，不能直接给 __class__ 赋值
        object.__setattr__(obj, "__class__", child_cls)
        return obj

    return create


def _make_state(index: Int32, phase: Int32) -> PipelineState:
    """用 index 和 phase 构造 PipelineState（count/stages 调用方不使用）。"""
    return PipelineState(stages=0, count=Int32(0), index=index, phase=phase)


class PipelineStateSimple:
    """
    流水线状态：包含与环形缓冲区当前位置对应的 index 和 phase 位。
    用单个 Int32 同时存储 index 和 phase 位，再用 divmod 取出 index 和 phase。
    若 stages 是 2 的幂，divmod 可退化为位运算。
    """
    # 讲解：这是软件流水线的核心状态 —— 生产者（TMA 加载）与消费者（GEMM 计算）
    # 通过 (index, phase) 判定环形缓冲区的槽位是否"满/空"；phase 位区分轮次，
    # 使上一轮的旧数据与新数据不会互相覆盖，从而在加载下一块的同时计算当前块。

    def __init__(self, stages: int, phase_index: Int32):
        self._stages = stages
        self._phase_index = phase_index

    def clone(self) -> "PipelineStateSimple":
        return PipelineStateSimple(self.stages, self._phase_index)

    @property
    def stages(self) -> int:
        return self._stages

    @property
    def index(self) -> Int32:
        if const_expr(self._stages == 1):
            return Int32(0)
        else:
            return self._phase_index % self._stages

    @property
    def phase(self) -> Int32:
        # PTX 文档要求 phase 奇偶位只能是 0 或 1，因此严格来说需要取模 2；
        # 但实践中不取模直接把 phase 传进去也能正常工作。
        if const_expr(self._stages == 1):
            return self._phase_index
        else:
            return self._phase_index // self._stages

    def advance(self):
        if const_expr(self._stages == 1):
            self._phase_index ^= 1
        else:
            self._phase_index += 1

    def __extract_mlir_values__(self):
        phase_index = self._phase_index
        return [phase_index.ir_value()]

    def __new_from_mlir_values__(self, values):
        return PipelineStateSimple(self.stages, Int32(values[0]))


def make_pipeline_state(type: PipelineUserType, stages: int):
    """
    创建流水线状态。生产方（Producer）假定从空缓冲区开始，且 phase 位翻转为 1。
    """
    if type is PipelineUserType.Producer:
        return PipelineStateSimple(stages, Int32(stages))
    elif type is PipelineUserType.Consumer:
        return PipelineStateSimple(stages, Int32(0))
    else:
        assert False, "Error: invalid PipelineUserType specified for make_pipeline_state."


# ── 共享辅助函数 ───────────────────────────────────────────────────────────


def _call_with_elect_one(parent_method, self, state, elect_one, syncwarp, loc, ip):
    """可选地给父类流水线方法调用包一层 sync_warp + elect_one。"""
    if const_expr(elect_one):
        if const_expr(syncwarp):
            cute.arch.sync_warp()
        with cute.arch.elect_one():
            parent_method(self, state, loc=loc, ip=ip)
    else:
        parent_method(self, state, loc=loc, ip=ip)


# ── Mixin：_w_index / _w_index_phase 变体，委托给父类 ───────────────────────
# 每个父类都有基于 PipelineState 的方法（producer_acquire、producer_commit、
# consumer_wait、consumer_release）。_w_index_phase 变体只是用 (index, phase)
# 构造一个 PipelineState 然后委托给父类。


class _PipelineIndexPhaseMixin:
    """提供 _w_index_phase / _w_index 方法的 Mixin，委托给基于 PipelineState 的父类。"""

    @dsl_user_op
    def producer_acquire_w_index_phase(
        self,
        index: Int32,
        phase: Int32,
        try_acquire_token: Optional[Boolean] = None,
        *,
        loc=None,
        ip=None,
    ):
        state = _make_state(index, phase)
        # 调用父类的 producer_acquire（它接收 PipelineState）
        self.producer_acquire(state, try_acquire_token, loc=loc, ip=ip)

    @dsl_user_op
    def producer_commit_w_index(self, index: Int32, *, loc=None, ip=None):
        state = _make_state(index, Int32(0))
        self.producer_commit(state, loc=loc, ip=ip)

    @dsl_user_op
    def consumer_wait_w_index_phase(
        self,
        index: Int32,
        phase: Int32,
        try_wait_token: Optional[Boolean] = None,
        *,
        loc=None,
        ip=None,
    ):
        state = _make_state(index, phase)
        self.consumer_wait(state, try_wait_token, loc=loc, ip=ip)

    @dsl_user_op
    def consumer_release_w_index(self, index: Int32, *, loc=None, ip=None):
        state = _make_state(index, Int32(0))
        self.consumer_release(state, loc=loc, ip=ip)


# ── NamedBarrier ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class NamedBarrier(NamedBarrierOg):
    create = _override_create(NamedBarrierOg, None)  # patched below

    @dsl_user_op
    def arrive_w_index(self, index: Int32, *, loc=None, ip=None) -> None:
        """
        arrive 的对齐（aligned）版本用于 CTA 内所有线程执行同一条指令的场景。
        参见 PTX 文档。
        """
        cute.arch.barrier_arrive(
            barrier_id=self.barrier_id + index,
            number_of_threads=self.num_threads,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def arrive_and_wait_w_index(self, index: Int32, *, loc=None, ip=None) -> None:
        cute.arch.barrier(
            barrier_id=self.barrier_id + index,
            number_of_threads=self.num_threads,
            loc=loc,
            ip=ip,
        )


NamedBarrier.create = _override_create(NamedBarrierOg, NamedBarrier)


# ── PipelineAsync ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PipelineAsync(_PipelineIndexPhaseMixin, PipelineAsyncOg):
    """
    PipelineAsync，支持 producer_commit 和 consumer_release 的可选 elect_one。

    当 elect_one_*=True（在 create 时设置）时，每个 warp 只有被选中的单个线程
    发出 barrier arrive。这在掩码计数（mask count）被设为每 warp 1 时很有用。

    create 参数：
        elect_one_commit: 若为 True，只有被选中的线程发出 producer_commit。
        syncwarp_before_commit: 若为 True（默认），在 elect_one 之前发出 syncwarp。
        elect_one_release: 若为 True，只有被选中的线程发出 consumer_release。
        syncwarp_before_release: 若为 True（默认），在 elect_one 之前发出 syncwarp。
            当线程已经汇聚（例如在 wgmma wait_group 之后）时，把 syncwarp 设为 False。
    """

    _elect_one_commit: bool = False
    _syncwarp_before_commit: bool = True
    _elect_one_release: bool = False
    _syncwarp_before_release: bool = True

    @staticmethod
    def create(
        *args,
        elect_one_commit: bool = False,
        syncwarp_before_commit: bool = True,
        elect_one_release: bool = False,
        syncwarp_before_release: bool = True,
        **kwargs,
    ):
        obj = PipelineAsyncOg.create(*args, **kwargs)
        object.__setattr__(obj, "__class__", PipelineAsync)
        object.__setattr__(obj, "_elect_one_commit", elect_one_commit)
        object.__setattr__(obj, "_syncwarp_before_commit", syncwarp_before_commit)
        object.__setattr__(obj, "_elect_one_release", elect_one_release)
        object.__setattr__(obj, "_syncwarp_before_release", syncwarp_before_release)
        return obj

    @dsl_user_op
    def producer_commit(self, state: PipelineState, *, loc=None, ip=None):
        _call_with_elect_one(
            PipelineAsyncOg.producer_commit,
            self,
            state,
            self._elect_one_commit,
            self._syncwarp_before_commit,
            loc,
            ip,
        )

    @dsl_user_op
    def consumer_release(self, state: PipelineState, *, loc=None, ip=None):
        _call_with_elect_one(
            PipelineAsyncOg.consumer_release,
            self,
            state,
            self._elect_one_release,
            self._syncwarp_before_release,
            loc,
            ip,
        )

    # _w_index 变体继承自 _PipelineIndexPhaseMixin，委托给上面的
    # producer_commit / consumer_release。


# ── PipelineCpAsync ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PipelineCpAsync(_PipelineIndexPhaseMixin, PipelineCpAsyncOg):
    _elect_one_release: bool = False
    _syncwarp_before_release: bool = True

    @staticmethod
    def create(
        *args,
        elect_one_release: bool = False,
        syncwarp_before_release: bool = True,
        **kwargs,
    ):
        obj = PipelineCpAsyncOg.create(*args, **kwargs)
        object.__setattr__(obj, "__class__", PipelineCpAsync)
        object.__setattr__(obj, "_elect_one_release", elect_one_release)
        object.__setattr__(obj, "_syncwarp_before_release", syncwarp_before_release)
        return obj

    @dsl_user_op
    def consumer_release(self, state: PipelineState, *, loc=None, ip=None):
        _call_with_elect_one(
            PipelineCpAsyncOg.consumer_release,
            self,
            state,
            self._elect_one_release,
            self._syncwarp_before_release,
            loc,
            ip,
        )

    # _w_index 变体继承自 _PipelineIndexPhaseMixin。


# ── PipelineTmaAsync ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PipelineTmaAsync(_PipelineIndexPhaseMixin, PipelineTmaAsyncOg):
    """重写 producer_acquire，使其接收 extra_tx_count 参数。"""

    @dsl_user_op
    def producer_acquire(
        self,
        state: PipelineState,
        try_acquire_token: Optional[Boolean] = None,
        extra_tx_count: int = 0,
        *,
        loc=None,
        ip=None,
    ):
        """
        TMA producer 提交：有条件地等待缓冲区为空，并为 leader 线程块设置事务屏障。
        """
        # 讲解：TMA（张量内存加速器）异步拷贝由硬件完成，屏障需知道预期的
        # 写事务数 —— arrive_and_expect_tx 声明了额外事务数，使屏障在全部
        # TMA 数据真正到达共享内存后才放行消费者（GEMM）。
        if_generate(
            try_acquire_token is None or try_acquire_token == 0,
            lambda: self.sync_object_empty.wait(state.index, state.phase, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
        if const_expr(extra_tx_count == 0):
            self.sync_object_full.arrive(state.index, self.producer_mask, loc=loc, ip=ip)
        else:
            tx_count = self.sync_object_full.tx_count + extra_tx_count
            self.sync_object_full.arrive_and_expect_tx(state.index, tx_count, loc=loc, ip=ip)


PipelineTmaAsync.create = _override_create(PipelineTmaAsyncOg, PipelineTmaAsync)


# ── PipelineTmaUmma ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PipelineTmaUmma(_PipelineIndexPhaseMixin, PipelineTmaUmmaOg):
    """重写 producer_acquire，使其接收 extra_tx_count 参数。"""

    @dsl_user_op
    def producer_acquire(
        self,
        state: PipelineState,
        try_acquire_token: Optional[Boolean] = None,
        extra_tx_count: int = 0,
        *,
        loc=None,
        ip=None,
    ):
        """
        TMA producer 提交：有条件地等待缓冲区为空，并为 leader 线程块设置事务屏障。
        """
        if_generate(
            try_acquire_token is None or try_acquire_token == 0,
            lambda: self.sync_object_empty.wait(state.index, state.phase, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
        if const_expr(extra_tx_count == 0):
            if_generate(
                self.is_leader_cta,
                lambda: self.sync_object_full.arrive(
                    state.index, self.producer_mask, loc=loc, ip=ip
                ),
                loc=loc,
                ip=ip,
            )
        else:
            tx_count = self.sync_object_full.tx_count + extra_tx_count
            if_generate(
                self.is_leader_cta,
                lambda: self.sync_object_full.arrive_and_expect_tx(
                    state.index, tx_count, loc=loc, ip=ip
                ),
                loc=loc,
                ip=ip,
            )


PipelineTmaUmma.create = _override_create(PipelineTmaUmmaOg, PipelineTmaUmma)


# ── PipelineUmmaAsync ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class PipelineUmmaAsync(_PipelineIndexPhaseMixin, PipelineUmmaAsyncOg):
    pass


PipelineUmmaAsync.create = _override_create(PipelineUmmaAsyncOg, PipelineUmmaAsync)


# ── PipelineAsyncUmma ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class PipelineAsyncUmma(_PipelineIndexPhaseMixin, PipelineAsyncUmmaOg):
    pass


PipelineAsyncUmma.create = _override_create(PipelineAsyncUmmaOg, PipelineAsyncUmma)
