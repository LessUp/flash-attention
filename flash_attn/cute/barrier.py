import cutlass
import cutlass.cute as cute
from cutlass import Int32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm


# 讲解：以下是对全局内存锁的原子操作封装，用于 CTA 之间的栅栏（barrier）同步。
# ld_acquire 带 acquire 语义：读锁值时会确保此前其他线程对相关内存的写入已经可见，
# 与 red_release 的 release 语义配对，构成跨 CTA 的"到达-等待"（arrive-wait）同步原语。
@dsl_user_op
def ld_acquire(lock_ptr: cute.Pointer, *, loc=None, ip=None) -> cutlass.Int32:
    lock_ptr_i64 = lock_ptr.toint(loc=loc, ip=ip).ir_value()
    state = llvm.inline_asm(
        T.i32(),
        [lock_ptr_i64],
        "ld.global.acquire.gpu.b32 $0, [$1];",
        "=r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return cutlass.Int32(state)


@dsl_user_op
def red_relaxed(
    lock_ptr: cute.Pointer, val: cutlass.Constexpr[Int32], *, loc=None, ip=None
) -> None:
    lock_ptr_i64 = lock_ptr.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [lock_ptr_i64, Int32(val).ir_value(loc=loc, ip=ip)],
        "red.relaxed.gpu.global.add.s32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def red_release(
    lock_ptr: cute.Pointer, val: cutlass.Constexpr[Int32], *, loc=None, ip=None
) -> None:
    lock_ptr_i64 = lock_ptr.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [lock_ptr_i64, Int32(val).ir_value(loc=loc, ip=ip)],
        "red.release.gpu.global.add.s32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def wait_eq(lock_ptr: cute.Pointer, thread_idx: int | Int32, flag_offset: int, val: Int32) -> None:
    flag_ptr = lock_ptr + flag_offset
    if thread_idx == 0:
        read_val = Int32(0)
        while read_val != val:
            read_val = ld_acquire(flag_ptr)


@cute.jit
def arrive_inc(
    lock_ptr: cute.Pointer, thread_idx: int | Int32, flag_offset: int, val: cutlass.Constexpr[Int32]
) -> None:
    flag_ptr = lock_ptr + flag_offset
    if thread_idx == 0:
        red_release(flag_ptr, val)
        # red_relaxed(flag_ptr, val)
