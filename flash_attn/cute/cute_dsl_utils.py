# Copyright (c) 2025, Tri Dao.

import os
from typing import Tuple
from functools import lru_cache

import torch
from torch._subclasses.fake_tensor import FakeTensor

try:
    from triton.tools.disasm import extract
except ImportError:
    extract = None

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import NumericMeta
from cutlass.cute.runtime import from_dlpack

StaticTypes = (cutlass.Constexpr, NumericMeta, int, bool, str, float, type(None))


load_cubin_module_data_og = cutlass.base_dsl.runtime.cuda.load_cubin_module_data
cute_compile_og = cute.compile


torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
    torch.float8_e4m3fn: cutlass.Float8E4M3FN,
    torch.float8_e5m2: cutlass.Float8E5M2,
}


@lru_cache
def get_max_active_clusters(cluster_size):
    return cutlass.utils.HardwareInfo().get_max_active_clusters(cluster_size=cluster_size)


@lru_cache
def get_device_capacity(device: torch.device = None) -> Tuple[int, int]:
    return torch.cuda.get_device_capability(device)


@lru_cache
def _get_device_arch_and_num_sms(device_index: int) -> tuple[int, int]:
    properties = torch.cuda.get_device_properties(device_index)
    return properties.major * 10 + properties.minor, properties.multi_processor_count


def get_num_sms_for_selection(device_index: int, arch: int) -> int:
    """返回匹配的本地 GPU 或交叉编译目标的 SM 数量。"""
    override = os.getenv("FLASH_ATTENTION_NUM_SMS")
    if override is not None:
        num_sms = int(override)
        if num_sms <= 0:
            raise ValueError("FLASH_ATTENTION_NUM_SMS must be positive")
        return num_sms
    if torch.cuda.is_available():
        device_arch, num_sms = _get_device_arch_and_num_sms(device_index)
        if device_arch == arch:
            return num_sms
    raise RuntimeError(
        "Cannot determine the target GPU's SM count; set FLASH_ATTENTION_NUM_SMS "
        "when cross-compiling without a matching local GPU"
    )


def _has_aligned_pointer(tensor: torch.Tensor, align_bytes: int) -> bool:
    address = (
        tensor.storage_offset() * tensor.element_size()
        if isinstance(tensor, FakeTensor)
        else tensor.data_ptr()
    )
    return address % align_bytes == 0


def _is_aligned_layout(tensor: torch.Tensor, align_bytes: int) -> bool:
    """返回张量是否满足 kernel 假定的指针与步长 ABI。"""
    if tensor.stride(-1) != 1 or not _has_aligned_pointer(tensor, align_bytes):
        return False
    stride_alignment = max(1, align_bytes // tensor.element_size())
    return all(stride == 0 or stride % stride_alignment == 0 for stride in tensor.stride()[:-1])


def maybe_contiguous(tensor: torch.Tensor | None, align_bytes: int = 16):
    """把输入规范化为 kernel 假定的指针与步长对齐形式。"""
    if tensor is None:
        return None
    if tensor.is_contiguous():
        return (
            tensor
            if _has_aligned_pointer(tensor, align_bytes)
            else tensor.clone(memory_format=torch.contiguous_format)
        )
    if not _has_aligned_pointer(tensor, align_bytes):
        return tensor.clone(memory_format=torch.contiguous_format)
    return tensor if _is_aligned_layout(tensor, align_bytes) else tensor.contiguous()


def validate_output_layout(tensor: torch.Tensor, name: str, align_bytes: int) -> None:
    """校验调用方提供的输出或 SplitKV 工作区。"""
    assert 0 not in tensor.stride(), f"{name} must not have broadcast dimensions"
    if tensor.is_contiguous():
        assert _has_aligned_pointer(tensor, align_bytes), (
            f"{name} must have aligned strides and a contiguous last dimension"
        )
        return
    assert _is_aligned_layout(tensor, align_bytes), (
        f"{name} must have aligned strides and a contiguous last dimension"
    )


def assume_strides_aligned(t):
    """假定除最后一维外的所有步长都能被 128 位整除。
    
    Python 整数步长（例如 GQA 扩展产生的 stride=0）保持原样，
    因为它们是静态的，不需要对齐假定。
    """
    divby = 128 // t.element_type.width
    strides = tuple(s if isinstance(s, int) else cute.assume(s, divby=divby) for s in t.stride[:-1])
    return (*strides, t.stride[-1])


def assume_tensor_aligned(t):
    """用 128 位对齐的步长假定重建张量。None 原样透传。"""
    if t is None:
        return None
    return cute.make_tensor(t.iterator, cute.make_layout(t.shape, stride=assume_strides_aligned(t)))


def to_cute_tensor(t, assumed_align=16, leading_dim=-1, fully_dynamic=False, enable_tvm_ffi=True):
    """把 torch 张量转换为 TVM FFI 用的 cute 张量。leading_dim=-1 默认取 t.ndim-1。"""
    if t is None:
        return None
    # 注意：torch 2.9.1 不支持通过 DLPack 传 fp8，但 2.11.0 nightly 支持
    # 目前先以 uint8 导出原始字节并告知 cutlass 正确的类型
    # can directly export as fp8 when torch supports it
    if t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        tensor = from_dlpack(
            t.view(torch.uint8).detach(),
            assumed_align=assumed_align,
            enable_tvm_ffi=enable_tvm_ffi,
        )
        tensor.element_type = (
            cutlass.Float8E4M3FN if t.dtype == torch.float8_e4m3fn else cutlass.Float8E5M2
        )
    else:
        tensor = from_dlpack(t.detach(), assumed_align=assumed_align, enable_tvm_ffi=enable_tvm_ffi)
    if fully_dynamic:
        return tensor.mark_layout_dynamic()
    if leading_dim == -1:
        leading_dim = t.ndim - 1
    return tensor.mark_layout_dynamic(leading_dim=leading_dim)


def to_cute_aux_tensor(t, enable_tvm_ffi=True):
    """把 torch 张量转换为 TVM FFI 用的 cute 张量，专为 FlexAttention aux 张量定制。
    允许用户为自定义 score_mod 可调用对象中使用的 aux 张量指定对齐和 leading 维。
    """
    assumed_align: int = getattr(t, "__assumed_align__", None)
    leading_dim: int = getattr(t, "__leading_dim__", None)
    fully_dynamic: bool = leading_dim is None

    return to_cute_tensor(
        t,
        assumed_align=assumed_align,
        leading_dim=leading_dim,
        fully_dynamic=fully_dynamic,
        enable_tvm_ffi=enable_tvm_ffi,
    )


def _resolve_aux_leading_dim(tensor: torch.Tensor) -> int | None:
    """选择 CuTe 保留为静态 stride-1 leading 维的 mode。"""
    leading_dim = getattr(tensor, "__leading_dim__", None)
    if leading_dim is not None:
        if tensor.ndim == 0:
            raise ValueError("Scalar aux tensors cannot declare __leading_dim__")
        leading_dim %= tensor.ndim
        if tensor.stride(leading_dim) != 1:
            raise ValueError("Aux tensor __leading_dim__ must identify a stride-1 dimension")
        return leading_dim

    unit_stride_dims = [dim for dim, stride in enumerate(tensor.stride()) if stride == 1]
    if len(unit_stride_dims) <= 1:
        return unit_stride_dims[0] if unit_stride_dims else None
    nontrivial_dims = [dim for dim in unit_stride_dims if tensor.shape[dim] > 1]
    if len(nontrivial_dims) != 1:
        raise ValueError("Aux tensor layout has no unique stride-1 leading dimension")
    return nontrivial_dims[0]


def get_aux_tensor_metadata(aux_tensors):
    """返回必须作为编译缓存 key 的静态 aux 张量 ABI 事实。"""
    metadata = []
    for tensor in aux_tensors:
        leading_dim = _resolve_aux_leading_dim(tensor)
        static_strides = tuple(
            0 if stride == 0 else 1 if dim == leading_dim else None
            for dim, stride in enumerate(tensor.stride())
        )
        metadata.append((tensor.dtype, getattr(tensor, "__assumed_align__", None), static_strides))
    return tuple(metadata)


def get_broadcast_dims(tensor: torch.Tensor) -> Tuple[bool, ...]:
    """返回布尔元组，指示哪些维是 stride=0（广播）。
    
    这对编译缓存 key 很有用：CuTe 的 mark_layout_dynamic() 会把
    stride=0 保留为静态，这意味着用不同广播模式编译的 kernel
    不可互换。
    """
    strides = tensor.stride()
    # Written this way for speed.
    if 0 not in strides:
        return (False,) * len(strides)
    return tuple(stride == 0 for stride in strides)


# credit: monellz (https://github.com/NVIDIA/cutlass/issues/2658#issuecomment-3630564264)
def dump_kernel_attributes(compiled_kernel):
    from cuda.bindings import driver
    from cutlass.utils import HardwareInfo
    import torch

    device_id = torch.cuda.current_device()
    hardware_info = HardwareInfo(device_id=device_id)
    cubin_data = compiled_kernel.artifacts.CUBIN
    assert cubin_data is not None, "cubin_data is None, need '--keep-cubin' option when compiling"
    cuda_library = hardware_info._checkCudaErrors(
        driver.cuLibraryLoadData(cubin_data, None, None, 0, None, None, 0)
    )
    kernels = hardware_info._checkCudaErrors(driver.cuLibraryEnumerateKernels(1, cuda_library))
    kernel = hardware_info._checkCudaErrors(driver.cuKernelGetFunction(kernels[0]))
    # more metrics: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html#group__CUDA__EXEC_1g5e92a1b0d8d1b82cb00dcfb2de15961b
    local_size_bytes = hardware_info._checkCudaErrors(
        driver.cuFuncGetAttribute(
            driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
            kernel,
        )
    )
    num_regs = hardware_info._checkCudaErrors(
        driver.cuFuncGetAttribute(
            driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS,
            kernel,
        )
    )

    print("--- Kernel Info ---")
    print(f"local_size_bytes: {local_size_bytes}")
    print(f"num_regs: {num_regs}")
    print("--- End Kernel Info ---")
