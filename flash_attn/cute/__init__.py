"""Flash Attention CUTE（CUDA 模板引擎）实现。

本包导出 FA4 的两个公共入口：flash_attn_func（标准注意力）与 flash_attn_varlen_func（变长序列）。"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fa4")
except PackageNotFoundError:
    __version__ = "0.0.0"

from .interface import (
    flash_attn_func,
    flash_attn_varlen_func,
)

__all__ = [
    "flash_attn_func",
    "flash_attn_varlen_func",
]
