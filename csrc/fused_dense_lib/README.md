这个 CUDA 扩展实现了融合的 matmul + bias（前向和反向），以及融合的 matmul + bias + gelu（前向和反向），改编自 Apex 的 [FusedDense](https://github.com/NVIDIA/apex/tree/master/apex/fused_dense)。
我们让它支持 bfloat16。

为了最佳性能，应使用 CUDA >= 11.8。此前的 CuBLAS 版本对 bfloat16 没有最好的 matmul + bias + gelu 性能。

它只在 A100 上测试过。

```sh
cd csrc/fused_dense_lib && pip install .
```
