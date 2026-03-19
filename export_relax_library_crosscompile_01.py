# import os
# os.environ["TVM_CUDA_COMPILE_MODE"] = "nvrtc"  # 加在import tvm之前

import os
import tvm.contrib.nvcc as _nvcc
from tvm.target import Target

_original_compile = _nvcc._compile_cuda_nvcc

# def _patched_compile_cuda_nvcc(code, target_format=None, arch=None, options=None, 
#                                 path_target=None, use_nvshmem=False):
#     # 强制指定目标arch和PTX格式
#     arch = ["-gencode", "arch=compute_90,code=sm_90"]
#     if target_format is None:
#         target_format = "ptx"  # PTX可以在目标设备上JIT，更安全
#     return _original_compile(code, target_format=target_format, arch=arch, 
#                              options=options, path_target=path_target, 
#                              use_nvshmem=use_nvshmem)


# 在_patched_compile_cuda_nvcc里加日志
def _patched_compile_cuda_nvcc(code, target_format=None, arch=None, options=None, 
                                path_target=None, use_nvshmem=False):
    arch = ["-gencode", "arch=compute_90,code=sm_90"]
    target_format = "ptx"
    print(f"[PATCH] target_format={target_format}, arch={arch}")  # 确认patch生效
    return _original_compile(code, target_format=target_format, arch=arch, 
                             options=options, path_target=path_target, 
                             use_nvshmem=use_nvshmem)

_nvcc._compile_cuda_nvcc = _patched_compile_cuda_nvcc

import numpy as np
import tvm
from tvm import relax
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T
from tvm import dlight as dl
import os

# 关键：在compile之前设置，强制生成PTX而非CUBIN
# os.environ["TVM_CUDA_NVCC_FLAGS"] = "-ptx"

@I.ir_module
class TVMScriptModule:
    @T.prim_func
    def addone(A_handle: T.handle, B_handle: T.handle) -> None:
        m = T.int64()
        n = T.int64()
        A = T.match_buffer(A_handle, (m, n), "int32")
        B = T.match_buffer(B_handle, (m, n), "int32")
        T.func_attr(({"global_symbol": "addone"}))
        for i, j in T.grid(m, n):
            with T.block("addone"):
                vi, vj = T.axis.remap("SS", [i, j])
                B[vi, vj] = A[vi, vj] + T.int32(1)

    @R.function
    def main(x: R.Tensor(("m", "n"), "int32")) -> R.Tensor(("m", "n"), "int32"):
        m, n = T.int64(), T.int64()
        gv0 = R.call_tir(TVMScriptModule.addone, (x,), R.Tensor((m, n), dtype="int32"))
        return gv0

mod = TVMScriptModule
mod = relax.transform.LegalizeOps()(mod)
mod = relax.get_pipeline("zero")(mod)

target = tvm.target.Target(
    "cuda -arch=sm_90",
    host="llvm -mtriple=aarch64-linux-gnu"
)

with target:
    mod = dl.ApplyDefaultSchedule(
        dl.gpu.Matmul(),
        dl.gpu.Fallback(),
    )(mod)
    executable = tvm.compile(mod, target=target)

executable.export_library(
    "compiled_artifact_gpu_thor_01.so",
    cc="aarch64-linux-gnu-gcc",
    options=[
        "-L/usr/local/cuda/targets/aarch64-linux/lib",
        "-L/usr/local/cuda/targets/aarch64-linux/lib/stubs",
        "-lcuda",
        "-lcudart",
    ]
)