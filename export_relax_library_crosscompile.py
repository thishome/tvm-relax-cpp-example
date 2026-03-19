import numpy as np
import tvm
from tvm import relax

from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T

from tvm import dlight as dl

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
# mod.show()

mod: tvm.IRModule = relax.transform.LegalizeOps()(mod)
mod.show()

mod: tvm.IRModule = relax.get_pipeline("zero")(mod)
mod.show()

so_name = "compiled_artifact_gpu_thor.so"
# so_name = "compiled_artifact_gpu_thor.tar"

# for thor crosscompiling
target = tvm.target.Target(
    "cuda -arch=sm_90", 
    host="llvm -mtriple=aarch64-linux-gnu"
    )
with target:
    mod = dl.ApplyDefaultSchedule(
            dl.gpu.Matmul(),
            dl.gpu.Fallback(),
        )(mod)

dev = tvm.cuda(0)

# relax.build is same with tvm.compile
# executable = relax.build(mod, target, exec_mode="compiled")
executable = tvm.compile(mod, target=target)
# executable.export_library(so_name)
executable.export_library(
    so_name,
    cc="aarch64-linux-gnu-gcc",
    options=[
        "-L/usr/local/cuda/targets/aarch64-linux/lib",
        "-L/usr/local/cuda/targets/aarch64-linux/lib/stubs",
        "-lcuda",
        "-lcudart",
    ]
)







