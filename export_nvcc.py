import tvm.contrib.nvcc as nvcc_mod
import os

# 查看原始compile_cuda函数
import inspect
# print(inspect.getsource(nvcc_mod.compile_cuda))
print(inspect.getsource(nvcc_mod._compile_cuda_nvcc))


# # 查看_compile_cuda_nvcc函数，找到arch是怎么传入的
# # python -c "
# import tvm.contrib.nvcc as m
# import inspect
# print(inspect.getsource(m._compile_cuda_nvcc))
# # "