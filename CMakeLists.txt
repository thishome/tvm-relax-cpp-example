cmake_minimum_required(VERSION 3.13)

project(tvm_relax_cpp_example LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(3rdparty/tvm/3rdparty/tvm-ffi tvm_ffi)

add_executable(main main.cc)
set_target_properties( main PROPERTIES
    CUDA_STANDARD 17
    CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(main PUBLIC
  ${CMAKE_SOURCE_DIR}/3rdparty/tvm/include
  ${CMAKE_SOURCE_DIR}/3rdparty/tvm/3rdparty/dlpack/include
  ${CMAKE_SOURCE_DIR}/3rdparty/tvm/3rdparty/dmlc-core/include
)

target_link_libraries(main
  ${CMAKE_SOURCE_DIR}/3rdparty/tvm/install-aarch64/lib/libtvm_runtime.so
  CUDA::cudart
)

target_link_libraries(main tvm_ffi_shared)

add_executable(main_resnet50 main_resnet50.cc)
set_target_properties( main_resnet50 PROPERTIES
    CUDA_STANDARD 17
    CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(main_resnet50 PUBLIC
  ${CMAKE_SOURCE_DIR}/3rdparty/tvm/include
  ${CMAKE_SOURCE_DIR}/3rdparty/tvm/3rdparty/dlpack/include
  ${CMAKE_SOURCE_DIR}/3rdparty/tvm/3rdparty/dmlc-core/include
)

target_link_libraries(main_resnet50
  ${CMAKE_SOURCE_DIR}/3rdparty/tvm/install-aarch64/lib/libtvm_runtime.so
  CUDA::cudart
)

target_link_libraries(main_resnet50 tvm_ffi_shared)