#include <iostream>
#include <optional>
#include <vector>
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/container/shape.h>
#include <tvm/runtime/vm/vm.h>
#include <tvm/runtime/device_api.h> 
#include <cuda_runtime.h>
#include <tvm/ffi/container/array.h>


struct GPUNDAlloc {
  void AllocData(DLTensor* tensor) {
    size_t data_size = tvm::ffi::GetDataSize(*tensor);
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, data_size);
    TVM_FFI_ICHECK_EQ(err, cudaSuccess) << "cudaMalloc failed: " << cudaGetErrorString(err);
    std::cout << "Allocated " << data_size << " bytes on GPU" << std::endl;
    tensor->data = ptr;
  }
  void FreeData(DLTensor* tensor) {
    if (tensor->data != nullptr) {
      size_t data_size = tvm::ffi::GetDataSize(*tensor);
      cudaError_t err = cudaFree(tensor->data);
      TVM_FFI_ICHECK_EQ(err, cudaSuccess) << "cudaFree failed: " << cudaGetErrorString(err);
      std::cout << "Freed " << data_size << " bytes on GPU" << std::endl;
      tensor->data = nullptr;
    }
  }
};

int main() {
    // std::string path = "./resnet50_with_params.0415.so";
    std::string path = "./28_parking_im_quality.strongly_fp16.so"; 

    tvm::ffi::Module m = tvm::ffi::Module::LoadFromFile(path);

    // 1. 获取并初始化 VM
    auto opt_vm_load = m->GetFunction("vm_load_executable");
    CHECK(opt_vm_load.has_value()) << "Function vm_load_executable not found";
    tvm::ffi::Module mod = opt_vm_load.value()().cast<tvm::ffi::Module>();

    auto opt_vm_init = mod->GetFunction("vm_initialization");
    CHECK(opt_vm_init.has_value()) << "Function vm_initialization not found";
    
    tvm::Device device{kDLCUDA, 0};
    opt_vm_init.value()(static_cast<int>(device.device_type), static_cast<int>(device.device_id),
                      static_cast<int>(tvm::runtime::memory::AllocatorType::kPooled), 
                      static_cast<int>(kDLCPU), 0,
                      static_cast<int>(tvm::runtime::memory::AllocatorType::kPooled));

    // 2. 创建专属的 CUDA Stream 并绑定到 TVM 运行时
    cudaStream_t infer_stream;
    cudaStreamCreate(&infer_stream);
    tvm::runtime::DeviceAPI::Get(device)->SetStream(device, infer_stream);

    // 3. 构造新的输入 Tensor (保持 float32 不变)
    GPUNDAlloc alloc;
    // tvm::ffi::Tensor input = tvm::ffi::Tensor::FromNDAlloc(alloc, {1, 3, 224, 224}, {kDLFloat, 32, 1}, device);
    // size_t numel = 1 * 3 * 224 * 224;

    tvm::ffi::Tensor input = tvm::ffi::Tensor::FromNDAlloc(alloc, {4, 3, 512, 640}, {kDLFloat, 32, 1}, device);
    size_t numel = 4 * 3 * 512 * 640;

    std::vector<float> host_data(numel, 0.5f); 

    // 使用 Async 异步拷贝绑定到流上
    cudaMemcpyAsync(input.data_ptr(), host_data.data(), numel * sizeof(float), cudaMemcpyHostToDevice, infer_stream);

    auto opt_main = mod->GetFunction("main");
    CHECK(opt_main.has_value()) << "Function main not found";
    tvm::ffi::Function main_func = opt_main.value();

    // 定义一个 Lambda 表达式，自动兼容处理 Tensor 或 Array 返回值
    auto extract_tensor = [](auto raw_ret) -> tvm::ffi::Tensor {
        try {
            // 先尝试按 Array 解析，提取第一个输出
            return raw_ret.template cast<tvm::ffi::Array<tvm::ffi::Any>>()[0].template cast<tvm::ffi::Tensor>();
            // return raw_ret.template cast<tvm::ffi::Array>()[0].template cast<tvm::ffi::Tensor>();
        } catch (...) {
            // 如果报错，说明它本身就是单个 Tensor
            return raw_ret.template cast<tvm::ffi::Tensor>();
        }
    };

    std::cout << "Running Warm-up..." << std::endl;
    {
        // 4. 预热 (Warm-up)为 CUDA Graph 准备
        tvm::ffi::Tensor warmup_output = extract_tensor(main_func(input));
        tvm::ffi::Tensor warmup_output2 = extract_tensor(main_func(input));
        cudaStreamSynchronize(infer_stream);
    } 

    // 5. 捕捉 CUDA Graph
    std::cout << "Capturing CUDA Graph..." << std::endl;
    cudaGraph_t graph;
    cudaGraphExec_t instance;

    cudaStreamBeginCapture(infer_stream, cudaStreamCaptureModeGlobal);
    auto raw_output = main_func(input); 
    cudaStreamEndCapture(infer_stream, &graph);


    // 获取兼容处理后的输出 Tensor
    tvm::ffi::Tensor output = extract_tensor(raw_output);


    // 编译出可执行实例
    cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
    std::cout << "CUDA Graph instantiation complete." << std::endl;


    // 获取一次输出内存池的底层指针
    float* d_output_ptr = static_cast<float*>(output.data_ptr());
    size_t numel_output = output.shape().Product();
    std::vector<float> host_output(numel_output);


    // 6. 高效的执行循环
    for (int i = 0; i < 20; ++i) 
    {
        // a. 一键发射整个算子图
        cudaGraphLaunch(instance, infer_stream);

        // b. 将输出张量数据异步拷回 Host
        cudaMemcpyAsync(host_output.data(), d_output_ptr, numel_output * sizeof(float), cudaMemcpyDeviceToHost, infer_stream);

        // c. 流同步
        cudaStreamSynchronize(infer_stream);

        std::cout << "Times: " << i << " ,Output first 5 elements: ";
        for (int j = 0; j < 5; ++j) {
            std::cout << host_output[j] << " ";
        }
        std::cout << std::endl;
    }

    // 7. 优雅释放资源
    cudaGraphExecDestroy(instance);
    cudaGraphDestroy(graph);
    
    tvm::runtime::DeviceAPI::Get(device)->SetStream(device, nullptr);
    cudaStreamDestroy(infer_stream);

    return 0;
}