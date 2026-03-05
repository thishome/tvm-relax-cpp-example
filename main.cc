#include <iostream>
#include <optional>
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/container/shape.h>
#include <tvm/runtime/vm/vm.h>
#include <cuda_runtime.h>

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

int main()
{
  std::string path = "./compiled_artifact_gpu.so";

  // Load the shared object
  tvm::ffi::Module m = tvm::ffi::Module::LoadFromFile(path);

  tvm::ffi::Optional<tvm::ffi::Function> vm_load_executable = m->GetFunction("vm_load_executable");
  CHECK(vm_load_executable != nullptr)
      << "Error: `vm_load_executable` does not exist in file `" << path << "`";
  std::cout << "Found vm_load_executable()" << std::endl;

  // Create a VM from the Executable
  tvm::ffi::Module mod = (*vm_load_executable)().cast<tvm::ffi::Module>();
  tvm::ffi::Optional<tvm::ffi::Function> vm_initialization = mod->GetFunction("vm_initialization");
  if (!vm_initialization.has_value()) {
    LOG(FATAL) << "Error: `vm_initialization` does not exist in file `" << path << "`";
  }
  std::cout << "Found vm_initialization()" << std::endl;


  // Initialize the VM
  // tvm::Device device{kDLCPU, 0};
  tvm::Device device{kDLCUDA, 0};
  (*vm_initialization)(static_cast<int>(device.device_type), static_cast<int>(device.device_id),
                       static_cast<int>(tvm::runtime::memory::AllocatorType::kPooled), static_cast<int>(kDLCPU), 0,
                       static_cast<int>(tvm::runtime::memory::AllocatorType::kPooled));
  std::cout << "vm initialized" << std::endl;

  // Create and initialize the input array
  GPUNDAlloc alloc;
  tvm::ffi::Tensor input = tvm::ffi::Tensor::FromNDAlloc(alloc, {3, 3}, {kDLInt, 32, 1}, device);
  int numel = input.shape().Product();
  std::vector<int> host_data(numel);
  for (int i = 0; i < numel; ++i) {
    host_data[i] = i;
  }
  cudaError_t err = cudaMemcpy(input.data_ptr(), host_data.data(), numel * sizeof(int), cudaMemcpyHostToDevice);
  TVM_FFI_ICHECK_EQ(err, cudaSuccess) << "cudaMemcpy failed: " << cudaGetErrorString(err);
  std::cout << "Input array initialized on GPU" << std::endl;
  
  tvm::ffi::Optional<tvm::ffi::Function> main = mod->GetFunction("main");
  CHECK(main != nullptr)
      << "Error: Entry function does not exist in file `" << path << "`";
  std::cout << "Found main()" << std::endl;
  // Run the main function
  tvm::ffi::Tensor output = (*main)(input).cast<tvm::ffi::Tensor>();

  cudaError_t err_sync = cudaDeviceSynchronize();
  TVM_FFI_ICHECK_EQ(err_sync, cudaSuccess) << "cudaDeviceSynchronize failed: " << cudaGetErrorString(err_sync);

  // Copy the output data back to the host
  int numel_output = output.shape().Product();
  std::vector<int> host_output(numel_output);
  err = cudaMemcpy(host_output.data(), output.data_ptr(), numel_output * sizeof(int), cudaMemcpyDeviceToHost);
  TVM_FFI_ICHECK_EQ(err, cudaSuccess) << "cudaMemcpy failed: " << cudaGetErrorString(err);

  std::cout << "output: " << std::endl;
  for (int i = 0; i < numel_output; ++i) {
    std::cout << host_output[i] << " ";
  }
  std::cout << std::endl;
}