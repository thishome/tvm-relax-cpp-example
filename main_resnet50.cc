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

#include <string>
#include <fstream>

#include <sys/stat.h>

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


size_t get_file_size(const char* path){
  struct stat st;
  stat(path, &st);
  size_t size = st.st_size;
  return size;
}

int main()
{
  std::string path = "./model.so";

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
  tvm::ffi::Tensor input = tvm::ffi::Tensor::FromNDAlloc(alloc, {1, 3, 224, 224}, {kDLFloat, 32, 1}, device);
  int numel = input.shape().Product();
  // std::vector<float> host_data(numel);
  // for (int i = 0; i < numel; ++i) {
  //   host_data[i] = i;
  // }
  std::string filename("input_onnx.dat");
  size_t filesize = get_file_size(filename.c_str());
  std::vector<char> input_img(filesize);
  std::ifstream ifs(filename);
  if(!ifs.is_open()) {
    std::cout << "open file " << filename << " error!" << std::endl;
    return -1;
  }
  ifs.read(input_img.data(), filesize);
  ifs.close();
  
  cudaError_t err = cudaMemcpy(input.data_ptr(), input_img.data(), numel * sizeof(float), cudaMemcpyHostToDevice);
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
  std::vector<float> host_output(numel_output);
  err = cudaMemcpy(host_output.data(), output.data_ptr(), numel_output * sizeof(float), cudaMemcpyDeviceToHost);
  TVM_FFI_ICHECK_EQ(err, cudaSuccess) << "cudaMemcpy failed: " << cudaGetErrorString(err);

  // for(int i=0; i<numel_output; ++i) {
  //   std::cout << host_output[i] << ", ";
  // }
  // std::cout << std::endl;

  std::cout << "argmax: " << std::endl;
  int max_idx = 0;
  float max_value = host_output[max_idx];
  for (int i = 1; i < numel_output; ++i) {
    if(max_value < host_output[i]){
      max_value = host_output[i];
      max_idx = i;
    }
  }
  std::cout<< "max idx:" << max_idx << std::endl;
  std::ofstream ofs("output_thor.dat");
  if(!ifs.is_open()) {
    std::cout << "open file " << filename << " error!" << std::endl;
    return -1;
  }
  ofs.write(reinterpret_cast<char*>(host_output.data()), host_output.size() * sizeof(host_output[0]));
  std::cout << "save output to dat." << std::endl;
}