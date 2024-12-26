#include <torch/extension.h>

#include "matmul_kernels.cuh"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

torch::Tensor matmul_naive(torch::Tensor a, torch::Tensor b) {
  // Input validation
  CHECK_INPUT(a);
  CHECK_INPUT(b);

  // Get matrix dimensions
  auto M = a.size(0);
  auto K = a.size(1);
  auto N = b.size(1);

  // Ensure inner dimensions match
  TORCH_CHECK(b.size(0) == K, "Inner dimensions of matrices must match");

  // Create output tensor
  torch::Tensor output = torch::zeros({M, N}, a.options());

  // Configure kernel launch parameters
  constexpr int BLOCK_SIZE = 32;
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid(cdiv(N, BLOCK_SIZE), cdiv(M, BLOCK_SIZE));

  // Launch kernel
  naive_kernel<<<grid, block>>>(a.const_data_ptr<float>(), b.const_data_ptr<float>(),
                                output.data_ptr<float>(), M, N, K);

  // Check for kernel launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    TORCH_CHECK(false, "CUDA kernel failed: ", cudaGetErrorString(err));
  }

  return output;
}

torch::Tensor matmul_tiled_smem(torch::Tensor a, torch::Tensor b) {
  CHECK_INPUT(a);
  CHECK_INPUT(b);

  int M = a.size(0);
  int K = a.size(1);
  int N = b.size(1);

  // Ensure the inner dimensions match
  TORCH_CHECK(b.size(0) == K, "Inner dimensions of a and b must match.");

  torch::Tensor output = torch::zeros({M, N}, a.options());

  constexpr int BLOCK_SIZE = 16;
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim(cdiv(N, BLOCK_SIZE), cdiv(M, BLOCK_SIZE));

  tiled_smem_kernel<BLOCK_SIZE><<<gridDim, blockDim>>>(
      a.const_data_ptr<float>(), b.const_data_ptr<float>(), output.data_ptr<float>(), M, N, K);

  // Check for kernel launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    TORCH_CHECK(false, "CUDA kernel failed: ", cudaGetErrorString(err));
  }

  return output;
}

torch::Tensor matmul_strided_tiled(torch::Tensor a, torch::Tensor b) {
  CHECK_INPUT(a);
  CHECK_INPUT(b);

  int M = a.size(0);
  int K = a.size(1);
  int N = b.size(1);

  // Ensure the inner dimensions match
  TORCH_CHECK(b.size(0) == K, "Inner dimensions of a and b must match.");

  torch::Tensor output = torch::zeros({M, N}, a.options());

  // 根据kernel函数的实现，每个线程处理 2x2 个元素
  constexpr int BLOCK_SIZE = 16;
  constexpr int STRIDE = 2;
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);  // 保持完整的block大小
  // 由于每个线程处理4个元素，所以在x方向上grid大小需要相应调整
  dim3 gridDim(cdiv(N, BLOCK_SIZE * STRIDE), cdiv(M, BLOCK_SIZE * STRIDE));

  strided_tiled_kernel<BLOCK_SIZE, STRIDE><<<gridDim, blockDim>>>(
      a.const_data_ptr<float>(), b.const_data_ptr<float>(), output.data_ptr<float>(), M, N, K);

  // Check for kernel launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    TORCH_CHECK(false, "CUDA kernel failed: ", cudaGetErrorString(err));
  }

  return output;
}

torch::Tensor matmul_float4_smem_coalesced(torch::Tensor a, torch::Tensor b) {
  CHECK_INPUT(a);
  CHECK_INPUT(b);

  int M = a.size(0);
  int K = a.size(1);
  int N = b.size(1);

  // Ensure the inner dimensions match
  TORCH_CHECK(b.size(0) == K, "Inner dimensions of a and b must match.");

  torch::Tensor output = torch::zeros({M, N}, a.options());

  // 根据kernel函数的实现，每个线程处理 2x2 个元素
  constexpr int BLOCK_SIZE = 16;
  constexpr int STRIDE = 2;
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);  // 保持完整的block大小
  // 由于每个线程处理4个元素，所以在x方向上grid大小需要相应调整
  dim3 gridDim(cdiv(N, BLOCK_SIZE * STRIDE), cdiv(M, BLOCK_SIZE * STRIDE));

  float4_smem_coalesced_kernel<BLOCK_SIZE, STRIDE><<<gridDim, blockDim>>>(
      a.const_data_ptr<float>(), b.const_data_ptr<float>(), output.data_ptr<float>(), M, N, K);

  // Check for kernel launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    TORCH_CHECK(false, "CUDA kernel failed: ", cudaGetErrorString(err));
  }

  return output;
}

PYBIND11_MODULE(matmul, m) {
  m.def("matmul_naive", &matmul_naive, "Basic element-wise matrix multiplication");
  m.def("matmul_tiled_smem", &matmul_tiled_smem,
        "Matrix multiplication using shared memory tiling for better cache utilization");
  m.def("matmul_strided_tiled", &matmul_strided_tiled,
        "Matrix multiplication using shared memory tiling with strided thread mapping for improved "
        "memory access");

  m.def("matmul_float4_smem_coalesced", &matmul_float4_smem_coalesced,
        "Matrix multiplication using float4 vectorized loads/stores with shared memory and "
        "coalesced memory access");
}
