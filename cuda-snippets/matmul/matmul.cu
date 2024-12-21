#include <torch/torch.h>

__global__ void matmul_kernel(const float *a, const float *b, float *c, int m,
                              int n, int k) {
  int gid_x = blockIdx.x * blockDim.x + threadIdx.x;
  int gid_y = blockIdx.y * blockDim.y + threadIdx.y;

  if (gid_x >= n || gid_y >= m) {
    return;
  }

  float sum = 0;
  for (int i = 0; i < k; ++i) {
    sum += a[gid_y * k + i] * b[i * n + gid_x];
  }
  c[gid_y * n + gid_x] = sum;
}

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

inline constexpr int cdiv(int x, int y) { return (x + y - 1) / y; }

torch::Tensor matmul(torch::Tensor a, torch::Tensor b) {
  CHECK_CUDA(a);
  CHECK_CUDA(b);

  auto m = a.size(0);
  auto k = a.size(1);
  auto n = b.size(1);

  torch::Tensor output = torch::zeros({m, n}, a.options());

  constexpr int BLOCK_SIZE = 32;
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim(cdiv(n, blockDim.x), cdiv(m, blockDim.y));
  matmul_kernel<<<gridDim, blockDim>>>(
      a.const_data_ptr<float>(), b.const_data_ptr<float>(),
      output.mutable_data_ptr<float>(), m, n, k);

  return output;
}
