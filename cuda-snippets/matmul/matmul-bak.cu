#include <cublas_v2.h>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

#include "common/cuda_helper.h"
#include "common/gpu_timer.h"
#include "common/random.h"

constexpr int BLOCK_SIZE = 32;
constexpr int TILE_K = 32;
cublasHandle_t blas_handle = nullptr;

// MatMulCPUV0 是CPU版本的参考实现，它并非vanilla版本实现，而是优化了b的访存
void MatMulCPUv0(float *a, float *b, float *c, int m, int n, int k) {
  memset(c, 0, sizeof(float) * m * n);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      for (int l = 0; l < k; ++l) {
        c[i * m + j] += a[i * k + l] * b[l * n + j];
      }
    }
  }
}

// MatMulCPUV1 是CPU版本的参考实现，它并非vanilla版本实现，而是优化了b的访存
void MatMulCPUv1(float *a, float *b, float *c, int m, int n, int k) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float sum = 0;
      for (int l = 0; l < k; ++l) {
        sum += a[i * k + l] * b[l * n + j];
      }
      c[i * m + j] = sum;
    }
  }
}

// MatMulCPUV2 是CPU版本的参考实现，它并非vanilla版本实现，而是优化了b的访存
void MatMulCPUv2(float *a, float *b, float *c, int m, int n, int k) {
  memset(c, 0, sizeof(float) * m * n);
  for (int i = 0; i < m; ++i) {
    for (int l = 0; l < k; ++l) {
      // 用 A 的第 i 行 第 l个元素，计算 C 的第 i行的元素的第 l/k 部分的结果
      float ail = a[i * k + l];
      for (int j = 0; j < n; ++j) {
        c[i * n + j] += ail * b[l * n + j];
      }
    }
  }
}

// MatMulCPUv3 是CPU版本的参考实现，它并非vanilla版本实现，而是优化了b的访存
template <int T>
void MatMulCPUv3(float *a, float *b, float *c, int m, int n, int k) {
  memset(c, 0, sizeof(float) * m * n);
  for (int i = 0; i < m; i += T) {
    for (int j = 0; j < n; j += T) {
      for (int l = 0; l < k; l += T) {
        const int min_mt = std::min(i + T, m);
        const int min_nt = std::min(j + T, n);
        const int min_kt = std::min(l + T, k);
        for (int mt = i; mt < min_mt; ++mt) {
          for (int nt = j; nt < min_nt; ++nt) {
            float sum = 0;
            for (int kt = l; kt < min_kt; ++kt) {
              sum += a[mt * n + kt] * b[kt * n + nt];
            }
            c[mt * n + nt] += sum;
          }
        }
      }
    }
  }
}

template <int tileSize>
inline void matmulImplTiling(float *left, float *right, float *result, int rows, int columns,
                             int inners) {
  memset(result, 0, sizeof(float) * rows * columns);
  for (int innerTile = 0; innerTile < inners; innerTile += tileSize) {
    for (int row = 0; row < rows; row++) {
      int innerTileEnd = std::min(inners, innerTile + tileSize);
      for (int inner = innerTile; inner < innerTileEnd; inner++) {
        for (int column = 0; column < columns; column++) {
          result[row * columns + column] +=
              left[row * inners + inner] * right[inner * columns + column];
        }
      }
    }
  }
}

void CuBlasBase(float *d_a, float *d_b, float *d_c, int m, int n, int k) {
  float alpha = 1.0;
  float beta = 0;
  checkCudaErrors(cublasSgemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_b, n, d_a,
                              k, &beta, d_c, n));
}

__global__ void kernel_v0(float *a, float *b, float *c, int m, int n, int k) {
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

// MatMulV0 是最基本的矩阵乘法，每个线程计算一个输出
void MatMulV0(float *d_a, float *d_b, float *d_c, int m, int n, int k) {
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (m + blockDim.y - 1) / blockDim.y);
  kernel_v0<<<gridDim, blockDim>>>(d_a, d_b, d_c, m, n, k);
  cudaDeviceSynchronize();
}

// matmul_v1 使用了共享内存来加速矩阵乘法，对于每个线程块(BLOCK_SIZExBLOCK_SIZE)
// 内的所有线程，它们计算对应位置的输出时，读取共享内存，而不是全局内存
__global__ void kernel_v1(float *a, float *b, float *c, int m, int n, int k) {
  __shared__ float sa[BLOCK_SIZE][TILE_K];
  __shared__ float sb[TILE_K][BLOCK_SIZE];

  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;
  int gid_x = blockDim.x * blockIdx.x + tid_x;
  int gid_y = blockDim.y * blockIdx.y + tid_y;
  float sum = 0;

  int num_tile_k = (k + TILE_K - 1) / TILE_K;
  for (int l = 0; l < num_tile_k * TILE_K; l += TILE_K) {
    // 先将a中的一个tile加载到共享内存
    if (gid_y < m && tid_x < TILE_K && l + tid_x < k) {
      sa[tid_y][tid_x] = a[gid_y * k + l + tid_x];
    } else if (tid_x < TILE_K) {
      sa[tid_y][tid_x] = 0;
    }
    // 再将b中的一个tile加载到共享内存
    if (gid_x < n && tid_y < TILE_K && l + tid_y < k) {
      sb[tid_y][tid_x] = b[(l + tid_y) * n + gid_x];
    } else if (tid_y < TILE_K) {
      sb[tid_y][tid_x] = 0;
    }

    __syncthreads();
    for (int i = 0; i < TILE_K; ++i) {
      sum += sa[tid_y][i] * sb[i][tid_x];
    }
    __syncthreads();
  }
  if (gid_y < m && gid_x < n) {
    c[gid_y * n + gid_x] = sum;
  }
}

void MatMulV1(float *d_a, float *d_b, float *d_c, int m, int n, int k) {
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (m + blockDim.y - 1) / blockDim.y);

  kernel_v1<<<gridDim, blockDim, 2 * BLOCK_SIZE * TILE_K * sizeof(float)>>>(d_a, d_b, d_c, m, n, k);
  cudaDeviceSynchronize();
}

// AverageDiff 计算两个向量的平均差异
float AverageDiff(const std::vector<float> &lhs, const std::vector<float> &rhs) {
  float sum = 0;
  for (int i = 0; i < lhs.size(); ++i) {
    sum += std::fabs(lhs[i] - rhs[i]);
  }
  return sum / lhs.size();
}

using matmul_fn = void(float *, float *, float *, int, int, int);

void test_matmul_func(matmul_fn fn, const char *fun_name, float *d_a, float *d_b, float *d_c, int m,
                      int n, int k, const std::vector<float> &ref = {}, int warmup = 1,
                      int num_iter = 1) {
  for (int i = 0; i < warmup; ++i) {
    fn(d_a, d_b, d_c, m, n, k);
  }
  checkLastCudaError();

  if (!ref.empty()) {
    std::vector<float> c(ref.size());
    checkCudaErrors(cudaMemcpy(c.data(), d_c, sizeof(float) * c.size(), cudaMemcpyDeviceToHost));
    float diff = AverageDiff(c, ref);
    if (diff > 1e-4) {
      printf("diff between c and c_cpu: %f\n", diff);
      exit(EXIT_FAILURE);
    }
  }

  GpuTimer timer;
  timer.Start();

  for (int i = 0; i < num_iter; ++i) {
    fn(d_a, d_b, d_c, m, n, k);
  }
  checkLastCudaError();
  timer.Stop();
  float elapsed = timer.Elapsed() / num_iter;

  float gflops = 2.0 * m * n * k * 1e-6 / elapsed;
  printf("%s: average elapsed: %.3f ms, GFlops: %.2f GFLOPS\n", fun_name, elapsed, gflops);
}

int main() {
  // 检查Cuda设备的可用情况
  // checkCudaDevice();

  constexpr int m = 1024;
  constexpr int k = 1024;
  constexpr int n = 1024;

  // 创建Host侧的矩阵，并填充随机数
  std::vector<float> a(m * k);
  std::vector<float> b(k * n);
  std::vector<float> c_cpu(m * n, 0);
  FillRandomNumbers(a);
  FillRandomNumbers(b);

  // 使用cpu实现来计算标准的输出
  test_matmul_func(MatMulCPUv0, "MatMulCPUv0", a.data(), b.data(), c_cpu.data(), m, n, k);
  test_matmul_func(MatMulCPUv1, "MatMulCPUv1", a.data(), b.data(), c_cpu.data(), m, n, k);
  test_matmul_func(MatMulCPUv2, "MatMulCPUv2", a.data(), b.data(), c_cpu.data(), m, n, k);
  test_matmul_func(MatMulCPUv3<128>, "MatMulCPUv3", a.data(), b.data(), c_cpu.data(), m, n, k);
  test_matmul_func(matmulImplTiling<16>, "matmulImplTiling", a.data(), b.data(), c_cpu.data(), m, n,
                   k);

  checkCudaErrors(cublasCreate(&blas_handle));
  // 分配设备侧的存储，并拷贝数据到设备侧
  float *d_a, *d_b, *d_c;
  checkCudaErrors(cudaMalloc(&d_a, sizeof(float) * a.size()));
  checkCudaErrors(cudaMalloc(&d_b, sizeof(float) * b.size()));
  checkCudaErrors(cudaMalloc(&d_c, sizeof(float) * c_cpu.size()));

  checkCudaErrors(cudaMemcpy(d_a, a.data(), sizeof(float) * a.size(), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_b, b.data(), sizeof(float) * b.size(), cudaMemcpyHostToDevice));

  test_matmul_func(CuBlasBase, "CuBlasBase", d_a, d_b, d_c, m, n, k, c_cpu);
  test_matmul_func(MatMulV0, "MatMulV0", d_a, d_b, d_c, m, n, k, c_cpu);
  test_matmul_func(MatMulV1, "MatMulV1", d_a, d_b, d_c, m, n, k, c_cpu);

  // 释放设备侧的存储
  checkCudaErrors(cudaFree(d_a));
  checkCudaErrors(cudaFree(d_b));
  checkCudaErrors(cudaFree(d_c));
  checkCudaErrors(cublasDestroy(blas_handle));

  return 0;
}