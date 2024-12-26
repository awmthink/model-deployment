#include <cublas_v2.h>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

#include "common/cuda_helper.h"
#include "common/gpu_timer.h"
#include "common/random.h"

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
  constexpr int BLOCK_SIZE = 16;
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (m + blockDim.y - 1) / blockDim.y);
  kernel_v0<<<gridDim, blockDim>>>(d_a, d_b, d_c, m, n, k);
  cudaDeviceSynchronize();
}

template <int BLOCK_SIZE>
__global__ void kernel_v1(const float *a, const float *b, float *c, int M, int N, int K) {
  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;
  int gid_x = blockIdx.x * blockDim.x + threadIdx.x;
  int gid_y = blockIdx.y * blockDim.y + threadIdx.y;

  // Declare shared memory for tiles
  __shared__ float tiled_a[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float tiled_b[BLOCK_SIZE][BLOCK_SIZE];

  float sum = 0.0f;
  for (int k = 0; k < K; k += BLOCK_SIZE) {
    // Copy tile_a to shared memory
    if (gid_y < M && (k + tid_x) < K) {
      tiled_a[tid_y][tid_x] = a[gid_y * K + k + tid_x];
    } else {
      tiled_a[tid_y][tid_x] = 0;
    }
    // Copy tile_b to shared memory
    if ((k + tid_y) < K && gid_x < N) {
      tiled_b[tid_y][tid_x] = b[(k + tid_y) * N + gid_x];
    } else {
      tiled_b[tid_y][tid_x] = 0;
    }
    __syncthreads();

    // Compute partial product
    for (int i = 0; i < BLOCK_SIZE; ++i) {
      sum += tiled_a[tid_y][i] * tiled_b[i][tid_x];
    }
    __syncthreads();
  }

  // Write the result
  if (gid_x < N && gid_y < M) {
    c[gid_y * N + gid_x] = sum;
  }
}

void MatMulV1(float *d_a, float *d_b, float *d_c, int m, int n, int k) {
  constexpr int BLOCK_SIZE = 16;
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (m + blockDim.y - 1) / blockDim.y);

  kernel_v1<BLOCK_SIZE><<<gridDim, blockDim>>>(d_a, d_b, d_c, m, n, k);
  cudaDeviceSynchronize();
}

template <int BLOCK_SIZE, int STRIDE>
__global__ void kernel_v2(const float *a, const float *b, float *c, int M, int N, int K) {
  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;

  constexpr int TILE_SIZE = BLOCK_SIZE * STRIDE;

  float *c_block = c + blockIdx.y * TILE_SIZE * N + blockIdx.x * TILE_SIZE;

  __shared__ float tiled_a[TILE_SIZE][TILE_SIZE];
  __shared__ float tiled_b[TILE_SIZE][TILE_SIZE];

  float accum_sum[STRIDE][STRIDE] = {.0f};

  for (int k = 0; k < K; k += TILE_SIZE) {
    for (int i = 0; i < STRIDE; ++i) {
      for (int j = 0; j < STRIDE; ++j) {
        int row = blockIdx.y * TILE_SIZE + tid_y * STRIDE + i;
        int col = k + tid_x * STRIDE + j;
        if (row < M && col < K) {
          tiled_a[tid_y * STRIDE + i][tid_x * STRIDE + j] = a[row * K + col];
        } else {
          tiled_a[tid_y * STRIDE + i][tid_x * STRIDE + j] = .0f;
        }

        row = k + tid_y * STRIDE + i;
        col = blockIdx.x * TILE_SIZE + tid_x * STRIDE + j;
        if (col < N && row < K) {
          tiled_b[tid_y * STRIDE + i][tid_x * STRIDE + j] = b[row * N + col];
        } else {
          tiled_b[tid_y * STRIDE + i][tid_x * STRIDE + j] = .0f;
        }
      }
    }
    __syncthreads();
    float register_a[STRIDE][TILE_SIZE];
    float register_b[STRIDE][TILE_SIZE];
    // cache shared memory to registrer memory
    for (int i = 0; i < STRIDE; ++i) {
      for (int s = 0; s < TILE_SIZE; ++s) {
        register_a[i][s] = tiled_a[tid_y * STRIDE + i][s];
        register_b[i][s] = tiled_b[s][tid_x * STRIDE + i];
      }
    }
    for (int i = 0; i < STRIDE; ++i) {
      for (int j = 0; j < STRIDE; ++j) {
        for (int s = 0; s < TILE_SIZE; ++s) {
          accum_sum[i][j] += register_a[i][s] * register_b[j][s];
        }
      }
    }
    __syncthreads();
  }
  for (int i = 0; i < STRIDE; ++i) {
    for (int j = 0; j < STRIDE; ++j) {
      c_block[(tid_y * STRIDE + i) * N + tid_x * STRIDE + j] = accum_sum[i][j];
    }
  }
}

void MatMulV2(float *d_a, float *d_b, float *d_c, int m, int n, int k) {
  constexpr int BLOCK_SIZE = 16;
  constexpr int STRIDE = 2;
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  constexpr int TILE_SIZE = STRIDE * BLOCK_SIZE;
  dim3 gridDim((n + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE);

  kernel_v2<BLOCK_SIZE, STRIDE><<<gridDim, blockDim>>>(d_a, d_b, d_c, m, n, k);
  cudaDeviceSynchronize();
}

template <int BLOCK_SIZE, int STRIDE>
__global__ void kernel_v3(const float *a, const float *b, float *c, int M, int N, int K) {
  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;

  constexpr int TILE_SIZE = BLOCK_SIZE * STRIDE;

  // Calculate the pointer to the current block in the output matrix
  float *c_tile = c + blockIdx.y * TILE_SIZE * N + blockIdx.x * TILE_SIZE;

  // Declare shared memory for tiles of input matrices
  __shared__ float smem_a_tile[TILE_SIZE][TILE_SIZE];
  __shared__ float smem_b_tile[TILE_SIZE][TILE_SIZE];

  // Initialize accumulation array for the result
  float accum_sum[STRIDE][STRIDE] = {0.0f};

  // Loop over the tiles of the input matrices
  for (int k = 0; k < K; k += TILE_SIZE) {
    // Step1: Copy the tiles data from input matrix to shared memory

    // Step1: Copy the tiles data from input matrix to shared memory
    float4 *a_shared_ptr = reinterpret_cast<float4 *>(&smem_a_tile[0][0]);
    float4 *b_shared_ptr = reinterpret_cast<float4 *>(&smem_b_tile[0][0]);

    int thread_idx_in_block = tid_y * BLOCK_SIZE + tid_x;

    // Calculate how many float4 loads each thread needs to perform
    constexpr int elements_per_thread = (TILE_SIZE * TILE_SIZE) / (BLOCK_SIZE * BLOCK_SIZE * 4);

    // Load data in multiple iterations
    for (int i = 0; i < elements_per_thread; i++) {
      int linear_idx = thread_idx_in_block + i * (BLOCK_SIZE * BLOCK_SIZE);
      int tile_row = linear_idx / (TILE_SIZE / 4);
      int tile_col = linear_idx % (TILE_SIZE / 4);

      if (tile_row < TILE_SIZE && tile_col < TILE_SIZE / 4) {
        // Load data into shared memory from global memory
        a_shared_ptr[tile_row * TILE_SIZE / 4 + tile_col] = reinterpret_cast<const float4 *>(
            a + (blockIdx.y * TILE_SIZE + tile_row) * K + k)[tile_col];
        b_shared_ptr[tile_row * TILE_SIZE / 4 + tile_col] = reinterpret_cast<const float4 *>(
            b + (k + tile_row) * N + blockIdx.x * TILE_SIZE)[tile_col];
      }
    }

    // Synchronize threads to ensure all data is loaded
    __syncthreads();

    // Step2: Perform the matrix multiplication using the outer product
    for (int s = 0; s < TILE_SIZE; ++s) {
      for (int i = 0; i < STRIDE; ++i) {
        for (int j = 0; j < STRIDE; ++j) {
          accum_sum[i][j] +=
              smem_a_tile[tid_y * STRIDE + i][s] * smem_b_tile[s][tid_x * STRIDE + j];
        }
      }
    }
    // Synchronize threads before the next iteration
    __syncthreads();
  }

  // Write the accumulated results to the output matrix
  for (int i = 0; i < STRIDE; ++i) {
    for (int j = 0; j < STRIDE; ++j) {
      c_tile[(tid_y * STRIDE + i) * N + tid_x * STRIDE + j] = accum_sum[i][j];
    }
  }
}

void MatMulV3(float *d_a, float *d_b, float *d_c, int m, int n, int k) {
  constexpr int BLOCK_SIZE = 16;
  constexpr int STRIDE = 2;
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  constexpr int TILE_SIZE = STRIDE * BLOCK_SIZE;
  dim3 gridDim((n + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE);

  kernel_v3<BLOCK_SIZE, STRIDE><<<gridDim, blockDim>>>(d_a, d_b, d_c, m, n, k);
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
  // test_matmul_func(MatMulCPUv1, "MatMulCPUv1", a.data(), b.data(), c_cpu.data(), m, n, k);
  // test_matmul_func(MatMulCPUv2, "MatMulCPUv2", a.data(), b.data(), c_cpu.data(), m, n, k);
  // test_matmul_func(MatMulCPUv3<128>, "MatMulCPUv3", a.data(), b.data(), c_cpu.data(), m, n, k);
  // test_matmul_func(matmulImplTiling<16>, "matmulImplTiling", a.data(), b.data(), c_cpu.data(), m,
  // n,
  //                  k);

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
  test_matmul_func(MatMulV2, "MatMulV2", d_a, d_b, d_c, m, n, k, c_cpu);
  test_matmul_func(MatMulV3, "MatMulV3", d_a, d_b, d_c, m, n, k, c_cpu);

  // 释放设备侧的存储
  checkCudaErrors(cudaFree(d_a));
  checkCudaErrors(cudaFree(d_b));
  checkCudaErrors(cudaFree(d_c));
  checkCudaErrors(cublasDestroy(blas_handle));

  return 0;
}