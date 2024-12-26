#include "matmul_kernels.cuh"

__global__ void naive_kernel(const float *a, const float *b, float *c, const int M, const int N,
                             const int K) {
  // Calculate global thread indices
  int gid_x = blockIdx.x * blockDim.x + threadIdx.x;
  int gid_y = blockIdx.y * blockDim.y + threadIdx.y;

  // Early exit if thread is outside matrix bounds
  if (gid_x >= N || gid_y >= M) {
    return;
  }

  // Accumulate dot product
  float sum = 0.0f;
  for (int i = 0; i < K; ++i) {
    sum += a[gid_y * K + i] * b[i * N + gid_x];
  }
  c[gid_y * N + gid_x] = sum;
}

template <int BLOCK_SIZE>
__global__ void tiled_smem_kernel(const float *a, const float *b, float *c, int M, int N, int K) {
  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;
  int gid_x = blockIdx.x * blockDim.x + threadIdx.x;
  int gid_y = blockIdx.y * blockDim.y + threadIdx.y;

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

// 每个 thread 处理 STRIDE * STRIDE 个元素
template <int BLOCK_SIZE, int STRIDE>
__global__ void strided_tiled_kernel(const float *a, const float *b, float *c, int M, int N,
                                     int K) {
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
    for (int i = 0; i < STRIDE; ++i) {
      for (int j = 0; j < STRIDE; ++j) {
        for (int s = 0; s < TILE_SIZE; ++s) {
          accum_sum[i][j] += tiled_a[tid_y * STRIDE + i][s] * tiled_b[s][tid_x * STRIDE + j];
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

template <int BLOCK_SIZE, int STRIDE>
__global__ void float4_smem_coalesced_kernel(const float *a, const float *b, float *c, int M, int N,
                                             int K) {
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

// Explicit instantiations for tiled_smem_kernel
template __global__ void tiled_smem_kernel<16>(const float *a, const float *b, float *c, int M,
                                               int N, int K);
template __global__ void tiled_smem_kernel<32>(const float *a, const float *b, float *c, int M,
                                               int N, int K);

// Explicit instantiations for strided_tiled_kernel
template __global__ void strided_tiled_kernel<16, 2>(const float *a, const float *b, float *c,
                                                     int M, int N, int K);
template __global__ void strided_tiled_kernel<8, 4>(const float *a, const float *b, float *c, int M,
                                                    int N, int K);

// Explicit instantiations for float4_smem_coalesced_kernel
template __global__ void float4_smem_coalesced_kernel<16, 2>(const float *a, const float *b,
                                                             float *c, int M, int N, int K);
template __global__ void float4_smem_coalesced_kernel<8, 4>(const float *a, const float *b,
                                                            float *c, int M, int N, int K);
