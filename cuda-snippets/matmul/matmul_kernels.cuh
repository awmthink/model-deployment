#pragma once
#include <cuda_runtime.h>

__host__ __device__ inline constexpr int cdiv(int x, int y) { return (x + y - 1) / y; }

template <typename T>
struct Range {
  int start;
  int end;

  __host__ __device__ Range(int start, int end) : start(start), end(end) {}
  __host__ __device__ int size() const { return end - start; }
};

template <typename T>
class MatrixView {
 private:
  T *data_;
  int rows_;
  int cols_;
  int stride_;  // 实际的列跨度，用于支持子矩阵视图

 public:
  // 构造函数
  __host__ __device__ MatrixView(T *data, int rows, int cols, int stride = -1)
      : data_(data), rows_(rows), cols_(cols), stride_(stride < 0 ? cols : stride) {}

  // 基本访问方法
  __host__ __device__ __forceinline__ T &operator()(int row, int col) {
    return data_[row * stride_ + col];
  }

  __host__ __device__ __forceinline__ const T &operator()(int row, int col) const {
    return data_[row * stride_ + col];
  }

  // 获取子矩阵视图
  __host__ __device__ MatrixView<T> submatrix(Range<int> rows, Range<int> cols) {
    T *new_data = &(*this)(rows.start, cols.start);
    return MatrixView<T>(new_data, rows.size(), cols.size(), stride_);
  }

  // 使用float4加载数据的辅助方法
  __host__ __device__ __forceinline__ float4 &as_float4(int row, int col4) {
    static_assert(std::is_same<T, float>::value, "as_float4 only works with float matrices");
    return *reinterpret_cast<float4 *>(&data_[row * stride_ + col4 * 4]);
  }

  // 基本属性访问
  __host__ __device__ __forceinline__ int rows() const { return rows_; }
  __host__ __device__ __forceinline__ int cols() const { return cols_; }
  __host__ __device__ __forceinline__ int stride() const { return stride_; }
  __host__ __device__ __forceinline__ T *data() { return data_; }
  __host__ __device__ __forceinline__ const T *data() const { return data_; }
};

__global__ void naive_kernel(const float *a, const float *b, float *c, const int M, const int N,
                             const int K);

template <int BLOCK_SIZE>
__global__ void tiled_smem_kernel(const float *a, const float *b, float *c, int M, int N, int K);

template <int BLOCK_SIZE, int STRIDE>
__global__ void strided_tiled_kernel(const float *a, const float *b, float *c, int M, int N, int K);

template <int BLOCK_SIZE, int STRIDE>
__global__ void float4_smem_coalesced_kernel(const float *a, const float *b, float *c, int M, int N,
                                             int K);