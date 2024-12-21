#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

std::string Hello(const std::string& greeting) { return "hello, " + greeting; }

// 测试 C++中的一些基本类型与 Python 中基本类型的对应关系
std::tuple<int, float, double, std::string, bool> GetBasicTypes() {
  int i = 42;               // int
  float f = 3.14f;          // float
  double d = 1.0;           // float
  std::string s = "hello";  // str
  bool b = true;            // bool
  return std::tuple<int, float, double, std::string, bool>{i, f, d, s, b};
}

// 返回一个 std::vector，在 python 层面会返回一个 list
std::vector<int> RangeList(int from, int to) {
  std::vector<int> vec(to - from);
  std::iota(vec.begin(), vec.end(), from);
  return vec;
}

// 返回 pybind11::bytes 对象，在 python中拿到的是 bytes 类型
pybind11::bytes ReadFileBytes(const char* file) {
  std::ifstream f(file, std::ifstream::binary);
  if (!f) {
    throw std::ios_base::failure("Failed to open the file: " +
                                 std::string(file));
  }
  f.seekg(0, f.end);
  std::size_t size = f.tellg();
  f.seekg(0, f.beg);

  char* buffer = new char[size];
  f.read(buffer, size);
  f.close();

  pybind11::bytes contents(buffer, size);
  delete[] buffer;

  return contents;
}

// 使用 buffer_info 来访问 arr 的一些属性
// 相较于直接访问 array_t 的属性，buffer_info 更加底层一些
void PrintArrayInfoBuffer(pybind11::array_t<float> arr) {
  pybind11::buffer_info buf = arr.request();  // 请求缓冲区信息
  std::cout << "Number of dimensions: " << buf.ndim << std::endl;
  std::cout << "Shape: ";
  for (ssize_t i = 0; i < buf.ndim; i++) {
    std::cout << buf.shape[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "ItemSize: " << buf.itemsize << std::endl;
}

pybind11::array_t<float> Transpose2D(pybind11::array_t<float> arr) {
  if (arr.ndim() != 2) {
    throw std::runtime_error("Input array must be 2 dimension");
  }
  int rows = arr.shape(0);
  int cols = arr.shape(1);
  pybind11::array_t<float> transposed({cols, rows});
  for (int i = 0; i < cols; ++i) {
    for (int j = 0; j < rows; ++j) {
      transposed.mutable_at(i, j) = arr.index_at(j, i);
    }
  }
  return transposed;
}

// ArraySquare 主要用于演示 index_at 接口的用法
void ArraySquare(pybind11::array_t<float> arr) {
  if (arr.ndim() != 2) {
    throw std::runtime_error("Input array must be 2 dimension");
  }
  int rows = arr.shape(0);
  int cols = arr.shape(1);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      auto idx = arr.index_at(i, j);
      arr.mutable_data()[idx] = arr.data()[idx] * arr.data()[idx];
    }
  }
}

class Matrix {
 public:
  Matrix() = default;
  Matrix(int rows, int cols)
      : data(new float[rows * cols]),
        size(rows * cols),
        nrows(rows),
        ncols(cols) {}

  ~Matrix() { delete[] data; }

  float& At(int i, int j) {
    int index = i * ncols + j;
    return data[index];
  }

  const float& At(int i, int j) const {
    int index = i * ncols + j;
    return data[index];
  }

  int nrows = 0;
  int ncols = 0;

 private:
  float* data = nullptr;
  std::size_t size = 0;
};

PYBIND11_MODULE(pybindx, m) {
  m.doc() = "pybind11 examples";

  using namespace pybind11::literals;
  m.def("hello", &Hello, "greeting"_a);
  m.def("range_list", &RangeList, "from"_a, "to"_a);
  m.def("get_basic_types", &GetBasicTypes);
  m.def("read_file_bytes", &ReadFileBytes, "file"_a);
  m.def("print_array_info_buffer", &PrintArrayInfoBuffer, "array"_a);
  m.def("transpose2d", &Transpose2D, "matrix"_a);
  m.def("array_square", &ArraySquare);

  pybind11::class_<Matrix, std::shared_ptr<Matrix>>(m, "Matrix")
      .def(pybind11::init<>())
      .def(pybind11::init<int, int>(), "rows"_a, "cols"_a)
      .def("__getitem__",
           [](const Matrix& m, std::pair<int, int> index) -> float {
             return m.At(index.first, index.second);
           })
      .def("__setitem__",
           [](Matrix& m, std::pair<int, int> index, float value) -> float {
             return m.At(index.first, index.second) = value;
           })
      .def_readwrite("nrows", &Matrix::nrows)
      .def_readwrite("ncols", &Matrix::ncols);
}