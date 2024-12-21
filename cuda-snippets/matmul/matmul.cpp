#include <torch/extension.h>

torch::Tensor matmul(torch::Tensor a, torch::Tensor b);

PYBIND11_MODULE(matmul, m) {
  m.def("matmul", &matmul, "matrix multiplication (CUDA)");
}