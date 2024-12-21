#include <torch/extension.h>

torch::Tensor Softmax(torch::Tensor logits) {
  auto max_logits = torch::amax(logits, -1, true);
  auto stable_logits = logits - max_logits;

  auto exp_logits = torch::exp(stable_logits);
  auto sum_exp_logits = torch::sum(exp_logits, -1, true);
  return exp_logits / sum_exp_logits;
}

PYBIND11_MODULE(torch_ext, m) {
  using namespace pybind11::literals;
  namespace py = pybind11;
  m.def("softmax", &Softmax, "logits"_a);
}