
#include <torch/script.h>

#include <iostream>
#include <vector>

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <script module path>" << std::endl;
    return -1;
  }
  torch::jit::script::Module module = torch::jit::load(argv[1]);
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(torch::randn({3, 5}));
  inputs.push_back(torch::randn({3, 5}));

  torch::Tensor output = module.forward(std::move(inputs)).toTensor();

  std::cout << output << std::endl;

  return 0;
}