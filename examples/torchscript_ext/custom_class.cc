
#include <torch/script.h>

#include <iostream>
#include <vector>

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <script module path>" << std::endl;
    return -1;
  }
  torch::jit::script::Module module;
  try {
    module = torch::jit::load(argv[1]);
  } catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  std::vector<torch::jit::IValue> inputs{"foobarbaz"};

  auto output = module.forward(inputs).toString();
  std::cout << output->string() << std::endl;

  return 0;
}