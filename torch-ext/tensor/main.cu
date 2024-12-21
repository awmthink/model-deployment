#include <torch/torch.h>

__global__ void print_packed_tensor_value(
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits>
        packed_tensor) {
  printf("[cuda PackedTensorAccessor] Tensor[1][2]: %f\n", packed_tensor[1][2]);
}

void tensor_accessor_test() {
  auto t = torch::rand({2, 3});
  std::cout << "Random Tensor: " << std::endl << t << std::endl;

  // 1. 直接使用下标索引
  float value = t[0][1].item<float>();
  std::cout << "Tensor[0][1]: " << value << std::endl;

  // 2. 使用指针来访问数据
  const float *const_data_ptr = t.const_data_ptr<float>();
  int i = 1;
  int j = 2;
  int index = i * t.stride(0) + j * t.stride(1);
  std::cout << "[data pointer] Tensor[1][2]: " << const_data_ptr[index]
            << std::endl;
  float *mutable_data_ptr = t.mutable_data_ptr<float>();
  mutable_data_ptr[index] = 3.14f;
  std::cout << "After set value at [1][2]: " << std::endl << t << std::endl;

  // 3. 使用 accessor。 accessor
  // 提供了一种高效访问多维张量元素的方式，尤其是当你希望避免较多的动态检查时。
  // accessor 方法只能用于 CPU 上的张量。
  auto accessor = t.accessor<float, 2>();
  value = accessor[1][2];
  std::cout << "[accessor] Tensor[1][2]: " << value << std::endl;

  // 使用 packed_accessor32 来获取一个 CUDA 平台下可以访问的
  // GenericPackedTensorAccessor 数据结构，它的成员函数都是 __device__ 的
  // 所以一般我们是在 kernel 函数中来使用该数据结构
  auto opt = torch::TensorOptions{}.device(torch::kCUDA).dtype(torch::kF32);
  auto dtensor = torch::rand({2, 3}, opt);
  std::cout << "Device Random Tensor: " << std::endl << dtensor << std::endl;
  auto dtensor_accessor =
      dtensor.packed_accessor32<float, 2, torch::RestrictPtrTraits>();
  print_packed_tensor_value<<<1, 1>>>(dtensor_accessor);
}

void tensor_slicing_test() {
  auto t = torch::rand({5, 5});
  std::cout << "Original tensor Size: " << t.sizes() << std::endl;  // [5, 5]
  auto sub_tensor =
      t.index({torch::indexing::Slice(0, 5, 2), torch::indexing::Slice(2, 3)});
  std::cout << "Sliced subtensor size: " << sub_tensor.sizes()
            << std::endl;  // [3, 1]
  auto sub_tensor1 = t.slice(0, 0, 5, 2).slice(1, 2, 3);
  std::cout << "Allcose subtensor and subtensor1: "
            << torch::allclose(sub_tensor, sub_tensor1) << std::endl;  // true

  // slice 的 tensor 和原 tensor 共享底层的数据
  sub_tensor[0][0] = 10.0f;
  std::cout << "Original tensor[0][0]: " << t[0][0].item<float>() << std::endl;

  // 如果不想共享数据，则可以使用 clone
  auto sub_tensor_clone = t.slice(0, 0, 5, 2).slice(1, 2, 3).clone();
  sub_tensor_clone[0][0] = 20.0f;
  std::cout << "Original tensor[0][0]: " << t[0][0].item<float>() << std::endl;
}

int main() {
  tensor_accessor_test();
  tensor_slicing_test();
}