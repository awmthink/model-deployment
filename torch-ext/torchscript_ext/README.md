# 为TorchScript编写C++/CUDA扩展

PyTorch 1.0发布引入了一种名为TorchScript的新的编程模型。TorchScript是Python编程语言的一个子集，可以被TorchScript编译器解析、编译和优化。此外，编译后的TorchScript模型可以选择序列化为磁盘文件格式，然后可以从纯C++（以及Python）中加载和运行用于推理。

TorchScript支持torch包提供的大部分操作，允许您纯粹地将许多种复杂模型表达为一系列来自PyTorch的张量操作。然而，有时您可能需要使用自定义的C++或CUDA函数扩展TorchScript。虽然我们建议只在无法将想法（以足够高效的方式）表达为简单的Python函数时才采用这个选项，但我们提供了一个非常友好和简单的接口，使用ATen、PyTorch的高性能C++张量库，定义自定义的C++和CUDA内核。一旦绑定到TorchScript中，您可以将这些自定义内核（或“ops”）嵌入到TorchScript模型中，并在Python中和在它们的序列化形式直接在C++中执行它们。


```cpp
#include <torch/torch.h>

#include <opencv2/opencv.hpp>

torch::Tensor warp_perspective(torch::Tensor image, torch::Tensor warp) {
  cv::Mat image_mat(image.size(0), image.size(1), CV_32FC1,
                    image.data_ptr<float>());
  cv::Mat warp_mat(warp.size(0), warp.size(1), CV_32FC1,
                   warp.data_ptr<float>());
  cv::Mat output_mat;
  cv::warpPerspective(image_mat, output_mat, warp_mat, {8, 8});
  torch::Tensor output = torch::from_blob(output_mat.ptr<float>(), {8, 8});
  return output.clone();
}

TORCH_LIBRARY(myops, m) {
  m.def("warp_perspective", warp_perspective);
}
```

`TORCH_LIBRARY`宏创建一个在程序启动时将被调用的函数。my_ops作为第一个参数给出（不应该加引号）。第二个参数（m）定义了一个`torch::Library`类型的变量，它是注册操作符的主要接口。Library::def方法实际上创建了一个名为`warp_perspective`的操作符，将其暴露给`Python`和`TorchScript`。您可以通过多次调用def来定义任意数量的操作符。我们把宏展开为：

```cpp
static void TORCH_LIBRARY_init_myops(torch::Library&);
static const torch::detail::TorchLibraryInit TORCH_LIBRARY_static_init_myops(
    torch::Library::DEF, &TORCH_LIBRARY_init_myops, "myops", c10::nullopt,
    file_name, line_no);
void TORCH_LIBRARY_init_myops(torch::Library& m) {
  m.def("warp_perspective", warp_perspective);
}
```

在背后，def函数实际上做了很多工作：它使用模板元编程来检查函数的类型签名，并将其转换为操作符Schema，该Schema在TorchScript的类型系统中指定了操作符的类型。

# 构建方法

## 使用CMake构建

像编译正常的依赖于torch的动态库一样：

```cmake
add_library(warp_perspective SHARED ${CMAKE_CURRENT_SOURCE_DIR}/torchscript_ext/ext_ops.cc)
target_link_libraries(warp_perspective ${TORCH_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
```

但使用这种方法，当前测试时，在python环境中，加载动态库时，出现了CXXABI的兼容问题。反而是使用C++加载scripted module时，链接这个库是没问题的。

## 使用setuptools来构建

和之前用setuptools编写torch的C++扩展没有区别

```python
setup(
    name="warp_perspective",
    ext_modules=[
        CppExtension(
            "warp_perspective",
            ["ext_ops.cc"]
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
)
```

## 使用JIT构建

```cpp
import torch.utils.cpp_extension

torch.utils.cpp_extension.load(
    name="warp_perspective",
    sources=["ext_ops.cc"],
    is_python_module=False,
    verbose=True
)

print(torch.ops.myops.add)
print(torch.ops.myops.warp_perspective)
```

# 在Python中的使用

```python
import torch
torch.ops.load_library("libwarp_perspective.so")
print(torch.ops.my_ops.warp_perspective(torch.randn(32, 32), torch.rand(3, 3)))
```

在幕后发生的是，当您首次在Python中访问`torch.ops.my_ops.warp_perspective`时，TorchScript编译器（在C++环境中）会检查是否已注册了函数`my_ops::warp_perspective`，如果是，则返回一个Python句柄来调用我们的C++操作符实现。这是TorchScript自定义操作符与C++扩展之间的一个显著区别：C++扩展是通过pybind11手动绑定的，而TorchScript自定义操作符则由PyTorch自己根据需要动态绑定。pybind11在绑定到Python时对于哪些类型和类可以绑定提供了更大的灵活性，因此推荐用于纯粹的eager代码，但它不支持TorchScript操作符。事实上，“标准库”函数如torch.matmul在TorchScript中的注册路径与自定义运算符基本相同。


# 将新的op运用在script module中

```python
import torch


torch.ops.load_library("build/lib.linux-x86_64-cpython-38/warp_perspective.so")

@torch.jit.script
def compute(x, y):
    return torch.ops.myops.add(x, y)

print(compute.graph)
compute.save("compute_scripted.pt")
```



# 在C++中调用scripted Module

我们写了一个myops_add.cc的C++调用程序，加载保存好的scripted module，但这里特别要注意的就是，该程序的链接需要显式的链接到`warp_perspective`这个动态库上。

上述示例中嵌入了一个关键细节：warp_perspective链接行之前的“-Wl,--no-as-needed”前缀。这是必需的，因为我们实际上不会在应用程序代码中调用warp_perspective共享库中的任何函数。我们只需要`TORCH_LIBRARY`函数来运行。然而，这会让链接器产生困惑，并认为它可以完全跳过与库的链接。在Linux上，`-Wl,--no-as-needed`标志强制进行链接（注意：该标志仅适用于Linux！）。此问题还有其他解决方法。最简单的方法是在所需从主应用程序中调用的运算符库中定义一些函数。这可以是一个简单的函数，比如在某个头文件中声明为`void init()`;，然后在运算符库中定义为`void init() { }`。在主应用程序中调用此init()函数将使链接器认为这是一个值得链接的库。不幸的是，这超出了我们的控制范围，我们宁愿让您知道这个原因以及简单的解决方法，而不是给您一些不透明的宏来插入您的代码中。