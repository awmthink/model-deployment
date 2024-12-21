# Pytorch的C++扩展

Pytorch中已经有大量的operations了，支持对Tensor数据进行丰富的代数运算。但有时候，我们还是有使用新的operator的需求，比如写新的激活函数，或为新的研究算法编写新的Layer。

最简单的扩展Pytorch算子的方式是，扩展 `pytorch` 的 `Function` 和 `Module`，它可以利用Pytorch的自动微分机制，使得我们可以只 `Forward` 过程。但有时候，我们为了进一步追求性能，或者我们的扩展 operator 会被非常频繁的调用，那么就可以考虑使用`C++/CUDA`来扩展实现了。还有一些情况是，我们需要实现的新operator需要调用一些第三方C++库，那么我们也需要使用C++来编写扩展。

在 Pytorch 中，我们可以经过继承 `torch.autograd.Function` 和 `torch.nn.Module` 来扩展自己的算子。但二者的使用场景不同：

1. `torch.autograd.Function` 一般有两类使用场景：1）不希望依赖自动微分机制，需要对`backward`过程有一些特别的处理；2）在前向与后向过程中，使用到了一些其他的不支持自动微分的库，比如用 pybind11 扩展的 C++实现的库。
2. `torch.nn.Module` 则主要用于定义神经网络中的块或层，Pytorch 自动管理它的梯度计算以及参数等。

让我们考虑性能优化方面：

Pytorch中的算子底层实际已经使用CPU和GPU进行了高度优化，借助了cuDNN、InterlMKL等库的支持。但如果我们看一些大的粒度，比如Layer这个层次，它里面往往是多个算子的组合，Pytorch层面必须逐个执行操作，启动CUDA内核。同时，Python解释器本身也有开销。

我们可以使用C++来进行多个算子的融合，把多个函数实现为一个函数，减少cuda kernel的启动次数，避免中间结果的写出与读入，提升访存利用率。

示例中的LLTM和LSTM区别：

1. 取消了LSTM中的forget gate
2. 使用了elu来替换了tanh

```python
# LSTM Cell
def __call__(self, x, hidden_state):
    # x.shape: (batch_size, input_dim)
    h, c = hidden_state
    i, f, g, o = np.split(self.weight_ih @ x + self.weight_hh @ h + self.bais_hh, 4)
    i, f, g, o = sigmoid(i), sigmoid(f), np.tanh(g), sigmoid(o)
    c_out = f * c + i * g
    h_out = o * np.tanh(c_out)
    return h_out, c_out

# LLTM Cell
def __call__(self, x, hidden_state):
    # x.shape: (batch_size, input_dim)
    h, c = hidden_state
    X = torch.cat([x, h], dim=1)
    i, g, o = np.split(self.weight @ X + self.bais, 3)
    i, g, o = sigmoid(i), np.elu(g), sigmoid(o)
    c_out = c + i * g
    h_out = o * np.tanh(c_out)
    return h_out, c_out
```

# 编写C++扩展

我们有三种方式来往pytorch中添加算子

1. 通过JIT的方法，也就是torch.utils.cpp_extension.load()
2. 通过 CMake 来显式的构建 pybind11 绑定后的 C++ 动态链接库
3. Ahead of time：通过setuptools


## CMake 的方式

和使用 `pybind11` 来其他的 C++ python包一样，我们将相关的 C++ 源文件编译为一个 `<module_name>.so` 的动态库，比如在该示例中，编译的库名为：`torch_ext.cpython-310-x86_64-linux-gnu.so`。

需要注意的是我们构建的动态链接库，因为使用到了 `torch::Tensor` 以及对应的 pybind11 的一些扩展，所以我们需要链接相应的库，除了链接 `TORCH_LIBRARIES`外，还需要手动指定 `libtorch_python.so`。

## setuptools的方式

首先我们需要定义`setup.py`脚本

```python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

setup(name="lltm", 
      ext_modules=[CUDAExtension("lltm", ["lltm_cuda.cpp", "lltm_cuda_kernel.cu"])],
      cmdclass={"build_ext": BuildExtension})
```

其中`CppExtension`和`CUDAExtension`等都是一个对`setuptools.Extension`的一个上层便利性的封装。它等价于：

```python
Extension(
   name='lltm',
   sources=["lltm_cuda.cpp", "lltm_cuda_kernel.cu"],
   include_dirs=cpp_extension.include_paths(),
   library_dirs=cpp_extension.library_paths(cuda=True)
   libraries=['c10', 'torch_cpu', 'torch', 'torch_python' 'cudart', 'torch_cuda'],
   language='c++')
```

`BuildExtension` 负责对整个编译过程进行配置（ninja）和编译，包括对CUDA代码进行混合编译。

我们需要扩展的函数实现，一般化的形式如下：

```cpp
#include <torch/extension.h>
#include <vector>

torch::Tensor add_forward(torch::Tensor lhs, torch::Tensor rhs) {
    return lhs + rhs;
}
std::vector<torch::Tensor> add_backward(torch::Tensor grad) {
    return {grad, grad}
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("add_forward", &add_forward, "Tensor add forward");
    m.def("add_backward", &add_backward, "Tensor add backward");
}
```

其中 `TORCH_EXTENSION_NAME` 是一个宏定义，当我们使用`setuptools` 来进行编译时，该宏会被 `BuildExtension` 定义为 `setup`中的`name`字段的值。

```python
def _define_torch_extension_name(self, extension):
        # pybind11 doesn't support dots in the names
        # so in order to support extensions in the packages
        # like torch._C, we take the last part of the string
        # as the library name
        names = extension.name.split('.')
        name = names[-1]
        define = f'-DTORCH_EXTENSION_NAME={name}'
        self._add_compile_flag(extension, define)
```

## JIT的方式

使用JIT的方式，我们在python的源代码中，直接`load`相关的C++扩展，这种方式应该是最方便的

```python
from torch.utils.cpp_extension import load

lltm = load(name="lltm", sources=["lltm.cpp"])
```

`load`的背后，实际的执行过程是：

1. 创建一个临时目录：$HOME/.cache/torch_extensions/py38_cu118/lltm
2. Emit 一个 Ninja 的构建文件
3. 把源码编译为一个 shard 库
4. 将这个动态库导入为一个python Module

当我们第一次`load`时，比较耗时，但如果我们对源码没有修改，那么后续加载就很快了。Ninja本身也是一种增量构建的过程。

如果我们的 setup 比较复杂，那么我们就需要使用 setuptools。

# 自定义 CUDA kernel

大部分情况，我们像上面使用 torch::Tensor 来编写扩展实现就可以了，因为torch::Tensor本身也支持使用GPU。

但为了追求进一步的性能提升，我们可以将上面的一些运算，全部写在一个cuda kernel里。我们一般的扩展编写策略是：

1. 编写一个C++文件，定义好暴露给Python侧的接口
2. 使用pybind11来绑定这些接口
3. 这个接口的实现会再次调用额外的cuda kernel函数
4. cuda kernel函数在cpp文件中前置声明，然后在一个.cu文件中实现
5. 由于cuda kernel一般要求连续的内存，所以我们需要在cpp层面加一些对输入数据的检查

```cpp

// -- add_ext.cpp

#include <torch/extension.h>
#include <vector>

// 前置声明
torch::Tensor add_forward_cuda(torch::Tensor lhs, torch::Tensor rhs);
std::vector<torch::Tensor> add_backward_cuda(torch::Tensor grad)

torch::Tensor add_forward(torch::Tensor lhs, torch::Tensor rhs) {
    CHECK_INPUT(lhs);
    CHECK_INPUT(rhs);
    return add_forward_cuda(lhs, rhs);
}
std::vector<torch::Tensor> add_backward(torch::Tensor grad) {
    CHECK_INPUT(grad);
    return add_backward_cuda(grad);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("add_forward", &add_forward, "Tensor add forward");
    m.def("add_backward", &add_backward, "Tensor add backward");
}

// -- add_ext_cuda.cu
// 注意在setuptools方式中，不要将cu文件与cpp文件使用同样的文件名

__global__ void add_forward_cuda_kernel() { ... }

__global__ void add_backward_cuda_kernel() { ... }

torch::Tensor add_forward_cuda(torch::Tensor lhs, torch::Tensor rhs) { ... }

std::vector<torch::Tensor> add_backward_cuda(torch::Tensor grad) { ... }

```

## AT_DISPATCH_FLOATING_TYPES

`AT_DISPATCH_FLOATING_TYPES`的目的是为我们处理类型分派。它接受一个类型（在我们的例子中是`gates.type()`）、一个名称（用于错误消息）和一个lambda函数。在这个lambda函数内部，类型别名`scalar_t`可用，并且在该上下文中定义为张量在运行时的实际类型。因此，如果我们有一个模板函数（这将是我们的CUDA内核），我们可以使用这个`scalar_t`别名进行实例化，并调用正确的函数。在这种情况下，我们还希望将张量的数据指针作为`scalar_t`类型的指针进行检索。如果您想要对所有类型进行分派，而不仅仅是浮点类型（Float和Double），可以使用`AT_DISPATCH_ALL_TYPES`。

## Using accessors

就一个`Tenosr`转换为一个在cuda kernel中可以以多维数组进行访问的形式：`torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits>`

其中`2`表示数组的维度，`torch::RestrictPtrTraits`表明了`__restrict__`。`PackedAccessor32`表示使用int32来表示size和stride。
