# Pybind11 Tutorials

## 生成动态链接库

在当前示例中，我们希望生成的 Python 包的名称为："pybindx"，那么我们通过 `pybind11` 生成的 C++ 动态链接库的名称格式需要是：`pybindx.so`或者带上 Python的版本与操作系统信息，例如：`pybindx.so.cpython-310-x86_64-linux-gnu.so`，不可以是默认的 `libpybinx.so`。

除了上述规则外，我们在使用`pybind11`时，`PYBIND11_MODULE(pybindx, m)`中定义的 module name 必须保持一致。否则会报下面的错误：

```bash
----> 1 import pybindx

ImportError: dynamic module does not define module export function (PyInit_pybindx)
```

## 与 Numpy的互操作

在 `Pybind11`中提供了一个 `pybind11::array_t`的模板数据结构用于与 `numpy.ndarray`进行相互转换。


### `c_style`与`forcecast`

```cpp
py::array_t<float> process_array(py::array_t<float, py::array::c_style | py::array::forcecast> input_array) {
    // 处理数组...
}
```

`py::array::c_style` 用于确保传递的 NumPy 数组是 C 风格（row-major） 的内存布局。这意味着数组在内存中是按行存储的（行优先），即连续的内存元素首先按行排列，而不是按列排列。当你使用 `py::array::c_style` 时，`pybind11` 将确保输入的 NumPy 数组是 C 风格的。如果传入的数组不是 C 风格的，pybind11 会尝试生成一个 C 风格的副本，确保你的 C++ 代码可以以预期的方式访问数据。

`py::array::forcecast` 是一个强制转换标志，它确保传递给 C++ 函数的 NumPy 数组 具有匹配的类型和维度。如果传递的数组类型与 C++ 函数期望的类型不匹配，`py::array::forcecast` 会强制将数据转换为目标类型。通常情况下，如果类型不匹配（例如，C++ 代码期望 `double` 类型，但 Python 传递的是 `int` 类型的 NumPy 数组），`pybind11` 可能会抛出异常。然而，使用 `py::array::forcecast`，它会自动进行类型转换，而不是抛出错误。

### `array_t`的几个常用接口

|接口名|功能描述|
|:---:|---|
|`array_t<T> array`|默认构造函数|
|`array_t<T> array({rows, cols})`|构造一个指定大小的 Numpy 数组|
|`array_t<T> array({rows, cols}, data_pointer)`|从现有的数据构造一个 Numpy 数组，底层会发生数据拷贝|
|`index_at(i,j,k,...)`| 根据多维下标计算对应元素在整个地址中的偏移|
|`data()`|返回 `const T*`的指针|
|`mutable_data()`|返回可以修改的非常量指针`T*`|
|`at(i,j,k,...)`|返回对应下标的常量元素的引用，不可对其进行修改|
|`mutable_at(i,j,k,...)`|返回对应下标的元素的引用，可以对其进行修改|
|`unchecked<N>()`|返回一个代理类，通过代理类进行元素访问时，不会进行越界检查|
|`mutable_unchecked<N>()`|同`unchecked`一样，只是返回的代理类可以修改数据|