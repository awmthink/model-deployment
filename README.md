# Pytorch Model Deployment

## Torchscript

* [Tutorial](./torchscript.ipynb)

## Pytorch Model Export ONNX

* [Tutorial](./export_onnx.ipynb)


## Extend Torch with C++


- [x] [为PyTorch扩展C++的算子，并在nn.Module中调用，运行在Python语言中](./examples/torch_ext/README.md)
- [x] [为Pytorch编写C++的算子或自定类的类型，并在torchscript中调用，可以同时运行在Python与C++语言中](./examples/torchscript_ext/README.md)
- [x] [使用Libtorch的C++前端编写推理或训练的代码](./examples/cpp_front/main.cc)
- [ ] [TODO: 在C++前端中使用自动微分机制，以及自定义C++的Autograd函数](https://pytorch.org/tutorials/advanced/cpp_autograd.html)
- [ ] [TODO: 注册一个Dispatched的C++算子](https://pytorch.org/tutorials/advanced/dispatcher.html)
- [ ] [TODO: 在C++中为Pytorch Dispatcher扩展一个新的Backend](https://pytorch.org/tutorials/advanced/extend_dispatcher.html)


## TensorRT