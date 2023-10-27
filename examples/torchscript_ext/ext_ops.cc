#include <torch/custom_class.h>
#include <torch/torch.h>

#include <string>

torch::Tensor warp_perspective(torch::Tensor image, torch::Tensor warp) {
  return image;
}

torch::Tensor add(torch::Tensor lhs, torch::Tensor rhs) { return lhs + rhs; }

// 所有扩展的自定义Class都必须继承自CustomClassHolder，它里面有引用计数
template <typename T>
struct MyStackClass : torch::CustomClassHolder {
  std::vector<T> stack_;
  MyStackClass(std::vector<T> init) : stack_(init.begin(), init.end()) {}

  void push(T x) { stack_.push_back(x); }

  T pop() {
    if (stack_.empty()) {
      throw std::runtime_error("empty stack");
    }
    auto val = stack_.back();
    stack_.pop_back();
    return val;
  }

  // Python层面看到的都是
  // c10::intrusive_ptr<MyStackClass>对象，类似于一个shared_ptr
  c10::intrusive_ptr<MyStackClass> clone() const {
    return c10::make_intrusive<MyStackClass>(stack_);
  }

  void merge(const c10::intrusive_ptr<MyStackClass>& c) {
    stack_.insert(stack_.end(), c->stack_.begin(), c->stack_.end());
  }
};

c10::intrusive_ptr<MyStackClass<std::string>> manipulate_instance(
    const c10::intrusive_ptr<MyStackClass<std::string>>& instance) {
  instance->pop();
  return instance;
}

TORCH_LIBRARY(myops, m) {
  m.def("warp_perspective", warp_perspective);
  m.def("add", add);

  m.class_<MyStackClass<std::string>>("MyStackClass")
      .def(torch::init<std::vector<std::string>>())
      .def("top",
           [](const c10::intrusive_ptr<MyStackClass<std::string>>& self) {
             return self->stack_.back();
           })
      .def("push", &MyStackClass<std::string>::push)
      .def("pop", &MyStackClass<std::string>::pop)
      .def("clone", &MyStackClass<std::string>::clone)
      .def("merge", &MyStackClass<std::string>::merge)

      // class_<>::def_pickle allows you to define the serialization
      // and deserialization methods for your C++ class.
      // Currently, we only support passing stateless lambda functions
      // as arguments to def_pickle
      .def_pickle(
          // __getstate__
          // This function defines what data structure should be produced
          // when we serialize an instance of this class. The function
          // must take a single `self` argument, which is an intrusive_ptr
          // to the instance of the object. The function can return
          // any type that is supported as a return value of the TorchScript
          // custom operator API. In this instance, we've chosen to return
          // a std::vector<std::string> as the salient data to preserve
          // from the class.
          [](const c10::intrusive_ptr<MyStackClass<std::string>>& self)
              -> std::vector<std::string> { return self->stack_; },
          // __setstate__
          // This function defines how to create a new instance of the C++
          // class when we are deserializing. The function must take a
          // single argument of the same type as the return value of
          // `__getstate__`. The function must return an intrusive_ptr
          // to a new instance of the C++ class, initialized however
          // you would like given the serialized state.
          [](std::vector<std::string> state)
              -> c10::intrusive_ptr<MyStackClass<std::string>> {
            // A convenient way to instantiate an object and get an
            // intrusive_ptr to it is via `make_intrusive`. We use
            // that here to allocate an instance of MyStackClass<std::string>
            // and call the single-argument std::vector<std::string>
            // constructor with the serialized state.
            return c10::make_intrusive<MyStackClass<std::string>>(
                std::move(state));
          });
  m.def("manipulate_instance", manipulate_instance);
}
