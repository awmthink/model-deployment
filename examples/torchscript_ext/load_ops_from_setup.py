import torch


torch.ops.load_library("build/lib.linux-x86_64-cpython-38/warp_perspective.so")
torch.classes.load_library("build/lib.linux-x86_64-cpython-38/warp_perspective.so")

@torch.jit.script
def compute(x, y):
    return torch.ops.myops.add(x, y)

print(compute.graph)
compute.save("compute_scripted.pt")

print(torch.classes.loaded_libraries)

s = torch.classes.myops.MyStackClass(["foo", "bar"])
s.push("pushed")
assert s.pop() == "pushed"

s2 = s.clone()
s.merge(s2)
for expected in ["bar", "foo", "bar", "foo"]:
    assert s.pop() == expected

try:
    s.pop()
except RuntimeError:
    print("can't pop a empty stack")
except :
    print("unknow exception")


# 下面代码为了展示，如何保存一个nn.Module，它的属性中，有一个是自定义的class
# 也就是说当我们序列化Foo时，也需要序列化MyStackClass
# 这就要求我们的MyStackClass提供def_pickle的实现
class Foo(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = torch.classes.myops.MyStackClass(["hi", "mom"])
    
    def forward(self, s:str) -> str:
        return self.stack.pop() + s
    
scripted_foo = torch.jit.script(Foo())
print(scripted_foo.graph)
scripted_foo.save("foo.pt")
    
loaded = torch.jit.load('foo.pt')
print(loaded.stack.pop())


# 下面演示的是如何命名用自定义的Operator，这个Operator本身的接口用到了自定义的MyStackClass
class TryCustomOp(torch.nn.Module):
    def __init__(self):
        super(TryCustomOp, self).__init__()
        self.f = torch.classes.myops.MyStackClass(["foo", "bar"])

    def forward(self):
        return torch.ops.myops.manipulate_instance(self.f)
    
custom_op_use_custom_class = TryCustomOp()
my_stack = custom_op_use_custom_class()
print(my_stack.pop()) # foo
