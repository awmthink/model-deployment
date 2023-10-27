import torch

import torch.utils.cpp_extension

torch.utils.cpp_extension.load(
    name="warp_perspective",
    sources=["ext_ops.cc"],
    is_python_module=False,
    verbose=True
)

print(torch.ops.myops.add)
print(torch.ops.myops.warp_perspective)
t1 = torch.randn(3,3)
t2 = torch.randn(3,3)
print(t1, t2, torch.ops.myops.add(t1, t2))


def compute(x, y):
    return torch.ops.myops.add(x, y)

traced_mod = torch.jit.trace(compute, (t1, t2))
print(traced_mod(t1, t2))