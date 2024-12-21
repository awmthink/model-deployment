from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

setup(
    name="torch_ext",
    ext_modules=[CUDAExtension("torch_ext", ["lltm_cuda.cpp", "lltm_cuda_kernel.cu"])],
    cmdclass={"build_ext": BuildExtension},
)
