from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

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