cmake_minimum_required(VERSION 3.20)
project(model-deployment C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES "75;80;86")
find_package(CUDAToolkit REQUIRED)

find_package(Threads REQUIRED)

# list(APPEND CMAKE_PREFIX_PATH $ENV{TORCH_ROOT})
execute_process(
    COMMAND python3 -c "import torch;print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE torch_cmake_prefix_path
    ERROR_VARIABLE error_output  # 获取错误输出
    RESULT_VARIABLE result       # 获取命令的返回码
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(${result} EQUAL 0)
    message(STATUS "PyTorch CMake Prefix Path: ${torch_cmake_prefix_path}")
    list(APPEND CMAKE_PREFIX_PATH ${torch_cmake_prefix_path})
else()
    message(FATAL_ERROR "Command failed with error: ${error_output}")
endif()
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(PythonInterp 3 REQUIRED)
find_package(PythonLibs 3 REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

add_subdirectory(cuda-snippets)
add_subdirectory(torch-ext)