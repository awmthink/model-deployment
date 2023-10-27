list(APPEND CMAKE_PREFIX_PATH $ENV{TORCH_ROOT})
enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES "75;80;86")


find_package(Torch REQUIRED)
find_package(Threads REQUIRED)

find_package(PythonInterp 3 REQUIRED)
find_package(PythonLibs 3 REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

link_directories(${TORCH_INCLUDE_DIRS})

add_executable(cpp_front ${CMAKE_CURRENT_SOURCE_DIR}/cpp_front/main.cc)
target_link_libraries(cpp_front ${TORCH_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

add_library(cpp_ext_cpp SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/torch_ext/lltm_cpp.cpp)
target_link_libraries(cpp_ext_cpp ${TORCH_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

add_library(cpp_ext_cuda SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/torch_ext/lltm_cuda.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/torch_ext/lltm_cuda_kernel.cu)
target_link_libraries(cpp_ext_cuda ${TORCH_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

add_library(warp_perspective SHARED ${CMAKE_CURRENT_SOURCE_DIR}/torchscript_ext/ext_ops.cc)
target_link_libraries(warp_perspective ${TORCH_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

add_executable(myops_add ${CMAKE_CURRENT_SOURCE_DIR}/torchscript_ext/myops_add.cc)
target_link_libraries(myops_add ${TORCH_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(myops_add -Wl,--no-as-needed warp_perspective)

add_executable(custom_class ${CMAKE_CURRENT_SOURCE_DIR}/torchscript_ext/custom_class.cc)
target_link_libraries(custom_class ${TORCH_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(custom_class -Wl,--no-as-needed warp_perspective)

