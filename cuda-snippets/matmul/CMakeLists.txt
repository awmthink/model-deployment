add_library(matmul SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul_kernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul_pybind.cu
)


find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")

target_include_directories(matmul PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(matmul PRIVATE ${TORCH_LIBRARIES}  ${TORCH_PYTHON_LIBRARY}  ${CMAKE_THREAD_LIBS_INIT})

string(TOLOWER ${CMAKE_SYSTEM_NAME} system_name)
set(system_processor ${CMAKE_SYSTEM_PROCESSOR})

set_target_properties(matmul 
    PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}" 
    SUFFIX ".cpython-${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}-${system_processor}-${system_name}-gnu.so"
    OUTPUT_NAME "matmul")

add_executable(matmul_kernel_test 
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul_kernel_test.cu
)
target_include_directories(matmul_kernel_test PRIVATE ${CMAKE_SOURCE_DIR}/cuda-snippets)
target_link_libraries(matmul_kernel_test PRIVATE CUDA::cudart CUDA::cublas)