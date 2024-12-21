add_library(torch_ext SHARED 
    ${CMAKE_CURRENT_SOURCE_DIR}/torch_ext.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/lltm_cpp.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/lltm_cuda.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lltm_cuda_kernel.cu
)

find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")
target_link_libraries(torch_ext ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} ${CMAKE_THREAD_LIBS_INIT} ) 

find_package(PythonInterp 3 REQUIRED) # 用于获取 PYTHON_VERSION_MAJOR, PYTHON_VERSION_MINOR

string(TOLOWER ${CMAKE_SYSTEM_NAME} system_name)
set(system_processor ${CMAKE_SYSTEM_PROCESSOR})

set_target_properties(torch_ext 
    PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}" 
    SUFFIX ".cpython-${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}-${system_processor}-${system_name}-gnu.so"
    OUTPUT_NAME "torch_ext")
