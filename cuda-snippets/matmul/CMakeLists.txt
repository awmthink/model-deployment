add_library(matmul SHARED ${CMAKE_CURRENT_SOURCE_DIR}/matmul.cu ${CMAKE_CURRENT_SOURCE_DIR}/matmul.cpp)


find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")

target_include_directories(matmul PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(matmul PRIVATE ${TORCH_LIBRARIES}  ${TORCH_PYTHON_LIBRARY}  ${CMAKE_THREAD_LIBS_INIT})

string(TOLOWER ${CMAKE_SYSTEM_NAME} system_name)
set(system_processor ${CMAKE_SYSTEM_PROCESSOR})

set_target_properties(matmul 
    PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}" 
    SUFFIX ".cpython-${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}-${system_processor}-${system_name}-gnu.so"
    OUTPUT_NAME "matmul")