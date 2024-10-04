add_library(cpp_ext_cpp SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/lltm_cpp.cpp)
target_link_libraries(cpp_ext_cpp ${TORCH_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

add_library(cpp_ext_cuda SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/lltm_cuda.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lltm_cuda_kernel.cu)
target_link_libraries(cpp_ext_cuda ${TORCH_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})