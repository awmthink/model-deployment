# enable_language(CUDA)
add_executable(tensor_test ${CMAKE_CURRENT_SOURCE_DIR}/main.cu)
target_link_libraries(tensor_test ${TORCH_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})