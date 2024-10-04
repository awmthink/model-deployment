add_executable(cpp_front ${CMAKE_CURRENT_SOURCE_DIR}/main.cc)
target_link_libraries(cpp_front ${TORCH_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})