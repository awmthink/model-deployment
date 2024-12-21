add_library(warp_perspective SHARED ${CMAKE_CURRENT_SOURCE_DIR}/ext_ops.cc)
target_link_libraries(warp_perspective ${TORCH_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

add_executable(myops_add ${CMAKE_CURRENT_SOURCE_DIR}/myops_add.cc)
target_link_libraries(myops_add ${TORCH_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(myops_add -Wl,--no-as-needed warp_perspective)

add_executable(custom_class ${CMAKE_CURRENT_SOURCE_DIR}/custom_class.cc)
target_link_libraries(custom_class ${TORCH_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(custom_class -Wl,--no-as-needed warp_perspective)