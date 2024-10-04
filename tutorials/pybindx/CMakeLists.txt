add_library(pybindx SHARED ${CMAKE_CURRENT_SOURCE_DIR}/pybindx.cpp)

find_package(PythonInterp 3 REQUIRED) # 用于获取 PYTHON_VERSION_MAJOR, PYTHON_VERSION_MINOR

string(TOLOWER ${CMAKE_SYSTEM_NAME} system_name)
set(system_processor ${CMAKE_SYSTEM_PROCESSOR})

set_target_properties(pybindx 
    PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}" 
    SUFFIX ".cpython-${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}-${system_processor}-${system_name}-gnu.so"
    OUTPUT_NAME "pybindx")
