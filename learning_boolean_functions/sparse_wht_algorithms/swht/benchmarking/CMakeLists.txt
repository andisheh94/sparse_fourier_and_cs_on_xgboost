#=======================
# Benchmarking building
#=======================

# cmake minimum version
cmake_minimum_required(VERSION 3.13.4)

# Utilities
set(BENCH_UTILS
    benchmark_utils.h
    "${CMAKE_BINARY_DIR}/include/build_info.h"
)

# Require Python 3
find_package(Python3 COMPONENTS Development REQUIRED)
add_library(python_utils OBJECT python_utils.cpp python_utils.h)
target_include_directories(python_utils PUBLIC ${Python3_INCLUDE_DIRS})
target_link_libraries(python_utils PUBLIC ${Python3_LIBRARIES})

# Define raw benchamrking target
add_executable(raw_timing raw_global_timing.cpp ${BENCH_UTILS})
target_include_directories(raw_timing PRIVATE "${SRC_DIR}" "${CMAKE_BINARY_DIR}/include")
target_link_libraries(raw_timing PRIVATE python_utils swht_kernel RandomSignal)

# Define python benchamrking target
add_executable(py_timing python_global_timing.cpp ${BENCH_UTILS})
target_include_directories(py_timing PRIVATE "${INCLUDE_SRC_DIR}" "${SRC_DIR}/utils" "${CMAKE_BINARY_DIR}/include")
target_link_libraries(py_timing PRIVATE python_utils RandomSignal)
