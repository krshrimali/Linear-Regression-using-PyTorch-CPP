cmake_minimum_required(VERSION 3.0)

project(lr-example)

find_package(Torch REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

link_directories(${CMAKE_BINARY_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

add_executable(lr-example src/regression.cpp)

target_link_libraries(lr-example "${TORCH_LIBRARIES}")

set_property(TARGET lr-example PROPERTY CXX_STANDARD 11)
