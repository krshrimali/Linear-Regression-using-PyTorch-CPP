 cmake_minimum_required(VERSION 3.0)

 project(lr-example)

 # list(APPEND CMAKE_PREFIX_PATH "/home/Users/krshrimali/Documents/libtorch/")
 # set(CMAKE_MODULE_PATH "/Users/krshrimali/Documents/libtorch/share/cmake/Torch/")

 find_package(Torch REQUIRED)
 find_package(OpenCV 4.1.0 REQUIRED)

 include_directories(${OpenCV_INCLUDE_DIRS})
 include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

 link_directories(${CMAKE_BINARY_DIR}/bin)
 set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

 add_executable(lr-example src/main.cpp src/regression.cpp)

 target_link_libraries(lr-example ${OpenCV_LIBS})
 target_link_libraries(lr-example "${TORCH_LIBRARIES}")

 set_property(TARGET lr-example PROPERTY CXX_STANDARD 11)
