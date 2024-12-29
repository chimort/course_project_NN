cmake_minimum_required(VERSION 3.10)
project(CourseProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_CLANG_TIDY "clang-tidy")

include_directories(external/eigen)
include_directories(external/EigenRand)
include_directories(${PROJECT_SOURCE_DIR}/utils)

add_subdirectory(LossFunctions)

add_executable(CourseProject main.cpp)

target_link_libraries(CourseProject PRIVATE LossFunctions)
target_include_directories(CourseProject PRIVATE LossFunctions)
