cmake_minimum_required(VERSION 3.10)
project(CourseProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_CLANG_TIDY "clang-tidy")

file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE HEADERS "src/*.h")

include_directories(external/eigen)
include_directories(external/EigenRand)

add_executable(CourseProject main.cpp ${SOURCES})

target_include_directories(CourseProject PRIVATE src)
target_include_directories(CourseProject PRIVATE external/eigen)
target_include_directories(CourseProject PRIVATE external/EigenRand)

macro(add_subdirectories_recursive dir)
    file(GLOB SUBDIRS LIST_DIRECTORIES true ${dir}/*)
    foreach(subdir ${SUBDIRS})
        if(IS_DIRECTORY ${subdir})
            target_include_directories(CourseProject PRIVATE ${subdir})
            add_subdirectories_recursive(${subdir})
        endif()
    endforeach()
endmacro()

add_subdirectories_recursive(${CMAKE_SOURCE_DIR}/src)