cmake_minimum_required(VERSION 3.10)
project(CourseProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_CLANG_TIDY "clang-tidy")

set(EXTERNAL_LIB_DIRS
    external/eigen
    external/EigenRand
    ${PROJECT_SOURCE_DIR}/utils
)

include_directories(${EXTERNAL_LIB_DIRS})

set(SUBDIRECTORIES
    LossFunctions
    ActivationFunctions
    #NetCore
    #Optimizers
)

add_executable(CourseProject main.cpp)

foreach(subdir ${SUBDIRECTORIES})
    add_subdirectory(${subdir})
    target_link_libraries(CourseProject PRIVATE ${subdir})
    target_include_directories(CourseProject PRIVATE ${subdir})
endforeach()

