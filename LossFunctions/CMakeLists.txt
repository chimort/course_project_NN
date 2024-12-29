add_library(LossFunctions
    LossFunction.cpp
)

target_include_directories(LossFunctions PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})