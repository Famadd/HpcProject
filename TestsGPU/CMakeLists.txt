cmake_minimum_required(VERSION 3.18)
project(GPU VERSION 1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Sources & Includes
file(GLOB SOURCES "src/*.c" "src/*.cu")
file(GLOB HEADERS "includes/*.h" "includes/*.tpp")

include_directories(includes)

# Exécutable
add_executable(GPU ${SOURCES} ${HEADERS})

target_link_libraries(GPU PRIVATE cublas)

set_target_properties(GPU PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/../bin"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES 75
)

