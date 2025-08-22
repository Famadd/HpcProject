cmake_minimum_required(VERSION 4.0.0)
project(CPU VERSION 1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Sources & Includes
file(GLOB SOURCES "src/*.cpp" "src/*.tpp")
file(GLOB HEADERS "includes/*.h" "includes/*.tpp")

include_directories(includes)

# Exécutable
add_executable(CPU ${SOURCES} ${HEADERS})

set_target_properties(CPU PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/../bin"
)



