cmake_minimum_required(VERSION 3.10)
project(NeuralNet C)

set(CMAKE_C_STANDARD 99)

# Include directory
include_directories(include)

# All .c files in src
file(GLOB_RECURSE ALL_SRC_FILES src/*.c)

# Main executable (with main.c)
add_executable(nn ${ALL_SRC_FILES})
