cmake_minimum_required(VERSION 3.10)
project(NeuralNet VERSION 1.0.0 LANGUAGES C)

# Set C standard
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Include directories
include_directories(include)

# Library source files
set(LIB_SOURCES
    src/nn.c
)

# Create static library
add_library(nn_static STATIC ${LIB_SOURCES})
target_include_directories(nn_static PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Create shared library
add_library(nn_shared SHARED ${LIB_SOURCES})
target_include_directories(nn_shared PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Set library properties
set_target_properties(nn_static PROPERTIES OUTPUT_NAME nn)
set_target_properties(nn_shared PROPERTIES OUTPUT_NAME nn)

# Link math library for both static and shared
target_link_libraries(nn_static m)
target_link_libraries(nn_shared m)

# Optional: Build examples
option(BUILD_EXAMPLES "Build example programs" ON)
if(BUILD_EXAMPLES)
    add_executable(xor_example examples/xor.c)
    target_link_libraries(xor_example nn_static)
endif()

# Optional: Build tests
option(BUILD_TESTS "Build test programs" ON)
if(BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests")
    add_executable(test_nn tests/test_nn.c)
    target_link_libraries(test_nn nn_static)
endif()

# Installation
include(GNUInstallDirs)

# Install libraries
install(TARGETS nn_static nn_shared
    EXPORT NeuralNetTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Install headers
install(FILES include/nn.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install CMake config files
install(EXPORT NeuralNetTargets
    FILE NeuralNetTargets.cmake
    NAMESPACE NeuralNet::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/NeuralNet
)

# Create config file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    NeuralNetConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    ${CMAKE_CURRENT_LIST_DIR}/cmake/NeuralNetConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/NeuralNetConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/NeuralNet
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/NeuralNetConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/NeuralNetConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/NeuralNet
)

# Create pkg-config file
configure_file(
    ${CMAKE_CURRENT_LIST_DIR}/cmake/neuralnet.pc.in
    ${CMAKE_CURRENT_BINARY_DIR}/neuralnet.pc
    @ONLY
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/neuralnet.pc
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
)