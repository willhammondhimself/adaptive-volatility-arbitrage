# C++ Execution Gateway - CMake Build Configuration
#
# PURPOSE:
# Cross-platform build system for macOS (Apple Silicon), Windows, and Linux.
# Finds dependencies (ZeroMQ, Protobuf), generates Protobuf C++ code, and compiles the executable.
#
# DESIGN RATIONALE:
# - Modern CMake (≥3.15): Uses targets, not variables
# - Cross-platform: Handles Homebrew (macOS), vcpkg (Windows), apt (Linux)
# - Protobuf generation: Automatically creates .pb.h/.pb.cc from .proto
# - Warnings: Treat code quality seriously (-Wall -Wextra)
#
# BUILD INSTRUCTIONS:
#
# macOS (Apple Silicon):
#   brew install cmake zeromq cppzmq protobuf
#   mkdir build && cd build
#   cmake .. -DCMAKE_BUILD_TYPE=Release
#   cmake --build .
#   ./execution_gateway
#
# Windows (vcpkg):
#   vcpkg install zeromq:x64-windows cppzmq:x64-windows protobuf:x64-windows
#   mkdir build && cd build
#   cmake .. -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
#   cmake --build . --config Release
#   .\Release\execution_gateway.exe
#
# Linux (Ubuntu/Debian):
#   sudo apt install cmake libzmq3-dev libprotobuf-dev protobuf-compiler
#   mkdir build && cd build
#   cmake .. -DCMAKE_BUILD_TYPE=Release
#   cmake --build .
#   ./execution_gateway

cmake_minimum_required(VERSION 3.15)

# Project metadata
project(execution_gateway
    VERSION 1.0.0
    DESCRIPTION "Low-latency order matching engine for quant trading"
    LANGUAGES CXX
)

#=============================================================================
# C++ STANDARD AND COMPILER SETTINGS
#=============================================================================

# Why C++17? Balance of modern features and compiler support
# - std::optional (for try_pop)
# - Structured bindings (for readability)
# - Inline variables (for constants)
# Alternative: C++20 adds concepts and ranges, but less widely supported
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)  # Force -std=c++17, not -std=gnu++17

# Warning flags: Catch bugs early
# Why treat warnings as errors? Enforce code quality at build time
if(MSVC)
    # MSVC-specific warnings
    add_compile_options(/W4)  # Warning level 4 (high)
    # /WX would treat warnings as errors (commented for flexibility)
    # add_compile_options(/WX)
else()
    # GCC/Clang warnings
    add_compile_options(
        -Wall          # Enable all common warnings
        -Wextra        # Enable extra warnings
        -Wpedantic     # Enforce ISO C++ compliance
        -Wno-gnu-zero-variadic-macro-arguments  # Suppress warnings from Protobuf/Abseil C++17 compat
        -Wno-nullability-extension              # Suppress Clang nullability extension warnings from Abseil
        -Wno-gcc-compat                         # Suppress GCC compatibility warnings (enable_if, diagnose_if from Abseil)
        -Wno-unused-parameter                   # Suppress unused parameter warnings
        # -Werror would treat warnings as errors (commented for flexibility)
    )
endif()

#=============================================================================
# PLATFORM DETECTION AND USER GUIDANCE
#=============================================================================

# Provide helpful messages about dependency installation
if(APPLE)
    message(STATUS "========================================")
    message(STATUS "macOS detected")
    message(STATUS "Install dependencies with:")
    message(STATUS "  brew install cmake zeromq cppzmq protobuf")
    message(STATUS "========================================")
elseif(WIN32)
    message(STATUS "========================================")
    message(STATUS "Windows detected")
    message(STATUS "Install dependencies with vcpkg:")
    message(STATUS "  vcpkg install zeromq:x64-windows cppzmq:x64-windows protobuf:x64-windows")
    message(STATUS "Then run CMake with:")
    message(STATUS "  cmake .. -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake")
    message(STATUS "========================================")
else()
    message(STATUS "========================================")
    message(STATUS "Linux detected")
    message(STATUS "Install dependencies with:")
    message(STATUS "  sudo apt install cmake libzmq3-dev libprotobuf-dev protobuf-compiler")
    message(STATUS "========================================")
endif()

#=============================================================================
# FIND DEPENDENCIES
#=============================================================================

# ZeroMQ: High-performance asynchronous messaging library
# Why ZeroMQ? Low latency, language-agnostic, battle-tested in HFT
# find_package looks for ZeroMQConfig.cmake (provided by vcpkg/Homebrew)
find_package(ZeroMQ QUIET)

if(NOT ZeroMQ_FOUND)
    # Fallback: Use pkg-config (common on Linux)
    # Why fallback? Not all installations provide CMake config files
    find_package(PkgConfig)
    if(PkgConfig_FOUND)
        pkg_check_modules(ZeroMQ REQUIRED libzmq)
        message(STATUS "ZeroMQ found via pkg-config")
    else()
        message(FATAL_ERROR "ZeroMQ not found. Install it using your package manager.")
    endif()
else()
    message(STATUS "ZeroMQ found: ${ZeroMQ_VERSION}")
endif()

# cppzmq: C++ header-only wrapper for ZeroMQ
# Why cppzmq? Provides RAII wrappers (zmq::context_t, zmq::socket_t)
# This is header-only, so we just need to find the include path
find_path(CPPZMQ_INCLUDE_DIR
    NAMES zmq.hpp
    HINTS
        /usr/local/include              # Homebrew default (Intel Mac)
        /opt/homebrew/include           # Homebrew default (Apple Silicon)
        /usr/include                    # Linux default
        ${ZeroMQ_INCLUDE_DIRS}          # Same location as libzmq
)

if(NOT CPPZMQ_INCLUDE_DIR)
    message(FATAL_ERROR "cppzmq (zmq.hpp) not found. Install with: brew install cppzmq (macOS) or vcpkg install cppzmq (Windows)")
endif()

message(STATUS "cppzmq found: ${CPPZMQ_INCLUDE_DIR}")

# Protobuf: Google Protocol Buffers for serialization
# Why Protobuf? Compact, fast, language-agnostic, with code generation
# Alternative: Cap'n Proto (faster), FlatBuffers (zero-copy)
find_package(Protobuf REQUIRED)
message(STATUS "Protobuf found: ${Protobuf_VERSION}")
message(STATUS "Protobuf compiler: ${Protobuf_PROTOC_EXECUTABLE}")

#=============================================================================
# GENERATE PROTOBUF C++ CODE
#=============================================================================

# Protobuf source file
set(PROTO_FILE ${CMAKE_CURRENT_SOURCE_DIR}/proto/messages.proto)

# Generate C++ files from .proto
# Output: messages.pb.h, messages.pb.cc in ${CMAKE_CURRENT_BINARY_DIR}
# Why in binary dir? Keeps generated files separate from source
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILE})

# Show generated files (helpful for debugging)
message(STATUS "Protobuf generated headers: ${PROTO_HDRS}")
message(STATUS "Protobuf generated sources: ${PROTO_SRCS}")

#=============================================================================
# LIBRARY: ORDER BOOK LOGIC
#=============================================================================

# Why separate library? Enables unit testing without linking full server
# Alternative: Single executable (simpler, but harder to test)
add_library(execution_lib
    src/order_book.cpp
    ${PROTO_SRCS}  # Include generated Protobuf code
)

# Include directories (PUBLIC = propagates to dependents)
target_include_directories(execution_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include     # Our headers (order_book.hpp, etc.)
    ${CMAKE_CURRENT_BINARY_DIR}             # Generated Protobuf headers
    ${CPPZMQ_INCLUDE_DIR}                   # cppzmq (zmq.hpp)
    ${Protobuf_INCLUDE_DIRS}                # Protobuf headers
)

# Link libraries (PUBLIC = propagates to dependents)
# Why target_link_libraries? Automatically handles transitive dependencies
if(ZeroMQ_FOUND AND TARGET libzmq)
    # Modern CMake: link to imported target
    target_link_libraries(execution_lib PUBLIC libzmq)
elseif(ZeroMQ_LIBRARIES)
    # Fallback: link to libraries found by pkg-config
    target_link_libraries(execution_lib PUBLIC ${ZeroMQ_LIBRARIES})
    target_include_directories(execution_lib PUBLIC ${ZeroMQ_INCLUDE_DIRS})
endif()

target_link_libraries(execution_lib PUBLIC
    ${Protobuf_LIBRARIES}
)

#=============================================================================
# EXECUTABLE: MAIN SERVER
#=============================================================================

# Main server: Thread A (network) + Thread B (matching)
# Why separate executable? Clear entry point, easier to run
add_executable(execution_gateway
    src/server.cpp
)

# Link to our library (transitively gets ZeroMQ, Protobuf)
target_link_libraries(execution_gateway PRIVATE
    execution_lib
)

# Platform-specific threading library
# Why? std::thread requires linking to pthread on Unix
if(UNIX AND NOT APPLE)
    target_link_libraries(execution_gateway PRIVATE pthread)
endif()

#=============================================================================
# INSTALLATION
#=============================================================================

# Install executable to CMAKE_INSTALL_PREFIX/bin
# Example: make install → /usr/local/bin/execution_gateway
install(TARGETS execution_gateway
    RUNTIME DESTINATION bin
)

#=============================================================================
# OPTIONAL: UNIT TESTS
#=============================================================================

# Enable testing with: cmake .. -DBUILD_TESTS=ON
option(BUILD_TESTS "Build unit tests (requires Google Test)" OFF)

if(BUILD_TESTS)
    enable_testing()

    # Find Google Test
    find_package(GTest REQUIRED)

    # Test executable: order book tests
    add_executable(test_order_book tests/test_order_book.cpp)
    target_link_libraries(test_order_book PRIVATE execution_lib GTest::GTest GTest::Main)
    add_test(NAME OrderBookTests COMMAND test_order_book)

    # Test executable: queue tests
    add_executable(test_queue tests/test_queue.cpp)
    target_link_libraries(test_queue PRIVATE execution_lib GTest::GTest GTest::Main)
    add_test(NAME QueueTests COMMAND test_queue)

    message(STATUS "Unit tests enabled")
else()
    message(STATUS "Unit tests disabled (use -DBUILD_TESTS=ON to enable)")
endif()

#=============================================================================
# BUILD SUMMARY
#=============================================================================

message(STATUS "========================================")
message(STATUS "Configuration Summary")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: C++${CMAKE_CXX_STANDARD}")
message(STATUS "  ZeroMQ: Found")
message(STATUS "  Protobuf: ${Protobuf_VERSION}")
message(STATUS "  cppzmq: Found")
message(STATUS "  Output: ${CMAKE_CURRENT_BINARY_DIR}/execution_gateway")
message(STATUS "========================================")
message(STATUS "Build with:")
message(STATUS "  cmake --build .")
message(STATUS "Run with:")
if(WIN32)
    message(STATUS "  .\\Release\\execution_gateway.exe")
else()
    message(STATUS "  ./execution_gateway")
endif()
message(STATUS "========================================")

#=============================================================================
# PERFORMANCE BUILD NOTES
#=============================================================================
#
# RELEASE BUILD (recommended for benchmarking):
#   cmake .. -DCMAKE_BUILD_TYPE=Release
#   Enables: -O3 optimization, inlining, loop unrolling
#
# DEBUG BUILD (for development):
#   cmake .. -DCMAKE_BUILD_TYPE=Debug
#   Enables: -g debug symbols, -O0 no optimization, assertions
#
# RELWITHDEBINFO BUILD (for profiling):
#   cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
#   Enables: -O2 optimization + -g debug symbols
#   Useful for: perf, Instruments, gprof
#
# LTO (Link-Time Optimization):
#   cmake .. -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON
#   Cross-file inlining, can reduce binary size 10-20%
#   Trade-off: Slower compile times
#
# NATIVE OPTIMIZATION (x86-64 specific):
#   add_compile_options(-march=native)
#   Uses CPU-specific instructions (AVX2, AVX-512)
#   Trade-off: Binary not portable to older CPUs
#
#=============================================================================
