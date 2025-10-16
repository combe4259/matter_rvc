cmake_minimum_required(VERSION 3.16)

project(FlowerRVCClient VERSION 0.1
  DESCRIPTION "Flower C++ Client for RVC (Robot Vacuum Cleaner)"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define NDEBUG to match protobuf build mode (inline destructors)
add_compile_definitions(NDEBUG)

message(STATUS "========================================")
message(STATUS "Flower RVC Client Build Configuration")
message(STATUS "========================================")

######################
### Use custom protobuf 3.20.3 and gRPC 1.48.2 - direct linking to avoid anaconda conflicts
message(STATUS "Using custom protobuf 3.20.3 and gRPC 1.48.2...")
set(PROTOBUF_ROOT "/Users/inter4259/protobuf-3.20.3")
set(GRPC_ROOT "/Users/inter4259/grpc-1.48.2")

# Include directories
include_directories(
  ${PROTOBUF_ROOT}/include
  ${GRPC_ROOT}/include
)

# Manually list all necessary gRPC/protobuf/abseil libraries
set(GRPC_LIBRARIES
  ${GRPC_ROOT}/lib/libgrpc++_reflection.a
  ${GRPC_ROOT}/lib/libgrpc++.a
  ${GRPC_ROOT}/lib/libgrpc.a
  ${PROTOBUF_ROOT}/lib/libprotobuf.a
  ${GRPC_ROOT}/lib/libaddress_sorting.a
  ${GRPC_ROOT}/lib/libgpr.a
  ${GRPC_ROOT}/lib/libupb.a
  ${GRPC_ROOT}/lib/libcares.a
  ${GRPC_ROOT}/lib/libre2.a
)

# Glob all abseil libraries from our gRPC build
file(GLOB GRPC_ABSL_LIBS "${GRPC_ROOT}/lib/libabsl_*.a")

message(STATUS "Protobuf root: ${PROTOBUF_ROOT}")
message(STATUS "gRPC root: ${GRPC_ROOT}")
message(STATUS "Found ${CMAKE_MATCH_COUNT} abseil libraries")

######################
### Flower C++ SDK
message(STATUS "Setting up Flower C++ SDK...")
set(FLWR_SDK_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/flower-cpp")

file(GLOB FLWR_SRCS "${FLWR_SDK_PATH}/src/*.cc")
file(GLOB FLWR_PROTO_SRCS "${FLWR_SDK_PATH}/include/flwr/proto/*.cc")
set(FLWR_INCLUDE_DIR "${FLWR_SDK_PATH}/include")

message(STATUS "Flower SDK Path: ${FLWR_SDK_PATH}")

add_library(flwr STATIC ${FLWR_SRCS} ${FLWR_PROTO_SRCS})

target_include_directories(flwr PUBLIC
  ${FLWR_INCLUDE_DIR}
  ${PROTOBUF_ROOT}/include
  ${GRPC_ROOT}/include
)

target_link_libraries(flwr
  ${GRPC_LIBRARIES}
  ${GRPC_ABSL_LIBS}
)

######################
### Simplified RVC Client (without TFLM for now)
message(STATUS "Building simplified Flower RVC client...")

add_executable(flower-rvc-client-simple
  simple_test.cpp
)

target_include_directories(flower-rvc-client-simple PRIVATE
  ${FLWR_INCLUDE_DIR}
  ../rvc-common/include
  ${PROTOBUF_ROOT}/include
  ${GRPC_ROOT}/include
)

# Find system libraries
find_package(OpenSSL REQUIRED)
find_package(Threads REQUIRED)
find_package(ZLIB REQUIRED)

target_link_libraries(flower-rvc-client-simple
  flwr
  ${GRPC_ROOT}/lib/libgrpc++_reflection.a
  ${GRPC_ROOT}/lib/libgrpc++.a
  ${GRPC_ROOT}/lib/libgrpc.a
  ${GRPC_ABSL_LIBS}
  ${GRPC_ROOT}/lib/libupb.a
  ${GRPC_ROOT}/lib/libgpr.a
  ${GRPC_ROOT}/lib/libaddress_sorting.a
  ${GRPC_ROOT}/lib/libcares.a
  ${GRPC_ROOT}/lib/libre2.a
  ${PROTOBUF_ROOT}/lib/libprotobuf.a
  OpenSSL::SSL
  OpenSSL::Crypto
  Threads::Threads
  ZLIB::ZLIB
  "-framework CoreFoundation"
  -lresolv
)

message(STATUS "========================================")
message(STATUS "Configuration complete!")
message(STATUS "Build with: cmake --build build")
message(STATUS "Run with: ./build/flower-rvc-client-simple 0 127.0.0.1:9092")
message(STATUS "========================================")
