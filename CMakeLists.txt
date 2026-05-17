cmake_minimum_required(VERSION 3.27)

project(us4 VERSION 0.1.21 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(US4_BUILD_TESTS "Build runtime smoke tests" ON)
option(US4_BUILD_BENCHMARKS "Build runtime benchmark placeholders" ON)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/runtime/core/version.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/generated/us4/version.h
  @ONLY
)

add_subdirectory(runtime)
add_subdirectory(apps)

if(US4_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
