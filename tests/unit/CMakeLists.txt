set(us4_runtime_smoke_test_sources
  runtime_smoke_test.cpp
)

set(us4_runtime_native_contract_runner_sources
  runtime_contract_runner.cpp
)

set(us4_runtime_contract_test_sources
  runtime_mode_contract_smoke_test.cpp
  hardware_probe_contract_smoke_test.cpp
  telemetry_contract_smoke_test.cpp
  backend_selector_contract_test.cpp
  model_asset_contract_test.cpp
  kv_contract_test.cpp
  moe_contract_test.cpp
  runtime_acceleration_contract_test.cpp
  tensor_contract_test.cpp
  scalar_kernels_contract_test.cpp
  adapter_generation_contract_test.cpp
)

add_executable(us4_runtime_smoke_test
  ${us4_runtime_smoke_test_sources}
)

target_link_libraries(us4_runtime_smoke_test PRIVATE us4_runtime_core)

target_include_directories(us4_runtime_smoke_test
  PRIVATE
    ${CMAKE_SOURCE_DIR}/runtime
    ${CMAKE_BINARY_DIR}/generated
)

add_test(NAME us4_runtime_smoke_test COMMAND us4_runtime_smoke_test)

add_executable(us4_runtime_contract_runner
  ${us4_runtime_native_contract_runner_sources}
)

target_link_libraries(us4_runtime_contract_runner PRIVATE us4_runtime_core)

target_compile_definitions(us4_runtime_contract_runner PRIVATE US4_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

target_include_directories(us4_runtime_contract_runner
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/runtime
    ${CMAKE_BINARY_DIR}/generated
)

add_test(NAME us4_runtime_contract_runner COMMAND us4_runtime_contract_runner)

find_package(GTest CONFIG QUIET)

if(NOT GTest_FOUND)
  find_package(GTest QUIET)
endif()

if(GTest_FOUND)
  include(GoogleTest)

  add_executable(us4_runtime_contract_tests
    ${us4_runtime_contract_test_sources}
  )

  target_link_libraries(us4_runtime_contract_tests PRIVATE GTest::gtest_main us4_runtime_core)
  target_compile_definitions(us4_runtime_contract_tests PRIVATE US4_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

  target_include_directories(us4_runtime_contract_tests
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_SOURCE_DIR}/runtime
      ${CMAKE_BINARY_DIR}/generated
  )

  gtest_discover_tests(us4_runtime_contract_tests)
endif()
