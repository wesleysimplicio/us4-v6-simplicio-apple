add_library(us4_runtime_core
  core/backend_selector.cpp
  core/runtime_mode.cpp
  core/hardware_probe.cpp
  core/gqa_attention.cpp
  core/model_asset.cpp
  core/rope.cpp
  core/runtime_context.cpp
  core/tensor.cpp
  cpu/scalar_matmul.cpp
  cpu/scalar_attention.cpp
  adapters/dense_adapter_base.cpp
  adapters/adapter_registry.cpp
  adapters/bitnet/bitnet_adapter.cpp
  adapters/deepseek/deepseek_moe_adapter.cpp
  adapters/qwen/qwen_adapter.cpp
  adapters/gemma/gemma_adapter.cpp
  adapters/kimi/kimi_moe_adapter.cpp
  adapters/llama/llama_adapter.cpp
  adapters/ternary/ternary_adapter.cpp
  kv/kv_pager.cpp
  kv/prefix_cache.cpp
  kv/ssd_cold_store.cpp
  kv/summarizer.cpp
  memory/unified_allocator.cpp
  metal/command_queue.cpp
  mlx/mlx_bridge.cpp
  moe/expert_pager.cpp
  moe/router.cpp
  neon/neon_attention.cpp
  neon/neon_matmul.cpp
  telemetry/telemetry_types.cpp
  telemetry/telemetry_sink.cpp
)

target_include_directories(us4_runtime_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/../generated
)

target_compile_features(us4_runtime_core PUBLIC cxx_std_20)

if(US4_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
