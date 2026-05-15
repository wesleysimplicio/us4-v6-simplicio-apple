#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <set>
#include <string_view>
#include <vector>

#include "adapters/adapter_registry.h"
#include "adapters/llama/llama_adapter.h"
#include "adapters/qwen/qwen_adapter.h"
#include "core/backend_selector.h"
#include "core/model_asset.h"
#include "core/runtime_context.h"
#include "cpu/scalar_attention.h"
#include "cpu/scalar_matmul.h"
#include "kv/kv_pager.h"
#include "kv/prefix_cache.h"
#include "kv/summarizer.h"
#include "memory/unified_allocator.h"
#include "metal/autorelease_scope.h"
#include "metal/command_queue.h"
#include "metal/dense_dispatch.h"
#include "metal/device_info.h"
#include "metal/kernel_library.h"
#include "mlx/dense_plan.h"
#include "mlx/mlx_bridge.h"
#include "moe/expert_pager.h"
#include "moe/router.h"
#include "neon/dequant_int4.h"
#include "neon/dequant_int8.h"
#include "neon/kernel_profile.h"
#include "neon/neon_attention.h"
#include "neon/neon_matmul.h"
#include "sprint_01_contract_placeholders.h"

namespace {

bool Expect(const bool condition, const char *message) {
  if (!condition) {
    std::cerr << message << "\n";
    return false;
  }
  return true;
}

us4::HardwareProbeResult MakeProbe() {
  us4::HardwareProbeResult probe;
  probe.platform = "macos";
  probe.architecture = "arm64";
  probe.chip = "apple-m";
  probe.unifiedMemoryGiB = 24;
  probe.isAppleSilicon = true;
  probe.hasNeon = true;
  probe.neonVectorBits = 128;
  probe.hasPerformanceCores = true;
  probe.hasEfficiencyCores = true;
  probe.recommendedMode = us4::RuntimeMode::kDegraded;
  return probe;
}

std::filesystem::path RepoRoot() {
#ifdef US4_SOURCE_DIR
  return std::filesystem::path(US4_SOURCE_DIR);
#else
  return std::filesystem::path(__FILE__)
      .parent_path()
      .parent_path()
      .parent_path();
#endif
}

void FillHalfTensor(us4::Tensor &tensor, const std::vector<float> &values,
                    const bool bfloat16) {
  std::uint16_t *data = tensor.MutableDataAsUInt16();
  if (data == nullptr || values.size() != tensor.ElementCount()) {
    return;
  }
  for (std::size_t index = 0; index < values.size(); ++index) {
    data[index] = bfloat16 ? us4::EncodeBFloat16(values[index])
                           : us4::EncodeFloat16(values[index]);
  }
}

float QuantizeHalfValue(const float value, const bool bfloat16) {
  return bfloat16 ? us4::DecodeBFloat16(us4::EncodeBFloat16(value))
                  : us4::DecodeFloat16(us4::EncodeFloat16(value));
}

bool FillHalfReferenceTensor(us4::Tensor &tensor,
                             const std::vector<float> &values,
                             const bool bfloat16) {
  float *data = tensor.MutableDataAsFloat32();
  if (data == nullptr || values.size() != tensor.ElementCount()) {
    return false;
  }
  for (std::size_t index = 0; index < values.size(); ++index) {
    data[index] = QuantizeHalfValue(values[index], bfloat16);
  }
  return true;
}

} // namespace

int main() {
  bool ok = true;

  {
    constexpr auto &modes = sprint_01_contract::kCanonicalRuntimeModes;
    ok &= Expect(modes.size() == 7U,
                 "runtime mode contract should keep 7 canonical modes");
    std::set<int> seen;
    for (const auto mode : modes) {
      ok &= Expect(seen.insert(static_cast<int>(mode)).second,
                   "runtime mode values should stay unique");
    }
  }

  {
    constexpr auto &responsibilities =
        sprint_01_contract::kProbeResponsibilities;
    ok &= Expect(responsibilities.size() == 5U,
                 "hardware probe contract should keep 5 responsibilities");
    std::set<std::string_view> seen;
    for (const auto &responsibility : responsibilities) {
      ok &= Expect(seen.insert(responsibility.name).second,
                   "hardware responsibilities should stay unique");
      ok &= Expect(!responsibility.description.empty(),
                   "hardware responsibility description should not be empty");
    }
  }

  {
    constexpr auto &categories = sprint_01_contract::kTelemetryCategories;
    ok &= Expect(categories.size() == 6U,
                 "telemetry contract should keep 6 categories");
  }

  {
    const auto cpu = us4::ParseBackendType("CPU");
    const auto metal = us4::ParseBackendType("metal");
    const auto invalid = us4::ParseBackendType("cuda");
    ok &= Expect(cpu.has_value() && *cpu == us4::BackendType::kScalarCpu,
                 "cpu alias should parse");
    ok &= Expect(metal.has_value() && *metal == us4::BackendType::kMetal,
                 "metal alias should parse");
    ok &= Expect(!invalid.has_value(), "invalid backend should not parse");

    us4::HardwareProbeResult probe = MakeProbe();
    probe.hasMetal = true;
    probe.hasMlx = true;
    probe.hasNeon = true;
    probe.neonVectorBits = 128;
    probe.hasPerformanceCores = true;
    probe.hasEfficiencyCores = true;
    const us4::LlamaAdapter llama;
    const us4::BackendSelection autoSelection =
        us4::SelectBackend(probe, us4::RuntimeMode::kBalancedPlus, llama);
    ok &= Expect(autoSelection.selected == us4::BackendType::kMetal,
                 "auto backend should prefer metal");
    ok &= Expect(autoSelection.reason == "auto-metal",
                 "auto metal reason should stay explicit");

    const us4::BackendSelection degradedSelection =
        us4::SelectBackend(probe, us4::RuntimeMode::kDegraded, llama);
    ok &= Expect(degradedSelection.selected == us4::BackendType::kMlx,
                 "degraded should prefer mlx before neon");
    ok &= Expect(degradedSelection.reason == "auto-mlx",
                 "auto mlx reason should stay explicit");

    const us4::QwenAdapter qwen;
    const us4::BackendSelection fallbackSelection =
        us4::SelectBackend(MakeProbe(), us4::RuntimeMode::kDegraded, qwen,
                           us4::BackendType::kMetal);
    ok &= Expect(fallbackSelection.selected == us4::BackendType::kNeon,
                 "fallback should pick neon on arm64");
    ok &= Expect(fallbackSelection.fellBack,
                 "requested unavailable backend should mark fallback");
  }

  {
    us4::ModelAsset asset;
    std::string error;
    const auto manifest = RepoRoot() / "tests" / "fixtures" / "models" /
                          "qwen-0.5b" / "model.us4manifest";
    ok &= Expect(us4::LoadModelAsset(manifest, asset, &error),
                 "qwen manifest should load");
    ok &=
        Expect(asset.family == "qwen", "qwen manifest should preserve family");
    ok &= Expect(asset.modelName == "qwen-0.5b-fixture",
                 "qwen manifest should preserve model name");
  }

  {
    us4::UnifiedAllocator allocator;
    allocator.Allocate(32, false);
    const auto shared = allocator.Allocate(128, true);
    ok &= Expect(allocator.SharedAllocationCount() == 1U,
                 "allocator should count unified-shared allocations");
    ok &=
        Expect(shared->visibility == us4::AllocationVisibility::kUnifiedShared,
               "shared allocation should be tagged unified-shared");

    us4::HardwareProbeResult probe = MakeProbe();
    probe.hasMetal = true;
    probe.hasMlx = true;
    us4::RuntimeContext context(probe);
    ok &= Expect(context.metalQueue().Available(),
                 "metal queue should be available on metal probe");
    ok &= Expect(context.metalQueue().Profile().stage ==
                     us4::MetalInitializationStage::kQueueReady,
                 "metal queue should surface queue-ready init stage");
    ok &= Expect(context.metalQueue().Profile().queueCreated,
                 "metal queue should mark queue creation");
    ok &= Expect(
        context.metalQueue().Profile().requiresAutoreleaseBoundary,
        "metal queue should request autorelease boundary on macos probe");
    ok &=
        Expect(context.metalQueue().Device().queueLabel == "us4.metal.default",
               "metal queue should expose queue label");
    ok &= Expect(context.metalQueue().Device().supportsUnifiedMemory,
                 "metal queue should expose unified memory support");
    ok &= Expect(context.mlxBridge().Available(),
                 "mlx bridge should be available on mlx probe");
    ok &= Expect(context.metalQueue().Dispatch(us4::MetalKernelKind::kSoftmax,
                                               2, 64, shared),
                 "metal queue should record dispatch contract");
    ok &= Expect(context.metalQueue().DispatchCount() == 1U,
                 "metal queue should count dispatches");
    ok &= Expect(context.metalQueue().Dispatches().front().entryPoint ==
                     "us4_softmax_rows",
                 "metal queue should surface dispatch entry point");
    ok &= Expect(context.metalQueue().Dispatches().front().relativePath ==
                     "runtime/metal/kernels/softmax.metal",
                 "metal queue should surface dispatch kernel path");
    ok &= Expect(
        context.metalQueue().Dispatches().front().autoreleaseBoundaryRequested,
        "metal queue should record autorelease boundary request");
    const us4::DenseMetalDispatchPlan densePlan =
        us4::BuildDenseMetalDispatchPlan(8, 8, 16);
    ok &= Expect(densePlan.steps.size() == 3U,
                 "dense metal dispatch plan should keep 3 stages");
    const us4::MlxDensePlan mlxPlan = us4::BuildMlxDensePlan(8, 8, 16);
    ok &= Expect(mlxPlan.operations.size() == 3U,
                 "mlx dense plan should keep 3 operations");
    ok &= Expect(context.mlxBridge().BuildDensePlan("llama", 32, shared),
                 "mlx bridge should build dense plan");
    ok &= Expect(context.mlxBridge().EvaluateLastPlan(),
                 "mlx bridge should evaluate last plan");
    ok &= Expect(context.mlxBridge().LastPlan().has_value() &&
                     context.mlxBridge().LastPlan()->operations.size() == 3U,
                 "mlx bridge should surface 3 recorded operations");
    ok &= Expect(us4::GetMetalKernelCatalog().size() == 3U,
                 "metal kernel catalog should keep 3 kernels");
    ok &=
        Expect(us4::FindMetalKernel(us4::MetalKernelKind::kRmsNorm) != nullptr,
               "metal kernel catalog should resolve rmsnorm");
    const us4::ScopedAutoreleasePool pool(true);
    ok &= Expect(pool.Requested(),
                 "autorelease scope should record request intent");
    if (pool.Kind() == us4::AutoreleaseBoundaryKind::kObjectiveC) {
      ok &= Expect(
          pool.Active(),
          "objective-c autorelease scope should be active on apple hosts");
    } else {
      ok &= Expect(
          !pool.Active(),
          "noop autorelease scope should remain inactive off apple hosts");
    }
  }

  {
    us4::PrefixCache cache;
    cache.Retain("hello");
    cache.Retain("hello");
    const auto retained = cache.Lookup("hello");
    ok &= Expect(retained.has_value() && retained->refCount == 2U,
                 "prefix cache should count retains");
    cache.Release("hello");
    cache.Release("hello");
    ok &= Expect(!cache.Lookup("hello").has_value(),
                 "prefix cache should erase on last release");

    us4::KvPager pager(1);
    pager.Append("prompt-a", {1.0F, 2.0F});
    pager.Append("prompt-b", {3.0F, 4.0F});
    const auto page = pager.Lookup("prompt-a");
    ok &= Expect(page.has_value(), "kv pager should find stored page");
    ok &= Expect(page->hitCount >= 1U,
                 "kv pager lookup should increase hit count");

    us4::Summarizer summarizer;
    const auto summary = summarizer.Summarize({2.0F, 4.0F, 6.0F});
    ok &= Expect(summary.size() == 1U && summary[0] == 4.0F,
                 "summarizer should produce arithmetic mean");
  }

  {
    us4::Router router;
    const auto topk = router.TopK({0.1F, 0.8F, 0.4F, 0.7F}, 2);
    ok &= Expect(topk.size() == 2U,
                 "router should clamp top-k to requested size");
    ok &=
        Expect(topk[0].expert == 1U, "router should sort highest score first");

    us4::ExpertPager pager(2);
    pager.Touch("expert-a");
    pager.Touch("expert-b");
    pager.Touch("expert-a");
    pager.Touch("expert-c");
    ok &= Expect(pager.ResidentCount() == 2U,
                 "expert pager should enforce resident limit");
    ok &= Expect(pager.IsResident("expert-a"),
                 "expert pager should keep hot expert resident");
  }

  {
    const us4::IUS4V6Adapter *qwen = us4::FindAdapterByModel("QWEN-0.5B");
    const us4::IUS4V6Adapter *llama = us4::FindAdapterByModel("llama-3.1-8b");
    const us4::IUS4V6Adapter *bitnet =
        us4::FindAdapterByModel("bitnet-b1.58-2b");
    const us4::IUS4V6Adapter *ternaryAdapter =
        us4::FindAdapterByModel("pt-bitnet-ternary-2b");
    const us4::IUS4V6Adapter *ternary =
        us4::FindAdapterByModel("pt-bitnet-ternary-2b");
    ok &= Expect(qwen != nullptr, "registry should find qwen by model");
    ok &= Expect(llama != nullptr, "registry should find llama by model");
    ok &= Expect(bitnet != nullptr, "registry should find bitnet by model");
    ok &= Expect(ternaryAdapter != nullptr,
                 "registry should find ternary adapter by model");
    ok &= Expect(ternary != nullptr,
                 "registry should find ternary by exact model");
    ok &= Expect(ternary != nullptr && ternary->Family() == "ternary",
                 "registry should not misroute ternary to bitnet");

    us4::ModelAsset asset;
    std::string error;
    const auto manifest = RepoRoot() / "tests" / "fixtures" / "models" /
                          "qwen-0.5b" / "model.us4manifest";
    ok &= Expect(us4::LoadModelAsset(manifest, asset, &error),
                 "adapter generation manifest should load");
    us4::RuntimeContext context(MakeProbe());
    qwen->ConfigureRuntime(context);
    const us4::GenerationResult result =
        qwen->Generate({.prompt = "Hi, US4!",
                        .maxTokens = 4,
                        .asset = &asset,
                        .requestedBackend = us4::BackendType::kMetal},
                       context);
    ok &= Expect(result.modelName == "qwen-0.5b-fixture",
                 "generation should surface manifest model name");
    ok &= Expect(result.backendReason == "requested-backend-unavailable",
                 "generation should expose fallback reason");
    ok &= Expect(result.fellBack, "generation should mark backend fallback");
    ok &= Expect(result.sharedAllocations == 0U,
                 "scalar fallback should not record shared allocations");
    ok &= Expect(result.metalDispatches == 0U,
                 "scalar fallback should not record metal dispatches");
    ok &= Expect(result.mlxOperationCount == 0U,
                 "scalar fallback should not record mlx operations");

    const us4::GenerationResult autoResult = qwen->Generate(
        {.prompt = "Hi, US4!", .maxTokens = 4, .asset = &asset}, context);
    ok &= Expect(autoResult.backendReason == "auto-neon" ||
                     autoResult.backendReason == "auto-scalar",
                 "auto generation should expose explicit backend reason");
    ok &= Expect(result.weightDType == "fp16",
                 "generation should surface asset weight dtype");
    ok &= Expect(result.neonKernelFlavor == "fp16-lane8",
                 "generation should surface neon flavor for fp16 assets");
    ok &= Expect(result.dequantPath == "none",
                 "fp16 assets should not request dequant path");

    us4::HardwareProbeResult appleProbe = MakeProbe();
    appleProbe.hasMetal = true;
    appleProbe.hasMlx = true;
    appleProbe.unifiedMemoryGiB = 96;
    appleProbe.recommendedMode = us4::RuntimeMode::kBalancedPlus;
    us4::RuntimeContext acceleratedContext(appleProbe);
    const us4::GenerationResult llamaResult =
        llama->Generate({.prompt = "metal path",
                         .maxTokens = 4,
                         .requestedBackend = us4::BackendType::kMetal},
                        acceleratedContext);
    ok &= Expect(llamaResult.backend == "metal",
                 "llama should keep metal backend when available");
    ok &= Expect(acceleratedContext.metalQueue().DispatchCount() == 3U,
                 "llama generation should touch the metal scaffold");
    ok &= Expect(acceleratedContext.allocator().SharedAllocationCount() == 1U,
                 "metal scaffold should allocate unified-shared memory");
    ok &= Expect(llamaResult.sharedAllocations == 1U,
                 "result should surface shared allocation count");
    ok &= Expect(llamaResult.metalDispatches == 3U,
                 "result should surface metal dispatch count");
    ok &= Expect(llamaResult.mlxOperationCount == 0U,
                 "metal path should not report mlx operations");
    ok &= Expect(llamaResult.metalQueueLabel == "us4.metal.default",
                 "result should surface metal queue label");
    ok &= Expect(qwen->SupportsMetalBackend(),
                 "qwen should declare metal support");

    us4::ModelAsset bitnetAsset;
    const auto bitnetManifest = RepoRoot() / "tests" / "fixtures" / "models" /
                                "bitnet-b1.58-2b" / "model.us4manifest";
    ok &= Expect(us4::LoadModelAsset(bitnetManifest, bitnetAsset, &error),
                 "bitnet manifest should load");
    us4::RuntimeContext bitnetContext(MakeProbe());
    bitnet->ConfigureRuntime(bitnetContext);
    const us4::GenerationResult bitnetResult = bitnet->Generate(
        {.prompt = "hi", .maxTokens = 2, .asset = &bitnetAsset}, bitnetContext);
    ok &= Expect(bitnetResult.weightDType == "int8",
                 "bitnet generation should surface int8 weight dtype");
    ok &= Expect(bitnetResult.dequantPath == "groupwise-int8",
                 "bitnet generation should surface int8 dequant path");
    ok &= Expect(bitnetResult.neonKernelFlavor == "int8-dot",
                 "bitnet generation should surface int8 dot neon flavor");

    us4::ModelAsset ternaryAsset;
    const auto ternaryManifest = RepoRoot() / "tests" / "fixtures" / "models" /
                                 "pt-bitnet-ternary-2b" / "model.us4manifest";
    ok &= Expect(us4::LoadModelAsset(ternaryManifest, ternaryAsset, &error),
                 "ternary manifest should load");
    us4::RuntimeContext ternaryContext(MakeProbe());
    ternaryAdapter->ConfigureRuntime(ternaryContext);
    const us4::GenerationResult ternaryResult = ternaryAdapter->Generate(
        {.prompt = "hi", .maxTokens = 2, .asset = &ternaryAsset},
        ternaryContext);
    ok &= Expect(ternaryResult.weightDType == "int4",
                 "ternary generation should surface int4 weight dtype");
    ok &= Expect(ternaryResult.dequantPath == "groupwise-int4",
                 "ternary generation should surface int4 dequant path");
    ok &= Expect(ternaryResult.neonKernelFlavor == "scalar-bridge",
                 "ternary generation should surface scalar-bridge neon flavor");
  }

  {
    us4::HardwareProbeResult neonProbe = MakeProbe();
    neonProbe.hasNeon = true;
    neonProbe.neonVectorBits = 128;
    neonProbe.hasPerformanceCores = true;
    neonProbe.hasEfficiencyCores = true;

    const us4::Tensor fp16Lhs({8, 16}, us4::DType::kFloat16,
                              us4::DeviceType::kCpu);
    const us4::Tensor fp16Rhs({16, 32}, us4::DType::kFloat16,
                              us4::DeviceType::kCpu);
    const us4::NeonMatmulProfile matmulProfile =
        us4::PlanNeonMatmul(neonProbe, fp16Lhs, fp16Rhs);
    ok &= Expect(matmulProfile.flavor == us4::NeonKernelFlavor::kFp16Lane8,
                 "neon matmul should pick fp16 lane8 profile on arm64");
    ok &= Expect(matmulProfile.tileRows == 8U && matmulProfile.tileCols == 8U,
                 "neon matmul should keep 8x8 tile contract");
    const us4::Tensor bf16Lhs({8, 16}, us4::DType::kBFloat16,
                              us4::DeviceType::kCpu);
    const us4::Tensor bf16Rhs({16, 24}, us4::DType::kBFloat16,
                              us4::DeviceType::kCpu);
    const us4::NeonMatmulProfile bf16MatmulProfile =
        us4::PlanNeonMatmul(neonProbe, bf16Lhs, bf16Rhs);
    ok &= Expect(
        bf16MatmulProfile.flavor == us4::NeonKernelFlavor::kBf16Lane8,
        "neon matmul should pick bf16 lane8 profile on arm64");
    ok &= Expect(
        bf16MatmulProfile.tileRows == 8U && bf16MatmulProfile.tileCols == 8U,
        "neon matmul should keep 8x8 tile contract for bf16");

    const us4::Tensor query({1, 8, 64}, us4::DType::kFloat32,
                            us4::DeviceType::kCpu);
    const us4::Tensor key({1, 8, 64}, us4::DType::kFloat32,
                          us4::DeviceType::kCpu);
    const us4::Tensor value({1, 8, 64}, us4::DType::kFloat32,
                            us4::DeviceType::kCpu);
    const us4::NeonAttentionProfile attentionProfile =
        us4::PlanNeonAttention(neonProbe, query, key, value, true);
    ok &=
        Expect(attentionProfile.fusesSoftmaxRescale,
               "neon attention should preserve fused softmax-rescale contract");
    ok &=
        Expect(attentionProfile.headDimBlock == 32U,
               "neon attention should keep 32-wide head blocks when possible");

    us4::Tensor attentionQuery({2, 4}, us4::DType::kFloat32,
                               us4::DeviceType::kCpu);
    us4::Tensor attentionKey({3, 4}, us4::DType::kFloat32,
                             us4::DeviceType::kCpu);
    us4::Tensor attentionValue({3, 3}, us4::DType::kFloat32,
                               us4::DeviceType::kCpu);
    us4::Tensor neonAttentionOut({2, 3}, us4::DType::kFloat32,
                                 us4::DeviceType::kCpu);
    us4::Tensor scalarAttentionOut({2, 3}, us4::DType::kFloat32,
                                   us4::DeviceType::kCpu);
    float *attentionQueryData = attentionQuery.MutableDataAsFloat32();
    float *attentionKeyData = attentionKey.MutableDataAsFloat32();
    float *attentionValueData = attentionValue.MutableDataAsFloat32();
    for (std::size_t index = 0; index < 8U; ++index) {
      attentionQueryData[index] =
          (static_cast<float>((index % 7U) + 1U) * 0.25F) - 0.5F;
    }
    for (std::size_t index = 0; index < 12U; ++index) {
      attentionKeyData[index] =
          (static_cast<float>((index % 7U) + 1U) * 0.20F) - 0.25F;
    }
    for (std::size_t index = 0; index < 9U; ++index) {
      attentionValueData[index] =
          (static_cast<float>((index % 7U) + 1U) * 0.15F) + 0.10F;
    }
    ok &=
        Expect(us4::NeonAttention(attentionQuery, attentionKey, attentionValue,
                                  neonAttentionOut, false, {}, nullptr),
               "neon attention should execute fp32 path");
    ok &= Expect(us4::ScalarAttention(attentionQuery, attentionKey,
                                      attentionValue, scalarAttentionOut, false,
                                      {}, nullptr),
                 "scalar attention should provide reference path");
    const float *neonAttentionValues = neonAttentionOut.DataAsFloat32();
    const float *scalarAttentionValues = scalarAttentionOut.DataAsFloat32();
    bool attentionMatches =
        neonAttentionValues != nullptr && scalarAttentionValues != nullptr;
    for (std::size_t index = 0; attentionMatches && index < 6U; ++index) {
      const float diff =
          neonAttentionValues[index] - scalarAttentionValues[index];
      attentionMatches = std::abs(diff) <= 1e-5F;
    }
    ok &= Expect(attentionMatches,
                 "neon attention should match scalar fp32 outputs");

    us4::Tensor cacheKeys({1, 4}, us4::DType::kFloat32, us4::DeviceType::kCpu);
    us4::Tensor cacheValues({1, 3}, us4::DType::kFloat32,
                            us4::DeviceType::kCpu);
    us4::Tensor causalNeonOut({2, 3}, us4::DType::kFloat32,
                              us4::DeviceType::kCpu);
    us4::Tensor causalScalarOut({2, 3}, us4::DType::kFloat32,
                                us4::DeviceType::kCpu);
    float *cacheKeyData = cacheKeys.MutableDataAsFloat32();
    float *cacheValueData = cacheValues.MutableDataAsFloat32();
    for (std::size_t index = 0; index < 4U; ++index) {
      cacheKeyData[index] =
          (static_cast<float>((index % 7U) + 1U) * 0.07F) + 0.11F;
    }
    for (std::size_t index = 0; index < 3U; ++index) {
      cacheValueData[index] =
          (static_cast<float>((index % 7U) + 1U) * 0.18F) + 0.02F;
    }
    const us4::AttentionCacheView cache{&cacheKeys, &cacheValues};
    ok &=
        Expect(us4::NeonAttention(attentionQuery, attentionKey, attentionValue,
                                  causalNeonOut, true, cache, nullptr),
               "neon attention should support causal cache path");
    ok &= Expect(us4::ScalarAttention(attentionQuery, attentionKey,
                                      attentionValue, causalScalarOut, true,
                                      cache, nullptr),
                 "scalar attention should support causal cache reference");
    const float *causalNeonValues = causalNeonOut.DataAsFloat32();
    const float *causalScalarValues = causalScalarOut.DataAsFloat32();
    bool causalMatches =
        causalNeonValues != nullptr && causalScalarValues != nullptr;
    for (std::size_t index = 0; causalMatches && index < 6U; ++index) {
      const float diff = causalNeonValues[index] - causalScalarValues[index];
      causalMatches = std::abs(diff) <= 1e-5F;
    }
    ok &= Expect(causalMatches,
                 "neon attention should match scalar outputs with cache");

    us4::Tensor wideValue({3, 5}, us4::DType::kFloat32, us4::DeviceType::kCpu);
    us4::Tensor wideCacheValues({1, 5}, us4::DType::kFloat32,
                                us4::DeviceType::kCpu);
    us4::Tensor wideNeonOut({2, 5}, us4::DType::kFloat32,
                            us4::DeviceType::kCpu);
    us4::Tensor wideScalarOut({2, 5}, us4::DType::kFloat32,
                              us4::DeviceType::kCpu);
    float *wideValueData = wideValue.MutableDataAsFloat32();
    float *wideCacheValueData = wideCacheValues.MutableDataAsFloat32();
    for (std::size_t index = 0; index < 15U; ++index) {
      wideValueData[index] =
          (static_cast<float>((index % 7U) + 1U) * 0.11F) - 0.04F;
    }
    for (std::size_t index = 0; index < 5U; ++index) {
      wideCacheValueData[index] =
          (static_cast<float>((index % 7U) + 1U) * 0.16F) + 0.07F;
    }
    const us4::AttentionCacheView wideCache{&cacheKeys, &wideCacheValues};
    ok &= Expect(us4::NeonAttention(attentionQuery, attentionKey, wideValue,
                                    wideNeonOut, true, wideCache, nullptr),
                 "neon attention should support wide value tail accumulation");
    ok &= Expect(us4::ScalarAttention(attentionQuery, attentionKey, wideValue,
                                      wideScalarOut, true, wideCache, nullptr),
                 "scalar attention should provide wide value tail reference");
    const float *wideNeonValues = wideNeonOut.DataAsFloat32();
    const float *wideScalarValues = wideScalarOut.DataAsFloat32();
    bool wideMatches = wideNeonValues != nullptr && wideScalarValues != nullptr;
    for (std::size_t index = 0; wideMatches && index < 10U; ++index) {
      const float diff = wideNeonValues[index] - wideScalarValues[index];
      wideMatches = std::abs(diff) <= 1e-5F;
    }
    ok &= Expect(
        wideMatches,
        "neon attention should match scalar outputs for wide value tails");

    us4::Tensor fp16MatmulLhs({2, 3}, us4::DType::kFloat16,
                              us4::DeviceType::kCpu);
    us4::Tensor fp16MatmulRhs({3, 5}, us4::DType::kFloat16,
                              us4::DeviceType::kCpu);
    us4::Tensor fp16NeonOut({2, 5}, us4::DType::kFloat32,
                            us4::DeviceType::kCpu);
    us4::Tensor fp16ScalarLhs({2, 3}, us4::DType::kFloat32,
                              us4::DeviceType::kCpu);
    us4::Tensor fp16ScalarRhs({3, 5}, us4::DType::kFloat32,
                              us4::DeviceType::kCpu);
    us4::Tensor fp16ScalarOut({2, 5}, us4::DType::kFloat32,
                              us4::DeviceType::kCpu);
    const std::vector<float> fp16LhsValues = {0.5F, -1.0F, 2.0F,
                                              1.5F, 0.25F, -0.75F};
    const std::vector<float> fp16RhsValues = {
        0.25F, -0.5F, 1.0F,   0.75F, -1.25F, 1.5F,  0.0F, -0.25F,
        0.5F,  1.0F,  -0.75F, 0.25F, 0.5F,   -1.5F, 0.25F};
    FillHalfTensor(fp16MatmulLhs, fp16LhsValues, false);
    FillHalfTensor(fp16MatmulRhs, fp16RhsValues, false);
    ok &= Expect(FillHalfReferenceTensor(fp16ScalarLhs, fp16LhsValues, false),
                 "scalar matmul should receive fp16-rounded lhs reference");
    ok &= Expect(FillHalfReferenceTensor(fp16ScalarRhs, fp16RhsValues, false),
                 "scalar matmul should receive fp16-rounded rhs reference");
    ok &= Expect(
        us4::NeonMatmul(fp16MatmulLhs, fp16MatmulRhs, fp16NeonOut, nullptr),
        "neon matmul should execute fp16 inputs");
    ok &= Expect(
        us4::ScalarMatmul(fp16ScalarLhs, fp16ScalarRhs, fp16ScalarOut, nullptr),
        "scalar matmul should provide fp16 reference");
    const float *fp16NeonValues = fp16NeonOut.DataAsFloat32();
    const float *fp16ScalarValues = fp16ScalarOut.DataAsFloat32();
    bool fp16Matches = fp16NeonValues != nullptr && fp16ScalarValues != nullptr;
    for (std::size_t index = 0; fp16Matches && index < 10U; ++index) {
      const float diff = fp16NeonValues[index] - fp16ScalarValues[index];
      fp16Matches = std::abs(diff) <= 1e-3F;
    }
    ok &= Expect(fp16Matches,
                 "neon matmul should match scalar outputs for fp16 inputs");

    us4::Tensor bf16MatmulLhs({3, 4}, us4::DType::kBFloat16,
                              us4::DeviceType::kCpu);
    us4::Tensor bf16MatmulRhs({4, 6}, us4::DType::kBFloat16,
                              us4::DeviceType::kCpu);
    us4::Tensor bf16NeonOut({3, 6}, us4::DType::kFloat32,
                            us4::DeviceType::kCpu);
    us4::Tensor bf16ScalarLhs({3, 4}, us4::DType::kFloat32,
                              us4::DeviceType::kCpu);
    us4::Tensor bf16ScalarRhs({4, 6}, us4::DType::kFloat32,
                              us4::DeviceType::kCpu);
    us4::Tensor bf16ScalarOut({3, 6}, us4::DType::kFloat32,
                              us4::DeviceType::kCpu);
    const std::vector<float> bf16LhsValues = {0.125F, 0.5F,   -0.75F, 1.25F,
                                              -1.0F,  0.375F, 0.875F, -0.5F,
                                              0.625F, -0.25F, 1.5F,   0.75F};
    const std::vector<float> bf16RhsValues = {
        0.25F, -0.5F, 0.75F,  1.0F,    -0.25F, 0.5F,    -0.75F, 0.125F,
        0.5F,  -1.0F, 0.25F,  0.875F,  1.125F, -0.625F, 0.375F, 0.25F,
        -0.5F, 0.75F, 0.625F, -0.125F, 1.0F,   -0.75F,  0.5F,   -0.25F};
    FillHalfTensor(bf16MatmulLhs, bf16LhsValues, true);
    FillHalfTensor(bf16MatmulRhs, bf16RhsValues, true);
    ok &= Expect(FillHalfReferenceTensor(bf16ScalarLhs, bf16LhsValues, true),
                 "scalar matmul should receive bf16-rounded lhs reference");
    ok &= Expect(FillHalfReferenceTensor(bf16ScalarRhs, bf16RhsValues, true),
                 "scalar matmul should receive bf16-rounded rhs reference");
    ok &= Expect(
        us4::NeonMatmul(bf16MatmulLhs, bf16MatmulRhs, bf16NeonOut, nullptr),
        "neon matmul should execute bf16 inputs");
    ok &= Expect(
        us4::ScalarMatmul(bf16ScalarLhs, bf16ScalarRhs, bf16ScalarOut, nullptr),
        "scalar matmul should provide bf16 reference");
    const float *bf16NeonValues = bf16NeonOut.DataAsFloat32();
    const float *bf16ScalarValues = bf16ScalarOut.DataAsFloat32();
    bool bf16Matches = bf16NeonValues != nullptr && bf16ScalarValues != nullptr;
    for (std::size_t index = 0; bf16Matches && index < 18U; ++index) {
      const float diff = bf16NeonValues[index] - bf16ScalarValues[index];
      bf16Matches = std::abs(diff) <= 1e-2F;
    }
    ok &= Expect(bf16Matches,
                 "neon matmul should match scalar outputs for bf16 inputs");

    us4::HardwareProbeResult narrowNeonProbe = neonProbe;
    narrowNeonProbe.neonVectorBits = 64;
    const us4::QwenAdapter qwenAdapter;
    const us4::BackendSelection narrowSelection = us4::SelectBackend(
        narrowNeonProbe, us4::RuntimeMode::kMicroPlus, qwenAdapter);
    ok &= Expect(narrowSelection.selected == us4::BackendType::kScalarCpu,
                 "narrow neon vectors should fall back to scalar");

    us4::Tensor matmulLhs({2, 3}, us4::DType::kFloat32, us4::DeviceType::kCpu);
    us4::Tensor matmulRhs({3, 4}, us4::DType::kFloat32, us4::DeviceType::kCpu);
    us4::Tensor matmulOut({2, 4}, us4::DType::kFloat32, us4::DeviceType::kCpu);
    float *lhsData = matmulLhs.MutableDataAsFloat32();
    float *rhsData = matmulRhs.MutableDataAsFloat32();
    lhsData[0] = 1.0F;
    lhsData[1] = 2.0F;
    lhsData[2] = 3.0F;
    lhsData[3] = 4.0F;
    lhsData[4] = 5.0F;
    lhsData[5] = 6.0F;
    rhsData[0] = 1.0F;
    rhsData[1] = 0.0F;
    rhsData[2] = 2.0F;
    rhsData[3] = 1.0F;
    rhsData[4] = 0.0F;
    rhsData[5] = 1.0F;
    rhsData[6] = 3.0F;
    rhsData[7] = 0.0F;
    rhsData[8] = 1.0F;
    rhsData[9] = 1.0F;
    rhsData[10] = 0.0F;
    rhsData[11] = 2.0F;
    ok &= Expect(us4::NeonMatmul(matmulLhs, matmulRhs, matmulOut, nullptr),
                 "neon matmul should execute fp32 fast path");
    const float *matmulValues = matmulOut.DataAsFloat32();
    ok &= Expect(matmulValues != nullptr && matmulValues[6] == 23.0F,
                 "neon matmul should preserve expected fp32 result");

    us4::Tensor tailLhs({3, 7}, us4::DType::kFloat32, us4::DeviceType::kCpu);
    us4::Tensor tailRhs({7, 6}, us4::DType::kFloat32, us4::DeviceType::kCpu);
    us4::Tensor tailNeon({3, 6}, us4::DType::kFloat32, us4::DeviceType::kCpu);
    us4::Tensor tailScalar({3, 6}, us4::DType::kFloat32, us4::DeviceType::kCpu);
    float *tailLhsData = tailLhs.MutableDataAsFloat32();
    float *tailRhsData = tailRhs.MutableDataAsFloat32();
    for (std::size_t index = 0; index < 21U; ++index) {
      tailLhsData[index] = static_cast<float>((index % 5U) - 2U);
    }
    for (std::size_t index = 0; index < 42U; ++index) {
      tailRhsData[index] = static_cast<float>((index % 7U) - 3U) * 0.5F;
    }
    ok &= Expect(us4::NeonMatmul(tailLhs, tailRhs, tailNeon, nullptr),
                 "neon matmul should handle tail columns");
    ok &= Expect(us4::ScalarMatmul(tailLhs, tailRhs, tailScalar, nullptr),
                 "scalar matmul should provide tail-column reference");
    const float *tailNeonValues = tailNeon.DataAsFloat32();
    const float *tailScalarValues = tailScalar.DataAsFloat32();
    bool tailMatches = tailNeonValues != nullptr && tailScalarValues != nullptr;
    for (std::size_t index = 0; tailMatches && index < 18U; ++index) {
      tailMatches = tailNeonValues[index] == tailScalarValues[index];
    }
    ok &= Expect(tailMatches,
                 "neon matmul should match scalar results for tail columns");

    us4::Tensor int8Weights({4}, us4::DType::kInt8, us4::DeviceType::kCpu);
    us4::Tensor int8Output({4}, us4::DType::kFloat32, us4::DeviceType::kCpu);
    auto *int8Bytes =
        reinterpret_cast<std::int8_t *>(int8Weights.MutableData());
    int8Bytes[0] = 4;
    int8Bytes[1] = -2;
    int8Bytes[2] = 3;
    int8Bytes[3] = -1;
    std::string error;
    ok &= Expect(us4::DequantizeInt8Groups(int8Weights, 2, {0.5F, 0.25F},
                                           int8Output, &error),
                 "neon int8 dequant should succeed for 2 groups");
    const float *int8Values = int8Output.DataAsFloat32();
    ok &= Expect(int8Values != nullptr && int8Values[2] == 0.75F,
                 "neon int8 dequant should scale the second group");

    us4::Tensor int4Weights({8}, us4::DType::kInt4, us4::DeviceType::kCpu);
    us4::Tensor int4Output({8}, us4::DType::kFloat32, us4::DeviceType::kCpu);
    auto *int4Bytes =
        reinterpret_cast<std::uint8_t *>(int4Weights.MutableData());
    int4Bytes[0] = 0x2F;
    int4Bytes[1] = 0x91;
    int4Bytes[2] = 0x47;
    int4Bytes[3] = 0x8C;
    ok &= Expect(us4::DequantizeInt4Groups(int4Weights, 8, 4, {0.5F, 0.25F},
                                           int4Output, &error),
                 "neon int4 dequant should unpack signed nibbles");
    const float *int4Values = int4Output.DataAsFloat32();
    ok &= Expect(int4Values != nullptr && int4Values[7] == -2.0F,
                 "neon int4 dequant should preserve tail values");
  }

  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
