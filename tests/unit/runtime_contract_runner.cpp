#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <set>
#include <string_view>

#include "adapters/adapter_registry.h"
#include "adapters/llama/llama_adapter.h"
#include "adapters/qwen/qwen_adapter.h"
#include "core/backend_selector.h"
#include "core/model_asset.h"
#include "core/runtime_context.h"
#include "kv/kv_pager.h"
#include "kv/prefix_cache.h"
#include "kv/summarizer.h"
#include "memory/unified_allocator.h"
#include "metal/command_queue.h"
#include "mlx/mlx_bridge.h"
#include "moe/expert_pager.h"
#include "moe/router.h"
#include "sprint_01_contract_placeholders.h"

namespace {

bool Expect(const bool condition, const char* message) {
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
  probe.hasNeon = true;
  probe.recommendedMode = us4::RuntimeMode::kDegraded;
  return probe;
}

std::filesystem::path RepoRoot() {
#ifdef US4_SOURCE_DIR
  return std::filesystem::path(US4_SOURCE_DIR);
#else
  return std::filesystem::path(__FILE__).parent_path().parent_path().parent_path();
#endif
}

}  // namespace

int main() {
  bool ok = true;

  {
    constexpr auto& modes = sprint_01_contract::kCanonicalRuntimeModes;
    ok &= Expect(modes.size() == 7U, "runtime mode contract should keep 7 canonical modes");
    std::set<int> seen;
    for (const auto mode : modes) {
      ok &= Expect(seen.insert(static_cast<int>(mode)).second, "runtime mode values should stay unique");
    }
  }

  {
    constexpr auto& responsibilities = sprint_01_contract::kProbeResponsibilities;
    ok &= Expect(responsibilities.size() == 5U, "hardware probe contract should keep 5 responsibilities");
    std::set<std::string_view> seen;
    for (const auto& responsibility : responsibilities) {
      ok &= Expect(seen.insert(responsibility.name).second, "hardware responsibilities should stay unique");
      ok &= Expect(!responsibility.description.empty(), "hardware responsibility description should not be empty");
    }
  }

  {
    constexpr auto& categories = sprint_01_contract::kTelemetryCategories;
    ok &= Expect(categories.size() == 6U, "telemetry contract should keep 6 categories");
  }

  {
    const auto cpu = us4::ParseBackendType("CPU");
    const auto metal = us4::ParseBackendType("metal");
    const auto invalid = us4::ParseBackendType("cuda");
    ok &= Expect(cpu.has_value() && *cpu == us4::BackendType::kScalarCpu, "cpu alias should parse");
    ok &= Expect(metal.has_value() && *metal == us4::BackendType::kMetal, "metal alias should parse");
    ok &= Expect(!invalid.has_value(), "invalid backend should not parse");

    us4::HardwareProbeResult probe = MakeProbe();
    probe.hasMetal = true;
    probe.hasMlx = true;
    probe.hasNeon = true;
    const us4::LlamaAdapter llama;
    const us4::BackendSelection autoSelection =
        us4::SelectBackend(probe, us4::RuntimeMode::kBalancedPlus, llama);
    ok &= Expect(autoSelection.selected == us4::BackendType::kMetal, "auto backend should prefer metal");

    const us4::QwenAdapter qwen;
    const us4::BackendSelection fallbackSelection =
        us4::SelectBackend(MakeProbe(), us4::RuntimeMode::kDegraded, qwen, us4::BackendType::kMetal);
    ok &= Expect(fallbackSelection.selected == us4::BackendType::kNeon, "fallback should pick neon on arm64");
    ok &= Expect(fallbackSelection.fellBack, "requested unavailable backend should mark fallback");
  }

  {
    us4::ModelAsset asset;
    std::string error;
    const auto manifest = RepoRoot() / "tests" / "fixtures" / "models" / "qwen-0.5b" / "model.us4manifest";
    ok &= Expect(us4::LoadModelAsset(manifest, asset, &error), "qwen manifest should load");
    ok &= Expect(asset.family == "qwen", "qwen manifest should preserve family");
    ok &= Expect(asset.modelName == "qwen-0.5b-fixture", "qwen manifest should preserve model name");
  }

  {
    us4::UnifiedAllocator allocator;
    allocator.Allocate(32, false);
    const auto shared = allocator.Allocate(128, true);
    ok &= Expect(allocator.SharedAllocationCount() == 1U, "allocator should count unified-shared allocations");
    ok &= Expect(shared->visibility == us4::AllocationVisibility::kUnifiedShared,
                 "shared allocation should be tagged unified-shared");

    us4::HardwareProbeResult probe = MakeProbe();
    probe.hasMetal = true;
    probe.hasMlx = true;
    us4::RuntimeContext context(probe);
    ok &= Expect(context.metalQueue().Available(), "metal queue should be available on metal probe");
    ok &= Expect(context.mlxBridge().Available(), "mlx bridge should be available on mlx probe");
    ok &= Expect(context.metalQueue().Dispatch(us4::MetalKernelKind::kSoftmax, 2, 64, shared),
                 "metal queue should record dispatch contract");
    ok &= Expect(context.metalQueue().DispatchCount() == 1U, "metal queue should count dispatches");
    ok &= Expect(context.mlxBridge().BuildDensePlan("llama", 32, shared), "mlx bridge should build dense plan");
    ok &= Expect(context.mlxBridge().EvaluateLastPlan(), "mlx bridge should evaluate last plan");
  }

  {
    us4::PrefixCache cache;
    cache.Retain("hello");
    cache.Retain("hello");
    const auto retained = cache.Lookup("hello");
    ok &= Expect(retained.has_value() && retained->refCount == 2U, "prefix cache should count retains");
    cache.Release("hello");
    cache.Release("hello");
    ok &= Expect(!cache.Lookup("hello").has_value(), "prefix cache should erase on last release");

    us4::KvPager pager(1);
    pager.Append("prompt-a", {1.0F, 2.0F});
    pager.Append("prompt-b", {3.0F, 4.0F});
    const auto page = pager.Lookup("prompt-a");
    ok &= Expect(page.has_value(), "kv pager should find stored page");
    ok &= Expect(page->hitCount >= 1U, "kv pager lookup should increase hit count");

    us4::Summarizer summarizer;
    const auto summary = summarizer.Summarize({2.0F, 4.0F, 6.0F});
    ok &= Expect(summary.size() == 1U && summary[0] == 4.0F, "summarizer should produce arithmetic mean");
  }

  {
    us4::Router router;
    const auto topk = router.TopK({0.1F, 0.8F, 0.4F, 0.7F}, 2);
    ok &= Expect(topk.size() == 2U, "router should clamp top-k to requested size");
    ok &= Expect(topk[0].expert == 1U, "router should sort highest score first");

    us4::ExpertPager pager(2);
    pager.Touch("expert-a");
    pager.Touch("expert-b");
    pager.Touch("expert-a");
    pager.Touch("expert-c");
    ok &= Expect(pager.ResidentCount() == 2U, "expert pager should enforce resident limit");
    ok &= Expect(pager.IsResident("expert-a"), "expert pager should keep hot expert resident");
  }

  {
    const us4::IUS4V6Adapter* qwen = us4::FindAdapterByModel("QWEN-0.5B");
    const us4::IUS4V6Adapter* llama = us4::FindAdapterByModel("llama-3.1-8b");
    const us4::IUS4V6Adapter* ternary = us4::FindAdapterByModel("pt-bitnet-ternary-2b");
    ok &= Expect(qwen != nullptr, "registry should find qwen by model");
    ok &= Expect(llama != nullptr, "registry should find llama by model");
    ok &= Expect(ternary != nullptr, "registry should find ternary by exact model");
    ok &= Expect(ternary != nullptr && ternary->Family() == "ternary", "registry should not misroute ternary to bitnet");

    us4::ModelAsset asset;
    std::string error;
    const auto manifest = RepoRoot() / "tests" / "fixtures" / "models" / "qwen-0.5b" / "model.us4manifest";
    ok &= Expect(us4::LoadModelAsset(manifest, asset, &error), "adapter generation manifest should load");
    us4::RuntimeContext context(MakeProbe());
    qwen->ConfigureRuntime(context);
    const us4::GenerationResult result = qwen->Generate(
        {.prompt = "Hi, US4!", .maxTokens = 4, .asset = &asset, .requestedBackend = us4::BackendType::kMetal},
        context);
    ok &= Expect(result.modelName == "qwen-0.5b-fixture", "generation should surface manifest model name");
    ok &= Expect(result.backendReason == "requested-backend-unavailable", "generation should expose fallback reason");
    ok &= Expect(result.fellBack, "generation should mark backend fallback");
    ok &= Expect(result.sharedAllocations == 0U, "scalar fallback should not record shared allocations");
    ok &= Expect(result.metalDispatches == 0U, "scalar fallback should not record metal dispatches");

    us4::HardwareProbeResult appleProbe = MakeProbe();
    appleProbe.hasMetal = true;
    appleProbe.hasMlx = true;
    us4::RuntimeContext acceleratedContext(appleProbe);
    const us4::GenerationResult llamaResult =
        llama->Generate({.prompt = "metal path", .maxTokens = 4, .requestedBackend = us4::BackendType::kMetal},
                        acceleratedContext);
    ok &= Expect(llamaResult.backend == "metal", "llama should keep metal backend when available");
    ok &= Expect(acceleratedContext.metalQueue().DispatchCount() == 1U,
                 "llama generation should touch the metal scaffold");
    ok &= Expect(acceleratedContext.allocator().SharedAllocationCount() == 1U,
                 "metal scaffold should allocate unified-shared memory");
    ok &= Expect(llamaResult.sharedAllocations == 1U, "result should surface shared allocation count");
    ok &= Expect(llamaResult.metalDispatches == 1U, "result should surface metal dispatch count");
  }

  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
