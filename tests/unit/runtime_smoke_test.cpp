#include <cstdlib>
#include <iostream>

#include "adapters/adapter_registry.h"
#include "core/backend_selector.h"
#include "core/gqa_attention.h"
#include "core/hardware_probe.h"
#include "core/rope.h"
#include "core/tensor.h"
#include "kv/kv_pager.h"
#include "kv/prefix_cache.h"
#include "kv/ssd_cold_store.h"
#include "kv/summarizer.h"
#include "memory/unified_allocator.h"
#include "metal/command_queue.h"
#include "mlx/mlx_bridge.h"
#include "moe/expert_pager.h"
#include "moe/router.h"
#include "core/runtime_context.h"
#include "core/runtime_mode.h"
#include "telemetry/telemetry_sink.h"

namespace {

bool Expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << message << "\n";
    return false;
  }
  return true;
}

}  // namespace

int main() {
  bool ok = true;

  const us4::HardwareProbeResult probe = us4::HardwareProbe::Detect();
  ok &= Expect(!probe.platform.empty(), "platform should not be empty");
  ok &= Expect(!probe.architecture.empty(), "architecture should not be empty");

  const us4::RuntimeMode expected_mode =
      us4::SelectRuntimeModeFromMemoryGiB(probe.unifiedMemoryGiB);
  ok &= Expect(expected_mode == probe.recommendedMode, "recommended mode should match memory tier");

  us4::RuntimeContext context(probe);
  ok &= Expect(context.mode() == probe.recommendedMode, "runtime context should start with recommended mode");

  context.SetMode(us4::RuntimeMode::kMicro);
  ok &= Expect(context.mode() == us4::RuntimeMode::kMicro, "runtime context should allow mode override");
  context.SetBackend(us4::BackendType::kScalarCpu);
  ok &= Expect(context.backend() == us4::BackendType::kScalarCpu, "runtime context should allow backend override");

  us4::TelemetrySink sink;
  ok &= Expect(sink.Empty(), "telemetry sink should start empty");
  sink.Record({12.5, 34.0, probe.unifiedMemoryGiB, 8, 4, context.mode()});
  ok &= Expect(!sink.Empty(), "telemetry sink should store snapshots");
  ok &= Expect(sink.Snapshots().size() == 1, "telemetry sink should keep one snapshot");

  const auto adapters = us4::ListAdapters();
  ok &= Expect(adapters.size() >= 7, "adapter registry should expose sprint 02-08 families");

  const us4::BackendSelection backend = us4::SelectBackend(probe, context.mode(), *adapters.front());
  ok &= Expect(!backend.reason.empty(), "backend selection should produce a reason");

  us4::UnifiedAllocator allocator;
  const auto allocation = allocator.Allocate(128, false);
  ok &= Expect(allocation->bytes.size() == 128, "unified allocator should reserve bytes");
  const auto sharedAllocation = allocator.Allocate(256, true);
  ok &= Expect(sharedAllocation->visibility == us4::AllocationVisibility::kUnifiedShared,
               "unified allocator should tag shared allocations");
  ok &= Expect(allocator.SharedAllocationCount() == 1, "unified allocator should count shared allocations");

  ok &= Expect(context.metalQueue().Available() == probe.hasMetal, "metal queue should mirror probe availability");
  ok &= Expect(context.mlxBridge().Available() == probe.hasMlx, "mlx bridge should mirror probe availability");
  if (probe.hasMetal) {
    ok &= Expect(context.metalQueue().Dispatch(us4::MetalKernelKind::kMatmul, 2, 32, sharedAllocation),
                 "metal queue should record dispatches when available");
  }
  if (probe.hasMlx) {
    ok &= Expect(context.mlxBridge().BuildDensePlan("qwen", 16, sharedAllocation),
                 "mlx bridge should build plan when available");
    ok &= Expect(context.mlxBridge().EvaluateLastPlan(), "mlx bridge should evaluate built plan");
  }

  us4::KvPager pager(1);
  pager.Append("prompt", {1.0F, 2.0F, 3.0F});
  ok &= Expect(pager.PageCount() == 1, "kv pager should store pages");

  us4::PrefixCache prefixCache;
  prefixCache.Retain("hello world");
  ok &= Expect(prefixCache.EntryCount() == 1, "prefix cache should retain entries");

  us4::SsdColdStore coldStore;
  ok &= Expect(coldStore.Flush("prompt", {1.0F, 2.0F}), "cold store should flush");
  ok &= Expect(coldStore.Restore("prompt").has_value(), "cold store should restore");

  us4::Summarizer summarizer;
  ok &= Expect(!summarizer.Summarize({1.0F, 3.0F, 5.0F}).empty(), "summarizer should produce a summary");

  us4::Tensor ropeTensor({1, 4}, us4::DType::kFloat32);
  float* ropeData = ropeTensor.MutableDataAsFloat32();
  ropeData[0] = 1.0F;
  ropeData[1] = 0.0F;
  ropeData[2] = 0.5F;
  ropeData[3] = 0.25F;
  us4::ApplyRopeInPlace(ropeTensor, 1, 10000.0F);
  ok &= Expect(ropeTensor.DataAsFloat32()[0] != 1.0F, "rope should rotate tensor values");

  us4::Tensor query({1, 2}, us4::DType::kFloat32);
  us4::Tensor key({1, 2}, us4::DType::kFloat32);
  us4::Tensor value({1, 2}, us4::DType::kFloat32);
  us4::Tensor out({1, 2}, us4::DType::kFloat32);
  query.MutableDataAsFloat32()[0] = 1.0F;
  query.MutableDataAsFloat32()[1] = 0.0F;
  key.MutableDataAsFloat32()[0] = 1.0F;
  key.MutableDataAsFloat32()[1] = 0.0F;
  value.MutableDataAsFloat32()[0] = 2.0F;
  value.MutableDataAsFloat32()[1] = 4.0F;
  std::string attentionError;
  ok &= Expect(us4::GqaAttention(query, key, value, 2, 1, out, &attentionError), "gqa attention should succeed");

  us4::Router router;
  ok &= Expect(router.TopK({0.1F, 0.9F, 0.4F}, 2).size() == 2, "router should return top-k experts");

  us4::ExpertPager expertPager(1);
  expertPager.Touch("expert-a");
  expertPager.Touch("expert-b");
  ok &= Expect(expertPager.ResidentCount() == 1, "expert pager should enforce resident limit");

  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
