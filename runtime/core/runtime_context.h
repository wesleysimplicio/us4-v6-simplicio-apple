#pragma once

#include "core/backend_selector.h"
#include "core/hardware_probe.h"
#include "kv/kv_pager.h"
#include "kv/prefix_cache.h"
#include "kv/ssd_cold_store.h"
#include "kv/summarizer.h"
#include "memory/unified_allocator.h"
#include "metal/command_queue.h"
#include "mlx/mlx_bridge.h"
#include "moe/expert_pager.h"
#include "moe/router.h"

namespace us4 {

class RuntimeContext {
 public:
  RuntimeContext() = default;
  explicit RuntimeContext(HardwareProbeResult probe_result);

  const HardwareProbeResult& hardware() const;
  RuntimeMode mode() const;
  BackendType backend() const;
  UnifiedAllocator& allocator();
  const UnifiedAllocator& allocator() const;
  MetalCommandQueue& metalQueue();
  const MetalCommandQueue& metalQueue() const;
  MlxBridge& mlxBridge();
  const MlxBridge& mlxBridge() const;
  KvPager& kvPager();
  PrefixCache& prefixCache();
  SsdColdStore& coldStore();
  Summarizer& summarizer();
  Router& router();
  ExpertPager& expertPager();
  void SetMode(RuntimeMode mode);
  void SetBackend(BackendType backend);

 private:
  HardwareProbeResult hardware_;
  RuntimeMode mode_ = RuntimeMode::kNano;
  BackendType backend_ = BackendType::kScalarCpu;
  UnifiedAllocator allocator_;
  MetalCommandQueue metalQueue_;
  MlxBridge mlxBridge_;
  KvPager kvPager_;
  PrefixCache prefixCache_;
  SsdColdStore coldStore_;
  Summarizer summarizer_;
  Router router_;
  ExpertPager expertPager_;
};

}  // namespace us4
