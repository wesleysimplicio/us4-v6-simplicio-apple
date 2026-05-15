#include "core/runtime_context.h"

namespace us4 {

RuntimeContext::RuntimeContext(HardwareProbeResult probe_result)
    : hardware_(std::move(probe_result)),
      mode_(hardware_.recommendedMode),
      metalQueue_(hardware_),
      mlxBridge_(hardware_) {}

const HardwareProbeResult& RuntimeContext::hardware() const {
  return hardware_;
}

RuntimeMode RuntimeContext::mode() const {
  return mode_;
}

BackendType RuntimeContext::backend() const {
  return backend_;
}

UnifiedAllocator& RuntimeContext::allocator() { return allocator_; }

const UnifiedAllocator& RuntimeContext::allocator() const { return allocator_; }

MetalCommandQueue& RuntimeContext::metalQueue() { return metalQueue_; }

const MetalCommandQueue& RuntimeContext::metalQueue() const { return metalQueue_; }

MlxBridge& RuntimeContext::mlxBridge() { return mlxBridge_; }

const MlxBridge& RuntimeContext::mlxBridge() const { return mlxBridge_; }

KvPager& RuntimeContext::kvPager() { return kvPager_; }

PrefixCache& RuntimeContext::prefixCache() { return prefixCache_; }

SsdColdStore& RuntimeContext::coldStore() { return coldStore_; }

Summarizer& RuntimeContext::summarizer() { return summarizer_; }

Router& RuntimeContext::router() { return router_; }

ExpertPager& RuntimeContext::expertPager() { return expertPager_; }

void RuntimeContext::SetMode(RuntimeMode mode) {
  mode_ = mode;
}

void RuntimeContext::SetBackend(BackendType backend) {
  backend_ = backend;
}

}  // namespace us4
