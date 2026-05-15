#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "core/hardware_probe.h"
#include "memory/unified_allocator.h"

namespace us4 {

enum class MetalKernelKind {
  kMatmul,
  kSoftmax,
  kRmsNorm,
};

struct MetalDispatchRecord {
  MetalKernelKind kernel = MetalKernelKind::kMatmul;
  std::size_t threadgroups = 0;
  std::size_t threadsPerGroup = 0;
  bool usesSharedAllocation = false;
};

std::string_view ToString(MetalKernelKind kernel);

class MetalCommandQueue {
 public:
  MetalCommandQueue() = default;
  explicit MetalCommandQueue(const HardwareProbeResult& hardware);

  bool Available() const;
  std::string_view Reason() const;
  bool Dispatch(MetalKernelKind kernel,
                std::size_t threadgroups,
                std::size_t threadsPerGroup,
                const std::shared_ptr<UnifiedAllocation>& allocation = nullptr);
  void Reset();
  std::size_t DispatchCount() const;
  const std::vector<MetalDispatchRecord>& Dispatches() const;

 private:
  bool available_ = false;
  std::string reason_ = "metal-unavailable";
  std::vector<MetalDispatchRecord> dispatches_;
};

}  // namespace us4
