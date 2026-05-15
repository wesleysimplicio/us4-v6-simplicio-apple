#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "core/hardware_probe.h"
#include "memory/unified_allocator.h"

namespace us4 {

struct MlxGraphPlan {
  std::string family;
  std::size_t tokenCount = 0;
  bool usesUnifiedAllocation = false;
};

class MlxBridge {
 public:
  MlxBridge() = default;
  explicit MlxBridge(const HardwareProbeResult& hardware);

  bool Available() const;
  std::string_view Reason() const;
  bool BuildDensePlan(std::string_view family,
                      std::size_t tokenCount,
                      const std::shared_ptr<UnifiedAllocation>& allocation = nullptr);
  bool EvaluateLastPlan();
  const std::optional<MlxGraphPlan>& LastPlan() const;
  bool LastEvaluationSucceeded() const;

 private:
  bool available_ = false;
  bool lastEvaluationSucceeded_ = false;
  std::string reason_ = "mlx-unavailable";
  std::optional<MlxGraphPlan> lastPlan_;
};

}  // namespace us4
