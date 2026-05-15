#include "mlx/mlx_bridge.h"

namespace us4 {

MlxBridge::MlxBridge(const HardwareProbeResult& hardware)
    : available_(hardware.hasMlx),
      reason_(hardware.hasMlx ? "mlx-bridge-ready" : "mlx-unavailable") {}

bool MlxBridge::Available() const { return available_; }

std::string_view MlxBridge::Reason() const { return reason_; }

bool MlxBridge::BuildDensePlan(const std::string_view family,
                               const std::size_t tokenCount,
                               const std::shared_ptr<UnifiedAllocation>& allocation) {
  if (!available_ || family.empty() || tokenCount == 0) {
    return false;
  }

  lastPlan_ = MlxGraphPlan{
      .family = std::string(family),
      .tokenCount = tokenCount,
      .usesUnifiedAllocation = allocation != nullptr && allocation->gpuVisible,
  };
  lastEvaluationSucceeded_ = false;
  reason_ = "mlx-plan-built";
  return true;
}

bool MlxBridge::EvaluateLastPlan() {
  if (!available_ || !lastPlan_.has_value()) {
    return false;
  }

  lastEvaluationSucceeded_ = true;
  reason_ = "mlx-plan-evaluated";
  return true;
}

const std::optional<MlxGraphPlan>& MlxBridge::LastPlan() const { return lastPlan_; }

bool MlxBridge::LastEvaluationSucceeded() const { return lastEvaluationSucceeded_; }

}  // namespace us4
