#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "core/backend_selector.h"
#include "core/runtime_context.h"

namespace us4 {

struct ModelAsset;

enum class ArchitectureType {
  kDense,
  kMoe,
  kTernary,
  kUnknown,
};

struct GenerationRequest {
  std::string prompt;
  std::size_t maxTokens = 16;
  const ModelAsset* asset = nullptr;
  std::optional<BackendType> requestedBackend = std::nullopt;
};

struct GenerationResult {
  std::string family;
  std::string modelName;
  std::string assetFormat;
  std::string assetPath;
  std::string backend;
  std::string backendReason;
  std::vector<std::string> promptTokens;
  std::vector<std::string> generatedTokens;
  std::string text;
  std::size_t sharedAllocations = 0;
  std::size_t metalDispatches = 0;
  bool mlxPlanBuilt = false;
  bool mlxEvaluated = false;
  RuntimeMode mode = RuntimeMode::kNano;
  bool fellBack = false;
};

class IUS4V6Adapter {
 public:
  virtual ~IUS4V6Adapter() = default;

  virtual std::string_view Family() const = 0;
  virtual std::string_view ModelName() const = 0;
  virtual ArchitectureType Architecture() const = 0;

  virtual bool SupportsMoe() const = 0;
  virtual bool SupportsMlxBackend() const = 0;
  virtual bool SupportsMetalBackend() const { return false; }
  virtual bool SupportsAneBackend() const { return false; }
  virtual bool SupportsSpeculativeDecoding() const = 0;
  virtual bool SupportsPromptRun() const = 0;

  virtual RuntimeMode MinimumMode() const = 0;
  virtual RuntimeMode RecommendedMode(const HardwareProbeResult& hardware) const = 0;
  virtual void ConfigureRuntime(RuntimeContext& context) const = 0;
  virtual std::vector<std::string> Tokenize(std::string_view text) const = 0;
  virtual GenerationResult Generate(const GenerationRequest& request, const RuntimeContext& context) const = 0;
};

}  // namespace us4
