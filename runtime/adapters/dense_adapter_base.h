#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "core/backend_selector.h"
#include "core/ius4v6_adapter.h"
#include "core/tensor.h"

namespace us4 {

class DenseAdapterBase : public IUS4V6Adapter {
public:
  DenseAdapterBase(std::string family, std::string modelName);

  std::string_view Family() const override;
  std::string_view ModelName() const override;
  ArchitectureType Architecture() const override;

  bool SupportsMoe() const override;
  bool SupportsMlxBackend() const override;
  bool SupportsSpeculativeDecoding() const override;
  bool SupportsPromptRun() const override;

  RuntimeMode MinimumMode() const override;
  RuntimeMode
  RecommendedMode(const HardwareProbeResult &hardware) const override;
  void ConfigureRuntime(RuntimeContext &context) const override;

  std::vector<std::string> Tokenize(std::string_view text) const override;
  GenerationResult Generate(const GenerationRequest &request,
                            const RuntimeContext &context) const override;

protected:
  virtual std::uint32_t Seed() const = 0;
  virtual std::vector<std::string> Vocabulary() const = 0;
  virtual std::string DefaultPromptToken() const = 0;
  std::size_t TokenIdFor(std::string_view token,
                         const std::vector<std::string> &vocabulary) const;
  std::vector<float> BuildTokenEmbedding(std::size_t tokenId,
                                         std::size_t hiddenSize,
                                         std::uint32_t seed) const;
  std::vector<float>
  BuildOutputProjection(const std::vector<std::string> &vocabulary,
                        std::size_t hiddenSize, std::uint32_t seed) const;
  std::string JoinTokens(const std::vector<std::string> &tokens) const;
  std::string BuildPromptCacheKeyForFamily(
      std::uint32_t seed, const std::vector<std::string> &promptTokens) const;
  void CopyVectorToTensorValues(const std::vector<float> &source,
                                Tensor &tensor) const;
  bool MaterializeProjectionForAsset(const std::vector<float> &source,
                                     const std::vector<std::size_t> &shape,
                                     const ModelAsset *asset,
                                     Tensor &projection,
                                     std::string *error) const;
  std::string ResolveDequantPathForAsset(const ModelAsset *asset) const;
  bool TryRestorePromptKvFromColdStore(RuntimeContext &context,
                                       const std::string &prefixKey,
                                       std::size_t rowWidth,
                                       std::size_t rowCount,
                                       std::vector<float> &keyBuffer,
                                       std::vector<float> &valueBuffer) const;
  std::size_t MaybeCompactPromptKv(RuntimeContext &context,
                                   const std::string &prefixKey,
                                   std::vector<float> &keyBuffer,
                                   std::vector<float> &valueBuffer,
                                   std::size_t rowWidth) const;
  GenerationResult FinalizeGenerationResult(
      const GenerationRequest &request, const RuntimeContext &context,
      const BackendSelection &backendSelection,
      std::vector<std::string> promptTokens,
      std::vector<std::string> generatedTokens, bool kvCacheHit,
      bool kvRestoredFromColdStore, std::size_t kvSummaryRows,
      std::size_t planHiddenSize) const;

private:
  std::string family_;
  std::string model_name_;
};

} // namespace us4
