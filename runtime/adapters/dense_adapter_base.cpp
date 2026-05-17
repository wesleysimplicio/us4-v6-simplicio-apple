#include "adapters/dense_adapter_base.h"

#include <cctype>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "core/model_asset.h"
#include "core/tensor.h"
#include "cpu/scalar_attention.h"
#include "cpu/scalar_matmul.h"
#include "metal/dense_dispatch.h"
#include "neon/dequant_int4.h"
#include "neon/dequant_int8.h"
#include "neon/kernel_profile.h"
#include "neon/neon_attention.h"
#include "neon/neon_matmul.h"

namespace us4 {

namespace {

constexpr std::size_t kHiddenSize = 8;

bool IsStandalonePunctuation(const std::string_view token) {
  return token == "." || token == "," || token == "!" || token == "?" ||
         token == ":" || token == ";";
}

std::string NormalizeToken(const std::string_view token) {
  std::string normalized;
  normalized.reserve(token.size());
  for (const char ch : token) {
    if (std::isalnum(static_cast<unsigned char>(ch)) != 0 || ch == '-' ||
        ch == '_' || ch == '.') {
      normalized.push_back(
          static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    }
  }
  return normalized;
}

std::string BuildPromptCacheKey(const std::string_view family,
                                const std::uint32_t seed,
                                const std::vector<std::string> &promptTokens) {
  std::ostringstream stream;
  stream << family << ":" << seed << ":";
  for (const std::string &token : promptTokens) {
    stream << NormalizeToken(token) << '|';
  }
  return family.empty()
             ? "kv:anonymous:" +
                   std::to_string(std::hash<std::string>{}(stream.str()))
             : "kv:" + std::string(family) + ":" +
                   std::to_string(std::hash<std::string>{}(stream.str()));
}

float DeterministicValue(const std::uint32_t seed, const std::uint32_t a,
                         const std::uint32_t b) {
  std::uint32_t value = seed;
  value ^= 0x9E3779B9U + a + (value << 6U) + (value >> 2U);
  value ^= 0x85EBCA6BU + b + (value << 6U) + (value >> 2U);
  const std::uint32_t bucket = value % 2001U;
  return static_cast<float>(bucket) / 1000.0F - 1.0F;
}

void CopyVectorToTensor(const std::vector<float> &source, Tensor &tensor) {
  float *target = tensor.MutableDataAsFloat32();
  for (std::size_t index = 0; index < source.size(); ++index) {
    target[index] = source[index];
  }
}

void RecordBackendScaffold(const IUS4V6Adapter &adapter,
                           const BackendSelection &backendSelection,
                           const GenerationRequest &request,
                           const RuntimeContext &context) {
  RuntimeContext &mutableContext = const_cast<RuntimeContext &>(context);
  const bool needsSharedAllocation =
      backendSelection.selected == BackendType::kMetal ||
      backendSelection.selected == BackendType::kMlx;
  const auto allocation = mutableContext.allocator().Allocate(
      std::max<std::size_t>(request.maxTokens, 1U) * kHiddenSize *
          sizeof(float),
      needsSharedAllocation);

  switch (backendSelection.selected) {
  case BackendType::kMetal:
    (void)ExecuteDenseMetalDispatchPlan(
        mutableContext.metalQueue(),
        BuildDenseMetalDispatchPlan(
            std::max<std::size_t>(request.maxTokens, 1U), kHiddenSize, 16U),
        allocation);
    break;
  case BackendType::kMlx:
    if (mutableContext.mlxBridge().BuildDensePlan(
            adapter.Family(), std::max<std::size_t>(request.maxTokens, 1U),
            allocation)) {
      (void)mutableContext.mlxBridge().EvaluateLastPlan();
    }
    break;
  case BackendType::kScalarCpu:
  case BackendType::kNeon:
  case BackendType::kAne:
    break;
  }
}

std::string ResolveDequantPath(const ModelAsset *asset) {
  if (asset == nullptr) {
    return "none";
  }
  switch (asset->weightDType) {
  case DType::kInt8:
    return "groupwise-int8";
  case DType::kInt4:
    return "groupwise-int4";
  default:
    return "none";
  }
}

struct GroupwiseQuantizedProjection {
  Tensor tensor;
  std::vector<float> scales;
};

std::vector<float> BuildGroupScales(const std::vector<float> &source,
                                    const std::size_t groupSize,
                                    const float maxQuantAbs) {
  const std::size_t groupCount = (source.size() + groupSize - 1U) / groupSize;
  std::vector<float> scales(groupCount, 1.0F);
  for (std::size_t groupIndex = 0; groupIndex < groupCount; ++groupIndex) {
    const std::size_t begin = groupIndex * groupSize;
    const std::size_t end = std::min(source.size(), begin + groupSize);
    float maxAbs = 0.0F;
    for (std::size_t index = begin; index < end; ++index) {
      maxAbs = std::max(maxAbs, std::fabs(source[index]));
    }
    scales[groupIndex] = maxAbs > std::numeric_limits<float>::epsilon()
                             ? maxAbs / maxQuantAbs
                             : 1.0F;
  }
  return scales;
}

GroupwiseQuantizedProjection
QuantizeProjectionInt8(const std::vector<float> &source,
                       const std::vector<std::size_t> &shape,
                       const std::size_t groupSize) {
  GroupwiseQuantizedProjection projection{
      .tensor = Tensor(shape, DType::kInt8),
      .scales = BuildGroupScales(source, groupSize, 127.0F),
  };

  auto *bytes =
      reinterpret_cast<std::int8_t *>(projection.tensor.MutableData());
  for (std::size_t index = 0; index < source.size(); ++index) {
    const std::size_t groupIndex = index / groupSize;
    const float scale = projection.scales[groupIndex];
    const float normalized = scale > std::numeric_limits<float>::epsilon()
                                 ? source[index] / scale
                                 : 0.0F;
    const long rounded = std::lround(normalized);
    bytes[index] =
        static_cast<std::int8_t>(std::clamp<long>(rounded, -127L, 127L));
  }

  return projection;
}

std::uint8_t EncodeSignedNibble(const std::int8_t value) {
  return static_cast<std::uint8_t>(value < 0 ? value + 16 : value) & 0x0FU;
}

GroupwiseQuantizedProjection
QuantizeProjectionInt4(const std::vector<float> &source,
                       const std::vector<std::size_t> &shape,
                       const std::size_t groupSize) {
  GroupwiseQuantizedProjection projection{
      .tensor = Tensor(shape, DType::kInt4),
      .scales = BuildGroupScales(source, groupSize, 7.0F),
  };

  auto *bytes =
      reinterpret_cast<std::uint8_t *>(projection.tensor.MutableData());
  std::fill(bytes, bytes + projection.tensor.ByteSize(), 0U);
  for (std::size_t index = 0; index < source.size(); ++index) {
    const std::size_t groupIndex = index / groupSize;
    const float scale = projection.scales[groupIndex];
    const float normalized = scale > std::numeric_limits<float>::epsilon()
                                 ? source[index] / scale
                                 : 0.0F;
    const long rounded = std::lround(normalized);
    const std::int8_t clamped =
        static_cast<std::int8_t>(std::clamp<long>(rounded, -8L, 7L));
    const std::uint8_t nibble = EncodeSignedNibble(clamped);
    const std::size_t byteIndex = index / 2U;
    if (index % 2U == 0U) {
      bytes[byteIndex] =
          static_cast<std::uint8_t>((bytes[byteIndex] & 0xF0U) | nibble);
    } else {
      bytes[byteIndex] = static_cast<std::uint8_t>((bytes[byteIndex] & 0x0FU) |
                                                   (nibble << 4U));
    }
  }

  return projection;
}

bool MaterializeProjectionTensor(const std::vector<float> &source,
                                 const std::vector<std::size_t> &shape,
                                 const ModelAsset *asset, Tensor &projection,
                                 std::string *error) {
  if (asset == nullptr || asset->weightDType == DType::kFloat32 ||
      asset->weightDType == DType::kFloat16 ||
      asset->weightDType == DType::kBFloat16) {
    CopyVectorToTensor(source, projection);
    return true;
  }

  constexpr std::size_t kQuantGroupSize = 8U;
  if (asset->weightDType == DType::kInt8) {
    GroupwiseQuantizedProjection quantized =
        QuantizeProjectionInt8(source, shape, kQuantGroupSize);
    return DequantizeInt8Groups(quantized.tensor, kQuantGroupSize,
                                quantized.scales, projection, error);
  }
  if (asset->weightDType == DType::kInt4) {
    GroupwiseQuantizedProjection quantized =
        QuantizeProjectionInt4(source, shape, kQuantGroupSize);
    return DequantizeInt4Groups(quantized.tensor, source.size(),
                                kQuantGroupSize, quantized.scales, projection,
                                error);
  }

  CopyVectorToTensor(source, projection);
  return true;
}

} // namespace

DenseAdapterBase::DenseAdapterBase(std::string family, std::string modelName)
    : family_(std::move(family)), model_name_(std::move(modelName)) {}

std::string_view DenseAdapterBase::Family() const { return family_; }

std::string_view DenseAdapterBase::ModelName() const { return model_name_; }

ArchitectureType DenseAdapterBase::Architecture() const {
  return ArchitectureType::kDense;
}

bool DenseAdapterBase::SupportsMoe() const { return false; }

bool DenseAdapterBase::SupportsMlxBackend() const { return false; }

bool DenseAdapterBase::SupportsSpeculativeDecoding() const { return false; }

bool DenseAdapterBase::SupportsPromptRun() const { return true; }

RuntimeMode DenseAdapterBase::MinimumMode() const { return RuntimeMode::kNano; }

RuntimeMode
DenseAdapterBase::RecommendedMode(const HardwareProbeResult &hardware) const {
  return hardware.recommendedMode;
}

void DenseAdapterBase::ConfigureRuntime(RuntimeContext &context) const {
  context.SetMode(
      MaxRuntimeMode(RecommendedMode(context.hardware()), MinimumMode()));
}

std::vector<std::string>
DenseAdapterBase::Tokenize(const std::string_view text) const {
  std::vector<std::string> tokens;
  std::string current;

  for (const char rawChar : text) {
    const unsigned char ch = static_cast<unsigned char>(rawChar);
    if (std::isspace(ch) != 0) {
      if (!current.empty()) {
        tokens.push_back(current);
        current.clear();
      }
      continue;
    }

    if (rawChar == '.' || rawChar == ',' || rawChar == '!' || rawChar == '?' ||
        rawChar == ':' || rawChar == ';') {
      if (!current.empty()) {
        tokens.push_back(current);
        current.clear();
      }
      tokens.emplace_back(1, rawChar);
      continue;
    }

    current.push_back(static_cast<char>(std::tolower(ch)));
  }

  if (!current.empty()) {
    tokens.push_back(current);
  }

  return tokens;
}

GenerationResult
DenseAdapterBase::Generate(const GenerationRequest &request,
                           const RuntimeContext &context) const {
  RuntimeContext &mutableContext = const_cast<RuntimeContext &>(context);
  const BackendSelection backendSelection = SelectBackend(
      context.hardware(), context.mode(), *this, request.requestedBackend);
  RecordBackendScaffold(*this, backendSelection, request, context);
  const std::vector<std::string> vocabulary =
      (request.asset != nullptr && !request.asset->vocabulary.empty())
          ? request.asset->vocabulary
          : Vocabulary();
  const std::uint32_t activeSeed =
      (request.asset != nullptr && request.asset->seed != 0U)
          ? request.asset->seed
          : Seed();
  std::vector<std::string> promptTokens = Tokenize(request.prompt);
  if (promptTokens.empty()) {
    if (request.asset != nullptr &&
        !request.asset->defaultPromptToken.empty()) {
      promptTokens.push_back(request.asset->defaultPromptToken);
    } else {
      promptTokens.push_back(DefaultPromptToken());
    }
  }

  std::vector<std::size_t> tokenIds;
  tokenIds.reserve(promptTokens.size() + request.maxTokens);
  for (const std::string &token : promptTokens) {
    tokenIds.push_back(TokenIdFor(token, vocabulary));
  }

  const std::string prefixKey =
      BuildPromptCacheKeyForFamily(activeSeed, promptTokens);
  mutableContext.prefixCache().Retain(prefixKey);

  std::vector<float> keyBuffer;
  std::vector<float> valueBuffer;
  bool kvCacheHit = false;
  bool kvRestoredFromColdStore = false;
  std::size_t kvSummaryRows = 0U;
  const std::optional<KvPage> cachedPrefix =
      mutableContext.kvPager().Lookup(prefixKey);
  if (cachedPrefix.has_value() && cachedPrefix->rowWidth == kHiddenSize &&
      cachedPrefix->rowCount > 0U &&
      cachedPrefix->rowCount <= tokenIds.size() &&
      cachedPrefix->keys.size() == cachedPrefix->rowCount * kHiddenSize &&
      cachedPrefix->values.size() == cachedPrefix->rowCount * kHiddenSize) {
    keyBuffer = cachedPrefix->keys;
    valueBuffer = cachedPrefix->values;
    kvCacheHit = true;
  } else {
    if (TryRestorePromptKvFromColdStore(mutableContext, prefixKey, kHiddenSize,
                                        tokenIds.size(), keyBuffer,
                                        valueBuffer)) {
      kvCacheHit = true;
      kvRestoredFromColdStore = true;
    } else {
      keyBuffer.reserve(tokenIds.size() * kHiddenSize);
      valueBuffer.reserve(tokenIds.size() * kHiddenSize);
      for (std::size_t index = 0; index < tokenIds.size(); ++index) {
        const std::vector<float> embedding =
            BuildTokenEmbedding(tokenIds[index], kHiddenSize, activeSeed);
        for (std::size_t hidden = 0; hidden < kHiddenSize; ++hidden) {
          keyBuffer.push_back(embedding[hidden]);
          valueBuffer.push_back(embedding[hidden] +
                                static_cast<float>(index % 3U) * 0.01F);
        }
      }
    }

    kvSummaryRows = MaybeCompactPromptKv(mutableContext, prefixKey, keyBuffer,
                                         valueBuffer, kHiddenSize);
    mutableContext.kvPager().Append(prefixKey, keyBuffer, valueBuffer,
                                    kHiddenSize);
  }

  std::vector<std::string> generatedTokens;
  generatedTokens.reserve(request.maxTokens);
  for (std::size_t step = 0; step < request.maxTokens; ++step) {
    const std::size_t sequenceLength = keyBuffer.size() / kHiddenSize;
    Tensor key({sequenceLength, kHiddenSize}, DType::kFloat32);
    Tensor value({sequenceLength, kHiddenSize}, DType::kFloat32);
    Tensor query({1, kHiddenSize}, DType::kFloat32);
    Tensor contextTensor({1, kHiddenSize}, DType::kFloat32);
    Tensor projection({kHiddenSize, vocabulary.size()}, DType::kFloat32);
    Tensor logits({1, vocabulary.size()}, DType::kFloat32);

    CopyVectorToTensor(keyBuffer, key);
    CopyVectorToTensor(valueBuffer, value);

    std::vector<float> queryVector =
        BuildTokenEmbedding(tokenIds.back(), kHiddenSize, activeSeed);
    for (std::size_t hidden = 0; hidden < kHiddenSize; ++hidden) {
      queryVector[hidden] += static_cast<float>((step + hidden) % 5U) * 0.02F;
    }
    CopyVectorToTensor(queryVector, query);

    std::string error;
    const std::vector<float> outputProjection =
        BuildOutputProjection(vocabulary, kHiddenSize, activeSeed);
    if (!MaterializeProjectionTensor(outputProjection,
                                     {kHiddenSize, vocabulary.size()},
                                     request.asset, projection, &error)) {
      generatedTokens.push_back("projection-error");
      break;
    }

    const bool attentionOk =
        backendSelection.selected == BackendType::kNeon
            ? NeonAttention(query, key, value, contextTensor, false, {}, &error)
            : ScalarAttention(query, key, value, contextTensor, false, {},
                              &error);
    if (!attentionOk) {
      generatedTokens.push_back("attention-error");
      break;
    }
    const bool matmulOk =
        backendSelection.selected == BackendType::kNeon
            ? NeonMatmul(contextTensor, projection, logits, &error)
            : ScalarMatmul(contextTensor, projection, logits, &error);
    if (!matmulOk) {
      generatedTokens.push_back("matmul-error");
      break;
    }

    const float *logitData = logits.DataAsFloat32();
    std::size_t bestIndex = 0;
    float bestValue = logitData[0];
    for (std::size_t index = 1; index < vocabulary.size(); ++index) {
      const float bias =
          static_cast<float>(((step + 1U) * (index + 3U)) % 7U) * 0.005F;
      const float candidate = logitData[index] + bias;
      if (candidate > bestValue) {
        bestValue = candidate;
        bestIndex = index;
      }
    }

    generatedTokens.push_back(vocabulary[bestIndex]);
    tokenIds.push_back(bestIndex);

    const std::vector<float> nextEmbedding =
        BuildTokenEmbedding(bestIndex, kHiddenSize, activeSeed);
    for (std::size_t hidden = 0; hidden < kHiddenSize; ++hidden) {
      keyBuffer.push_back(nextEmbedding[hidden]);
      valueBuffer.push_back(nextEmbedding[hidden] +
                            static_cast<float>((sequenceLength + step) % 3U) *
                                0.01F);
    }
  }

  return FinalizeGenerationResult(
      request, context, backendSelection, std::move(promptTokens),
      std::move(generatedTokens), kvCacheHit, kvRestoredFromColdStore,
      kvSummaryRows, kHiddenSize);
}

std::size_t
DenseAdapterBase::TokenIdFor(const std::string_view token,
                             const std::vector<std::string> &vocabulary) const {
  const std::string normalized = NormalizeToken(token);
  for (std::size_t index = 0; index < vocabulary.size(); ++index) {
    if (vocabulary[index] == normalized || vocabulary[index] == token) {
      return index;
    }
  }

  if (vocabulary.empty()) {
    return 0;
  }

  return std::hash<std::string>{}(normalized.empty() ? std::string(token)
                                                     : normalized) %
         vocabulary.size();
}

std::vector<float>
DenseAdapterBase::BuildTokenEmbedding(const std::size_t tokenId,
                                      const std::size_t hiddenSize,
                                      const std::uint32_t seed) const {
  std::vector<float> embedding(hiddenSize, 0.0F);
  for (std::size_t hidden = 0; hidden < hiddenSize; ++hidden) {
    embedding[hidden] =
        DeterministicValue(seed, static_cast<std::uint32_t>(tokenId + 1U),
                           static_cast<std::uint32_t>(hidden + 1U));
  }
  return embedding;
}

std::vector<float> DenseAdapterBase::BuildOutputProjection(
    const std::vector<std::string> &vocabulary, const std::size_t hiddenSize,
    const std::uint32_t seed) const {
  std::vector<float> projection(hiddenSize * vocabulary.size(), 0.0F);
  for (std::size_t hidden = 0; hidden < hiddenSize; ++hidden) {
    for (std::size_t token = 0; token < vocabulary.size(); ++token) {
      projection[hidden * vocabulary.size() + token] = DeterministicValue(
          seed + 17U, static_cast<std::uint32_t>(hidden + 1U),
          static_cast<std::uint32_t>(token + 1U));
    }
  }
  return projection;
}

std::string
DenseAdapterBase::JoinTokens(const std::vector<std::string> &tokens) const {
  std::ostringstream stream;
  bool first = true;
  for (const std::string &token : tokens) {
    if (!first && !IsStandalonePunctuation(token)) {
      stream << ' ';
    }
    stream << token;
    first = false;
  }
  return stream.str();
}

std::string DenseAdapterBase::BuildPromptCacheKeyForFamily(
    const std::uint32_t seed,
    const std::vector<std::string> &promptTokens) const {
  return BuildPromptCacheKey(family_, seed, promptTokens);
}

void DenseAdapterBase::CopyVectorToTensorValues(
    const std::vector<float> &source, Tensor &tensor) const {
  CopyVectorToTensor(source, tensor);
}

bool DenseAdapterBase::MaterializeProjectionForAsset(
    const std::vector<float> &source, const std::vector<std::size_t> &shape,
    const ModelAsset *asset, Tensor &projection, std::string *error) const {
  return MaterializeProjectionTensor(source, shape, asset, projection, error);
}

std::string
DenseAdapterBase::ResolveDequantPathForAsset(const ModelAsset *asset) const {
  return ResolveDequantPath(asset);
}

bool DenseAdapterBase::TryRestorePromptKvFromColdStore(
    RuntimeContext &context, const std::string &prefixKey,
    const std::size_t rowWidth, const std::size_t rowCount,
    std::vector<float> &keyBuffer, std::vector<float> &valueBuffer) const {
  const std::optional<std::vector<float>> restoredKeys =
      context.coldStore().Restore(prefixKey + "-keys");
  const std::optional<std::vector<float>> restoredValues =
      context.coldStore().Restore(prefixKey + "-values");
  if (!restoredKeys.has_value() || !restoredValues.has_value()) {
    return false;
  }
  if (restoredKeys->size() != rowCount * rowWidth ||
      restoredValues->size() != rowCount * rowWidth) {
    return false;
  }
  keyBuffer = *restoredKeys;
  valueBuffer = *restoredValues;
  return true;
}

std::size_t DenseAdapterBase::MaybeCompactPromptKv(
    RuntimeContext &context, const std::string &prefixKey,
    std::vector<float> &keyBuffer, std::vector<float> &valueBuffer,
    const std::size_t rowWidth) const {
  constexpr std::size_t kKeepRecentRows = 3U;
  if (RuntimeModeRank(context.mode()) > RuntimeModeRank(RuntimeMode::kMicro)) {
    return 0U;
  }
  if (rowWidth == 0U || keyBuffer.size() != valueBuffer.size() ||
      (keyBuffer.size() % rowWidth) != 0U) {
    return 0U;
  }

  const std::size_t rowCount = keyBuffer.size() / rowWidth;
  if (rowCount <= (kKeepRecentRows + 1U)) {
    return 0U;
  }

  const std::size_t summaryInputRows = rowCount - kKeepRecentRows;
  const std::size_t summaryInputValues = summaryInputRows * rowWidth;
  const std::vector<float> summaryKeys = context.summarizer().SummarizeRows(
      {keyBuffer.begin(),
       keyBuffer.begin() + static_cast<std::ptrdiff_t>(summaryInputValues)},
      rowWidth);
  const std::vector<float> summaryValues = context.summarizer().SummarizeRows(
      {valueBuffer.begin(),
       valueBuffer.begin() + static_cast<std::ptrdiff_t>(summaryInputValues)},
      rowWidth);
  if (summaryKeys.size() != rowWidth || summaryValues.size() != rowWidth) {
    return 0U;
  }

  (void)context.coldStore().Flush(prefixKey + "-keys", keyBuffer);
  (void)context.coldStore().Flush(prefixKey + "-values", valueBuffer);

  std::vector<float> compactedKeys;
  std::vector<float> compactedValues;
  compactedKeys.reserve((kKeepRecentRows + 1U) * rowWidth);
  compactedValues.reserve((kKeepRecentRows + 1U) * rowWidth);
  compactedKeys.insert(compactedKeys.end(), summaryKeys.begin(),
                       summaryKeys.end());
  compactedValues.insert(compactedValues.end(), summaryValues.begin(),
                         summaryValues.end());

  const std::size_t recentOffset = (rowCount - kKeepRecentRows) * rowWidth;
  compactedKeys.insert(compactedKeys.end(),
                       keyBuffer.begin() +
                           static_cast<std::ptrdiff_t>(recentOffset),
                       keyBuffer.end());
  compactedValues.insert(compactedValues.end(),
                         valueBuffer.begin() +
                             static_cast<std::ptrdiff_t>(recentOffset),
                         valueBuffer.end());

  keyBuffer = std::move(compactedKeys);
  valueBuffer = std::move(compactedValues);
  return summaryInputRows - 1U;
}

GenerationResult DenseAdapterBase::FinalizeGenerationResult(
    const GenerationRequest &request, const RuntimeContext &context,
    const BackendSelection &backendSelection,
    std::vector<std::string> promptTokens,
    std::vector<std::string> generatedTokens, const bool kvCacheHit,
    const bool kvRestoredFromColdStore, const std::size_t kvSummaryRows,
    const std::size_t planHiddenSize) const {
  RuntimeContext &mutableContext = const_cast<RuntimeContext &>(context);

  GenerationResult result;
  result.family = (request.asset != nullptr && !request.asset->family.empty())
                      ? request.asset->family
                      : family_;
  result.modelName =
      (request.asset != nullptr && !request.asset->modelName.empty())
          ? request.asset->modelName
          : model_name_;
  result.assetFormat = request.asset != nullptr
                           ? std::string(ToString(request.asset->format))
                           : "builtin";
  result.assetPath =
      request.asset != nullptr ? request.asset->sourcePath.string() : "";
  result.backend = std::string(ToString(backendSelection.selected));
  result.backendReason = std::string(backendSelection.reason);
  result.promptTokens = std::move(promptTokens);
  result.generatedTokens = std::move(generatedTokens);
  result.text = JoinTokens(result.generatedTokens);
  result.sharedAllocations = context.allocator().SharedAllocationCount();
  result.metalDispatches = context.metalQueue().DispatchCount();
  result.mlxOperationCount =
      context.mlxBridge().LastPlan().has_value()
          ? context.mlxBridge().LastPlan()->operations.size()
          : 0U;
  result.kvCacheHit = kvCacheHit;
  result.kvRestoredFromColdStore = kvRestoredFromColdStore;
  result.kvPageCount = mutableContext.kvPager().PageCount();
  result.kvHotPages = mutableContext.kvPager().HotPageCount();
  result.kvWarmPages = mutableContext.kvPager().WarmPageCount();
  result.kvColdPages = mutableContext.kvPager().ColdPageCount();
  result.kvSummaryRows = kvSummaryRows;
  result.prefixCacheEntries = mutableContext.prefixCache().EntryCount();
  result.mlxPlanBuilt = context.mlxBridge().LastPlan().has_value();
  result.mlxEvaluated = context.mlxBridge().LastEvaluationSucceeded();
  result.weightDType = request.asset != nullptr
                           ? std::string(ToString(request.asset->weightDType))
                           : "fp32";
  result.dequantPath = ResolveDequantPathForAsset(request.asset);
  result.neonKernelFlavor = "none";
  if (backendSelection.selected == BackendType::kNeon) {
    const DType planDType =
        request.asset != nullptr ? request.asset->weightDType : DType::kFloat32;
    const Tensor lhs(
        {std::max<std::size_t>(request.maxTokens, 1U), planHiddenSize},
        planDType, DeviceType::kCpu);
    const Tensor rhs({planHiddenSize, planHiddenSize}, planDType,
                     DeviceType::kCpu);
    result.neonKernelFlavor = std::string(
        ToString(PlanNeonMatmul(context.hardware(), lhs, rhs).flavor));
  }
  result.metalDevice = context.metalQueue().Device().deviceName;
  result.metalQueueLabel = context.metalQueue().Device().queueLabel;
  result.mode = context.mode();
  result.fellBack = backendSelection.fellBack;
  return result;
}

} // namespace us4
