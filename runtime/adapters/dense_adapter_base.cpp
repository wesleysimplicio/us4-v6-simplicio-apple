#include "adapters/dense_adapter_base.h"

#include <cctype>
#include <cstdint>
#include <functional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "core/model_asset.h"
#include "core/tensor.h"
#include "cpu/scalar_attention.h"
#include "cpu/scalar_matmul.h"

namespace us4 {

namespace {

constexpr std::size_t kHiddenSize = 8;

bool IsStandalonePunctuation(const std::string_view token) {
  return token == "." || token == "," || token == "!" || token == "?" || token == ":" || token == ";";
}

std::string NormalizeToken(const std::string_view token) {
  std::string normalized;
  normalized.reserve(token.size());
  for (const char ch : token) {
    if (std::isalnum(static_cast<unsigned char>(ch)) != 0 || ch == '-' || ch == '_' || ch == '.') {
      normalized.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    }
  }
  return normalized;
}

float DeterministicValue(const std::uint32_t seed, const std::uint32_t a, const std::uint32_t b) {
  std::uint32_t value = seed;
  value ^= 0x9E3779B9U + a + (value << 6U) + (value >> 2U);
  value ^= 0x85EBCA6BU + b + (value << 6U) + (value >> 2U);
  const std::uint32_t bucket = value % 2001U;
  return static_cast<float>(bucket) / 1000.0F - 1.0F;
}

void CopyVectorToTensor(const std::vector<float>& source, Tensor& tensor) {
  float* target = tensor.MutableDataAsFloat32();
  for (std::size_t index = 0; index < source.size(); ++index) {
    target[index] = source[index];
  }
}

void RecordBackendScaffold(const IUS4V6Adapter& adapter,
                           const BackendSelection& backendSelection,
                           const GenerationRequest& request,
                           const RuntimeContext& context) {
  RuntimeContext& mutableContext = const_cast<RuntimeContext&>(context);
  const bool needsSharedAllocation =
      backendSelection.selected == BackendType::kMetal || backendSelection.selected == BackendType::kMlx;
  const auto allocation =
      mutableContext.allocator().Allocate(std::max<std::size_t>(request.maxTokens, 1U) * kHiddenSize * sizeof(float),
                                          needsSharedAllocation);

  switch (backendSelection.selected) {
    case BackendType::kMetal:
      (void)mutableContext.metalQueue().Dispatch(MetalKernelKind::kMatmul,
                                                 std::max<std::size_t>(request.maxTokens, 1U),
                                                 kHiddenSize,
                                                 allocation);
      break;
    case BackendType::kMlx:
      if (mutableContext.mlxBridge().BuildDensePlan(adapter.Family(), std::max<std::size_t>(request.maxTokens, 1U),
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

}  // namespace

DenseAdapterBase::DenseAdapterBase(std::string family, std::string modelName)
    : family_(std::move(family)), model_name_(std::move(modelName)) {}

std::string_view DenseAdapterBase::Family() const { return family_; }

std::string_view DenseAdapterBase::ModelName() const { return model_name_; }

ArchitectureType DenseAdapterBase::Architecture() const { return ArchitectureType::kDense; }

bool DenseAdapterBase::SupportsMoe() const { return false; }

bool DenseAdapterBase::SupportsMlxBackend() const { return false; }

bool DenseAdapterBase::SupportsSpeculativeDecoding() const { return false; }

bool DenseAdapterBase::SupportsPromptRun() const { return true; }

RuntimeMode DenseAdapterBase::MinimumMode() const { return RuntimeMode::kNano; }

RuntimeMode DenseAdapterBase::RecommendedMode(const HardwareProbeResult& hardware) const {
  return hardware.recommendedMode;
}

void DenseAdapterBase::ConfigureRuntime(RuntimeContext& context) const {
  context.SetMode(MaxRuntimeMode(RecommendedMode(context.hardware()), MinimumMode()));
}

std::vector<std::string> DenseAdapterBase::Tokenize(const std::string_view text) const {
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

    if (rawChar == '.' || rawChar == ',' || rawChar == '!' || rawChar == '?' || rawChar == ':' || rawChar == ';') {
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

GenerationResult DenseAdapterBase::Generate(const GenerationRequest& request, const RuntimeContext& context) const {
  const BackendSelection backendSelection =
      SelectBackend(context.hardware(), context.mode(), *this, request.requestedBackend);
  RecordBackendScaffold(*this, backendSelection, request, context);
  const std::vector<std::string> vocabulary =
      (request.asset != nullptr && !request.asset->vocabulary.empty()) ? request.asset->vocabulary : Vocabulary();
  const std::uint32_t activeSeed = (request.asset != nullptr && request.asset->seed != 0U) ? request.asset->seed : Seed();
  std::vector<std::string> promptTokens = Tokenize(request.prompt);
  if (promptTokens.empty()) {
    if (request.asset != nullptr && !request.asset->defaultPromptToken.empty()) {
      promptTokens.push_back(request.asset->defaultPromptToken);
    } else {
      promptTokens.push_back(DefaultPromptToken());
    }
  }

  std::vector<std::size_t> tokenIds;
  tokenIds.reserve(promptTokens.size() + request.maxTokens);
  for (const std::string& token : promptTokens) {
    tokenIds.push_back(TokenIdFor(token, vocabulary));
  }

  std::vector<std::string> generatedTokens;
  generatedTokens.reserve(request.maxTokens);
  for (std::size_t step = 0; step < request.maxTokens; ++step) {
    const std::size_t sequenceLength = tokenIds.size();
    Tensor key({sequenceLength, kHiddenSize}, DType::kFloat32);
    Tensor value({sequenceLength, kHiddenSize}, DType::kFloat32);
    Tensor query({1, kHiddenSize}, DType::kFloat32);
    Tensor contextTensor({1, kHiddenSize}, DType::kFloat32);
    Tensor projection({kHiddenSize, vocabulary.size()}, DType::kFloat32);
    Tensor logits({1, vocabulary.size()}, DType::kFloat32);

    std::vector<float> keyBuffer(sequenceLength * kHiddenSize, 0.0F);
    std::vector<float> valueBuffer(sequenceLength * kHiddenSize, 0.0F);
    for (std::size_t index = 0; index < sequenceLength; ++index) {
      const std::vector<float> embedding = BuildTokenEmbedding(tokenIds[index], kHiddenSize, activeSeed);
      for (std::size_t hidden = 0; hidden < kHiddenSize; ++hidden) {
        keyBuffer[index * kHiddenSize + hidden] = embedding[hidden];
        valueBuffer[index * kHiddenSize + hidden] = embedding[hidden] + static_cast<float>((index + step) % 3U) * 0.01F;
      }
    }

    CopyVectorToTensor(keyBuffer, key);
    CopyVectorToTensor(valueBuffer, value);

    std::vector<float> queryVector = BuildTokenEmbedding(tokenIds.back(), kHiddenSize, activeSeed);
    for (std::size_t hidden = 0; hidden < kHiddenSize; ++hidden) {
      queryVector[hidden] += static_cast<float>((step + hidden) % 5U) * 0.02F;
    }
    CopyVectorToTensor(queryVector, query);
    CopyVectorToTensor(BuildOutputProjection(vocabulary, kHiddenSize, activeSeed), projection);

    std::string error;
    if (!ScalarAttention(query, key, value, contextTensor, false, {}, &error)) {
      generatedTokens.push_back("attention-error");
      break;
    }
    if (!ScalarMatmul(contextTensor, projection, logits, &error)) {
      generatedTokens.push_back("matmul-error");
      break;
    }

    const float* logitData = logits.DataAsFloat32();
    std::size_t bestIndex = 0;
    float bestValue = logitData[0];
    for (std::size_t index = 1; index < vocabulary.size(); ++index) {
      const float bias = static_cast<float>(((step + 1U) * (index + 3U)) % 7U) * 0.005F;
      const float candidate = logitData[index] + bias;
      if (candidate > bestValue) {
        bestValue = candidate;
        bestIndex = index;
      }
    }

    generatedTokens.push_back(vocabulary[bestIndex]);
    tokenIds.push_back(bestIndex);
  }

  GenerationResult result;
  result.family = (request.asset != nullptr && !request.asset->family.empty()) ? request.asset->family : family_;
  result.modelName = (request.asset != nullptr && !request.asset->modelName.empty()) ? request.asset->modelName : model_name_;
  result.assetFormat = request.asset != nullptr ? std::string(ToString(request.asset->format)) : "builtin";
  result.assetPath = request.asset != nullptr ? request.asset->sourcePath.string() : "";
  result.backend = std::string(ToString(backendSelection.selected));
  result.backendReason = std::string(backendSelection.reason);
  result.promptTokens = std::move(promptTokens);
  result.generatedTokens = generatedTokens;
  result.text = JoinTokens(generatedTokens);
  result.sharedAllocations = context.allocator().SharedAllocationCount();
  result.metalDispatches = context.metalQueue().DispatchCount();
  result.mlxPlanBuilt = context.mlxBridge().LastPlan().has_value();
  result.mlxEvaluated = context.mlxBridge().LastEvaluationSucceeded();
  result.mode = context.mode();
  result.fellBack = backendSelection.fellBack;
  return result;
}

std::size_t DenseAdapterBase::TokenIdFor(const std::string_view token,
                                         const std::vector<std::string>& vocabulary) const {
  const std::string normalized = NormalizeToken(token);
  for (std::size_t index = 0; index < vocabulary.size(); ++index) {
    if (vocabulary[index] == normalized || vocabulary[index] == token) {
      return index;
    }
  }

  if (vocabulary.empty()) {
    return 0;
  }

  return std::hash<std::string>{}(normalized.empty() ? std::string(token) : normalized) % vocabulary.size();
}

std::vector<float> DenseAdapterBase::BuildTokenEmbedding(const std::size_t tokenId,
                                                         const std::size_t hiddenSize,
                                                         const std::uint32_t seed) const {
  std::vector<float> embedding(hiddenSize, 0.0F);
  for (std::size_t hidden = 0; hidden < hiddenSize; ++hidden) {
    embedding[hidden] = DeterministicValue(seed, static_cast<std::uint32_t>(tokenId + 1U),
                                           static_cast<std::uint32_t>(hidden + 1U));
  }
  return embedding;
}

std::vector<float> DenseAdapterBase::BuildOutputProjection(const std::vector<std::string>& vocabulary,
                                                           const std::size_t hiddenSize,
                                                           const std::uint32_t seed) const {
  std::vector<float> projection(hiddenSize * vocabulary.size(), 0.0F);
  for (std::size_t hidden = 0; hidden < hiddenSize; ++hidden) {
    for (std::size_t token = 0; token < vocabulary.size(); ++token) {
      projection[hidden * vocabulary.size() + token] = DeterministicValue(
          seed + 17U, static_cast<std::uint32_t>(hidden + 1U), static_cast<std::uint32_t>(token + 1U));
    }
  }
  return projection;
}

std::string DenseAdapterBase::JoinTokens(const std::vector<std::string>& tokens) const {
  std::ostringstream stream;
  bool first = true;
  for (const std::string& token : tokens) {
    if (!first && !IsStandalonePunctuation(token)) {
      stream << ' ';
    }
    stream << token;
    first = false;
  }
  return stream.str();
}

}  // namespace us4
