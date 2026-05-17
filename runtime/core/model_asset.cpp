#include "core/model_asset.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <unordered_map>

namespace us4 {

namespace {

bool WriteError(std::string *error, const std::string &message) {
  if (error != nullptr) {
    *error = message;
  }
  return false;
}

std::string Trim(const std::string &value) {
  const auto begin =
      std::find_if_not(value.begin(), value.end(),
                       [](unsigned char ch) { return std::isspace(ch) != 0; });
  const auto end =
      std::find_if_not(value.rbegin(), value.rend(), [](unsigned char ch) {
        return std::isspace(ch) != 0;
      }).base();
  if (begin >= end) {
    return {};
  }
  return std::string(begin, end);
}

std::string ToLower(std::string value) {
  std::transform(
      value.begin(), value.end(), value.begin(),
      [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  return value;
}

DType ParseDType(const std::string &value) {
  const std::string normalized = ToLower(value);
  if (normalized == "fp16") {
    return DType::kFloat16;
  }
  if (normalized == "bf16") {
    return DType::kBFloat16;
  }
  if (normalized == "int8") {
    return DType::kInt8;
  }
  if (normalized == "int4") {
    return DType::kInt4;
  }
  return DType::kFloat32;
}

std::vector<std::string> SplitCsv(const std::string &value) {
  std::vector<std::string> parts;
  std::stringstream stream(value);
  std::string item;
  while (std::getline(stream, item, ',')) {
    const std::string trimmed = Trim(item);
    if (!trimmed.empty()) {
      parts.push_back(trimmed);
    }
  }
  return parts;
}

bool LoadFixtureManifest(const std::filesystem::path &path, ModelAsset &asset,
                         std::string *error) {
  std::ifstream stream(path);
  if (!stream.is_open()) {
    return WriteError(error, "unable to open model manifest: " + path.string());
  }

  std::unordered_map<std::string, std::string> values;
  std::string line;
  while (std::getline(stream, line)) {
    const std::string trimmed = Trim(line);
    if (trimmed.empty() || trimmed[0] == '#') {
      continue;
    }

    const std::size_t eq = trimmed.find('=');
    if (eq == std::string::npos) {
      return WriteError(error, "invalid manifest line: " + trimmed);
    }

    values.emplace(ToLower(Trim(trimmed.substr(0, eq))),
                   Trim(trimmed.substr(eq + 1)));
  }

  asset.format = ModelFormat::kFixtureManifest;
  asset.family = values["family"];
  asset.modelName = values["model_name"];
  asset.weightDType = ParseDType(values["weight_dtype"]);
  asset.seed = values.contains("seed")
                   ? static_cast<std::uint32_t>(std::stoul(values["seed"]))
                   : 0U;
  asset.vocabulary = SplitCsv(values["vocabulary"]);
  asset.defaultPromptToken = values["default_prompt_token"];
  asset.sourcePath = path;
  asset.metadata = values;

  if (asset.family.empty() || asset.modelName.empty()) {
    return WriteError(error, "manifest must define family and model_name");
  }
  if (asset.vocabulary.empty()) {
    return WriteError(error, "manifest must define vocabulary");
  }
  if (asset.defaultPromptToken.empty()) {
    asset.defaultPromptToken = asset.vocabulary.front();
  }
  return true;
}

void HydrateFromSiblingManifest(const std::filesystem::path &assetPath,
                                ModelAsset &asset) {
  const std::filesystem::path siblingManifest =
      assetPath.parent_path() / "model.us4manifest";
  if (!std::filesystem::exists(siblingManifest)) {
    return;
  }

  ModelAsset manifestAsset;
  std::string ignoredError;
  if (!LoadFixtureManifest(siblingManifest, manifestAsset, &ignoredError)) {
    return;
  }

  if (asset.family.empty()) {
    asset.family = manifestAsset.family;
  }
  asset.weightDType = manifestAsset.weightDType;
  asset.seed = manifestAsset.seed;
  asset.vocabulary = manifestAsset.vocabulary;
  asset.defaultPromptToken = manifestAsset.defaultPromptToken;
  asset.metadata = manifestAsset.metadata;

  const std::filesystem::path tokenizerPath =
      assetPath.parent_path() / "tokenizer.json";
  if (std::filesystem::exists(tokenizerPath)) {
    asset.metadata["tokenizer_json"] = tokenizerPath.string();
  }
}

std::string InferFamilyFromStem(const std::string &stem) {
  const std::string normalized = ToLower(stem);
  if (normalized.find("qwen") != std::string::npos) {
    return "qwen";
  }
  if (normalized.find("gemma") != std::string::npos) {
    return "gemma";
  }
  if (normalized.find("llama") != std::string::npos) {
    return "llama";
  }
  if (normalized.find("ternary") != std::string::npos) {
    return "ternary";
  }
  if (normalized.find("bitnet") != std::string::npos) {
    return "bitnet";
  }
  if (normalized.find("deepseek") != std::string::npos) {
    return "deepseek";
  }
  if (normalized.find("kimi") != std::string::npos) {
    return "kimi";
  }
  return {};
}

} // namespace

std::string_view ToString(const ModelFormat format) {
  switch (format) {
  case ModelFormat::kBuiltin:
    return "builtin";
  case ModelFormat::kFixtureManifest:
    return "fixture-manifest";
  case ModelFormat::kGguf:
    return "gguf";
  case ModelFormat::kSafetensors:
    return "safetensors";
  case ModelFormat::kUnknown:
    return "unknown";
  }
  return "unknown";
}

bool LoadModelAsset(const std::filesystem::path &path, ModelAsset &asset,
                    std::string *error) {
  std::filesystem::path resolved = path;
  if (std::filesystem::is_directory(path)) {
    resolved /= "model.us4manifest";
  }

  const std::string extension = ToLower(resolved.extension().string());
  if (extension == ".us4manifest") {
    return LoadFixtureManifest(resolved, asset, error);
  }

  if (!std::filesystem::exists(resolved)) {
    return WriteError(error,
                      "model asset path does not exist: " + resolved.string());
  }

  asset = {};
  asset.sourcePath = resolved;
  asset.modelName = resolved.stem().string();
  asset.family = InferFamilyFromStem(asset.modelName);
  if (extension == ".gguf") {
    asset.format = ModelFormat::kGguf;
    asset.weightDType = DType::kFloat16;
    HydrateFromSiblingManifest(resolved, asset);
    return true;
  }
  if (extension == ".safetensors") {
    asset.format = ModelFormat::kSafetensors;
    asset.weightDType = DType::kFloat16;
    HydrateFromSiblingManifest(resolved, asset);
    return true;
  }

  return WriteError(error,
                    "unsupported model asset format: " + resolved.string());
}

} // namespace us4
