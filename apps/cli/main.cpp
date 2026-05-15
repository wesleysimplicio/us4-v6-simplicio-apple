#include <algorithm>
#include <filesystem>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>

#include "adapters/adapter_registry.h"
#include "core/hardware_probe.h"
#include "core/ius4v6_adapter.h"
#include "core/model_asset.h"
#include "core/runtime_context.h"
#include "core/runtime_mode.h"
#include "us4/version.h"

namespace {

std::string_view ArchitectureToString(const us4::ArchitectureType architecture) {
  switch (architecture) {
    case us4::ArchitectureType::kDense:
      return "dense";
    case us4::ArchitectureType::kMoe:
      return "moe";
    case us4::ArchitectureType::kTernary:
      return "ternary";
    case us4::ArchitectureType::kUnknown:
      return "unknown";
  }
  return "unknown";
}

std::string EscapeJson(const std::string_view value) {
  std::string escaped;
  escaped.reserve(value.size() + 8);
  for (const char ch : value) {
    switch (ch) {
      case '\\':
        escaped += "\\\\";
        break;
      case '"':
        escaped += "\\\"";
        break;
      case '\n':
        escaped += "\\n";
        break;
      case '\r':
        escaped += "\\r";
        break;
      case '\t':
        escaped += "\\t";
        break;
      default:
        escaped.push_back(ch);
        break;
    }
  }
  return escaped;
}

void PrintHelp() {
  std::cout
      << "US4 V6 Apple Edition CLI\n"
      << "Usage:\n"
      << "  us4-cli --version\n"
      << "  us4-cli --probe [--json]\n"
      << "  us4-cli --mode auto [--json]\n"
      << "  us4-cli list-models [--json]\n"
      << "  us4-cli run --model <name> [--model-path <path>] [--backend <scalar|neon|mlx|metal|ane>] --prompt <text> [--max-tokens N] [--json]\n";
}

void PrintProbeText(const us4::HardwareProbeResult& probe) {
  std::cout
      << "US4 V6 Apple Edition\n"
      << "version: " << us4::kUs4Version << "\n"
      << "platform: " << probe.platform << "\n"
      << "architecture: " << probe.architecture << "\n"
      << "chip: " << probe.chip << "\n"
      << "memory_gib: " << probe.unifiedMemoryGiB << "\n"
      << "is_apple_silicon: " << (probe.isAppleSilicon ? "true" : "false") << "\n"
      << "has_mlx: " << (probe.hasMlx ? "true" : "false") << "\n"
      << "has_metal: " << (probe.hasMetal ? "true" : "false") << "\n"
      << "has_neon: " << (probe.hasNeon ? "true" : "false") << "\n"
      << "has_ane: " << (probe.hasAne ? "true" : "false") << "\n"
      << "recommended_mode: " << us4::ToString(probe.recommendedMode) << "\n";
}

void PrintProbeJson(const us4::HardwareProbeResult& probe) {
  std::cout
      << "{"
      << "\"version\":\"" << EscapeJson(us4::kUs4Version) << "\","
      << "\"platform\":\"" << EscapeJson(probe.platform) << "\","
      << "\"architecture\":\"" << EscapeJson(probe.architecture) << "\","
      << "\"chip\":\"" << EscapeJson(probe.chip) << "\","
      << "\"memory_gib\":" << probe.unifiedMemoryGiB << ","
      << "\"is_apple_silicon\":" << (probe.isAppleSilicon ? "true" : "false") << ","
      << "\"has_mlx\":" << (probe.hasMlx ? "true" : "false") << ","
      << "\"has_metal\":" << (probe.hasMetal ? "true" : "false") << ","
      << "\"has_neon\":" << (probe.hasNeon ? "true" : "false") << ","
      << "\"has_ane\":" << (probe.hasAne ? "true" : "false") << ","
      << "\"recommended_mode\":\"" << us4::ToString(probe.recommendedMode) << "\""
      << "}\n";
}

void PrintRunText(const us4::GenerationResult& result) {
  std::cout
      << "family: " << result.family << "\n"
      << "model: " << result.modelName << "\n"
      << "asset_format: " << result.assetFormat << "\n"
      << "asset_path: " << (result.assetPath.empty() ? "<builtin>" : result.assetPath) << "\n"
      << "mode: " << us4::ToString(result.mode) << "\n"
      << "backend: " << result.backend << "\n"
      << "backend_reason: " << result.backendReason << "\n"
      << "fallback: " << (result.fellBack ? "true" : "false") << "\n"
      << "shared_allocations: " << result.sharedAllocations << "\n"
      << "metal_dispatches: " << result.metalDispatches << "\n"
      << "mlx_plan_built: " << (result.mlxPlanBuilt ? "true" : "false") << "\n"
      << "mlx_evaluated: " << (result.mlxEvaluated ? "true" : "false") << "\n"
      << "prompt_tokens: " << result.promptTokens.size() << "\n"
      << "generated_tokens: " << result.generatedTokens.size() << "\n"
      << "text: " << result.text << "\n";
}

void PrintRunJson(const us4::GenerationResult& result) {
  std::ostringstream promptTokens;
  std::ostringstream generatedTokens;

  for (std::size_t index = 0; index < result.promptTokens.size(); ++index) {
    if (index > 0) {
      promptTokens << ",";
    }
    promptTokens << "\"" << EscapeJson(result.promptTokens[index]) << "\"";
  }

  for (std::size_t index = 0; index < result.generatedTokens.size(); ++index) {
    if (index > 0) {
      generatedTokens << ",";
    }
    generatedTokens << "\"" << EscapeJson(result.generatedTokens[index]) << "\"";
  }

  std::cout
      << "{"
      << "\"family\":\"" << EscapeJson(result.family) << "\","
      << "\"model\":\"" << EscapeJson(result.modelName) << "\","
      << "\"asset_format\":\"" << EscapeJson(result.assetFormat) << "\","
      << "\"asset_path\":\"" << EscapeJson(result.assetPath) << "\","
      << "\"mode\":\"" << EscapeJson(us4::ToString(result.mode)) << "\","
      << "\"backend\":\"" << EscapeJson(result.backend) << "\","
      << "\"backend_reason\":\"" << EscapeJson(result.backendReason) << "\","
      << "\"fallback\":" << (result.fellBack ? "true" : "false") << ","
      << "\"shared_allocations\":" << result.sharedAllocations << ","
      << "\"metal_dispatches\":" << result.metalDispatches << ","
      << "\"mlx_plan_built\":" << (result.mlxPlanBuilt ? "true" : "false") << ","
      << "\"mlx_evaluated\":" << (result.mlxEvaluated ? "true" : "false") << ","
      << "\"prompt_tokens\":[" << promptTokens.str() << "],"
      << "\"generated_tokens\":[" << generatedTokens.str() << "],"
      << "\"text\":\"" << EscapeJson(result.text) << "\""
      << "}\n";
}

void PrintAdapterList() {
  std::cout << "Available models:\n";
  for (const us4::IUS4V6Adapter* adapter : us4::ListAdapters()) {
    std::cout
        << "  - " << adapter->ModelName()
        << " [" << adapter->Family() << "]"
        << " arch=" << ArchitectureToString(adapter->Architecture())
        << " min_mode=" << us4::ToString(adapter->MinimumMode())
        << " mlx=" << (adapter->SupportsMlxBackend() ? "true" : "false")
        << " moe=" << (adapter->SupportsMoe() ? "true" : "false")
        << "\n";
  }
}

void PrintAdapterListJson() {
  std::ostringstream models;
  bool first = true;
  for (const us4::IUS4V6Adapter* adapter : us4::ListAdapters()) {
    if (!first) {
      models << ",";
    }
    models << "{"
           << "\"family\":\"" << EscapeJson(adapter->Family()) << "\","
           << "\"model\":\"" << EscapeJson(adapter->ModelName()) << "\","
           << "\"architecture\":\"" << EscapeJson(ArchitectureToString(adapter->Architecture())) << "\","
           << "\"minimum_mode\":\"" << EscapeJson(us4::ToString(adapter->MinimumMode())) << "\","
           << "\"supports_moe\":" << (adapter->SupportsMoe() ? "true" : "false") << ","
           << "\"supports_mlx\":" << (adapter->SupportsMlxBackend() ? "true" : "false") << ","
           << "\"supports_metal\":" << (adapter->SupportsMetalBackend() ? "true" : "false") << ","
            << "\"supports_prompt_run\":" << (adapter->SupportsPromptRun() ? "true" : "false")
            << "}";
    first = false;
  }
  std::cout << "{\"models\":[" << models.str() << "]}\n";
}

}  // namespace

int main(int argc, char** argv) {
  bool outputJson = false;
  bool showProbe = false;
  bool showVersion = false;
  bool showHelp = false;
  bool listModels = false;
  bool runCommand = false;
  std::optional<std::string> modeValue;
  std::optional<std::string> modelName;
  std::optional<std::string> modelPath;
  std::optional<std::string> backendValue;
  std::optional<std::string> promptValue;
  std::size_t maxTokens = 16;

  for (int index = 1; index < argc; ++index) {
    const std::string_view arg = argv[index];
    if (arg == "run") {
      runCommand = true;
    } else if (arg == "list-models") {
      listModels = true;
    } else if (arg == "--json") {
      outputJson = true;
    } else if (arg == "--probe") {
      showProbe = true;
    } else if (arg == "--version") {
      showVersion = true;
    } else if (arg == "--help" || arg == "-h") {
      showHelp = true;
    } else if (arg == "--mode" && index + 1 < argc) {
      modeValue = argv[++index];
    } else if (arg == "--model" && index + 1 < argc) {
      modelName = argv[++index];
    } else if (arg == "--model-path" && index + 1 < argc) {
      modelPath = argv[++index];
    } else if (arg == "--backend" && index + 1 < argc) {
      backendValue = argv[++index];
    } else if (arg == "--prompt" && index + 1 < argc) {
      promptValue = argv[++index];
    } else if (arg == "--max-tokens" && index + 1 < argc) {
      const std::string tokenText = argv[++index];
      try {
        const unsigned long long parsed = std::stoull(tokenText);
        maxTokens = static_cast<std::size_t>(
            std::min<unsigned long long>(parsed, static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max())));
      } catch (...) {
        std::cerr << "Invalid --max-tokens value: " << tokenText << "\n";
        return 1;
      }
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      PrintHelp();
      return 1;
    }
  }

  if (showHelp || argc == 1) {
    PrintHelp();
    return 0;
  }

  if (showVersion) {
    std::cout << us4::kUs4Version << "\n";
    return 0;
  }

  const us4::HardwareProbeResult probe = us4::HardwareProbe::Detect();

  if (showProbe) {
    if (outputJson) {
      PrintProbeJson(probe);
    } else {
      PrintProbeText(probe);
    }
    return 0;
  }

  if (modeValue.has_value() && !runCommand) {
    const auto parsedMode = us4::ParseRuntimeMode(*modeValue);
    const us4::RuntimeMode mode =
        (*modeValue == "auto" || !parsedMode.has_value()) ? probe.recommendedMode : *parsedMode;

    if (outputJson) {
      std::cout << "{\"mode\":\"" << us4::ToString(mode) << "\"}\n";
    } else {
      std::cout << us4::ToString(mode) << "\n";
    }
    return 0;
  }

  if (runCommand) {
    const std::optional<us4::BackendType> parsedBackend =
        backendValue.has_value() ? us4::ParseBackendType(*backendValue) : std::nullopt;
    if (backendValue.has_value() && !parsedBackend.has_value()) {
      std::cerr << "Invalid --backend value: " << *backendValue << "\n";
      return 1;
    }

    std::optional<us4::ModelAsset> loadedAsset;
    if (modelPath.has_value()) {
      us4::ModelAsset asset;
      std::string error;
      if (!us4::LoadModelAsset(std::filesystem::path(*modelPath), asset, &error)) {
        std::cerr << "Failed to load --model-path: " << error << "\n";
        return 1;
      }
      loadedAsset = asset;
    }

    if (!modelName.has_value() && (!loadedAsset.has_value() || loadedAsset->modelName.empty())) {
      std::cerr << "--model is required for run\n";
      PrintAdapterList();
      return 1;
    }

    const std::string resolvedModelName =
        modelName.value_or(loadedAsset.has_value() ? loadedAsset->modelName : std::string{});
    const us4::IUS4V6Adapter* adapter = us4::FindAdapterByModel(resolvedModelName);
    if (adapter == nullptr) {
      std::cerr << "Unknown model: " << resolvedModelName << "\n";
      PrintAdapterList();
      return 1;
    }

    us4::RuntimeContext context(probe);
    if (modeValue.has_value()) {
      const auto parsedMode = us4::ParseRuntimeMode(*modeValue);
      context.SetMode(parsedMode.has_value() ? *parsedMode : probe.recommendedMode);
    } else {
      adapter->ConfigureRuntime(context);
    }

    const us4::GenerationRequest request{
        .prompt = promptValue.value_or("hi"),
        .maxTokens = maxTokens,
        .asset = loadedAsset.has_value() ? &*loadedAsset : nullptr,
        .requestedBackend = parsedBackend,
    };
    const us4::GenerationResult result = adapter->Generate(request, context);
    if (outputJson) {
      PrintRunJson(result);
    } else {
      PrintRunText(result);
    }
    return 0;
  }

  if (listModels) {
    if (outputJson) {
      PrintAdapterListJson();
    } else {
      PrintAdapterList();
    }
    return 0;
  }

  PrintHelp();
  return 0;
}
