#include <chrono>
#include <filesystem>
#include <iostream>
#include <optional>

#include "adapters/adapter_registry.h"
#include "core/hardware_probe.h"
#include "core/model_asset.h"
#include "core/runtime_context.h"
#include "core/runtime_mode.h"

namespace {

std::filesystem::path RepoRoot() {
#ifdef US4_SOURCE_DIR
  return std::filesystem::path(US4_SOURCE_DIR);
#else
  return std::filesystem::path(__FILE__)
      .parent_path()
      .parent_path()
      .parent_path();
#endif
}

int RunCase(const us4::HardwareProbeResult &probe, const std::string_view label,
            const std::string_view model,
            const std::optional<std::filesystem::path> &manifest,
            const std::optional<us4::BackendType> requestedBackend) {
  const us4::IUS4V6Adapter *adapter = us4::FindAdapterByModel(model);
  if (adapter == nullptr) {
    std::cerr << "missing adapter for " << model << "\n";
    return 1;
  }

  std::optional<us4::ModelAsset> asset;
  std::string error;
  if (manifest.has_value()) {
    us4::ModelAsset loaded;
    if (!us4::LoadModelAsset(*manifest, loaded, &error)) {
      std::cerr << "failed to load manifest " << manifest->string() << ": "
                << error << "\n";
      return 1;
    }
    asset = loaded;
  }

  us4::RuntimeContext context(probe);
  adapter->ConfigureRuntime(context);
  const auto start = std::chrono::steady_clock::now();
  const us4::GenerationResult result =
      adapter->Generate({.prompt = "hi",
                         .maxTokens = 8,
                         .asset = asset.has_value() ? &*asset : nullptr,
                         .requestedBackend = requestedBackend},
                        context);
  const auto end = std::chrono::steady_clock::now();
  const auto elapsedMs =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();

  std::cout << "case=" << label << "\n";
  std::cout << "model=" << result.modelName << "\n";
  std::cout << "backend=" << result.backend << "\n";
  std::cout << "backend_reason=" << result.backendReason << "\n";
  std::cout << "weight_dtype=" << result.weightDType << "\n";
  std::cout << "neon_kernel_flavor=" << result.neonKernelFlavor << "\n";
  std::cout << "dequant_path=" << result.dequantPath << "\n";
  std::cout << "generated_tokens=" << result.generatedTokens.size() << "\n";
  std::cout << "elapsed_ms=" << elapsedMs << "\n";
  std::cout << "text=" << result.text << "\n";
  std::cout << "--\n";
  return 0;
}

} // namespace

int main() {
  const us4::HardwareProbeResult probe = us4::HardwareProbe::Detect();
  std::cout << "benchmark=dense_baseline\n";
  std::cout << "recommended_mode=" << us4::ToString(probe.recommendedMode)
            << "\n";
  std::cout << "neon_vector_bits=" << probe.neonVectorBits << "\n";
  std::cout << "has_performance_cores="
            << (probe.hasPerformanceCores ? "true" : "false") << "\n";
  std::cout << "has_efficiency_cores="
            << (probe.hasEfficiencyCores ? "true" : "false") << "\n";
  std::cout << "--\n";

  const std::filesystem::path repoRoot = RepoRoot();
  if (RunCase(probe, "dense-default", "qwen-0.5b", std::nullopt,
              std::nullopt) != 0) {
    return 1;
  }
  if (RunCase(probe, "lowbit-int8", "bitnet-b1.58-2b",
              repoRoot / "tests" / "fixtures" / "models" / "bitnet-b1.58-2b" /
                  "model.us4manifest",
              us4::BackendType::kNeon) != 0) {
    return 1;
  }
  if (RunCase(probe, "lowbit-int4", "pt-bitnet-ternary-2b",
              repoRoot / "tests" / "fixtures" / "models" /
                  "pt-bitnet-ternary-2b" / "model.us4manifest",
              us4::BackendType::kNeon) != 0) {
    return 1;
  }

  return 0;
}
