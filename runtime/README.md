# Runtime Scaffold

This directory hosts the early C++ runtime scaffold for US4 V6 Apple Edition.

It is no longer just a planned placeholder: the repo already contains a
buildable skeleton for Sprint 01, while the real inference runtime is still
ahead of us.

## What exists now

- root `CMakeLists.txt` configures the runtime build
- `runtime/CMakeLists.txt` builds `us4_runtime_core`
- `apps/CMakeLists.txt` builds the `us4-cli` smoke executable
- `core/` contains compileable contracts for hardware probe, runtime mode,
  runtime context, and backend selection
- `adapters/` contains a native registry plus scaffold adapters for Qwen,
  Gemma, BitNet, ternary, Llama, DeepSeek MoE, and Kimi MoE
- `metal/` now contains a cross-platform command queue skeleton used by native
  contracts
- `mlx/` now contains a cross-platform bridge skeleton used by native
  contracts
- `kv/` contains small `KvPager` and `PrefixCache` foundations used by unit
  coverage
- `moe/` contains small `Router` and `ExpertPager` foundations used by MoE
  adapter scaffolding
- `telemetry/` contains minimal sink/types placeholders used by smoke tests
- `benchmarks/dense_baseline.cpp` now exercises the scalar dense baseline path
- `tests/unit/` contains smoke coverage plus backend selection, KV, and MoE
  contract tests

## What the scaffold already proves

- the runtime tree layout is real
- build entrypoints are explicit
- `us4-cli` already exposes `--version`, `--probe`, `--mode <value>`, and a
  first `run --model ... --prompt ...` scalar path
- `us4-cli list-models` exposes the native adapter registry
- `run --model-path ...` can load fixture manifests and detect GGUF /
  Safetensors file types without external libraries
- backend selection and fallback are explicit in CLI output
- the backend contract already accepts `scalar`, `neon`, `mlx`, `metal`, and
  `ane`, with automatic fallback when a requested path is unavailable
- `RuntimeContext` now exposes acceleration services for Metal and MLX even
  before the real Apple-only backend code lands
- the native registry already exposes dense, ternary, llama, and MoE-family
  adapters, but they still execute through the shared scaffold path today
- hardware probe and mode selection contracts compile and run
- KV and MoE directories now contain contract-grade foundations, not just empty
  placeholders
- dense adapters can generate deterministic scalar tokens for fixture-grade
  validation

## What is still missing

- full tensor/view ownership model and execution graph
- real weight loading from GGUF / Safetensors payloads
- real MLX bridge and Metal kernels used by generation
- NEON / Accelerate hot paths used by generation
- backend-specific execution beyond the selection and fallback contract
- production KV tiering, SSD cold storage, and summarization flows
- production MoE routing, expert lazy loading, and expert telemetry
- production-capable `run`, `serve`, `bench`, and `tune` CLI flows
- correctness fixtures and backend regression coverage

## Directory intent

| Path | Intent today | Evolves into |
|---|---|---|
| `core/` | stable contracts and orchestration skeleton | runtime orchestration, selection, and shared primitives |
| `adapters/` | native registry and scaffold family adapters | dense, MoE, and low-memory adapters |
| `mlx/` | reserved primary backend surface | MLX graph/build/eval integration |
| `metal/` | reserved accelerated backend surface | measured hot kernels only |
| `neon/` | reserved CPU fallback surface | scalar/NEON low-memory and safety paths |
| `ane/` | reserved opt-in backend surface | validated M5+ offload paths |
| `kv/` | contract-grade pager and prefix-cache foundation | KV lifecycle, tiering, and reuse |
| `moe/` | contract-grade router and expert-pager foundation | routed experts and MoE scheduling |
| `memory/`, `cache/`, `speculative/`, `tuning/` | roadmap-aligned placeholders | runtime subsystems landed by later sprints |
| `telemetry/` | smoke-level instrumentation contract | structured runtime metrics and fallback observability |
| `benchmarks/` | baseline scaffold harness | correctness and throughput evidence |

## Build entrypoints

From repo root:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/us4-cli --version
./build/us4-cli --probe
./build/us4-cli --mode auto
./build/us4-cli list-models
./build/us4-cli run --model qwen-0.5b --prompt "hi"
./build/us4-cli run --model qwen-0.5b --model-path tests/fixtures/models/qwen-0.5b/model.us4manifest --prompt "hi"
./build/us4-cli run --model llama-3.1-8b --backend metal --prompt "hello"
```

These commands validate the scaffold and CLI contract. They do not validate
real inference yet.

Requesting `--backend metal`, `--backend mlx`, or `--backend neon` currently
validates selection and reporting behavior; it does not prove that generation is
already running on those accelerated paths.

## Transition rule

During the starter-to-runtime transition, this tree must stay honest about the
current repo state:

- document what already builds and runs;
- mark placeholders as placeholders;
- land real runtime behavior inside the existing ownership boundaries from
  `PATTERNS.md`;
- avoid treating registry presence or backend selection as proof of
  production-grade family/back-end execution;
- avoid claiming MLX, Metal, NEON, or adapter support before the code and tests
  exist.

See [STARTER-TO-RUNTIME.md](STARTER-TO-RUNTIME.md) for the short migration map.
