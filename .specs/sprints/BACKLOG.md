# Backlog - US4 V6 Apple Edition

## Cadence

- 12 sprints
- 2 weeks each
- Sprint 01 starts from planning/bootstrap and creates the runtime skeleton

## Current planning inventory

- 12 sprint directories exist under `.specs/sprints/`
- 9 authored task files currently exist on disk
- All authored `*.task.md` files still live in `sprint-01`
- Sprints 02-12 already have scope/theme/timeline, but still need finer task
  decomposition as implementation continues

## Quality gates by maturity

| Phase | Required gates |
|---|---|
| Planning/bootstrap | docs consistency, starter tests, scaffold validation |
| Skeleton/runtime bootstrap | build, format/lint, unit, CLI smoke |
| Inference baseline onward | correctness, regression, benchmark evidence |

## Sprint matrix

| Sprint | Theme | Main result |
|---|---|---|
| 01 | Foundations and Skeleton | runtime skeleton, CLI contract, probe, mode selector, starter-to-runtime transition plan |
| 02 | CPU Scalar Baseline | tensor core, scalar dense baseline, first text generation path |
| 03 | MLX and Metal Skeleton | MLX path, first Metal kernels, unified-memory baseline |
| 04 | NEON Hot Paths | CPU SIMD acceleration and dequant hot loops |
| 05 | BitNet and Ternary | low-memory adapters and ternary kernel path |
| 06 | KV Memory Architecture | tiered KV lifecycle and prefix reuse |
| 07 | Llama Adapter | dense adapter expansion and public alpha confidence |
| 08 | MoE Foundation | DeepSeek/Kimi adapter foundation and expert paging |
| 09 | MoE Advanced | MiniMax/GLM, speculative expert prefetch |
| 10 | Continuous Batching and Speculative Decoding | multi-session scheduling and draft/verify flow |
| 11 | ANE M5+ Offload | opt-in ANE path with fallback discipline |
| 12 | Auto-Tune and Release | profiling, release hardening, v1.0 |
