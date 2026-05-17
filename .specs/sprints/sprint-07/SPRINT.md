---
sprint: sprint-07
status: in_progress
start: 2026-08-06
end: 2026-08-19
owner: us4-core
---

# Sprint 07 - Llama Adapter (Apple)

## Objetivo

Adapter Llama 3/4 com GQA, RoPE scaling (linear/dynamic/YaRN), ALiBi
opcional, KV reutilizavel e caminho de attention consistente nos backends ja
existentes.

## Estado atual no repo em 2026-05-16

- `LlamaAdapter` ja esta no registry nativo e aparece no CLI.
- `LlamaConfig` resolve `hidden_size`, `query_heads`, `kv_heads`,
  `head_dim`, `rope_theta`, `rope_scaling` e `rope_scale` a partir de
  metadata do manifesto fixture, com cobertura de normalizacao segura.
- `runtime/core/rope.{h,cpp}` deixou de ser placeholder e hoje cobre scaling
  `linear`, `dynamic` e `YaRN` com contrato deterministico.
- `runtime/core/gqa_attention.{h,cpp}` ja executa grouped-query attention no
  caminho dedicado de Llama, com smoke/contracts protegendo shapes e
  regressao.
- O adapter dedicado de Llama ja reutiliza `KvPager`, `PrefixCache`,
  `SsdColdStore` e `Summarizer` via helpers compartilhados do
  `DenseAdapterBase`, inclusive em `MICRO` com restore de cold-store e
  telemetria de `kvSummaryRows`.
- O caminho dedicado de `neon` continua sendo o mais avancado; `scalar/mlx/metal`
  ainda dependem do shared path do `DenseAdapterBase`.
- Ainda faltam loader real de Llama, corretude contra referencia externa e
  benchmark forte nos backends Apple para fechar o sprint.

## Tasks

- [x] T07.1 - `runtime/adapters/llama/LlamaConfig` (rope_theta,
  rope_scaling, gqa heads)
- [x] T07.2 - `runtime/adapters/llama/LlamaAdapter` (forward pass, KV via pager)
- [x] T07.3 - `runtime/core/rope.{h,cpp}` (linear + dynamic + YaRN scaling)
- [x] T07.4 - `runtime/core/gqa_attention.{h,cpp}` (grouped-query attention)
- [ ] T07.5 - Loader: Llama GGUF + safetensors + tokenizer.json
- [ ] T07.6 - Bench Llama 3.x 8B em Metal + NEON

## Test plan

- Unit: RoPE vs reference Python (atol 1e-5); GQA shape contract.
- Regression: outros adapters intactos.
- E2E: Llama 3 8B Q4 em M3 Max gera 200 tokens em <= 30s.
- Correctness: diff vs HF reference <= 1e-3 nos primeiros 64 tokens.

## Contract prep before implementation

- Unit contract should keep Llama fixture coverage for:
  - manifest directory loading without explicit `--model`
  - default prompt token fallback from `model.us4manifest`
  - GGUF asset detection and family/model routing
  - requested backend fallback telemetry when `metal` is unavailable
  - shared KV reuse within the same `RuntimeContext`
  - cold-store restore in `MICRO` mode with summary telemetry
- Native E2E should keep host-aware Llama evidence for both:
  - `tests/fixtures/models/llama-3.1-8b/`
  - `tests/fixtures/models/llama-3.1-8b/toy-llama.gguf`
- Bench evidence for T07.6 should record at minimum:
  - requested backend, observed backend, and fallback reason
  - runtime mode and hardware profile
  - generated token count, elapsed time, and text fingerprint
  - correctness delta once the HF reference path exists

## DoD

- Llama 3 + 4 funcionando em FULL/BALANCED_PLUS/DEGRADED.
- Coverage >=80% em `runtime/adapters/llama`.
