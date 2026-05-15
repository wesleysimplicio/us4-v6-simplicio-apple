---
sprint: sprint-04
status: in_progress
start: 2026-06-25
end: 2026-07-08
owner: us4-core
---

# Sprint 04 — NEON Hot Paths (Apple)

## Objetivo

Caminhos quentes em NEON (ARM SIMD): matmul, attention, dequantizacao INT8/INT4. Block GEMM tipo BLAS.

## Estado atual no repo em 2026-05-14

- NEON ja faz parte do contrato de backend e pode ser solicitado no CLI, com fallback automatico quando indisponivel.
- Ainda nao ha hot paths reais em `runtime/neon/`; a geracao segue no caminho scalar compartilhado.
- Este sprint continua sendo o alvo para SIMD de verdade, bench e cobertura de corretude.

## Estado atual no repo em 2026-05-15

- `runtime/neon/neon_matmul.cpp` e `runtime/neon/neon_attention.cpp` ainda mantem o bridge para o caminho scalar.
- `runtime/neon/kernel_profile.{h,cpp}` agora descreve o shape SIMD pretendido para matmul e attention (`fp32-lane4`, `fp16-lane8`, `bf16-lane8`, `int8-dot`).
- O contract runner e a suite GTest ja verificam tile shape, lane width, fused softmax-rescale e fallback para hosts nao-ARM.
- Ainda faltam intrinsics reais, dequantizacao INT8/INT4, bench e integracao de geracao no caminho NEON.

## Tasks

- [ ] T04.1 — `runtime/neon/neon_matmul.cpp` (FP16/BF16/INT8 via dotprod, vdotq_s32)
- [ ] T04.2 — `runtime/neon/neon_attention.cpp` (causal, fused softmax-rescale)
- [ ] T04.3 — `runtime/neon/dequant_int8.cpp` + `dequant_int4.cpp` (group-wise scales)
- [ ] T04.4 — Block GEMM tiling 8x8 / 4x16 + cache-aware prefetch
- [ ] T04.5 — Auto-select NEON vs scalar via probe (vector width, cluster type P/E)
- [ ] T04.6 — Re-bench Qwen/Gemma com NEON vs scalar

## Test plan

- Unit: NEON matmul vs scalar (atol 1e-3 BF16); dequant INT8 vs FP32 ground truth.
- Regression: Sprint 01-03 verde.
- E2E: `us4-cli run --backend neon` >=2x speedup vs scalar baseline.
- Correctness: diff NEON vs scalar <= 1e-3.

## DoD

- NEON path em DEGRADED + ULTRA_LOW modes.
- Coverage >=80% em `runtime/neon`.
- Bench tabela atualizada com numeros NEON.

## Riscos

- E-core throttling em workloads sustained -> stick com P-core para hot path.
