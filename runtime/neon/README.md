# NEON

Home for NEON and Accelerate-based CPU fallback paths.

What already exists:

- `neon_matmul.cpp` and `neon_attention.cpp` still bridge to the scalar
  reference path, which keeps regression behavior stable
- `kernel_profile.{h,cpp}` now describes the intended SIMD shape for matmul and
  attention (`fp32-lane4`, `fp16-lane8`, `bf16-lane8`, `int8-dot`)
- the NEON profile layer already locks tile expectations such as `8x8`,
  `4x16`, dot-product usage, and fused softmax-rescale intent

What is still missing:

- real ARM intrinsics and Accelerate hot paths behind those profiles
- dequantization kernels for INT8 and INT4
- benchmark and correctness wiring for the NEON path
