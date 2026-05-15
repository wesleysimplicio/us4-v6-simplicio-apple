# MLX

Home for the primary MLX execution path on Apple Silicon.

What already exists:

- `mlx_bridge.{h,cpp}` records a dense-plan build/evaluate lifecycle
- availability is driven by `HardwareProbeResult::hasMlx`
- the bridge can reference unified-shared allocations from
  `UnifiedAllocator`

What is still missing:

- real MLX graph construction
- shared buffer interop with Metal / unified memory
- execution wired into generation for dense and Llama-family adapters
