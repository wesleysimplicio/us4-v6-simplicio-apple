# Metal

Home for the Metal execution path.

What already exists:

- `command_queue.{h,cpp}` records dispatch intent for `matmul`, `softmax`, and
  `rmsnorm`
- queue availability is driven by `HardwareProbeResult::hasMetal`
- shared allocations from `UnifiedAllocator` can be attached to dispatch
  records

What is still missing:

- real `MTLDevice` and command-queue ownership
- autorelease helpers for Objective-C++ boundaries
- `.metal` kernels and dispatch wrappers used by generation
