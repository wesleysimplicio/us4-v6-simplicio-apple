#include "memory/unified_allocator.h"

namespace us4 {

std::string_view ToString(const AllocationVisibility visibility) {
  switch (visibility) {
    case AllocationVisibility::kCpuOnly:
      return "cpu-only";
    case AllocationVisibility::kUnifiedShared:
      return "unified-shared";
  }
  return "cpu-only";
}

std::shared_ptr<UnifiedAllocation> UnifiedAllocator::Allocate(const std::size_t byteCount, const bool gpuVisible) {
  auto allocation = std::make_shared<UnifiedAllocation>();
  allocation->bytes.resize(byteCount);
  allocation->gpuVisible = gpuVisible;
  allocation->visibility = gpuVisible ? AllocationVisibility::kUnifiedShared : AllocationVisibility::kCpuOnly;
  allocations_.push_back(allocation);
  return allocation;
}

std::size_t UnifiedAllocator::AllocationCount() const { return allocations_.size(); }

std::size_t UnifiedAllocator::ResidentBytes() const {
  std::size_t total = 0;
  for (const auto& allocation : allocations_) {
    total += allocation->bytes.size();
  }
  return total;
}

std::size_t UnifiedAllocator::SharedAllocationCount() const {
  std::size_t total = 0;
  for (const auto& allocation : allocations_) {
    if (allocation->visibility == AllocationVisibility::kUnifiedShared) {
      ++total;
    }
  }
  return total;
}

}  // namespace us4
