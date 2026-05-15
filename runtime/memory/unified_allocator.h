#pragma once

#include <cstddef>
#include <memory>
#include <string_view>
#include <vector>

namespace us4 {

enum class AllocationVisibility {
  kCpuOnly,
  kUnifiedShared,
};

struct UnifiedAllocation {
  std::vector<std::byte> bytes;
  bool cpuVisible = true;
  bool gpuVisible = false;
  AllocationVisibility visibility = AllocationVisibility::kCpuOnly;
};

class UnifiedAllocator {
 public:
  std::shared_ptr<UnifiedAllocation> Allocate(std::size_t byteCount, bool gpuVisible);
  std::size_t AllocationCount() const;
  std::size_t ResidentBytes() const;
  std::size_t SharedAllocationCount() const;

 private:
  std::vector<std::shared_ptr<UnifiedAllocation>> allocations_;
};

std::string_view ToString(AllocationVisibility visibility);

}  // namespace us4
