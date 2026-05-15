#include <gtest/gtest.h>

#include "neon/kernel_profile.h"

namespace {

us4::HardwareProbeResult MakeArmProbe() {
  us4::HardwareProbeResult probe;
  probe.platform = "macos";
  probe.architecture = "arm64";
  probe.chip = "apple-m";
  probe.hasNeon = true;
  return probe;
}

} // namespace

TEST(NeonContractTest, MatmulProfileUsesLaneAwareTilesForArm64) {
  const us4::Tensor lhs({8, 16}, us4::DType::kFloat16, us4::DeviceType::kCpu);
  const us4::Tensor rhs({16, 32}, us4::DType::kFloat16, us4::DeviceType::kCpu);

  const us4::NeonMatmulProfile profile =
      us4::PlanNeonMatmul(MakeArmProbe(), lhs, rhs);

  EXPECT_EQ(profile.flavor, us4::NeonKernelFlavor::kFp16Lane8);
  EXPECT_EQ(profile.tileRows, 8U);
  EXPECT_EQ(profile.tileCols, 8U);
  EXPECT_EQ(profile.vectorLanes, 8U);
  EXPECT_FALSE(profile.usesAccelerateFallback);
}

TEST(NeonContractTest, Int8MatmulProfilePrefersDotProductPath) {
  const us4::Tensor lhs({8, 32}, us4::DType::kInt8, us4::DeviceType::kCpu);
  const us4::Tensor rhs({32, 32}, us4::DType::kInt8, us4::DeviceType::kCpu);

  const us4::NeonMatmulProfile profile =
      us4::PlanNeonMatmul(MakeArmProbe(), lhs, rhs);

  EXPECT_EQ(profile.flavor, us4::NeonKernelFlavor::kInt8Dot);
  EXPECT_TRUE(profile.usesDotProduct);
  EXPECT_EQ(profile.vectorLanes, 16U);
}

TEST(NeonContractTest, AttentionProfileKeepsFusedSoftmaxContract) {
  const us4::Tensor query({1, 8, 64}, us4::DType::kFloat32,
                          us4::DeviceType::kCpu);
  const us4::Tensor key({1, 8, 64}, us4::DType::kFloat32,
                        us4::DeviceType::kCpu);
  const us4::Tensor value({1, 8, 64}, us4::DType::kFloat32,
                          us4::DeviceType::kCpu);

  const us4::NeonAttentionProfile profile =
      us4::PlanNeonAttention(MakeArmProbe(), query, key, value, true);

  EXPECT_EQ(profile.flavor, us4::NeonKernelFlavor::kFp32Lane4);
  EXPECT_EQ(profile.vectorLanes, 4U);
  EXPECT_EQ(profile.headDimBlock, 32U);
  EXPECT_TRUE(profile.fusesSoftmaxRescale);
  EXPECT_TRUE(profile.supportsCausalMask);
}

TEST(NeonContractTest, NonArmHostsStayOnScalarBridgePlan) {
  us4::HardwareProbeResult probe;
  probe.platform = "windows";
  probe.architecture = "x64";
  probe.hasNeon = false;

  const us4::Tensor lhs({4, 4}, us4::DType::kFloat32, us4::DeviceType::kCpu);
  const us4::Tensor rhs({4, 4}, us4::DType::kFloat32, us4::DeviceType::kCpu);

  const us4::NeonMatmulProfile profile = us4::PlanNeonMatmul(probe, lhs, rhs);

  EXPECT_EQ(profile.flavor, us4::NeonKernelFlavor::kScalarBridge);
  EXPECT_FALSE(profile.usesDotProduct);
}
