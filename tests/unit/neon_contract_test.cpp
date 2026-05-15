#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "cpu/scalar_attention.h"
#include "cpu/scalar_matmul.h"
#include "neon/kernel_profile.h"
#include "neon/neon_attention.h"
#include "neon/neon_matmul.h"

namespace {

us4::HardwareProbeResult MakeArmProbe() {
  us4::HardwareProbeResult probe;
  probe.platform = "macos";
  probe.architecture = "arm64";
  probe.chip = "apple-m";
  probe.hasNeon = true;
  return probe;
}

void FillSequence(float *data, const std::size_t count, const float scale,
                  const float bias) {
  for (std::size_t index = 0; index < count; ++index) {
    data[index] = (static_cast<float>((index % 7U) + 1U) * scale) + bias;
  }
}

void FillFloat16Tensor(us4::Tensor &tensor, const std::vector<float> &values) {
  std::uint16_t *data = tensor.MutableDataAsUInt16();
  ASSERT_NE(data, nullptr);
  ASSERT_EQ(values.size(), tensor.ElementCount());
  for (std::size_t index = 0; index < values.size(); ++index) {
    data[index] = us4::EncodeFloat16(values[index]);
  }
}

void FillBFloat16Tensor(us4::Tensor &tensor, const std::vector<float> &values) {
  std::uint16_t *data = tensor.MutableDataAsUInt16();
  ASSERT_NE(data, nullptr);
  ASSERT_EQ(values.size(), tensor.ElementCount());
  for (std::size_t index = 0; index < values.size(); ++index) {
    data[index] = us4::EncodeBFloat16(values[index]);
  }
}

float QuantizeHalfValue(const float value, const bool bfloat16) {
  return bfloat16 ? us4::DecodeBFloat16(us4::EncodeBFloat16(value))
                  : us4::DecodeFloat16(us4::EncodeFloat16(value));
}

void FillHalfReferenceTensor(us4::Tensor &tensor,
                             const std::vector<float> &values,
                             const bool bfloat16) {
  float *data = tensor.MutableDataAsFloat32();
  ASSERT_NE(data, nullptr);
  ASSERT_EQ(values.size(), tensor.ElementCount());
  for (std::size_t index = 0; index < values.size(); ++index) {
    data[index] = QuantizeHalfValue(values[index], bfloat16);
  }
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

TEST(NeonContractTest, NeonMatmulMatchesScalarResultForFp32) {
  us4::Tensor lhs({2, 3}, us4::DType::kFloat32, us4::DeviceType::kCpu);
  us4::Tensor rhs({3, 4}, us4::DType::kFloat32, us4::DeviceType::kCpu);
  us4::Tensor output({2, 4}, us4::DType::kFloat32, us4::DeviceType::kCpu);

  float *lhsData = lhs.MutableDataAsFloat32();
  float *rhsData = rhs.MutableDataAsFloat32();
  lhsData[0] = 1.0F;
  lhsData[1] = 2.0F;
  lhsData[2] = 3.0F;
  lhsData[3] = 4.0F;
  lhsData[4] = 5.0F;
  lhsData[5] = 6.0F;

  rhsData[0] = 1.0F;
  rhsData[1] = 0.0F;
  rhsData[2] = 2.0F;
  rhsData[3] = 1.0F;
  rhsData[4] = 0.0F;
  rhsData[5] = 1.0F;
  rhsData[6] = 3.0F;
  rhsData[7] = 0.0F;
  rhsData[8] = 1.0F;
  rhsData[9] = 1.0F;
  rhsData[10] = 0.0F;
  rhsData[11] = 2.0F;

  std::string error;
  ASSERT_TRUE(us4::NeonMatmul(lhs, rhs, output, &error)) << error;

  const float *values = output.DataAsFloat32();
  ASSERT_NE(values, nullptr);
  EXPECT_FLOAT_EQ(values[0], 4.0F);
  EXPECT_FLOAT_EQ(values[1], 5.0F);
  EXPECT_FLOAT_EQ(values[2], 8.0F);
  EXPECT_FLOAT_EQ(values[3], 7.0F);
  EXPECT_FLOAT_EQ(values[4], 10.0F);
  EXPECT_FLOAT_EQ(values[5], 11.0F);
  EXPECT_FLOAT_EQ(values[6], 23.0F);
  EXPECT_FLOAT_EQ(values[7], 16.0F);
}

TEST(NeonContractTest, NeonMatmulMatchesScalarResultForTailColumns) {
  us4::Tensor lhs({3, 7}, us4::DType::kFloat32, us4::DeviceType::kCpu);
  us4::Tensor rhs({7, 6}, us4::DType::kFloat32, us4::DeviceType::kCpu);
  us4::Tensor neonOutput({3, 6}, us4::DType::kFloat32, us4::DeviceType::kCpu);
  us4::Tensor scalarOutput({3, 6}, us4::DType::kFloat32, us4::DeviceType::kCpu);

  float *lhsData = lhs.MutableDataAsFloat32();
  float *rhsData = rhs.MutableDataAsFloat32();
  for (std::size_t index = 0; index < 21U; ++index) {
    lhsData[index] = static_cast<float>((index % 5U) - 2U);
  }
  for (std::size_t index = 0; index < 42U; ++index) {
    rhsData[index] = static_cast<float>((index % 7U) - 3U) * 0.5F;
  }

  std::string error;
  ASSERT_TRUE(us4::NeonMatmul(lhs, rhs, neonOutput, &error)) << error;
  ASSERT_TRUE(us4::ScalarMatmul(lhs, rhs, scalarOutput, &error)) << error;

  const float *neonValues = neonOutput.DataAsFloat32();
  const float *scalarValues = scalarOutput.DataAsFloat32();
  ASSERT_NE(neonValues, nullptr);
  ASSERT_NE(scalarValues, nullptr);
  for (std::size_t index = 0; index < 18U; ++index) {
    EXPECT_FLOAT_EQ(neonValues[index], scalarValues[index]) << index;
  }
}

TEST(NeonContractTest, NeonMatmulExecutesFp16AndAccumulatesIntoFp32) {
  us4::Tensor lhs({2, 3}, us4::DType::kFloat16, us4::DeviceType::kCpu);
  us4::Tensor rhs({3, 5}, us4::DType::kFloat16, us4::DeviceType::kCpu);
  us4::Tensor neonOutput({2, 5}, us4::DType::kFloat32, us4::DeviceType::kCpu);
  us4::Tensor scalarLhs({2, 3}, us4::DType::kFloat32, us4::DeviceType::kCpu);
  us4::Tensor scalarRhs({3, 5}, us4::DType::kFloat32, us4::DeviceType::kCpu);
  us4::Tensor scalarOutput({2, 5}, us4::DType::kFloat32, us4::DeviceType::kCpu);

  const std::vector<float> lhsValues = {0.5F, -1.0F, 2.0F, 1.5F, 0.25F, -0.75F};
  const std::vector<float> rhsValues = {0.25F,  -0.5F, 1.0F,   0.75F, -1.25F,
                                        1.5F,   0.0F,  -0.25F, 0.5F,  1.0F,
                                        -0.75F, 0.25F, 0.5F,   -1.5F, 0.25F};
  FillFloat16Tensor(lhs, lhsValues);
  FillFloat16Tensor(rhs, rhsValues);
  FillHalfReferenceTensor(scalarLhs, lhsValues, false);
  FillHalfReferenceTensor(scalarRhs, rhsValues, false);

  std::string error;
  ASSERT_TRUE(us4::NeonMatmul(lhs, rhs, neonOutput, &error)) << error;
  ASSERT_TRUE(us4::ScalarMatmul(scalarLhs, scalarRhs, scalarOutput, &error))
      << error;

  const float *neonValues = neonOutput.DataAsFloat32();
  const float *scalarValues = scalarOutput.DataAsFloat32();
  ASSERT_NE(neonValues, nullptr);
  ASSERT_NE(scalarValues, nullptr);
  for (std::size_t index = 0; index < 10U; ++index) {
    EXPECT_NEAR(neonValues[index], scalarValues[index], 1e-3F) << index;
  }
}

TEST(NeonContractTest, NeonMatmulExecutesBf16AndKeepsTailParity) {
  us4::Tensor lhs({3, 4}, us4::DType::kBFloat16, us4::DeviceType::kCpu);
  us4::Tensor rhs({4, 6}, us4::DType::kBFloat16, us4::DeviceType::kCpu);
  us4::Tensor neonOutput({3, 6}, us4::DType::kFloat32, us4::DeviceType::kCpu);
  us4::Tensor scalarLhs({3, 4}, us4::DType::kFloat32, us4::DeviceType::kCpu);
  us4::Tensor scalarRhs({4, 6}, us4::DType::kFloat32, us4::DeviceType::kCpu);
  us4::Tensor scalarOutput({3, 6}, us4::DType::kFloat32, us4::DeviceType::kCpu);

  const std::vector<float> lhsValues = {0.125F, 0.5F,   -0.75F, 1.25F,
                                        -1.0F,  0.375F, 0.875F, -0.5F,
                                        0.625F, -0.25F, 1.5F,   0.75F};
  const std::vector<float> rhsValues = {
      0.25F, -0.5F, 0.75F,  1.0F,    -0.25F, 0.5F,    -0.75F, 0.125F,
      0.5F,  -1.0F, 0.25F,  0.875F,  1.125F, -0.625F, 0.375F, 0.25F,
      -0.5F, 0.75F, 0.625F, -0.125F, 1.0F,   -0.75F,  0.5F,   -0.25F};
  FillBFloat16Tensor(lhs, lhsValues);
  FillBFloat16Tensor(rhs, rhsValues);
  FillHalfReferenceTensor(scalarLhs, lhsValues, true);
  FillHalfReferenceTensor(scalarRhs, rhsValues, true);

  std::string error;
  ASSERT_TRUE(us4::NeonMatmul(lhs, rhs, neonOutput, &error)) << error;
  ASSERT_TRUE(us4::ScalarMatmul(scalarLhs, scalarRhs, scalarOutput, &error))
      << error;

  const float *neonValues = neonOutput.DataAsFloat32();
  const float *scalarValues = scalarOutput.DataAsFloat32();
  ASSERT_NE(neonValues, nullptr);
  ASSERT_NE(scalarValues, nullptr);
  for (std::size_t index = 0; index < 18U; ++index) {
    EXPECT_NEAR(neonValues[index], scalarValues[index], 1e-2F) << index;
  }
}

TEST(NeonContractTest, NeonAttentionMatchesScalarForFp32) {
  us4::Tensor query({2, 4}, us4::DType::kFloat32, us4::DeviceType::kCpu);
  us4::Tensor key({3, 4}, us4::DType::kFloat32, us4::DeviceType::kCpu);
  us4::Tensor value({3, 3}, us4::DType::kFloat32, us4::DeviceType::kCpu);
  us4::Tensor neonOutput({2, 3}, us4::DType::kFloat32, us4::DeviceType::kCpu);
  us4::Tensor scalarOutput({2, 3}, us4::DType::kFloat32, us4::DeviceType::kCpu);

  FillSequence(query.MutableDataAsFloat32(), 8U, 0.25F, -0.5F);
  FillSequence(key.MutableDataAsFloat32(), 12U, 0.20F, -0.25F);
  FillSequence(value.MutableDataAsFloat32(), 9U, 0.15F, 0.10F);

  std::string error;
  ASSERT_TRUE(
      us4::NeonAttention(query, key, value, neonOutput, false, {}, &error))
      << error;
  ASSERT_TRUE(
      us4::ScalarAttention(query, key, value, scalarOutput, false, {}, &error))
      << error;

  const float *neonValues = neonOutput.DataAsFloat32();
  const float *scalarValues = scalarOutput.DataAsFloat32();
  ASSERT_NE(neonValues, nullptr);
  ASSERT_NE(scalarValues, nullptr);
  for (std::size_t index = 0; index < 6U; ++index) {
    EXPECT_NEAR(neonValues[index], scalarValues[index], 1e-5F) << index;
  }
}

TEST(NeonContractTest, NeonAttentionMatchesScalarWithCausalMask) {
  us4::Tensor query({3, 4}, us4::DType::kFloat32, us4::DeviceType::kCpu);
  us4::Tensor key({3, 4}, us4::DType::kFloat32, us4::DeviceType::kCpu);
  us4::Tensor value({3, 2}, us4::DType::kFloat32, us4::DeviceType::kCpu);
  us4::Tensor neonOutput({3, 2}, us4::DType::kFloat32, us4::DeviceType::kCpu);
  us4::Tensor scalarOutput({3, 2}, us4::DType::kFloat32, us4::DeviceType::kCpu);

  FillSequence(query.MutableDataAsFloat32(), 12U, 0.10F, -0.20F);
  FillSequence(key.MutableDataAsFloat32(), 12U, 0.05F, 0.30F);
  FillSequence(value.MutableDataAsFloat32(), 6U, 0.40F, -0.10F);

  std::string error;
  ASSERT_TRUE(
      us4::NeonAttention(query, key, value, neonOutput, true, {}, &error))
      << error;
  ASSERT_TRUE(
      us4::ScalarAttention(query, key, value, scalarOutput, true, {}, &error))
      << error;

  const float *neonValues = neonOutput.DataAsFloat32();
  const float *scalarValues = scalarOutput.DataAsFloat32();
  ASSERT_NE(neonValues, nullptr);
  ASSERT_NE(scalarValues, nullptr);
  for (std::size_t index = 0; index < 6U; ++index) {
    EXPECT_NEAR(neonValues[index], scalarValues[index], 1e-5F) << index;
  }
}

TEST(NeonContractTest, NeonAttentionMatchesScalarWithCache) {
  us4::Tensor query({2, 4}, us4::DType::kFloat32, us4::DeviceType::kCpu);
  us4::Tensor key({2, 4}, us4::DType::kFloat32, us4::DeviceType::kCpu);
  us4::Tensor value({2, 3}, us4::DType::kFloat32, us4::DeviceType::kCpu);
  us4::Tensor cacheKeys({1, 4}, us4::DType::kFloat32, us4::DeviceType::kCpu);
  us4::Tensor cacheValues({1, 3}, us4::DType::kFloat32, us4::DeviceType::kCpu);
  us4::Tensor neonOutput({2, 3}, us4::DType::kFloat32, us4::DeviceType::kCpu);
  us4::Tensor scalarOutput({2, 3}, us4::DType::kFloat32, us4::DeviceType::kCpu);

  FillSequence(query.MutableDataAsFloat32(), 8U, 0.30F, -0.20F);
  FillSequence(key.MutableDataAsFloat32(), 8U, 0.12F, 0.15F);
  FillSequence(value.MutableDataAsFloat32(), 6U, 0.22F, -0.05F);
  FillSequence(cacheKeys.MutableDataAsFloat32(), 4U, 0.07F, 0.11F);
  FillSequence(cacheValues.MutableDataAsFloat32(), 3U, 0.18F, 0.02F);

  const us4::AttentionCacheView cache{&cacheKeys, &cacheValues};
  std::string error;
  ASSERT_TRUE(
      us4::NeonAttention(query, key, value, neonOutput, false, cache, &error))
      << error;
  ASSERT_TRUE(us4::ScalarAttention(query, key, value, scalarOutput, false,
                                   cache, &error))
      << error;

  const float *neonValues = neonOutput.DataAsFloat32();
  const float *scalarValues = scalarOutput.DataAsFloat32();
  ASSERT_NE(neonValues, nullptr);
  ASSERT_NE(scalarValues, nullptr);
  for (std::size_t index = 0; index < 6U; ++index) {
    EXPECT_NEAR(neonValues[index], scalarValues[index], 1e-5F) << index;
  }
}

TEST(NeonContractTest, NeonAttentionMatchesScalarForWideValueTail) {
  us4::Tensor query({2, 4}, us4::DType::kFloat32, us4::DeviceType::kCpu);
  us4::Tensor key({3, 4}, us4::DType::kFloat32, us4::DeviceType::kCpu);
  us4::Tensor value({3, 5}, us4::DType::kFloat32, us4::DeviceType::kCpu);
  us4::Tensor cacheKeys({1, 4}, us4::DType::kFloat32, us4::DeviceType::kCpu);
  us4::Tensor cacheValues({1, 5}, us4::DType::kFloat32, us4::DeviceType::kCpu);
  us4::Tensor neonOutput({2, 5}, us4::DType::kFloat32, us4::DeviceType::kCpu);
  us4::Tensor scalarOutput({2, 5}, us4::DType::kFloat32, us4::DeviceType::kCpu);

  FillSequence(query.MutableDataAsFloat32(), 8U, 0.28F, -0.12F);
  FillSequence(key.MutableDataAsFloat32(), 12U, 0.14F, 0.08F);
  FillSequence(value.MutableDataAsFloat32(), 15U, 0.11F, -0.04F);
  FillSequence(cacheKeys.MutableDataAsFloat32(), 4U, 0.09F, 0.03F);
  FillSequence(cacheValues.MutableDataAsFloat32(), 5U, 0.16F, 0.07F);

  const us4::AttentionCacheView cache{&cacheKeys, &cacheValues};
  std::string error;
  ASSERT_TRUE(
      us4::NeonAttention(query, key, value, neonOutput, true, cache, &error))
      << error;
  ASSERT_TRUE(us4::ScalarAttention(query, key, value, scalarOutput, true, cache,
                                   &error))
      << error;

  const float *neonValues = neonOutput.DataAsFloat32();
  const float *scalarValues = scalarOutput.DataAsFloat32();
  ASSERT_NE(neonValues, nullptr);
  ASSERT_NE(scalarValues, nullptr);
  for (std::size_t index = 0; index < 10U; ++index) {
    EXPECT_NEAR(neonValues[index], scalarValues[index], 1e-5F) << index;
  }
}

TEST(NeonContractTest, NeonAttentionFallsBackOnNonArmHosts) {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  GTEST_SKIP() << "This host builds the NEON path directly.";
#else
  us4::Tensor query({1, 4}, us4::DType::kFloat32, us4::DeviceType::kCpu);
  us4::Tensor key({2, 4}, us4::DType::kFloat32, us4::DeviceType::kCpu);
  us4::Tensor value({2, 2}, us4::DType::kFloat32, us4::DeviceType::kCpu);
  us4::Tensor neonOutput({1, 2}, us4::DType::kFloat32, us4::DeviceType::kCpu);
  us4::Tensor scalarOutput({1, 2}, us4::DType::kFloat32, us4::DeviceType::kCpu);

  FillSequence(query.MutableDataAsFloat32(), 4U, 0.50F, -0.25F);
  FillSequence(key.MutableDataAsFloat32(), 8U, 0.10F, 0.20F);
  FillSequence(value.MutableDataAsFloat32(), 4U, 0.35F, -0.10F);

  std::string error;
  ASSERT_TRUE(
      us4::NeonAttention(query, key, value, neonOutput, false, {}, &error))
      << error;
  ASSERT_TRUE(
      us4::ScalarAttention(query, key, value, scalarOutput, false, {}, &error))
      << error;
  EXPECT_NEAR(neonOutput.DataAsFloat32()[0], scalarOutput.DataAsFloat32()[0],
              1e-5F);
  EXPECT_NEAR(neonOutput.DataAsFloat32()[1], scalarOutput.DataAsFloat32()[1],
              1e-5F);
#endif
}
