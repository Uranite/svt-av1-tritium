/*
 * Copyright(c) 2026 Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and the
 * Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License was
 * not distributed with this source code in the LICENSE file, you can obtain it
 * at https://www.aomedia.org/license/software-license. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at
 * https://www.aomedia.org/license/patent-license.
 */

#include <stdint.h>
#include <initializer_list>
#include <tuple>
#include <utility>

#include "aom_dsp_rtcd.h"
#include "gtest/gtest.h"
#include "random.h"
#include "svt_malloc.h"
#include "util.h"

using svt_av1_test_tool::SVTRandom;

namespace {
#if OPT_TUNE_VMAF

static const int kSvtAv1MaxWidth = 16384;
static const int kSvtAv1MaxHeight = 8704;

// Cover minimal 8x8 tiling, a typical 1080p frame, and max legal width/height
// cases.
static const std::pair<int, int> kVmafSizes[] = {
    {8, 8},
    {1920, 1080},
    {kSvtAv1MaxWidth, 8},
    {8, kSvtAv1MaxHeight},
};
static const int kVmafWidths[] = {8, 1920, kSvtAv1MaxWidth};

using VmafAvgMadFunc = uint32_t (*)(const uint8_t *src, int width, int height,
                                    int stride);

using VmafAvgMadParam = std::tuple<std::pair<int, int>, VmafAvgMadFunc>;

class VmafAvgMadTest : public ::testing::TestWithParam<VmafAvgMadParam> {
  public:
    VmafAvgMadTest() : size_(TEST_GET_PARAM(0)), test_func_(TEST_GET_PARAM(1)) {
    }

    ~VmafAvgMadTest() override = default;

  protected:
    static const int kIterations = 10;

    void SetUp() override {
        const int width = size_.first;
        const int height = size_.second;
        src_ = static_cast<uint8_t *>(
            svt_aom_memalign(16, (size_t)width * height));
        ASSERT_NE(src_, nullptr);
    }

    void TearDown() override {
        svt_aom_free(src_);
        src_ = nullptr;
    }

    std::pair<int, int> size_;
    VmafAvgMadFunc test_func_;
    uint8_t *src_ = nullptr;

    void run_match_test() {
        SVTRandom rnd(8, false);
        const int width = size_.first;
        const int height = size_.second;
        const int stride = width;

        for (int i = 0; i < kIterations; ++i) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < stride; ++x) {
                    src_[y * stride + x] = rnd.Rand8();
                }
            }

            const uint32_t res_ref =
                svt_vmaf_compute_avg_mad_c(src_, width, height, stride);
            const uint32_t res_tst = test_func_(src_, width, height, stride);
            ASSERT_EQ(res_ref, res_tst) << "iteration " << i;
        }
    }

    void run_max_input_test() {
        const int width = size_.first;
        const int height = size_.second;
        const int stride = width;

        for (int i = 0; i < height * stride; ++i) {
            src_[i] = 255;
        }

        const uint32_t res_ref =
            svt_vmaf_compute_avg_mad_c(src_, width, height, stride);
        const uint32_t res_tst = test_func_(src_, width, height, stride);
        ASSERT_EQ(res_ref, res_tst);
    }

    void run_max_activity_test() {
        const int width = size_.first;
        const int height = size_.second;
        const int stride = width;

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < stride; ++x) {
                src_[y * stride + x] = (x & 1) ? 255 : 0;
            }
        }

        const uint32_t res_ref =
            svt_vmaf_compute_avg_mad_c(src_, width, height, stride);
        const uint32_t res_tst = test_func_(src_, width, height, stride);
        ASSERT_EQ(res_ref, res_tst);
    }
};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(VmafAvgMadTest);

TEST_P(VmafAvgMadTest, MatchTest) {
    run_match_test();
}

TEST_P(VmafAvgMadTest, MaxInputTest) {
    run_max_input_test();
}

TEST_P(VmafAvgMadTest, MaxActivityTest) {
    run_max_activity_test();
}

#ifdef ARCH_X86_64
INSTANTIATE_TEST_SUITE_P(
    AVX2, VmafAvgMadTest,
    ::testing::Combine(::testing::ValuesIn(kVmafSizes),
                       ::testing::Values(&svt_vmaf_compute_avg_mad_avx2)));
#endif  // ARCH_X86_64

#ifdef ARCH_AARCH64
INSTANTIATE_TEST_SUITE_P(
    NEON, VmafAvgMadTest,
    ::testing::Combine(::testing::ValuesIn(kVmafSizes),
                       ::testing::Values(&svt_vmaf_compute_avg_mad_neon)));

#if HAVE_NEON_DOTPROD
INSTANTIATE_TEST_SUITE_P(
    NEON_DOTPROD, VmafAvgMadTest,
    ::testing::Combine(
        ::testing::ValuesIn(kVmafSizes),
        ::testing::Values(&svt_vmaf_compute_avg_mad_neon_dotprod)));
#endif  // HAVE_NEON_DOTPROD

#if HAVE_NEON_I8MM
INSTANTIATE_TEST_SUITE_P(
    NEON_I8MM, VmafAvgMadTest,
    ::testing::Combine(::testing::ValuesIn(kVmafSizes),
                       ::testing::Values(&svt_vmaf_compute_avg_mad_neon_i8mm)));
#endif  // HAVE_NEON_I8MM
#endif  // ARCH_AARCH64

using VmafUnsharpRowFunc = void (*)(const uint8_t *src, const int16_t *blur,
                                    uint8_t *dst, int width, int amount,
                                    int32_t max_delta);

using VmafUnsharpRowParam = std::tuple<int, int, int, VmafUnsharpRowFunc>;

class VmafUnsharpRowTest
    : public ::testing::TestWithParam<VmafUnsharpRowParam> {
  public:
    VmafUnsharpRowTest()
        : width_(TEST_GET_PARAM(0)),
          amount_(TEST_GET_PARAM(1)),
          max_delta_(TEST_GET_PARAM(2)),
          test_func_(TEST_GET_PARAM(3)) {
    }

    ~VmafUnsharpRowTest() override = default;

  protected:
    static const int kIterations = 100;

    void SetUp() override {
        src_ = static_cast<uint8_t *>(
            svt_aom_memalign(16, (size_t)width_ * sizeof(*src_)));
        blur_ = static_cast<int16_t *>(
            svt_aom_memalign(16, (size_t)width_ * sizeof(*blur_)));
        dst_ref_ = static_cast<uint8_t *>(
            svt_aom_memalign(16, (size_t)width_ * sizeof(*dst_ref_)));
        dst_tst_ = static_cast<uint8_t *>(
            svt_aom_memalign(16, (size_t)width_ * sizeof(*dst_tst_)));

        ASSERT_NE(src_, nullptr);
        ASSERT_NE(blur_, nullptr);
        ASSERT_NE(dst_ref_, nullptr);
        ASSERT_NE(dst_tst_, nullptr);
    }

    void TearDown() override {
        svt_aom_free(src_);
        svt_aom_free(blur_);
        svt_aom_free(dst_ref_);
        svt_aom_free(dst_tst_);
    }

    int width_;
    int amount_;
    int max_delta_;
    VmafUnsharpRowFunc test_func_;
    uint8_t *src_ = nullptr;
    int16_t *blur_ = nullptr;
    uint8_t *dst_ref_ = nullptr;
    uint8_t *dst_tst_ = nullptr;

    void run_match_test() {
        SVTRandom rnd(8, false);

        for (int i = 0; i < kIterations; ++i) {
            for (int x = 0; x < width_; ++x) {
                src_[x] = rnd.Rand8();
                blur_[x] = rnd.Rand8();
            }

            svt_vmaf_apply_unsharp_row_c(
                src_, blur_, dst_ref_, width_, amount_, max_delta_);
            test_func_(src_, blur_, dst_tst_, width_, amount_, max_delta_);
            for (int x = 0; x < width_; ++x) {
                ASSERT_EQ(dst_ref_[x], dst_tst_[x])
                    << "x " << x << ", iteration " << i;
            }
        }
    }
};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(VmafUnsharpRowTest);

TEST_P(VmafUnsharpRowTest, MatchTest) {
    run_match_test();
}

#ifdef ARCH_X86_64
INSTANTIATE_TEST_SUITE_P(
    AVX2, VmafUnsharpRowTest,
    ::testing::Combine(::testing::ValuesIn(kVmafWidths),
                       ::testing::Values(19660, 32767),
                       ::testing::Values(8, 12),
                       ::testing::Values(&svt_vmaf_apply_unsharp_row_avx2)));
#endif  // ARCH_X86_64

#ifdef ARCH_AARCH64
INSTANTIATE_TEST_SUITE_P(
    NEON, VmafUnsharpRowTest,
    ::testing::Combine(::testing::ValuesIn(kVmafWidths),
                       ::testing::Values(19660, 32767),
                       ::testing::Values(8, 12),
                       ::testing::Values(&svt_vmaf_apply_unsharp_row_neon)));
#if HAVE_SVE2
INSTANTIATE_TEST_SUITE_P(
    SVE2, VmafUnsharpRowTest,
    ::testing::Combine(::testing::ValuesIn(kVmafWidths),
                       ::testing::Values(19660, 32767),
                       ::testing::Values(8, 12),
                       ::testing::Values(&svt_vmaf_apply_unsharp_row_sve2)));
#endif  // HAVE_SVE2
#endif  // ARCH_AARCH64

using VmafVpassRowFunc = void (*)(const uint32_t *hpass, uint32_t *sc0,
                                  uint32_t *sc1, uint32_t *sc2, uint32_t *sc3,
                                  int16_t *blur_row, int alloc_width, int width,
                                  int steps_x, int do_output);

using VmafVpassRowParam = std::tuple<int, int, VmafVpassRowFunc>;

class VmafVpassRowTest : public ::testing::TestWithParam<VmafVpassRowParam> {
  public:
    VmafVpassRowTest()
        : width_(TEST_GET_PARAM(0)),
          do_output_(TEST_GET_PARAM(1)),
          test_func_(TEST_GET_PARAM(2)) {
    }

    ~VmafVpassRowTest() override = default;

  protected:
    static const int kIterations = 100;

    void SetUp() override {
        const int steps_x = 2;
        const int alloc_width = width_ + 2 * steps_x;
        hpass_ = static_cast<uint32_t *>(
            svt_aom_memalign(16, (size_t)alloc_width * sizeof(*hpass_)));
        sc0_ref_ = static_cast<uint32_t *>(
            svt_aom_memalign(16, (size_t)alloc_width * sizeof(*sc0_ref_)));
        sc1_ref_ = static_cast<uint32_t *>(
            svt_aom_memalign(16, (size_t)alloc_width * sizeof(*sc1_ref_)));
        sc2_ref_ = static_cast<uint32_t *>(
            svt_aom_memalign(16, (size_t)alloc_width * sizeof(*sc2_ref_)));
        sc3_ref_ = static_cast<uint32_t *>(
            svt_aom_memalign(16, (size_t)alloc_width * sizeof(*sc3_ref_)));
        sc0_tst_ = static_cast<uint32_t *>(
            svt_aom_memalign(16, (size_t)alloc_width * sizeof(*sc0_tst_)));
        sc1_tst_ = static_cast<uint32_t *>(
            svt_aom_memalign(16, (size_t)alloc_width * sizeof(*sc1_tst_)));
        sc2_tst_ = static_cast<uint32_t *>(
            svt_aom_memalign(16, (size_t)alloc_width * sizeof(*sc2_tst_)));
        sc3_tst_ = static_cast<uint32_t *>(
            svt_aom_memalign(16, (size_t)alloc_width * sizeof(*sc3_tst_)));
        blur_ref_ = static_cast<int16_t *>(
            svt_aom_memalign(16, (size_t)width_ * sizeof(*blur_ref_)));
        blur_tst_ = static_cast<int16_t *>(
            svt_aom_memalign(16, (size_t)width_ * sizeof(*blur_tst_)));

        ASSERT_NE(hpass_, nullptr);
        ASSERT_NE(sc0_ref_, nullptr);
        ASSERT_NE(sc1_ref_, nullptr);
        ASSERT_NE(sc2_ref_, nullptr);
        ASSERT_NE(sc3_ref_, nullptr);
        ASSERT_NE(sc0_tst_, nullptr);
        ASSERT_NE(sc1_tst_, nullptr);
        ASSERT_NE(sc2_tst_, nullptr);
        ASSERT_NE(sc3_tst_, nullptr);
        ASSERT_NE(blur_ref_, nullptr);
        ASSERT_NE(blur_tst_, nullptr);
    }

    void TearDown() override {
        svt_aom_free(hpass_);
        svt_aom_free(sc0_ref_);
        svt_aom_free(sc1_ref_);
        svt_aom_free(sc2_ref_);
        svt_aom_free(sc3_ref_);
        svt_aom_free(sc0_tst_);
        svt_aom_free(sc1_tst_);
        svt_aom_free(sc2_tst_);
        svt_aom_free(sc3_tst_);
        svt_aom_free(blur_ref_);
        svt_aom_free(blur_tst_);
    }

    int width_;
    int do_output_;
    VmafVpassRowFunc test_func_;
    uint32_t *hpass_ = nullptr;
    uint32_t *sc0_ref_ = nullptr;
    uint32_t *sc1_ref_ = nullptr;
    uint32_t *sc2_ref_ = nullptr;
    uint32_t *sc3_ref_ = nullptr;
    uint32_t *sc0_tst_ = nullptr;
    uint32_t *sc1_tst_ = nullptr;
    uint32_t *sc2_tst_ = nullptr;
    uint32_t *sc3_tst_ = nullptr;
    int16_t *blur_ref_ = nullptr;
    int16_t *blur_tst_ = nullptr;

    void run_match_test() {
        SVTRandom rnd(8, false);
        const int steps_x = 2;
        const int alloc_width = width_ + 2 * steps_x;

        for (int i = 0; i < kIterations; ++i) {
            for (int x = 0; x < alloc_width; ++x) {
                hpass_[x] = rnd.Rand8();
                sc0_ref_[x] = rnd.Rand8();
                sc1_ref_[x] = rnd.Rand8();
                sc2_ref_[x] = rnd.Rand8();
                sc3_ref_[x] = rnd.Rand8();
                sc0_tst_[x] = sc0_ref_[x];
                sc1_tst_[x] = sc1_ref_[x];
                sc2_tst_[x] = sc2_ref_[x];
                sc3_tst_[x] = sc3_ref_[x];
            }
            for (int x = 0; x < width_; ++x) {
                blur_ref_[x] = rnd.Rand8();
                blur_tst_[x] = blur_ref_[x];
            }

            svt_vmaf_vpass_row_c(hpass_,
                                 sc0_ref_,
                                 sc1_ref_,
                                 sc2_ref_,
                                 sc3_ref_,
                                 blur_ref_,
                                 alloc_width,
                                 width_,
                                 steps_x,
                                 do_output_);
            test_func_(hpass_,
                       sc0_tst_,
                       sc1_tst_,
                       sc2_tst_,
                       sc3_tst_,
                       blur_tst_,
                       alloc_width,
                       width_,
                       steps_x,
                       do_output_);

            for (int x = 0; x < alloc_width; ++x) {
                ASSERT_EQ(sc0_ref_[x], sc0_tst_[x])
                    << "sc0 mismatch, x " << x << ", alloc_width "
                    << alloc_width << ", iteration " << i;
                ASSERT_EQ(sc1_ref_[x], sc1_tst_[x])
                    << "sc1 mismatch, x " << x << ", alloc_width "
                    << alloc_width << ", iteration " << i;
                ASSERT_EQ(sc2_ref_[x], sc2_tst_[x])
                    << "sc2 mismatch, x " << x << ", alloc_width "
                    << alloc_width << ", iteration " << i;
                ASSERT_EQ(sc3_ref_[x], sc3_tst_[x])
                    << "sc3 mismatch, x " << x << ", alloc_width "
                    << alloc_width << ", iteration " << i;
            }
            for (int x = 0; x < width_; ++x) {
                ASSERT_EQ(blur_ref_[x], blur_tst_[x])
                    << "blur_row mismatch, x " << x << ", alloc_width "
                    << alloc_width << ", iteration " << i;
            }
        }
    }
};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(VmafVpassRowTest);

TEST_P(VmafVpassRowTest, MatchTest) {
    run_match_test();
}

#ifdef ARCH_X86_64
INSTANTIATE_TEST_SUITE_P(
    AVX2, VmafVpassRowTest,
    ::testing::Combine(::testing::ValuesIn(kVmafWidths),
                       ::testing::Values(0, 1),
                       ::testing::Values(&svt_vmaf_vpass_row_avx2)));
#endif  // ARCH_X86_64

#ifdef ARCH_AARCH64
INSTANTIATE_TEST_SUITE_P(
    NEON, VmafVpassRowTest,
    ::testing::Combine(::testing::ValuesIn(kVmafWidths),
                       ::testing::Values(0, 1),
                       ::testing::Values(&svt_vmaf_vpass_row_neon)));
#endif  // ARCH_AARCH64

#endif  // OPT_TUNE_VMAF

}  // namespace
