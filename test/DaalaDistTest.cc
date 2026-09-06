/*
 * Copyright (c) 2019, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at https://www.aomedia.org/license/software-license. If the
 * Alliance for Open Media Patent License 1.0 was not distributed with this
 * source code in the PATENTS file, you can obtain it at
 * https://www.aomedia.org/license/patent-license.
 */

#include <stdlib.h>
#include <string.h>

#include "gtest/gtest.h"
#include "definitions.h"
#include "random.h"
#include "svt_time.h"
#include "svt_malloc.h"
#include "util.h"
#include "utility.h"
#include "common_dsp_rtcd.h"

using svt_av1_test_tool::SVTRandom;

namespace {

// Test 1: svt_aom_od_compute_diff_and_filter_h
typedef void (*ComputeDiffAndFilterHFunc)(
    const uint16_t *x, uint32_t x_stride,
    const uint16_t *y, uint32_t y_stride,
    int32_t *tmp, uint32_t tmp_stride,
    uint32_t width, uint32_t height);

typedef std::tuple<uint32_t, uint32_t, ComputeDiffAndFilterHFunc, ComputeDiffAndFilterHFunc> DiffFilterHParam;

class DaalaDiffFilterHTest : public ::testing::TestWithParam<DiffFilterHParam> {
  protected:
    DaalaDiffFilterHTest() {
        width_ = std::get<0>(GetParam());
        height_ = std::get<1>(GetParam());
        ref_func_ = std::get<2>(GetParam());
        tst_func_ = std::get<3>(GetParam());
    }

    void SetUp() override {
        x_stride_ = width_ + 8;
        y_stride_ = width_ + 16;
        tmp_stride_ = width_ + 4;

        x_ = (uint16_t*)svt_aom_memalign(32, sizeof(uint16_t) * x_stride_ * height_);
        y_ = (uint16_t*)svt_aom_memalign(32, sizeof(uint16_t) * y_stride_ * height_);
        tmp_ref_ = (int32_t*)svt_aom_memalign(32, sizeof(int32_t) * tmp_stride_ * height_);
        tmp_tst_ = (int32_t*)svt_aom_memalign(32, sizeof(int32_t) * tmp_stride_ * height_);
    }

    void TearDown() override {
        svt_aom_free(x_);
        svt_aom_free(y_);
        svt_aom_free(tmp_ref_);
        svt_aom_free(tmp_tst_);
    }

    void RunCheck() {
        SVTRandom rnd(0, 255);
        for (uint32_t i = 0; i < height_; i++) {
            for (uint32_t j = 0; j < width_; j++) {
                x_[i * x_stride_ + j] = rnd.random();
                y_[i * y_stride_ + j] = rnd.random();
            }
        }
        memset(tmp_ref_, 0, sizeof(int32_t) * tmp_stride_ * height_);
        memset(tmp_tst_, 0, sizeof(int32_t) * tmp_stride_ * height_);

        ref_func_(x_, x_stride_, y_, y_stride_, tmp_ref_, tmp_stride_, width_, height_);
        tst_func_(x_, x_stride_, y_, y_stride_, tmp_tst_, tmp_stride_, width_, height_);

        for (uint32_t i = 0; i < height_; i++) {
            for (uint32_t j = 0; j < width_; j++) {
                ASSERT_EQ(tmp_ref_[i * tmp_stride_ + j], tmp_tst_[i * tmp_stride_ + j])
                    << "Mismatch at row " << i << ", col " << j;
            }
        }
    }

    void RunSpeed() {
        SVTRandom rnd(0, 255);
        for (uint32_t i = 0; i < height_; i++) {
            for (uint32_t j = 0; j < width_; j++) {
                x_[i * x_stride_ + j] = rnd.random();
                y_[i * y_stride_ + j] = rnd.random();
            }
        }

        uint64_t start_sec, start_usec;
        uint64_t finish_sec, finish_usec;

        svt_av1_get_time(&start_sec, &start_usec);
        for (int k = 0; k < 100000; k++) {
            ref_func_(x_, x_stride_, y_, y_stride_, tmp_ref_, tmp_stride_, width_, height_);
        }
        svt_av1_get_time(&finish_sec, &finish_usec);
        double ref_time = svt_av1_compute_overall_elapsed_time_ms(start_sec, start_usec, finish_sec, finish_usec);

        svt_av1_get_time(&start_sec, &start_usec);
        for (int k = 0; k < 100000; k++) {
            tst_func_(x_, x_stride_, y_, y_stride_, tmp_tst_, tmp_stride_, width_, height_);
        }
        svt_av1_get_time(&finish_sec, &finish_usec);
        double tst_time = svt_av1_compute_overall_elapsed_time_ms(start_sec, start_usec, finish_sec, finish_usec);

        printf("DaalaDiffFilterH [%ux%u]: Ref = %6.2f ms, Test = %6.2f ms (Speedup: %.2f)\n",
               width_, height_, ref_time, tst_time,
               tst_time > 0 ? (ref_time / tst_time) : 0.0);
    }

    uint32_t width_, height_;
    uint16_t *x_, *y_;
    int32_t *tmp_ref_, *tmp_tst_;
    uint32_t x_stride_, y_stride_, tmp_stride_;
    ComputeDiffAndFilterHFunc ref_func_;
    ComputeDiffAndFilterHFunc tst_func_;
};

TEST_P(DaalaDiffFilterHTest, Correctness) {
    RunCheck();
}

TEST_P(DaalaDiffFilterHTest, Speed) {
    RunSpeed();
}

// Test 2: svt_aom_od_filter_v
typedef void (*FilterVFunc)(
    const int32_t *tmp, uint32_t tmp_stride,
    int32_t *e_lp, uint32_t e_lp_stride,
    uint32_t width, uint32_t height);

typedef std::tuple<uint32_t, uint32_t, FilterVFunc, FilterVFunc> FilterVParam;

class DaalaFilterVTest : public ::testing::TestWithParam<FilterVParam> {
  protected:
    DaalaFilterVTest() {
        width_ = std::get<0>(GetParam());
        height_ = std::get<1>(GetParam());
        ref_func_ = std::get<2>(GetParam());
        tst_func_ = std::get<3>(GetParam());
    }

    void SetUp() override {
        tmp_stride_ = width_ + 8;
        e_lp_stride_ = width_ + 16;

        tmp_ = (int32_t*)svt_aom_memalign(32, sizeof(int32_t) * tmp_stride_ * height_);
        e_lp_ref_ = (int32_t*)svt_aom_memalign(32, sizeof(int32_t) * e_lp_stride_ * height_);
        e_lp_tst_ = (int32_t*)svt_aom_memalign(32, sizeof(int32_t) * e_lp_stride_ * height_);
    }

    void TearDown() override {
        svt_aom_free(tmp_);
        svt_aom_free(e_lp_ref_);
        svt_aom_free(e_lp_tst_);
    }

    void RunCheck() {
        SVTRandom rnd(-1000, 1000);
        for (uint32_t i = 0; i < height_; i++) {
            for (uint32_t j = 0; j < width_; j++) {
                tmp_[i * tmp_stride_ + j] = rnd.random();
            }
        }
        memset(e_lp_ref_, 0, sizeof(int32_t) * e_lp_stride_ * height_);
        memset(e_lp_tst_, 0, sizeof(int32_t) * e_lp_stride_ * height_);

        ref_func_(tmp_, tmp_stride_, e_lp_ref_, e_lp_stride_, width_, height_);
        tst_func_(tmp_, tmp_stride_, e_lp_tst_, e_lp_stride_, width_, height_);

        for (uint32_t i = 0; i < height_; i++) {
            for (uint32_t j = 0; j < width_; j++) {
                ASSERT_EQ(e_lp_ref_[i * e_lp_stride_ + j], e_lp_tst_[i * e_lp_stride_ + j])
                    << "Mismatch at row " << i << ", col " << j;
            }
        }
    }

    void RunSpeed() {
        SVTRandom rnd(-1000, 1000);
        for (uint32_t i = 0; i < height_; i++) {
            for (uint32_t j = 0; j < width_; j++) {
                tmp_[i * tmp_stride_ + j] = rnd.random();
            }
        }

        uint64_t start_sec, start_usec;
        uint64_t finish_sec, finish_usec;

        svt_av1_get_time(&start_sec, &start_usec);
        for (int k = 0; k < 100000; k++) {
            ref_func_(tmp_, tmp_stride_, e_lp_ref_, e_lp_stride_, width_, height_);
        }
        svt_av1_get_time(&finish_sec, &finish_usec);
        double ref_time = svt_av1_compute_overall_elapsed_time_ms(start_sec, start_usec, finish_sec, finish_usec);

        svt_av1_get_time(&start_sec, &start_usec);
        for (int k = 0; k < 100000; k++) {
            tst_func_(tmp_, tmp_stride_, e_lp_tst_, e_lp_stride_, width_, height_);
        }
        svt_av1_get_time(&finish_sec, &finish_usec);
        double tst_time = svt_av1_compute_overall_elapsed_time_ms(start_sec, start_usec, finish_sec, finish_usec);

        printf("DaalaFilterV [%ux%u]: Ref = %6.2f ms, Test = %6.2f ms (Speedup: %.2f)\n",
               width_, height_, ref_time, tst_time,
               tst_time > 0 ? (ref_time / tst_time) : 0.0);
    }

    uint32_t width_, height_;
    int32_t *tmp_;
    int32_t *e_lp_ref_, *e_lp_tst_;
    uint32_t tmp_stride_, e_lp_stride_;
    FilterVFunc ref_func_;
    FilterVFunc tst_func_;
};

TEST_P(DaalaFilterVTest, Correctness) {
    RunCheck();
}

TEST_P(DaalaFilterVTest, Speed) {
    RunSpeed();
}

// Test 3: svt_aom_od_compute_var_and_dist_8x8
typedef void (*ComputeVarDist8x8Func)(
    const uint16_t *x, uint32_t x_stride,
    const uint16_t *y, uint32_t y_stride,
    const int32_t *e_lp, uint32_t e_lp_stride,
    uint32_t *varx, uint32_t *vary,
    uint64_t *sum_e_lp_sq);

typedef std::tuple<ComputeVarDist8x8Func, ComputeVarDist8x8Func> VarDist8x8Param;

class DaalaVarDist8x8Test : public ::testing::TestWithParam<VarDist8x8Param> {
  protected:
    DaalaVarDist8x8Test() {
        ref_func_ = std::get<0>(GetParam());
        tst_func_ = std::get<1>(GetParam());
    }

    void SetUp() override {
        x_stride_ = 16;
        y_stride_ = 24;
        e_lp_stride_ = 12;

        x_ = (uint16_t*)svt_aom_memalign(32, sizeof(uint16_t) * x_stride_ * 8);
        y_ = (uint16_t*)svt_aom_memalign(32, sizeof(uint16_t) * y_stride_ * 8);
        e_lp_ = (int32_t*)svt_aom_memalign(32, sizeof(int32_t) * e_lp_stride_ * 8);
    }

    void TearDown() override {
        svt_aom_free(x_);
        svt_aom_free(y_);
        svt_aom_free(e_lp_);
    }

    void RunCheck() {
        SVTRandom rnd_p(0, 255);
        SVTRandom rnd_e(-10000, 10000);
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                x_[i * x_stride_ + j] = rnd_p.random();
                y_[i * y_stride_ + j] = rnd_p.random();
                e_lp_[i * e_lp_stride_ + j] = rnd_e.random();
            }
        }

        uint32_t varx_ref[9] = {0};
        uint32_t vary_ref[9] = {0};
        uint64_t sum_ref = 0;

        uint32_t varx_tst[9] = {0};
        uint32_t vary_tst[9] = {0};
        uint64_t sum_tst = 0;

        ref_func_(x_, x_stride_, y_, y_stride_, e_lp_, e_lp_stride_, varx_ref, vary_ref, &sum_ref);
        tst_func_(x_, x_stride_, y_, y_stride_, e_lp_, e_lp_stride_, varx_tst, vary_tst, &sum_tst);

        ASSERT_EQ(sum_ref, sum_tst) << "Sum of squares mismatch";
        for (int k = 0; k < 9; k++) {
            ASSERT_EQ(varx_ref[k], varx_tst[k]) << "VarX mismatch at index " << k;
            ASSERT_EQ(vary_ref[k], vary_tst[k]) << "VarY mismatch at index " << k;
        }
    }

    void RunSpeed() {
        SVTRandom rnd_p(0, 255);
        SVTRandom rnd_e(-10000, 10000);
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                x_[i * x_stride_ + j] = rnd_p.random();
                y_[i * y_stride_ + j] = rnd_p.random();
                e_lp_[i * e_lp_stride_ + j] = rnd_e.random();
            }
        }

        uint32_t varx_ref[9] = {0};
        uint32_t vary_ref[9] = {0};
        uint64_t sum_ref = 0;

        uint32_t varx_tst[9] = {0};
        uint32_t vary_tst[9] = {0};
        uint64_t sum_tst = 0;

        uint64_t start_sec, start_usec;
        uint64_t finish_sec, finish_usec;

        svt_av1_get_time(&start_sec, &start_usec);
        for (int k = 0; k < 200000; k++) {
            ref_func_(x_, x_stride_, y_, y_stride_, e_lp_, e_lp_stride_, varx_ref, vary_ref, &sum_ref);
        }
        svt_av1_get_time(&finish_sec, &finish_usec);
        double ref_time = svt_av1_compute_overall_elapsed_time_ms(start_sec, start_usec, finish_sec, finish_usec);

        svt_av1_get_time(&start_sec, &start_usec);
        for (int k = 0; k < 200000; k++) {
            tst_func_(x_, x_stride_, y_, y_stride_, e_lp_, e_lp_stride_, varx_tst, vary_tst, &sum_tst);
        }
        svt_av1_get_time(&finish_sec, &finish_usec);
        double tst_time = svt_av1_compute_overall_elapsed_time_ms(start_sec, start_usec, finish_sec, finish_usec);

        printf("DaalaVarDist8x8: Ref = %6.2f ms, Test = %6.2f ms (Speedup: %.2f)\n",
               ref_time, tst_time,
               tst_time > 0 ? (ref_time / tst_time) : 0.0);
    }

    uint16_t *x_, *y_;
    int32_t *e_lp_;
    uint32_t x_stride_, y_stride_, e_lp_stride_;
    ComputeVarDist8x8Func ref_func_;
    ComputeVarDist8x8Func tst_func_;
};

TEST_P(DaalaVarDist8x8Test, Correctness) {
    RunCheck();
}

TEST_P(DaalaVarDist8x8Test, Speed) {
    RunSpeed();
}

// Instantiate test suites for x86_64
#ifdef ARCH_X86_64
INSTANTIATE_TEST_SUITE_P(
    SSE4_1, DaalaDiffFilterHTest,
    ::testing::Values(
        std::make_tuple(8, 8, svt_aom_od_compute_diff_and_filter_h_c, svt_aom_od_compute_diff_and_filter_h_sse4_1),
        std::make_tuple(16, 16, svt_aom_od_compute_diff_and_filter_h_c, svt_aom_od_compute_diff_and_filter_h_sse4_1),
        std::make_tuple(32, 32, svt_aom_od_compute_diff_and_filter_h_c, svt_aom_od_compute_diff_and_filter_h_sse4_1),
        std::make_tuple(64, 64, svt_aom_od_compute_diff_and_filter_h_c, svt_aom_od_compute_diff_and_filter_h_sse4_1),
        std::make_tuple(128, 128, svt_aom_od_compute_diff_and_filter_h_c, svt_aom_od_compute_diff_and_filter_h_sse4_1)
    )
);

INSTANTIATE_TEST_SUITE_P(
    AVX2, DaalaDiffFilterHTest,
    ::testing::Values(
        std::make_tuple(16, 16, svt_aom_od_compute_diff_and_filter_h_c, svt_aom_od_compute_diff_and_filter_h_avx2),
        std::make_tuple(32, 32, svt_aom_od_compute_diff_and_filter_h_c, svt_aom_od_compute_diff_and_filter_h_avx2),
        std::make_tuple(64, 64, svt_aom_od_compute_diff_and_filter_h_c, svt_aom_od_compute_diff_and_filter_h_avx2),
        std::make_tuple(128, 128, svt_aom_od_compute_diff_and_filter_h_c, svt_aom_od_compute_diff_and_filter_h_avx2)
    )
);

INSTANTIATE_TEST_SUITE_P(
    SSE4_1, DaalaFilterVTest,
    ::testing::Values(
        std::make_tuple(8, 8, svt_aom_od_filter_v_c, svt_aom_od_filter_v_sse4_1),
        std::make_tuple(16, 16, svt_aom_od_filter_v_c, svt_aom_od_filter_v_sse4_1),
        std::make_tuple(32, 32, svt_aom_od_filter_v_c, svt_aom_od_filter_v_sse4_1),
        std::make_tuple(64, 64, svt_aom_od_filter_v_c, svt_aom_od_filter_v_sse4_1),
        std::make_tuple(128, 128, svt_aom_od_filter_v_c, svt_aom_od_filter_v_sse4_1)
    )
);

INSTANTIATE_TEST_SUITE_P(
    AVX2, DaalaFilterVTest,
    ::testing::Values(
        std::make_tuple(8, 8, svt_aom_od_filter_v_c, svt_aom_od_filter_v_avx2),
        std::make_tuple(16, 16, svt_aom_od_filter_v_c, svt_aom_od_filter_v_avx2),
        std::make_tuple(32, 32, svt_aom_od_filter_v_c, svt_aom_od_filter_v_avx2),
        std::make_tuple(64, 64, svt_aom_od_filter_v_c, svt_aom_od_filter_v_avx2),
        std::make_tuple(128, 128, svt_aom_od_filter_v_c, svt_aom_od_filter_v_avx2)
    )
);

INSTANTIATE_TEST_SUITE_P(
    SSE4_1, DaalaVarDist8x8Test,
    ::testing::Values(
        std::make_tuple(svt_aom_od_compute_var_and_dist_8x8_c, svt_aom_od_compute_var_and_dist_8x8_sse4_1)
    )
);

INSTANTIATE_TEST_SUITE_P(
    AVX2, DaalaVarDist8x8Test,
    ::testing::Values(
        std::make_tuple(svt_aom_od_compute_var_and_dist_8x8_c, svt_aom_od_compute_var_and_dist_8x8_avx2)
    )
);
#endif

} // namespace
