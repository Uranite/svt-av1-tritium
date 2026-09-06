/*
* Copyright (c) 2016, Alliance for Open Media. All rights reserved
*
* This source code is subject to the terms of the BSD 2 Clause License and
* the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
* was not distributed with this source code in the LICENSE file, you can
* obtain it at www.aomedia.org/license/software. If the Alliance for Open
* Media Patent License 1.0 was not distributed with this source code in the
* PATENTS file, you can obtain it at www.aomedia.org/license/patent.
*/

#include <math.h>
#include <limits.h>
#include "daala_dist.h"
#include "common_dsp_rtcd.h"

#define OD_MINI(a, b) ((a) < (b) ? (a) : (b))
#define OD_MAXI(a, b) ((a) > (b) ? (a) : (b))

static int od_compute_var_4x4(uint16_t *x, int stride) {
    int sum;
    int s2;
    int i;
    sum = 0;
    s2 = 0;
    for (i = 0; i < 4; i++) {
        int j;
        for (j = 0; j < 4; j++) {
            int t;

            t = x[i * stride + j];
            sum += t;
            s2 += t * t;
        }
    }

    return (s2 - (sum * sum >> 4)) >> 4;
}

/* OD_DIST_LP_MID controls the frequency weighting filter used for computing
   the distortion. For a value X, the filter is [1 X 1]/(X + 2) and
   is applied both horizontally and vertically. For X=5, the filter is
   a good approximation for the OD_QM8_Q4_HVS quantization matrix. */
#define OD_DIST_LP_MID (5)
#define OD_DIST_LP_NORM (OD_DIST_LP_MID + 2)

void svt_aom_od_compute_diff_and_filter_h_c(
    const uint16_t *x, uint32_t x_stride,
    const uint16_t *y, uint32_t y_stride,
    int32_t *tmp, uint32_t tmp_stride,
    uint32_t width, uint32_t height)
{
    int mid = OD_DIST_LP_MID;
    for (uint32_t i = 0; i < height; i++) {
        const uint16_t *row_x = x + i * x_stride;
        const uint16_t *row_y = y + i * y_stride;
        int32_t *row_tmp = tmp + i * tmp_stride;

        // Compute difference
        int32_t e[128]; // width is at most 128 (MAX_TX_SIZE)
        for (uint32_t j = 0; j < width; j++) {
            e[j] = (int32_t)row_x[j] - (int32_t)row_y[j];
        }

        // Apply horizontal filter
        row_tmp[0] = mid * e[0] + 2 * e[1];
        row_tmp[width - 1] = mid * e[width - 1] + 2 * e[width - 2];
        for (uint32_t j = 1; j < width - 1; j++) {
            row_tmp[j] = mid * e[j] + e[j - 1] + e[j + 1];
        }
    }
}

void svt_aom_od_filter_v_c(
    const int32_t *tmp, uint32_t tmp_stride,
    int32_t *e_lp, uint32_t e_lp_stride,
    uint32_t width, uint32_t height)
{
    int mid = OD_DIST_LP_MID;
    for (uint32_t j = 0; j < width; j++) {
        e_lp[j] = mid * tmp[j] + 2 * tmp[tmp_stride + j];
        e_lp[(height - 1) * e_lp_stride + j] =
            mid * tmp[(height - 1) * tmp_stride + j] + 2 * tmp[(height - 2) * tmp_stride + j];
    }
    for (uint32_t i = 1; i < height - 1; i++) {
        for (uint32_t j = 0; j < width; j++) {
            e_lp[i * e_lp_stride + j] = mid * tmp[i * tmp_stride + j] +
                                        tmp[(i - 1) * tmp_stride + j] +
                                        tmp[(i + 1) * tmp_stride + j];
        }
    }
}

void svt_aom_od_compute_var_and_dist_8x8_c(
    const uint16_t *x, uint32_t x_stride,
    const uint16_t *y, uint32_t y_stride,
    const int32_t *e_lp, uint32_t e_lp_stride,
    uint32_t *varx, uint32_t *vary,
    uint64_t *sum_e_lp_sq)
{
    int idx = 0;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            varx[idx] = od_compute_var_4x4((uint16_t*)(x + 2 * i * x_stride + 2 * j), x_stride);
            vary[idx] = od_compute_var_4x4((uint16_t*)(y + 2 * i * y_stride + 2 * j), y_stride);
            idx++;
        }
    }

    uint64_t sum = 0;
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            sum += e_lp[i * e_lp_stride + j] * (int64_t)e_lp[i * e_lp_stride + j];
        }
    }
    *sum_e_lp_sq = sum;
}

double svt_aom_od_compute_dist(uint16_t *x, uint16_t *y, int bsize_w,
                               int bsize_h, int qindex, int activity_masking) {
    assert(bsize_w >= 8 && bsize_h >= 8);

    DECLARE_ALIGNED(16, od_coeff, tmp[MAX_TX_SQUARE]);
    DECLARE_ALIGNED(16, od_coeff, e_lp[MAX_TX_SQUARE]);
    svt_memset(tmp, 0, sizeof(tmp));
    svt_memset(e_lp, 0, sizeof(e_lp));

    svt_aom_od_compute_diff_and_filter_h(x, bsize_w, y, bsize_w, tmp, bsize_w, bsize_w, bsize_h);
    svt_aom_od_filter_v(tmp, bsize_w, e_lp, bsize_w, bsize_w, bsize_h);

    double sum = 0;

    for (int i = 0; i < bsize_h; i += 8) {
        for (int j = 0; j < bsize_w; j += 8) {
            uint32_t varx_arr[9];
            uint32_t vary_arr[9];
            uint64_t sum_e_lp_sq;
            svt_aom_od_compute_var_and_dist_8x8(
                &x[i * bsize_w + j], bsize_w,
                &y[i * bsize_w + j], bsize_w,
                &e_lp[i * bsize_w + j], bsize_w,
                varx_arr, vary_arr, &sum_e_lp_sq);

            int min_var = INT_MAX;
            double mean_var = 0;
            double vardist = 0;
            for (int k = 0; k < 9; k++) {
                int vx = varx_arr[k];
                int vy = vary_arr[k];
                min_var = OD_MINI(min_var, vx);
                mean_var += 1. / (1 + vx);
                vardist += vx - 2 * sqrt(vx * (double)vy) + vy;
            }

            double var_stat;
            double calibration;
            if (activity_masking) {
                calibration = 1.95;
                var_stat = 9. / mean_var;
            } else {
                calibration = 1.62;
                var_stat = min_var;
            }
            double activity = calibration * pow(.25 + var_stat, -1. / 6);
            double sum_val = (double)sum_e_lp_sq;
            sum_val *= 1. / (OD_DIST_LP_NORM * OD_DIST_LP_NORM * OD_DIST_LP_NORM * OD_DIST_LP_NORM);

            sum += activity * activity * (sum_val + vardist);
        }
    }

    /* Scale according to linear regression against SSE, for 8x8 blocks. */
    if (activity_masking) {
        sum *= 2.2 + (1.7 - 2.2) * (qindex - 99) / (210 - 99) +
               (qindex < 99 ? 2.5 * (qindex - 99) / 99 * (qindex - 99) / 99 : 0);
    } else {
        sum *= qindex >= 128
                   ? 1.4 + (0.9 - 1.4) * (qindex - 128) / (209 - 128)
                   : qindex <= 43 ? 1.5 + (2.0 - 1.5) * (qindex - 43) / (16 - 43)
                                  : 1.5 + (1.4 - 1.5) * (qindex - 43) / (128 - 43);
    }

    return sum;
}
