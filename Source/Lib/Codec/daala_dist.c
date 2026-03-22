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

static double od_compute_dist_8x8(int use_activity_masking, uint16_t *x,
                                  uint16_t *y, od_coeff *e_lp, int stride) {
    double sum;
    int min_var;
    double mean_var;
    double var_stat;
    double activity;
    double calibration;
    int i;
    int j;
    double vardist;

    vardist = 0;

    min_var = INT_MAX;
    mean_var = 0;
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            int varx;
            int vary;
            varx = od_compute_var_4x4(x + 2 * i * stride + 2 * j, stride);
            vary = od_compute_var_4x4(y + 2 * i * stride + 2 * j, stride);
            min_var = OD_MINI(min_var, varx);
            mean_var += 1. / (1 + varx);
            /* The cast to (double) is to avoid an overflow before the sqrt.*/
            vardist += varx - 2 * sqrt(varx * (double)vary) + vary;
        }
    }
    /* We use a different variance statistic depending on whether activity
       masking is used, since the harmonic mean appeared slightly worse with
       masking off. The calibration constant just ensures that we preserve the
       rate compared to activity=1. */
    if (use_activity_masking) {
        calibration = 1.95;
        var_stat = 9. / mean_var;
    } else {
        calibration = 1.62;
        var_stat = min_var;
    }
    /* 1.62 is a calibration constant, 0.25 is a noise floor and 1/6 is the
       activity masking constant. */
    activity = calibration * pow(.25 + var_stat, -1. / 6);

    sum = 0;
    for (i = 0; i < 8; i++) {
        for (j = 0; j < 8; j++)
            sum += e_lp[i * stride + j] * (double)e_lp[i * stride + j];
    }
    /* Normalize the filter to unit DC response. */
    sum *= 1. / (OD_DIST_LP_NORM * OD_DIST_LP_NORM * OD_DIST_LP_NORM *
                 OD_DIST_LP_NORM);
    return activity * activity * (sum + vardist);
}

double svt_aom_od_compute_dist(uint16_t *x, uint16_t *y, int bsize_w,
                               int bsize_h, int qindex, int activity_masking) {
    assert(bsize_w >= 8 && bsize_h >= 8);

    int i, j;
    DECLARE_ALIGNED(16, od_coeff, e[MAX_TX_SQUARE]);
    DECLARE_ALIGNED(16, od_coeff, tmp[MAX_TX_SQUARE]);
    DECLARE_ALIGNED(16, od_coeff, e_lp[MAX_TX_SQUARE]);
    svt_memset(e, 0, sizeof(e));
    svt_memset(tmp, 0, sizeof(tmp));
    svt_memset(e_lp, 0, sizeof(e_lp));
    for (i = 0; i < bsize_h; i++) {
        for (j = 0; j < bsize_w; j++) {
            e[i * bsize_w + j] = x[i * bsize_w + j] - y[i * bsize_w + j];
        }
    }
    int mid = OD_DIST_LP_MID;
    for (i = 0; i < bsize_h; i++) {
        tmp[i * bsize_w] = mid * e[i * bsize_w] + 2 * e[i * bsize_w + 1];
        tmp[i * bsize_w + bsize_w - 1] =
            mid * e[i * bsize_w + bsize_w - 1] + 2 * e[i * bsize_w + bsize_w - 2];
        for (j = 1; j < bsize_w - 1; j++) {
            tmp[i * bsize_w + j] = mid * e[i * bsize_w + j] + e[i * bsize_w + j - 1] +
                                   e[i * bsize_w + j + 1];
        }
    }

    double sum = 0;

    for (j = 0; j < bsize_w; j++) {
        e_lp[j] = mid * tmp[j] + 2 * tmp[bsize_w + j];
        e_lp[(bsize_h - 1) * bsize_w + j] = mid * tmp[(bsize_h - 1) * bsize_w + j] +
                                            2 * tmp[(bsize_h - 2) * bsize_w + j];
    }
    for (i = 1; i < bsize_h - 1; i++) {
        for (j = 0; j < bsize_w; j++) {
            e_lp[i * bsize_w + j] = mid * tmp[i * bsize_w + j] +
                                    tmp[(i - 1) * bsize_w + j] +
                                    tmp[(i + 1) * bsize_w + j];
        }
    }
    for (i = 0; i < bsize_h; i += 8) {
        for (j = 0; j < bsize_w; j += 8) {
            sum += od_compute_dist_8x8(activity_masking, &x[i * bsize_w + j],
                                       &y[i * bsize_w + j], &e_lp[i * bsize_w + j],
                                       bsize_w);
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
