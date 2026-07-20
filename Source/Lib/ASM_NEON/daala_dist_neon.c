/*
* Copyright (c) 2026, Alliance for Open Media. All rights reserved
*
* This source code is subject to the terms of the BSD 2 Clause License and
* the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
* was not distributed with this source code in the LICENSE file, you can
* obtain it at www.aomedia.org/license/software. If the Alliance for Open
* Media Patent License 1.0 was not distributed with this source code in the
* PATENTS file, you can obtain it at www.aomedia.org/license/patent.
*/

#include "daala_dist.h"
#include "common_dsp_rtcd.h"

void svt_aom_od_compute_diff_and_filter_h_neon(
    const uint16_t *x, uint32_t x_stride,
    const uint16_t *y, uint32_t y_stride,
    int32_t *tmp, uint32_t tmp_stride,
    uint32_t width, uint32_t height)
{
    svt_aom_od_compute_diff_and_filter_h_c(x, x_stride, y, y_stride, tmp, tmp_stride, width, height);
}

void svt_aom_od_filter_v_neon(
    const int32_t *tmp, uint32_t tmp_stride,
    int32_t *e_lp, uint32_t e_lp_stride,
    uint32_t width, uint32_t height)
{
    svt_aom_od_filter_v_c(tmp, tmp_stride, e_lp, e_lp_stride, width, height);
}

void svt_aom_od_compute_var_and_dist_8x8_neon(
    const uint16_t *x, uint32_t x_stride,
    const uint16_t *y, uint32_t y_stride,
    const int32_t *e_lp, uint32_t e_lp_stride,
    uint32_t *varx, uint32_t *vary,
    uint64_t *sum_e_lp_sq)
{
    svt_aom_od_compute_var_and_dist_8x8_c(x, x_stride, y, y_stride, e_lp, e_lp_stride, varx, vary, sum_e_lp_sq);
}
