/*
 * Copyright (c) 2026, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
 */

#include <arm_sve.h>

#include "definitions.h"

void svt_vmaf_apply_unsharp_row_sve2(const uint8_t* src, const uint8_t* blur, uint8_t* dst, int width, int amount,
                                     int32_t max_delta) {
    const int16_t amount_s16    = (int16_t)(amount > INT16_MAX ? INT16_MAX : amount);
    const int16_t max_delta_s16 = (int16_t)(max_delta > INT16_MAX ? INT16_MAX : max_delta);

    const svint16_t clamp_max = svdup_n_s16(max_delta_s16);
    const svint16_t clamp_min = svdup_n_s16(-max_delta_s16);

    const svbool_t ptrue = svptrue_b16();
    const int      step  = svcnth();

    int j = 0;
    for (; j + step <= width; j += step) {
        const svint16_t b = svld1ub_s16(ptrue, blur + j);
        const svint16_t s = svld1ub_s16(ptrue, src + j);

        svint16_t detail = svsub_s16_x(ptrue, s, b);
        detail           = svmin_s16_x(ptrue, detail, clamp_max);
        detail           = svmax_s16_x(ptrue, detail, clamp_min);

        svint16_t res = svqdmulh_n_s16(detail, amount_s16);
        res           = svadd_s16_x(ptrue, res, s);

        svst1b_u16(ptrue, dst + j, svreinterpret_u16_u8(svqxtunb_s16(res)));
    }
    if (j != width) {
        const svbool_t pg_s16 = svwhilelt_b16_s32(j, width);

        const svint16_t b = svld1ub_s16(pg_s16, blur + j);
        const svint16_t s = svld1ub_s16(pg_s16, src + j);

        svint16_t detail = svsub_s16_x(pg_s16, s, b);
        detail           = svmin_s16_x(pg_s16, detail, clamp_max);
        detail           = svmax_s16_x(pg_s16, detail, clamp_min);

        svint16_t res = svqdmulh_n_s16(detail, amount_s16);
        res           = svadd_s16_x(pg_s16, res, s);

        svst1b_u16(pg_s16, dst + j, svreinterpret_u16_u8(svqxtunb_s16(res)));
    }
}
