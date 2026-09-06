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

#include <immintrin.h>
#include <limits.h>
#include "daala_dist.h"
#include "common_dsp_rtcd.h"

#define OD_DIST_LP_MID (5)

void svt_aom_od_compute_diff_and_filter_h_avx2(
    const uint16_t *x, uint32_t x_stride,
    const uint16_t *y, uint32_t y_stride,
    int32_t *tmp, uint32_t tmp_stride,
    uint32_t width, uint32_t height)
{
    for (uint32_t i = 0; i < height; i++) {
        const uint16_t *row_x = x + i * x_stride;
        const uint16_t *row_y = y + i * y_stride;
        int32_t *row_tmp = tmp + i * tmp_stride;

        // Padded buffer to simplify shifts and boundary handling
        DECLARE_ALIGNED(16, int16_t, e[258]);

        for (uint32_t j = 0; j < width; j++) {
            e[j + 1] = (int16_t)row_x[j] - (int16_t)row_y[j];
        }

        // Apply filter to middle elements using AVX2.
        // We can process 16 elements at a time.
        uint32_t j = 0;
        for (; j + 15 < width; j += 16) {
            __m256i prev = _mm256_loadu_si256((const __m256i*)&e[j]);
            __m256i curr = _mm256_loadu_si256((const __m256i*)&e[j + 1]);
            __m256i next = _mm256_loadu_si256((const __m256i*)&e[j + 2]);

            __m256i curr_5 = _mm256_add_epi16(_mm256_slli_epi16(curr, 2), curr);
            __m256i filtered_256 = _mm256_add_epi16(_mm256_add_epi16(curr_5, prev), next);

            __m256i filtered_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(filtered_256));
            __m256i filtered_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(filtered_256, 1));

            _mm256_storeu_si256((__m256i*)&row_tmp[j], filtered_lo);
            _mm256_storeu_si256((__m256i*)&row_tmp[j + 8], filtered_hi);
        }

        // Fallback to SSE/scalar if width is not a multiple of 16
        for (; j < width; j += 8) {
            __m128i prev = _mm_loadu_si128((const __m128i*)&e[j]);
            __m128i curr = _mm_loadu_si128((const __m128i*)&e[j + 1]);
            __m128i next = _mm_loadu_si128((const __m128i*)&e[j + 2]);

            __m128i curr_5 = _mm_add_epi16(_mm_slli_epi16(curr, 2), curr);
            __m128i filtered_16 = _mm_add_epi16(_mm_add_epi16(curr_5, prev), next);

            __m128i filtered_lo = _mm_cvtepi16_epi32(filtered_16);
            __m128i filtered_hi = _mm_cvtepi16_epi32(_mm_srli_si128(filtered_16, 8));

            _mm_storeu_si128((__m128i*)&row_tmp[j], filtered_lo);
            _mm_storeu_si128((__m128i*)&row_tmp[j + 4], filtered_hi);
        }

        // Correct boundary elements
        int32_t e0 = e[1];
        int32_t e1 = e[2];
        row_tmp[0] = OD_DIST_LP_MID * e0 + 2 * e1;

        int32_t e_last = e[width];
        int32_t e_prev_last = e[width - 1];
        row_tmp[width - 1] = OD_DIST_LP_MID * e_last + 2 * e_prev_last;
    }
}

void svt_aom_od_filter_v_avx2(
    const int32_t *tmp, uint32_t tmp_stride,
    int32_t *e_lp, uint32_t e_lp_stride,
    uint32_t width, uint32_t height)
{
    // Top boundary: 5 * tmp[j] + 2 * tmp[w + j]
    uint32_t j = 0;
    for (; j + 7 < width; j += 8) {
        __m256i t0 = _mm256_loadu_si256((const __m256i*)&tmp[j]);
        __m256i t1 = _mm256_loadu_si256((const __m256i*)&tmp[tmp_stride + j]);

        __m256i t0_5 = _mm256_add_epi32(_mm256_slli_epi32(t0, 2), t0);
        __m256i t1_2 = _mm256_slli_epi32(t1, 1);
        __m256i res = _mm256_add_epi32(t0_5, t1_2);
        _mm256_storeu_si256((__m256i*)&e_lp[j], res);
    }
    for (; j < width; j += 4) {
        __m128i t0 = _mm_loadu_si128((const __m128i*)&tmp[j]);
        __m128i t1 = _mm_loadu_si128((const __m128i*)&tmp[tmp_stride + j]);

        __m128i t0_5 = _mm_add_epi32(_mm_slli_epi32(t0, 2), t0);
        __m128i t1_2 = _mm_slli_epi32(t1, 1);
        __m128i res = _mm_add_epi32(t0_5, t1_2);
        _mm_storeu_si128((__m128i*)&e_lp[j], res);
    }

    // Middle rows: 5 * tmp[i] + tmp[i-1] + tmp[i+1]
    for (uint32_t i = 1; i < height - 1; i++) {
        const int32_t *row_prev = tmp + (i - 1) * tmp_stride;
        const int32_t *row_curr = tmp + i * tmp_stride;
        const int32_t *row_next = tmp + (i + 1) * tmp_stride;
        int32_t *row_lp = e_lp + i * e_lp_stride;

        uint32_t col = 0;
        for (; col + 7 < width; col += 8) {
            __m256i prev = _mm256_loadu_si256((const __m256i*)&row_prev[col]);
            __m256i curr = _mm256_loadu_si256((const __m256i*)&row_curr[col]);
            __m256i next = _mm256_loadu_si256((const __m256i*)&row_next[col]);

            __m256i curr_5 = _mm256_add_epi32(_mm256_slli_epi32(curr, 2), curr);
            __m256i res = _mm256_add_epi32(_mm256_add_epi32(curr_5, prev), next);
            _mm256_storeu_si256((__m256i*)&row_lp[col], res);
        }
        for (; col < width; col += 4) {
            __m128i prev = _mm_loadu_si128((const __m128i*)&row_prev[col]);
            __m128i curr = _mm_loadu_si128((const __m128i*)&row_curr[col]);
            __m128i next = _mm_loadu_si128((const __m128i*)&row_next[col]);

            __m128i curr_5 = _mm_add_epi32(_mm_slli_epi32(curr, 2), curr);
            __m128i res = _mm_add_epi32(_mm_add_epi32(curr_5, prev), next);
            _mm_storeu_si128((__m128i*)&row_lp[col], res);
        }
    }

    // Bottom boundary: 5 * tmp[h-1] + 2 * tmp[h-2]
    const int32_t *row_last = tmp + (height - 1) * tmp_stride;
    const int32_t *row_prev_last = tmp + (height - 2) * tmp_stride;
    int32_t *row_lp_last = e_lp + (height - 1) * e_lp_stride;

    j = 0;
    for (; j + 7 < width; j += 8) {
        __m256i t_last = _mm256_loadu_si256((const __m256i*)&row_last[j]);
        __m256i t_prev_last = _mm256_loadu_si256((const __m256i*)&row_prev_last[j]);

        __m256i t_last_5 = _mm256_add_epi32(_mm256_slli_epi32(t_last, 2), t_last);
        __m256i t_prev_last_2 = _mm256_slli_epi32(t_prev_last, 1);
        __m256i res = _mm256_add_epi32(t_last_5, t_prev_last_2);
        _mm256_storeu_si256((__m256i*)&row_lp_last[j], res);
    }
    for (; j < width; j += 4) {
        __m128i t_last = _mm_loadu_si128((const __m128i*)&row_last[j]);
        __m128i t_prev_last = _mm_loadu_si128((const __m128i*)&row_prev_last[j]);

        __m128i t_last_5 = _mm_add_epi32(_mm_slli_epi32(t_last, 2), t_last);
        __m128i t_prev_last_2 = _mm_slli_epi32(t_prev_last, 1);
        __m128i res = _mm_add_epi32(t_last_5, t_prev_last_2);
        _mm_storeu_si128((__m128i*)&row_lp_last[j], res);
    }
}

static inline int od_compute_var_4x4_avx2(const uint16_t *x, uint32_t stride) {
    __m128i r0 = _mm_loadl_epi64((const __m128i*)(x + 0 * stride));
    __m128i r1 = _mm_loadl_epi64((const __m128i*)(x + 1 * stride));
    __m128i r2 = _mm_loadl_epi64((const __m128i*)(x + 2 * stride));
    __m128i r3 = _mm_loadl_epi64((const __m128i*)(x + 3 * stride));

    __m128i r01 = _mm_unpacklo_epi64(r0, r1);
    __m128i r23 = _mm_unpacklo_epi64(r2, r3);

    __m128i ones = _mm_set1_epi16(1);
    __m128i sum_01 = _mm_madd_epi16(r01, ones);
    __m128i sum_23 = _mm_madd_epi16(r23, ones);
    __m128i sum_all = _mm_add_epi32(sum_01, sum_23);

    __m128i shuf = _mm_shuffle_epi32(sum_all, _MM_SHUFFLE(1, 0, 3, 2));
    __m128i sum_2 = _mm_add_epi32(sum_all, shuf);
    __m128i shuf2 = _mm_shuffle_epi32(sum_2, _MM_SHUFFLE(2, 3, 0, 1));
    __m128i sum_final = _mm_add_epi32(sum_2, shuf2);
    int sum = _mm_cvtsi128_si32(sum_final);

    __m128i sq_01 = _mm_madd_epi16(r01, r01);
    __m128i sq_23 = _mm_madd_epi16(r23, r23);
    __m128i sq_all = _mm_add_epi32(sq_01, sq_23);

    __m128i sq_shuf = _mm_shuffle_epi32(sq_all, _MM_SHUFFLE(1, 0, 3, 2));
    __m128i sq_2 = _mm_add_epi32(sq_all, sq_shuf);
    __m128i sq_shuf2 = _mm_shuffle_epi32(sq_2, _MM_SHUFFLE(2, 3, 0, 1));
    __m128i sq_final = _mm_add_epi32(sq_2, sq_shuf2);
    int s2 = _mm_cvtsi128_si32(sq_final);

    return (s2 - (sum * sum >> 4)) >> 4;
}

void svt_aom_od_compute_var_and_dist_8x8_avx2(
    const uint16_t *x, uint32_t x_stride,
    const uint16_t *y, uint32_t y_stride,
    const int32_t *e_lp, uint32_t e_lp_stride,
    uint32_t *varx, uint32_t *vary,
    uint64_t *sum_e_lp_sq)
{
    // Compute 9 variances
    int idx = 0;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            varx[idx] = od_compute_var_4x4_avx2(x + 2 * i * x_stride + 2 * j, x_stride);
            vary[idx] = od_compute_var_4x4_avx2(y + 2 * i * y_stride + 2 * j, y_stride);
            idx++;
        }
    }

    // Compute sum of squares of e_lp (8x8 block of int32_t) using AVX2 256-bit vectors
    __m256i accum = _mm256_setzero_si256();

    for (int i = 0; i < 8; i++) {
        const int32_t *row = e_lp + i * e_lp_stride;
        __m256i val = _mm256_loadu_si256((const __m256i*)row);

        __m256i sq = _mm256_mullo_epi32(val, val);

        __m256i sq_lo = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(sq));
        __m256i sq_hi = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(sq, 1));

        accum = _mm256_add_epi64(accum, sq_lo);
        accum = _mm256_add_epi64(accum, sq_hi);
    }

    // Extract 256-bit horizontal sum
    DECLARE_ALIGNED(32, uint64_t, sum_val[4]);
    _mm256_storeu_si256((__m256i*)sum_val, accum);
    *sum_e_lp_sq = sum_val[0] + sum_val[1] + sum_val[2] + sum_val[3];
}
