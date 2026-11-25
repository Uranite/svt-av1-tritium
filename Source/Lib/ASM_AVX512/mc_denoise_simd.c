/*
 * Copyright(c) 2026 Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at https://www.aomedia.org/license/software-license. If the
 * Alliance for Open Media Patent License 1.0 was not distributed with this
 * source code in the PATENTS file, you can obtain it at
 * https://www.aomedia.org/license/patent-license.
 */
/*
 * mc_denoise_simd.c - AVX-512 kernels for the motion-compensated denoiser.
 *
 * This translation unit is compiled with -mavx512f,bw,dq,vl / /arch:AVX512 by the build system (see
 * Source/Lib/ASM_AVX512/CMakeLists.txt), so the instructions are available
 * unconditionally here. The scalar implementations in mc_denoise_*.c stay the
 * default compile path and the correctness oracle; dispatching happens in
 * common_dsp_rtcd.c.
 *
 * Exactness contract:
 *   - Integer kernels (SAD, SATD, half-pel filters, degrain helpers) are
 *     BIT-IDENTICAL to the scalar versions: lane-wise integer operations
 *     only, accumulation order changes are associative-safe.
 *   - Float kernel (dfttest) is BIT-IDENTICAL to the scalar version: pure
 *     lane-wise IEEE-754 operations (mul/add/sub/div/compare/sqrt-free),
 *     NO fused multiply-adds and NO reassociation, so every element takes
 *     exactly the scalar operation sequence. (The library is compiled for
 *     generic x86-64 without FMA, so the scalar reference has no contractions
 *     to match — verified by disassembly.)
 *
 * Testing hook: environment variable MCD_NO_SIMD (any value) forces every
 * dispatch to the scalar path (see svt_mcd_simd_disabled).
 */

/* AVX-512 kernels: compiled with -mavx512f/... / /arch:AVX512 by the build system. */
#include <immintrin.h>
// for the CRT secure-API macros on Windows
#if defined(_WIN32) && !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#endif
#include <stdlib.h>

#include "mc_denoise.h"
#include "mc_denoise_simd.h"
#include "mc_denoise_dfttest_ctx.h"
#include "svt_threads.h"
#define svt_mcd_simd_satd_u8u8 mcd_avx512_satd_u8u8
#define svt_mcd_simd_satd_u16u16 mcd_avx512_satd_u16u16
#define svt_mcd_simd_halfpel_h_u16 mcd_avx512_halfpel_h_u16
#define svt_mcd_simd_halfpel_v_u16 mcd_avx512_halfpel_v_u16
#define svt_mcd_simd_dfttest_plane_with_ctx mcd_avx512_dfttest_plane_with_ctx
#define mcd_merge_plane512 mcd_avx512_merge_plane512

#if EN_AVX512_SUPPORT

uint64_t svt_mcd_simd_satd_u8u8(const uint8_t* a, int32_t sa, const uint8_t* b, int32_t sb, uint32_t w, uint32_t h,
                                uint64_t bound) {
    if (w < 16 || h < 16) {
        return 0; /* caller falls back to scalar (unreachable at level 0) */
    }
    /* permutex indices: lane s <- element 4*(s mod 4) + c of the widened row */
    const __m256i vidx[4] = {
        _mm256_setr_epi16(0, 4, 8, 12, 0, 4, 8, 12, 0, 4, 8, 12, 0, 4, 8, 12),
        _mm256_setr_epi16(1, 5, 9, 13, 1, 5, 9, 13, 1, 5, 9, 13, 1, 5, 9, 13),
        _mm256_setr_epi16(2, 6, 10, 14, 2, 6, 10, 14, 2, 6, 10, 14, 2, 6, 10, 14),
        _mm256_setr_epi16(3, 7, 11, 15, 3, 7, 11, 15, 3, 7, 11, 15, 3, 7, 11, 15),
    };
    uint64_t total = 0;
    for (uint32_t by = 0; by + 16 <= h; by += 16) {
        for (uint32_t bx = 0; bx + 16 <= w; bx += 16) {
            __m256i d[4][4];
            for (uint32_t r = 0; r < 4; r++) {
                const __m256i df[4] = {
                    _mm256_sub_epi16(_mm256_cvtepu8_epi16(_mm_loadu_si128(
                                         (const __m128i*)(const void*)(a + (by + 4 * 0 + r) * sa + bx))),
                                     _mm256_cvtepu8_epi16(_mm_loadu_si128(
                                         (const __m128i*)(const void*)(b + (by + 4 * 0 + r) * sb + bx)))),
                    _mm256_sub_epi16(_mm256_cvtepu8_epi16(_mm_loadu_si128(
                                         (const __m128i*)(const void*)(a + (by + 4 * 1 + r) * sa + bx))),
                                     _mm256_cvtepu8_epi16(_mm_loadu_si128(
                                         (const __m128i*)(const void*)(b + (by + 4 * 1 + r) * sb + bx)))),
                    _mm256_sub_epi16(_mm256_cvtepu8_epi16(_mm_loadu_si128(
                                         (const __m128i*)(const void*)(a + (by + 4 * 2 + r) * sa + bx))),
                                     _mm256_cvtepu8_epi16(_mm_loadu_si128(
                                         (const __m128i*)(const void*)(b + (by + 4 * 2 + r) * sb + bx)))),
                    _mm256_sub_epi16(_mm256_cvtepu8_epi16(_mm_loadu_si128(
                                         (const __m128i*)(const void*)(a + (by + 4 * 3 + r) * sa + bx))),
                                     _mm256_cvtepu8_epi16(_mm_loadu_si128(
                                         (const __m128i*)(const void*)(b + (by + 4 * 3 + r) * sb + bx)))),
                };
                for (uint32_t c = 0; c < 4; c++) {
                    /* permutex output lanes 4g..4g+3 repeat group 0's values;
                     * take the low 64 bits of each group's permute */
                    const __m256i v = _mm256_setr_epi64x(
                        (long long)_mm_cvtsi128_si64(_mm256_castsi256_si128(_mm256_permutexvar_epi16(vidx[c], df[0]))),
                        (long long)_mm_cvtsi128_si64(_mm256_castsi256_si128(_mm256_permutexvar_epi16(vidx[c], df[1]))),
                        (long long)_mm_cvtsi128_si64(_mm256_castsi256_si128(_mm256_permutexvar_epi16(vidx[c], df[2]))),
                        (long long)_mm_cvtsi128_si64(_mm256_castsi256_si128(_mm256_permutexvar_epi16(vidx[c], df[3]))));
                    d[r][c] = v;
                }
            }
            /* horizontal pass (columns of the 4x4) */
            __m256i t[4][4];
            for (uint32_t r = 0; r < 4; r++) {
                const __m256i a0 = _mm256_add_epi16(d[r][0], d[r][3]);
                const __m256i a1 = _mm256_add_epi16(d[r][1], d[r][2]);
                const __m256i a2 = _mm256_sub_epi16(d[r][1], d[r][2]);
                const __m256i a3 = _mm256_sub_epi16(d[r][0], d[r][3]);
                t[r][0]          = _mm256_add_epi16(a0, a1);
                t[r][1]          = _mm256_add_epi16(a2, a3);
                t[r][2]          = _mm256_sub_epi16(a0, a1);
                t[r][3]          = _mm256_sub_epi16(a3, a2);
            }
            /* vertical pass + accumulate |s| per lane (u16 lanes hold at
             * most 16 sub-blocks x 4080 = 65280, no wrap) */
            __m256i acc = _mm256_setzero_si256();
            for (uint32_t c = 0; c < 4; c++) {
                const __m256i a0 = _mm256_add_epi16(t[0][c], t[3][c]);
                const __m256i a1 = _mm256_add_epi16(t[1][c], t[2][c]);
                const __m256i a2 = _mm256_sub_epi16(t[1][c], t[2][c]);
                const __m256i a3 = _mm256_sub_epi16(t[0][c], t[3][c]);
                const __m256i s0 = _mm256_add_epi16(a0, a1);
                const __m256i s1 = _mm256_add_epi16(a2, a3);
                const __m256i s2 = _mm256_sub_epi16(a0, a1);
                const __m256i s3 = _mm256_sub_epi16(a3, a2);
                acc              = _mm256_add_epi16(acc, _mm256_abs_epi16(s0));
                acc              = _mm256_add_epi16(acc, _mm256_abs_epi16(s1));
                acc              = _mm256_add_epi16(acc, _mm256_abs_epi16(s2));
                acc              = _mm256_add_epi16(acc, _mm256_abs_epi16(s3));
            }
            const __m512i wide = _mm512_cvtepu16_epi32(acc);
            total += _mm512_reduce_add_epi32(wide);
            if (bound && total >= bound) {
                return total; /* partial-region abort: see mc_denoise_me.c */
            }
        }
    }
    return total;
}

uint64_t svt_mcd_simd_satd_u16u16(const uint16_t* a, int32_t sa, const uint16_t* b, int32_t sb, uint32_t w, uint32_t h,
                                  uint64_t bound) {
    if (w < 16 || h < 16) {
        return 0;
    }
    const __m512i vidx[4] = {
        _mm512_setr_epi32(0, 4, 8, 12, 0, 4, 8, 12, 0, 4, 8, 12, 0, 4, 8, 12),
        _mm512_setr_epi32(1, 5, 9, 13, 1, 5, 9, 13, 1, 5, 9, 13, 1, 5, 9, 13),
        _mm512_setr_epi32(2, 6, 10, 14, 2, 6, 10, 14, 2, 6, 10, 14, 2, 6, 10, 14),
        _mm512_setr_epi32(3, 7, 11, 15, 3, 7, 11, 15, 3, 7, 11, 15, 3, 7, 11, 15),
    };
    uint64_t total = 0;
    for (uint32_t by = 0; by + 16 <= h; by += 16) {
        for (uint32_t bx = 0; bx + 16 <= w; bx += 16) {
            __m512i d[4][4];
            for (uint32_t r = 0; r < 4; r++) {
                const __m512i df[4] = {
                    _mm512_sub_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256(
                                         (const __m256i*)(const void*)(a + (by + 4 * 0 + r) * sa + bx))),
                                     _mm512_cvtepu16_epi32(_mm256_loadu_si256(
                                         (const __m256i*)(const void*)(b + (by + 4 * 0 + r) * sb + bx)))),
                    _mm512_sub_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256(
                                         (const __m256i*)(const void*)(a + (by + 4 * 1 + r) * sa + bx))),
                                     _mm512_cvtepu16_epi32(_mm256_loadu_si256(
                                         (const __m256i*)(const void*)(b + (by + 4 * 1 + r) * sb + bx)))),
                    _mm512_sub_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256(
                                         (const __m256i*)(const void*)(a + (by + 4 * 2 + r) * sa + bx))),
                                     _mm512_cvtepu16_epi32(_mm256_loadu_si256(
                                         (const __m256i*)(const void*)(b + (by + 4 * 2 + r) * sb + bx)))),
                    _mm512_sub_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256(
                                         (const __m256i*)(const void*)(a + (by + 4 * 3 + r) * sa + bx))),
                                     _mm512_cvtepu16_epi32(_mm256_loadu_si256(
                                         (const __m256i*)(const void*)(b + (by + 4 * 3 + r) * sb + bx)))),
                };
                for (uint32_t c = 0; c < 4; c++) {
                    /* low 128 bits of each group's permute hold the 4
                     * sub-block values of that group */
                    const __m128i q0 = _mm512_castsi512_si128(_mm512_permutexvar_epi32(vidx[c], df[0]));
                    const __m128i q1 = _mm512_castsi512_si128(_mm512_permutexvar_epi32(vidx[c], df[1]));
                    const __m128i q2 = _mm512_castsi512_si128(_mm512_permutexvar_epi32(vidx[c], df[2]));
                    const __m128i q3 = _mm512_castsi512_si128(_mm512_permutexvar_epi32(vidx[c], df[3]));
                    const __m512i v  = _mm512_inserti64x4(
                        _mm512_castsi256_si512(_mm256_set_m128i(q1, q0)), _mm256_set_m128i(q3, q2), 1);
                    d[r][c] = v;
                }
            }
            __m512i t[4][4];
            for (uint32_t r = 0; r < 4; r++) {
                const __m512i a0 = _mm512_add_epi32(d[r][0], d[r][3]);
                const __m512i a1 = _mm512_add_epi32(d[r][1], d[r][2]);
                const __m512i a2 = _mm512_sub_epi32(d[r][1], d[r][2]);
                const __m512i a3 = _mm512_sub_epi32(d[r][0], d[r][3]);
                t[r][0]          = _mm512_add_epi32(a0, a1);
                t[r][1]          = _mm512_add_epi32(a2, a3);
                t[r][2]          = _mm512_sub_epi32(a0, a1);
                t[r][3]          = _mm512_sub_epi32(a3, a2);
            }
            __m512i acc = _mm512_setzero_si512();
            for (uint32_t c = 0; c < 4; c++) {
                const __m512i a0 = _mm512_add_epi32(t[0][c], t[3][c]);
                const __m512i a1 = _mm512_add_epi32(t[1][c], t[2][c]);
                const __m512i a2 = _mm512_sub_epi32(t[1][c], t[2][c]);
                const __m512i a3 = _mm512_sub_epi32(t[0][c], t[3][c]);
                const __m512i s0 = _mm512_add_epi32(a0, a1);
                const __m512i s1 = _mm512_add_epi32(a2, a3);
                const __m512i s2 = _mm512_sub_epi32(a0, a1);
                const __m512i s3 = _mm512_sub_epi32(a3, a2);
                acc              = _mm512_add_epi32(acc, _mm512_abs_epi32(s0));
                acc              = _mm512_add_epi32(acc, _mm512_abs_epi32(s1));
                acc              = _mm512_add_epi32(acc, _mm512_abs_epi32(s2));
                acc              = _mm512_add_epi32(acc, _mm512_abs_epi32(s3));
            }
            total += _mm512_reduce_add_epi32(acc);
            if (bound && total >= bound) {
                return total; /* partial-region abort: see mc_denoise_me.c */
            }
        }
    }
    return total;
}

#endif /* EN_AVX512_SUPPORT */

#if EN_AVX512_SUPPORT
/* Symmetric mirror reflection of i into [0, w) (same as the scalar
 * mcd_mirror in mc_denoise_dfttest.c). */
static int32_t mcd_simd_mirror(int32_t i, uint32_t w) {
    if (w == 1) {
        return 0;
    }
    while (i < 0 || i >= (int32_t)w) {
        if (i < 0) {
            i = -i - 1;
        } else {
            i = 2 * (int32_t)w - 1 - i;
        }
    }
    return i;
}

#endif /* EN_AVX512_SUPPORT */

#if EN_AVX512_SUPPORT
static inline __m512i mcd_halfpel_load16(const uint16_t* p, uint32_t X) {
    return _mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*)(const void*)(p + X)));
}

void svt_mcd_simd_halfpel_h_u16(const uint16_t* src, uint32_t sstride, uint32_t w, uint32_t h, uint32_t pad,
                                uint16_t* dst, uint32_t dstride, uint32_t maxval) {
    const uint32_t tw   = w + 2 * pad;
    const int32_t  wm1  = (int32_t)w - 1;
    const __m512i  zero = _mm512_setzero_si512();
    const __m512i  vmax = _mm512_set1_epi32((int)maxval);
    for (uint32_t Y = 0; Y < h + 2 * pad; Y++) {
        /* see the 8-bit H pass for the row-base / tap-offset scheme */
        const uint16_t* srow = src + (size_t)Y * sstride;
        const uint16_t* vrow = srow + pad; /* visible origin (scalar path) */
        uint16_t*       drow = dst + (size_t)Y * dstride;
        uint32_t        X    = 0;
        for (; X < pad + 2 && X < tw; X++) { /* taps clamp into the core */
            const int32_t x = (int32_t)X - (int32_t)pad;
            const int32_t v = (vrow[mcd_simd_clampi(x - 2, 0, wm1)] - 5 * vrow[mcd_simd_clampi(x - 1, 0, wm1)] +
                               20 * vrow[mcd_simd_clampi(x, 0, wm1)] + 20 * vrow[mcd_simd_clampi(x + 1, 0, wm1)] -
                               5 * vrow[mcd_simd_clampi(x + 2, 0, wm1)] + vrow[mcd_simd_clampi(x + 3, 0, wm1)] + 16) >>
                5;
            drow[X] = (uint16_t)(v < 0 ? 0 : (v > (int32_t)maxval ? (int32_t)maxval : v));
        }
        for (X = pad + 2; X + 19 <= tw; X += 16) {
            const __m512i s0 = mcd_halfpel_load16(srow, X - 2);
            const __m512i s1 = mcd_halfpel_load16(srow, X - 1);
            const __m512i s2 = mcd_halfpel_load16(srow, X);
            const __m512i s3 = mcd_halfpel_load16(srow, X + 1);
            const __m512i s4 = mcd_halfpel_load16(srow, X + 2);
            const __m512i s5 = mcd_halfpel_load16(srow, X + 3);
            const __m512i t  = _mm512_add_epi32(
                _mm512_add_epi32(_mm512_add_epi32(s0, s5),
                                 _mm512_mullo_epi32(_mm512_add_epi32(s2, s3), _mm512_set1_epi32(20))),
                _mm512_mullo_epi32(_mm512_add_epi32(s1, s4), _mm512_set1_epi32(-5)));
            const __m512i v = _mm512_srai_epi32(_mm512_add_epi32(t, _mm512_set1_epi32(16)), 5);
            _mm256_storeu_si256((__m256i*)(void*)(drow + X),
                                /* signed max handles negative Wiener
                                 * overshoot; value is then <= 0x7FFF so the
                                 * unsigned min against vmax stays correct */
                                _mm512_cvtusepi32_epi16(_mm512_min_epu32(_mm512_max_epi32(v, zero), vmax)));
        }
        for (; X < tw; X++) {
            const int32_t x = (int32_t)X - (int32_t)pad;
            const int32_t v = (vrow[mcd_simd_clampi(x - 2, 0, wm1)] - 5 * vrow[mcd_simd_clampi(x - 1, 0, wm1)] +
                               20 * vrow[mcd_simd_clampi(x, 0, wm1)] + 20 * vrow[mcd_simd_clampi(x + 1, 0, wm1)] -
                               5 * vrow[mcd_simd_clampi(x + 2, 0, wm1)] + vrow[mcd_simd_clampi(x + 3, 0, wm1)] + 16) >>
                5;
            drow[X] = (uint16_t)(v < 0 ? 0 : (v > (int32_t)maxval ? (int32_t)maxval : v));
        }
    }
}

void svt_mcd_simd_halfpel_v_u16(const uint16_t* src, uint32_t sstride, uint32_t w, uint32_t h, uint32_t pad,
                                uint16_t* dst, uint32_t dstride, uint32_t maxval) {
    const uint32_t tw   = w + 2 * pad;
    const int32_t  hm1  = (int32_t)h - 1;
    const __m512i  zero = _mm512_setzero_si512();
    const __m512i  vmax = _mm512_set1_epi32((int)maxval);
    for (uint32_t Y = 0; Y < h + 2 * pad; Y++) {
        const int32_t   y    = (int32_t)Y - (int32_t)pad;
        const uint16_t* r0   = src + (size_t)(mcd_simd_clampi(y - 2, 0, hm1) + (int32_t)pad) * sstride;
        const uint16_t* r1   = src + (size_t)(mcd_simd_clampi(y - 1, 0, hm1) + (int32_t)pad) * sstride;
        const uint16_t* r2   = src + (size_t)(mcd_simd_clampi(y, 0, hm1) + (int32_t)pad) * sstride;
        const uint16_t* r3   = src + (size_t)(mcd_simd_clampi(y + 1, 0, hm1) + (int32_t)pad) * sstride;
        const uint16_t* r4   = src + (size_t)(mcd_simd_clampi(y + 2, 0, hm1) + (int32_t)pad) * sstride;
        const uint16_t* r5   = src + (size_t)(mcd_simd_clampi(y + 3, 0, hm1) + (int32_t)pad) * sstride;
        uint16_t*       drow = dst + (size_t)Y * dstride;
        uint32_t        X    = 0;
        for (; X + 16 <= tw; X += 16) {
            const __m512i s0 = mcd_halfpel_load16(r0, X);
            const __m512i s1 = mcd_halfpel_load16(r1, X);
            const __m512i s2 = mcd_halfpel_load16(r2, X);
            const __m512i s3 = mcd_halfpel_load16(r3, X);
            const __m512i s4 = mcd_halfpel_load16(r4, X);
            const __m512i s5 = mcd_halfpel_load16(r5, X);
            const __m512i t  = _mm512_add_epi32(
                _mm512_add_epi32(_mm512_add_epi32(s0, s5),
                                 _mm512_mullo_epi32(_mm512_add_epi32(s2, s3), _mm512_set1_epi32(20))),
                _mm512_mullo_epi32(_mm512_add_epi32(s1, s4), _mm512_set1_epi32(-5)));
            const __m512i v = _mm512_srai_epi32(_mm512_add_epi32(t, _mm512_set1_epi32(16)), 5);
            _mm256_storeu_si256((__m256i*)(void*)(drow + X),
                                /* signed max handles negative Wiener
                                 * overshoot; value is then <= 0x7FFF so the
                                 * unsigned min against vmax stays correct */
                                _mm512_cvtusepi32_epi16(_mm512_min_epu32(_mm512_max_epi32(v, zero), vmax)));
        }
        for (; X < tw; X++) {
            const int32_t v = (r0[X] - 5 * r1[X] + 20 * r2[X] + 20 * r3[X] - 5 * r4[X] + r5[X] + 16) >> 5;
            drow[X]         = (uint16_t)(v < 0 ? 0 : (v > (int32_t)maxval ? (int32_t)maxval : v));
        }
    }
}

#endif /* EN_AVX512_SUPPORT */

#if EN_AVX512_SUPPORT
static inline __m512 mcd_merge_mask512(__m512 yv, __m512 kK, __m512 v16, __m512 v59, __m512 v1, __m512 v0) {
    const __m512 y8 = _mm512_mul_ps(yv, kK);
    const __m512 m  = _mm512_div_ps(_mm512_sub_ps(y8, v16), v59);
    return _mm512_min_ps(_mm512_max_ps(m, v0), v1);
}

static inline __m512i mcd_merge_out512(__m512 sv, __m512 fv, __m512 m, __m512 vhalf, __m512i vpmaxi) {
    const __m512 v  = _mm512_add_ps(sv, _mm512_mul_ps(m, _mm512_sub_ps(fv, sv)));
    __m512i      iv = _mm512_cvttps_epi32(_mm512_add_ps(v, vhalf));
    iv              = _mm512_max_epi32(iv, _mm512_setzero_si512());
    return _mm512_min_epi32(iv, vpmaxi);
}

/* Luma-mask row: dst[x] = mask of luma pixel (x, ly), for x in [0, yl->w).
 * Same per-element op sequence as the scalar mcd_merge_mask. */
static void mcd_merge_mask_row512(const McdPlane* yl, uint32_t ly, float* dst, float kK) {
    const __m512 vK  = _mm512_set1_ps(kK);
    const __m512 v16 = _mm512_set1_ps(16.0f);
    const __m512 v59 = _mm512_set1_ps(59.0f);
    const __m512 v1  = _mm512_set1_ps(1.0f);
    const __m512 v0  = _mm512_setzero_ps();
    uint32_t     x   = 0;
    for (; x + 16 <= yl->w; x += 16) {
        __m512 yv;
        if (yl->highbd) {
            yv = _mm512_cvtepu32_ps(_mm512_cvtepu16_epi32(_mm256_loadu_si256(
                (const __m256i*)(const void*)((const uint16_t*)yl->data + (size_t)ly * yl->stride + x))));
        } else {
            yv = _mm512_cvtepu32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128(
                (const __m128i*)(const void*)((const uint8_t*)yl->data + (size_t)ly * yl->stride + x))));
        }
        _mm512_storeu_ps(dst + x, mcd_merge_mask512(yv, vK, v16, v59, v1, v0));
    }
    for (; x < yl->w; x++) {
        dst[x] = mcd_merge_mask(yl, x, ly, kK);
    }
}

/* One plane, 16 lanes per chunk. For the luma plane (is_luma) the mask is
 * the pixel's own luma mask; chroma (4:2:0 only) gathers precomputed luma
 * mask rows kept in `mrows` (2 * yl->w floats; rows min(2y(+1), h-1),
 * values = the same luma mask formula), keeping the scalar box tap order
 * ((m00 + m01) + m10) + m11. */
void mcd_merge_plane512(const McdPlane* s, const McdPlane* f, McdPlane* o, const McdPlane* yl, int is_luma,
                        float* mrows, float inv_pmax, uint32_t pmax) {
    const float    kK     = 255.0f * inv_pmax;
    const __m512   vK     = _mm512_set1_ps(kK);
    const __m512   v16    = _mm512_set1_ps(16.0f);
    const __m512   v59    = _mm512_set1_ps(59.0f);
    const __m512   v1     = _mm512_set1_ps(1.0f);
    const __m512   v0     = _mm512_setzero_ps();
    const __m512   vhal   = _mm512_set1_ps(0.5f);
    const __m512i  vpmaxi = _mm512_set1_epi32((int)pmax);
    const uint32_t lhm1   = yl->h - 1;

    for (uint32_t y = 0; y < s->h; y++) {
        const uint8_t* srow  = (const uint8_t*)s->data + (size_t)y * s->stride * (s->highbd ? 2 : 1);
        const uint8_t* frow  = (const uint8_t*)f->data + (size_t)y * f->stride * (s->highbd ? 2 : 1);
        uint8_t*       orow  = (uint8_t*)o->data + (size_t)y * o->stride * (s->highbd ? 2 : 1);
        const float *  mrow0 = NULL, *mrow1 = NULL;
        if (!is_luma) {
            /* each chroma row reads its own pair of luma rows (clamped at
             * the bottom edge), matching the scalar box mask */
            const uint32_t r0 = (2 * y < lhm1) ? 2 * y : lhm1;
            const uint32_t r1 = (2 * y + 1 < lhm1) ? 2 * y + 1 : lhm1;
            mcd_merge_mask_row512(yl, r0, mrows, kK);
            mcd_merge_mask_row512(yl, r1, mrows + yl->w, kK);
            mrow0 = mrows;
            mrow1 = mrows + yl->w;
        }
        const uint32_t lwm1 = yl->w - 1;
        uint32_t       x    = 0;
        for (; x + 16 <= s->w; x += 16) {
            __m512 m;
            if (is_luma) {
                __m512 yv;
                if (s->highbd) {
                    yv = _mm512_cvtepu32_ps(_mm512_cvtepu16_epi32(_mm256_loadu_si256(
                        (const __m256i*)(const void*)((const uint16_t*)yl->data + (size_t)y * yl->stride + x))));
                } else {
                    yv = _mm512_cvtepu32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128(
                        (const __m128i*)(const void*)((const uint8_t*)yl->data + (size_t)y * yl->stride + x))));
                }
                m = mcd_merge_mask512(yv, vK, v16, v59, v1, v0);
            } else {
                /* box mask over the 2x2 luma footprint, tap order
                 * ((m00 + m01) + m10) + m11, per-lane clamped indices */
                const __m512i twox = _mm512_add_epi32(
                    _mm512_set1_epi32((int)(2 * x)),
                    _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30));
                const __m512i xx0 = _mm512_min_epi32(twox, _mm512_set1_epi32((int)lwm1));
                const __m512i xx1 = _mm512_min_epi32(_mm512_add_epi32(twox, _mm512_set1_epi32(1)),
                                                     _mm512_set1_epi32((int)lwm1));
                const __m512  g00 = _mm512_i32gather_ps(xx0, mrow0, 4);
                const __m512  g01 = _mm512_i32gather_ps(xx1, mrow0, 4);
                const __m512  g10 = _mm512_i32gather_ps(xx0, mrow1, 4);
                const __m512  g11 = _mm512_i32gather_ps(xx1, mrow1, 4);
                m                 = _mm512_div_ps(_mm512_add_ps(_mm512_add_ps(_mm512_add_ps(g00, g01), g10), g11),
                                                  _mm512_set1_ps(4.0f));
            }
            __m512 sv, fv;
            if (s->highbd) {
                sv = _mm512_cvtepu32_ps(_mm512_cvtepu16_epi32(
                    _mm256_loadu_si256((const __m256i*)(const void*)(((const uint16_t*)srow) + x))));
                fv = _mm512_cvtepu32_ps(_mm512_cvtepu16_epi32(
                    _mm256_loadu_si256((const __m256i*)(const void*)(((const uint16_t*)frow) + x))));
            } else {
                sv = _mm512_cvtepu32_ps(
                    _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(const void*)((const uint8_t*)srow + x))));
                fv = _mm512_cvtepu32_ps(
                    _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(const void*)((const uint8_t*)frow + x))));
            }
            const __m512i iv = mcd_merge_out512(sv, fv, m, vhal, vpmaxi);
            if (s->highbd) {
                _mm256_storeu_si256((__m256i*)(void*)(((uint16_t*)orow) + x), _mm512_cvtusepi32_epi16(iv));
            } else {
                const __m256i p16 = _mm512_cvtusepi32_epi16(iv);
                _mm_storeu_si128((__m128i*)(void*)((uint8_t*)orow + x),
                                 _mm_packus_epi16(_mm256_castsi256_si128(p16), _mm256_extracti128_si256(p16, 1)));
            }
        }
        for (; x < s->w; x++) {
            /* chroma reaching here is always 4:2:0 (dispatcher gate) */
            const float m = is_luma ? mcd_merge_mask(yl, x, y, kK) : mcd_merge_box(yl, x, y, 1u, 1u, kK);
            mcd_merge_pix(s, f, o, x, y, m, pmax);
        }
    }
}

#endif /* EN_AVX512_SUPPORT */

#define MCD_SIMD_EPSSH (1e-15f)
#define MCD_SIMD_ACC_MARGIN 96

#if EN_AVX512_SUPPORT

/* One 1D FFT16 over the 16 vector slots at re/im, where slot k lives at
 * base + k * sstride floats (sstride == 16: contiguous row slots,
 * sstride == 256: column slots, transforming in place with no staging
 * copies — the same 16-lane slot values are loaded/stored, only the
 * addresses differ, so results are identical to the staged version). */
static inline void mcd_fft16_vec(float* re, float* im, const struct McdDftCtx* c, int inverse, uint32_t sstride) {
    for (uint32_t k = 0; k < MCD_DFT_N; k++) {
        const uint32_t j = c->brev[k];
        if (j > k) {
            __m512 t = _mm512_loadu_ps(re + k * sstride);
            _mm512_storeu_ps(re + k * sstride, _mm512_loadu_ps(re + j * sstride));
            _mm512_storeu_ps(re + j * sstride, t);
            t = _mm512_loadu_ps(im + k * sstride);
            _mm512_storeu_ps(im + k * sstride, _mm512_loadu_ps(im + j * sstride));
            _mm512_storeu_ps(im + j * sstride, t);
        }
    }
    for (uint32_t len = 2; len <= MCD_DFT_N; len <<= 1) {
        const uint32_t half  = len >> 1;
        const uint32_t tstep = MCD_DFT_N / len;
        for (uint32_t ib = 0; ib < MCD_DFT_N; ib += len) {
            for (uint32_t j = 0; j < half; j++) {
                const __m512 wr = _mm512_set1_ps(c->twr[j * tstep]);
                const __m512 wi = _mm512_set1_ps(inverse ? -c->twi[j * tstep] : c->twi[j * tstep]);
                float*       pr = re + (ib + j) * sstride;
                float*       pi = im + (ib + j) * sstride;
                float*       qr = re + (ib + j + half) * sstride;
                float*       qi = im + (ib + j + half) * sstride;
                const __m512 ar = _mm512_loadu_ps(pr);
                const __m512 ai = _mm512_loadu_ps(pi);
                const __m512 br = _mm512_loadu_ps(qr);
                const __m512 bi = _mm512_loadu_ps(qi);
                const __m512 tr = _mm512_sub_ps(_mm512_mul_ps(br, wr), _mm512_mul_ps(bi, wi));
                const __m512 ti = _mm512_add_ps(_mm512_mul_ps(br, wi), _mm512_mul_ps(bi, wr));
                _mm512_storeu_ps(pr, _mm512_add_ps(ar, tr));
                _mm512_storeu_ps(pi, _mm512_add_ps(ai, ti));
                _mm512_storeu_ps(qr, _mm512_sub_ps(ar, tr));
                _mm512_storeu_ps(qi, _mm512_sub_ps(ai, ti));
            }
        }
    }
}

/* One 2D transform of the group buffer: rows (contiguous slots), then
 * columns (slots with fixed x, stride 16 slots) transformed in place. */
static inline void mcd_fft2d_vec(float* re, float* im, const struct McdDftCtx* c, int inverse) {
    for (uint32_t y = 0; y < MCD_DFT_N; y++) {
        mcd_fft16_vec(re + y * (MCD_DFT_N * 16), im + y * (MCD_DFT_N * 16), c, inverse, 16);
    }
    for (uint32_t x = 0; x < MCD_DFT_N; x++) {
        mcd_fft16_vec(re + x * 16, im + x * 16, c, inverse, MCD_DFT_N * 16);
    }
}

/* dfttest of a 16-block horizontal strip (lanes = block index b of
 * bx + 4*b). rex/imx: caller-provided 256*16-float scratch buffers (the
 * column FFTs transform in place, no staging scratch needed). */
static void mcd_dfttest_group(const struct McdDftCtx* ctx, const float* pad, uint32_t pdw, int32_t bx, int32_t by,
                              uint32_t aw, int32_t ymax, float* acc, float* rex, float* imx) {
    /* gather offsets for the windowed input load: prow already points at
     * column (bx + PAD), so lane b (block bx + 4b) reads at + (4b + x) */
    __m512i idx4[16];
    for (uint32_t x = 0; x < 16; x++) {
        idx4[x] = _mm512_add_epi32(_mm512_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60),
                                   _mm512_set1_epi32((int)x));
    }
    /* gather offsets for the OPA: element (y, x, lane) of the group buffer
     * sits at (y*16 + x)*16 + lane; per (y, x) the offset is 256*y + 16*x */
    __m512i idxg[16];
    for (uint32_t y = 0; y < 16; y++) {
        idxg[y] = _mm512_add_epi32(
            _mm512_setr_epi32(0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240),
            _mm512_set1_epi32(256 * (int)y));
    }

    /* windowed load: slot (y*16+x) = pad[by+y][bx + 4*lane + x] * win */
    for (uint32_t y = 0; y < MCD_DFT_N; y++) {
        const float* prow = pad + (size_t)(by + y + MCD_DFT_PAD) * pdw + (bx + MCD_DFT_PAD);
        for (uint32_t x = 0; x < MCD_DFT_N; x++) {
            const __m512 v = _mm512_mul_ps(_mm512_i32gather_ps(idx4[x], prow, 4),
                                           _mm512_set1_ps(ctx->win[y * MCD_DFT_N + x]));
            _mm512_storeu_ps(rex + (y * MCD_DFT_N + x) * 16, v);
            _mm512_storeu_ps(imx + (y * MCD_DFT_N + x) * 16, _mm512_setzero_ps());
        }
    }

    mcd_fft2d_vec(rex, imx, ctx, 0);

    if (ctx->sigma_pct) {
        /* zmean + Wiener shrinkage + zmean restore, fused into ONE pass over
         * the 256 slots (two slots per iteration to hide the division
         * latency). Per element the float operation sequence is exactly the
         * scalar/SIMD-reference one — gf*wfre mul, zmean sub, psd muls/add,
         * num sub, den add, div, blend, shrink muls, restore adds — merely
         * interleaved across slots; slots are independent, so results are
         * bit-identical to the previous three-pass version. No FMA (this TU
         * is compiled with -ffp-contract=off). */
        const __m512 gf   = _mm512_div_ps(_mm512_loadu_ps(rex), _mm512_set1_ps(ctx->wfre[0]));
        const __m512 eps  = _mm512_set1_ps(MCD_SIMD_EPSSH);
        const __m512 zero = _mm512_setzero_ps();
        for (uint32_t k = 0; k < MCD_DFT_N2; k += 2) {
            float*       pr0 = rex + k * 16;
            float*       pi0 = imx + k * 16;
            float*       pr1 = pr0 + 16;
            float*       pi1 = pi0 + 16;
            const __m512 wr0 = _mm512_mul_ps(gf, _mm512_set1_ps(ctx->wfre[k]));
            const __m512 wi0 = _mm512_mul_ps(gf, _mm512_set1_ps(ctx->wfim[k]));
            const __m512 wr1 = _mm512_mul_ps(gf, _mm512_set1_ps(ctx->wfre[k + 1]));
            const __m512 wi1 = _mm512_mul_ps(gf, _mm512_set1_ps(ctx->wfim[k + 1]));
            /* zmean: remove the windowed local mean (per block = per lane) */
            const __m512 re0 = _mm512_sub_ps(_mm512_loadu_ps(pr0), wr0);
            const __m512 im0 = _mm512_sub_ps(_mm512_loadu_ps(pi0), wi0);
            const __m512 re1 = _mm512_sub_ps(_mm512_loadu_ps(pr1), wr1);
            const __m512 im1 = _mm512_sub_ps(_mm512_loadu_ps(pi1), wi1);
            /* type-0 Wiener shrinkage */
            const __m512    psd0  = _mm512_add_ps(_mm512_mul_ps(re0, re0), _mm512_mul_ps(im0, im0));
            const __m512    psd1  = _mm512_add_ps(_mm512_mul_ps(re1, re1), _mm512_mul_ps(im1, im1));
            const __m512    num0  = _mm512_sub_ps(psd0, _mm512_set1_ps(ctx->sigma2[k]));
            const __m512    num1  = _mm512_sub_ps(psd1, _mm512_set1_ps(ctx->sigma2[k + 1]));
            const __m512    den0  = _mm512_add_ps(psd0, eps);
            const __m512    den1  = _mm512_add_ps(psd1, eps);
            const __mmask16 mask0 = _mm512_cmp_ps_mask(num0, zero, _CMP_GT_OQ);
            const __mmask16 mask1 = _mm512_cmp_ps_mask(num1, zero, _CMP_GT_OQ);
            const __m512    mult0 = _mm512_mask_blend_ps(mask0, zero, _mm512_div_ps(num0, den0));
            const __m512    mult1 = _mm512_mask_blend_ps(mask1, zero, _mm512_div_ps(num1, den1));
            /* restore the mean */
            _mm512_storeu_ps(pr0, _mm512_add_ps(_mm512_mul_ps(re0, mult0), wr0));
            _mm512_storeu_ps(pi0, _mm512_add_ps(_mm512_mul_ps(im0, mult0), wi0));
            _mm512_storeu_ps(pr1, _mm512_add_ps(_mm512_mul_ps(re1, mult1), wr1));
            _mm512_storeu_ps(pi1, _mm512_add_ps(_mm512_mul_ps(im1, mult1), wi1));
        }
    }

    mcd_fft2d_vec(rex, imx, ctx, 1);

    /* overlap-add synthesis, per block in ascending order (matching the
     * scalar block order exactly): block b's element (y, x) lives at
     * rex[(y*16+x)*16 + b]; gather it row-wise, multiply by the synthesis
     * window, and add into the accumulator with a read-add-write. The acc
     * buffer carries MCD_SIMD_ACC_MARGIN extra samples before the visible
     * core, so blocks at negative/beyond-frame positions stay in bounds;
     * finalization reads only the visible core. Per element the sequence
     * of float additions is identical to the scalar loop. */
    for (uint32_t b = 0; b < 16; b++) {
        const int32_t obx = bx + 4 * (int32_t)b;
        for (uint32_t y = 0; y < MCD_DFT_N; y++) {
            const int32_t oy = by + (int32_t)y;
            if (oy < 0 || oy > ymax) {
                continue;
            }
            const __m512 v = _mm512_mul_ps(
                _mm512_mask_i32gather_ps(_mm512_setzero_ps(),
                                         (__mmask16)0xffff,
                                         _mm512_add_epi32(idxg[y], _mm512_set1_epi32((int)b)),
                                         rex,
                                         4),
                _mm512_loadu_ps(ctx->win + y * MCD_DFT_N));
            float* ar = acc + (size_t)oy * aw + obx;
            _mm512_storeu_ps(ar, _mm512_add_ps(_mm512_loadu_ps(ar), v));
        }
    }
}

/* SIMD dfttest plane: mirrors svt_mcd_dfttest_plane_with_ctx (same padding,
 * same block grid, same arithmetic order) with 16-block vector groups. */
int svt_mcd_simd_dfttest_plane_with_ctx(const struct McdDftCtx* ctx, const McdPlane* src, McdPlane* dst,
                                        uint32_t bitdepth) {
    if (!ctx || !src || !dst || !src->data || (bitdepth != 8 && bitdepth != 10 && bitdepth != 12) ||
        src->highbd != (bitdepth > 8)) {
        return MCD_ERR_PARAM;
    }
    const uint32_t w = src->w;
    const uint32_t h = src->h;
    if (w == 0 || h == 0 || src->stride < w) {
        return MCD_ERR_PARAM;
    }

    const uint32_t pmax   = (1u << bitdepth) - 1;
    const float    iscale = 1.0f / (float)(1u << (bitdepth - 8));
    const float    oscale = (float)(1u << (bitdepth - 8));

    if (!dst->data) {
        dst->data = malloc((size_t)w * h * (src->highbd ? sizeof(uint16_t) : sizeof(uint8_t)));
        if (!dst->data) {
            return MCD_ERR_NOMEM;
        }
        dst->w      = w;
        dst->h      = h;
        dst->stride = w;
        dst->highbd = src->highbd;
    } else if (dst->w != w || dst->h != h || dst->stride < w || dst->highbd != src->highbd) {
        return MCD_ERR_PARAM;
    }

    const uint32_t pw = w + 2 * MCD_DFT_PAD;
    /* extra columns so the group's gathers for off-grid blocks of the last
     * strip stay in bounds (their values are never accumulated) */
    const uint32_t pdw = pw + 4 * (MCD_DFT_N - 1) + MCD_DFT_N;
    const uint32_t ph  = h + 2 * MCD_DFT_PAD;
    /* accumulation buffer with a per-row margin on both sides so block
     * overhang past the frame edges lands in the row's own margin (visible
     * core of row y = [MARGIN, MARGIN + w) within the padded row) */
    const uint32_t aw = w + 2 * MCD_SIMD_ACC_MARGIN;

    /* NOTE (deliberately NOT pooled): a per-thread grow-to-fit pool for
     * these buffers was measured A/B at 1080p and kept ~18 MB per PA thread
     * resident (~+260 MB max RSS in total) for an uncertain ~3.5% wall-time
     * gain. The malloc/free churn of the two large buffers is mmap-backed
     * (free returns the pages; calloc maps fresh zero pages lazily), so the
     * per-call allocation is nearly free while the resident cost is real.
     * pad/acc keep their calloc: pad so off-grid gathers read (unused)
     * initialized columns (MSan quiet), acc because it is a += target. */
    int32_t* xmap    = malloc(sizeof(int32_t) * pdw);
    int32_t* ymap    = malloc(sizeof(int32_t) * ph);
    float*   pad     = calloc((size_t)pdw * ph, sizeof(float));
    float*   acc     = calloc((size_t)aw * h, sizeof(float));
    float*   scratch = malloc(2 * 256 * 16 * sizeof(float));
    int      ret     = MCD_SUCCESS;

    if (!xmap || !ymap || !pad || !acc || !scratch) {
        ret = MCD_ERR_NOMEM;
        goto cleanup;
    }

    for (uint32_t x = 0; x < pw; x++) {
        xmap[x] = mcd_simd_mirror((int32_t)x - MCD_DFT_PAD, w);
    }
    for (uint32_t x = pw; x < pdw; x++) {
        xmap[x] = w - 1; /* off-grid gather clamp (values never used) */
    }
    for (uint32_t y = 0; y < ph; y++) {
        ymap[y] = mcd_simd_mirror((int32_t)y - MCD_DFT_PAD, h);
    }

    /* mirror-padded copy in the internal 8-bit-range float scale: the
     * core region is a contiguous conversion (vector), the margins keep
     * the mirrored gather (scalar) */
    for (uint32_t y = 0; y < ph; y++) {
        const size_t srow = (size_t)ymap[y] * src->stride;
        float*       pr   = pad + (size_t)y * pdw;
        /* interior output column x mirrors source column x - MCD_DFT_PAD */
        if (src->highbd) {
            const uint16_t* s = src->data;
            uint32_t        x = MCD_DFT_PAD;
            for (; x + 16 <= MCD_DFT_PAD + w; x += 16) {
                const __m512 f = _mm512_cvtepu32_ps(_mm512_cvtepu16_epi32(
                    _mm256_loadu_si256((const __m256i*)(const void*)(s + srow + x - MCD_DFT_PAD))));
                _mm512_storeu_ps(pr + x, _mm512_mul_ps(f, _mm512_set1_ps(iscale)));
            }
            for (; x < MCD_DFT_PAD + w; x++) {
                pr[x] = (float)s[srow + x - MCD_DFT_PAD] * iscale;
            }
            for (x = 0; x < MCD_DFT_PAD; x++) {
                pr[x] = (float)s[srow + xmap[x]] * iscale;
            }
            for (x = MCD_DFT_PAD + w; x < pw; x++) {
                pr[x] = (float)s[srow + xmap[x]] * iscale;
            }
        } else {
            const uint8_t* s = src->data;
            uint32_t       x = MCD_DFT_PAD;
            for (; x + 16 <= MCD_DFT_PAD + w; x += 16) {
                const __m512 f = _mm512_cvtepu32_ps(
                    _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(const void*)(s + srow + x - MCD_DFT_PAD))));
                _mm512_storeu_ps(pr + x, _mm512_mul_ps(f, _mm512_set1_ps(iscale)));
            }
            for (; x < MCD_DFT_PAD + w; x++) {
                pr[x] = (float)s[srow + x - MCD_DFT_PAD] * iscale;
            }
            for (x = 0; x < MCD_DFT_PAD; x++) {
                pr[x] = (float)s[srow + xmap[x]] * iscale;
            }
            for (x = MCD_DFT_PAD + w; x < pw; x++) {
                pr[x] = (float)s[srow + xmap[x]] * iscale;
            }
        }
    }

    {
        const int32_t xmax = (int32_t)w - 1;
        const int32_t ymax = (int32_t)h - 1;
        float*        rex  = scratch;
        float*        imx  = scratch + 256 * 16;

        for (int32_t by = -(MCD_DFT_N - MCD_DFT_STEP); by <= ymax; by += MCD_DFT_STEP) {
            for (int32_t bx = -(MCD_DFT_N - MCD_DFT_STEP); bx <= xmax; bx += MCD_DFT_STEP) {
                mcd_dfttest_group(ctx, pad, pdw, bx, by, aw, ymax, acc + MCD_SIMD_ACC_MARGIN, rex, imx);
                /* 16 blocks per group: skip the 15 covered positions */
                bx += 15 * MCD_DFT_STEP;
                if (bx > xmax) {
                    break;
                }
            }
        }
    }

    /* back to native scale, rounded and clamped (visible core only):
     * (int32_t)(acc * oscale + 0.5f) is a truncating conversion, so the
     * vector path uses cvttps and identical clamps - bit-exact */
    {
        const __m512 vos = _mm512_set1_ps(oscale);
        const __m512 vh  = _mm512_set1_ps(0.5f);
        for (uint32_t y = 0; y < h; y++) {
            const float* ar = acc + MCD_SIMD_ACC_MARGIN + (size_t)y * aw;
            uint32_t     x  = 0;
            if (dst->highbd) {
                uint16_t* dr = (uint16_t*)dst->data + (size_t)y * dst->stride;
                for (; x + 16 <= w; x += 16) {
                    __m512i v = _mm512_cvttps_epi32(_mm512_add_ps(_mm512_mul_ps(_mm512_loadu_ps(ar + x), vos), vh));
                    v         = _mm512_max_epi32(v, _mm512_setzero_si512());
                    v         = _mm512_min_epi32(v, _mm512_set1_epi32((int)pmax));
                    _mm256_storeu_si256((__m256i*)(void*)(dr + x), _mm512_cvtusepi32_epi16(v));
                }
            } else {
                uint8_t* dr = (uint8_t*)dst->data + (size_t)y * dst->stride;
                for (; x + 16 <= w; x += 16) {
                    __m512i v   = _mm512_cvttps_epi32(_mm512_add_ps(_mm512_mul_ps(_mm512_loadu_ps(ar + x), vos), vh));
                    v           = _mm512_max_epi32(v, _mm512_setzero_si512());
                    v           = _mm512_min_epi32(v, _mm512_set1_epi32((int)pmax));
                    __m256i p16 = _mm512_cvtusepi32_epi16(v);
                    __m128i p8  = _mm_packus_epi16(_mm256_castsi256_si128(p16), _mm256_extracti128_si256(p16, 1));
                    _mm_storeu_si128((__m128i*)(void*)(dr + x), p8);
                }
            }
            for (; x < w; x++) {
                int32_t v = (int32_t)(ar[x] * oscale + 0.5f);
                if (v < 0) {
                    v = 0;
                }
                if (v > (int32_t)pmax) {
                    v = (int32_t)pmax;
                }
                if (dst->highbd) {
                    ((uint16_t*)dst->data)[(size_t)y * dst->stride + x] = (uint16_t)v;
                } else {
                    ((uint8_t*)dst->data)[(size_t)y * dst->stride + x] = (uint8_t)v;
                }
            }
        }
    }

cleanup:
    free(xmap);
    free(ymap);
    free(pad);
    free(acc);
    free(scratch);
    return ret;
}

#endif /* EN_AVX512_SUPPORT */

/* Mask-merge RTCD kernel: validates the call and drives the per-plane
 * vector merge. Returns MCD_ERR_PARAM without writing when unsupported. */
int mcd_avx512_mask_merge(const McdPlane* src_planes, const McdPlane* filt_planes, McdPlane* out_planes,
                          uint32_t n_planes, uint32_t x_ss, uint32_t y_ss, uint32_t bitdepth) {
    if (n_planes != 1 && n_planes != 3) {
        return MCD_ERR_PARAM;
    }
    if (n_planes == 3 && !(x_ss == 1 && y_ss == 1)) {
        return MCD_ERR_PARAM;
    }
    if (bitdepth != 8 && bitdepth != 10) {
        return MCD_ERR_PARAM;
    }
    for (uint32_t p = 0; p < n_planes; p++) {
        const McdPlane* s = &src_planes[p];
        const McdPlane* f = &filt_planes[p];
        const McdPlane* o = &out_planes[p];
        if (!s->data || !f->data || !o->data || s->w == 0 || s->h == 0 || s->stride < s->w || f->w != s->w ||
            f->h != s->h || f->stride < f->w || o->w != s->w || o->h != s->h || o->stride < o->w ||
            f->highbd != s->highbd || o->highbd != s->highbd) {
            return MCD_ERR_PARAM;
        }
    }
    const uint32_t  pmax     = (1u << bitdepth) - 1;
    const float     inv_pmax = 1.0f / (float)pmax;
    const McdPlane* yl       = &src_planes[0];
    float*          mrows    = n_planes == 3 ? (float*)malloc(2 * (size_t)yl->w * sizeof(float)) : NULL;
    if (n_planes == 3 && !mrows) {
        return MCD_ERR_NOMEM;
    }
    for (uint32_t p = 0; p < n_planes; p++) {
        mcd_avx512_merge_plane512(&src_planes[p], &filt_planes[p], &out_planes[p], yl, p == 0, mrows, inv_pmax, pmax);
    }
    free(mrows);
    return MCD_SUCCESS;
}
