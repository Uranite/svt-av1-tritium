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
 * mc_denoise_simd.c - AVX2 kernels for the motion-compensated denoiser.
 *
 * This translation unit is compiled with -mavx2 / /arch:AVX2 by the build system (see
 * Source/Lib/ASM_AVX2/CMakeLists.txt), so the instructions are available
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

/* AVX2 kernels: compiled with -mavx2 / /arch:AVX2 by the build system. */
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
#define svt_mcd_simd_sad_u8u8 mcd_avx2_sad_u8u8
#define svt_mcd_simd_sad_u16u16 mcd_avx2_sad_u16u16
#define svt_mcd_simd_sad_u8u16 mcd_avx2_sad_u8u16
#define svt_mcd_simd_sad_u16u8 mcd_avx2_sad_u16u8
#define svt_mcd_simd_fetch_u8_row mcd_avx2_fetch_u8_row
#define svt_mcd_simd_halfpel_h_u8 mcd_avx2_halfpel_h_u8
#define svt_mcd_simd_halfpel_v_u8 mcd_avx2_halfpel_v_u8
#define svt_mcd_simd_degrain_init_row mcd_avx2_degrain_init_row
#define svt_mcd_simd_degrain_add_ref mcd_avx2_degrain_add_ref
#define svt_mcd_simd_degrain_shift mcd_avx2_degrain_shift
#define svt_mcd_simd_degrain_acc_row mcd_avx2_degrain_acc_row
#define svt_mcd_simd_degrain_finalize_row mcd_avx2_degrain_finalize_row
#define mcd_merge_plane256 mcd_avx2_merge_plane256

uint64_t svt_mcd_simd_sad_u8u8(const uint8_t* a, int32_t sa, const uint8_t* b, int32_t sb, uint32_t w, uint32_t h,
                               uint64_t bound) {
    __m128i  vacc = _mm_setzero_si128(); /* 2 x u64 partial sums */
    uint64_t tail = 0;
    uint32_t y;
    /* bound > 0: partial-row early abort (see mcd_sad_* in mc_denoise_me.c).
     * Exact: partial sums are monotone, so returning once the accumulated
     * total reaches `bound` only affects candidates that cannot win; the
     * returned value is a lower bound of the full SAD and >= bound. */
    for (y = 0; y < h; y++) {
        uint32_t x = 0;
        for (; x + 16 <= w; x += 16) {
            const __m128i va = _mm_loadu_si128((const __m128i*)(const void*)(a + x));
            const __m128i vb = _mm_loadu_si128((const __m128i*)(const void*)(b + x));
            vacc             = _mm_add_epi64(vacc, _mm_sad_epu8(va, vb));
        }
        if (x + 8 <= w) {
            const __m128i va = _mm_loadl_epi64((const __m128i*)(const void*)(a + x));
            const __m128i vb = _mm_loadl_epi64((const __m128i*)(const void*)(b + x));
            tail += (uint64_t)_mm_cvtsi128_si64(_mm_sad_epu8(va, vb));
            x += 8;
        }
        for (; x < w; x++) {
            const int d = (int)a[x] - (int)b[x];
            tail += (uint64_t)(d < 0 ? -d : d);
        }
        if (bound) {
            const __m128i  acc    = _mm_add_epi64(vacc, _mm_unpackhi_epi64(vacc, vacc));
            const uint64_t so_far = tail + (uint64_t)_mm_cvtsi128_si64(acc);
            if (so_far >= bound) {
                return so_far;
            }
        }
        a += sa;
        b += sb;
    }
    const __m128i v = _mm_add_epi64(vacc, _mm_unpackhi_epi64(vacc, vacc));
    return tail + (uint64_t)_mm_cvtsi128_si64(v);
}

/* 16-bit SAD reduction: |a-b| per u16 lane, pairwise-widened to u32 with
 * madd(ones), u32 lane accumulation. Safe for every block this module uses
 * (w,h <= 32): max sum = 1024 * 65535 < 2^31; the guard falls back to the
 * scalar path (via the caller) for anything larger. */

static inline uint64_t mcd_sad_u16_reduce(const __m256i vacc, uint64_t tail) {
    __m128i v = _mm_add_epi32(_mm256_castsi256_si128(vacc), _mm256_extracti128_si256(vacc, 1));
    v         = _mm_add_epi32(v, _mm_shuffle_epi32(v, _MM_SHUFFLE(1, 0, 3, 2)));
    v         = _mm_add_epi32(v, _mm_shuffle_epi32(v, _MM_SHUFFLE(2, 3, 0, 1)));
    return tail + (uint64_t)(uint32_t)_mm_cvtsi128_si32(v);
}

uint64_t svt_mcd_simd_sad_u16u16(const uint16_t* a, int32_t sa, const uint16_t* b, int32_t sb, uint32_t w, uint32_t h,
                                 uint64_t bound) {
    if ((uint64_t)w * h * 65535ull >= 0x7fffffffull) {
        return 0; /* unreachable for module block sizes */
    }
    const __m256i ones = _mm256_set1_epi16(1);
    __m256i       vacc = _mm256_setzero_si256(); /* 8 x u32 */
    uint64_t      tail = 0;
    for (uint32_t y = 0; y < h; y++) {
        uint32_t x = 0;
        for (; x + 16 <= w; x += 16) {
            const __m256i va = _mm256_loadu_si256((const __m256i*)(const void*)(a + x));
            const __m256i vb = _mm256_loadu_si256((const __m256i*)(const void*)(b + x));
            __m256i       d  = _mm256_abs_epi16(_mm256_sub_epi16(va, vb));
            vacc             = _mm256_add_epi32(vacc, _mm256_madd_epi16(d, ones));
        }
        if (x + 8 <= w) {
            const __m256i va = _mm256_inserti128_si256(
                _mm256_setzero_si256(), _mm_loadu_si128((const __m128i*)(const void*)(a + x)), 0);
            const __m256i vb = _mm256_inserti128_si256(
                _mm256_setzero_si256(), _mm_loadu_si128((const __m128i*)(const void*)(b + x)), 0);
            __m256i d = _mm256_abs_epi16(_mm256_sub_epi16(va, vb));
            vacc      = _mm256_add_epi32(vacc, _mm256_madd_epi16(d, ones));
            x += 8;
        }
        for (; x < w; x++) {
            const int d = (int)a[x] - (int)b[x];
            tail += (uint64_t)(d < 0 ? -d : d);
        }
        if (bound && (y & 3) == 3) {
            const uint64_t so_far = mcd_sad_u16_reduce(vacc, tail);
            if (so_far >= bound) {
                return so_far;
            }
        }
        a += sa;
        b += sb;
    }
    return mcd_sad_u16_reduce(vacc, tail);
}

uint64_t svt_mcd_simd_sad_u8u16(const uint8_t* a, int32_t sa, const uint16_t* b, int32_t sb, uint32_t w, uint32_t h,
                                uint64_t bound) {
    if ((uint64_t)w * h * 65535ull >= 0x7fffffffull) {
        return 0;
    }
    const __m256i ones = _mm256_set1_epi16(1);
    __m256i       vacc = _mm256_setzero_si256();
    uint64_t      tail = 0;
    for (uint32_t y = 0; y < h; y++) {
        uint32_t x = 0;
        for (; x + 8 <= w; x += 8) {
            const __m256i va = _mm256_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)(const void*)(a + x)));
            const __m256i vb = _mm256_inserti128_si256(
                _mm256_setzero_si256(), _mm_loadu_si128((const __m128i*)(const void*)(b + x)), 0);
            __m256i d = _mm256_abs_epi16(_mm256_sub_epi16(va, vb));
            vacc      = _mm256_add_epi32(vacc, _mm256_madd_epi16(d, ones));
        }
        for (; x < w; x++) {
            const int d = (int)a[x] - (int)b[x];
            tail += (uint64_t)(d < 0 ? -d : d);
        }
        if (bound && (y & 3) == 3) {
            const uint64_t so_far = mcd_sad_u16_reduce(vacc, tail);
            if (so_far >= bound) {
                return so_far;
            }
        }
        a += sa;
        b += sb;
    }
    return mcd_sad_u16_reduce(vacc, tail);
}

uint64_t svt_mcd_simd_sad_u16u8(const uint16_t* a, int32_t sa, const uint8_t* b, int32_t sb, uint32_t w, uint32_t h,
                                uint64_t bound) {
    if ((uint64_t)w * h * 65535ull >= 0x7fffffffull) {
        return 0;
    }
    const __m256i ones = _mm256_set1_epi16(1);
    __m256i       vacc = _mm256_setzero_si256();
    uint64_t      tail = 0;
    for (uint32_t y = 0; y < h; y++) {
        uint32_t x = 0;
        for (; x + 8 <= w; x += 8) {
            const __m256i va = _mm256_inserti128_si256(
                _mm256_setzero_si256(), _mm_loadu_si128((const __m128i*)(const void*)(a + x)), 0);
            const __m256i vb = _mm256_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)(const void*)(b + x)));
            __m256i       d  = _mm256_abs_epi16(_mm256_sub_epi16(va, vb));
            vacc             = _mm256_add_epi32(vacc, _mm256_madd_epi16(d, ones));
        }
        for (; x < w; x++) {
            const int d = (int)a[x] - (int)b[x];
            tail += (uint64_t)(d < 0 ? -d : d);
        }
        if (bound && (y & 3) == 3) {
            const uint64_t so_far = mcd_sad_u16_reduce(vacc, tail);
            if (so_far >= bound) {
                return so_far;
            }
        }
        a += sa;
        b += sb;
    }
    return mcd_sad_u16_reduce(vacc, tail);
}

/* ================================================================== */
/* SATD: sum over 4x4 sub-blocks of the abs-sums of a 2D Hadamard of   */
/* the prediction error (unnormalized; matches mcd_satd_4x4 exactly).  */
/*                                                                     */
/* 16 lanes = the 16 sub-blocks of a 16x16 region; the d[r][c] arrays  */
/* hold, per lane, the error at offset (r,c) of that sub-block. Both   */
/* Hadamard passes combine values ACROSS the d/t arrays (element-wise  */
/* vector ops, no shuffles), so the lane arithmetic is identical to    */
/* the scalar int32 butterfly, and integer addition is associative.    */
/* Only used at ME level 0, i.e. w==h==blksize in {8,16,32}; 16x16     */
/* regions (the default) are vectorized, other sizes use the scalar    */
/* fallback in mc_denoise_me.c.                                        */
/* ================================================================== */

void svt_mcd_simd_fetch_u8_row(const uint8_t* row, uint16_t* dst, uint32_t w) {
    uint32_t x = 0;
    for (; x + 16 <= w; x += 16) {
        _mm256_storeu_si256((__m256i*)(void*)(dst + x),
                            _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(const void*)(row + x))));
    }
    for (; x < w; x++) {
        dst[x] = (uint16_t)row[x];
    }
}

static inline void mcd_halfpel_h_row_u8(const uint8_t* srow, uint32_t X, uint8_t* drow) {
    const __m256i s0 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(const void*)(srow + X - 2)));
    const __m256i s1 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(const void*)(srow + X - 1)));
    const __m256i s2 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(const void*)(srow + X)));
    const __m256i s3 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(const void*)(srow + X + 1)));
    const __m256i s4 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(const void*)(srow + X + 2)));
    const __m256i s5 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(const void*)(srow + X + 3)));
    const __m256i t  = _mm256_add_epi16(
        _mm256_add_epi16(_mm256_add_epi16(s0, s5), _mm256_mullo_epi16(_mm256_add_epi16(s2, s3), _mm256_set1_epi16(20))),
        _mm256_mullo_epi16(_mm256_add_epi16(s1, s4), _mm256_set1_epi16(-5)));
    const __m256i v = _mm256_srai_epi16(_mm256_add_epi16(t, _mm256_set1_epi16(16)), 5);
    _mm_storeu_si128((__m128i*)(void*)(drow + X),
                     _mm_packus_epi16(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1)));
}

void svt_mcd_simd_halfpel_h_u8(const uint8_t* src, uint32_t sstride, uint32_t w, uint32_t h, uint32_t pad, uint8_t* dst,
                               uint32_t dstride, uint32_t maxval) {
    const uint32_t tw  = w + 2 * pad;
    const int32_t  wm1 = (int32_t)w - 1;
    for (uint32_t Y = 0; Y < h + 2 * pad; Y++) {
        /* row base WITHOUT the +pad offset: core column c sits at
         * srow[pad + c]; the vector kernel's taps srow[X-2 .. X+18] for
         * output X therefore evaluate core positions x = X - pad, and
         * taps past a core edge land in the replicated margin, which is
         * exactly the scalar clamp. Only the final 18 columns need a
         * scalar (memory-bounded) tail. */
        const uint8_t* srow = src + (size_t)Y * sstride;
        const uint8_t* vrow = srow + pad; /* visible origin (scalar path) */
        uint8_t*       drow = dst + (size_t)Y * dstride;
        uint32_t       X    = 0;
        for (; X < pad + 2 && X < tw; X++) { /* taps clamp into the core */
            const int32_t x = (int32_t)X - (int32_t)pad;
            const int32_t v = (vrow[mcd_simd_clampi(x - 2, 0, wm1)] - 5 * vrow[mcd_simd_clampi(x - 1, 0, wm1)] +
                               20 * vrow[mcd_simd_clampi(x, 0, wm1)] + 20 * vrow[mcd_simd_clampi(x + 1, 0, wm1)] -
                               5 * vrow[mcd_simd_clampi(x + 2, 0, wm1)] + vrow[mcd_simd_clampi(x + 3, 0, wm1)] + 16) >>
                5;
            drow[X] = (uint8_t)(v < 0 ? 0 : (v > (int32_t)maxval ? (int32_t)maxval : v));
        }
        for (X = pad + 2; X + 19 <= tw; X += 16) {
            mcd_halfpel_h_row_u8(srow, X, drow);
        }
        for (; X < tw; X++) {
            const int32_t x = (int32_t)X - (int32_t)pad;
            const int32_t v = (vrow[mcd_simd_clampi(x - 2, 0, wm1)] - 5 * vrow[mcd_simd_clampi(x - 1, 0, wm1)] +
                               20 * vrow[mcd_simd_clampi(x, 0, wm1)] + 20 * vrow[mcd_simd_clampi(x + 1, 0, wm1)] -
                               5 * vrow[mcd_simd_clampi(x + 2, 0, wm1)] + vrow[mcd_simd_clampi(x + 3, 0, wm1)] + 16) >>
                5;
            drow[X] = (uint8_t)(v < 0 ? 0 : (v > (int32_t)maxval ? (int32_t)maxval : v));
        }
    }
}

void svt_mcd_simd_halfpel_v_u8(const uint8_t* src, uint32_t sstride, uint32_t w, uint32_t h, uint32_t pad, uint8_t* dst,
                               uint32_t dstride, uint32_t maxval) {
    const uint32_t tw  = w + 2 * pad;
    const int32_t  hm1 = (int32_t)h - 1;
    for (uint32_t Y = 0; Y < h + 2 * pad; Y++) {
        const int32_t y = (int32_t)Y - (int32_t)pad;
        /* clamped CORE row indices offset by pad into the padded plane;
         * X below indexes padded columns directly */
        const uint8_t* r0   = src + (size_t)(mcd_simd_clampi(y - 2, 0, hm1) + (int32_t)pad) * sstride;
        const uint8_t* r1   = src + (size_t)(mcd_simd_clampi(y - 1, 0, hm1) + (int32_t)pad) * sstride;
        const uint8_t* r2   = src + (size_t)(mcd_simd_clampi(y, 0, hm1) + (int32_t)pad) * sstride;
        const uint8_t* r3   = src + (size_t)(mcd_simd_clampi(y + 1, 0, hm1) + (int32_t)pad) * sstride;
        const uint8_t* r4   = src + (size_t)(mcd_simd_clampi(y + 2, 0, hm1) + (int32_t)pad) * sstride;
        const uint8_t* r5   = src + (size_t)(mcd_simd_clampi(y + 3, 0, hm1) + (int32_t)pad) * sstride;
        uint8_t*       drow = dst + (size_t)Y * dstride;
        uint32_t       X    = 0;
        for (; X + 16 <= tw; X += 16) {
            const __m256i s0 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(const void*)(r0 + X)));
            const __m256i s1 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(const void*)(r1 + X)));
            const __m256i s2 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(const void*)(r2 + X)));
            const __m256i s3 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(const void*)(r3 + X)));
            const __m256i s4 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(const void*)(r4 + X)));
            const __m256i s5 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(const void*)(r5 + X)));
            const __m256i t  = _mm256_add_epi16(
                _mm256_add_epi16(_mm256_add_epi16(s0, s5),
                                 _mm256_mullo_epi16(_mm256_add_epi16(s2, s3), _mm256_set1_epi16(20))),
                _mm256_mullo_epi16(_mm256_add_epi16(s1, s4), _mm256_set1_epi16(-5)));
            const __m256i v = _mm256_srai_epi16(_mm256_add_epi16(t, _mm256_set1_epi16(16)), 5);
            _mm_storeu_si128((__m128i*)(void*)(drow + X),
                             _mm_packus_epi16(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1)));
        }
        for (; X < tw; X++) {
            const int32_t v = (r0[X] - 5 * r1[X] + 20 * r2[X] + 20 * r3[X] - 5 * r4[X] + r5[X] + 16) >> 5;
            drow[X]         = (uint8_t)(v < 0 ? 0 : (v > (int32_t)maxval ? (int32_t)maxval : v));
        }
    }
}

/* 16-bit passes: i32 math (52 * 1023 exceeds i16), 16 outputs per zmm.
 * Saturation to [0, maxval] with maxval <= 0xFFFF, then a truncating
 * narrowing store is exact. */

void svt_mcd_simd_degrain_init_row(uint32_t* bacc, const McdPlane* sp, uint32_t Y, uint32_t x0_p, uint32_t wsrc,
                                   uint32_t n) {
    const uint32_t vs = wsrc;
    uint32_t       i  = 0;
    if (!sp->highbd) {
        const uint8_t* row  = (const uint8_t*)sp->data + (size_t)Y * sp->stride + x0_p;
        const __m256i  c128 = _mm256_set1_epi32(128);
        const __m256i  vvs  = _mm256_set1_epi32((int)vs);
        for (; i + 8 <= n; i += 8) {
            const __m256i b = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(const void*)(row + i)));
            _mm256_storeu_si256((__m256i*)(void*)(bacc + i), _mm256_add_epi32(c128, _mm256_mullo_epi32(vvs, b)));
        }
        for (; i < n; i++) {
            bacc[i] = 128 + vs * row[i];
        }
    } else {
        const uint16_t* row  = (const uint16_t*)sp->data + (size_t)Y * sp->stride + x0_p;
        const __m256i   c128 = _mm256_set1_epi32(128);
        const __m256i   vvs  = _mm256_set1_epi32((int)vs);
        for (; i + 8 <= n; i += 8) {
            const __m256i b = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(const void*)(row + i)));
            _mm256_storeu_si256((__m256i*)(void*)(bacc + i), _mm256_add_epi32(c128, _mm256_mullo_epi32(vvs, b)));
        }
        for (; i < n; i++) {
            bacc[i] = 128 + vs * row[i];
        }
    }
}

void svt_mcd_simd_degrain_add_ref(uint32_t* bacc, const uint16_t* comp, uint32_t w, uint32_t n) {
    const __m256i vw = _mm256_set1_epi32((int)w);
    uint32_t      i  = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256i c = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(const void*)(comp + i)));
        _mm256_storeu_si256(
            (__m256i*)(void*)(bacc + i),
            _mm256_add_epi32(_mm256_loadu_si256((const __m256i*)(const void*)(bacc + i)), _mm256_mullo_epi32(vw, c)));
    }
    for (; i < n; i++) {
        bacc[i] += w * comp[i];
    }
}

void svt_mcd_simd_degrain_shift(uint32_t* bacc, uint32_t n) {
    uint32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_si256((__m256i*)(void*)(bacc + i),
                            _mm256_srai_epi32(_mm256_loadu_si256((const __m256i*)(const void*)(bacc + i)), 8));
    }
    for (; i < n; i++) {
        bacc[i] >>= 8;
    }
}

void svt_mcd_simd_degrain_acc_row(uint32_t* accrow, const uint32_t* brow, const uint32_t* win, uint32_t n) {
    uint32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256i b  = _mm256_loadu_si256((const __m256i*)(const void*)(brow + i));
        const __m256i wv = _mm256_loadu_si256((const __m256i*)(const void*)(win + i));
        _mm256_storeu_si256(
            (__m256i*)(void*)(accrow + i),
            _mm256_add_epi32(_mm256_loadu_si256((const __m256i*)(const void*)(accrow + i)), _mm256_mullo_epi32(b, wv)));
    }
    for (; i < n; i++) {
        accrow[i] += brow[i] * win[i];
    }
}

int svt_mcd_simd_degrain_finalize_row(McdPlane* dp, uint32_t y, const uint32_t* acc, const McdPlane* sp, int32_t lim,
                                      uint32_t pmax) {
    if (lim) {
        return 0; /* limit clamp kept scalar (rare knob) */
    }
    const uint32_t  w  = sp->w;
    const uint32_t* ar = acc + (size_t)y * w;
    if (pmax > 32767) {
        return 0; /* narrowing below assumes the sample range fits i16 */
    }
    uint32_t x = 0;
    if (!dp->highbd) {
        uint8_t*      dr    = (uint8_t*)dp->data + (size_t)y * dp->stride;
        const __m256i c1024 = _mm256_set1_epi32(1024);
        const __m256i vpmax = _mm256_set1_epi32((int)pmax);
        for (; x + 8 <= w; x += 8) {
            __m256i o = _mm256_srai_epi32(
                _mm256_add_epi32(_mm256_loadu_si256((const __m256i*)(const void*)(ar + x)), c1024), 11);
            o = _mm256_min_epu32(o, vpmax); /* values are non-negative */
            /* i32 -> i16 (lane-fixed) -> u8 */
            __m256i pk = _mm256_permute4x64_epi64(_mm256_packs_epi32(o, o), _MM_SHUFFLE(3, 1, 2, 0));
            __m128i b  = _mm_packus_epi16(_mm256_castsi256_si128(pk), _mm256_extracti128_si256(pk, 1));
            _mm_storel_epi64((__m128i*)(void*)(dr + x), b);
        }
        for (; x < w; x++) {
            uint32_t o = (ar[x] + 1024) >> 11;
            if (o > pmax) {
                o = pmax;
            }
            dr[x] = (uint8_t)o;
        }
    } else {
        uint16_t*     dr    = (uint16_t*)dp->data + (size_t)y * dp->stride;
        const __m256i c1024 = _mm256_set1_epi32(1024);
        const __m256i vpmax = _mm256_set1_epi32((int)pmax);
        for (; x + 8 <= w; x += 8) {
            __m256i o = _mm256_srai_epi32(
                _mm256_add_epi32(_mm256_loadu_si256((const __m256i*)(const void*)(ar + x)), c1024), 11);
            o          = _mm256_min_epu32(o, vpmax);
            __m256i pk = _mm256_permute4x64_epi64(_mm256_packs_epi32(o, o), _MM_SHUFFLE(3, 1, 2, 0));
            _mm_storeu_si128((__m128i*)(void*)(dr + x), _mm256_castsi256_si128(pk));
        }
        for (; x < w; x++) {
            uint32_t o = (ar[x] + 1024) >> 11;
            if (o > pmax) {
                o = pmax;
            }
            dr[x] = (uint16_t)o;
        }
    }
    return 1;
}

/* ================================================================== */
/* luma-masked merge: out = src + m * (filt - src), m from the SOURCE  */
/* luma (see svt_mcd_prefilter_mask_merge).                                */
/*                                                                     */
/* Bit-exactness: lane-wise IEEE float ops in the exact scalar         */
/* per-pixel order — y8 = y * (255 * inv_pmax), m = (y8 - 16) / 59,    */
/* clamp [0,1], lerp mul+add, +0.5f, truncating conversion, clamps —   */
/* no FMA (TU compiled with -ffp-contract=off), no reassociation.      */
/* Planes are processed in order 0,1,2 and the chroma box taps are     */
/* summed (((m00+m01)+m10)+m11) exactly like the scalar loop. The      */
/* chroma mask reads the source luma plane handed in by the caller;    */
/* svt_mcd_prefilter_mask_merge snapshots that plane when the output       */
/* aliases the source, so the mask never observes merged luma. Scalar  */
/* tails replicate the scalar formulas verbatim.                       */
/* ================================================================== */

/* One plane, 8 lanes per chunk (AVX2). Same structure as the AVX-512
 * kernel; chroma gathers precomputed luma mask rows (see above). */
static void mcd_merge_mask_row256(const McdPlane* yl, uint32_t ly, float* dst, float kK) {
    const __m256 vK  = _mm256_set1_ps(kK);
    const __m256 v16 = _mm256_set1_ps(16.0f);
    const __m256 v59 = _mm256_set1_ps(59.0f);
    const __m256 v1  = _mm256_set1_ps(1.0f);
    const __m256 v0  = _mm256_setzero_ps();
    uint32_t     x   = 0;
    for (; x + 8 <= yl->w; x += 8) {
        __m256 yv;
        if (yl->highbd) {
            yv = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm_loadu_si128(
                (const __m128i*)(const void*)((const uint16_t*)yl->data + (size_t)ly * yl->stride + x))));
        } else {
            yv = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(
                (const __m128i*)(const void*)((const uint8_t*)yl->data + (size_t)ly * yl->stride + x))));
        }
        _mm256_storeu_ps(
            dst + x,
            _mm256_min_ps(_mm256_max_ps(_mm256_div_ps(_mm256_sub_ps(_mm256_mul_ps(yv, vK), v16), v59), v0), v1));
    }
    for (; x < yl->w; x++) {
        dst[x] = mcd_merge_mask(yl, x, ly, kK);
    }
}

void mcd_merge_plane256(const McdPlane* s, const McdPlane* f, McdPlane* o, const McdPlane* yl, int is_luma,
                        float* mrows, float inv_pmax, uint32_t pmax) {
    const float    kK     = 255.0f * inv_pmax;
    const __m256   vK     = _mm256_set1_ps(kK);
    const __m256   v16    = _mm256_set1_ps(16.0f);
    const __m256   v59    = _mm256_set1_ps(59.0f);
    const __m256   v1     = _mm256_set1_ps(1.0f);
    const __m256   v0     = _mm256_setzero_ps();
    const __m256   vhal   = _mm256_set1_ps(0.5f);
    const __m256i  vpmaxi = _mm256_set1_epi32((int)pmax);
    const __m256i  vzeroi = _mm256_setzero_si256();
    const uint32_t lhm1   = yl->h - 1;

    for (uint32_t y = 0; y < s->h; y++) {
        const uint8_t* srow  = (const uint8_t*)s->data + (size_t)y * s->stride * (s->highbd ? 2 : 1);
        const uint8_t* frow  = (const uint8_t*)f->data + (size_t)y * f->stride * (s->highbd ? 2 : 1);
        uint8_t*       orow  = (uint8_t*)o->data + (size_t)y * o->stride * (s->highbd ? 2 : 1);
        const float *  mrow0 = NULL, *mrow1 = NULL;
        if (!is_luma) {
            const uint32_t r0 = (2 * y < lhm1) ? 2 * y : lhm1;
            const uint32_t r1 = (2 * y + 1 < lhm1) ? 2 * y + 1 : lhm1;
            mcd_merge_mask_row256(yl, r0, mrows, kK);
            mcd_merge_mask_row256(yl, r1, mrows + yl->w, kK);
            mrow0 = mrows;
            mrow1 = mrows + yl->w;
        }
        const uint32_t lwm1 = yl->w - 1;
        uint32_t       x    = 0;
        for (; x + 8 <= s->w; x += 8) {
            __m256 m;
            if (is_luma) {
                __m256 yv;
                if (s->highbd) {
                    yv = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm_loadu_si128(
                        (const __m128i*)(const void*)((const uint16_t*)yl->data + (size_t)y * yl->stride + x))));
                } else {
                    yv = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(
                        (const __m128i*)(const void*)((const uint8_t*)yl->data + (size_t)y * yl->stride + x))));
                }
                m = _mm256_min_ps(_mm256_max_ps(_mm256_div_ps(_mm256_sub_ps(_mm256_mul_ps(yv, vK), v16), v59), v0), v1);
            } else {
                const __m256i twox = _mm256_add_epi32(_mm256_set1_epi32((int)(2 * x)),
                                                      _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14));
                const __m256i xx0  = _mm256_min_epi32(twox, _mm256_set1_epi32((int)lwm1));
                const __m256i xx1  = _mm256_min_epi32(_mm256_add_epi32(twox, _mm256_set1_epi32(1)),
                                                      _mm256_set1_epi32((int)lwm1));
                const __m256  g00  = _mm256_i32gather_ps(mrow0, xx0, 4);
                const __m256  g01  = _mm256_i32gather_ps(mrow0, xx1, 4);
                const __m256  g10  = _mm256_i32gather_ps(mrow1, xx0, 4);
                const __m256  g11  = _mm256_i32gather_ps(mrow1, xx1, 4);
                m                  = _mm256_div_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(g00, g01), g10), g11),
                                                   _mm256_set1_ps(4.0f));
            }
            __m256 sv, fv;
            if (s->highbd) {
                sv = _mm256_cvtepi32_ps(
                    _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(const void*)((const uint16_t*)srow + x))));
                fv = _mm256_cvtepi32_ps(
                    _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(const void*)((const uint16_t*)frow + x))));
            } else {
                sv = _mm256_cvtepi32_ps(
                    _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(const void*)((const uint8_t*)srow + x))));
                fv = _mm256_cvtepi32_ps(
                    _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(const void*)((const uint8_t*)frow + x))));
            }
            const __m256 v  = _mm256_add_ps(sv, _mm256_mul_ps(m, _mm256_sub_ps(fv, sv)));
            __m256i      iv = _mm256_cvttps_epi32(_mm256_add_ps(v, vhal));
            iv              = _mm256_max_epi32(iv, vzeroi);
            iv              = _mm256_min_epi32(iv, vpmaxi);
            if (s->highbd) {
                /* 8 x i32 -> 8 x u16 (values in [0, pmax] fit u16) */
                __m256i pk = _mm256_permute4x64_epi64(_mm256_packs_epi32(iv, iv), _MM_SHUFFLE(3, 1, 2, 0));
                _mm_storeu_si128((__m128i*)(void*)((uint16_t*)orow + x), _mm256_castsi256_si128(pk));
            } else {
                __m256i pk = _mm256_permute4x64_epi64(_mm256_packs_epi32(iv, iv), _MM_SHUFFLE(3, 1, 2, 0));
                __m128i b  = _mm_packus_epi16(_mm256_castsi256_si128(pk), _mm256_extracti128_si256(pk, 1));
                _mm_storel_epi64((__m128i*)(void*)((uint8_t*)orow + x), b);
            }
        }
        for (; x < s->w; x++) {
            /* chroma reaching here is always 4:2:0 (dispatcher gate) */
            const float m = is_luma ? mcd_merge_mask(yl, x, y, kK) : mcd_merge_box(yl, x, y, 1u, 1u, kK);
            mcd_merge_pix(s, f, o, x, y, m, pmax);
        }
    }
}

/* Mask-merge RTCD kernel: validates the call and drives the per-plane
 * vector merge. Returns MCD_ERR_PARAM without writing when unsupported. */
int mcd_avx2_mask_merge(const McdPlane* src_planes, const McdPlane* filt_planes, McdPlane* out_planes,
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
        mcd_avx2_merge_plane256(&src_planes[p], &filt_planes[p], &out_planes[p], yl, p == 0, mrows, inv_pmax, pmax);
    }
    free(mrows);
    return MCD_SUCCESS;
}
