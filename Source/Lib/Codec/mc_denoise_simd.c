/*
 * mc_denoise_simd.c — SIMD kernels for the motion-compensated denoiser.
 *
 * Baseline-compiled translation unit: every SIMD function carries a GCC
 * target attribute, so the file (and the whole library) still builds for a
 * generic x86-64 target and the scalar code in mc_denoise_*.c stays the
 * default compile path and the correctness oracle. Non-x86 builds get
 * unreachable stubs.
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
 * dispatch to the scalar path (see mcd_simd_disabled).
 */
#include <stdlib.h>

#include "mc_denoise.h"
#include "mc_denoise_simd.h"

/* ------------------------------------------------------------------ */
/* forced-scalar test hook                                             */

static int g_mcd_no_simd = -1;

int mcd_simd_disabled(void) {
    /* plain read/write: idempotent env-derived value, benign race
     * (any racing thread computes the same result) */
    int v = g_mcd_no_simd;
    if (v < 0) {
        v = getenv("MCD_NO_SIMD") != NULL;
        g_mcd_no_simd = v;
    }
    return v;
}

/* ------------------------------------------------------------------ */
/* non-x86 stubs (never called: the dispatch macros check the arch)     */

#if !((defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__)))

uint64_t mcd_simd_sad_u8u8(const uint8_t *a, int32_t sa, const uint8_t *b, int32_t sb,
                           uint32_t w, uint32_t h) { (void)a;(void)sa;(void)b;(void)sb;(void)w;(void)h; return 0; }
uint64_t mcd_simd_sad_u16u16(const uint16_t *a, int32_t sa, const uint16_t *b, int32_t sb,
                             uint32_t w, uint32_t h) { (void)a;(void)sa;(void)b;(void)sb;(void)w;(void)h; return 0; }
uint64_t mcd_simd_sad_u8u16(const uint8_t *a, int32_t sa, const uint16_t *b, int32_t sb,
                            uint32_t w, uint32_t h) { (void)a;(void)sa;(void)b;(void)sb;(void)w;(void)h; return 0; }
uint64_t mcd_simd_sad_u16u8(const uint16_t *a, int32_t sa, const uint8_t *b, int32_t sb,
                            uint32_t w, uint32_t h) { (void)a;(void)sa;(void)b;(void)sb;(void)w;(void)h; return 0; }
uint64_t mcd_simd_satd_u8u8(const uint8_t *a, int32_t sa, const uint8_t *b, int32_t sb,
                            uint32_t w, uint32_t h) { (void)a;(void)sa;(void)b;(void)sb;(void)w;(void)h; return 0; }
uint64_t mcd_simd_satd_u16u16(const uint16_t *a, int32_t sa, const uint16_t *b, int32_t sb,
                              uint32_t w, uint32_t h) { (void)a;(void)sa;(void)b;(void)sb;(void)w;(void)h; return 0; }
void mcd_simd_halfpel_h_u8(const uint8_t *s, uint32_t ss, uint32_t w, uint32_t h, uint32_t p,
                           uint8_t *d, uint32_t ds, uint32_t m)
    { (void)s;(void)ss;(void)w;(void)h;(void)p;(void)d;(void)ds;(void)m; }
void mcd_simd_halfpel_v_u8(const uint8_t *s, uint32_t ss, uint32_t w, uint32_t h, uint32_t p,
                           uint8_t *d, uint32_t ds, uint32_t m)
    { (void)s;(void)ss;(void)w;(void)h;(void)p;(void)d;(void)ds;(void)m; }
void mcd_simd_halfpel_h_u16(const uint16_t *s, uint32_t ss, uint32_t w, uint32_t h, uint32_t p,
                            uint16_t *d, uint32_t ds, uint32_t m)
    { (void)s;(void)ss;(void)w;(void)h;(void)p;(void)d;(void)ds;(void)m; }
void mcd_simd_halfpel_v_u16(const uint16_t *s, uint32_t ss, uint32_t w, uint32_t h, uint32_t p,
                            uint16_t *d, uint32_t ds, uint32_t m)
    { (void)s;(void)ss;(void)w;(void)h;(void)p;(void)d;(void)ds;(void)m; }
void mcd_simd_fetch_u8_row(const uint8_t *row, uint16_t *dst, uint32_t w)
    { (void)row;(void)dst;(void)w; }
int mcd_simd_dfttest_plane_with_ctx(const struct McdDftCtx *ctx, const McdPlane *src,
                                    McdPlane *dst)
    { (void)ctx;(void)src;(void)dst; return 1; }

#else /* x86-64 */

#include <immintrin.h>

/* ================================================================== */
/* SAD: sum of absolute differences, uint64 accumulation.              */
/* Exactness: |a-b| per sample summed as integers (order irrelevant).  */
/* Block sizes in this module are powers of two, 2..32; the kernels    */
/* handle any w/h with 16/8-wide vectors plus a scalar tail.           */
/* ================================================================== */

__attribute__((target("avx2")))
uint64_t mcd_simd_sad_u8u8(const uint8_t *a, int32_t sa, const uint8_t *b, int32_t sb,
                           uint32_t w, uint32_t h) {
    __m128i vacc = _mm_setzero_si128(); /* 2 x u64 partial sums */
    uint64_t tail = 0;
    uint32_t y;
    for (y = 0; y < h; y++) {
        uint32_t x = 0;
        for (; x + 16 <= w; x += 16) {
            const __m128i va = _mm_loadu_si128((const __m128i *)(const void *)(a + x));
            const __m128i vb = _mm_loadu_si128((const __m128i *)(const void *)(b + x));
            vacc = _mm_add_epi64(vacc, _mm_sad_epu8(va, vb));
        }
        if (x + 8 <= w) {
            const __m128i va = _mm_loadl_epi64((const __m128i *)(const void *)(a + x));
            const __m128i vb = _mm_loadl_epi64((const __m128i *)(const void *)(b + x));
            tail += (uint64_t)_mm_cvtsi128_si64(_mm_sad_epu8(va, vb));
            x += 8;
        }
        for (; x < w; x++) {
            const int d = (int)a[x] - (int)b[x];
            tail += (uint64_t)(d < 0 ? -d : d);
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

__attribute__((target("avx2")))
static inline uint64_t mcd_sad_u16_reduce(const __m256i vacc, uint64_t tail) {
    __m128i v = _mm_add_epi32(_mm256_castsi256_si128(vacc), _mm256_extracti128_si256(vacc, 1));
    v = _mm_add_epi32(v, _mm_shuffle_epi32(v, _MM_SHUFFLE(1, 0, 3, 2)));
    v = _mm_add_epi32(v, _mm_shuffle_epi32(v, _MM_SHUFFLE(2, 3, 0, 1)));
    return tail + (uint64_t)(uint32_t)_mm_cvtsi128_si32(v);
}

__attribute__((target("avx2")))
uint64_t mcd_simd_sad_u16u16(const uint16_t *a, int32_t sa, const uint16_t *b, int32_t sb,
                             uint32_t w, uint32_t h) {
    if ((uint64_t)w * h * 65535ull >= 0x7fffffffull)
        return 0; /* unreachable for module block sizes */
    const __m256i ones = _mm256_set1_epi16(1);
    __m256i vacc = _mm256_setzero_si256(); /* 8 x u32 */
    uint64_t tail = 0;
    for (uint32_t y = 0; y < h; y++) {
        uint32_t x = 0;
        for (; x + 16 <= w; x += 16) {
            const __m256i va = _mm256_loadu_si256((const __m256i *)(const void *)(a + x));
            const __m256i vb = _mm256_loadu_si256((const __m256i *)(const void *)(b + x));
            __m256i d = _mm256_abs_epi16(_mm256_sub_epi16(va, vb));
            vacc = _mm256_add_epi32(vacc, _mm256_madd_epi16(d, ones));
        }
        if (x + 8 <= w) {
            const __m256i va = _mm256_inserti128_si256(
                _mm256_setzero_si256(), _mm_loadu_si128((const __m128i *)(const void *)(a + x)), 0);
            const __m256i vb = _mm256_inserti128_si256(
                _mm256_setzero_si256(), _mm_loadu_si128((const __m128i *)(const void *)(b + x)), 0);
            __m256i d = _mm256_abs_epi16(_mm256_sub_epi16(va, vb));
            vacc = _mm256_add_epi32(vacc, _mm256_madd_epi16(d, ones));
            x += 8;
        }
        for (; x < w; x++) {
            const int d = (int)a[x] - (int)b[x];
            tail += (uint64_t)(d < 0 ? -d : d);
        }
        a += sa;
        b += sb;
    }
    return mcd_sad_u16_reduce(vacc, tail);
}

__attribute__((target("avx2")))
uint64_t mcd_simd_sad_u8u16(const uint8_t *a, int32_t sa, const uint16_t *b, int32_t sb,
                            uint32_t w, uint32_t h) {
    if ((uint64_t)w * h * 65535ull >= 0x7fffffffull)
        return 0;
    const __m256i ones = _mm256_set1_epi16(1);
    __m256i vacc = _mm256_setzero_si256();
    uint64_t tail = 0;
    for (uint32_t y = 0; y < h; y++) {
        uint32_t x = 0;
        for (; x + 8 <= w; x += 8) {
            const __m256i va =
                _mm256_cvtepu8_epi16(_mm_loadl_epi64((const __m128i *)(const void *)(a + x)));
            const __m256i vb = _mm256_inserti128_si256(
                _mm256_setzero_si256(), _mm_loadu_si128((const __m128i *)(const void *)(b + x)), 0);
            __m256i d = _mm256_abs_epi16(_mm256_sub_epi16(va, vb));
            vacc = _mm256_add_epi32(vacc, _mm256_madd_epi16(d, ones));
        }
        for (; x < w; x++) {
            const int d = (int)a[x] - (int)b[x];
            tail += (uint64_t)(d < 0 ? -d : d);
        }
        a += sa;
        b += sb;
    }
    return mcd_sad_u16_reduce(vacc, tail);
}

__attribute__((target("avx2")))
uint64_t mcd_simd_sad_u16u8(const uint16_t *a, int32_t sa, const uint8_t *b, int32_t sb,
                            uint32_t w, uint32_t h) {
    if ((uint64_t)w * h * 65535ull >= 0x7fffffffull)
        return 0;
    const __m256i ones = _mm256_set1_epi16(1);
    __m256i vacc = _mm256_setzero_si256();
    uint64_t tail = 0;
    for (uint32_t y = 0; y < h; y++) {
        uint32_t x = 0;
        for (; x + 8 <= w; x += 8) {
            const __m256i va = _mm256_inserti128_si256(
                _mm256_setzero_si256(), _mm_loadu_si128((const __m128i *)(const void *)(a + x)), 0);
            const __m256i vb =
                _mm256_cvtepu8_epi16(_mm_loadl_epi64((const __m128i *)(const void *)(b + x)));
            __m256i d = _mm256_abs_epi16(_mm256_sub_epi16(va, vb));
            vacc = _mm256_add_epi32(vacc, _mm256_madd_epi16(d, ones));
        }
        for (; x < w; x++) {
            const int d = (int)a[x] - (int)b[x];
            tail += (uint64_t)(d < 0 ? -d : d);
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

__attribute__((target("avx512bw,avx512vl")))
uint64_t mcd_simd_satd_u8u8(const uint8_t *a, int32_t sa, const uint8_t *b, int32_t sb,
                            uint32_t w, uint32_t h) {
    if (w < 16 || h < 16)
        return 0; /* caller falls back to scalar (unreachable at level 0) */
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
                    _mm256_sub_epi16(
                        _mm256_cvtepu8_epi16(_mm_loadu_si128(
                            (const __m128i *)(const void *)(a + (by + 4 * 0 + r) * sa + bx))),
                        _mm256_cvtepu8_epi16(_mm_loadu_si128(
                            (const __m128i *)(const void *)(b + (by + 4 * 0 + r) * sb + bx)))),
                    _mm256_sub_epi16(
                        _mm256_cvtepu8_epi16(_mm_loadu_si128(
                            (const __m128i *)(const void *)(a + (by + 4 * 1 + r) * sa + bx))),
                        _mm256_cvtepu8_epi16(_mm_loadu_si128(
                            (const __m128i *)(const void *)(b + (by + 4 * 1 + r) * sb + bx)))),
                    _mm256_sub_epi16(
                        _mm256_cvtepu8_epi16(_mm_loadu_si128(
                            (const __m128i *)(const void *)(a + (by + 4 * 2 + r) * sa + bx))),
                        _mm256_cvtepu8_epi16(_mm_loadu_si128(
                            (const __m128i *)(const void *)(b + (by + 4 * 2 + r) * sb + bx)))),
                    _mm256_sub_epi16(
                        _mm256_cvtepu8_epi16(_mm_loadu_si128(
                            (const __m128i *)(const void *)(a + (by + 4 * 3 + r) * sa + bx))),
                        _mm256_cvtepu8_epi16(_mm_loadu_si128(
                            (const __m128i *)(const void *)(b + (by + 4 * 3 + r) * sb + bx)))),
                };
                for (uint32_t c = 0; c < 4; c++) {
                    /* permutex output lanes 4g..4g+3 repeat group 0's values;
                     * take the low 64 bits of each group's permute */
                    const __m256i v = _mm256_setr_epi64x(
                        (long long)_mm_cvtsi128_si64(
                            _mm256_castsi256_si128(_mm256_permutexvar_epi16(vidx[c], df[0]))),
                        (long long)_mm_cvtsi128_si64(
                            _mm256_castsi256_si128(_mm256_permutexvar_epi16(vidx[c], df[1]))),
                        (long long)_mm_cvtsi128_si64(
                            _mm256_castsi256_si128(_mm256_permutexvar_epi16(vidx[c], df[2]))),
                        (long long)_mm_cvtsi128_si64(
                            _mm256_castsi256_si128(_mm256_permutexvar_epi16(vidx[c], df[3]))));
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
                t[r][0] = _mm256_add_epi16(a0, a1);
                t[r][1] = _mm256_add_epi16(a2, a3);
                t[r][2] = _mm256_sub_epi16(a0, a1);
                t[r][3] = _mm256_sub_epi16(a3, a2);
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
                acc = _mm256_add_epi16(acc, _mm256_abs_epi16(s0));
                acc = _mm256_add_epi16(acc, _mm256_abs_epi16(s1));
                acc = _mm256_add_epi16(acc, _mm256_abs_epi16(s2));
                acc = _mm256_add_epi16(acc, _mm256_abs_epi16(s3));
            }
            const __m512i wide = _mm512_cvtepu16_epi32(acc);
            total += _mm512_reduce_add_epi32(wide);
        }
    }
    return total;
}

__attribute__((target("avx512f,avx512bw,avx512vl")))
uint64_t mcd_simd_satd_u16u16(const uint16_t *a, int32_t sa, const uint16_t *b, int32_t sb,
                              uint32_t w, uint32_t h) {
    if (w < 16 || h < 16)
        return 0;
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
                    _mm512_sub_epi32(
                        _mm512_cvtepu16_epi32(_mm256_loadu_si256(
                            (const __m256i *)(const void *)(a + (by + 4 * 0 + r) * sa + bx))),
                        _mm512_cvtepu16_epi32(_mm256_loadu_si256(
                            (const __m256i *)(const void *)(b + (by + 4 * 0 + r) * sb + bx)))),
                    _mm512_sub_epi32(
                        _mm512_cvtepu16_epi32(_mm256_loadu_si256(
                            (const __m256i *)(const void *)(a + (by + 4 * 1 + r) * sa + bx))),
                        _mm512_cvtepu16_epi32(_mm256_loadu_si256(
                            (const __m256i *)(const void *)(b + (by + 4 * 1 + r) * sb + bx)))),
                    _mm512_sub_epi32(
                        _mm512_cvtepu16_epi32(_mm256_loadu_si256(
                            (const __m256i *)(const void *)(a + (by + 4 * 2 + r) * sa + bx))),
                        _mm512_cvtepu16_epi32(_mm256_loadu_si256(
                            (const __m256i *)(const void *)(b + (by + 4 * 2 + r) * sb + bx)))),
                    _mm512_sub_epi32(
                        _mm512_cvtepu16_epi32(_mm256_loadu_si256(
                            (const __m256i *)(const void *)(a + (by + 4 * 3 + r) * sa + bx))),
                        _mm512_cvtepu16_epi32(_mm256_loadu_si256(
                            (const __m256i *)(const void *)(b + (by + 4 * 3 + r) * sb + bx)))),
                };
                for (uint32_t c = 0; c < 4; c++) {
                    /* low 128 bits of each group's permute hold the 4
                     * sub-block values of that group */
                    const __m128i q0 = _mm512_castsi512_si128(
                        _mm512_permutexvar_epi32(vidx[c], df[0]));
                    const __m128i q1 = _mm512_castsi512_si128(
                        _mm512_permutexvar_epi32(vidx[c], df[1]));
                    const __m128i q2 = _mm512_castsi512_si128(
                        _mm512_permutexvar_epi32(vidx[c], df[2]));
                    const __m128i q3 = _mm512_castsi512_si128(
                        _mm512_permutexvar_epi32(vidx[c], df[3]));
                    const __m512i v = _mm512_inserti64x4(
                        _mm512_castsi256_si512(_mm256_set_m128i(q1, q0)),
                        _mm256_set_m128i(q3, q2), 1);
                    d[r][c] = v;
                }
            }
            __m512i t[4][4];
            for (uint32_t r = 0; r < 4; r++) {
                const __m512i a0 = _mm512_add_epi32(d[r][0], d[r][3]);
                const __m512i a1 = _mm512_add_epi32(d[r][1], d[r][2]);
                const __m512i a2 = _mm512_sub_epi32(d[r][1], d[r][2]);
                const __m512i a3 = _mm512_sub_epi32(d[r][0], d[r][3]);
                t[r][0] = _mm512_add_epi32(a0, a1);
                t[r][1] = _mm512_add_epi32(a2, a3);
                t[r][2] = _mm512_sub_epi32(a0, a1);
                t[r][3] = _mm512_sub_epi32(a3, a2);
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
                acc = _mm512_add_epi32(acc, _mm512_abs_epi32(s0));
                acc = _mm512_add_epi32(acc, _mm512_abs_epi32(s1));
                acc = _mm512_add_epi32(acc, _mm512_abs_epi32(s2));
                acc = _mm512_add_epi32(acc, _mm512_abs_epi32(s3));
            }
            total += _mm512_reduce_add_epi32(acc);
        }
    }
    return total;
}


/* Symmetric mirror reflection of i into [0, w) (same as the scalar
 * mcd_mirror in mc_denoise_dfttest.c). */
static int32_t mcd_simd_mirror(int32_t i, uint32_t w)
{
    if (w == 1)
        return 0;
    while (i < 0 || i >= (int32_t)w) {
        if (i < 0)
            i = -i - 1;
        else
            i = 2 * (int32_t)w - 1 - i;
    }
    return i;
}

/* ================================================================== */
/* pelset: half-pel phase plane filters (6-tap Wiener 1,-5,20,20,-5,1) */
/* and the u8->u16 fetch row copy. Bit-exact: integer lane arithmetic  */
/* with the same tap math; u8/u16 saturating pack == the scalar clamp. */
/*                                                                     */
/* H pass: the plane margins are exact edge replicas, so unclamped     */
/* taps equal the scalar's clamped taps for every in-memory position;  */
/* the vector run stops where the 6-tap window would leave the padded  */
/* row, and a scalar clamped tail covers the last columns.             */
/* V pass: taps are rows (clamped pointer selection per output row),   */
/* X needs no clamp at all - fully vectorized.                         */
/* ================================================================== */

static inline int32_t mcd_simd_clampi(int32_t v, int32_t lo, int32_t hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__attribute__((target("avx2")))
void mcd_simd_fetch_u8_row(const uint8_t *row, uint16_t *dst, uint32_t w) {
    uint32_t x = 0;
    for (; x + 16 <= w; x += 16)
        _mm256_storeu_si256((__m256i *)(void *)(dst + x),
                            _mm256_cvtepu8_epi16(_mm_loadu_si128(
                                (const __m128i *)(const void *)(row + x))));
    for (; x < w; x++)
        dst[x] = (uint16_t)row[x];
}

__attribute__((target("avx2")))
static inline void mcd_halfpel_h_row_u8(const uint8_t *srow, uint32_t X,
                                        uint8_t *drow) {
    const __m256i s0 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(const void *)(srow + X - 2)));
    const __m256i s1 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(const void *)(srow + X - 1)));
    const __m256i s2 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(const void *)(srow + X)));
    const __m256i s3 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(const void *)(srow + X + 1)));
    const __m256i s4 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(const void *)(srow + X + 2)));
    const __m256i s5 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(const void *)(srow + X + 3)));
    const __m256i t = _mm256_add_epi16(
        _mm256_add_epi16(_mm256_add_epi16(s0, s5),
                         _mm256_mullo_epi16(_mm256_add_epi16(s2, s3), _mm256_set1_epi16(20))),
        _mm256_mullo_epi16(_mm256_add_epi16(s1, s4), _mm256_set1_epi16(-5)));
    const __m256i v = _mm256_srai_epi16(_mm256_add_epi16(t, _mm256_set1_epi16(16)), 5);
    _mm_storeu_si128((__m128i *)(void *)(drow + X),
                     _mm_packus_epi16(_mm256_castsi256_si128(v),
                                      _mm256_extracti128_si256(v, 1)));
}

__attribute__((target("avx2")))
void mcd_simd_halfpel_h_u8(const uint8_t *src, uint32_t sstride, uint32_t w, uint32_t h,
                           uint32_t pad, uint8_t *dst, uint32_t dstride, uint32_t maxval) {
    const uint32_t tw = w + 2 * pad;
    const int32_t wm1 = (int32_t)w - 1;
    (void)maxval;
    for (uint32_t Y = 0; Y < h + 2 * pad; Y++) {
        /* row base WITHOUT the +pad offset: core column c sits at
         * srow[pad + c]; the vector kernel's taps srow[X-2 .. X+18] for
         * output X therefore evaluate core positions x = X - pad, and
         * taps past a core edge land in the replicated margin, which is
         * exactly the scalar clamp. Only the final 18 columns need a
         * scalar (memory-bounded) tail. */
        const uint8_t *srow = src + (size_t)Y * sstride;
        const uint8_t *vrow = srow + pad; /* visible origin (scalar path) */
        uint8_t *drow = dst + (size_t)Y * dstride;
        uint32_t X = 0;
        for (; X < pad + 2 && X < tw; X++) {   /* taps clamp into the core */
            const int32_t x = (int32_t)X - (int32_t)pad;
            const int32_t v = (vrow[mcd_simd_clampi(x - 2, 0, wm1)] -
                               5 * vrow[mcd_simd_clampi(x - 1, 0, wm1)] +
                              20 * vrow[mcd_simd_clampi(x, 0, wm1)] +
                              20 * vrow[mcd_simd_clampi(x + 1, 0, wm1)] -
                               5 * vrow[mcd_simd_clampi(x + 2, 0, wm1)] +
                                   vrow[mcd_simd_clampi(x + 3, 0, wm1)] + 16) >> 5;
            drow[X] = (uint8_t)(v < 0 ? 0 : (v > (int32_t)maxval ? (int32_t)maxval : v));
        }
        for (X = pad + 2; X + 19 <= tw; X += 16)
            mcd_halfpel_h_row_u8(srow, X, drow);
        for (; X < tw; X++) {
            const int32_t x = (int32_t)X - (int32_t)pad;
            const int32_t v = (vrow[mcd_simd_clampi(x - 2, 0, wm1)] -
                               5 * vrow[mcd_simd_clampi(x - 1, 0, wm1)] +
                              20 * vrow[mcd_simd_clampi(x, 0, wm1)] +
                              20 * vrow[mcd_simd_clampi(x + 1, 0, wm1)] -
                               5 * vrow[mcd_simd_clampi(x + 2, 0, wm1)] +
                                   vrow[mcd_simd_clampi(x + 3, 0, wm1)] + 16) >> 5;
            drow[X] = (uint8_t)(v < 0 ? 0 : (v > (int32_t)maxval ? (int32_t)maxval : v));
        }
    }
}

__attribute__((target("avx2")))
void mcd_simd_halfpel_v_u8(const uint8_t *src, uint32_t sstride, uint32_t w, uint32_t h,
                           uint32_t pad, uint8_t *dst, uint32_t dstride, uint32_t maxval) {
    const uint32_t tw = w + 2 * pad;
    const int32_t hm1 = (int32_t)h - 1;
    (void)maxval;
    for (uint32_t Y = 0; Y < h + 2 * pad; Y++) {
        const int32_t y = (int32_t)Y - (int32_t)pad;
        /* clamped CORE row indices offset by pad into the padded plane;
         * X below indexes padded columns directly */
        const uint8_t *r0 = src + (size_t)(mcd_simd_clampi(y - 2, 0, hm1) + (int32_t)pad) * sstride;
        const uint8_t *r1 = src + (size_t)(mcd_simd_clampi(y - 1, 0, hm1) + (int32_t)pad) * sstride;
        const uint8_t *r2 = src + (size_t)(mcd_simd_clampi(y, 0, hm1) + (int32_t)pad) * sstride;
        const uint8_t *r3 = src + (size_t)(mcd_simd_clampi(y + 1, 0, hm1) + (int32_t)pad) * sstride;
        const uint8_t *r4 = src + (size_t)(mcd_simd_clampi(y + 2, 0, hm1) + (int32_t)pad) * sstride;
        const uint8_t *r5 = src + (size_t)(mcd_simd_clampi(y + 3, 0, hm1) + (int32_t)pad) * sstride;
        uint8_t *drow = dst + (size_t)Y * dstride;
        uint32_t X = 0;
        for (; X + 16 <= tw; X += 16) {
            const __m256i s0 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(const void *)(r0 + X)));
            const __m256i s1 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(const void *)(r1 + X)));
            const __m256i s2 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(const void *)(r2 + X)));
            const __m256i s3 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(const void *)(r3 + X)));
            const __m256i s4 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(const void *)(r4 + X)));
            const __m256i s5 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(const void *)(r5 + X)));
            const __m256i t = _mm256_add_epi16(
                _mm256_add_epi16(_mm256_add_epi16(s0, s5),
                                 _mm256_mullo_epi16(_mm256_add_epi16(s2, s3), _mm256_set1_epi16(20))),
                _mm256_mullo_epi16(_mm256_add_epi16(s1, s4), _mm256_set1_epi16(-5)));
            const __m256i v = _mm256_srai_epi16(_mm256_add_epi16(t, _mm256_set1_epi16(16)), 5);
            _mm_storeu_si128((__m128i *)(void *)(drow + X),
                             _mm_packus_epi16(_mm256_castsi256_si128(v),
                                      _mm256_extracti128_si256(v, 1)));
        }
        for (; X < tw; X++) {
            const int32_t v = (r0[X] - 5 * r1[X] + 20 * r2[X] + 20 * r3[X] -
                               5 * r4[X] + r5[X] + 16) >> 5;
            drow[X] = (uint8_t)(v < 0 ? 0 : (v > (int32_t)maxval ? (int32_t)maxval : v));
        }
    }
}

/* 16-bit passes: i32 math (52 * 1023 exceeds i16), 16 outputs per zmm.
 * Saturation to [0, maxval] with maxval <= 0xFFFF, then a truncating
 * narrowing store is exact. */
__attribute__((target("avx512bw,avx512vl")))
static inline __m512i mcd_halfpel_load16(const uint16_t *p, uint32_t X) {
    return _mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)(const void *)(p + X)));
}

__attribute__((target("avx512bw,avx512vl")))
void mcd_simd_halfpel_h_u16(const uint16_t *src, uint32_t sstride, uint32_t w, uint32_t h,
                            uint32_t pad, uint16_t *dst, uint32_t dstride, uint32_t maxval) {
    const uint32_t tw = w + 2 * pad;
    const int32_t wm1 = (int32_t)w - 1;
    const __m512i zero = _mm512_setzero_si512();
    const __m512i vmax = _mm512_set1_epi32((int)maxval);
    for (uint32_t Y = 0; Y < h + 2 * pad; Y++) {
        /* see the 8-bit H pass for the row-base / tap-offset scheme */
        const uint16_t *srow = src + (size_t)Y * sstride;
        const uint16_t *vrow = srow + pad; /* visible origin (scalar path) */
        uint16_t *drow = dst + (size_t)Y * dstride;
        uint32_t X = 0;
        for (; X < pad + 2 && X < tw; X++) {   /* taps clamp into the core */
            const int32_t x = (int32_t)X - (int32_t)pad;
            const int32_t v = (vrow[mcd_simd_clampi(x - 2, 0, wm1)] -
                               5 * vrow[mcd_simd_clampi(x - 1, 0, wm1)] +
                              20 * vrow[mcd_simd_clampi(x, 0, wm1)] +
                              20 * vrow[mcd_simd_clampi(x + 1, 0, wm1)] -
                               5 * vrow[mcd_simd_clampi(x + 2, 0, wm1)] +
                                   vrow[mcd_simd_clampi(x + 3, 0, wm1)] + 16) >> 5;
            drow[X] = (uint16_t)(v < 0 ? 0 : (v > (int32_t)maxval ? (int32_t)maxval : v));
        }
        for (X = pad + 2; X + 19 <= tw; X += 16) {
            const __m512i s0 = mcd_halfpel_load16(srow, X - 2);
            const __m512i s1 = mcd_halfpel_load16(srow, X - 1);
            const __m512i s2 = mcd_halfpel_load16(srow, X);
            const __m512i s3 = mcd_halfpel_load16(srow, X + 1);
            const __m512i s4 = mcd_halfpel_load16(srow, X + 2);
            const __m512i s5 = mcd_halfpel_load16(srow, X + 3);
            const __m512i t = _mm512_add_epi32(
                _mm512_add_epi32(_mm512_add_epi32(s0, s5),
                                 _mm512_mullo_epi32(_mm512_add_epi32(s2, s3), _mm512_set1_epi32(20))),
                _mm512_mullo_epi32(_mm512_add_epi32(s1, s4), _mm512_set1_epi32(-5)));
            const __m512i v = _mm512_srai_epi32(_mm512_add_epi32(t, _mm512_set1_epi32(16)), 5);
            _mm256_storeu_si256((__m256i *)(void *)(drow + X),
                                /* signed max handles negative Wiener
                                 * overshoot; value is then <= 0x7FFF so the
                                 * unsigned min against vmax stays correct */
                                _mm512_cvtusepi32_epi16(_mm512_min_epu32(
                                    _mm512_max_epi32(v, zero), vmax)));
        }
        for (; X < tw; X++) {
            const int32_t x = (int32_t)X - (int32_t)pad;
            const int32_t v = (vrow[mcd_simd_clampi(x - 2, 0, wm1)] -
                               5 * vrow[mcd_simd_clampi(x - 1, 0, wm1)] +
                              20 * vrow[mcd_simd_clampi(x, 0, wm1)] +
                              20 * vrow[mcd_simd_clampi(x + 1, 0, wm1)] -
                               5 * vrow[mcd_simd_clampi(x + 2, 0, wm1)] +
                                   vrow[mcd_simd_clampi(x + 3, 0, wm1)] + 16) >> 5;
            drow[X] = (uint16_t)(v < 0 ? 0 : (v > (int32_t)maxval ? (int32_t)maxval : v));
        }
    }
}

__attribute__((target("avx512bw,avx512vl")))
void mcd_simd_halfpel_v_u16(const uint16_t *src, uint32_t sstride, uint32_t w, uint32_t h,
                            uint32_t pad, uint16_t *dst, uint32_t dstride, uint32_t maxval) {
    const uint32_t tw = w + 2 * pad;
    const int32_t hm1 = (int32_t)h - 1;
    const __m512i zero = _mm512_setzero_si512();
    const __m512i vmax = _mm512_set1_epi32((int)maxval);
    for (uint32_t Y = 0; Y < h + 2 * pad; Y++) {
        const int32_t y = (int32_t)Y - (int32_t)pad;
        const uint16_t *r0 = src + (size_t)(mcd_simd_clampi(y - 2, 0, hm1) + (int32_t)pad) * sstride;
        const uint16_t *r1 = src + (size_t)(mcd_simd_clampi(y - 1, 0, hm1) + (int32_t)pad) * sstride;
        const uint16_t *r2 = src + (size_t)(mcd_simd_clampi(y, 0, hm1) + (int32_t)pad) * sstride;
        const uint16_t *r3 = src + (size_t)(mcd_simd_clampi(y + 1, 0, hm1) + (int32_t)pad) * sstride;
        const uint16_t *r4 = src + (size_t)(mcd_simd_clampi(y + 2, 0, hm1) + (int32_t)pad) * sstride;
        const uint16_t *r5 = src + (size_t)(mcd_simd_clampi(y + 3, 0, hm1) + (int32_t)pad) * sstride;
        uint16_t *drow = dst + (size_t)Y * dstride;
        uint32_t X = 0;
        for (; X + 16 <= tw; X += 16) {
            const __m512i s0 = mcd_halfpel_load16(r0, X);
            const __m512i s1 = mcd_halfpel_load16(r1, X);
            const __m512i s2 = mcd_halfpel_load16(r2, X);
            const __m512i s3 = mcd_halfpel_load16(r3, X);
            const __m512i s4 = mcd_halfpel_load16(r4, X);
            const __m512i s5 = mcd_halfpel_load16(r5, X);
            const __m512i t = _mm512_add_epi32(
                _mm512_add_epi32(_mm512_add_epi32(s0, s5),
                                 _mm512_mullo_epi32(_mm512_add_epi32(s2, s3), _mm512_set1_epi32(20))),
                _mm512_mullo_epi32(_mm512_add_epi32(s1, s4), _mm512_set1_epi32(-5)));
            const __m512i v = _mm512_srai_epi32(_mm512_add_epi32(t, _mm512_set1_epi32(16)), 5);
            _mm256_storeu_si256((__m256i *)(void *)(drow + X),
                                /* signed max handles negative Wiener
                                 * overshoot; value is then <= 0x7FFF so the
                                 * unsigned min against vmax stays correct */
                                _mm512_cvtusepi32_epi16(_mm512_min_epu32(
                                    _mm512_max_epi32(v, zero), vmax)));
        }
        for (; X < tw; X++) {
            const int32_t v = (r0[X] - 5 * r1[X] + 20 * r2[X] + 20 * r3[X] -
                               5 * r4[X] + r5[X] + 16) >> 5;
            drow[X] = (uint16_t)(v < 0 ? 0 : (v > (int32_t)maxval ? (int32_t)maxval : v));
        }
    }
}

/* ================================================================== */
/* degrain: element-wise loops. Integer lane ops, bit-exact.           */
/* ================================================================== */

__attribute__((target("avx2")))
void mcd_simd_degrain_init_row(uint32_t *bacc, const McdPlane *sp, uint32_t Y,
                               uint32_t x0_p, uint32_t wsrc, uint32_t n) {
    const uint32_t vs = wsrc;
    uint32_t i = 0;
    if (!sp->highbd) {
        const uint8_t *row = (const uint8_t *)sp->data + (size_t)Y * sp->stride + x0_p;
        const __m256i c128 = _mm256_set1_epi32(128);
        const __m256i vvs = _mm256_set1_epi32((int)vs);
        for (; i + 8 <= n; i += 8) {
            const __m256i b = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(const void *)(row + i)));
            _mm256_storeu_si256((__m256i *)(void *)(bacc + i),
                                _mm256_add_epi32(c128, _mm256_mullo_epi32(vvs, b)));
        }
        for (; i < n; i++)
            bacc[i] = 128 + vs * row[i];
    } else {
        const uint16_t *row = (const uint16_t *)sp->data + (size_t)Y * sp->stride + x0_p;
        const __m256i c128 = _mm256_set1_epi32(128);
        const __m256i vvs = _mm256_set1_epi32((int)vs);
        for (; i + 8 <= n; i += 8) {
            const __m256i b = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(const void *)(row + i)));
            _mm256_storeu_si256((__m256i *)(void *)(bacc + i),
                                _mm256_add_epi32(c128, _mm256_mullo_epi32(vvs, b)));
        }
        for (; i < n; i++)
            bacc[i] = 128 + vs * row[i];
    }
}

__attribute__((target("avx2")))
void mcd_simd_degrain_add_ref(uint32_t *bacc, const uint16_t *comp, uint32_t w, uint32_t n) {
    (void)w;
    const __m256i vw = _mm256_set1_epi32((int)w);
    uint32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256i c = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(const void *)(comp + i)));
        _mm256_storeu_si256((__m256i *)(void *)(bacc + i),
                            _mm256_add_epi32(_mm256_loadu_si256((const __m256i *)(const void *)(bacc + i)),
                                             _mm256_mullo_epi32(vw, c)));
    }
    for (; i < n; i++)
        bacc[i] += w * comp[i];
}

__attribute__((target("avx2")))
void mcd_simd_degrain_shift(uint32_t *bacc, uint32_t n) {
    uint32_t i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_si256((__m256i *)(void *)(bacc + i),
                            _mm256_srai_epi32(_mm256_loadu_si256((const __m256i *)(const void *)(bacc + i)), 8));
    for (; i < n; i++)
        bacc[i] >>= 8;
}

__attribute__((target("avx2")))
void mcd_simd_degrain_acc_row(uint32_t *accrow, const uint32_t *brow,
                              const uint32_t *win, uint32_t n) {
    uint32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256i b = _mm256_loadu_si256((const __m256i *)(const void *)(brow + i));
        const __m256i wv = _mm256_loadu_si256((const __m256i *)(const void *)(win + i));
        _mm256_storeu_si256((__m256i *)(void *)(accrow + i),
                            _mm256_add_epi32(_mm256_loadu_si256((const __m256i *)(const void *)(accrow + i)),
                                             _mm256_mullo_epi32(b, wv)));
    }
    for (; i < n; i++)
        accrow[i] += brow[i] * win[i];
}

__attribute__((target("avx2")))
int mcd_simd_degrain_finalize_row(McdPlane *dp, uint32_t y, const uint32_t *acc,
                                  const McdPlane *sp, int32_t lim, uint32_t pmax) {
    if (lim)
        return 0; /* limit clamp kept scalar (rare knob) */
    const uint32_t w = sp->w;
    const uint32_t *ar = acc + (size_t)y * w;
    if (pmax > 32767)
        return 0; /* narrowing below assumes the sample range fits i16 */
    uint32_t x = 0;
    if (!dp->highbd) {
        uint8_t *dr = (uint8_t *)dp->data + (size_t)y * dp->stride;
        const __m256i c1024 = _mm256_set1_epi32(1024);
        const __m256i vpmax = _mm256_set1_epi32((int)pmax);
        for (; x + 8 <= w; x += 8) {
            __m256i o = _mm256_srai_epi32(_mm256_add_epi32(
                _mm256_loadu_si256((const __m256i *)(const void *)(ar + x)), c1024), 11);
            o = _mm256_min_epu32(o, vpmax); /* values are non-negative */
            /* i32 -> i16 (lane-fixed) -> u8 */
            __m256i pk = _mm256_permute4x64_epi64(_mm256_packs_epi32(o, o),
                                                  _MM_SHUFFLE(3, 1, 2, 0));
            __m128i b = _mm_packus_epi16(_mm256_castsi256_si128(pk),
                                         _mm256_extracti128_si256(pk, 1));
            _mm_storel_epi64((__m128i *)(void *)(dr + x), b);
        }
        for (; x < w; x++) {
            uint32_t o = (ar[x] + 1024) >> 11;
            if (o > pmax) o = pmax;
            dr[x] = (uint8_t)o;
        }
    } else {
        uint16_t *dr = (uint16_t *)dp->data + (size_t)y * dp->stride;
        const __m256i c1024 = _mm256_set1_epi32(1024);
        const __m256i vpmax = _mm256_set1_epi32((int)pmax);
        for (; x + 8 <= w; x += 8) {
            __m256i o = _mm256_srai_epi32(_mm256_add_epi32(
                _mm256_loadu_si256((const __m256i *)(const void *)(ar + x)), c1024), 11);
            o = _mm256_min_epu32(o, vpmax);
            __m256i pk = _mm256_permute4x64_epi64(_mm256_packs_epi32(o, o),
                                                  _MM_SHUFFLE(3, 1, 2, 0));
            _mm_storeu_si128((__m128i *)(void *)(dr + x), _mm256_castsi256_si128(pk));
        }
        for (; x < w; x++) {
            uint32_t o = (ar[x] + 1024) >> 11;
            if (o > pmax) o = pmax;
            dr[x] = (uint16_t)o;
        }
    }
    return 1;
}

/* ================================================================== */
/* dfttest: overlapped 16x16 block-DFT denoise, 16 blocks per pass.    */
/*                                                                     */
/* Layout: lanes (16) = block index within a horizontal strip of 16    */
/* blocks (bx + 4*b); the 256 transform coefficients form a vector     */
/* ARRAY indexed slot = (y*16 + x), each slot a full __m512. Both 1D   */
/* passes are butterflies between vector-array slots (rows: contiguous */
/* slots, columns: stride-16 slots), so no in-register transposes and  */
/* the lane dimension rides along untouched.                           */
/*                                                                     */
/* Bit-exactness: every floating operation is a lane-wise copy of the  */
/* scalar sequence (mul/add/sub/div/compare), no FMA and no            */
/* reassociation; the scalar reference is compiled without FMA (the    */
/* library targets generic x86-64 - verified by disassembly). Gathers  */
/* and scatters touch disjoint elements, so results are deterministic  */
/* and identical to the scalar per-element arithmetic.                 */
/* ================================================================== */

#include "mc_denoise_dfttest_ctx.h"

#define MCD_SIMD_EPSSH      (1e-15f)
#define MCD_SIMD_ACC_MARGIN 96

/* One 1D FFT16 over the 16 vector slots at re/im (stride = one slot). */
__attribute__((target("avx512f")))
static inline void mcd_fft16_vec(float *re, float *im, const struct McdDftCtx *c,
                                 int inverse) {
    for (uint32_t k = 0; k < MCD_DFT_N; k++) {
        const uint32_t j = c->brev[k];
        if (j > k) {
            __m512 t = _mm512_loadu_ps(re + k * 16);
            _mm512_storeu_ps(re + k * 16, _mm512_loadu_ps(re + j * 16));
            _mm512_storeu_ps(re + j * 16, t);
            t = _mm512_loadu_ps(im + k * 16);
            _mm512_storeu_ps(im + k * 16, _mm512_loadu_ps(im + j * 16));
            _mm512_storeu_ps(im + j * 16, t);
        }
    }
    for (uint32_t len = 2; len <= MCD_DFT_N; len <<= 1) {
        const uint32_t half  = len >> 1;
        const uint32_t tstep = MCD_DFT_N / len;
        for (uint32_t ib = 0; ib < MCD_DFT_N; ib += len) {
            for (uint32_t j = 0; j < half; j++) {
                const __m512 wr = _mm512_set1_ps(c->twr[j * tstep]);
                const __m512 wi = _mm512_set1_ps(inverse ? -c->twi[j * tstep]
                                                         : c->twi[j * tstep]);
                float *pr = re + (ib + j) * 16;
                float *pi = im + (ib + j) * 16;
                float *qr = re + (ib + j + half) * 16;
                float *qi = im + (ib + j + half) * 16;
                const __m512 ar = _mm512_loadu_ps(pr);
                const __m512 ai = _mm512_loadu_ps(pi);
                const __m512 br = _mm512_loadu_ps(qr);
                const __m512 bi = _mm512_loadu_ps(qi);
                const __m512 tr = _mm512_sub_ps(_mm512_mul_ps(br, wr),
                                                _mm512_mul_ps(bi, wi));
                const __m512 ti = _mm512_add_ps(_mm512_mul_ps(br, wi),
                                                _mm512_mul_ps(bi, wr));
                _mm512_storeu_ps(pr, _mm512_add_ps(ar, tr));
                _mm512_storeu_ps(pi, _mm512_add_ps(ai, ti));
                _mm512_storeu_ps(qr, _mm512_sub_ps(ar, tr));
                _mm512_storeu_ps(qi, _mm512_sub_ps(ai, ti));
            }
        }
    }
}

/* One 2D transform of the group buffer: rows (contiguous slots), then
 * columns (slots with fixed x, stride 16 slots, staged through scratch). */
__attribute__((target("avx512f")))
static inline void mcd_fft2d_vec(float *re, float *im, const struct McdDftCtx *c,
                                 int inverse, float *cr, float *ci) {
    for (uint32_t y = 0; y < MCD_DFT_N; y++)
        mcd_fft16_vec(re + y * (MCD_DFT_N * 16), im + y * (MCD_DFT_N * 16), c, inverse);
    for (uint32_t x = 0; x < MCD_DFT_N; x++) {
        for (uint32_t y = 0; y < MCD_DFT_N; y++) {
            _mm512_storeu_ps(cr + y * 16, _mm512_loadu_ps(re + (y * MCD_DFT_N + x) * 16));
            _mm512_storeu_ps(ci + y * 16, _mm512_loadu_ps(im + (y * MCD_DFT_N + x) * 16));
        }
        mcd_fft16_vec(cr, ci, c, inverse);
        for (uint32_t y = 0; y < MCD_DFT_N; y++) {
            _mm512_storeu_ps(re + (y * MCD_DFT_N + x) * 16, _mm512_loadu_ps(cr + y * 16));
            _mm512_storeu_ps(im + (y * MCD_DFT_N + x) * 16, _mm512_loadu_ps(ci + y * 16));
        }
    }
}

/* dfttest of a 16-block horizontal strip (lanes = block index b of
 * bx + 4*b). rex/imx: caller-provided 256*16-float scratch buffers;
 * cr/ci: 16*16 scratch for the column pass. */
__attribute__((target("avx512f,avx512dq")))
static void mcd_dfttest_group(const struct McdDftCtx *ctx, const float *pad,
                              uint32_t pdw, int32_t bx, int32_t by,
                              uint32_t aw, int32_t ymax, float *acc,
                              float *rex, float *imx, float *cr, float *ci) {
    /* gather offsets for the windowed input load: prow already points at
     * column (bx + PAD), so lane b (block bx + 4b) reads at + (4b + x) */
    __m512i idx4[16];
    for (uint32_t x = 0; x < 16; x++)
        idx4[x] = _mm512_add_epi32(_mm512_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28,
                                                     32, 36, 40, 44, 48, 52, 56, 60),
                                   _mm512_set1_epi32((int)x));
    /* gather offsets for the OPA: element (y, x, lane) of the group buffer
     * sits at (y*16 + x)*16 + lane; per (y, x) the offset is 256*y + 16*x */
    __m512i idxg[16];
    for (uint32_t y = 0; y < 16; y++)
        idxg[y] = _mm512_add_epi32(_mm512_setr_epi32(0, 16, 32, 48, 64, 80, 96, 112,
                                                     128, 144, 160, 176, 192, 208, 224, 240),
                                   _mm512_set1_epi32(256 * (int)y));

    /* windowed load: slot (y*16+x) = pad[by+y][bx + 4*lane + x] * win */
    for (uint32_t y = 0; y < MCD_DFT_N; y++) {
        const float *prow = pad + (size_t)(by + y + MCD_DFT_PAD) * pdw +
                            (bx + MCD_DFT_PAD);
        for (uint32_t x = 0; x < MCD_DFT_N; x++) {
            const __m512 v = _mm512_mul_ps(
                _mm512_i32gather_ps(idx4[x], prow, 4),
                _mm512_set1_ps(ctx->win[y * MCD_DFT_N + x]));
            _mm512_storeu_ps(rex + (y * MCD_DFT_N + x) * 16, v);
            _mm512_storeu_ps(imx + (y * MCD_DFT_N + x) * 16, _mm512_setzero_ps());
        }
    }

    mcd_fft2d_vec(rex, imx, ctx, 0, cr, ci);

    if (ctx->sigma_pct) {
        /* zmean: remove the windowed local mean (per block = per lane) */
        const __m512 gf = _mm512_div_ps(_mm512_loadu_ps(rex), _mm512_set1_ps(ctx->wfre[0]));
        for (uint32_t k = 0; k < MCD_DFT_N2; k++) {
            float *pr = rex + k * 16;
            float *pi = imx + k * 16;
            const __m512 wr = _mm512_mul_ps(gf, _mm512_set1_ps(ctx->wfre[k]));
            const __m512 wi = _mm512_mul_ps(gf, _mm512_set1_ps(ctx->wfim[k]));
            _mm512_storeu_ps(pr, _mm512_sub_ps(_mm512_loadu_ps(pr), wr));
            _mm512_storeu_ps(pi, _mm512_sub_ps(_mm512_loadu_ps(pi), wi));
        }
        /* type-0 Wiener shrinkage */
        for (uint32_t k = 0; k < MCD_DFT_N2; k++) {
            float *pr = rex + k * 16;
            float *pi = imx + k * 16;
            const __m512 re = _mm512_loadu_ps(pr);
            const __m512 im = _mm512_loadu_ps(pi);
            const __m512 psd = _mm512_add_ps(_mm512_mul_ps(re, re), _mm512_mul_ps(im, im));
            const __m512 num = _mm512_sub_ps(psd, _mm512_set1_ps(ctx->sigma2[k]));
            const __m512 den = _mm512_add_ps(psd, _mm512_set1_ps(MCD_SIMD_EPSSH));
            const __mmask16 mask = _mm512_cmp_ps_mask(num, _mm512_setzero_ps(), _CMP_GT_OQ);
            const __m512 mult = _mm512_mask_blend_ps(mask, _mm512_setzero_ps(),
                                                     _mm512_div_ps(num, den));
            _mm512_storeu_ps(pr, _mm512_mul_ps(re, mult));
            _mm512_storeu_ps(pi, _mm512_mul_ps(im, mult));
        }
        /* restore the mean */
        for (uint32_t k = 0; k < MCD_DFT_N2; k++) {
            float *pr = rex + k * 16;
            float *pi = imx + k * 16;
            const __m512 wr = _mm512_mul_ps(gf, _mm512_set1_ps(ctx->wfre[k]));
            const __m512 wi = _mm512_mul_ps(gf, _mm512_set1_ps(ctx->wfim[k]));
            _mm512_storeu_ps(pr, _mm512_add_ps(_mm512_loadu_ps(pr), wr));
            _mm512_storeu_ps(pi, _mm512_add_ps(_mm512_loadu_ps(pi), wi));
        }
    }

    mcd_fft2d_vec(rex, imx, ctx, 1, cr, ci);

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
            if (oy < 0 || oy > ymax)
                continue;
            const __m512 v = _mm512_mul_ps(
                _mm512_mask_i32gather_ps(_mm512_setzero_ps(), (__mmask16)0xffff,
                                         _mm512_add_epi32(idxg[y], _mm512_set1_epi32((int)b)),
                                         rex, 4),
                _mm512_loadu_ps(ctx->win + y * MCD_DFT_N));
            float *ar = acc + (size_t)oy * aw + obx;
            _mm512_storeu_ps(ar, _mm512_add_ps(_mm512_loadu_ps(ar), v));
        }
    }
}

/* SIMD dfttest plane: mirrors mcd_dfttest_plane_with_ctx (same padding,
 * same block grid, same arithmetic order) with 16-block vector groups. */
__attribute__((target("avx512f,avx512bw,avx512dq")))
int mcd_simd_dfttest_plane_with_ctx(const struct McdDftCtx *ctx, const McdPlane *src,
                                    McdPlane *dst) {
    if (!ctx || !src || !dst || !src->data)
        return MCD_ERR_PARAM;
    const uint32_t w = src->w;
    const uint32_t h = src->h;
    if (w == 0 || h == 0 || src->stride < w)
        return MCD_ERR_PARAM;

    const uint32_t bd     = src->highbd ? 10 : 8;
    const uint32_t pmax   = (1u << bd) - 1;
    const float    iscale = 1.0f / (float)(1u << (bd - 8));
    const float    oscale = (float)(1u << (bd - 8));

    if (!dst->data) {
        dst->data = malloc((size_t)w * h * (src->highbd ? sizeof(uint16_t)
                                                        : sizeof(uint8_t)));
        if (!dst->data)
            return MCD_ERR_NOMEM;
        dst->w      = w;
        dst->h      = h;
        dst->stride = w;
        dst->highbd = src->highbd;
    } else if (dst->w != w || dst->h != h || dst->stride < w ||
               dst->highbd != src->highbd) {
        return MCD_ERR_PARAM;
    }


    const uint32_t pw = w + 2 * MCD_DFT_PAD;
    /* extra columns so the group's gathers for off-grid blocks of the last
     * strip stay in bounds (their values are never accumulated) */
    const uint32_t pdw = pw + 4 * (MCD_DFT_N - 1) + MCD_DFT_N;
    const uint32_t ph = h + 2 * MCD_DFT_PAD;

    int32_t *xmap = malloc(sizeof(int32_t) * pdw);
    int32_t *ymap = malloc(sizeof(int32_t) * ph);
    /* calloc: gathers for off-grid lanes read (unused) pad columns; keep
     * them initialized so MSan builds stay quiet */
    float   *pad  = calloc((size_t)pdw * ph, sizeof(float));
    /* accumulation buffer with a per-row margin on both sides so block
     * overhang past the frame edges lands in the row's own margin (visible
     * core of row y = [MARGIN, MARGIN + w) within the padded row) */
    const uint32_t aw = w + 2 * MCD_SIMD_ACC_MARGIN;
    float   *acc  = calloc((size_t)aw * h, sizeof(float));
    float   *scratch = malloc(6 * 256 * 16 * sizeof(float));
    int      ret  = MCD_SUCCESS;

    if (!xmap || !ymap || !pad || !acc || !scratch) {
        ret = MCD_ERR_NOMEM;
        goto cleanup;
    }

    for (uint32_t x = 0; x < pw; x++)
        xmap[x] = mcd_simd_mirror((int32_t)x - MCD_DFT_PAD, w);
    for (uint32_t x = pw; x < pdw; x++)
        xmap[x] = w - 1; /* off-grid gather clamp (values never used) */
    for (uint32_t y = 0; y < ph; y++)
        ymap[y] = mcd_simd_mirror((int32_t)y - MCD_DFT_PAD, h);

    /* mirror-padded copy in the internal 8-bit-range float scale: the
     * core region is a contiguous conversion (vector), the margins keep
     * the mirrored gather (scalar) */
    for (uint32_t y = 0; y < ph; y++) {
        const size_t srow = (size_t)ymap[y] * src->stride;
        float *pr = pad + (size_t)y * pdw;
        /* interior output column x mirrors source column x - MCD_DFT_PAD */
        if (src->highbd) {
            const uint16_t *s = src->data;
            uint32_t x = MCD_DFT_PAD;
            for (; x + 16 <= MCD_DFT_PAD + w; x += 8) {
                const __m512 f = _mm512_cvtepu32_ps(_mm512_cvtepu16_epi32(
                    _mm256_loadu_si256((const __m256i *)(const void *)(s + srow + x - MCD_DFT_PAD))));
                _mm512_storeu_ps(pr + x, _mm512_mul_ps(f, _mm512_set1_ps(iscale)));
            }
            for (; x < MCD_DFT_PAD + w; x++)
                pr[x] = (float)s[srow + x - MCD_DFT_PAD] * iscale;
            for (x = 0; x < MCD_DFT_PAD; x++)
                pr[x] = (float)s[srow + xmap[x]] * iscale;
            for (x = MCD_DFT_PAD + w; x < pw; x++)
                pr[x] = (float)s[srow + xmap[x]] * iscale;
        } else {
            const uint8_t *s = src->data;
            uint32_t x = MCD_DFT_PAD;
            for (; x + 16 <= MCD_DFT_PAD + w; x += 16) {
                const __m512 f = _mm512_cvtepu32_ps(_mm512_cvtepu8_epi32(
                    _mm_loadu_si128((const __m128i *)(const void *)(s + srow + x - MCD_DFT_PAD))));
                _mm512_storeu_ps(pr + x, _mm512_mul_ps(f, _mm512_set1_ps(iscale)));
            }
            for (; x < MCD_DFT_PAD + w; x++)
                pr[x] = (float)s[srow + x - MCD_DFT_PAD] * iscale;
            for (x = 0; x < MCD_DFT_PAD; x++)
                pr[x] = (float)s[srow + xmap[x]] * iscale;
            for (x = MCD_DFT_PAD + w; x < pw; x++)
                pr[x] = (float)s[srow + xmap[x]] * iscale;
        }
    }

    {
        const int32_t xmax = (int32_t)w - 1;
        const int32_t ymax = (int32_t)h - 1;
        float *rex = scratch;
        float *imx = scratch + 256 * 16;
        float *cr  = scratch + 2 * 256 * 16;
        float *ci  = scratch + 3 * 256 * 16;

        for (int32_t by = -(MCD_DFT_N - MCD_DFT_STEP); by <= ymax; by += MCD_DFT_STEP) {
            for (int32_t bx = -(MCD_DFT_N - MCD_DFT_STEP); bx <= xmax; bx += MCD_DFT_STEP) {
                mcd_dfttest_group(ctx, pad, pdw, bx, by, aw, ymax,
                                  acc + MCD_SIMD_ACC_MARGIN, rex, imx, cr, ci);
                /* 16 blocks per group: skip the 15 covered positions */
                bx += 15 * MCD_DFT_STEP;
                if (bx > xmax) break;
            }
        }
    }

    /* back to native scale, rounded and clamped (visible core only):
     * (int32_t)(acc * oscale + 0.5f) is a truncating conversion, so the
     * vector path uses cvttps and identical clamps - bit-exact */
    {
        const __m512 vos = _mm512_set1_ps(oscale);
        const __m512 vh = _mm512_set1_ps(0.5f);
        for (uint32_t y = 0; y < h; y++) {
            const float *ar = acc + MCD_SIMD_ACC_MARGIN + (size_t)y * aw;
            uint32_t x = 0;
            if (dst->highbd) {
                uint16_t *dr = (uint16_t *)dst->data + (size_t)y * dst->stride;
                for (; x + 16 <= w; x += 16) {
                    __m512i v = _mm512_cvttps_epi32(_mm512_add_ps(
                        _mm512_mul_ps(_mm512_loadu_ps(ar + x), vos), vh));
                    v = _mm512_max_epi32(v, _mm512_setzero_si512());
                    v = _mm512_min_epi32(v, _mm512_set1_epi32((int)pmax));
                    _mm256_storeu_si256((__m256i *)(void *)(dr + x),
                                        _mm512_cvtusepi32_epi16(v));
                }
            } else {
                uint8_t *dr = (uint8_t *)dst->data + (size_t)y * dst->stride;
                for (; x + 16 <= w; x += 16) {
                    __m512i v = _mm512_cvttps_epi32(_mm512_add_ps(
                        _mm512_mul_ps(_mm512_loadu_ps(ar + x), vos), vh));
                    v = _mm512_max_epi32(v, _mm512_setzero_si512());
                    v = _mm512_min_epi32(v, _mm512_set1_epi32((int)pmax));
                    __m256i p16 = _mm512_cvtusepi32_epi16(v);
                    __m128i p8 = _mm_packus_epi16(_mm256_castsi256_si128(p16),
                                                  _mm256_extracti128_si256(p16, 1));
                    _mm_storeu_si128((__m128i *)(void *)(dr + x), p8);
                }
            }
            for (; x < w; x++) {
                int32_t v = (int32_t)(ar[x] * oscale + 0.5f);
                if (v < 0)
                    v = 0;
                if (v > (int32_t)pmax)
                    v = (int32_t)pmax;
                if (dst->highbd)
                    ((uint16_t *)dst->data)[(size_t)y * dst->stride + x] = (uint16_t)v;
                else
                    ((uint8_t *)dst->data)[(size_t)y * dst->stride + x] = (uint8_t)v;
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

#endif /* x86-64 */

