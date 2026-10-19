/*
 * mc_denoise_simd.h — runtime dispatch layer for the mc_denoise SIMD kernels.
 *
 * The scalar implementations in mc_denoise_*.c remain the reference (and the
 * fallback); each hot kernel gains a SIMD twin in mc_denoise_simd.c selected
 * at runtime by CPU feature. Dispatch is a cheap feature check, no state.
 *
 * Exactness contract (see mc_denoise.h and the simd .c file):
 *   - integer kernels are bit-identical to the scalar versions;
 *   - float kernels (dfttest) are bit-identical: lane-wise IEEE operations,
 *     no FMA contraction, no reassociation, same operation order.
 *
 * Environment override for testing: setting MCD_NO_SIMD (any value)
 * paths everywhere (checked once per process). This is a debug/test hook only;
 * the encoder CLI is unchanged.
 */
#ifndef MC_DENOISE_SIMD_H
#define MC_DENOISE_SIMD_H

#include <stdint.h>
#include "mc_denoise.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- forced-scalar test hook (1 = use scalar paths) ------------------- */
int mcd_simd_disabled(void);

/* ---- CPU feature checks (cheap, uncached reads of compiler-provided
 * cpu model; safe from any thread).
 * GCC/Clang only: __builtin_cpu_supports does not exist on MSVC, so MSVC
 * (and any other compiler) gets the scalar paths. ---------------------------------- */
#if (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__))
static inline int mcd_have_avx2(void) {
    return __builtin_cpu_supports("avx2");
}
static inline int mcd_have_avx512(void) {
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
           __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512dq");
}
#else
static inline int mcd_have_avx2(void) { return 0; }
static inline int mcd_have_avx512(void) { return 0; }
#endif

static inline int mcd_use_avx2(void) {
    return !mcd_simd_disabled() && mcd_have_avx2();
}
static inline int mcd_use_avx512(void) {
    return !mcd_simd_disabled() && mcd_have_avx512();
}

/* ---- SAD kernels (bit-exact vs scalar) -------------------------------- */
/* bound != 0 enables the partial-row early abort: once the accumulated sum
 * reaches `bound` the kernel returns early with a value that is both >=
 * bound and a lower bound of the full sum. Partial sums are monotone, so
 * this only short-circuits candidates that cannot beat `bound` — the
 * comparison outcomes at the call sites are unchanged (mc_denoise_me.c).
 * bound == 0 computes the full sum. */
uint64_t mcd_simd_sad_u8u8(const uint8_t *a, int32_t sa, const uint8_t *b, int32_t sb,
                           uint32_t w, uint32_t h, uint64_t bound);
uint64_t mcd_simd_sad_u16u16(const uint16_t *a, int32_t sa, const uint16_t *b, int32_t sb,
                             uint32_t w, uint32_t h, uint64_t bound);
uint64_t mcd_simd_sad_u8u16(const uint8_t *a, int32_t sa, const uint16_t *b, int32_t sb,
                            uint32_t w, uint32_t h, uint64_t bound);
uint64_t mcd_simd_sad_u16u8(const uint16_t *a, int32_t sa, const uint8_t *b, int32_t sb,
                            uint32_t w, uint32_t h, uint64_t bound);

/* ---- SATD kernel (bit-exact vs scalar; bound as for the SADs) ---------- */
uint64_t mcd_simd_satd_u8u8(const uint8_t *a, int32_t sa, const uint8_t *b, int32_t sb,
                            uint32_t w, uint32_t h, uint64_t bound);
uint64_t mcd_simd_satd_u16u16(const uint16_t *a, int32_t sa, const uint16_t *b, int32_t sb,
                              uint32_t w, uint32_t h, uint64_t bound);

/* ---- pelset: half-pel phase plane filters (bit-exact vs scalar) -------- */
void mcd_simd_halfpel_h_u8(const uint8_t *src, uint32_t sstride, uint32_t w, uint32_t h,
                           uint32_t pad, uint8_t *dst, uint32_t dstride, uint32_t maxval);
void mcd_simd_halfpel_v_u8(const uint8_t *src, uint32_t sstride, uint32_t w, uint32_t h,
                           uint32_t pad, uint8_t *dst, uint32_t dstride, uint32_t maxval);
void mcd_simd_halfpel_h_u16(const uint16_t *src, uint32_t sstride, uint32_t w, uint32_t h,
                            uint32_t pad, uint16_t *dst, uint32_t dstride, uint32_t maxval);
void mcd_simd_halfpel_v_u16(const uint16_t *src, uint32_t sstride, uint32_t w, uint32_t h,
                            uint32_t pad, uint16_t *dst, uint32_t dstride, uint32_t maxval);
/* row copy u8 -> u16 (pelset fetch), w bytes per row */
void mcd_simd_fetch_u8_row(const uint8_t *row, uint16_t *dst, uint32_t w);

/* ---- degrain element-wise loops (bit-exact vs scalar) ----------------- */
/* bacc row init: bacc[i] = 128 + wsrc * plane_sample(Y, x0_p + i)        */
void mcd_simd_degrain_init_row(uint32_t *bacc, const McdPlane *sp, uint32_t Y,
                               uint32_t x0_p, uint32_t wsrc, uint32_t n);
/* bacc[i] += w * comp[i] (comp is the fetched uint16 reference block)    */
void mcd_simd_degrain_add_ref(uint32_t *bacc, const uint16_t *comp, uint32_t w,
                              uint32_t n);
/* bacc[i] >>= 8                                                          */
void mcd_simd_degrain_shift(uint32_t *bacc, uint32_t n);
/* accrow[i] += brow[i] * win[i]                                          */
void mcd_simd_degrain_acc_row(uint32_t *accrow, const uint32_t *brow,
                              const uint32_t *win, uint32_t n);
/* finalize: out = clamp(acc[i]+1024)>>11 with limit clamp vs the source;
 * returns 1 when the SIMD path handled the row, 0 = caller does scalar   */
int mcd_simd_degrain_finalize_row(McdPlane *dp, uint32_t y, const uint32_t *acc,
                                  const McdPlane *sp, int32_t lim, uint32_t pmax);

/* ---- luma-masked merge (bit-exact vs scalar: lane-wise float ops, no FMA)
 * Computes mcd_prefilter_mask_merge over the validated plane triplet.
 * Returns MCD_SUCCESS when it handled the whole call; any unsupported
 * geometry returns MCD_ERR_PARAM and the caller must use the scalar path
 * (no partial output is written in that case). */
int mcd_simd_mask_merge(const McdPlane *src_planes, const McdPlane *filt_planes,
                        McdPlane *out_planes, uint32_t n_planes,
                        uint32_t x_ss, uint32_t y_ss, uint32_t bitdepth);

/* ---- dfttest (bit-exact vs scalar: no FMA, no reassociation) ----------- */
/* Returns MCD_SUCCESS or MCD_ERR_* like the scalar entry; falls back to the
 * scalar implementation internally when the geometry is unsupported. */
int mcd_simd_dfttest_plane_with_ctx(const struct McdDftCtx *ctx, const McdPlane *src,
                                    McdPlane *dst);

/* The dfttest context struct is shared with the scalar implementation; the
 * definition lives in mc_denoise_dfttest.c. */
struct McdDftCtx;

#ifdef __cplusplus
}
#endif

#endif /* MC_DENOISE_SIMD_H */
