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
 * Environment override for testing: setting MCD_NO_SIMD (any value) forces
 * the scalar paths everywhere (checked once per process). This is a debug/test
 * hook only; the encoder CLI is unchanged.
 */
#ifndef MC_DENOISE_SIMD_H_
#define MC_DENOISE_SIMD_H_

#include <stdint.h>
#include "mc_denoise.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- forced-scalar test hook (1 = use scalar paths) ------------------- */
int svt_mcd_simd_disabled(void);

/* The dfttest context struct is shared with the scalar implementation; the
 * definition lives in mc_denoise_dfttest.c. */
struct McdDftCtx;

/* Per-ISA mask-merge kernels */
int mcd_avx2_mask_merge(const McdPlane* src_planes, const McdPlane* filt_planes, McdPlane* out_planes,
                        uint32_t n_planes, uint32_t x_ss, uint32_t y_ss, uint32_t bitdepth);
int mcd_avx512_mask_merge(const McdPlane* src_planes, const McdPlane* filt_planes, McdPlane* out_planes,
                          uint32_t n_planes, uint32_t x_ss, uint32_t y_ss, uint32_t bitdepth);

/* Scalar C implementations (RTCD defaults) */
uint64_t svt_mcd_simd_sad_u8u8_c(const uint8_t* a, int32_t sa, const uint8_t* b, int32_t sb, uint32_t w, uint32_t h,
                                 uint64_t bound);
uint64_t svt_mcd_simd_sad_u16u16_c(const uint16_t* a, int32_t sa, const uint16_t* b, int32_t sb, uint32_t w, uint32_t h,
                                   uint64_t bound);
uint64_t svt_mcd_simd_sad_u8u16_c(const uint8_t* a, int32_t sa, const uint16_t* b, int32_t sb, uint32_t w, uint32_t h,
                                  uint64_t bound);
uint64_t svt_mcd_simd_sad_u16u8_c(const uint16_t* a, int32_t sa, const uint8_t* b, int32_t sb, uint32_t w, uint32_t h,
                                  uint64_t bound);
uint64_t svt_mcd_simd_satd_u8u8_c(const uint8_t* a, int32_t sa, const uint8_t* b, int32_t sb, uint32_t w, uint32_t h,
                                  uint64_t bound);
uint64_t svt_mcd_simd_satd_u16u16_c(const uint16_t* a, int32_t sa, const uint16_t* b, int32_t sb, uint32_t w,
                                    uint32_t h, uint64_t bound);
void     svt_mcd_simd_halfpel_h_u8_c(const uint8_t* src, uint32_t sstride, uint32_t w, uint32_t h, uint32_t pad,
                                     uint8_t* dst, uint32_t dstride, uint32_t maxval);
void     svt_mcd_simd_halfpel_v_u8_c(const uint8_t* src, uint32_t sstride, uint32_t w, uint32_t h, uint32_t pad,
                                     uint8_t* dst, uint32_t dstride, uint32_t maxval);
void     svt_mcd_simd_halfpel_h_u16_c(const uint16_t* src, uint32_t sstride, uint32_t w, uint32_t h, uint32_t pad,
                                      uint16_t* dst, uint32_t dstride, uint32_t maxval);
void     svt_mcd_simd_halfpel_v_u16_c(const uint16_t* src, uint32_t sstride, uint32_t w, uint32_t h, uint32_t pad,
                                      uint16_t* dst, uint32_t dstride, uint32_t maxval);
void     svt_mcd_simd_fetch_u8_row_c(const uint8_t* row, uint16_t* dst, uint32_t w);
void     svt_mcd_simd_degrain_init_row_c(uint32_t* bacc, const McdPlane* sp, uint32_t Y, uint32_t x0_p, uint32_t wsrc,
                                         uint32_t n);
void     svt_mcd_simd_degrain_add_ref_c(uint32_t* bacc, const uint16_t* comp, uint32_t w, uint32_t n);
void     svt_mcd_simd_degrain_shift_c(uint32_t* bacc, uint32_t n);
void     svt_mcd_simd_degrain_acc_row_c(uint32_t* accrow, const uint32_t* brow, const uint32_t* win, uint32_t n);
int svt_mcd_simd_degrain_finalize_row_c(McdPlane* dp, uint32_t y, const uint32_t* acc, const McdPlane* sp, int32_t lim,
                                        uint32_t pmax);
int svt_mcd_simd_mask_merge_c(const McdPlane* src_planes, const McdPlane* filt_planes, McdPlane* out_planes,
                              uint32_t n_planes, uint32_t x_ss, uint32_t y_ss, uint32_t bitdepth);
int svt_mcd_simd_dfttest_plane_with_ctx_c(const struct McdDftCtx* ctx, const McdPlane* src, McdPlane* dst,
                                          uint32_t bitdepth);

#ifdef __cplusplus
}
#endif

#endif // MC_DENOISE_SIMD_H_
