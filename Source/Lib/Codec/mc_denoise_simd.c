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
 * dispatch to the scalar path (see svt_mcd_simd_disabled).
 */

/* Baseline-compiled mc-denoise SIMD support: the forced-scalar test hook and
 * stub kernels used as RTCD defaults. The real kernels live in ASM_AVX2 and
 * ASM_AVX512 and are selected in common_dsp_rtcd.c. */
/* Per-ISA kernels live in mc_denoise_simd_avx2.c and mc_denoise_simd_avx512.c,
 * compiled as separate object libraries with their own arch flags (the repo
 * convention). This TU is baseline-compiled and only dispatches. Non-x86
 * targets get the unreachable stubs below. */
//for getenv on windows
#if defined(_WIN32) && !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#endif
#include <stdlib.h>

#include "mc_denoise.h"
#include "mc_denoise_simd.h"

int svt_mcd_simd_disabled(void) {
    return getenv("MCD_NO_SIMD") != NULL;
}

/* -------------------- RTCD pointers ------------------------------------ */
uint64_t(*svt_mcd_simd_sad_u8u8)(const uint8_t* a, int32_t sa, const uint8_t* b, int32_t sb, uint32_t w, uint32_t h,
                               uint64_t bound) = svt_mcd_simd_sad_u8u8_c;
uint64_t(*svt_mcd_simd_sad_u16u16)(const uint16_t* a, int32_t sa, const uint16_t* b, int32_t sb, uint32_t w, uint32_t h,
                                 uint64_t bound) = svt_mcd_simd_sad_u16u16_c;
uint64_t(*svt_mcd_simd_sad_u8u16)(const uint8_t* a, int32_t sa, const uint16_t* b, int32_t sb, uint32_t w, uint32_t h,
                                uint64_t bound) = svt_mcd_simd_sad_u8u16_c;
uint64_t(*svt_mcd_simd_sad_u16u8)(const uint16_t* a, int32_t sa, const uint8_t* b, int32_t sb, uint32_t w, uint32_t h,
                                uint64_t bound) = svt_mcd_simd_sad_u16u8_c;
uint64_t(*svt_mcd_simd_satd_u8u8)(const uint8_t* a, int32_t sa, const uint8_t* b, int32_t sb, uint32_t w, uint32_t h,
                                uint64_t bound) = svt_mcd_simd_satd_u8u8_c;
uint64_t(*svt_mcd_simd_satd_u16u16)(const uint16_t* a, int32_t sa, const uint16_t* b, int32_t sb, uint32_t w, uint32_t h,
                                  uint64_t bound) = svt_mcd_simd_satd_u16u16_c;
void(*svt_mcd_simd_halfpel_h_u8)(const uint8_t* src, uint32_t sstride, uint32_t w, uint32_t h, uint32_t pad, uint8_t* dst,
                               uint32_t dstride, uint32_t maxval) = svt_mcd_simd_halfpel_h_u8_c;
void(*svt_mcd_simd_halfpel_v_u8)(const uint8_t* src, uint32_t sstride, uint32_t w, uint32_t h, uint32_t pad, uint8_t* dst,
                               uint32_t dstride, uint32_t maxval) = svt_mcd_simd_halfpel_v_u8_c;
void(*svt_mcd_simd_halfpel_h_u16)(const uint16_t* src, uint32_t sstride, uint32_t w, uint32_t h, uint32_t pad,
                                uint16_t* dst, uint32_t dstride, uint32_t maxval) = svt_mcd_simd_halfpel_h_u16_c;
void(*svt_mcd_simd_halfpel_v_u16)(const uint16_t* src, uint32_t sstride, uint32_t w, uint32_t h, uint32_t pad,
                                uint16_t* dst, uint32_t dstride, uint32_t maxval) = svt_mcd_simd_halfpel_v_u16_c;
void(*svt_mcd_simd_fetch_u8_row)(const uint8_t* row, uint16_t* dst, uint32_t w) = svt_mcd_simd_fetch_u8_row_c;
void(*svt_mcd_simd_degrain_init_row)(uint32_t* bacc, const McdPlane* sp, uint32_t Y, uint32_t x0_p, uint32_t wsrc,
                                   uint32_t n) = svt_mcd_simd_degrain_init_row_c;
void(*svt_mcd_simd_degrain_add_ref)(uint32_t* bacc, const uint16_t* comp, uint32_t w, uint32_t n) = svt_mcd_simd_degrain_add_ref_c;
void(*svt_mcd_simd_degrain_shift)(uint32_t* bacc, uint32_t n) = svt_mcd_simd_degrain_shift_c;
void(*svt_mcd_simd_degrain_acc_row)(uint32_t* accrow, const uint32_t* brow, const uint32_t* win, uint32_t n) = svt_mcd_simd_degrain_acc_row_c;
int(*svt_mcd_simd_degrain_finalize_row)(McdPlane* dp, uint32_t y, const uint32_t* acc, const McdPlane* sp, int32_t lim,
                                      uint32_t pmax) = svt_mcd_simd_degrain_finalize_row_c;
int(*svt_mcd_simd_mask_merge)(const McdPlane* src_planes, const McdPlane* filt_planes, McdPlane* out_planes,
                            uint32_t n_planes, uint32_t x_ss, uint32_t y_ss, uint32_t bitdepth) = svt_mcd_simd_mask_merge_c;
int(*svt_mcd_simd_dfttest_plane_with_ctx)(const struct McdDftCtx* ctx, const McdPlane* src, McdPlane* dst,
                                        uint32_t bitdepth) = svt_mcd_simd_dfttest_plane_with_ctx_c;
