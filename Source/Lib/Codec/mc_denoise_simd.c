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

