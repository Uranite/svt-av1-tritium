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
 * mc_denoise_simd.c - mc-denoise SIMD support.
 *
 * Baseline-compiled: holds the forced-scalar test hook and the RTCD pointer
 * storage. The AVX2 and AVX-512 kernels live in Source/Lib/ASM_AVX2 and
 * Source/Lib/ASM_AVX512 and are selected in common_dsp_rtcd.c from the CPU
 * flags; every pointer falls back to the scalar C implementation.
 *
 * Testing hook: environment variable MCD_NO_SIMD (any value) makes the RTCD
 * setup select the scalar implementations for every kernel.
 */
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
