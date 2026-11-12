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
 * mc_denoise_dfttest_ctx.h — internal (shared scalar/SIMD) definition of the
 * dfttest constant-table context. Not part of the module contract; include
 * only from mc_denoise_dfttest.c and mc_denoise_simd.c.
 */
#ifndef MC_DENOISE_DFTTEST_CTX_H_
#define MC_DENOISE_DFTTEST_CTX_H_

#include <stdint.h>

#include "mc_denoise.h"

#define MCD_DFT_N 16 /* transform block size        */
#define MCD_DFT_STEP 4 /* block stepping (75% overlap)*/
#define MCD_DFT_PAD 16 /* mirror padding per side     */
#define MCD_DFT_N2 (MCD_DFT_N * MCD_DFT_N)

struct McdDftCtx {
    float    twr[8], twi[8]; /* fwd twiddles e^{-2*pi*i*k/16}        */
    uint8_t  brev[MCD_DFT_N]; /* bit-reversal permutation             */
    float    win[MCD_DFT_N2]; /* analysis = synthesis window          */
    float    wscale; /* sum of win^2 over the block          */
    float    wfre[MCD_DFT_N2]; /* DFT(win): zmean profile, real part   */
    float    wfim[MCD_DFT_N2]; /* DFT(win): imaginary part             */
    float    sigma2[MCD_DFT_N2]; /* per-coefficient noise variance       */
    uint32_t sigma_pct; /* the sigma this context was built for */
};

#endif // MC_DENOISE_DFTTEST_CTX_H_
