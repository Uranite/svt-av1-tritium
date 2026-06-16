/*
 * Copyright (c) 2024, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

// Low-bit-depth (8-bit) forward transforms, int16 intermediates, all tx_types.
// Wholesale-adapted from libaom (av1/encoder/arm/av1_fwd_txfm2d_neon.c). This
// is the comprehensive int16 forward path; sizes are added incrementally and
// each is validated bit-exact vs svt_av1_transform_two_d_*_c (FwdTxfm2dAsmTest).
//
// Ported sizes so far: 4x4.

#include <arm_neon.h>
#include "aom_dsp_rtcd.h"
#include "definitions.h"
#include "transforms.h"
#include "transpose_neon.h"

#define TXFM_COS_BIT_MAX 13
#define TXFM_COS_BIT_MIN 10
#define NEW_SQRT2_BITS 12
// 2^12 * sqrt(2)
#define NEW_SQRT2 5793

// cospi constants in Q2.13, indexed [cos_bit - 10] (libaom av1_cospi_arr_q13_data).
static const int16_t fwd_cospi_arr_q13[4][128] = {
    {
        5792, 5792, -5792, -5792, 7568, 3136, -7568, -3136, 8032, 1600, -8032, -1600, 6808, 4552, -6808, -4552,
        8152, 800,  -8152, -800,  7840, 2376, -7840, -2376, 7224, 3864, -7224, -3864, 6336, 5200, -6336, -5200,
        8184, 400,  -8184, -400,  8104, 1200, -8104, -1200, 7944, 1992, -7944, -1992, 7712, 2760, -7712, -2760,
        7408, 3504, -7408, -3504, 7024, 4208, -7024, -4208, 6576, 4880, -6576, -4880, 6072, 5504, -6072, -5504,
        8192, 200,  -8192, -200,  8168, 600,  -8168, -600,  8128, 1000, -8128, -1000, 8072, 1400, -8072, -1400,
        7992, 1792, -7992, -1792, 7896, 2184, -7896, -2184, 7776, 2568, -7776, -2568, 7640, 2952, -7640, -2952,
        7488, 3320, -7488, -3320, 7320, 3680, -7320, -3680, 7128, 4040, -7128, -4040, 6920, 4384, -6920, -4384,
        6696, 4720, -6696, -4720, 6456, 5040, -6456, -5040, 6200, 5352, -6200, -5352, 5936, 5648, -5936, -5648,
    },
    {
        5792, 5792, -5792, -5792, 7568, 3136, -7568, -3136, 8036, 1600, -8036, -1600, 6812, 4552, -6812, -4552,
        8152, 804,  -8152, -804,  7840, 2380, -7840, -2380, 7224, 3860, -7224, -3860, 6332, 5196, -6332, -5196,
        8184, 400,  -8184, -400,  8104, 1204, -8104, -1204, 7948, 1992, -7948, -1992, 7712, 2760, -7712, -2760,
        7404, 3504, -7404, -3504, 7028, 4212, -7028, -4212, 6580, 4880, -6580, -4880, 6068, 5500, -6068, -5500,
        8188, 200,  -8188, -200,  8168, 604,  -8168, -604,  8132, 1004, -8132, -1004, 8072, 1400, -8072, -1400,
        7992, 1796, -7992, -1796, 7896, 2184, -7896, -2184, 7780, 2568, -7780, -2568, 7644, 2948, -7644, -2948,
        7488, 3320, -7488, -3320, 7316, 3684, -7316, -3684, 7128, 4036, -7128, -4036, 6920, 4384, -6920, -4384,
        6696, 4716, -6696, -4716, 6460, 5040, -6460, -5040, 6204, 5352, -6204, -5352, 5932, 5648, -5932, -5648,
    },
    {
        5792, 5792, -5792, -5792, 7568, 3134, -7568, -3134, 8034, 1598, -8034, -1598, 6812, 4552, -6812, -4552,
        8152, 802,  -8152, -802,  7840, 2378, -7840, -2378, 7224, 3862, -7224, -3862, 6332, 5196, -6332, -5196,
        8182, 402,  -8182, -402,  8104, 1202, -8104, -1202, 7946, 1990, -7946, -1990, 7714, 2760, -7714, -2760,
        7406, 3502, -7406, -3502, 7026, 4212, -7026, -4212, 6580, 4880, -6580, -4880, 6070, 5502, -6070, -5502,
        8190, 202,  -8190, -202,  8170, 602,  -8170, -602,  8130, 1002, -8130, -1002, 8072, 1400, -8072, -1400,
        7992, 1794, -7992, -1794, 7896, 2184, -7896, -2184, 7778, 2570, -7778, -2570, 7644, 2948, -7644, -2948,
        7490, 3320, -7490, -3320, 7318, 3684, -7318, -3684, 7128, 4038, -7128, -4038, 6922, 4382, -6922, -4382,
        6698, 4718, -6698, -4718, 6458, 5040, -6458, -5040, 6204, 5350, -6204, -5350, 5934, 5648, -5934, -5648,
    },
    {
        5793, 5793, -5793, -5793, 7568, 3135, -7568, -3135, 8035, 1598, -8035, -1598, 6811, 4551, -6811, -4551,
        8153, 803,  -8153, -803,  7839, 2378, -7839, -2378, 7225, 3862, -7225, -3862, 6333, 5197, -6333, -5197,
        8182, 402,  -8182, -402,  8103, 1202, -8103, -1202, 7946, 1990, -7946, -1990, 7713, 2760, -7713, -2760,
        7405, 3503, -7405, -3503, 7027, 4212, -7027, -4212, 6580, 4880, -6580, -4880, 6070, 5501, -6070, -5501,
        8190, 201,  -8190, -201,  8170, 603,  -8170, -603,  8130, 1003, -8130, -1003, 8071, 1401, -8071, -1401,
        7993, 1795, -7993, -1795, 7895, 2185, -7895, -2185, 7779, 2570, -7779, -2570, 7643, 2948, -7643, -2948,
        7489, 3320, -7489, -3320, 7317, 3683, -7317, -3683, 7128, 4038, -7128, -4038, 6921, 4383, -6921, -4383,
        6698, 4717, -6698, -4717, 6458, 5040, -6458, -5040, 6203, 5351, -6203, -5351, 5933, 5649, -5933, -5649,
    }};

// sinpi constants in Q2.13, indexed [cos_bit - 10] (libaom av1_sinpi_arr_q13_data).
static const int16_t fwd_sinpi_arr_q13[4][4] = {
    {2640, 4968, 6688, 7608}, {2640, 4964, 6688, 7604}, {2642, 4964, 6688, 7606}, {2642, 4964, 6689, 7606}};

static AOM_FORCE_INLINE const int16_t* fwd_cospi_q13(int cos_bit) {
    return fwd_cospi_arr_q13[cos_bit - TXFM_COS_BIT_MIN];
}

static AOM_FORCE_INLINE const int16_t* fwd_sinpi_q13(int cos_bit) {
    return fwd_sinpi_arr_q13[cos_bit - TXFM_COS_BIT_MIN];
}

// ---------------------------------------------------------------------------
// Shared helpers (x4 lane width)
// ---------------------------------------------------------------------------

static AOM_FORCE_INLINE void load_buffer_s16_x4(const int16_t* in, const int stride, int16x4_t* const out,
                                                const int out_size) {
    for (int i = 0; i < out_size; ++i) {
        out[i] = vld1_s16(in);
        in += stride;
    }
}

static AOM_FORCE_INLINE void store_buffer_s16_x4(const int16x4_t* const in, int32_t* const out, const int stride,
                                                 const int out_size) {
    for (int i = 0; i < out_size; ++i) {
        vst1q_s32(out + i * stride, vmovl_s16(in[i]));
    }
}

static AOM_FORCE_INLINE void shift_left_2_s16_x4(const int16x4_t* in, int16x4_t* out, int size) {
    for (int i = 0; i < size; ++i) {
        out[i] = vshl_n_s16(in[i], 2);
    }
}

static AOM_FORCE_INLINE void flip_buf_4_neon(int16x4_t* in, int16x4_t* out, int size) {
    for (int i = 0; i < size; ++i) {
        out[size - i - 1] = in[i];
    }
}

static AOM_FORCE_INLINE void ud_adjust_input_and_stride(int ud_flip, const int16_t** input, uint32_t* stride,
                                                        int out_size) {
    if (ud_flip) {
        *input  = *input + (out_size - 1) * *stride;
        *stride = -*stride;
    }
}

static AOM_FORCE_INLINE int16x4_t round_shift_sqrt2_s16_s16_4x1_neon(int16x4_t a) {
    return vqrshrn_n_s32(vmull_n_s16(a, NEW_SQRT2), NEW_SQRT2_BITS);
}

static AOM_FORCE_INLINE void round_shift_sqrt2_s16_s16_4xn_neon(const int16x4_t* in, int16x4_t* out, int size) {
    for (int i = 0; i < size; ++i) {
        out[i] = round_shift_sqrt2_s16_s16_4x1_neon(in[i]);
    }
}

// ---------------------------------------------------------------------------
// 4x4 1D primitives
// ---------------------------------------------------------------------------

static AOM_FORCE_INLINE void fdct4x4_neon(const int16x4_t* input, int16x4_t* output, int cos_bit) {
    const int16_t*  cospi   = fwd_cospi_q13(cos_bit);
    const int16x4_t cospi16 = vld1_s16(&cospi[4 * 1]);

    int16x4_t in12a = vadd_s16(input[1], input[2]);
    int16x4_t in12s = vsub_s16(input[1], input[2]);
    int16x4_t in03a = vadd_s16(input[0], input[3]);
    int16x4_t in03s = vsub_s16(input[0], input[3]);

    // cospi32 single-product butterfly via vqrdmulh (1 op/output vs widening
    // mull+add+rshrn). Bit-exact: out = (in03a +/- in12a)*cospi32 >> 13, and
    // |in03a|+|in12a| <= ~4080 << 32767 so the int16 sum never saturates.
    const int16x4_t w32 = vdup_n_s16((int16_t)(cospi[4 * 0] * 4)); // Q13 -> Q15
    output[0]           = vqrdmulh_s16(vqadd_s16(in12a, in03a), w32);
    output[2]           = vqrdmulh_s16(vqsub_s16(in03a, in12a), w32);

    // cospi16 two-product butterfly (kept widening: distinct weights).
    int32x4_t u2 = vmull_lane_s16(in12s, cospi16, 1);
    u2           = vmlal_lane_s16(u2, in03s, cospi16, 0);
    int32x4_t u3 = vmull_lane_s16(in03s, cospi16, 1);
    u3           = vmlsl_lane_s16(u3, in12s, cospi16, 0);
    output[1]    = vrshrn_n_s32(u2, TXFM_COS_BIT_MAX);
    output[3]    = vrshrn_n_s32(u3, TXFM_COS_BIT_MAX);
}

static AOM_FORCE_INLINE void fadst4x4_neon(const int16x4_t* input, int16x4_t* output, int cos_bit) {
    int32x4_t       u[6], v[6];
    const int16x4_t sinpi = vld1_s16(fwd_sinpi_q13(cos_bit));
    const int16x4_t u01   = vqadd_s16(input[0], input[1]);

    v[5] = vmull_lane_s16(input[2], sinpi, 2);
    v[0] = vmull_lane_s16(input[1], sinpi, 1);
    v[0] = vmlal_lane_s16(v[0], input[0], sinpi, 0);
    v[1] = vmlal_lane_s16(v[5], input[3], sinpi, 3);
    v[2] = vmull_lane_s16(u01, sinpi, 2);
    v[3] = vmull_lane_s16(input[0], sinpi, 3);
    v[3] = vmlsl_lane_s16(v[3], input[1], sinpi, 0);
    v[4] = vmlsl_lane_s16(v[5], input[3], sinpi, 1);

    u[0] = vaddq_s32(v[0], v[1]);
    u[1] = vmlsl_lane_s16(v[2], input[3], sinpi, 2);
    u[2] = vsubq_s32(v[3], v[4]);
    u[3] = vsubq_s32(u[2], u[0]);
    u[3] = vmlaq_n_s32(u[3], v[5], 3);

    output[0] = vrshrn_n_s32(u[0], TXFM_COS_BIT_MAX);
    output[1] = vrshrn_n_s32(u[1], TXFM_COS_BIT_MAX);
    output[2] = vrshrn_n_s32(u[2], TXFM_COS_BIT_MAX);
    output[3] = vrshrn_n_s32(u[3], TXFM_COS_BIT_MAX);
}

static AOM_FORCE_INLINE void fidentity4x4_neon(const int16x4_t* const input, int16x4_t* const output,
                                               const int cos_bit) {
    (void)cos_bit;
    round_shift_sqrt2_s16_s16_4xn_neon(input, output, 4);
}

// Column wrappers: load 8-bit residual, shift left 2, run 1D.
#define TRANSFORM_COL_4(name)                                                                            \
    static void name##_col_neon(const int16_t* input, int16x4_t* output, uint32_t stride, int cos_bit) { \
        int16x4_t buf0[4];                                                                               \
        load_buffer_s16_x4(input, stride, buf0, 4);                                                      \
        shift_left_2_s16_x4(buf0, buf0, 4);                                                              \
        name##_neon(buf0, output, cos_bit);                                                              \
    }
TRANSFORM_COL_4(fdct4x4)
TRANSFORM_COL_4(fadst4x4)
TRANSFORM_COL_4(fidentity4x4)

// 4x4 forward, 8-bit residual, all tx_types. Mirrors libaom's lowbd 4x4 driver
// but appends a final transpose so the int32 output matches SVT's coefficient
// layout (SVT's 32-bit svt_av1_fwd_txfm2d_4x4_neon transposes before storing).
void svt_lbd_fwd_txfm2d_4x4_neon(int16_t* input, int32_t* output, uint32_t stride, TxType tx_type) {
    int ud_flip, lr_flip;
    get_flip_cfg(tx_type, &ud_flip, &lr_flip);
    const int16_t* in = input;
    ud_adjust_input_and_stride(ud_flip, &in, &stride, 4);

    int16x4_t buf0[4], buf1[4], rbuf[4];

    // Column (vertical) transform: load + shift-left-2 + 1D.
    switch (tx_type) {
    case DCT_DCT:
    case DCT_ADST:
    case DCT_FLIPADST:
    case V_DCT:
        fdct4x4_col_neon(in, buf0, stride, 13);
        break;
    case ADST_DCT:
    case ADST_ADST:
    case FLIPADST_DCT:
    case FLIPADST_FLIPADST:
    case ADST_FLIPADST:
    case FLIPADST_ADST:
    case V_ADST:
    case V_FLIPADST:
        fadst4x4_col_neon(in, buf0, stride, 13);
        break;
    default:
        fidentity4x4_col_neon(in, buf0, stride, 13);
        break; // IDTX, H_DCT, H_ADST, H_FLIPADST
    }

    transpose_arrays_s16_4x4(buf0, buf1);
    if (lr_flip) {
        flip_buf_4_neon(buf1, buf0, 4);
    }
    const int16x4_t* rin = lr_flip ? buf0 : buf1;

    // Row (horizontal) transform.
    switch (tx_type) {
    case DCT_DCT:
    case ADST_DCT:
    case FLIPADST_DCT:
    case H_DCT:
        fdct4x4_neon(rin, rbuf, 13);
        break;
    case DCT_ADST:
    case ADST_ADST:
    case DCT_FLIPADST:
    case FLIPADST_FLIPADST:
    case ADST_FLIPADST:
    case FLIPADST_ADST:
    case H_ADST:
    case H_FLIPADST:
        fadst4x4_neon(rin, rbuf, 13);
        break;
    default:
        fidentity4x4_neon(rin, rbuf, 13);
        break; // IDTX, V_DCT, V_ADST, V_FLIPADST
    }

    transpose_arrays_s16_4x4(rbuf, buf1);
    store_buffer_s16_x4(buf1, output, 4, 4);
}
