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

// ---------------------------------------------------------------------------
// Shared x8 helpers + butterflies (used by 8x8 and larger)
// ---------------------------------------------------------------------------

static AOM_FORCE_INLINE void load_buffer_s16_x8(const int16_t* in, int stride, int16x8_t* out, int n) {
    for (int i = 0; i < n; ++i) {
        out[i] = vld1q_s16(in + i * stride);
    }
}

static AOM_FORCE_INLINE void store_buffer_s16_x8(const int16x8_t* in, int32_t* out, int stride, int n) {
    for (int i = 0; i < n; ++i) {
        vst1q_s32(out + i * stride + 0, vmovl_s16(vget_low_s16(in[i])));
        vst1q_s32(out + i * stride + 4, vmovl_s16(vget_high_s16(in[i])));
    }
}

static AOM_FORCE_INLINE void shift_left_2_s16_x8(const int16x8_t* in, int16x8_t* out, int n) {
    for (int i = 0; i < n; ++i) {
        out[i] = vshlq_n_s16(in[i], 2);
    }
}

static AOM_FORCE_INLINE void shift_left_1_s16_x8(const int16x8_t* in, int16x8_t* out, int n) {
    for (int i = 0; i < n; ++i) {
        out[i] = vaddq_s16(in[i], in[i]);
    }
}

static AOM_FORCE_INLINE void shift_right_1_round_s16_x8(const int16x8_t* in, int16x8_t* out, int n) {
    const int16x8_t zero = vdupq_n_s16(0);
    for (int i = 0; i < n; ++i) {
        out[i] = vrhaddq_s16(in[i], zero);
    }
}

static AOM_FORCE_INLINE void flip_buf_8_neon(int16x8_t* in, int16x8_t* out, int size) {
    for (int i = 0; i < size; ++i) {
        out[size - i - 1] = in[i];
    }
}

// out0 = in0*w[l0] + in1*w[l1]; out1 = in0*w[l2] + in1*w[l3], rounded >> 13.
#define butterfly_s16_s32_x8_neon(wvec, l0, l1, l2, l3, in0, in1, out0, out1)    \
    do {                                                                         \
        int32x4_t u0 = vmull_lane_s16(vget_low_s16(in0), wvec, l0);              \
        u0           = vmlal_lane_s16(u0, vget_low_s16(in1), wvec, l1);          \
        int32x4_t u1 = vmull_lane_s16(vget_high_s16(in0), wvec, l0);             \
        u1           = vmlal_lane_s16(u1, vget_high_s16(in1), wvec, l1);         \
        int32x4_t v0 = vmull_lane_s16(vget_low_s16(in0), wvec, l2);              \
        v0           = vmlal_lane_s16(v0, vget_low_s16(in1), wvec, l3);          \
        int32x4_t v1 = vmull_lane_s16(vget_high_s16(in0), wvec, l2);             \
        v1           = vmlal_lane_s16(v1, vget_high_s16(in1), wvec, l3);         \
        *(out0)      = vcombine_s16(vrshrn_n_s32(u0, 13), vrshrn_n_s32(u1, 13)); \
        *(out1)      = vcombine_s16(vrshrn_n_s32(v0, 13), vrshrn_n_s32(v1, 13)); \
    } while (0)

static AOM_FORCE_INLINE void butterfly_0112_x8(const int16x4_t w, const int16x8_t in0, const int16x8_t in1,
                                               int16x8_t* out0, int16x8_t* out1) {
    butterfly_s16_s32_x8_neon(w, 0, 1, 1, 2, in0, in1, out0, out1);
}

static AOM_FORCE_INLINE void butterfly_1003_x8(const int16x4_t w, const int16x8_t in0, const int16x8_t in1,
                                               int16x8_t* out0, int16x8_t* out1) {
    butterfly_s16_s32_x8_neon(w, 1, 0, 0, 3, in0, in1, out0, out1);
}

static AOM_FORCE_INLINE void butterfly_1223_x8(const int16x4_t w, const int16x8_t in0, const int16x8_t in1,
                                               int16x8_t* out0, int16x8_t* out1) {
    butterfly_s16_s32_x8_neon(w, 1, 2, 2, 3, in0, in1, out0, out1);
}

// cospi32 single-product butterflies via vqrdmulh (bit-exact when |in0|+|in1|
// <= 32767). w must be cospi32_q13 << 2 (i.e. Q15). Two output patterns:
//   0112: out0 = (in0+in1)*w,  out1 = (in0-in1)*w
//   0332: out0 = (in0-in1)*w,  out1 = -(in0+in1)*w
static AOM_FORCE_INLINE void btf_cospi32_0112_x8(const int16x8_t w, const int16x8_t in0, const int16x8_t in1,
                                                 int16x8_t* out0, int16x8_t* out1) {
    *out0 = vqrdmulhq_s16(vqaddq_s16(in0, in1), w);
    *out1 = vqrdmulhq_s16(vqsubq_s16(in0, in1), w);
}

static AOM_FORCE_INLINE void btf_cospi32_0332_x8(const int16x8_t w, const int16x8_t in0, const int16x8_t in1,
                                                 int16x8_t* out0, int16x8_t* out1) {
    *out0 = vqrdmulhq_s16(vqsubq_s16(in0, in1), w);
    *out1 = vqrdmulhq_s16(vqaddq_s16(in0, in1), vnegq_s16(w));
}

static AOM_FORCE_INLINE void butterfly_dct_pre_s16_x8(const int16x8_t* input, int16x8_t* output, int n) {
    for (int i = 0; i < n / 2; ++i) {
        output[i] = vqaddq_s16(input[i], input[n - i - 1]);
    }
    for (int i = 0; i < n / 2; ++i) {
        output[n / 2 + i] = vqsubq_s16(input[n / 2 - i - 1], input[n / 2 + i]);
    }
}

static AOM_FORCE_INLINE void butterfly_dct_post_s16_x8(const int16x8_t* in0, const int16x8_t* in1, int16x8_t* output,
                                                       int n) {
    for (int i = 0; i < n / 4; ++i) {
        output[i] = vqaddq_s16(in0[i], in1[n / 2 - i - 1]);
    }
    for (int i = 0; i < n / 4; ++i) {
        output[n / 4 + i] = vqsubq_s16(in0[n / 4 - i - 1], in1[n / 4 + i]);
    }
    for (int i = 0; i < n / 4; ++i) {
        output[n / 2 + i] = vqsubq_s16(in0[n - i - 1], in1[n / 2 + i]);
    }
    for (int i = 0; i < n / 4; ++i) {
        output[(3 * n) / 4 + i] = vqaddq_s16(in0[(3 * n) / 4 + i], in1[(3 * n) / 4 - i - 1]);
    }
}

// ---------------------------------------------------------------------------
// 8x8 1D primitives (cospi32 butterflies use vqrdmulh)
// ---------------------------------------------------------------------------

static AOM_FORCE_INLINE void fdct8x8_neon(const int16x8_t* input, int16x8_t* output, int cos_bit) {
    const int16_t*  cospi      = fwd_cospi_q13(cos_bit);
    const int16x8_t cospi32_16 = vld1q_s16(&cospi[4 * 0]);
    const int16x8_t cospi8_24  = vld1q_s16(&cospi[4 * 2]);
    const int16x4_t cospi16    = vget_high_s16(cospi32_16);
    const int16x4_t cospi8     = vget_low_s16(cospi8_24);
    const int16x4_t cospi24    = vget_high_s16(cospi8_24);
    const int16x8_t w32        = vdupq_n_s16((int16_t)(cospi[4 * 0] * 4));

    int16x8_t x1[8];
    butterfly_dct_pre_s16_x8(input, x1, 8);
    int16x8_t x2[8];
    butterfly_dct_pre_s16_x8(x1, x2, 4);
    btf_cospi32_0112_x8(w32, x1[6], x1[5], &x2[6], &x2[5]);

    int16x8_t x3[8];
    btf_cospi32_0112_x8(w32, x2[0], x2[1], &output[0], &output[4]);
    butterfly_0112_x8(cospi16, x2[3], x2[2], &output[2], &output[6]);
    butterfly_dct_post_s16_x8(x1 + 4, x2 + 4, x3 + 4, 4);

    butterfly_0112_x8(cospi8, x3[7], x3[4], &output[1], &output[7]);
    butterfly_1003_x8(cospi24, x3[6], x3[5], &output[5], &output[3]);
}

static AOM_FORCE_INLINE void fadst8x8_neon(const int16x8_t* input, int16x8_t* output, int cos_bit) {
    const int16_t*  cospi      = fwd_cospi_q13(cos_bit);
    const int16x8_t cospi32_16 = vld1q_s16(&cospi[4 * 0]);
    const int16x8_t cospi4_12  = vld1q_s16(&cospi[4 * 4]);
    const int16x8_t cospi20_28 = vld1q_s16(&cospi[4 * 6]);
    const int16x4_t cospi16    = vget_high_s16(cospi32_16);
    const int16x4_t cospi4     = vget_low_s16(cospi4_12);
    const int16x4_t cospi12    = vget_high_s16(cospi4_12);
    const int16x4_t cospi20    = vget_low_s16(cospi20_28);
    const int16x4_t cospi28    = vget_high_s16(cospi20_28);
    const int16x8_t w32        = vdupq_n_s16((int16_t)(cospi[4 * 0] * 4));

    int16x8_t x2[8];
    btf_cospi32_0332_x8(w32, input[4], input[3], &x2[2], &x2[3]);
    btf_cospi32_0112_x8(w32, input[2], input[5], &x2[7], &x2[6]);

    int16x8_t x3[8];
    x3[0] = vqaddq_s16(input[0], x2[2]);
    x3[1] = vqsubq_s16(x2[3], input[7]);
    x3[2] = vqsubq_s16(input[0], x2[2]);
    x3[3] = vqaddq_s16(input[7], x2[3]);
    x3[4] = vqsubq_s16(x2[6], input[1]);
    x3[5] = vqaddq_s16(input[6], x2[7]);
    x3[6] = vqaddq_s16(input[1], x2[6]);
    x3[7] = vqsubq_s16(input[6], x2[7]);

    butterfly_0112_x8(cospi16, x3[4], x3[5], &x3[4], &x3[5]);
    butterfly_0112_x8(cospi16, x3[7], x3[6], &x3[6], &x3[7]);

    int16x8_t x5[8];
    x5[0] = vqaddq_s16(x3[0], x3[4]);
    x5[1] = vqaddq_s16(x3[1], x3[5]);
    x5[2] = vqaddq_s16(x3[2], x3[6]);
    x5[3] = vqsubq_s16(x3[7], x3[3]);
    x5[4] = vqsubq_s16(x3[0], x3[4]);
    x5[5] = vqsubq_s16(x3[1], x3[5]);
    x5[6] = vqsubq_s16(x3[2], x3[6]);
    x5[7] = vqaddq_s16(x3[3], x3[7]);

    butterfly_0112_x8(cospi4, x5[0], x5[1], &output[7], &output[0]);
    butterfly_0112_x8(cospi20, x5[2], x5[3], &output[5], &output[2]);
    butterfly_1003_x8(cospi28, x5[4], x5[5], &output[3], &output[4]);
    butterfly_0112_x8(cospi12, x5[6], x5[7], &output[6], &output[1]);
}

static AOM_FORCE_INLINE void fidentity8x8_neon(const int16x8_t* input, int16x8_t* output, int cos_bit) {
    (void)cos_bit;
    shift_left_1_s16_x8(input, output, 8);
}

#define TRANSFORM_COL_8(name)                                                                            \
    static void name##_col_neon(const int16_t* input, int16x8_t* output, uint32_t stride, int cos_bit) { \
        int16x8_t buf0[8];                                                                               \
        load_buffer_s16_x8(input, stride, buf0, 8);                                                      \
        shift_left_2_s16_x8(buf0, buf0, 8);                                                              \
        name##_neon(buf0, output, cos_bit);                                                              \
    }
TRANSFORM_COL_8(fdct8x8)
TRANSFORM_COL_8(fadst8x8)
TRANSFORM_COL_8(fidentity8x8)

// 8x8 forward, 8-bit residual, all tx_types. Final transpose matches SVT layout.
void svt_lbd_fwd_txfm2d_8x8_neon(int16_t* input, int32_t* output, uint32_t stride, TxType tx_type) {
    int ud_flip, lr_flip;
    get_flip_cfg(tx_type, &ud_flip, &lr_flip);
    const int16_t* in = input;
    ud_adjust_input_and_stride(ud_flip, &in, &stride, 8);

    int16x8_t buf0[8], buf1[8], rbuf[8];

    switch (tx_type) {
    case DCT_DCT:
    case DCT_ADST:
    case DCT_FLIPADST:
    case V_DCT:
        fdct8x8_col_neon(in, buf0, stride, 13);
        break;
    case ADST_DCT:
    case ADST_ADST:
    case FLIPADST_DCT:
    case FLIPADST_FLIPADST:
    case ADST_FLIPADST:
    case FLIPADST_ADST:
    case V_ADST:
    case V_FLIPADST:
        fadst8x8_col_neon(in, buf0, stride, 13);
        break;
    default:
        fidentity8x8_col_neon(in, buf0, stride, 13);
        break;
    }

    shift_right_1_round_s16_x8(buf0, buf0, 8);
    transpose_arrays_s16_8x8(buf0, buf1);
    if (lr_flip) {
        flip_buf_8_neon(buf1, buf0, 8);
    }
    const int16x8_t* rin = lr_flip ? buf0 : buf1;

    switch (tx_type) {
    case DCT_DCT:
    case ADST_DCT:
    case FLIPADST_DCT:
    case H_DCT:
        fdct8x8_neon(rin, rbuf, 13);
        break;
    case DCT_ADST:
    case ADST_ADST:
    case DCT_FLIPADST:
    case FLIPADST_FLIPADST:
    case ADST_FLIPADST:
    case FLIPADST_ADST:
    case H_ADST:
    case H_FLIPADST:
        fadst8x8_neon(rin, rbuf, 13);
        break;
    default:
        fidentity8x8_neon(rin, rbuf, 13);
        break;
    }

    transpose_arrays_s16_8x8(rbuf, buf1);
    store_buffer_s16_x8(buf1, output, 8, 8);
}

// ---------------------------------------------------------------------------
// 16x16 1D primitives + driver
// ---------------------------------------------------------------------------

static AOM_FORCE_INLINE void butterfly_0332_x8(const int16x4_t w, const int16x8_t in0, const int16x8_t in1,
                                               int16x8_t* out0, int16x8_t* out1) {
    butterfly_s16_s32_x8_neon(w, 0, 3, 3, 2, in0, in1, out0, out1);
}

static AOM_FORCE_INLINE void shift_right_2_round_s16_x8(const int16x8_t* in, int16x8_t* out, int n) {
    for (int i = 0; i < n; ++i) {
        out[i] = vrshrq_n_s16(in[i], 2);
    }
}

static AOM_FORCE_INLINE int16x8_t round_shift_2sqrt2_x8(int16x8_t a) {
    int16x4_t lo = vqrshrn_n_s32(vmull_n_s16(vget_low_s16(a), 2 * NEW_SQRT2), NEW_SQRT2_BITS);
    int16x4_t hi = vqrshrn_n_s32(vmull_n_s16(vget_high_s16(a), 2 * NEW_SQRT2), NEW_SQRT2_BITS);
    return vcombine_s16(lo, hi);
}

static AOM_FORCE_INLINE void fdct16x16_neon(const int16x8_t* input, int16x8_t* output, int cos_bit) {
    const int16_t*  cospi      = fwd_cospi_q13(cos_bit);
    const int16x8_t cospi32_16 = vld1q_s16(&cospi[4 * 0]);
    const int16x8_t cospi8_24  = vld1q_s16(&cospi[4 * 2]);
    const int16x8_t cospi4_12  = vld1q_s16(&cospi[4 * 4]);
    const int16x8_t cospi20_28 = vld1q_s16(&cospi[4 * 6]);
    const int16x4_t cospi32    = vget_low_s16(cospi32_16);
    const int16x4_t cospi16    = vget_high_s16(cospi32_16);
    const int16x4_t cospi8     = vget_low_s16(cospi8_24);
    const int16x4_t cospi24    = vget_high_s16(cospi8_24);
    const int16x4_t cospi4     = vget_low_s16(cospi4_12);
    const int16x4_t cospi12    = vget_high_s16(cospi4_12);
    const int16x4_t cospi20    = vget_low_s16(cospi20_28);
    const int16x4_t cospi28    = vget_high_s16(cospi20_28);
    const int16x8_t w32        = vdupq_n_s16((int16_t)(cospi[4 * 0] * 4));

    int16x8_t x1[16];
    butterfly_dct_pre_s16_x8(input, x1, 16);
    int16x8_t x2[16];
    butterfly_dct_pre_s16_x8(x1, x2, 8);
    btf_cospi32_0112_x8(w32, x1[13], x1[10], &x2[13], &x2[10]);
    btf_cospi32_0112_x8(w32, x1[12], x1[11], &x2[12], &x2[11]);

    int16x8_t x3[16];
    butterfly_dct_pre_s16_x8(x2, x3, 4);
    btf_cospi32_0112_x8(w32, x2[6], x2[5], &x3[6], &x3[5]);
    butterfly_dct_post_s16_x8(x1 + 8, x2 + 8, x3 + 8, 8);

    int16x8_t x4[16];
    // stage-4 cospi32 stays widening: x3[0]+x3[1] is a deep sum that can exceed
    // int16 in the cos_bit-12 row pass (proven by FwdTxfm2dAsmTest).
    butterfly_0112_x8(cospi32, x3[0], x3[1], &output[0], &output[8]);
    butterfly_0112_x8(cospi16, x3[3], x3[2], &output[4], &output[12]);
    butterfly_dct_post_s16_x8(x2 + 4, x3 + 4, x4 + 4, 4);
    butterfly_0112_x8(cospi16, x3[14], x3[9], &x4[14], &x4[9]);
    butterfly_1223_x8(cospi16, x3[13], x3[10], &x4[13], &x4[10]);

    int16x8_t x5[16];
    butterfly_0112_x8(cospi8, x4[7], x4[4], &output[2], &output[14]);
    butterfly_1003_x8(cospi24, x4[6], x4[5], &output[10], &output[6]);
    butterfly_dct_post_s16_x8(x3 + 8, x4 + 8, x5 + 8, 4);
    butterfly_dct_post_s16_x8(x3 + 12, x4 + 12, x5 + 12, 4);

    butterfly_0112_x8(cospi4, x5[15], x5[8], &output[1], &output[15]);
    butterfly_1003_x8(cospi28, x5[14], x5[9], &output[9], &output[7]);
    butterfly_0112_x8(cospi20, x5[13], x5[10], &output[5], &output[11]);
    butterfly_1003_x8(cospi12, x5[12], x5[11], &output[13], &output[3]);
}

static AOM_FORCE_INLINE void fadst16x16_neon(const int16x8_t* input, int16x8_t* output, int cos_bit) {
    const int16_t*  cospi      = fwd_cospi_q13(cos_bit);
    const int16x8_t cospi32_16 = vld1q_s16(&cospi[4 * 0]);
    const int16x8_t cospi8_24  = vld1q_s16(&cospi[4 * 2]);
    const int16x8_t cospi2_6   = vld1q_s16(&cospi[4 * 8]);
    const int16x8_t cospi10_14 = vld1q_s16(&cospi[4 * 10]);
    const int16x8_t cospi18_22 = vld1q_s16(&cospi[4 * 12]);
    const int16x8_t cospi26_30 = vld1q_s16(&cospi[4 * 14]);
    const int16x4_t cospi16    = vget_high_s16(cospi32_16);
    const int16x4_t cospi8     = vget_low_s16(cospi8_24);
    const int16x4_t cospi24    = vget_high_s16(cospi8_24);
    const int16x4_t cospi2     = vget_low_s16(cospi2_6);
    const int16x4_t cospi6     = vget_high_s16(cospi2_6);
    const int16x4_t cospi10    = vget_low_s16(cospi10_14);
    const int16x4_t cospi14    = vget_high_s16(cospi10_14);
    const int16x4_t cospi18    = vget_low_s16(cospi18_22);
    const int16x4_t cospi22    = vget_high_s16(cospi18_22);
    const int16x4_t cospi26    = vget_low_s16(cospi26_30);
    const int16x4_t cospi30    = vget_high_s16(cospi26_30);
    const int16x8_t w32        = vdupq_n_s16((int16_t)(cospi[4 * 0] * 4));

    // stage 2: cospi32 single-product butterflies (safe: operate on raw coeffs).
    int16x8_t x2[8];
    btf_cospi32_0332_x8(w32, input[8], input[7], &x2[0], &x2[1]);
    btf_cospi32_0112_x8(w32, input[4], input[11], &x2[3], &x2[2]);
    btf_cospi32_0112_x8(w32, input[6], input[9], &x2[5], &x2[4]);
    btf_cospi32_0332_x8(w32, input[10], input[5], &x2[6], &x2[7]);

    int16x8_t x3[16];
    x3[0]  = vqaddq_s16(input[0], x2[0]);
    x3[1]  = vqsubq_s16(x2[1], input[15]);
    x3[2]  = vqsubq_s16(input[0], x2[0]);
    x3[3]  = vqaddq_s16(input[15], x2[1]);
    x3[4]  = vqsubq_s16(x2[2], input[3]);
    x3[5]  = vqaddq_s16(input[12], x2[3]);
    x3[6]  = vqaddq_s16(input[3], x2[2]);
    x3[7]  = vqsubq_s16(input[12], x2[3]);
    x3[8]  = vqsubq_s16(x2[4], input[1]);
    x3[9]  = vqaddq_s16(input[14], x2[5]);
    x3[10] = vqaddq_s16(input[1], x2[4]);
    x3[11] = vqsubq_s16(input[14], x2[5]);
    x3[12] = vqaddq_s16(input[2], x2[6]);
    x3[13] = vqsubq_s16(x2[7], input[13]);
    x3[14] = vqsubq_s16(input[2], x2[6]);
    x3[15] = vqaddq_s16(input[13], x2[7]);

    // stage 4: cospi16 two-product (widening).
    butterfly_0112_x8(cospi16, x3[4], x3[5], &x3[4], &x3[5]);
    butterfly_0112_x8(cospi16, x3[7], x3[6], &x3[6], &x3[7]);
    butterfly_0112_x8(cospi16, x3[12], x3[13], &x3[12], &x3[13]);
    butterfly_0332_x8(cospi16, x3[14], x3[15], &x3[15], &x3[14]);

    int16x8_t x5[16];
    x5[0]  = vqaddq_s16(x3[0], x3[4]);
    x5[1]  = vqaddq_s16(x3[1], x3[5]);
    x5[2]  = vqaddq_s16(x3[2], x3[6]);
    x5[3]  = vqsubq_s16(x3[7], x3[3]);
    x5[4]  = vqsubq_s16(x3[0], x3[4]);
    x5[5]  = vqsubq_s16(x3[1], x3[5]);
    x5[6]  = vqsubq_s16(x3[2], x3[6]);
    x5[7]  = vqaddq_s16(x3[3], x3[7]);
    x5[8]  = vqaddq_s16(x3[8], x3[12]);
    x5[9]  = vqaddq_s16(x3[9], x3[13]);
    x5[10] = vqsubq_s16(x3[14], x3[10]);
    x5[11] = vqaddq_s16(x3[11], x3[15]);
    x5[12] = vqsubq_s16(x3[8], x3[12]);
    x5[13] = vqsubq_s16(x3[9], x3[13]);
    x5[14] = vqaddq_s16(x3[10], x3[14]);
    x5[15] = vqsubq_s16(x3[11], x3[15]);

    // stage 6: cospi8/24 two-product (widening).
    butterfly_0112_x8(cospi8, x5[8], x5[9], &x5[8], &x5[9]);
    butterfly_1003_x8(cospi24, x5[10], x5[11], &x5[10], &x5[11]);
    butterfly_1003_x8(cospi8, x5[13], x5[12], &x5[13], &x5[12]);
    butterfly_1003_x8(cospi24, x5[15], x5[14], &x5[14], &x5[15]);

    int16x8_t x7[16];
    x7[0]  = vqaddq_s16(x5[0], x5[8]);
    x7[1]  = vqaddq_s16(x5[1], x5[9]);
    x7[2]  = vqaddq_s16(x5[2], x5[10]);
    x7[3]  = vqaddq_s16(x5[3], x5[11]);
    x7[4]  = vqaddq_s16(x5[4], x5[12]);
    x7[5]  = vqaddq_s16(x5[5], x5[13]);
    x7[6]  = vqaddq_s16(x5[6], x5[14]);
    x7[7]  = vqsubq_s16(x5[15], x5[7]);
    x7[8]  = vqsubq_s16(x5[0], x5[8]);
    x7[9]  = vqsubq_s16(x5[1], x5[9]);
    x7[10] = vqsubq_s16(x5[2], x5[10]);
    x7[11] = vqsubq_s16(x5[3], x5[11]);
    x7[12] = vqsubq_s16(x5[4], x5[12]);
    x7[13] = vqsubq_s16(x5[5], x5[13]);
    x7[14] = vqsubq_s16(x5[6], x5[14]);
    x7[15] = vqaddq_s16(x5[7], x5[15]);

    // stage 8: cospi2/6/10/14/18/22/26/30 two-product (widening).
    butterfly_0112_x8(cospi2, x7[0], x7[1], &output[15], &output[0]);
    butterfly_0112_x8(cospi10, x7[2], x7[3], &output[13], &output[2]);
    butterfly_0112_x8(cospi18, x7[4], x7[5], &output[11], &output[4]);
    butterfly_0112_x8(cospi26, x7[6], x7[7], &output[9], &output[6]);
    butterfly_1003_x8(cospi30, x7[8], x7[9], &output[7], &output[8]);
    butterfly_1003_x8(cospi22, x7[10], x7[11], &output[5], &output[10]);
    butterfly_1003_x8(cospi14, x7[12], x7[13], &output[3], &output[12]);
    butterfly_0112_x8(cospi6, x7[14], x7[15], &output[14], &output[1]);
}

static AOM_FORCE_INLINE void fidentity16x16_neon(const int16x8_t* input, int16x8_t* output, int cos_bit) {
    (void)cos_bit;
    for (int i = 0; i < 16; ++i) {
        output[i] = round_shift_2sqrt2_x8(input[i]);
    }
}

#define TRANSFORM_COL_16(name)                                                                           \
    static void name##_col_neon(const int16_t* input, int16x8_t* output, uint32_t stride, int cos_bit) { \
        int16x8_t buf0[16];                                                                              \
        load_buffer_s16_x8(input, stride, buf0, 16);                                                     \
        shift_left_2_s16_x8(buf0, buf0, 16);                                                             \
        name##_neon(buf0, output, cos_bit);                                                              \
    }
TRANSFORM_COL_16(fdct16x16)
TRANSFORM_COL_16(fadst16x16)
TRANSFORM_COL_16(fidentity16x16)

// 16x16 forward, 8-bit residual, all tx_types. Final transpose -> SVT layout.
void svt_lbd_fwd_txfm2d_16x16_neon(int16_t* input, int32_t* output, uint32_t stride, TxType tx_type) {
    int ud_flip, lr_flip;
    get_flip_cfg(tx_type, &ud_flip, &lr_flip);
    const int16_t* in = input;
    ud_adjust_input_and_stride(ud_flip, &in, &stride, 16);

    int16x8_t buf0[16], buf1[32];
    for (int i = 0; i < 2; i++) {
        switch (tx_type) {
        case DCT_DCT:
        case DCT_ADST:
        case DCT_FLIPADST:
        case V_DCT:
            fdct16x16_col_neon(in + 8 * i, buf0, stride, 13);
            break;
        case ADST_DCT:
        case ADST_ADST:
        case FLIPADST_DCT:
        case FLIPADST_FLIPADST:
        case ADST_FLIPADST:
        case FLIPADST_ADST:
        case V_ADST:
        case V_FLIPADST:
            fadst16x16_col_neon(in + 8 * i, buf0, stride, 13);
            break;
        default:
            fidentity16x16_col_neon(in + 8 * i, buf0, stride, 13);
            break;
        }
        shift_right_2_round_s16_x8(buf0, buf0, 16);
        transpose_arrays_s16_8x8(buf0 + 0, buf1 + 0 * 16 + 8 * i);
        transpose_arrays_s16_8x8(buf0 + 8, buf1 + 1 * 16 + 8 * i);
    }

    for (int i = 0; i < 2; i++) {
        int16x8_t rin[16];
        if (lr_flip) {
            flip_buf_8_neon(buf1 + 16 * i, rin, 16);
        }
        const int16x8_t* row_in = lr_flip ? rin : (buf1 + 16 * i);

        int16x8_t rowout[16];
        switch (tx_type) {
        case DCT_DCT:
        case ADST_DCT:
        case FLIPADST_DCT:
        case H_DCT:
            fdct16x16_neon(row_in, rowout, 12);
            break;
        case DCT_ADST:
        case ADST_ADST:
        case DCT_FLIPADST:
        case FLIPADST_FLIPADST:
        case ADST_FLIPADST:
        case FLIPADST_ADST:
        case H_ADST:
        case H_FLIPADST:
            fadst16x16_neon(row_in, rowout, 12);
            break;
        default:
            fidentity16x16_neon(row_in, rowout, 12);
            break;
        }

        int16x8_t ta[8], tb[8];
        transpose_arrays_s16_8x8(rowout + 0, ta);
        transpose_arrays_s16_8x8(rowout + 8, tb);
        for (int r = 0; r < 8; r++) {
            int32_t* o = output + (8 * i + r) * 16;
            vst1q_s32(o + 0, vmovl_s16(vget_low_s16(ta[r])));
            vst1q_s32(o + 4, vmovl_s16(vget_high_s16(ta[r])));
            vst1q_s32(o + 8, vmovl_s16(vget_low_s16(tb[r])));
            vst1q_s32(o + 12, vmovl_s16(vget_high_s16(tb[r])));
        }
    }
}

// ---------------------------------------------------------------------------
// 32x32 1D primitives + driver (DCT/IDTX only; no ADST at size >= 32)
// ---------------------------------------------------------------------------

static AOM_FORCE_INLINE void shift_right_4_round_s16_x8(const int16x8_t* in, int16x8_t* out, int n) {
    for (int i = 0; i < n; ++i) {
        out[i] = vrshrq_n_s16(in[i], 4);
    }
}

static AOM_FORCE_INLINE void fdct32x32_neon(const int16x8_t* input, int16x8_t* output, int cos_bit) {
    const int16_t*  cospi      = fwd_cospi_q13(cos_bit);
    const int16x8_t cospi32_16 = vld1q_s16(&cospi[4 * 0]);
    const int16x8_t cospi8_24  = vld1q_s16(&cospi[4 * 2]);
    const int16x8_t cospi4_12  = vld1q_s16(&cospi[4 * 4]);
    const int16x8_t cospi20_28 = vld1q_s16(&cospi[4 * 6]);
    const int16x8_t cospi2_6   = vld1q_s16(&cospi[4 * 8]);
    const int16x8_t cospi10_14 = vld1q_s16(&cospi[4 * 10]);
    const int16x8_t cospi18_22 = vld1q_s16(&cospi[4 * 12]);
    const int16x8_t cospi26_30 = vld1q_s16(&cospi[4 * 14]);
    const int16x4_t cospi16    = vget_high_s16(cospi32_16);
    const int16x4_t cospi8     = vget_low_s16(cospi8_24);
    const int16x4_t cospi24    = vget_high_s16(cospi8_24);
    const int16x4_t cospi4     = vget_low_s16(cospi4_12);
    const int16x4_t cospi12    = vget_high_s16(cospi4_12);
    const int16x4_t cospi20    = vget_low_s16(cospi20_28);
    const int16x4_t cospi28    = vget_high_s16(cospi20_28);
    const int16x4_t cospi2     = vget_low_s16(cospi2_6);
    const int16x4_t cospi6     = vget_high_s16(cospi2_6);
    const int16x4_t cospi10    = vget_low_s16(cospi10_14);
    const int16x4_t cospi14    = vget_high_s16(cospi10_14);
    const int16x4_t cospi18    = vget_low_s16(cospi18_22);
    const int16x4_t cospi22    = vget_high_s16(cospi18_22);
    const int16x4_t cospi26    = vget_low_s16(cospi26_30);
    const int16x4_t cospi30    = vget_high_s16(cospi26_30);
    const int16x4_t cospi32    = vget_low_s16(cospi32_16);
    const int16x8_t w32        = vdupq_n_s16((int16_t)(cospi[4 * 0] * 4));

    int16x8_t x1[32];
    butterfly_dct_pre_s16_x8(input, x1, 32);
    // stage 2 cospi32: sum <= 4*A, safe -> vqrdmulh.
    int16x8_t x2[32];
    butterfly_dct_pre_s16_x8(x1, x2, 16);
    btf_cospi32_0112_x8(w32, x1[27], x1[20], &x2[27], &x2[20]);
    btf_cospi32_0112_x8(w32, x1[26], x1[21], &x2[26], &x2[21]);
    btf_cospi32_0112_x8(w32, x1[25], x1[22], &x2[25], &x2[22]);
    btf_cospi32_0112_x8(w32, x1[24], x1[23], &x2[24], &x2[23]);
    // stage 3 cospi32: sum <= 8*A, safe.
    int16x8_t x3[32];
    butterfly_dct_pre_s16_x8(x2, x3, 8);
    btf_cospi32_0112_x8(w32, x2[13], x2[10], &x3[13], &x3[10]);
    btf_cospi32_0112_x8(w32, x2[12], x2[11], &x3[12], &x3[11]);
    butterfly_dct_post_s16_x8(x1 + 16, x2 + 16, x3 + 16, 16);
    // stage 4 cospi32: sum <= 16*A_row (~23k), safe.
    int16x8_t x4[32];
    butterfly_dct_pre_s16_x8(x3, x4, 4);
    btf_cospi32_0112_x8(w32, x3[6], x3[5], &x4[6], &x4[5]);
    butterfly_dct_post_s16_x8(x2 + 8, x3 + 8, x4 + 8, 8);
    butterfly_0112_x8(cospi16, x3[29], x3[18], &x4[29], &x4[18]);
    butterfly_0112_x8(cospi16, x3[28], x3[19], &x4[28], &x4[19]);
    butterfly_1223_x8(cospi16, x3[27], x3[20], &x4[27], &x4[20]);
    butterfly_1223_x8(cospi16, x3[26], x3[21], &x4[26], &x4[21]);
    // stage 5 cospi32 (DC): sum <= 32*A_row (~46k) -> widening (overflow risk).
    int16x8_t x5[32];
    butterfly_0112_x8(cospi32, x4[0], x4[1], &output[0], &output[16]);
    butterfly_0112_x8(cospi16, x4[3], x4[2], &output[8], &output[24]);
    butterfly_dct_post_s16_x8(x3 + 4, x4 + 4, x5 + 4, 4);
    butterfly_0112_x8(cospi16, x4[14], x4[9], &x5[14], &x5[9]);
    butterfly_1223_x8(cospi16, x4[13], x4[10], &x5[13], &x5[10]);
    butterfly_dct_post_s16_x8(x3 + 16, x4 + 16, x5 + 16, 8);
    butterfly_dct_post_s16_x8(x3 + 24, x4 + 24, x5 + 24, 8);
    // stage 6
    int16x8_t x6[32];
    butterfly_0112_x8(cospi8, x5[7], x5[4], &output[4], &output[28]);
    butterfly_1003_x8(cospi24, x5[6], x5[5], &output[20], &output[12]);
    butterfly_dct_post_s16_x8(x4 + 8, x5 + 8, x6 + 8, 4);
    butterfly_dct_post_s16_x8(x4 + 12, x5 + 12, x6 + 12, 4);
    butterfly_0112_x8(cospi8, x5[30], x5[17], &x6[30], &x6[17]);
    butterfly_1223_x8(cospi8, x5[29], x5[18], &x6[29], &x6[18]);
    butterfly_1003_x8(cospi24, x5[26], x5[21], &x6[26], &x6[21]);
    butterfly_0332_x8(cospi24, x5[25], x5[22], &x6[25], &x6[22]);
    // stage 7
    int16x8_t x7[32];
    butterfly_0112_x8(cospi4, x6[15], x6[8], &output[2], &output[30]);
    butterfly_1003_x8(cospi28, x6[14], x6[9], &output[18], &output[14]);
    butterfly_0112_x8(cospi20, x6[13], x6[10], &output[10], &output[22]);
    butterfly_1003_x8(cospi12, x6[12], x6[11], &output[26], &output[6]);
    butterfly_dct_post_s16_x8(x5 + 16, x6 + 16, x7 + 16, 4);
    butterfly_dct_post_s16_x8(x5 + 20, x6 + 20, x7 + 20, 4);
    butterfly_dct_post_s16_x8(x5 + 24, x6 + 24, x7 + 24, 4);
    butterfly_dct_post_s16_x8(x5 + 28, x6 + 28, x7 + 28, 4);
    // stage 8
    butterfly_0112_x8(cospi2, x7[31], x7[16], &output[1], &output[31]);
    butterfly_1003_x8(cospi30, x7[30], x7[17], &output[17], &output[15]);
    butterfly_0112_x8(cospi18, x7[29], x7[18], &output[9], &output[23]);
    butterfly_1003_x8(cospi14, x7[28], x7[19], &output[25], &output[7]);
    butterfly_0112_x8(cospi10, x7[27], x7[20], &output[5], &output[27]);
    butterfly_1003_x8(cospi22, x7[26], x7[21], &output[21], &output[11]);
    butterfly_0112_x8(cospi26, x7[25], x7[22], &output[13], &output[19]);
    butterfly_1003_x8(cospi6, x7[24], x7[23], &output[29], &output[3]);
}

static AOM_FORCE_INLINE void fidentity32x32_neon(const int16x8_t* input, int16x8_t* output, int cos_bit) {
    (void)cos_bit;
    shift_left_2_s16_x8(input, output, 32);
}

#define TRANSFORM_COL_32(name)                                                                           \
    static void name##_col_neon(const int16_t* input, int16x8_t* output, uint32_t stride, int cos_bit) { \
        int16x8_t buf0[32];                                                                              \
        load_buffer_s16_x8(input, stride, buf0, 32);                                                     \
        shift_left_2_s16_x8(buf0, buf0, 32);                                                             \
        name##_neon(buf0, output, cos_bit);                                                              \
    }
TRANSFORM_COL_32(fdct32x32)
TRANSFORM_COL_32(fidentity32x32)

// 32x32 forward, 8-bit residual. DCT_DCT/IDTX/V_DCT/H_DCT only. cos_bit 12 both
// passes; shift_left_2 in, shift_right_4 between. Final transpose -> SVT layout.
void svt_lbd_fwd_txfm2d_32x32_neon(int16_t* input, int32_t* output, uint32_t stride, TxType tx_type) {
    int ud_flip, lr_flip;
    get_flip_cfg(tx_type, &ud_flip, &lr_flip);
    const int16_t* in = input;
    ud_adjust_input_and_stride(ud_flip, &in, &stride, 32);

    int16x8_t buf0[32], buf1[128];
    for (int i = 0; i < 4; i++) {
        switch (tx_type) {
        case DCT_DCT:
        case V_DCT:
            fdct32x32_col_neon(in + 8 * i, buf0, stride, 12);
            break;
        default:
            fidentity32x32_col_neon(in + 8 * i, buf0, stride, 12);
            break;
        }
        shift_right_4_round_s16_x8(buf0, buf0, 32);
        for (int j = 0; j < 4; j++) {
            transpose_arrays_s16_8x8(buf0 + j * 8, buf1 + j * 32 + 8 * i);
        }
    }

    for (int i = 0; i < 4; i++) {
        int16x8_t rin[32];
        if (lr_flip) {
            flip_buf_8_neon(buf1 + 32 * i, rin, 32);
        }
        const int16x8_t* row_in = lr_flip ? rin : (buf1 + 32 * i);

        int16x8_t rowout[32];
        switch (tx_type) {
        case DCT_DCT:
        case H_DCT:
            fdct32x32_neon(row_in, rowout, 12);
            break;
        default:
            fidentity32x32_neon(row_in, rowout, 12);
            break;
        }

        int16x8_t t[4][8];
        for (int j = 0; j < 4; j++) {
            transpose_arrays_s16_8x8(rowout + j * 8, t[j]);
        }
        for (int r = 0; r < 8; r++) {
            int32_t* o = output + (8 * i + r) * 32;
            for (int j = 0; j < 4; j++) {
                vst1q_s32(o + j * 8 + 0, vmovl_s16(vget_low_s16(t[j][r])));
                vst1q_s32(o + j * 8 + 4, vmovl_s16(vget_high_s16(t[j][r])));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Rectangular sizes (reuse the 1D kernels + sqrt2 rect scaling)
// ---------------------------------------------------------------------------

static AOM_FORCE_INLINE int16x8_t round_shift_sqrt2_x8(int16x8_t a) {
    int16x4_t lo = vqrshrn_n_s32(vmull_n_s16(vget_low_s16(a), NEW_SQRT2), NEW_SQRT2_BITS);
    int16x4_t hi = vqrshrn_n_s32(vmull_n_s16(vget_high_s16(a), NEW_SQRT2), NEW_SQRT2_BITS);
    return vcombine_s16(lo, hi);
}

static AOM_FORCE_INLINE void col16_dispatch(const int16_t* in, int stride, int16x8_t* out, TxType t) {
    int16x8_t s[16];
    load_buffer_s16_x8(in, stride, s, 16);
    shift_left_2_s16_x8(s, s, 16);
    switch (t) {
    case DCT_DCT:
    case DCT_ADST:
    case DCT_FLIPADST:
    case V_DCT:
        fdct16x16_neon(s, out, 13);
        break;
    case ADST_DCT:
    case ADST_ADST:
    case FLIPADST_DCT:
    case FLIPADST_FLIPADST:
    case ADST_FLIPADST:
    case FLIPADST_ADST:
    case V_ADST:
    case V_FLIPADST:
        fadst16x16_neon(s, out, 13);
        break;
    default:
        fidentity16x16_neon(s, out, 13);
        break;
    }
}

static AOM_FORCE_INLINE void row8_dispatch(const int16x8_t* in, int16x8_t* out, TxType t) {
    switch (t) {
    case DCT_DCT:
    case ADST_DCT:
    case FLIPADST_DCT:
    case H_DCT:
        fdct8x8_neon(in, out, 13);
        break;
    case DCT_ADST:
    case ADST_ADST:
    case DCT_FLIPADST:
    case FLIPADST_FLIPADST:
    case ADST_FLIPADST:
    case FLIPADST_ADST:
    case H_ADST:
    case H_FLIPADST:
        fadst8x8_neon(in, out, 13);
        break;
    default:
        fidentity8x8_neon(in, out, 13);
        break;
    }
}

// 8x16 forward, 8-bit. col=16pt, row=8pt(rect sqrt2). Final transpose -> SVT.
void svt_lbd_fwd_txfm2d_8x16_neon(int16_t* input, int32_t* output, uint32_t stride, TxType tx_type) {
    int ud_flip, lr_flip;
    get_flip_cfg(tx_type, &ud_flip, &lr_flip);
    const int16_t* in = input;
    ud_adjust_input_and_stride(ud_flip, &in, &stride, 16);

    int16x8_t buf0[16], buf1[16];
    col16_dispatch(in, (int)stride, buf0, tx_type);
    shift_right_2_round_s16_x8(buf0, buf0, 16);
    transpose_arrays_s16_8x8(buf0, buf1);
    transpose_arrays_s16_8x8(buf0 + 8, buf1 + 8);

    for (int i = 0; i < 2; i++) {
        int16x8_t rin[8];
        if (lr_flip) {
            flip_buf_8_neon(buf1 + 8 * i, rin, 8);
        }
        const int16x8_t* ri = lr_flip ? rin : (buf1 + 8 * i);
        int16x8_t        rout[8];
        row8_dispatch(ri, rout, tx_type);
        for (int r = 0; r < 8; r++) {
            rout[r] = round_shift_sqrt2_x8(rout[r]);
        }
        int16x8_t t[8];
        transpose_arrays_s16_8x8(rout, t);
        for (int r = 0; r < 8; r++) {
            int32_t* o = output + (8 * i + r) * 8;
            vst1q_s32(o + 0, vmovl_s16(vget_low_s16(t[r])));
            vst1q_s32(o + 4, vmovl_s16(vget_high_s16(t[r])));
        }
    }
}
