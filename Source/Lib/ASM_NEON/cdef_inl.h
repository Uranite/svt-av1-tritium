/*
 * Copyright (c) 2025, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

// CDEF 8-bit pixel-copy helpers (NEON).

#ifndef EbCdefInl_h
#define EbCdefInl_h

#include <arm_neon.h>
#include <string.h>

static inline void cdef_narrow_row(uint8_t* d, const uint16_t* s, int cols) {
    int c = 0;
    for (; c + 8 <= cols; c += 8) {
        vst1_u8(d + c, vmovn_u16(vld1q_u16(s + c)));
    }
    for (; c < cols; c++) {
        d[c] = (uint8_t)s[c];
    }
}

static inline void cdef_widen_row(uint16_t* d, const uint8_t* s, int cols) {
    int c = 0;
    for (; c + 8 <= cols; c += 8) {
        vst1q_u16(d + c, vmovl_u8(vld1_u8(s + c)));
    }
    for (; c < cols; c++) {
        d[c] = (uint16_t)s[c];
    }
}

static inline void svt_cdef_copy_rect8(uint8_t* dst, int dstride, const uint8_t* src, int src_voffset, int src_hoffset,
                                       int sstride, int v, int h) {
    const uint8_t* base = &src[src_voffset * sstride + src_hoffset];
    for (int r = 0; r < v; r++) {
        memcpy(dst, base, (size_t)h);
        dst += dstride;
        base += sstride;
    }
}

static inline void svt_cdef_narrow_rect(uint8_t* dst, int dstride, const uint16_t* src, int sstride, int v, int h) {
    for (int r = 0; r < v; r++) {
        cdef_narrow_row(dst, src, h);
        dst += dstride;
        src += sstride;
    }
}

static inline void svt_cdef_widen_rect(uint16_t* dst, int dstride, const uint8_t* src, int sstride, int v, int h) {
    for (int r = 0; r < v; r++) {
        cdef_widen_row(dst, src, h);
        dst += dstride;
        src += sstride;
    }
}

#endif // EbCdefInl_h
