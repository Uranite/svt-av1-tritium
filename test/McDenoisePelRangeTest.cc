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

/* Regression tests for the motion-compensated denoiser module:
 *  - native bit-depth clipping of the Wiener half-pel interpolation,
 *  - half-pel kernels (scalar and AVX2) against a scalar reference,
 *  - dfttest output range at 12-bit (must not be clamped to 10-bit),
 *  - in-place mask merge (out aliasing src or filt) matching the
 *    out-of-place result. */

#include <stdint.h>
#include <string.h>

#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

#include "mc_denoise.h"
#include "random.h"

namespace {

using svt_av1_test_tool::SVTRandom;

void check_depth(uint32_t bitdepth) {
    const uint32_t maxval = (1u << bitdepth) - 1u;
    uint8_t pixels8[8] = {0};
    uint16_t pixels16[8] = {0};
    const uint32_t stride = 8;
    McdPlane plane;
    McdPelSet* pel = nullptr;
    uint16_t out[8];

    plane.w = 8;
    plane.h = 1;
    plane.stride = stride;
    plane.highbd = bitdepth > 8;
    plane.data = bitdepth > 8 ? (void*)pixels16 : (void*)pixels8;

    if (bitdepth > 8) {
        pixels16[2] = (uint16_t)maxval;
        pixels16[3] = (uint16_t)maxval;
    } else {
        pixels8[2] = (uint8_t)maxval;
        pixels8[3] = (uint8_t)maxval;
    }
    ASSERT_EQ(MCD_SUCCESS,
              svt_mcd_pelset_create(&plane, 1, bitdepth, 32, &pel));
    for (int32_t xphase = 0; xphase <= 1; xphase++) {
        svt_mcd_pelset_fetch(pel, 0, xphase, 0, 8, 1, out);
        for (uint32_t x = 0; x < 8; x++) {
            EXPECT_LE(out[x], maxval)
                << bitdepth << "-bit phase " << xphase << " sample " << x;
        }
    }
    svt_mcd_pelset_destroy(&pel);
}

/* 6-tap Wiener (1,-5,20,20,-5,1)/32 used by the pelset phase planes,
 * clamped to the 8-bit range. Only valid for interior coordinates. */
uint32_t wiener_h(const uint8_t* p, uint32_t stride, uint32_t x, uint32_t y) {
    const int32_t v =
        (int32_t)p[y * stride + x - 2] - 5 * (int32_t)p[y * stride + x - 1] +
        20 * (int32_t)p[y * stride + x] + 20 * (int32_t)p[y * stride + x + 1] -
        5 * (int32_t)p[y * stride + x + 2] + (int32_t)p[y * stride + x + 3];
    const int32_t r = (v + 16) >> 5;
    return (uint32_t)(r < 0 ? 0 : (r > 255 ? 255 : r));
}

uint32_t wiener_v(const uint8_t* p, uint32_t stride, uint32_t x, uint32_t y) {
    const int32_t v =
        (int32_t)p[(y - 2) * stride + x] -
        5 * (int32_t)p[(y - 1) * stride + x] + 20 * (int32_t)p[y * stride + x] +
        20 * (int32_t)p[(y + 1) * stride + x] -
        5 * (int32_t)p[(y + 2) * stride + x] + (int32_t)p[(y + 3) * stride + x];
    const int32_t r = (v + 16) >> 5;
    return (uint32_t)(r < 0 ? 0 : (r > 255 ? 255 : r));
}

/* One plane triplet with its backing storage. */
struct PlaneSet {
    std::vector<uint8_t> buf[3];
    McdPlane p[3];
};

void fill_planes(PlaneSet& ps, uint32_t bitdepth, uint32_t seed) {
    static const uint32_t dims[3][2] = {{32, 16}, {16, 8}, {16, 8}};
    const uint32_t maxval = (1u << bitdepth) - 1u;
    const bool hb = bitdepth > 8;

    SVTRandom rnd(0, (int)maxval, seed);
    for (int i = 0; i < 3; i++) {
        const uint32_t w = dims[i][0];
        const uint32_t h = dims[i][1];
        const size_t n = (size_t)w * h;

        ps.buf[i].assign(n * (hb ? 2 : 1), 0);
        if (hb) {
            uint16_t* d = (uint16_t*)ps.buf[i].data();
            for (size_t k = 0; k < n; k++) {
                d[k] = (uint16_t)rnd.random();
            }
        } else {
            uint8_t* d = ps.buf[i].data();
            for (size_t k = 0; k < n; k++) {
                d[k] = (uint8_t)rnd.random();
            }
        }
        ps.p[i].w = w;
        ps.p[i].h = h;
        ps.p[i].stride = w;
        ps.p[i].highbd = hb;
        ps.p[i].data = ps.buf[i].data();
    }
}

TEST(McDenoisePelRange, LocateHandlesNullOutputs) {
    int32_t stride = 7;
    const void* row = reinterpret_cast<const void*>(uintptr_t(1));

    svt_mcd_pelset_locate(nullptr, 0, 0, 0, 1, 1, 0, nullptr, &stride);
    EXPECT_EQ(0, stride);
    svt_mcd_pelset_locate(nullptr, 0, 0, 0, 1, 1, 0, &row, nullptr);
    EXPECT_EQ(nullptr, row);
}

TEST(McDenoisePelRange, HalfPelSamplesStayWithinNativeBitDepth) {
    check_depth(8);
    check_depth(10);
    check_depth(12);
}

/* Exercises the 8-bit half-pel kernels (AVX2 when available, scalar
 * otherwise) against the scalar 6-tap reference for every phase. */
TEST(McDenoisePelRange, HalfPelMatchesScalarReference) {
    const uint32_t w = 16;
    const uint32_t h = 16;
    std::vector<uint8_t> src((size_t)w * h);
    std::vector<uint8_t> vphase((size_t)w * h);
    McdPlane plane = {w, h, w, false, src.data()};
    McdPelSet* pel = nullptr;

    SVTRandom rnd(0, 255, 0x12345678u);
    for (size_t i = 0; i < src.size(); i++) {
        src[i] = (uint8_t)rnd.random();
    }
    ASSERT_EQ(MCD_SUCCESS, svt_mcd_pelset_create(&plane, 1, 8, 8, &pel));

    /* Only the interior rows are checked, so only those need a vertical
     * phase reference (wiener_v reads y-2..y+3). */
    for (uint32_t y = 2; y + 4 <= h; y++) {
        for (uint32_t x = 0; x < w; x++) {
            vphase[(size_t)y * w + x] = (uint8_t)wiener_v(src.data(), w, x, y);
        }
    }

    for (uint32_t y = 2; y + 4 <= h; y++) {
        for (uint32_t x = 2; x + 4 <= w; x++) {
            uint16_t out = 0;
            svt_mcd_pelset_fetch(
                pel, 0, 2 * (int32_t)x, 2 * (int32_t)y, 1, 1, &out);
            EXPECT_EQ(src[(size_t)y * w + x], out)
                << "integer " << x << "," << y;
            svt_mcd_pelset_fetch(
                pel, 0, 2 * (int32_t)x + 1, 2 * (int32_t)y, 1, 1, &out);
            EXPECT_EQ(wiener_h(src.data(), w, x, y), out)
                << "xhalf " << x << "," << y;
            svt_mcd_pelset_fetch(
                pel, 0, 2 * (int32_t)x, 2 * (int32_t)y + 1, 1, 1, &out);
            EXPECT_EQ(wiener_v(src.data(), w, x, y), out)
                << "yhalf " << x << "," << y;
            svt_mcd_pelset_fetch(
                pel, 0, 2 * (int32_t)x + 1, 2 * (int32_t)y + 1, 1, 1, &out);
            EXPECT_EQ(wiener_h(vphase.data(), w, x, y), out)
                << "xyhalf " << x << "," << y;
        }
    }
    svt_mcd_pelset_destroy(&pel);
}

/* The 12-bit dfttest path used to assume 10-bit (pmax = 1023) and clamped
 * every output sample to the 10-bit range. */
TEST(McDenoiseDfttest, TwelveBitOutputKeepsNativeRange) {
    const uint32_t w = 32;
    const uint32_t h = 32;
    const uint32_t bd = 12;
    const uint32_t maxval = (1u << bd) - 1u;
    std::vector<uint16_t> srcbuf((size_t)w * h);
    std::vector<uint16_t> dstbuf((size_t)w * h);
    McdPlane src = {w, h, w, true, srcbuf.data()};
    McdPlane dst = {w, h, w, true, dstbuf.data()};
    uint32_t mn = maxval;
    uint32_t mx = 0;

    for (uint32_t y = 0; y < h; y++) {
        for (uint32_t x = 0; x < w; x++) {
            srcbuf[(size_t)y * w + x] =
                (uint16_t)(3000u + ((x * 7u + y * 13u) % 96u));
        }
    }

    ASSERT_EQ(MCD_SUCCESS, svt_mcd_dfttest_plane(&src, &dst, 50, bd));
    for (size_t i = 0; i < srcbuf.size(); i++) {
        mn = std::min(mn, (uint32_t)dstbuf[i]);
        mx = std::max(mx, (uint32_t)dstbuf[i]);
    }
    EXPECT_GT(mn, 1023u) << "12-bit output was clamped to 10-bit range";
    EXPECT_LE(mx, maxval);
}

/* In-place merging (out aliasing src or filt) must produce exactly the
 * same planes as the out-of-place merge. The chroma box masks read
 * neighbouring luma, so an out==src merge needs the luma snapshot. */
TEST(McDenoiseMaskMerge, InPlaceMatchesOutOfPlace) {
    for (uint32_t bd = 8; bd <= 12; bd += 2) {
        for (int alias = 0; alias <= 1; alias++) {
            PlaneSet src_ref, filt_ref, out_ref;
            PlaneSet src_alias, filt_alias, out_alias;

            fill_planes(src_ref, bd, 0x1111u + bd);
            fill_planes(filt_ref, bd, 0x2222u + bd);
            fill_planes(out_ref, bd, 0x3333u + bd);
            ASSERT_EQ(MCD_SUCCESS,
                      svt_mcd_prefilter_mask_merge(
                          src_ref.p, filt_ref.p, out_ref.p, 3, 1, 1, bd));

            fill_planes(src_alias, bd, 0x1111u + bd);
            fill_planes(filt_alias, bd, 0x2222u + bd);
            fill_planes(out_alias, bd, 0x3333u + bd);
            McdPlane* dst = alias == 0 ? src_alias.p : filt_alias.p;
            ASSERT_EQ(MCD_SUCCESS,
                      svt_mcd_prefilter_mask_merge(
                          src_alias.p, filt_alias.p, dst, 3, 1, 1, bd));

            for (int p = 0; p < 3; p++) {
                const std::vector<uint8_t>& got =
                    alias == 0 ? src_alias.buf[p] : filt_alias.buf[p];
                EXPECT_EQ(out_ref.buf[p].size(), got.size());
                EXPECT_EQ(0,
                          memcmp(out_ref.buf[p].data(),
                                 got.data(),
                                 std::min(out_ref.buf[p].size(), got.size())))
                    << "bd=" << bd << " plane=" << p
                    << " alias=" << (alias == 0 ? "src" : "filt");
            }
        }
    }
}

}  // namespace
