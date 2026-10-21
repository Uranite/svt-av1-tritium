/* Regression tests for bitdepth-aware Wiener half-pel clipping. */
#include <stdint.h>

#include "gtest/gtest.h"

extern "C" {
#include "mc_denoise.h"
}

namespace {

void check_depth(uint32_t bitdepth) {
    const uint32_t maxval = (1u << bitdepth) - 1u;
    uint8_t pixels8[8] = {0};
    uint16_t pixels16[8] = {0};
    const uint32_t stride = 8;
    McdPlane plane = {8, 1, stride, bitdepth > 8,
                      bitdepth > 8 ? (void *)pixels16 : (void *)pixels8};
    McdPelSet *pel = nullptr;
    uint16_t out[8];

    if (bitdepth > 8) {
        pixels16[2] = (uint16_t)maxval;
        pixels16[3] = (uint16_t)maxval;
    } else {
        pixels8[2] = (uint8_t)maxval;
        pixels8[3] = (uint8_t)maxval;
    }
    ASSERT_EQ(MCD_SUCCESS, mcd_pelset_create(&plane, 1, bitdepth, 32, &pel));
    for (int32_t xphase = 0; xphase <= 1; xphase++) {
        mcd_pelset_fetch(pel, 0, xphase, 0, 8, 1, out);
        for (uint32_t x = 0; x < 8; x++)
            EXPECT_LE(out[x], maxval) << bitdepth << "-bit phase " << xphase
                                      << " sample " << x;
    }
    mcd_pelset_destroy(&pel);
}

TEST(McDenoisePelRange, LocateHandlesNullOutputs) {
    int32_t stride = 7;
    const void *row = reinterpret_cast<const void *>(uintptr_t(1));

    mcd_pelset_locate(nullptr, 0, 0, 0, 1, 1, 0, nullptr, &stride);
    EXPECT_EQ(0, stride);
    mcd_pelset_locate(nullptr, 0, 0, 0, 1, 1, 0, &row, nullptr);
    EXPECT_EQ(nullptr, row);
}

TEST(McDenoisePelRange, HalfPelSamplesStayWithinNativeBitDepth) {
    check_depth(8);
    check_depth(10);
    check_depth(12);
}

} // namespace
