/* Standalone regression for bitdepth-aware Wiener half-pel clipping.
 * Build with mc_denoise_pel.c and mc_denoise_simd.c plus the codec include
 * paths. The 10-bit edge pattern overshoots 1023 before clipping. */
#include <stdint.h>
#include <stdio.h>

#include "mc_denoise.h"

static int check_depth(uint32_t bitdepth) {
    const uint32_t maxval = (1u << bitdepth) - 1u;
    uint8_t pixels8[8] = {0};
    uint16_t pixels16[8] = {0};
    const uint32_t stride = 8;
    McdPlane plane = {8, 1, stride, bitdepth > 8,
                      bitdepth > 8 ? (void *)pixels16 : (void *)pixels8};
    McdPelSet *pel = NULL;
    uint16_t out[8];

    if (bitdepth > 8) {
        pixels16[2] = (uint16_t)maxval;
        pixels16[3] = (uint16_t)maxval;
    } else {
        pixels8[2] = (uint8_t)maxval;
        pixels8[3] = (uint8_t)maxval;
    }
    if (mcd_pelset_create(&plane, 1, bitdepth, 32, &pel) != MCD_SUCCESS)
        return 1;
    for (int32_t xphase = 0; xphase <= 1; xphase++) {
        mcd_pelset_fetch(pel, 0, xphase, 0, 8, 1, out);
        for (uint32_t x = 0; x < 8; x++) {
            if (out[x] > maxval) {
                fprintf(stderr, "%u-bit phase %d sample %u exceeded max: %u\n",
                        bitdepth, xphase, x, out[x]);
                mcd_pelset_destroy(&pel);
                return 1;
            }
        }
    }
    mcd_pelset_destroy(&pel);
    return 0;
}

int main(void) {
    int32_t stride = 7;
    const void *row = (const void *)(uintptr_t)1;

    mcd_pelset_locate(NULL, 0, 0, 0, 1, 1, 0, NULL, &stride);
    if (stride != 0)
        return 1;
    mcd_pelset_locate(NULL, 0, 0, 0, 1, 1, 0, &row, NULL);
    if (row != NULL)
        return 1;

    return check_depth(8) || check_depth(10) || check_depth(12);
}
