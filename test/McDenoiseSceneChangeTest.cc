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

/* Pins the motion-compensated denoiser scene-change boundary.
 *
 * mcd_me_scene() used to test `changed*100 > thscd2_pct*n`, the strict form
 * of "more than thscd2 percent". The documented contract is "at least", so
 * the comparison is now inclusive. These tests lock the endpoints and the
 * exact-equality case so the off-by-one cannot come back silently:
 *   - thscd2_pct == 0 must still require one changed block,
 *   - thscd2_pct == 100 must fire when every block changed (it could never
 *     fire under the strict form),
 *   - exactly thscd2_pct changed blocks must fire when that is an integral
 *     block count. */

#include <stdint.h>

#include "gtest/gtest.h"

#include "mc_denoise.h"

namespace {

/* thscd2_pct == 0: a static frame (no changed blocks) is never a scene
 * change, but a single changed block is. */
TEST(McDenoiseSceneChange, ZeroThresholdRequiresOneChangedBlock) {
    EXPECT_EQ(0, mcd_scene_change_decision(0, 100, 0));
    EXPECT_EQ(0, mcd_scene_change_decision(0, 1, 0));
    EXPECT_EQ(1, mcd_scene_change_decision(1, 100, 0));
    EXPECT_EQ(1, mcd_scene_change_decision(1, 1, 0));
}

/* thscd2_pct == 100: every block must change. Under the old strict
 * comparison this required changed > n and could never fire. */
TEST(McDenoiseSceneChange, HundredPercentRequiresEveryBlock) {
    EXPECT_EQ(0, mcd_scene_change_decision(99, 100, 100));
    EXPECT_EQ(1, mcd_scene_change_decision(100, 100, 100));
    EXPECT_EQ(0, mcd_scene_change_decision(0, 1, 100));
    EXPECT_EQ(1, mcd_scene_change_decision(1, 1, 100));
}

/* The inclusive boundary: exactly thscd2_pct percent fires. This is the
 * case the strict form got wrong (it needed one extra block). */
TEST(McDenoiseSceneChange, FiresAtExactPercentage) {
    EXPECT_EQ(0, mcd_scene_change_decision(50, 100, 51));
    EXPECT_EQ(1, mcd_scene_change_decision(51, 100, 51));

    EXPECT_EQ(0, mcd_scene_change_decision(101, 200, 51));
    EXPECT_EQ(1, mcd_scene_change_decision(102, 200, 51));

    EXPECT_EQ(0, mcd_scene_change_decision(49, 100, 50));
    EXPECT_EQ(1, mcd_scene_change_decision(50, 100, 50));
}

/* When thscd2_pct*n is not a multiple of 100 the boundary is not integral
 * and the decision fires at the first block count that reaches it. */
TEST(McDenoiseSceneChange, NonIntegralBoundaryRoundsUp) {
    /* 51% of 3 blocks = 1.53 blocks -> 2 blocks are needed. */
    EXPECT_EQ(0, mcd_scene_change_decision(1, 3, 51));
    EXPECT_EQ(1, mcd_scene_change_decision(2, 3, 51));

    /* 1% of 3 blocks = 0.03 -> a single changed block suffices. */
    EXPECT_EQ(1, mcd_scene_change_decision(1, 3, 1));
}

/* Default configuration on a 1080p analysis grid (blksize 16, overlap 8):
 * 239 x 134 = 32026 blocks. 51% of 32026 is 16333.26, so 16334 blocks are
 * required. Documents that the default is unaffected by the boundary shift
 * for this geometry (32026 is not a multiple of 100). */
TEST(McDenoiseSceneChange, DefaultThresholdOn1080pGrid) {
    const uint32_t n = 239u * 134u;
    EXPECT_EQ(0, mcd_scene_change_decision(16333, n, 51));
    EXPECT_EQ(1, mcd_scene_change_decision(16334, n, 51));
}

/* Degenerate inputs must not divide by zero or fire. */
TEST(McDenoiseSceneChange, EmptyGridIsNotASceneChange) {
    EXPECT_EQ(0, mcd_scene_change_decision(0, 0, 0));
    EXPECT_EQ(0, mcd_scene_change_decision(0, 0, 51));
    EXPECT_EQ(0, mcd_scene_change_decision(0, 0, 100));
}

}  // namespace
