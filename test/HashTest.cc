/*
 * Copyright (c) 2026, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at https://www.aomedia.org/license/software-license. If the
 * Alliance for Open Media Patent License 1.0 was not distributed with this
 * source code in the PATENTS file, you can obtain it at
 * https://www.aomedia.org/license/patent-license.
 */

/******************************************************************************
 * @file HashTest.cc
 *
 * @brief Unit test for the hardware CRC-32C kernels used by hash-based
 * motion estimation:
 * - svt_av1_get_crc32c_value_arm_crc32
 * - svt_av1_get_crc32c_value_sse4_2
 *
 ******************************************************************************/

#include <cinttypes>
#include <cstring>

#include "gtest/gtest.h"
#include "aom_dsp_rtcd.h"
#include "common_dsp_rtcd.h"
#include "definitions.h"
#include "hash.h"
#include "random.h"
#include "svt_time.h"

using svt_av1_test_tool::SVTRandom;

#if defined(ARCH_AARCH64)
#if HAVE_ARM_CRC32
#define HW_CRC32C_FUNC svt_av1_get_crc32c_value_arm_crc32
#define HW_CRC32C_FLAG EB_CPU_FLAGS_ARM_CRC32
#endif
#elif defined(ARCH_X86_64)
#define HW_CRC32C_FUNC svt_av1_get_crc32c_value_sse4_2
#define HW_CRC32C_FLAG EB_CPU_FLAGS_SSE4_2
#endif

namespace {

uint32_t crc32c_ref(CRC32C *calc, const uint8_t *buf, size_t len) {
    return svt_av1_get_crc32c_value_c(calc, buf, len);
}

// CRC-32C check values, e.g. from RFC 3720 (iSCSI) appendix B.4.
TEST(HashCrc32cTest, KnownAnswer) {
    CRC32C calc;
    svt_av1_crc32c_calculator_init(&calc);

    const uint8_t check[] = "123456789";
    EXPECT_EQ(0xE3069283u, crc32c_ref(&calc, check, 9));

    const uint8_t zeros[32] = {0};
    EXPECT_EQ(0x8A9136AAu, crc32c_ref(&calc, zeros, 32));

    uint8_t ones[32];
    memset(ones, 0xFF, sizeof(ones));
    EXPECT_EQ(0x62A8AB43u, crc32c_ref(&calc, ones, 32));

    EXPECT_EQ(0x00000000u, crc32c_ref(&calc, check, 0));
}

#ifdef HW_CRC32C_FUNC

TEST(HashCrc32cTest, KnownAnswerHw) {
    if (!(svt_aom_get_cpu_flags() & HW_CRC32C_FLAG))
        GTEST_SKIP() << "Hardware CRC32C not supported on this CPU";

    const uint8_t check[] = "123456789";
    EXPECT_EQ(0xE3069283u, HW_CRC32C_FUNC(nullptr, check, 9));
    EXPECT_EQ(0x00000000u, HW_CRC32C_FUNC(nullptr, check, 0));
}

TEST(HashCrc32cTest, MatchC) {
    if (!(svt_aom_get_cpu_flags() & HW_CRC32C_FLAG))
        GTEST_SKIP() << "Hardware CRC32C not supported on this CPU";

    CRC32C calc;
    svt_av1_crc32c_calculator_init(&calc);

    constexpr size_t max_len = 1024;
    constexpr size_t max_offset = 16;
    uint8_t buf[max_len + max_offset];
    SVTRandom rnd(8, false);
    for (size_t i = 0; i < sizeof(buf); i++)
        buf[i] = rnd.Rand8();

    // All lengths up to a couple of 8-byte blocks to cover the head/tail
    // handling, plus some longer ones; all at varying alignments.
    for (size_t offset = 0; offset < max_offset; offset++) {
        for (size_t len = 0; len <= max_len;
             len = len < 32 ? len + 1 : len * 2) {
            const uint32_t ref_crc = crc32c_ref(&calc, buf + offset, len);
            const uint32_t hw_crc = HW_CRC32C_FUNC(&calc, buf + offset, len);
            ASSERT_EQ(ref_crc, hw_crc)
                << "CRC mismatch at offset " << offset << " length " << len;
        }
    }
}

TEST(HashCrc32cTest, DISABLED_Speed) {
    if (!(svt_aom_get_cpu_flags() & HW_CRC32C_FLAG))
        GTEST_SKIP() << "Hardware CRC32C not supported on this CPU";

    CRC32C calc;
    svt_av1_crc32c_calculator_init(&calc);

    // 16 bytes is what svt_av1_generate_block_hash_value and
    // svt_av1_get_block_hash_value hash on every call.
    const size_t lens[] = {16, 64, 1024};
    SVTRandom rnd(8, false);

    for (size_t len : lens) {
        uint8_t buf[1024];
        for (size_t i = 0; i < len; i++)
            buf[i] = rnd.Rand8();

        const uint64_t num_iter = 1000000000 / (len * 10);
        double time_c, time_o;
        uint64_t start_time_seconds, start_time_useconds;
        uint64_t finish_time_seconds, finish_time_useconds;

        uint32_t sum_c = 0;
        svt_av1_get_time(&start_time_seconds, &start_time_useconds);
        for (uint64_t i = 0; i < num_iter; i++) {
            buf[0] = (uint8_t)i;
            sum_c += crc32c_ref(&calc, buf, len);
        }
        svt_av1_get_time(&finish_time_seconds, &finish_time_useconds);
        time_c = svt_av1_compute_overall_elapsed_time_ms(start_time_seconds,
                                                         start_time_useconds,
                                                         finish_time_seconds,
                                                         finish_time_useconds);

        uint32_t sum_o = 0;
        svt_av1_get_time(&start_time_seconds, &start_time_useconds);
        for (uint64_t i = 0; i < num_iter; i++) {
            buf[0] = (uint8_t)i;
            sum_o += HW_CRC32C_FUNC(&calc, buf, len);
        }
        svt_av1_get_time(&finish_time_seconds, &finish_time_useconds);
        time_o = svt_av1_compute_overall_elapsed_time_ms(start_time_seconds,
                                                         start_time_useconds,
                                                         finish_time_seconds,
                                                         finish_time_useconds);

        EXPECT_EQ(sum_c, sum_o);
        printf("len = %4zu, iterations = %" PRIu64
               ": c_time = %f \t o_time = %f \t Gain = %4.2f\n",
               len,
               num_iter,
               time_c,
               time_o,
               time_c / time_o);
    }
}

#endif  // HW_CRC32C_FUNC

}  // namespace
