/*
 * Copyright(c) 2026, Alliance for Open Media
 *
 * This source code is subject to the terms of the BSD 3-Clause Clear License
 * and the Alliance for Open Media Patent License 1.0. If the BSD 3-Clause Clear
 * License was not distributed with this source code in the LICENSE file, you
 * can obtain it at https://www.aomedia.org/license. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at
 * https://www.aomedia.org/license/patent-license.
 */

/******************************************************************************
 * @file SvtAv1EncZonesApiTest.cc
 *
 * @brief Quality-zone (--zones) parsing and per-frame application tests.
 *
 * Parsing is exercised through the public svt_av1_enc_parse_parameter() entry
 * point. Per-frame selection is checked directly on svt_av1_get_effective_qp()
 * for a two-zone clip, and end-to-end by encoding a low-delay CQP clip and
 * checking the QP reported on each output packet.
 ******************************************************************************/

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>

#include "DummyVideoSource.h"
#include "EbSvtAv1.h"
#include "EbSvtAv1Enc.h"
#include "gtest/gtest.h"
#include "sequence_control_set.h"

namespace {

constexpr uint32_t kWidth = 320;
constexpr uint32_t kHeight = 240;
constexpr uint32_t kNumFrames = 16;

static void clear_zones(EbSvtAv1EncConfiguration* cfg) {
    EXPECT_EQ(EB_ErrorNone, svt_av1_enc_parse_parameter(cfg, "zones", ""));
}

TEST(ZonesApiTest, ParseSingleZone) {
    EbSvtAv1EncConfiguration cfg{};
    ASSERT_EQ(EB_ErrorNone,
              svt_av1_enc_parse_parameter(&cfg, "zones", "0,100,35"));
    ASSERT_EQ(1, cfg.num_zones);
    ASSERT_NE(nullptr, cfg.quality_zones);
    EXPECT_EQ(0u, cfg.quality_zones[0].start_frame);
    EXPECT_EQ(100u, cfg.quality_zones[0].end_frame);
    EXPECT_EQ(35, cfg.quality_zones[0].zone_baseq);
    EXPECT_EQ(0, cfg.quality_zones[0].zone_qsidx);
    clear_zones(&cfg);
    EXPECT_EQ(nullptr, cfg.quality_zones);
    EXPECT_EQ(0, cfg.num_zones);
}

TEST(ZonesApiTest, ParseTwoZonesAndQuarterSteps) {
    EbSvtAv1EncConfiguration cfg{};
    ASSERT_EQ(EB_ErrorNone,
              svt_av1_enc_parse_parameter(&cfg, "zones", "0,9,20;10,19,32.75"));
    ASSERT_EQ(2, cfg.num_zones);
    ASSERT_NE(nullptr, cfg.quality_zones);

    EXPECT_EQ(0u, cfg.quality_zones[0].start_frame);
    EXPECT_EQ(9u, cfg.quality_zones[0].end_frame);
    EXPECT_EQ(20, cfg.quality_zones[0].zone_baseq);
    EXPECT_EQ(0, cfg.quality_zones[0].zone_qsidx);

    EXPECT_EQ(10u, cfg.quality_zones[1].start_frame);
    EXPECT_EQ(19u, cfg.quality_zones[1].end_frame);
    EXPECT_EQ(32, cfg.quality_zones[1].zone_baseq);
    EXPECT_EQ(3, cfg.quality_zones[1].zone_qsidx);

    // Re-parsing replaces the previous zone array.
    ASSERT_EQ(EB_ErrorNone,
              svt_av1_enc_parse_parameter(&cfg, "zones", "5,6,20.5"));
    ASSERT_EQ(1, cfg.num_zones);
    EXPECT_EQ(20, cfg.quality_zones[0].zone_baseq);
    EXPECT_EQ(2, cfg.quality_zones[0].zone_qsidx);

    clear_zones(&cfg);
    EXPECT_EQ(nullptr, cfg.quality_zones);
    EXPECT_EQ(0, cfg.num_zones);
}

TEST(ZonesApiTest, RejectsInvalidZones) {
    const char* invalid[] = {
        "10,0,20",          // start > end
        "0,10,71",          // quality above the supported range
        "0,10,nan",         // NaN must not reach round()/cast
        "0,10,inf",         // infinity must be rejected
        "0,10",             // missing quality
        "a,b,c",            // malformed numbers
        "0,10,20;bogus",    // malformed trailing zone
        "0,4294967296,20",  // frame range does not fit in 32 bits
    };
    for (const char* value : invalid) {
        EbSvtAv1EncConfiguration cfg{};
        EXPECT_EQ(EB_ErrorBadParameter,
                  svt_av1_enc_parse_parameter(&cfg, "zones", value))
            << "expected rejection of \"" << value << "\"";
        EXPECT_EQ(0, cfg.num_zones)
            << "zones left behind by \"" << value << "\"";
        clear_zones(&cfg);
    }
}

/** A failed re-parse must not leave the previous zones behind. */
TEST(ZonesApiTest, FailedReparseClearsZones) {
    EbSvtAv1EncConfiguration cfg{};
    ASSERT_EQ(EB_ErrorNone,
              svt_av1_enc_parse_parameter(&cfg, "zones", "0,9,20"));
    ASSERT_EQ(1, cfg.num_zones);
    ASSERT_NE(nullptr, cfg.quality_zones);

    EXPECT_EQ(EB_ErrorBadParameter,
              svt_av1_enc_parse_parameter(&cfg, "zones", "10,0,20"));
    EXPECT_EQ(nullptr, cfg.quality_zones);
    EXPECT_EQ(0, cfg.num_zones);
}

TEST(ZonesApiTest, ZonesRequestNonzeroQuality) {
    std::unique_ptr<SequenceControlSet> scs(new SequenceControlSet());
    memset(scs.get(), 0, sizeof(*scs));
    scs->static_config.rate_control_mode = SVT_AV1_RC_MODE_CQP_OR_CRF;
    scs->enable_qp_scaling_flag = 1;
    EXPECT_FALSE(svt_av1_zones_request_nonzero_quality(scs.get()));

    SvtAv1QualityZone zero_zone = {0, 9, 0, 0};
    scs->static_config.quality_zones = &zero_zone;
    scs->static_config.num_zones = 1;
    EXPECT_FALSE(svt_av1_zones_request_nonzero_quality(scs.get()));

    // A quarter step on top of base 0 is still lossy: qindex 0 is the only
    // lossless value.
    SvtAv1QualityZone quarter_zone = {0, 9, 0, 2};
    scs->static_config.quality_zones = &quarter_zone;
    EXPECT_TRUE(svt_av1_zones_request_nonzero_quality(scs.get()));

    SvtAv1QualityZone zones[2] = {{0, 4, 0, 0}, {5, 9, 35, 0}};
    scs->static_config.quality_zones = zones;
    scs->static_config.num_zones = 2;
    EXPECT_TRUE(svt_av1_zones_request_nonzero_quality(scs.get()));
}

/** Verifies the per-frame effective QP/qindex selection for a two-zone clip. */
TEST(ZonesApiTest, EffectiveQpFollowsTwoZones) {
    SvtAv1QualityZone zones[2] = {
        {1, 7, 20, 0},
        {8, 15, 50, 2},
    };
    std::unique_ptr<SequenceControlSet> scs(new SequenceControlSet());
    memset(scs.get(), 0, sizeof(*scs));
    scs->static_config.qp = 30;
    scs->static_config.rate_control_mode = SVT_AV1_RC_MODE_CQP_OR_CRF;
    scs->static_config.quality_zones = zones;
    scs->static_config.num_zones = 2;
    scs->enable_qp_scaling_flag = 1;

    // Frame 0 is outside both zones: the global QP is used.
    SvtAv1EffectiveQp qp = svt_av1_get_effective_qp(scs.get(), 0);
    EXPECT_EQ(30, qp.qp);
    EXPECT_FALSE(qp.from_zone);

    // Zone 1: base 20, no quarter step.
    for (uint64_t f = 1; f <= 7; ++f) {
        qp = svt_av1_get_effective_qp(scs.get(), f);
        ASSERT_TRUE(qp.from_zone) << "frame " << f;
        EXPECT_EQ(20, qp.qp) << "frame " << f;
        EXPECT_EQ(0u, qp.qindex_offset) << "frame " << f;
        EXPECT_EQ(0u, qp.extended_crf_qindex_offset) << "frame " << f;
        EXPECT_FALSE(qp.qp_is_max) << "frame " << f;
    }

    // Zone 2: base 50 + 2 quarter steps.
    for (uint64_t f = 8; f <= 15; ++f) {
        qp = svt_av1_get_effective_qp(scs.get(), f);
        ASSERT_TRUE(qp.from_zone) << "frame " << f;
        EXPECT_EQ(50, qp.qp) << "frame " << f;
        EXPECT_EQ(2u, qp.qindex_offset) << "frame " << f;
        EXPECT_EQ(0u, qp.extended_crf_qindex_offset) << "frame " << f;
        EXPECT_FALSE(qp.qp_is_max) << "frame " << f;
    }

    // Outside the union of the zones the global QP applies again.
    qp = svt_av1_get_effective_qp(scs.get(), 16);
    EXPECT_EQ(30, qp.qp);
    EXPECT_FALSE(qp.from_zone);
}

/** Extended CRF (63-70) is carried in qindex_offset, not in qp. */
TEST(ZonesApiTest, EffectiveQpExtendedCrf) {
    SvtAv1QualityZone zones[2] = {
        {1, 7, 63, 1},   // 63.25
        {8, 15, 70, 0},  // 70.0
    };
    std::unique_ptr<SequenceControlSet> scs(new SequenceControlSet());
    memset(scs.get(), 0, sizeof(*scs));
    scs->static_config.qp = 30;
    scs->static_config.rate_control_mode = SVT_AV1_RC_MODE_CQP_OR_CRF;
    scs->static_config.quality_zones = zones;
    scs->static_config.num_zones = 2;
    scs->enable_qp_scaling_flag = 1;

    SvtAv1EffectiveQp qp = svt_av1_get_effective_qp(scs.get(), 1);
    ASSERT_TRUE(qp.from_zone);
    EXPECT_EQ(63, qp.qp);
    EXPECT_TRUE(qp.qp_is_max);
    EXPECT_EQ(1u, qp.qindex_offset);
    EXPECT_EQ(1u, qp.extended_crf_qindex_offset);

    qp = svt_av1_get_effective_qp(scs.get(), 8);
    ASSERT_TRUE(qp.from_zone);
    EXPECT_EQ(63, qp.qp);
    EXPECT_TRUE(qp.qp_is_max);
    EXPECT_EQ(28u, qp.qindex_offset);
    EXPECT_EQ(28u, qp.extended_crf_qindex_offset);
}

/** Overlapping zones are resolved by the last matching entry. */
TEST(ZonesApiTest, EffectiveQpOverlappingZonesLastWins) {
    SvtAv1QualityZone zones[2] = {
        {0, 9, 20, 0},
        {5, 15, 40, 2},
    };
    std::unique_ptr<SequenceControlSet> scs(new SequenceControlSet());
    memset(scs.get(), 0, sizeof(*scs));
    scs->static_config.qp = 30;
    scs->static_config.rate_control_mode = SVT_AV1_RC_MODE_CQP_OR_CRF;
    scs->static_config.quality_zones = zones;
    scs->static_config.num_zones = 2;
    scs->enable_qp_scaling_flag = 1;

    // Only the first zone matches.
    SvtAv1EffectiveQp qp = svt_av1_get_effective_qp(scs.get(), 4);
    ASSERT_TRUE(qp.from_zone);
    EXPECT_EQ(20, qp.qp);
    EXPECT_EQ(0u, qp.qindex_offset);

    // Both match: the later zone wins.
    qp = svt_av1_get_effective_qp(scs.get(), 5);
    ASSERT_TRUE(qp.from_zone);
    EXPECT_EQ(40, qp.qp);
    EXPECT_EQ(2u, qp.qindex_offset);

    // Only the second zone matches.
    qp = svt_av1_get_effective_qp(scs.get(), 10);
    ASSERT_TRUE(qp.from_zone);
    EXPECT_EQ(40, qp.qp);
    EXPECT_EQ(2u, qp.qindex_offset);
}

/** Encodes kNumFrames of low-delay CQP video and records the QP reported on
 * each output packet, keyed by the packet's pts. */
static void encode_and_collect_qp(
    const char* zones, uint32_t global_qp,
    std::map<uint32_t, uint32_t>* qp_by_pts,
    const SvtAv1QualityZone* direct_zones = nullptr,
    uint16_t direct_num_zones = 0) {
    EbComponentType* enc = nullptr;
    EbSvtAv1EncConfiguration cfg{};
    ASSERT_EQ(EB_ErrorNone, svt_av1_enc_init_handle(&enc, &cfg));

    cfg.source_width = kWidth;
    cfg.source_height = kHeight;
    cfg.frame_rate_numerator = 30;
    cfg.frame_rate_denominator = 1;
    cfg.encoder_bit_depth = 8;
    cfg.encoder_color_format = EB_YUV420;
    cfg.enc_mode = 11;
    cfg.rtc = true;
    cfg.tune = 1;
    cfg.level_of_parallelism = 1;
    cfg.intra_period_length = -1;
    cfg.hierarchical_levels = 0;  // resolved to AUTO for RTC low-delay
    cfg.pred_structure = LOW_DELAY;
    cfg.rate_control_mode = SVT_AV1_RC_MODE_CQP_OR_CRF;
    cfg.aq_mode = 0;  // CQP
    cfg.qp = global_qp;
    cfg.look_ahead_distance = 0;
    cfg.recode_loop = 0;
    cfg.scene_change_detection = 0;
    cfg.enable_overlays = false;
    cfg.max_qp_allowed = 63;
    cfg.min_qp_allowed = 4;

    if (direct_zones && direct_num_zones > 0) {
        // Exercise the public struct path: the library takes ownership of the
        // array and frees it in svt_av1_enc_set_parameter().
        cfg.quality_zones = static_cast<SvtAv1QualityZone*>(
            malloc(sizeof(*direct_zones) * direct_num_zones));
        ASSERT_NE(nullptr, cfg.quality_zones);
        memcpy(cfg.quality_zones,
               direct_zones,
               sizeof(*direct_zones) * direct_num_zones);
        cfg.num_zones = direct_num_zones;
    } else if (zones) {
        ASSERT_EQ(EB_ErrorNone,
                  svt_av1_enc_parse_parameter(&cfg, "zones", zones));
    }
    ASSERT_EQ(EB_ErrorNone, svt_av1_enc_set_parameter(enc, &cfg));
    ASSERT_EQ(EB_ErrorNone, svt_av1_enc_init(enc));

    svt_av1_video_source::DummyVideoSource src(IMG_FMT_420, kWidth, kHeight, 8);
    ASSERT_EQ(0, src.open_source(0, 0));

    EbBufferHeaderType in_hdr{};
    in_hdr.size = sizeof(EbBufferHeaderType);
    for (uint32_t fi = 0; fi < kNumFrames; ++fi) {
        EbSvtIOFormat* frame = src.get_next_frame();
        ASSERT_NE(nullptr, frame);
        in_hdr.p_buffer = reinterpret_cast<uint8_t*>(frame);
        in_hdr.n_filled_len = src.get_frame_size();
        in_hdr.pts = fi;
        in_hdr.pic_type =
            (fi == 0) ? EB_AV1_KEY_PICTURE : EB_AV1_INVALID_PICTURE;
        in_hdr.p_app_private = nullptr;

        ASSERT_EQ(EB_ErrorNone, svt_av1_enc_send_picture(enc, &in_hdr));
        EbBufferHeaderType* out = nullptr;
        if (svt_av1_enc_get_packet(enc, &out, 0) == EB_ErrorNone && out) {
            if (out->n_filled_len > 0)
                (*qp_by_pts)[static_cast<uint32_t>(out->pts)] = out->qp;
            svt_av1_enc_release_out_buffer(&out);
        }
    }

    EbBufferHeaderType eos{};
    eos.size = sizeof(EbBufferHeaderType);
    eos.flags = EB_BUFFERFLAG_EOS;
    eos.pic_type = EB_AV1_INVALID_PICTURE;
    ASSERT_EQ(EB_ErrorNone, svt_av1_enc_send_picture(enc, &eos));
    for (;;) {
        EbBufferHeaderType* out = nullptr;
        EbErrorType rc = svt_av1_enc_get_packet(enc, &out, 1);
        if (rc != EB_ErrorNone || !out)
            break;
        const bool is_eos = (out->flags & EB_BUFFERFLAG_EOS) != 0;
        if (!is_eos && out->n_filled_len > 0)
            (*qp_by_pts)[static_cast<uint32_t>(out->pts)] = out->qp;
        svt_av1_enc_release_out_buffer(&out);
        if (is_eos)
            break;
    }

    svt_av1_enc_deinit(enc);
    svt_av1_enc_deinit_handle(enc);
}

/** Encodes a full RA/CRF clip and returns the compressed packets. */
static void encode_crf_packets(
    const char* zones, uint32_t qp, uint32_t ext_offset,
    std::vector<std::vector<uint8_t>>* packets, uint32_t max_bit_rate = 0,
    uint32_t enc_mode = 8, uint32_t width = kWidth, uint32_t height = kHeight,
    bool variance_boost = true, uint32_t recode_loop = 0, uint32_t tune = 0,
    PredStructure pred_structure = RANDOM_ACCESS) {
    EbComponentType* enc = nullptr;
    EbSvtAv1EncConfiguration cfg{};
    ASSERT_EQ(EB_ErrorNone, svt_av1_enc_init_handle(&enc, &cfg));

    cfg.source_width = width;
    cfg.source_height = height;
    cfg.frame_rate_numerator = 30;
    cfg.frame_rate_denominator = 1;
    cfg.encoder_bit_depth = 8;
    cfg.encoder_color_format = EB_YUV420;
    cfg.enc_mode = enc_mode;
    cfg.tune = tune;
    cfg.intra_period_length = 32;
    cfg.pred_structure = pred_structure;
    cfg.rate_control_mode = SVT_AV1_RC_MODE_CQP_OR_CRF;
    cfg.aq_mode = 2;  // CRF; keeps TPL and TF enabled
    cfg.qp = qp;
    cfg.extended_crf_qindex_offset = ext_offset;
    cfg.max_bit_rate = max_bit_rate;
    cfg.enable_variance_boost = variance_boost;
    cfg.level_of_parallelism =
        1;  // remove host-dependent threading from the test
    cfg.enable_overlays = false;
    cfg.look_ahead_distance = 0;
    cfg.recode_loop = recode_loop;

    if (zones) {
        ASSERT_EQ(EB_ErrorNone,
                  svt_av1_enc_parse_parameter(&cfg, "zones", zones));
    }
    ASSERT_EQ(EB_ErrorNone, svt_av1_enc_set_parameter(enc, &cfg));
    ASSERT_EQ(EB_ErrorNone, svt_av1_enc_init(enc));

    svt_av1_video_source::DummyVideoSource src(IMG_FMT_420, width, height, 8);
    ASSERT_EQ(0, src.open_source(0, 0));

    EbBufferHeaderType in_hdr{};
    in_hdr.size = sizeof(EbBufferHeaderType);
    for (uint32_t fi = 0; fi < kNumFrames; ++fi) {
        EbSvtIOFormat* frame = src.get_next_frame();
        ASSERT_NE(nullptr, frame);
        in_hdr.p_buffer = reinterpret_cast<uint8_t*>(frame);
        in_hdr.n_filled_len = src.get_frame_size();
        in_hdr.pts = fi;
        in_hdr.pic_type =
            (fi == 0) ? EB_AV1_KEY_PICTURE : EB_AV1_INVALID_PICTURE;
        in_hdr.p_app_private = nullptr;

        ASSERT_EQ(EB_ErrorNone, svt_av1_enc_send_picture(enc, &in_hdr));
        EbBufferHeaderType* out = nullptr;
        if (svt_av1_enc_get_packet(enc, &out, 0) == EB_ErrorNone && out) {
            if (out->n_filled_len > 0)
                packets->emplace_back(out->p_buffer,
                                      out->p_buffer + out->n_filled_len);
            svt_av1_enc_release_out_buffer(&out);
        }
    }

    EbBufferHeaderType eos{};
    eos.size = sizeof(EbBufferHeaderType);
    eos.flags = EB_BUFFERFLAG_EOS;
    eos.pic_type = EB_AV1_INVALID_PICTURE;
    ASSERT_EQ(EB_ErrorNone, svt_av1_enc_send_picture(enc, &eos));
    for (;;) {
        EbBufferHeaderType* out = nullptr;
        EbErrorType rc = svt_av1_enc_get_packet(enc, &out, 1);
        if (rc != EB_ErrorNone || !out)
            break;
        const bool is_eos = (out->flags & EB_BUFFERFLAG_EOS) != 0;
        if (!is_eos && out->n_filled_len > 0)
            packets->emplace_back(out->p_buffer,
                                  out->p_buffer + out->n_filled_len);
        svt_av1_enc_release_out_buffer(&out);
        if (is_eos)
            break;
    }

    svt_av1_enc_deinit(enc);
    svt_av1_enc_deinit_handle(enc);
}

/** Asserts that two encodes produced the same number of packets and that every
 * packet is byte-identical. Also guards against a near-empty encode. */
static void expect_same_packets(const std::vector<std::vector<uint8_t>>& a,
                                const std::vector<std::vector<uint8_t>>& b) {
    ASSERT_EQ(kNumFrames, a.size());
    ASSERT_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        EXPECT_EQ(a[i].size(), b[i].size()) << "packet " << i;
        if (a[i].size() == b[i].size()) {
            EXPECT_TRUE(a[i] == b[i]) << "packet " << i;
        }
    }
}

/** A zone covering the whole clip must reproduce the equivalent global CRF
 * encode bit-for-bit: the base QP must not leak into TF, TPL or delta-q. */
TEST(ZonesApiTest, FullClipZoneMatchesGlobalCrf) {
    std::vector<std::vector<uint8_t>> global_packets;
    encode_crf_packets(nullptr, 35, 0, &global_packets);
    std::vector<std::vector<uint8_t>> zoned_packets;
    encode_crf_packets("0,15,35", 0, 1, &zoned_packets);  // --crf 0.25

    expect_same_packets(global_packets, zoned_packets);
}

/** Capped CRF with a full-clip zone must not depend on the global QP.
 *
 * Note: the max-rate clamp's base-QP choice is not externally observable in
 * this setup (the rate-control adjustment dominates it), so this is an
 * equivalence/smoke check, not a direct test of the clamp. */
TEST(ZonesApiTest, CappedCrfFullClipZoneIgnoresGlobalQp) {
    constexpr uint32_t kMaxBitRate = 200000;  // binding, forces re-encodes
    // The zone overrides every frame, so the global QP must not influence the
    // capped-CRF clamp. The global QP is below the zone quality, so a
    // global-QP clamp would let re-encodes go below the requested zone quality.
    std::vector<std::vector<uint8_t>> low_global_packets;
    encode_crf_packets("0,15,50",
                       20,
                       0,
                       &low_global_packets,
                       kMaxBitRate,
                       8,
                       kWidth,
                       kHeight,
                       true,
                       4);
    std::vector<std::vector<uint8_t>> high_global_packets;
    encode_crf_packets("0,15,50",
                       50,
                       0,
                       &high_global_packets,
                       kMaxBitRate,
                       8,
                       kWidth,
                       kHeight,
                       true,
                       4);

    expect_same_packets(low_global_packets, high_global_packets);
}

TEST(ZonesApiTest, EffectiveQpAtFrameZero) {
    SvtAv1QualityZone zones[2] = {{0, 9, 60, 0}, {10, 19, 20, 0}};
    std::unique_ptr<SequenceControlSet> scs(new SequenceControlSet());
    memset(scs.get(), 0, sizeof(*scs));
    scs->static_config.rate_control_mode = SVT_AV1_RC_MODE_CQP_OR_CRF;
    scs->enable_qp_scaling_flag = 1;
    scs->static_config.quality_zones = zones;
    scs->static_config.num_zones = 2;

    // Frame 0 is inside the first zone, which is what drives the sequence-level
    // super-block-size decision.
    SvtAv1EffectiveQp qp = svt_av1_get_effective_qp(scs.get(), 0);
    ASSERT_TRUE(qp.from_zone);
    EXPECT_EQ(60, qp.qp);
    EXPECT_EQ(0u, qp.qindex_offset);
}

/** Below preset 5 the super-block size is QP dependent. A full-clip zone must
 * reproduce the equivalent global CRF encode there too. */
TEST(ZonesApiTest, FullClipZoneMatchesGlobalCrfBelowPreset5) {
    constexpr uint32_t kEncMode = 4;
    constexpr uint32_t kW = 640;
    constexpr uint32_t kH = 360;
    std::vector<std::vector<uint8_t>> global_packets;
    encode_crf_packets(
        nullptr, 35, 0, &global_packets, 0, kEncMode, kW, kH, false);
    std::vector<std::vector<uint8_t>> zoned_packets;
    // Global QP is above the SB threshold; the zone is below it.
    encode_crf_packets(
        "0,15,35", 58, 0, &zoned_packets, 0, kEncMode, kW, kH, false);

    expect_same_packets(global_packets, zoned_packets);
}

/** The reverse direction: a high-quality zone must also match the equivalent
 * global encode when the global QP is low (super-block size is QP dependent).
 */
TEST(ZonesApiTest, FullClipZoneMatchesGlobalCrfBelowPreset5HighQuality) {
    constexpr uint32_t kEncMode = 4;
    constexpr uint32_t kW = 640;
    constexpr uint32_t kH = 360;
    std::vector<std::vector<uint8_t>> global_packets;
    // --crf 63.5
    encode_crf_packets(
        nullptr, 63, 2, &global_packets, 0, kEncMode, kW, kH, false);
    std::vector<std::vector<uint8_t>> zoned_packets;
    // --crf 0.25 with a full-clip zone at 63.5
    encode_crf_packets(
        "0,15,63.5", 0, 1, &zoned_packets, 0, kEncMode, kW, kH, false);

    expect_same_packets(global_packets, zoned_packets);
}

/** VMAF preprocessing is QP dependent; a full-clip zone must match. */
TEST(ZonesApiTest, FullClipZoneMatchesGlobalCrfVmaf) {
    std::vector<std::vector<uint8_t>> global_packets;
    encode_crf_packets(
        nullptr, 35, 0, &global_packets, 0, 8, kWidth, kHeight, true, 0, 5);
    std::vector<std::vector<uint8_t>> zoned_packets;
    encode_crf_packets(
        "0,15,35", 0, 1, &zoned_packets, 0, 8, kWidth, kHeight, true, 0, 5);
    expect_same_packets(global_packets, zoned_packets);
}

/** TUNE_IQ's max_tx_size is QP dependent; a full-clip zone must match. */
TEST(ZonesApiTest, FullClipZoneMatchesGlobalCrfTuneIqLowDelay) {
    std::vector<std::vector<uint8_t>> global_packets;
    encode_crf_packets(nullptr,
                       63,
                       2,
                       &global_packets,
                       0,
                       8,
                       kWidth,
                       kHeight,
                       true,
                       0,
                       3,
                       LOW_DELAY);
    std::vector<std::vector<uint8_t>> zoned_packets;
    encode_crf_packets("0,15,63.5",
                       0,
                       1,
                       &zoned_packets,
                       0,
                       8,
                       kWidth,
                       kHeight,
                       true,
                       0,
                       3,
                       LOW_DELAY);
    expect_same_packets(global_packets, zoned_packets);
}

TEST(ZonesApiTest, PerFrameQpFollowsTwoZones) {
    std::map<uint32_t, uint32_t> qp_by_pts;
    encode_and_collect_qp("1,7,10;8,15,55", 30, &qp_by_pts);

    ASSERT_EQ(kNumFrames, qp_by_pts.size());

    double sum_a = 0.0, sum_b = 0.0;
    uint32_t max_a = 0, min_b = 255;
    for (uint32_t i = 1; i <= 7; ++i) {
        ASSERT_NE(qp_by_pts.end(), qp_by_pts.find(i));
        sum_a += qp_by_pts[i];
        if (qp_by_pts[i] > max_a)
            max_a = qp_by_pts[i];
    }
    for (uint32_t i = 8; i < kNumFrames; ++i) {
        ASSERT_NE(qp_by_pts.end(), qp_by_pts.find(i));
        sum_b += qp_by_pts[i];
        if (qp_by_pts[i] < min_b)
            min_b = qp_by_pts[i];
    }

    const double mean_a = sum_a / 7.0;
    const double mean_b = sum_b / 8.0;
    // The RTC low-delay CQP path applies temporal-layer QP scaling, so assert
    // the zone separation rather than exact per-frame QP values.
    // The two zones must actually drive different frame QPs.
    EXPECT_LT(mean_a, mean_b - 15.0)
        << "zone 1 mean QP " << mean_a << " vs zone 2 mean QP " << mean_b;
    EXPECT_LT(max_a, min_b)
        << "every zone-1 frame must be higher quality than any zone-2 frame";
}

static double mean_qp(const std::map<uint32_t, uint32_t>& qp_by_pts,
                      uint32_t first, uint32_t last) {
    double sum = 0.0;
    uint32_t n = 0;
    for (uint32_t i = first; i <= last; ++i) {
        auto it = qp_by_pts.find(i);
        if (it != qp_by_pts.end()) {
            sum += it->second;
            ++n;
        }
    }
    return n ? sum / n : 0.0;
}

TEST(ZonesApiTest, ZonesOverrideGlobalQp) {
    std::map<uint32_t, uint32_t> global_qps;
    encode_and_collect_qp(nullptr, 30, &global_qps);
    std::map<uint32_t, uint32_t> zoned_qps;
    encode_and_collect_qp("1,7,10;8,15,55", 30, &zoned_qps);

    ASSERT_EQ(kNumFrames, global_qps.size());
    ASSERT_EQ(kNumFrames, zoned_qps.size());

    const double global_mean = mean_qp(global_qps, 1, kNumFrames - 1);
    const double zone_a_mean = mean_qp(zoned_qps, 1, 7);
    const double zone_b_mean = mean_qp(zoned_qps, 8, kNumFrames - 1);

    // A better-quality zone must beat the global setting, and a worse-quality
    // zone must lose to it.
    EXPECT_LT(zone_a_mean, global_mean - 5.0)
        << "zone A " << zone_a_mean << " vs global " << global_mean;
    EXPECT_GT(zone_b_mean, global_mean + 5.0)
        << "zone B " << zone_b_mean << " vs global " << global_mean;
}

/** The public struct path (no parser) must apply zones the same way. */
TEST(ZonesApiTest, DirectApiZonesApply) {
    SvtAv1QualityZone zones[2] = {{1, 7, 10, 0}, {8, 15, 55, 0}};
    std::map<uint32_t, uint32_t> qp_by_pts;
    encode_and_collect_qp(nullptr, 30, &qp_by_pts, zones, 2);

    ASSERT_EQ(kNumFrames, qp_by_pts.size());
    const double zone_a_mean = mean_qp(qp_by_pts, 1, 7);
    const double zone_b_mean = mean_qp(qp_by_pts, 8, 15);
    EXPECT_LT(zone_a_mean, zone_b_mean - 15.0)
        << "zone A " << zone_a_mean << " vs zone B " << zone_b_mean;
}

}  // namespace
