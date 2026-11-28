/*
 * Copyright(c) 2026 Alliance for Open Media. All rights reserved.
 *
 * Regression coverage for deterministic motion-compensated denoising across
 * the ordered and no-hold Picture Analysis paths.
 */

#include <cstdint>
#include <vector>

#include "DummyVideoSource.h"
#include "EbSvtAv1Enc.h"
#include "gtest/gtest.h"

namespace {

constexpr uint32_t kWidth = 322;
constexpr uint32_t kHeight = 242;
struct McDenoiseParam {
    const char* name;
    PredStructure pred_structure;
    uint8_t bit_depth;
    uint32_t level_of_parallelism;
    bool enable_overlays;
    uint32_t frame_count;
};

bool encode_mc_denoise(const McDenoiseParam& prm,
                       std::vector<uint8_t>* bitstream) {
    EbComponentType* enc = nullptr;
    bool enc_initialized = false;
    bool eos = false;

    auto cleanup = [&]() {
        if (enc_initialized) {
            svt_av1_enc_deinit(enc);
        }
        if (enc) {
            svt_av1_enc_deinit_handle(enc);
        }
    };

    EbSvtAv1EncConfiguration cfg{};
    if (svt_av1_enc_init_handle(&enc, &cfg) != EB_ErrorNone) {
        return false;
    }
    cfg.source_width = kWidth;
    cfg.source_height = kHeight;
    cfg.frame_rate_numerator = 30;
    cfg.frame_rate_denominator = 1;
    cfg.encoder_bit_depth = prm.bit_depth;
    cfg.encoder_color_format = EB_YUV420;
    cfg.enc_mode = 10;
    cfg.pred_structure = prm.pred_structure;
    cfg.intra_period_length = prm.pred_structure == ALL_INTRA ? 0 : -1;
    cfg.hierarchical_levels = 3;
    cfg.level_of_parallelism = prm.level_of_parallelism;
    cfg.enable_overlays = prm.enable_overlays;
    cfg.rate_control_mode = SVT_AV1_RC_MODE_CQP_OR_CRF;
    cfg.qp = 32;
    cfg.enable_tf = 0;
    cfg.mc_denoise = 1;
    cfg.mc_denoise_tr = 2;
    cfg.mc_denoise_prefilter = 0;

    if (svt_av1_enc_set_parameter(enc, &cfg) != EB_ErrorNone ||
        svt_av1_enc_init(enc) != EB_ErrorNone) {
        cleanup();
        return false;
    }
    enc_initialized = true;

    svt_av1_video_source::DummyVideoSource src(
        IMG_FMT_420, kWidth, kHeight, prm.bit_depth);
    if (src.open_source(0, prm.frame_count) != EB_ErrorNone) {
        cleanup();
        return false;
    }

    bool output_ok = true;
    auto consume_packet = [&](EbBufferHeaderType** pkt) {
        eos = ((*pkt)->flags & EB_BUFFERFLAG_EOS) != 0;
        if ((*pkt)->n_filled_len > 0) {
            bitstream->insert(bitstream->end(),
                              (*pkt)->p_buffer,
                              (*pkt)->p_buffer + (*pkt)->n_filled_len);
        }
        svt_av1_enc_release_out_buffer(pkt);
    };
    auto drain_available = [&]() {
        while (!eos && output_ok) {
            EbBufferHeaderType* pkt = nullptr;
            const EbErrorType rc = svt_av1_enc_get_packet(enc, &pkt, false);
            if (rc == EB_NoErrorEmptyQueue) {
                return;
            }
            if (rc != EB_ErrorNone || !pkt) {
                output_ok = false;
                if (pkt) {
                    svt_av1_enc_release_out_buffer(&pkt);
                }
                return;
            }
            consume_packet(&pkt);
        }
    };

    bool input_ok = true;
    for (uint32_t fi = 0; fi < prm.frame_count; ++fi) {
        EbSvtIOFormat* frame = src.get_next_frame();
        if (!frame) {
            input_ok = false;
            break;
        }
        EbBufferHeaderType in{};
        in.size = sizeof(in);
        in.p_buffer = reinterpret_cast<uint8_t*>(frame);
        in.n_filled_len = src.get_frame_size();
        in.pts = fi;
        in.pic_type = EB_AV1_INVALID_PICTURE;
        if (svt_av1_enc_send_picture(enc, &in) != EB_ErrorNone) {
            input_ok = false;
            break;
        }
        if (prm.pred_structure != LOW_DELAY) {
            // get_packet blocks unconditionally in low-delay mode, even when
            // pic_send_done is false. For other structures this safely drains
            // available output without waiting.
            drain_available();
        }
    }

    EbBufferHeaderType eos_hdr{};
    eos_hdr.size = sizeof(eos_hdr);
    eos_hdr.flags = EB_BUFFERFLAG_EOS;
    eos_hdr.pic_type = EB_AV1_INVALID_PICTURE;
    if (svt_av1_enc_send_picture(enc, &eos_hdr) != EB_ErrorNone) {
        input_ok = false;
    }
    // By now EOS has been submitted; blocking packet reads are safe for all
    // prediction structures and drain the final packets before teardown.
    while (!eos && output_ok && input_ok) {
        EbBufferHeaderType* pkt = nullptr;
        if (svt_av1_enc_get_packet(enc, &pkt, true) != EB_ErrorNone || !pkt) {
            output_ok = false;
            if (pkt) {
                svt_av1_enc_release_out_buffer(&pkt);
            }
            break;
        }
        consume_packet(&pkt);
    }

    src.close_source();
    cleanup();
    return input_ok && output_ok && eos && !bitstream->empty();
}

class McDenoiseOrderingApiTest
    : public ::testing::TestWithParam<McDenoiseParam> {};

TEST_P(McDenoiseOrderingApiTest,
       RepeatedEncodeHasIdenticalBitstreamAndCompletes) {
    const McDenoiseParam& prm = GetParam();
    std::vector<uint8_t> first;
    std::vector<uint8_t> second;

    ASSERT_TRUE(encode_mc_denoise(prm, &first))
        << prm.name << " first encode did not complete";
    ASSERT_TRUE(encode_mc_denoise(prm, &second))
        << prm.name << " second encode did not complete";
    EXPECT_EQ(first, second)
        << prm.name << " produced different output across identical runs";
}

INSTANTIATE_TEST_SUITE_P(
    McDenoiseOrdering, McDenoiseOrderingApiTest,
    ::testing::Values(
        McDenoiseParam{"LowDelayMultiThread8Bit", LOW_DELAY, 8, 4, false, 8},
        McDenoiseParam{"LowDelayMultiThread10Bit", LOW_DELAY, 10, 4, false, 8},
        McDenoiseParam{
            "AllIntraSingleThread10Bit", ALL_INTRA, 10, 1, false, 16},
        McDenoiseParam{"AllIntraMultiThread8Bit", ALL_INTRA, 8, 4, false, 16},
        McDenoiseParam{
            "RandomAccessOverlays8Bit", RANDOM_ACCESS, 8, 4, true, 40}),
    [](const ::testing::TestParamInfo<McDenoiseParam>& info) {
        return info.param.name;
    });

}  // namespace
