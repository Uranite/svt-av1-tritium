/*
* Copyright(c) 2020 Intel Corporation
*
* This source code is subject to the terms of the BSD 2 Clause License and
* the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
* was not distributed with this source code in the LICENSE file, you can
* obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
* Media Patent License 1.0 was not distributed with this source code in the
* PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
*/

/*
* This file contains only debug macros that are used during the development
* and are supposed to be cleaned up every tag cycle
* all macros must have the following format:
* - adding a new feature should be prefixed by FTR_
* - tuning a feature should be prefixed by TUNE_
* - enabling a feature should be prefixed by EN_
* - disabling a feature should be prefixed by DIS_
* - bug fixes should be prefixed by FIX_
* - code refactors should be prefixed by RFCTR_
* - code cleanups should be prefixed by CLN_
* - optimizations should be prefixed by OPT_
* - all macros must have a coherent comment explaining what the MACRO is doing
* - #if 0 / #if 1 are not to be used
*/

#ifndef EbDebugMacros_h
#define EbDebugMacros_h

// clang-format off

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#define OPT_NSC_STILL_IMAGE         1 // optimize the NSC path for still-image
#define OPT_SC_STILL_IMAGE          1 // optimize the SC path (scm1/scm2) for still-image

#define FIX_CDEF                    1
#if FIX_CDEF
#define FIX_Q_STRENGTH              1 // Fix initialization issue for use_qp_strength
#define OPT_CDEF_SKIP_CHROMA_BORDER 1 // Skip all chroma border copies when all UV strengths are 0 for the entire frame
#define CLN_FINISH_CDEF             1 // Remove dead allocation/write/free for use_qp_strength and use_reference_cdef_fs paths, unify CDEF strength-per-mi propagation through propagate_cdef_strength()
#define OPT_SC_CDEF_QP              1 // SC model for CDEF from QP for sc-class1
#endif

#define FTR_COUPLE_VLPD0_TXS        1 // Use fastest TXS when VERY_LIGHT_PD0 is active and default TXS is off
#if FTR_COUPLE_VLPD0_TXS
#define FTR_COUPLE_VLPD0_TXS_PER_SB 1
#endif
#define OPT_SC_RA                   1 // optimize the SC path (scm1) for RA
#define OPT_SC_RTC                  1 // optimize the SC path (scm1) for RTC

#define OPT_VLPD0_COST             1 // Optimize VLPD0 inter-depth partitioning
#define OPT_VLPD0_COST_BIS         1 // Faster VLPD0 inter-depth partitioning

#define TUNE_CDEF_LEVEL            1 // Simplify RTC CDEF level derivation (remove sc_class1 branch, split flat/3L)
#define OPT_CDEF_PRI_ONLY          1 // Only test pri-only strengths {0,15*4}
#define OPT_CDEF_SKIP_TH           1 // Replace use_skip_detector bool with configurable skip_th percentage

#define OPT_COEFF_SHAVING          1 // Post-quantization coefficient shaving: retract EOB and optionally zero block
#define OPT_COEFF_LEVEL            1 // Remove noise-level gating from derive_inter_coeff_level
#define OPT_PERIODIC_CDF_UPDATE    1 // Selective CDF disable for M12+ RTC
#define OPT_EC_INTERP              1 // Fixed EIGHTTAP_REGULAR when IFS is off
#define OPT_EC_DC_ONLY             1 // Fast entropy-coding path for eob==1 (DC-only blocks)
#define OPT_EC_MERGE_COEFF_LOOPS   1 // Merge backward/forward coefficient coding loops
#define OPT_GATE_SB_LAMBDA_MOD     1 // Gate stats_based_sb_lambda_modulation behind preset check

#define OPT_LPD1                   1 // Optimize LPD1: fixed-stage subpel, bias_fp, unify VLPD1, remove skip_zz_mv, fix rate-est
#define TUNE_LPD1_LEVEL            1 // Unified pic_lpd1_lvl derivation for RTC (remove sc_class1 special case)
#define OPT_STATS_MUTEX            1 // Use local accumulators for qindex stats instead of per-block mutex
#define OPT_SKIP_INTRA             1 // Skip INTRA using me-distortion
#define TUNE_SIMPLIFY_SETTINGS     1 // Unify settings by removing differences across prediction structures (flat vs 3L), content types (SC vs non-SC), and resolutions
#define TUNE_SHIFT_PRESETS_RTC     1 // Shift RTC presets: M10 -> M9, M11 -> M10, M12 -> M11; cap at M11

#define FIX_MR_STILL_IMAGE         1 // Restore MR for still-image

#define FTR_TUNE_VMAF  1 // Implement an unsharp preprocessing filter under TUNE-VMAF (--tune 5)
#define OPT_TUNE_VMAF  1 // TUNE-VMAF Optimizations: adaptive sharpening (per-QP + spatial MAD), noise gate (Laplacian),
                         // per-pixel High Frequency delta clip (QP-adaptive), chroma QP compensation, SIMD

#define FIX_CR_BAND_WRAPPING       1 // Handle wrapped range: sb_start > sb_end means [sb_start, total) union [0, sb_end)
#define OPT_CR_MOTION_GATE         1 // Cyclic-refresh motion gate: only boost SBs with low motion (dist < 2*norm_me_dist AND zero MV); disable CR for the frame if all SBs rejected to skip delta_q signaling overhead
#define OPT_ME_STATIC_B64          1 // Complete ME bypass for static 64x64 blocks: if L0/R0 zero-MV SAD < threshold, skip all HME + integer ME, set all MVs to (0,0), approximate sub-block SADs
#define FTR_ADD_RTC_M12_M13        1 // Add M12 and M13 RTC presets, with M12 as the fastest preset and M13 as an experimental faster option
#define OPT_LPD1_GLOBALMV_BYPASS   1 // Skip MDS0-2 (and MVP/ME refinement) for low-residual, zero-MV inter SQ blocks by injecting a forced GLOBALMV (IDENTITY) candidate straight into MDS3. GLOBALMV/GLOBAL_GLOBALMV code no mv_diff (AV1 spec 5.11.24): the decoder derives the MV directly from the frame-header global_motion[] params (IDENTITY -> (0,0)) without consulting the ref_mv_stack, so the MVP table is not needed for MV reconstruction.
#define OPT_LPD1_FAST_SKIP         1 // Predict skip from luma-only RD after luma TX, force chroma TX bypass
#define OPT_LPD1_CHROMA_SKIP       1 // Absolute chroma-residual SAD gate before svt_aom_full_loop_chroma_light_pd1
#define OPT_SUBPEL_FIXED_SEARCH    1 // Improve md_subpel_search_fixed_stage: th_normalizer early-exit, pred_variance_th check, remove ref bounds checks, early-exit in half/qpel loops
#define OPT_SUBPEL_CTRL            1 // Upgrade subpel-ctrl: new FIXED_STAGE cases 7-10, updated level assignments, remove early_neigh_check_exit param
#define OPT_VLPD0_PATH_INTER       1 // Bypass generate_md_stage_0_cand/md_stage_0, evaluate ME candidates directly on ref buffer (bit-exact)
#define OPT_PADDING                1 // Rewrite svt_aom_generate_padding: use svt_memset per-row for horizontal padding and svt_memcpy for vertical row replication; reduces 4 temp pointers to 2, eliminates sizeof(uint8_t) multiplier.
#define OPT_CDEF_PER_PLANE_SKIP    1 // Per-plane skip of src[] border copies + filter call in svt_av1_cdef_frame when that plane's (level,sec)=(0,0); linebuf/colbuf are saved directly from the unmodified rec_buff
#define OPT_APPROX_COEFF_RATE      1 // Mirror the existing luma cheap eob-based coeff-rate path on the chroma side (svt_aom_full_loop_uv / svt_aom_cuchroma_coding_loop): both for consistency and to bypass useless rate-estimation operation
#define OPT_SHAVE_COEFF_LIN        1 // Optimize coeff-shaving processing and restrict to isolated coeff removal (no energy-based skip)
#define OPT_MRP_HME_L0_DETECT      1 // Prune extra L0 refs in PD (RTC only) when the LAST-to-LAST2 (flat_ipp) or LAST-to-LAST3 (non-flat, base layer) HME-L0 SAD ratio exceeds early_hme_l0_prune_th/100
#define TUNE_RTC                   1 // Tune RTC
#define TUNE_RA                    1 // Tune RA
#define OPT_MAX_CAN_COUNT_RTC      1 // Derive tighter max_can_count from preset assuming 3L prediction structure
#define OPT_RTC_M13_FAST           1 // Apply aggressive speed optimizations for the experimental M13 preset


#define OPT_USE_HL0_FLAT  1 // Support hierarchical_levels 0 (flat) and 1 in LD CBR and RA 1L referencing

//FOR DEBUGGING - Do not remove
#define LOG_ENC_DONE            0 // log encoder job one
#define DEBUG_TPL               0 // Prints to debug TPL
#define DETAILED_FRAME_OUTPUT   0 // Prints detailed frame output from the library for debugging
#define DEBUG_BUFFERS           0 // Print process count and segments info
#define TUNE_CHROMA_SSIM        0 // Allows for Chroma and SSIM BDR-based Tuning
#define TUNE_CQP_CHROMA_SSIM    0 // Tune CQP qp scaling towards improved chroma and SSIM BDR

#define MIN_PIC_PARALLELIZATION 0 // Use the minimum amount of picture parallelization
#define SRM_REPORT              0 // Report SRM status
#define LAD_MG_PRINT            0 // Report LAD
#define RC_NO_R2R               0 // This is a debugging flag for RC and makes encoder to run with no R2R in RC mode
                                  // Note that the speed might impacted significantly
#if !RC_NO_R2R
#define FTR_KF_ON_FLY_SAMPLE         0 // Sample code to signal KF
#define FTR_RES_ON_FLY_SAMPLE        0 // Sample functions to change the resolution on the fly
#define FTR_RATE_ON_FLY_SAMPLE       0 // Sample functions to change bit rate
#define FTR_PRESET_ON_FLY_SAMPLE     0 // Sample functions to change preset on the fly
#define FTR_FRAME_RATE_ON_FLY_SAMPLE 0 // Sample functions to change frame rate
#define FTR_PER_FRAME_QUALITY_SAMPLE 0 // Sample functions to compute PSNR per frame
#endif
// Super-resolution debugging code
#define DEBUG_SCALING           0
#define DEBUG_TF                0
#define DEBUG_SUPERRES_RECODE   0
#define DEBUG_SUPERRES_ENERGY   0
#define DEBUG_RC_CAP_LOG        0 // Prints for RC cap

// Switch frame debugging code
#define DEBUG_SFRAME            0

// Variance Boost debugging code
#define DEBUG_VAR_BOOST         0
#define DEBUG_VAR_BOOST_QP      0
#define DEBUG_VAR_BOOST_STATS   0

// Anti-alias aware screen content mode debugging code
#define DEBUG_AA_SCM            0

// QP scaling debugging code
#define DEBUG_QP_SCALING        0

// Quantization matrices
#define DEBUG_QM_LEVEL          0
#define DEBUG_STARTUP_MG_SIZE   0
#define DEBUG_SEGMENT_QP        0
#define DEBUG_ROI               0
#ifdef __cplusplus
}
#endif // __cplusplus

// clang-format on

#endif // EbDebugMacros_h
