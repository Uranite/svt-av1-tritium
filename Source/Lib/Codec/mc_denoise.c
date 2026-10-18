/*
 * mc_denoise.c — encoder glue for the motion-compensated temporal denoiser.
 *
 * Bridges the mc_denoise.* algorithm modules (see mc_denoise.h) into the SVT
 * Picture Analysis stage. One SvtMcdState per encoder instance lives on the
 * shared EncodeContext (created in svt_av1_enc_init when --mc-denoise is on,
 * released by encode_context_dctor).
 *
 * Per-picture flow (called from the PA kernel before its regular body):
 *   1. Copy the arrived picture's visible planes into a ring slot (keyed by
 *      picture_number). The pcs goes on a small holding queue; its input
 *      buffers stay alive because the pcs wrapper is not released early.
 *   2. While the queue head has tr later frames copied (or at end of stream),
 *      denoise the head: lazily build the prefiltered search planes +
 *      pyramid + half-pel caches of every participating frame, run
 *      analyze+recalculate against each available neighbour at distance
 *      d=1..tr (both directions), degrain, and write the result back into the
 *      pcs input buffers.
 *   3. The released pcs values are returned to the PA kernel, which runs its
 *      normal body (padding, statistics, ...) on them.
 *
 * Buffer management rules (learned from the module self-tests):
 *   - Prefiltered search planes live in padded buffers (>= 64 samples on every
 *     side, edge-replicated margins, data pointer at the visible origin); ME
 *     reads up to 32 samples outside the visible area.
 *   - mcd_prefilter_mask_merge output is a third buffer set and never aliases
 *     the dfttest output; original pixels are never overwritten before degrain
 *     (pel caches compensate reference blocks from originals).
 *   - 10-bit pictures are stored split by SVT (8-bit MSB planes + packed
 *     2-bit LSB bitplanes); they are packed to native uint16 on the way in
 *     (svt_aom_compressed_pack_sb) and unpacked on the way out
 *     (svt_unpack_and_2bcompress).
 *
 * Known limitations:
 *   - Overlay pictures (enable-overlays) do not carry their own input pixels
 *     through the PA kernel, so they are not denoised; they pass through this
 *     stage untouched (their alt-ref pairs are denoised as usual).
 *   - In low-delay / all-intra prediction structures the holding queue is
 *     disabled (those modes never deliver an end-of-stream marker to the PA
 *     stage, so pictures must not be delayed): every picture is denoised on
 *     arrival, using whatever neighbour frames are already copied (the
 *     previous ones).
 */
#include <stdlib.h>
#include <string.h>

#include "mc_denoise.h"
#include "mc_denoise_glue.h"

#include "aom_dsp_rtcd.h"
#include "definitions.h"
#include "encode_context.h"
#include "pcs.h"
#include "pic_buffer_desc.h"
#include "pic_operators.h"
#include "sequence_control_set.h"
#include "svt_log.h"
#include "svt_malloc.h"

/* Margin replicated around every module-owned plane. ME reads at most 32
 * samples outside the visible area; the pyramid wants blk + 8 (<= 40 for
 * blksize 32) and the pel sets want >= 32. */
#define MCD_PAD 64
/* Grid alignment of module plane strides, in samples. */
#define MCD_STRIDE_ALIGN 16

typedef struct McdSlot {
    bool     valid;
    uint64_t picture_number;
    McdFrame frame;               /* planes[] point into pix_buf           */
    void*    pix_buf[3];          /* module copies of the original planes  */
    void*    fplane_buf[3];       /* padded prefiltered search planes      */
    void*    dft_buf[3];          /* dfttest scratch (freed after merge)   */
    void*    out_buf[3];          /* degrain destination planes            */
} McdSlot;

typedef struct McdHold {
    EbObjectWrapper*                wrapper;
    struct PictureParentControlSet* pcs;
    uint64_t                        picture_number;
} McdHold;

typedef struct SvtMcdState {
    EbHandle  mutex;
    McdConfig cfg;
    uint32_t  ring_len;                 /* 2*tr + 1                            */
    McdSlot*  ring;
    McdHold   hold[SVT_MCD_MAX_RELEASED]; /* sorted by picture_number           */
    uint32_t  hold_count;
    uint64_t  newest;                   /* newest picture_number copied        */
    bool      newest_valid;
} SvtMcdState;

/* ------------------------------------------------------------------ */
/* small helpers                                                       */

static void* mcd_alloc_zeroed(size_t bytes) {
    void* p = malloc(bytes);
    if (p) {
        memset(p, 0, bytes);
    }
    return p;
}

static uint32_t mcd_align_stride(uint32_t samples) {
    return (samples + (MCD_STRIDE_ALIGN - 1)) & ~(MCD_STRIDE_ALIGN - 1);
}

/* Allocate a padded plane: margin of MCD_PAD on all sides, data at the
 * visible origin. Returns MCD_SUCCESS or MCD_ERR_NOMEM. */
static int mcd_alloc_padded_plane(uint32_t w, uint32_t h, bool highbd, McdPlane* pl, void** base) {
    const uint32_t stride = mcd_align_stride(w + 2 * MCD_PAD);
    const size_t   esz    = highbd ? sizeof(uint16_t) : sizeof(uint8_t);
    uint8_t*       buf    = mcd_alloc_zeroed((size_t)stride * (h + 2 * MCD_PAD) * esz);
    if (!buf) {
        return MCD_ERR_NOMEM;
    }
    *base      = buf;
    pl->w      = w;
    pl->h      = h;
    pl->stride = stride;
    pl->highbd = highbd;
    pl->data   = buf + ((size_t)MCD_PAD * stride + MCD_PAD) * esz;
    return MCD_SUCCESS;
}

static void mcd_free_padded(void** buf) {
    free(*buf);
    *buf = NULL;
}

/* Edge-replicate the margin of a padded plane (mirrors the module e2e test:
 * first left/right margins on the visible rows, then whole padded rows for
 * top/bottom). */
static void mcd_replicate_margins(const McdPlane* pl) {
    const uint32_t pad    = MCD_PAD;
    const uint32_t stride = pl->stride;
    const size_t   esz    = pl->highbd ? sizeof(uint16_t) : sizeof(uint8_t);
    uint8_t*       vis    = (uint8_t*)pl->data;

    for (uint32_t y = 0; y < pl->h; ++y) {
        uint8_t* row = vis + (size_t)y * stride * esz;
        if (pl->highbd) {
            const uint16_t first = ((uint16_t*)row)[0];
            const uint16_t last  = ((uint16_t*)row)[pl->w - 1];
            for (uint32_t x = 1; x <= pad; ++x) {
                ((uint16_t*)row)[-(int32_t)x]      = first;
                ((uint16_t*)row)[pl->w - 1 + x] = last;
            }
        } else {
            memset(row - pad, row[0], pad);
            memset(row + pl->w, row[pl->w - 1], pad);
        }
    }
    /* Top/bottom margins: copy the whole padded row (including its left/right
     * margins, already filled above). base points at buffer row 0, column 0;
     * pl->data sits pad rows and pad columns into the buffer. The template
     * rows are the first/last VISIBLE padded rows (pl->data's row shifted
     * back to column 0), not the margin rows themselves. */
    uint8_t*     base     = vis - ((size_t)pad * stride + pad) * esz;
    const size_t rowb     = (size_t)(pl->w + 2 * pad) * esz;
    const uint8_t* firstrow = vis - (size_t)pad * esz;
    const uint8_t* lastrow  = vis + (size_t)(pl->h - 1) * stride * esz - (size_t)pad * esz;
    for (uint32_t y = 0; y < pad; ++y) {
        memcpy(base + (size_t)y * stride * esz, firstrow, rowb);
        memcpy(lastrow + (size_t)(y + 1) * stride * esz, lastrow, rowb);
    }
}

/* ------------------------------------------------------------------ */
/* picture geometry / copy                                             */

static void mcd_slot_free_caches(McdSlot* sl);

typedef struct McdGeom {
    uint32_t w[3], h[3];
    uint32_t x_ss, y_ss;
    uint32_t bitdepth;
    bool     highbd;
    bool     has_chroma;
} McdGeom;

static void mcd_geom_from_pic(McdGeom* g, const EbPictureBufferDesc* pic) {
    g->bitdepth   = pic->bit_depth;
    g->highbd     = pic->bit_depth > 8;
    g->x_ss       = (pic->color_format == EB_YUV444) ? 0 : 1;
    g->y_ss       = (pic->color_format == EB_YUV444 || pic->color_format == EB_YUV422) ? 0 : 1;
    g->w[0]       = pic->width;
    g->h[0]       = pic->height;
    g->has_chroma = (pic->u_buffer != NULL && pic->v_buffer != NULL);
    g->w[1]       = pic->width >> g->x_ss;
    g->h[1]       = pic->height >> g->y_ss;
    g->w[2]       = g->w[1];
    g->h[2]       = g->h[1];
}

/* Copy the visible planes of the SVT picture buffer into the slot's compact
 * module planes. Any caches from a previous picture in this slot are dropped. */
static int mcd_slot_copy_pixels(McdSlot* sl, const EbPictureBufferDesc* pic) {
    /* invalidate first: on a mid-copy failure (OOM) the slot must not keep a
     * stale picture_number over dangling plane pointers */
    sl->valid = false;
    McdGeom g;
    mcd_geom_from_pic(&g, pic);

    mcd_slot_free_caches(sl);

    sl->frame.num_planes  = g.has_chroma ? 3 : 1;
    sl->frame.x_ss        = g.x_ss;
    sl->frame.y_ss        = g.y_ss;
    sl->frame.bitdepth    = g.bitdepth;
    sl->frame.prefiltered = false;
    sl->frame.pyr_valid   = false;
    sl->frame.pel_valid   = false;

    for (uint32_t p = 0; p < sl->frame.num_planes; ++p) {
        const uint32_t w   = g.w[p];
        const uint32_t h   = g.h[p];
        const size_t   esz = g.highbd ? sizeof(uint16_t) : sizeof(uint8_t);

        free(sl->pix_buf[p]);
        sl->pix_buf[p] = mcd_alloc_zeroed((size_t)w * h * esz);
        if (!sl->pix_buf[p]) {
            return MCD_ERR_NOMEM;
        }

        McdPlane* pl = &sl->frame.planes[p];
        pl->w        = w;
        pl->h        = h;
        pl->stride   = w;
        pl->highbd   = g.highbd;
        pl->data     = sl->pix_buf[p];

        if (!g.highbd) {
            const uint8_t* src        = pic->buffer[p];
            const uint32_t src_stride = pic->stride[p];
            for (uint32_t y = 0; y < h; ++y) {
                memcpy((uint8_t*)sl->pix_buf[p] + (size_t)y * w, src + (size_t)y * src_stride, w);
            }
        } else {
            svt_aom_compressed_pack_sb(pic->buffer[p],
                                       pic->stride[p],
                                       pic->buffer_bit_inc[p],
                                       pic->stride_bit_inc[p] >> 2,
                                       (uint16_t*)sl->pix_buf[p],
                                       w,
                                       w,
                                       h);
        }
    }
    return MCD_SUCCESS;
}

/* Write the denoised compact planes back into the SVT picture buffer. */
static void mcd_write_back(McdSlot* sl, EbPictureBufferDesc* pic) {
    for (uint32_t p = 0; p < sl->frame.num_planes; ++p) {
        const McdPlane* pl = &sl->frame.planes[p];
        if (!pl->highbd) {
            uint8_t*       dst        = pic->buffer[p];
            const uint32_t dst_stride = pic->stride[p];
            const uint8_t* src        = (const uint8_t*)sl->out_buf[p];
            for (uint32_t y = 0; y < pl->h; ++y) {
                memcpy(dst + (size_t)y * dst_stride, src + (size_t)y * pl->w, pl->w);
            }
        } else {
            svt_unpack_and_2bcompress((uint16_t*)sl->out_buf[p],
                                      pl->w,
                                      pic->buffer[p],
                                      pic->stride[p],
                                      pic->buffer_bit_inc[p],
                                      pic->stride_bit_inc[p] >> 2,
                                      pl->w,
                                      pl->h);
        }
    }
}

/* ------------------------------------------------------------------ */
/* per-slot cache builders                                             */

/* Build the padded search planes (dfttest prefilter + luma-mask merge when
 * enabled, edge-replicated copies of the originals otherwise) and the search
 * pyramid over them. */
static int mcd_slot_build_search(SvtMcdState* s, McdSlot* sl) {
    McdFrame* fr = &sl->frame;
    int       rc;

    if (!fr->prefiltered) {
        /* a previous failed attempt may have left partially built buffers */
        mcd_slot_free_caches(sl);
        if (s->cfg.prefilter) {
            /* dfttest pass into a first padded buffer set */
            for (uint32_t p = 0; p < fr->num_planes; ++p) {
                rc = mcd_alloc_padded_plane(
                    fr->planes[p].w, fr->planes[p].h, fr->planes[p].highbd, &fr->fplanes[p], &sl->dft_buf[p]);
                if (rc) {
                    return rc;
                }
                rc = mcd_dfttest_plane(&fr->planes[p], &fr->fplanes[p], s->cfg.sigma_pct);
                if (rc) {
                    return rc;
                }
                mcd_replicate_margins(&fr->fplanes[p]);
            }
            /* luma-masked merge into a SECOND padded buffer set; the merge
             * output must not alias the dfttest output, and the original
             * planes are never touched. */
            McdPlane merged[3];
            void*    merged_base[3] = {NULL, NULL, NULL};
            for (uint32_t p = 0; p < fr->num_planes; ++p) {
                rc = mcd_alloc_padded_plane(
                    fr->planes[p].w, fr->planes[p].h, fr->planes[p].highbd, &merged[p], &merged_base[p]);
                if (rc) {
                    for (uint32_t q = 0; q < p; ++q) {
                        mcd_free_padded(&merged_base[q]);
                    }
                    return rc;
                }
            }
            rc = mcd_prefilter_mask_merge(
                fr->planes, fr->fplanes, merged, fr->num_planes, fr->x_ss, fr->y_ss, fr->bitdepth);
            for (uint32_t p = 0; p < fr->num_planes; ++p) {
                mcd_free_padded(&sl->dft_buf[p]); /* the dfttest scratch */
                fr->fplanes[p]    = merged[p];
                sl->fplane_buf[p] = merged_base[p];
                mcd_replicate_margins(&fr->fplanes[p]);
            }
            if (rc) {
                return rc;
            }
        } else {
            /* search over edge-replicated copies of the original pixels */
            for (uint32_t p = 0; p < fr->num_planes; ++p) {
                rc = mcd_alloc_padded_plane(
                    fr->planes[p].w, fr->planes[p].h, fr->planes[p].highbd, &fr->fplanes[p], &sl->fplane_buf[p]);
                if (rc) {
                    return rc;
                }
                const McdPlane* src     = &fr->planes[p];
                const size_t    esz     = src->highbd ? sizeof(uint16_t) : sizeof(uint8_t);
                uint8_t*        dst     = (uint8_t*)fr->fplanes[p].data;
                const uint8_t*  s_bytes = (const uint8_t*)src->data;
                for (uint32_t y = 0; y < src->h; ++y) {
                    memcpy(dst + (size_t)y * fr->fplanes[p].stride * esz,
                           s_bytes + (size_t)y * src->stride * esz,
                           (size_t)src->w * esz);
                }
                mcd_replicate_margins(&fr->fplanes[p]);
            }
        }
        fr->prefiltered = true;
    }

    if (!fr->pyr_valid) {
        rc = mcd_pyramid_create(fr->fplanes, fr->num_planes, MCD_PAD, 4, (McdPyramid**)&fr->pyr);
        if (rc) {
            return rc;
        }
        fr->pyr_valid = true;
    }
    return MCD_SUCCESS;
}

/* Half-pel phase planes over the ORIGINAL pixels, used by degrain to
 * compensate reference blocks. */
static int mcd_slot_build_pel(SvtMcdState* s, McdSlot* sl) {
    (void)s;
    if (!sl->frame.pel_valid) {
        int rc = mcd_pelset_create(sl->frame.planes, sl->frame.num_planes, MCD_PAD, (McdPelSet**)&sl->frame.pel);
        if (rc) {
            return rc;
        }
        sl->frame.pel_valid = true;
    }
    return MCD_SUCCESS;
}

static void mcd_slot_free_caches(McdSlot* sl) {
    if (sl->frame.pyr_valid) {
        mcd_pyramid_destroy((McdPyramid**)&sl->frame.pyr);
        sl->frame.pyr_valid = false;
        sl->frame.pyr       = NULL;
    }
    if (sl->frame.pel_valid) {
        mcd_pelset_destroy((McdPelSet**)&sl->frame.pel);
        sl->frame.pel_valid = false;
        sl->frame.pel       = NULL;
    }
    for (uint32_t p = 0; p < 3; ++p) {
        mcd_free_padded(&sl->dft_buf[p]);
        mcd_free_padded(&sl->fplane_buf[p]);
        free(sl->out_buf[p]);
        sl->out_buf[p] = NULL;
    }
    sl->frame.prefiltered = false;
}

/* ------------------------------------------------------------------ */
/* denoise one center frame                                            */

/* Denoise the frame in `center` and write the result back into `pic`.
 * Returns MCD_SUCCESS when the picture was denoised (write-back done),
 * MCD_ERR_PARAM when there was nothing to denoise, another MCD code on
 * module failure (the picture is then left untouched). */
static int mcd_denoise_center(SvtMcdState* s, McdSlot* center, EbPictureBufferDesc* pic) {
    McdFrame* fr = &center->frame;
    int       rc = mcd_slot_build_search(s, center);
    if (rc) {
        return rc;
    }

    const McdFrame*    refs[2 * MCD_MAX_TR];
    const McdMeResult* results[2 * MCD_MAX_TR];
    McdMeResult*       own[2 * MCD_MAX_TR];
    uint32_t           n_refs = 0;

    /* backward neighbours (in the past) first, then forward neighbours */
    for (int32_t pass = 0; pass < 2; ++pass) {
        for (uint32_t d = 1; d <= s->cfg.tr; ++d) {
            const int64_t neigh_num = (int64_t)center->picture_number + (pass == 0 ? -(int64_t)d : (int64_t)d);
            McdSlot*      neigh     = NULL;
            if (neigh_num >= 0) {
                McdSlot* cand = &s->ring[(uint64_t)neigh_num % s->ring_len];
                if (cand->valid && cand->picture_number == (uint64_t)neigh_num && cand != center) {
                    neigh = cand;
                }
            }
            if (!neigh) {
                continue; /* start/end of stream: just fewer refs */
            }
            rc = mcd_slot_build_search(s, neigh);
            if (rc == MCD_SUCCESS) {
                rc = mcd_slot_build_pel(s, neigh);
            }
            if (rc) {
                goto fail;
            }
            /* delta is positive when the reference is in the past */
            const int32_t delta = (pass == 0) ? (int32_t)d : -(int32_t)d;
            McdMeResult*  res   = NULL;
            rc                  = mcd_me_analyze(fr, &neigh->frame, delta, &s->cfg, &res);
            if (rc == MCD_SUCCESS) {
                rc = mcd_me_recalculate(fr, &neigh->frame, &s->cfg, res);
            }
            if (rc == MCD_ERR_PARAM) {
                /* geometry mismatch (e.g. dynamic resolution change): skip
                 * this neighbour and keep aggregating against the others */
                mcd_me_result_free(&res);
                continue;
            }
            if (rc) {
                mcd_me_result_free(&res);
                goto fail;
            }
            refs[n_refs]    = &neigh->frame;
            results[n_refs] = res;
            own[n_refs]     = res;
            ++n_refs;
        }
    }

    if (n_refs == 0) {
        /* nothing to aggregate against: leave the source untouched */
        return MCD_ERR_PARAM;
    }

    /* degrain into compact module planes */
    {
        McdPlane dst[3];
        for (uint32_t p = 0; p < fr->num_planes; ++p) {
            const size_t esz = fr->planes[p].highbd ? sizeof(uint16_t) : sizeof(uint8_t);
            if (!center->out_buf[p]) {
                center->out_buf[p] = mcd_alloc_zeroed((size_t)fr->planes[p].w * fr->planes[p].h * esz);
            }
            if (!center->out_buf[p]) {
                rc = MCD_ERR_NOMEM;
                goto fail;
            }
            dst[p].w      = fr->planes[p].w;
            dst[p].h      = fr->planes[p].h;
            dst[p].stride = fr->planes[p].w;
            dst[p].highbd = fr->planes[p].highbd;
            dst[p].data   = center->out_buf[p];
        }

        rc = mcd_degrain_frame(fr, refs, results, n_refs, &s->cfg, dst);
        if (rc == MCD_SUCCESS) {
            mcd_write_back(center, pic);
        }
    }

fail:
    for (uint32_t i = 0; i < n_refs; ++i) {
        mcd_me_result_free(&own[i]);
    }
    return rc;
}

/* ------------------------------------------------------------------ */
/* state lifecycle                                                     */

EbErrorType svt_mcd_state_create(void** state_ptr, const EbSvtAv1EncConfiguration* cfg) {
    SvtMcdState* s = NULL;

    EB_MALLOC(s, sizeof(*s));
    memset(s, 0, sizeof(*s));

    s->cfg.tr         = (uint8_t)(cfg->mc_denoise_tr ? cfg->mc_denoise_tr : 1);
    s->cfg.thsad      = cfg->mc_denoise_thsad;
    s->cfg.blksize    = cfg->mc_denoise_blksize;
    s->cfg.prefilter  = (uint8_t)(cfg->mc_denoise_prefilter ? 1 : 0);
    s->cfg.sigma_pct  = cfg->mc_denoise_sigma ? cfg->mc_denoise_sigma : 100;
    s->cfg.thscd1     = cfg->mc_denoise_thscd1;
    s->cfg.thscd2_pct = cfg->mc_denoise_thscd2;
    s->cfg.limit      = cfg->mc_denoise_limit;
    /* SATD (Hadamard) luma cost at the finest ME level, as in the
     * vsdenoise MVToolsPreset.HQ_SAD port target; not exposed as a knob */
    s->cfg.satd       = 1;

    s->ring_len = 2u * s->cfg.tr + 1u;
    EB_MALLOC(s->ring, sizeof(McdSlot) * s->ring_len);
    memset(s->ring, 0, sizeof(McdSlot) * s->ring_len);

    s->hold_count   = 0;
    s->newest_valid = false;

    EB_CREATE_MUTEX(s->mutex);

    *state_ptr = s;
    return EB_ErrorNone;
}

void svt_mcd_state_destroy(void** state_ptr) {
    SvtMcdState* s = (SvtMcdState*)*state_ptr;
    if (!s) {
        return;
    }
    for (uint32_t i = 0; i < s->ring_len; ++i) {
        McdSlot* sl = &s->ring[i];
        mcd_slot_free_caches(sl);
        for (uint32_t p = 0; p < 3; ++p) {
            free(sl->pix_buf[p]);
            sl->pix_buf[p] = NULL;
        }
        sl->valid = false;
    }
    EB_FREE(s->ring);
    EB_DESTROY_MUTEX(s->mutex);
    EB_FREE(s);
    *state_ptr = NULL;
}

/* ------------------------------------------------------------------ */
/* arrival hook                                                        */

/* Denoise the valid slot for pic_num (if any) into the pcs buffers.
 * Returns true when the pcs was denoised, false when it passes through
 * unchanged (never drops a picture). */
static bool mcd_process_picture(SvtMcdState* s, uint64_t pic_num, struct PictureParentControlSet* pcs) {
    if (!pcs || !pcs->enhanced_pic) {
        return false;
    }
    McdSlot* center = &s->ring[pic_num % s->ring_len];
    if (!center->valid || center->picture_number != pic_num) {
        return false;
    }
    const int rc = mcd_denoise_center(s, center, pcs->enhanced_pic);
    if (rc != MCD_SUCCESS && rc != MCD_ERR_PARAM) {
        SVT_ERROR("MC denoiser: failed to denoise picture %llu (rc=%d), passing it through undenoised\n",
                  (unsigned long long)pic_num,
                  rc);
        return false;
    }
    return rc == MCD_SUCCESS;
}

/* Insert into the holding queue keeping it sorted by picture_number. */
static void mcd_hold_push(SvtMcdState* s, EbObjectWrapper* wrapper, struct PictureParentControlSet* pcs,
                          uint64_t picture_number) {
    uint32_t pos = s->hold_count;
    while (pos > 0 && s->hold[pos - 1].picture_number > picture_number) {
        s->hold[pos] = s->hold[pos - 1];
        --pos;
    }
    s->hold[pos].wrapper        = wrapper;
    s->hold[pos].pcs            = pcs;
    s->hold[pos].picture_number = picture_number;
    ++s->hold_count;
}

/* Denoise + release the queue head into the caller's output arrays. */
static void mcd_release_head(SvtMcdState* s, EbObjectWrapper** out_wrappers,
                             struct PictureParentControlSet** out_pcs, uint32_t* n_out) {
    McdHold* head = &s->hold[0];
    mcd_process_picture(s, head->picture_number, head->pcs);
    out_wrappers[*n_out] = head->wrapper;
    out_pcs[*n_out]      = head->pcs;
    ++(*n_out);
    --s->hold_count;
    memmove(&s->hold[0], &s->hold[1], sizeof(McdHold) * s->hold_count);
}

uint32_t svt_mcd_pa_arrival(void* state,
                            struct PictureParentControlSet* pcs,
                            EbObjectWrapper* pcs_wrapper,
                            EbObjectWrapper** out_wrappers,
                            struct PictureParentControlSet** out_pcs) {
    SvtMcdState*        s       = (SvtMcdState*)state;
    SequenceControlSet* scs     = pcs->scs;
    EncodeContext*      enc_ctx = scs->enc_ctx;
    uint32_t            n_out   = 0;

    svt_block_on_mutex(s->mutex);

    /* End of stream: the last real picture arrives flagged in random-access
     * mode; the EncodeContext flag covers low-delay and overlay streams. */
    const bool eos = pcs->end_of_sequence_flag || enc_ctx->terminating_sequence_flag_received;
    /* Low-delay / all-intra: never hold pictures (see file header comment). */
    const bool no_hold = (scs->static_config.pred_structure == LOW_DELAY) || scs->allintra;

    if (!pcs->is_overlay) {
        /* copy the pixels into the ring slot for this picture number */
        McdSlot* sl = &s->ring[pcs->picture_number % s->ring_len];
        /* out-of-order guard: with several PA threads the mutex acquisition
         * order can differ from picture order. A very late arrival (older
         * than the whole live window) must neither clobber a slot still
         * needed as a neighbour nor regress "newest"; pass it through
         * undenoised instead. Requires a thread stalled across 2*tr+1
         * pictures, i.e. essentially never. */
        const bool stale = s->newest_valid &&
            pcs->picture_number + s->ring_len <= s->newest;
        if (stale) {
            out_wrappers[n_out] = pcs_wrapper;
            out_pcs[n_out]      = pcs;
            ++n_out;
        } else if (mcd_slot_copy_pixels(sl, pcs->enhanced_pic) == MCD_SUCCESS) {
            sl->picture_number = pcs->picture_number;
            sl->valid          = true;
            if (!s->newest_valid || pcs->picture_number > s->newest) {
                s->newest       = pcs->picture_number;
                s->newest_valid = true;
            }

            if (no_hold) {
                /* denoise immediately with whatever neighbours exist */
                mcd_process_picture(s, pcs->picture_number, pcs);
                out_wrappers[n_out] = pcs_wrapper;
                out_pcs[n_out]      = pcs;
                ++n_out;
            } else {
                /* capacity guard: ordered arrivals can hold at most tr + 1
                 * pictures, far below the output capacity; release the head
                 * early in case that assumption is ever violated */
                while (s->hold_count >= SVT_MCD_MAX_RELEASED - 1) {
                    mcd_release_head(s, out_wrappers, out_pcs, &n_out);
                }
                mcd_hold_push(s, pcs_wrapper, pcs, pcs->picture_number);
                /* release the head once all its temporal neighbours arrived */
                while (s->hold_count > 0 && s->newest_valid &&
                       s->hold[0].picture_number + s->cfg.tr <= s->newest) {
                    mcd_release_head(s, out_wrappers, out_pcs, &n_out);
                }
            }
        } else {
            /* could not copy (out of memory): pass the picture through
             * undenoised rather than dropping it */
            out_wrappers[n_out] = pcs_wrapper;
            out_pcs[n_out]      = pcs;
            ++n_out;
        }
    } else {
        /* overlay pictures are not denoised (they do not carry the input
         * pixels at this stage); forward them untouched */
        out_wrappers[n_out] = pcs_wrapper;
        out_pcs[n_out]      = pcs;
        ++n_out;
    }

    if (eos && s->hold_count > 0) {
        /* flush the holding queue: denoise every held picture with whatever
         * neighbours exist and release them all */
        while (s->hold_count > 0) {
            mcd_release_head(s, out_wrappers, out_pcs, &n_out);
        }
    }

    svt_release_mutex(s->mutex);
    return n_out;
}
