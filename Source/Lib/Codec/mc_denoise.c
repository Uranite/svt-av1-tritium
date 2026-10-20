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
 *   2. While the queue head has tr later frames settled (or at end of
 *      stream), pop the head and denoise it.
 *   3. The released pcs values are returned to the PA kernel, which runs its
 *      normal body (padding, statistics, ...) on them.
 *
 * Concurrency: the state mutex guards ONLY the cheap bookkeeping (ring
 * ownership, holding queue, watermark, end-of-stream flush, in-flight
 * count, build-job queue). The heavy work (prefilter, motion analysis,
 * degrain, write-back) runs OUTSIDE the mutex, so Picture Analysis threads
 * denoise different frames in parallel; popping a picture from the holding
 * queue grants the popping thread exclusive ownership of that picture's
 * denoise + write-back.
 *
 * Eager cache builds: a frame's neighbour caches (prefiltered search planes
 * + pyramid, degrain pelset, ME pelset) are pure functions of its copied
 * pixels, so each arrival posts its buffer to a small pool of helper
 * threads that builds all three caches while the current head's ME/degrain
 * runs, instead of every build serializing in front of whichever later
 * denoise needs the frame as a neighbour. The helpers call the SAME
 * idempotent, build_mutex-serialized builders the lazy path uses, so a
 * cache is built exactly once whichever thread gets there first, a denoise
 * that reaches a still-building cache simply waits on that buffer's
 * build_mutex (strictly cheaper than building it itself), and outputs are
 * bit-identical by construction. Releases keep keying on the pixel-copy
 * watermark; the helpers only change WHEN a build happens, never WHAT.
 *
 * DETERMINISM: the denoise result of a frame must depend only on the input,
 * never on thread timing. Two rules provide this:
 *   - Arrivals are ORDERED: a picture waits (with the mutex released) until
 *     the release watermark has reached its predecessor, so slots are filled
 *     and settled in picture order exactly as a serial arrival loop would.
 *     The wait is empty in the common case and only absorbs genuine
 *     reordering between PA threads; the copies it orders cost well under a
 *     millisecond at 1080p, against tens of milliseconds of denoise.
 *   - Heads are released strictly by the watermark (head + tr <= watermark),
 *     which guarantees every temporal neighbour is settled — and, because
 *     arrival is ordered, still resident in its slot — when the head pops.
 *     The queue therefore never exceeds tr + 1 entries in normal operation.
 *
 * Buffer lifetime: every ring slot owns a reference-counted McdBufs object
 * (pixel copies + caches). A denoising pass takes references (under the
 * mutex) on its center buffer and every neighbour buffer before releasing
 * the mutex, so a later arrival that would recycle one of their slots
 * installs a FRESH buffer object instead; the displaced object dies with its
 * last reference. Recycling pixel data under a live reader is never done.
 * Teardown safety: the state keeps a count of in-flight denoises;
 * svt_mcd_state_destroy waits on a semaphore until every in-flight pass has
 * finished. (The encoder teardown joins the PA threads before
 * encode_context_dctor runs, so the wait is purely defensive.)
 *
 * Buffer management rules (learned from the module self-tests):
 *   - Prefiltered search planes live in padded buffers (>= 64 samples on
 *     every side, edge-replicated margins, data pointer at the visible
 *     origin); ME reads up to 32 samples outside the visible area.
 *   - The luma-mask merge runs in place over the dfttest output (the padded
 *     set becomes the search planes); original pixels are never overwritten
 *     before degrain (pel caches compensate reference blocks from
 *     originals).
 *   - 10-bit pictures are stored split by SVT (8-bit MSB planes + packed
 *     2-bit LSB bitplanes); they are packed to native uint16 on the way in
 *     (svt_aom_compressed_pack_sb) and unpacked on the way out
 *     (svt_unpack_and_2bcompress).
 *
 * Known limitations:
 *   - Overlay pictures (enable-overlays) do not carry their own input pixels
 *     through the PA kernel, so they are not denoised; they pass through
 *     this stage untouched (their alt-ref pairs are denoised as usual).
 *   - In low-delay / all-intra prediction structures the holding queue is
 *     disabled (those modes never deliver an end-of-stream marker to the PA
 *     stage, so pictures must not be delayed): every picture is denoised on
 *     arrival, using whatever neighbour frames are already copied (the
 *     previous ones).
 */
#include <stdlib.h>
#include <string.h>
#ifndef _WIN32
#include <time.h>
#endif

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
#include "svt_threads.h"

/* Margin of the module-owned SEARCH planes (prefiltered fplanes and their
 * pyramid). ME clamps every candidate read to +-32 samples outside the
 * visible area, and the pyramid only reads the visible core of its source,
 * so 40 samples are enough; the plane CONTENT does not depend on the margin
 * width, which makes the trim bit-exact. The degrain pel sets get
 * 32 + blksize/2 for the same reason (see mcd_bufs_build_pel). */
#define MCD_PAD_SEARCH 40
/* Grid alignment of module plane strides, in samples. */
#define MCD_STRIDE_ALIGN 16

/* Poll cadence / bounds for the (rarely taken) ordered-arrival and
 * end-of-stream waits. The caps only matter when a picture never arrives
 * (encoder teardown); the bounds keep such a degenerate case from hanging. */
#define MCD_WAIT_US 200

/* Keep the ordered-arrival polls short without depending on POSIX usleep,
 * which is not available in the Windows build. Windows sleep granularity is
 * typically 1 ms; that is acceptable because these waits are rare. */
static void mcd_wait_short(void) {
#ifdef _WIN32
    Sleep(1);
#else
    const struct timespec delay = {0, MCD_WAIT_US * 1000};
    nanosleep(&delay, NULL);
#endif
}

/* Number of eager cache-build helper threads. Measured at 1080p (Ryzen
 * 8840HS, 8 physical cores) against the HEAD lazy baseline (~3.95 fps at
 * tr=1): 1 helper ~4.13 fps, 2 helpers ~4.18 fps, 3 helpers ~4.13 fps; at
 * tr=3 ~2.52 (1 helper) vs ~2.56 (2 helpers) fps. Two threads were picked:
 * consistently (if marginally) ahead of one at both tr=1 and tr=3 — the
 * second thread covers arrival bursts (stream start, scene changes, higher
 * tr) where up to ring_len builds can queue behind the one the current
 * head is waiting on — while a third bought nothing at tr=1 (once the
 * fresh neighbour's build overlaps the backward-neighbour ME, build
 * capacity stops being the bottleneck; the denoise threads saturate
 * first). */
#define MCD_BUILD_THREADS 2

typedef struct McdBufs {
    uint64_t picture_number;
    bool     valid;               /* pixels copied for picture_number      */
    bool     denoised;            /* had its (one) denoise turn; it can only
                                     serve as a neighbour afterwards       */
    McdFrame frame;               /* planes[] point into pix_buf           */
    void*    pix_buf[3];          /* module copies of the original planes;
                                     returned to the state's pix pool once
                                     the picture itself has been denoised
                                     (it is never read again afterwards)  */
    size_t   pix_bytes[3];        /* 0 when pix_buf[p] is not owned        */
    void*    fplane_buf[3];       /* padded prefiltered search planes      */
    void*    dft_buf[3];          /* dfttest scratch (freed after merge)   */
    void*    out_buf[3];          /* degrain destination planes            */
    EbHandle build_mutex;         /* serializes lazy per-buffer cache builds */
    int32_t  refs;                /* guarded by SvtMcdState.mutex          */
    /* eager-build job queue link (intrusive FIFO; every field guarded by
     * SvtMcdState.mutex). A queued/being-built buffer carries one extra
     * reference, so it can never be recycled or freed under the helpers. */
    struct McdBufs* build_next;
    bool            build_queued; /* on the queue or mid-build by a helper */
} McdBufs;

typedef struct McdSlot {
    McdBufs* buf;                 /* the slot holds one owning reference   */
} McdSlot;

typedef struct McdHold {
    EbObjectWrapper*                wrapper;
    struct PictureParentControlSet* pcs;
    uint64_t                        picture_number;
} McdHold;

/* One popped (owned) denoise job: the hold-queue head plus references on the
 * buffers it will read (center + temporal neighbours), all taken under the
 * state mutex before the heavy work starts. scratch: ME predictor-field
 * scratch taken from the state's pool for the duration of the denoise. */
typedef struct McdPopped {
    EbObjectWrapper*                wrapper;
    struct PictureParentControlSet* pcs;
    uint64_t                        picture_number;
    McdBufs*                        center;
    McdBufs*                        neigh[2 * MCD_MAX_TR];
    int32_t                         neigh_delta[2 * MCD_MAX_TR];
    uint32_t                        n_neigh;
    McdMeScratch*                   me_scratch;
} McdPopped;

/* Pooled ME scratch: analyzed frames need grow-to-fit predictor fields
 * (~0.5 MB at 1080p blk16); a small freelist removes the per-frame churn.
 * Take/put under the state mutex; entries are owned by exactly one denoise
 * at a time. */
#define MCD_ME_POOL_MAX 16

typedef struct SvtMcdState {
    EbHandle  mutex;
    McdConfig cfg;
    uint32_t  ring_len;                 /* 2*tr + 1                            */
    McdSlot*  ring;
    McdHold   hold[SVT_MCD_MAX_RELEASED]; /* sorted by picture_number           */
    uint32_t  hold_count;
    uint64_t  newest;                   /* newest picture_number copied        */
    bool      newest_valid;
    /* Release watermark: the largest N such that every picture 0..N has been
     * through the arrival hook (copied, or settled without a copy). Arrivals
     * are ordered on it (see DETERMINISM above) and heads are released once
     * head + tr <= watermark. */
    uint64_t  watermark;
    bool      watermark_valid;
    /* settled-set window over (watermark, watermark + 64]: a picture that
     * settles out of order records its bit here instead of being LOST (the
     * old single-step settle froze the watermark forever after any
     * out-of-order settle, which backed up the hold queue until the
     * capacity guard force-released heads with arbitrary missing
     * neighbours). */
    uint64_t  settled_bits;
    uint32_t  in_flight;                /* denoises running outside the mutex  */
    uint32_t  in_copy;                  /* pixel copies running outside the
                                           mutex (defensive teardown wait)    */
    EbHandle  done_semaphore;           /* posted once per finished denoise    */
    McdDftCtx* dft_ctx;                 /* shared dfttest constant tables      */
    /* scratch pools (all guarded by mutex) */
    McdMeScratch* me_pool[MCD_ME_POOL_MAX];
    uint32_t      me_pool_n;
    void*         pix_pool[3];          /* one recycled copy buffer per plane  */
    size_t        pix_pool_bytes[3];    /* 0 when the slot is empty            */
    /* eager cache-build helper pool (guarded by mutex where noted) */
    McdBufs*  build_head;               /* FIFO front                          */
    McdBufs*  build_tail;               /* FIFO back                           */
    EbHandle  build_semaphore;          /* one token per queued job            */
    EbHandle  build_threads[MCD_BUILD_THREADS];
    uint32_t  build_threads_n;          /* threads actually running            */
    bool      build_shutdown;           /* destroy asked helpers to exit       */
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

/* Allocate a padded plane: margin of `pad` on all sides, data at the
 * visible origin. Returns MCD_SUCCESS or MCD_ERR_NOMEM. */
static int mcd_alloc_padded_plane(uint32_t w, uint32_t h, bool highbd, McdPlane* pl, void** base, uint32_t pad) {
    const uint32_t stride = mcd_align_stride(w + 2 * pad);
    const size_t   esz    = highbd ? sizeof(uint16_t) : sizeof(uint8_t);
    /* uninitialized malloc is safe: every byte that is ever read is written
     * first — the visible core is filled by the copy/dfttest pass, the
     * replicated margin by mcd_replicate_margins; the stride-alignment slack
     * columns past w + 2*pad are never read by any module (all reads are
     * clamped to the core +/- the replicated margin) */
    uint8_t*       buf    = malloc((size_t)stride * (h + 2 * pad) * esz);
    if (!buf) {
        return MCD_ERR_NOMEM;
    }
    *base      = buf;
    pl->w      = w;
    pl->h      = h;
    pl->stride = stride;
    pl->highbd = highbd;
    pl->data   = buf + ((size_t)pad * stride + pad) * esz;
    return MCD_SUCCESS;
}

static void mcd_free_padded(void** buf) {
    free(*buf);
    *buf = NULL;
}

/* Edge-replicate the margin of a padded plane (mirrors the module e2e test:
 * first left/right margins on the visible rows, then whole padded rows for
 * top/bottom). */
static void mcd_replicate_margins(const McdPlane* pl, uint32_t pad) {
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
    uint8_t*       lastrow_w = (uint8_t*)lastrow; /* mutable alias for the dest */
    for (uint32_t y = 0; y < pad; ++y) {
        memcpy(base + (size_t)y * stride * esz, firstrow, rowb);
        memcpy(lastrow_w + (size_t)(y + 1) * stride * esz, lastrow, rowb);
    }
}

/* ------------------------------------------------------------------ */
/* reference-counted buffer objects                                    */

static void mcd_bufs_free_caches(McdBufs* b) {
    if (b->frame.pyr_valid) {
        mcd_pyramid_destroy((McdPyramid**)&b->frame.pyr);
        b->frame.pyr_valid = false;
        b->frame.pyr       = NULL;
    }
    if (b->frame.pel_valid) {
        mcd_pelset_destroy((McdPelSet**)&b->frame.pel);
        b->frame.pel_valid = false;
        b->frame.pel       = NULL;
    }
    if (b->frame.me_pel_valid) {
        mcd_pelset_destroy((McdPelSet**)&b->frame.me_pel);
        b->frame.me_pel_valid = false;
        b->frame.me_pel       = NULL;
    }
    for (uint32_t p = 0; p < 3; ++p) {
        mcd_free_padded(&b->dft_buf[p]);
        mcd_free_padded(&b->fplane_buf[p]);
        free(b->out_buf[p]);
        b->out_buf[p] = NULL;
    }
    b->frame.prefiltered = false;
}

/* Reset a buffer whose only reference is the owning slot's, so it can be
 * refilled for a new picture. Must only be called when b->refs == 1. */
static void mcd_bufs_reset(McdBufs* b) {
    /* clear `denoised` under build_mutex BEFORE freeing anything: the flag is
     * read (under build_mutex, inside mcd_bufs_maybe_recycle_pix) by later
     * centres' neighbour loops to decide whether this buffer's pixel copies
     * are dead. A stale `true` from the buffer's PREVIOUS occupant would pool
     * or free the pixel copies of the CURRENT picture while its own denoise
     * turn is still reading them (observed as SIGSEGV in degrain_init_row and
     * as silently wrong pixels). */
    svt_block_on_mutex(b->build_mutex);
    b->denoised = false;
    svt_release_mutex(b->build_mutex);
    mcd_bufs_free_caches(b);
    for (uint32_t p = 0; p < 3; ++p) {
        free(b->pix_buf[p]);
        b->pix_buf[p]   = NULL;
        b->pix_bytes[p] = 0;
    }
    b->valid = false;
}

/* Free a buffer whose reference count reached zero. */
static void mcd_bufs_destroy(McdBufs* b) {
    mcd_bufs_reset(b);
    if (b->build_mutex) {
        EB_DESTROY_MUTEX(b->build_mutex);
    }
    free(b);
}

/* Drop one reference (state mutex held); free on the last one. */
static void mcd_bufs_unref(SvtMcdState* s, McdBufs* b) {
    (void)s;
    if (b && --b->refs == 0) {
        mcd_bufs_destroy(b);
    }
}

/* ------------------------------------------------------------------ */
/* scratch pools (state mutex held by every caller)                    */

/* Take an ME scratch from the pool (or create one). */
static McdMeScratch* mcd_me_pool_take(SvtMcdState* s) {
    if (s->me_pool_n > 0) {
        return s->me_pool[--s->me_pool_n];
    }
    McdMeScratch* sc = NULL;
    if (mcd_me_scratch_create(&sc) != MCD_SUCCESS) {
        return NULL;
    }
    return sc;
}

/* Return an ME scratch to the pool (surplus is destroyed). */
static void mcd_me_pool_put(SvtMcdState* s, McdMeScratch* sc) {
    if (!sc) {
        return;
    }
    if (s->me_pool_n < MCD_ME_POOL_MAX) {
        s->me_pool[s->me_pool_n++] = sc;
    } else {
        mcd_me_scratch_destroy(&sc);
    }
}

/* Take a recycled copy buffer of exactly `bytes` for plane p (NULL = caller
 * must malloc; the caller records the ownership in b->pix_bytes). */
static void* mcd_pix_pool_take(SvtMcdState* s, uint32_t p, size_t bytes) {
    if (s->pix_pool[p] && s->pix_pool_bytes[p] == bytes) {
        void* buf     = s->pix_pool[p];
        s->pix_pool[p]      = NULL;
        s->pix_pool_bytes[p] = 0;
        return buf;
    }
    return NULL;
}

/* Return a copy buffer to the pool; keep the largest buffer seen for the
 * plane role, free everything else (geometry is constant per stream, so the
 * steady state is one exact-size hit per plane per frame). */
static void mcd_pix_pool_put(SvtMcdState* s, uint32_t p, void* buf, size_t bytes) {
    if (!buf) {
        return;
    }
    if (bytes > s->pix_pool_bytes[p]) {
        free(s->pix_pool[p]);
        s->pix_pool[p]       = buf;
        s->pix_pool_bytes[p]  = bytes;
    } else {
        free(buf);
    }
}

/* Return a buffer's original pixel copies to the state pool once they are
 * provably dead: the picture had its denoise turn (it can only be a
 * neighbour afterwards, and neighbours are read only through their caches)
 * AND every cache a future neighbour pass needs (search planes + pyramid +
 * both pelsets) is already built — early-stream frames have their caches
 * completed later, by the next centre's neighbour loop, so the buffers stay
 * alive until then.
 * Locking: state mutex held; the buffer's build_mutex is taken for the
 * cache-validity flags (they are written under it). The only other nesting
 * of these mutexes is this same s->mutex -> build_mutex order, so no
 * inversion can arise. */
static void mcd_bufs_maybe_recycle_pix(SvtMcdState* s, McdBufs* b) {
    const McdFrame* fr = &b->frame;
    svt_block_on_mutex(b->build_mutex);
    const int dead = b->denoised && fr->prefiltered && fr->pyr_valid &&
                     fr->pel_valid && fr->me_pel_valid;
    if (dead) {
        for (uint32_t p = 0; p < 3; ++p) {
            mcd_pix_pool_put(s, p, b->pix_buf[p], b->pix_bytes[p]);
            b->pix_buf[p]   = NULL;
            b->pix_bytes[p] = 0;
        }
        /* frame.planes[].data now dangle by design; nothing dereferences
         * them (the valid flags short-circuit every builder, degrain reads
         * only CENTRE planes, ME reads pyramids/pelsets) */
    }
    svt_release_mutex(b->build_mutex);
}

/* ------------------------------------------------------------------ */
/* picture geometry / copy                                             */

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

/* Copy the visible planes of the SVT picture buffer into the buffer object's
 * compact module planes. Any caches from a previous picture are dropped.
 * Runs OUTSIDE the state mutex (item: the 10-bit pack is expensive); pool
 * buffers are pre-taken under the mutex before the call. */
static int mcd_bufs_copy_pixels(SvtMcdState* s, McdBufs* b, const EbPictureBufferDesc* pic) {
    /* invalidate first: on a mid-copy failure (OOM) the buffer must not keep
     * a stale picture_number over dangling plane pointers */
    (void)s;
    b->valid = false;
    McdGeom g;
    mcd_geom_from_pic(&g, pic);

    mcd_bufs_free_caches(b);

    b->frame.num_planes  = g.has_chroma ? 3 : 1;
    b->frame.x_ss        = g.x_ss;
    b->frame.y_ss        = g.y_ss;
    b->frame.bitdepth    = g.bitdepth;
    b->frame.prefiltered = false;
    b->frame.pyr_valid   = false;
    b->frame.pel_valid   = false;

    for (uint32_t p = 0; p < b->frame.num_planes; ++p) {
        const uint32_t w   = g.w[p];
        const uint32_t h   = g.h[p];
        const size_t   esz = g.highbd ? sizeof(uint16_t) : sizeof(uint8_t);
        const size_t   bytes = (size_t)w * h * esz;

        /* keep an already-owned buffer of the right size (steady state);
         * otherwise the pool was pre-taken under the mutex, else malloc */
        if (!b->pix_buf[p] || b->pix_bytes[p] != bytes) {
            free(b->pix_buf[p]);
            b->pix_buf[p]   = malloc(bytes);
            b->pix_bytes[p] = b->pix_buf[p] ? bytes : 0;
        }
        /* uninitialized memory is fine: the compact planes are fully
         * overwritten below (row memcpy for 8-bit, full-plane
         * svt_aom_compressed_pack_sb for 10-bit) before any read */
        if (!b->pix_buf[p]) {
            return MCD_ERR_NOMEM;
        }

        McdPlane* pl = &b->frame.planes[p];
        pl->w        = w;
        pl->h        = h;
        pl->stride   = w;
        pl->highbd   = g.highbd;
        pl->data     = b->pix_buf[p];

        if (!g.highbd) {
            const uint8_t* src        = pic->buffer[p];
            const uint32_t src_stride = pic->stride[p];
            for (uint32_t y = 0; y < h; ++y) {
                memcpy((uint8_t*)b->pix_buf[p] + (size_t)y * w, src + (size_t)y * src_stride, w);
            }
        } else {
            svt_aom_compressed_pack_sb(pic->buffer[p],
                                       pic->stride[p],
                                       pic->buffer_bit_inc[p],
                                       pic->stride_bit_inc[p] >> 2,
                                       (uint16_t*)b->pix_buf[p],
                                       w,
                                       w,
                                       h);
        }
    }
    return MCD_SUCCESS;
}

/* Write the denoised compact planes back into the SVT picture buffer. */
static void mcd_write_back(McdBufs* b, EbPictureBufferDesc* pic) {
    for (uint32_t p = 0; p < b->frame.num_planes; ++p) {
        const McdPlane* pl = &b->frame.planes[p];
        if (!pl->highbd) {
            uint8_t*       dst        = pic->buffer[p];
            const uint32_t dst_stride = pic->stride[p];
            const uint8_t* src        = (const uint8_t*)b->out_buf[p];
            for (uint32_t y = 0; y < pl->h; ++y) {
                memcpy(dst + (size_t)y * dst_stride, src + (size_t)y * pl->w, pl->w);
            }
        } else {
            svt_unpack_and_2bcompress((uint16_t*)b->out_buf[p],
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
/* per-buffer cache builders                                           */
/*                                                                    */
/* All builders serialize on the buffer's build_mutex, so concurrent   */
/* denoising passes sharing a reference frame build each cache once.   */
/* The mutex is never held across the heavy module calls that merely   */
/* READ the caches (motion analysis, degrain), and never nested, so    */
/* no lock ordering can arise. NOTE: the cache-validity flags are      */
/* written under build_mutex and must only be read while holding it   */
/* (or after having held it on the same thread), so there is           */
/* deliberately no unlocked fast path in these builders.               */

/* Build the padded search planes (dfttest prefilter + luma-mask merge when
 * enabled, edge-replicated copies of the originals otherwise) and the search
 * pyramid over them. */
static int mcd_bufs_build_search(SvtMcdState* s, McdBufs* b) {
    McdFrame* fr = &b->frame;
    int       rc = MCD_SUCCESS;

    svt_block_on_mutex(b->build_mutex);
    if (fr->prefiltered && fr->pyr_valid) {
        goto unlock;
    }
    if (!fr->prefiltered) {
        /* a previous failed attempt may have left partially built buffers */
        mcd_bufs_free_caches(b);
        if (s->cfg.prefilter) {
            /* dfttest pass into a first padded buffer set */
            for (uint32_t p = 0; p < fr->num_planes; ++p) {
                rc = mcd_alloc_padded_plane(
                    fr->planes[p].w, fr->planes[p].h, fr->planes[p].highbd, &fr->fplanes[p], &b->dft_buf[p],
                    MCD_PAD_SEARCH);
                if (rc) {
                    goto unlock;
                }
                rc = mcd_dfttest_plane_with_ctx(s->dft_ctx, &fr->planes[p], &fr->fplanes[p]);
                if (rc) {
                    goto unlock;
                }
                /* NOTE: no margin replication here. The luma-mask merge below
                 * reads only the visible core of the dfttest output, and the
                 * dft_buf set is scratch: margins are replicated once, after
                 * the merge. */
            }
            /* luma-masked merge IN PLACE over the dfttest output: the merge
             * reads filt(x,y)/src(x,y) strictly before writing out(x,y) at
             * the same coordinate, so out == filt is per-pixel read-before-
             * write safe; the original planes are never touched. */
            rc = mcd_prefilter_mask_merge(
                fr->planes, fr->fplanes, fr->fplanes, fr->num_planes, fr->x_ss, fr->y_ss, fr->bitdepth);
            if (rc) {
                goto unlock;
            }
            for (uint32_t p = 0; p < fr->num_planes; ++p) {
                /* the padded set becomes the search planes (ownership moves
                 * from dft_buf to fplane_buf) */
                b->fplane_buf[p] = b->dft_buf[p];
                b->dft_buf[p]    = NULL;
                mcd_replicate_margins(&fr->fplanes[p], MCD_PAD_SEARCH);
            }
        } else {
            /* search over edge-replicated copies of the original pixels */
            for (uint32_t p = 0; p < fr->num_planes; ++p) {
                rc = mcd_alloc_padded_plane(
                    fr->planes[p].w, fr->planes[p].h, fr->planes[p].highbd, &fr->fplanes[p], &b->fplane_buf[p],
                    MCD_PAD_SEARCH);
                if (rc) {
                    goto unlock;
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
                mcd_replicate_margins(&fr->fplanes[p], MCD_PAD_SEARCH);
            }
        }
        fr->prefiltered = true;
    }

    if (!fr->pyr_valid) {
        rc = mcd_pyramid_create(fr->fplanes, fr->num_planes, MCD_PAD_SEARCH, 4, (McdPyramid**)&fr->pyr);
        if (rc) {
            goto unlock;
        }
        fr->pyr_valid = true;
    }

unlock:
    svt_release_mutex(b->build_mutex);
    return rc;
}

/* Half-pel phase planes over the ORIGINAL pixels, used by degrain to
 * compensate reference blocks. */
static int mcd_bufs_build_pel(SvtMcdState* s, McdBufs* b) {
    (void)s;
    int rc = MCD_SUCCESS;
    svt_block_on_mutex(b->build_mutex);
    if (!b->frame.pel_valid) {
        /* 32 (ME's candidate clamp) + blk/2 covers every degrain fetch for
         * motion up to ~tens of pixels; the fetch clamps internally, so
         * larger vectors stay in bounds (the content of the phase planes
         * does not depend on the margin width, so normal-motion fetches are
         * bit-identical to the previous 64-sample margin). */
        const uint32_t pad = 32u + (s->cfg.blksize >> 1);
        rc = mcd_pelset_create(b->frame.planes,
                               b->frame.num_planes,
                               b->frame.bitdepth,
                               pad,
                               (McdPelSet**)&b->frame.pel);
        if (rc == MCD_SUCCESS) {
            b->frame.pel_valid = true;
        }
    }
    svt_release_mutex(b->build_mutex);
    return rc;
}

/* Half-pel phase planes over the PREFILTERED luma, shared by every
 * analyze/recalculate pass that uses this buffer as its reference (the
 * motion module would otherwise rebuild an identical pelset per call — up to
 * 2 * tr times per frame). Built at the analyze grid's margin; the
 * recalculate pass clamps its fetches to its own (smaller) expected margin,
 * which is bit-identical to a local build. */
static int mcd_bufs_build_me_pel(SvtMcdState* s, McdBufs* b) {
    int rc = MCD_SUCCESS;
    svt_block_on_mutex(b->build_mutex);
    if (!b->frame.me_pel_valid) {
        const uint32_t pad = 32u + (s->cfg.blksize >> 1); /* MCD_ME_EDGE_PAD + blk/2 */
        const McdPlane* src = b->frame.prefiltered ? &b->frame.fplanes[0] : &b->frame.planes[0];
        /* The padded search plane's margin is MCD_PAD_SEARCH; when it equals
         * the ME margin, fplanes[0] IS byte-identical to what phase 0 of the
         * pelset would be (same pad, stride and edge replication), so adopt
         * its storage instead of rebuilding one padded luma plane per slot.
         * mcd_bufs_free_caches destroys me_pel before freeing fplane_buf. */
        const McdPlane* borrow =
            (b->fplane_buf[0] && pad == MCD_PAD_SEARCH) ? &b->frame.fplanes[0] : NULL;
        rc = mcd_pelset_create_borrow0(src,
                                       1,
                                       b->frame.bitdepth,
                                       pad,
                                       borrow,
                                       (McdPelSet**)&b->frame.me_pel);
        if (rc == MCD_SUCCESS) {
            b->frame.me_pel_valid = true;
        }
    }
    svt_release_mutex(b->build_mutex);
    return rc;
}

/* ------------------------------------------------------------------ */
/* eager cache-build helper pool                                       */
/*                                                                    */
/* Each arrival whose pixel copy succeeded posts its buffer to a FIFO  */
/* of "fully build" jobs (search planes + pyramid, degrain pelset, ME  */
/* pelset); MCD_BUILD_THREADS helpers drain it, overlapping the build  */
/* cost with the denoise/ME of the current head. The helpers run the   */
/* SAME builders as the lazy denoise path, so a cache is still built   */
/* exactly once (the per-buffer build_mutex serializes both paths) and */
/* a failed build leaves the validity flags false, which the lazy path */
/* retries at first use — identical outcomes either way.               */
/*                                                                    */
/* Locking: queue state is guarded by the state mutex; the builds      */
/* themselves run with NO state mutex held (the builders only take the */
/* buffer's build_mutex — never both at once, so the existing          */
/* s->mutex -> build_mutex nesting stays the only one). Waking uses a  */
/* counting semaphore (one token per job) instead of a condition       */
/* variable: the codebase's CondVar waits on a private mutex and can   */
/* not be paired with the state mutex, which would force a check-then- */
/* wait window; a token-per-job semaphore has no such window and       */
/* mirrors the done_semaphore idiom already used for teardown.         */
/*                                                                    */
/* Lifetime: a queued buffer carries one extra reference taken at post */
/* time, so it stays alive (and keeps its pixels) until its job        */
/* completes even if its ring slot is meanwhile displaced; the slot    */
/* recycle path (refs == 1) therefore never fires under a pending      */
/* job. The queue is implicitly bounded: a buffer is queued at most    */
/* once between job completions (build_queued flag) and live buffer    */
/* objects are bounded by the ring size plus references held by        */
/* in-flight passes.                                                   */

/* Queue a buffer for a full cache build (state mutex held). Takes one
 * reference on the buffer; the completing helper drops it. */
static void mcd_build_post(SvtMcdState* s, McdBufs* b) {
    if (s->build_threads_n == 0) {
        return; /* helpers failed to spawn: no queue, no semaphore (see
                 * svt_mcd_state_create); builds fall back to the lazy path */
    }
    if (b->build_queued) {
        return; /* one job per buffer between completions */
    }
    b->build_queued = true;
    b->build_next   = NULL;
    if (s->build_tail) {
        s->build_tail->build_next = b;
    } else {
        s->build_head = b;
    }
    s->build_tail = b;
    ++b->refs; /* job's owning reference */
    svt_post_semaphore(s->build_semaphore);
}

/* Helper main loop: block for a token, drain one job, repeat. Tokens are
 * posted one per queued job; the shutdown sequence posts one extra token
 * per live thread, so a helper that finds an EMPTY queue has consumed a
 * shutdown poke and exits (this also means every queued job is drained
 * before the helpers die — builds that a remaining denoise might wait on
 * are never lost, and on the normal teardown path every denoise has
 * already finished before shutdown is even signalled). */
static void* mcd_build_thread(void* arg) {
    SvtMcdState* s = (SvtMcdState*)arg;
    for (;;) {
        svt_block_on_semaphore(s->build_semaphore);
        svt_block_on_mutex(s->mutex);
        McdBufs* b = s->build_head;
        if (!b) {
            svt_release_mutex(s->mutex);
            break; /* empty queue: shutdown poke */
        }
        s->build_head = b->build_next;
        if (!s->build_head) {
            s->build_tail = NULL;
        }
        b->build_next = NULL;
        svt_release_mutex(s->mutex);

        /* the heavy part, outside the state mutex; failures leave the
         * cache flags false and are retried by the lazy path at use */
        int rc = mcd_bufs_build_search(s, b);
        if (rc == MCD_SUCCESS) {
            rc = mcd_bufs_build_pel(s, b);
        }
        if (rc == MCD_SUCCESS) {
            rc = mcd_bufs_build_me_pel(s, b);
        }

        svt_block_on_mutex(s->mutex);
        b->build_queued = false;
        mcd_bufs_unref(s, b); /* drop the job's reference (may free b) */
        svt_release_mutex(s->mutex);
    }
    return NULL;
}

/* Signal the helpers to exit (they drain the queue first), join them and
 * release whatever the queue still holds. Mutex NOT held on entry; used
 * by state destroy and by the partial-failure path of state create. */
static void mcd_build_stop(SvtMcdState* s) {
    svt_block_on_mutex(s->mutex);
    s->build_shutdown = true;
    svt_release_mutex(s->mutex);
    /* one poke per live thread: every job token is backed by a queued
     * buffer, so a helper can only see an empty queue on a poke */
    for (uint32_t i = 0; i < s->build_threads_n; ++i) {
        svt_post_semaphore(s->build_semaphore);
    }
    for (uint32_t i = 0; i < s->build_threads_n; ++i) {
        EB_DESTROY_THREAD(s->build_threads[i]); /* joins */
    }
    s->build_threads_n = 0;
    /* defensive: the helpers drain before exiting, so this is a no-op
     * unless a future edit breaks that invariant — never leak the refs */
    svt_block_on_mutex(s->mutex);
    while (s->build_head) {
        McdBufs* b   = s->build_head;
        s->build_head = b->build_next;
        b->build_next  = NULL;
        b->build_queued = false;
        mcd_bufs_unref(s, b);
    }
    s->build_tail = NULL;
    svt_release_mutex(s->mutex);
    EB_DESTROY_SEMAPHORE(s->build_semaphore);
    s->build_semaphore = NULL;
}

/* ------------------------------------------------------------------ */
/* denoise one center frame                                            */

/* Denoise the popped job and write the result back into the pcs buffers.
 * Runs OUTSIDE the state mutex; the popped buffers are kept alive by the
 * references taken at pop time. Returns MCD_SUCCESS when the picture was
 * denoised (write-back done), MCD_ERR_PARAM when there was nothing to
 * denoise, another MCD code on module failure (the picture is then left
 * untouched). */
static int mcd_denoise_center(SvtMcdState* s, McdPopped* pop) {
    McdBufs*  center = pop->center;
    McdFrame* fr     = &center->frame;
    int       rc     = mcd_bufs_build_search(s, center);
    if (rc) {
        return rc;
    }

    const McdFrame*    refs[2 * MCD_MAX_TR];
    const McdMeResult* results[2 * MCD_MAX_TR];
    McdMeResult*       own[2 * MCD_MAX_TR];
    uint32_t           n_refs = 0;

    /* backward neighbours (in the past) first, then forward neighbours */
    for (uint32_t i = 0; i < pop->n_neigh; ++i) {
        McdBufs* neigh = pop->neigh[i];
        rc = mcd_bufs_build_search(s, neigh);
        if (rc == MCD_SUCCESS) {
            rc = mcd_bufs_build_pel(s, neigh);
        }
        if (rc == MCD_SUCCESS) {
            rc = mcd_bufs_build_me_pel(s, neigh);
        }
        if (rc) {
            goto fail;
        }
        /* Recycle the neighbour's original pixels if they are provably
         * dead: mcd_bufs_maybe_recycle_pix re-validates every condition
         * (denoise turn + all caches built) under the buffer's build_mutex,
         * which is the authoritative check -- a bare peek at ->denoised
         * here would be a formal C11 data race on a flag written under the
         * state mutex. Taking s->mutex is safe: this thread holds no
         * build_mutex, and no path ever holds s->mutex while acquiring a
         * build_mutex except this same ordered nesting. */
        svt_block_on_mutex(s->mutex);
        mcd_bufs_maybe_recycle_pix(s, neigh);
        svt_release_mutex(s->mutex);
        McdMeResult* res = NULL;
        rc               = mcd_me_analyze_s(fr, &neigh->frame, pop->neigh_delta[i], &s->cfg,
                                            pop->me_scratch, &res);
        if (rc == MCD_SUCCESS) {
            rc = mcd_me_recalculate_s(fr, &neigh->frame, &s->cfg, pop->me_scratch, res);
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
                /* uninitialized malloc: on success mcd_degrain_frame writes
                 * every sample of the plane before mcd_write_back reads it;
                 * on failure the buffer is never read (no write-back) */
                center->out_buf[p] = malloc((size_t)fr->planes[p].w * fr->planes[p].h * esz);
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
            mcd_write_back(center, pop->pcs->enhanced_pic);
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

    *state_ptr = NULL;
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
    EB_MALLOC_NO_CHECK(s->ring, sizeof(McdSlot) * s->ring_len);
    if (!s->ring) {
        EB_FREE(s);
        return EB_ErrorInsufficientResources;
    }
    memset(s->ring, 0, sizeof(McdSlot) * s->ring_len);

    s->hold_count   = 0;
    s->newest_valid = false;
    s->in_flight    = 0;

    s->mutex = svt_create_mutex();
    EB_NO_THROW_ADD_MEM(s->mutex, 1, EB_MUTEX);
    if (!s->mutex) {
        EB_FREE(s->ring);
        EB_FREE(s);
        return EB_ErrorInsufficientResources;
    }
    s->done_semaphore = svt_create_semaphore(0, 0x7FFFFFFF);
    EB_NO_THROW_ADD_MEM(s->done_semaphore, 1, EB_SEMAPHORE);
    if (!s->done_semaphore) {
        EB_DESTROY_MUTEX(s->mutex);
        EB_FREE(s->ring);
        EB_FREE(s);
        return EB_ErrorInsufficientResources;
    }

    /* One dfttest constant-table context per encoder instance: identical for
     * every frame and plane of the same sigma, shared read-only by all PA
     * threads. */
    EbErrorType rc = mcd_dfttest_ctx_create(s->cfg.sigma_pct, &s->dft_ctx);
    if (rc != EB_ErrorNone) {
        EB_DESTROY_SEMAPHORE(s->done_semaphore);
        EB_DESTROY_MUTEX(s->mutex);
        EB_FREE(s->ring);
        EB_FREE(s);
        *state_ptr = NULL;
        return rc;
    }

    /* Eager cache-build helpers, created last (they read s->dft_ctx and the
     * rest of the state). A helper that fails to spawn degrades gracefully:
     * mcd_build_stop winds the pool down and the lazy path keeps the encoder
     * correct, just without the overlap. */
    s->build_semaphore = svt_create_semaphore(0, 0x7FFFFFFF);
    EB_NO_THROW_ADD_MEM(s->build_semaphore, 1, EB_SEMAPHORE);
    if (s->build_semaphore) {
        for (uint32_t i = 0; i < MCD_BUILD_THREADS; ++i) {
            char name[16];
            svt_format_thread_name(name, sizeof(name), "mcd_build", i);
            s->build_threads[i] = svt_create_thread(mcd_build_thread, s, name);
            EB_NO_THROW_ADD_MEM(s->build_threads[i], 1, EB_THREAD);
            if (!s->build_threads[i]) {
                break;
            }
            ++s->build_threads_n;
        }
    }
    if (s->build_threads_n < MCD_BUILD_THREADS) {
        mcd_build_stop(s);
    }

    *state_ptr = s;
    return EB_ErrorNone;
}

void svt_mcd_state_destroy(void** state_ptr) {
    SvtMcdState* s = (SvtMcdState*)*state_ptr;
    if (!s) {
        return;
    }
    /* Wait for every in-flight denoise to finish before tearing the buffers
     * down. (The encoder teardown joins the PA threads before releasing the
     * EncodeContext, so this loop never spins in practice; it keeps the state
     * self-contained against future teardown reorders.) Every denoising pass
     * posts the semaphore exactly once when it finishes. */
    svt_block_on_mutex(s->mutex);
    while (s->in_flight > 0) {
        svt_release_mutex(s->mutex);
        /* each finished denoise posts once and has already decremented
         * in_flight before posting; the token alone is the wakeup --
         * decrementing again here would let destroy race live denoises */
        svt_block_on_semaphore(s->done_semaphore);
        svt_block_on_mutex(s->mutex);
    }
    while (s->in_copy > 0) {
        /* purely defensive (PA threads are joined before destroy); a copy
         * never posts the semaphore, so poll briefly instead */
        svt_release_mutex(s->mutex);
        mcd_wait_short();
        svt_block_on_mutex(s->mutex);
    }
    svt_release_mutex(s->mutex);

    /* Stop the build helpers once no denoise is in flight: an in-flight
     * denoise may still be waiting on a build a helper is running, so the
     * helpers must outlive it; conversely a helper never waits on a
     * denoise, so this cannot deadlock. Helpers drain every queued job
     * before exiting (no build a future denoise could wait on is lost —
     * and on this path no denoise is coming anyway), then the queue's
     * leftover references (if any) are released before the buffers go. */
    mcd_build_stop(s);

    for (uint32_t i = 0; i < s->ring_len; ++i) {
        mcd_bufs_unref(s, s->ring[i].buf);
        s->ring[i].buf = NULL;
    }
    for (uint32_t i = 0; i < s->me_pool_n; ++i) {
        mcd_me_scratch_destroy(&s->me_pool[i]);
    }
    s->me_pool_n = 0;
    for (uint32_t p = 0; p < 3; ++p) {
        free(s->pix_pool[p]);
        s->pix_pool[p]       = NULL;
        s->pix_pool_bytes[p] = 0;
    }
    EB_DESTROY_SEMAPHORE(s->done_semaphore);
    EB_FREE(s->ring);
    EB_DESTROY_MUTEX(s->mutex);
    mcd_dfttest_ctx_destroy(&s->dft_ctx);
    EB_FREE(s);
    *state_ptr = NULL;
}

/* ------------------------------------------------------------------ */
/* pop / denoise / finish plumbing                                     */

/* Resolve the denoise job for picture_number: reference the center buffer
 * (when its slot still holds this picture) and every settled temporal
 * neighbour, in the same past-then-future order the modules expect. Under
 * ordered arrivals the slots are guaranteed to hold every neighbour; the
 * number check keeps a degenerate out-of-order arrival from resolving
 * against the wrong picture. State mutex held. */
static void mcd_pop_resolve(SvtMcdState* s, McdPopped* pop) {
    pop->center  = NULL;
    pop->n_neigh = 0;
    {
        McdBufs* b = s->ring[pop->picture_number % s->ring_len].buf;
        if (b && b->valid && b->picture_number == pop->picture_number) {
            ++b->refs;
            pop->center = b;
        }
    }
    for (int32_t pass = 0; pass < 2; ++pass) {
        for (uint32_t d = 1; d <= s->cfg.tr; ++d) {
            const int64_t neigh_num = (int64_t)pop->picture_number + (pass == 0 ? -(int64_t)d : (int64_t)d);
            if (neigh_num < 0) {
                continue; /* start of stream: just fewer refs */
            }
            McdBufs* cand = s->ring[(uint64_t)neigh_num % s->ring_len].buf;
            if (cand && cand->valid && cand->picture_number == (uint64_t)neigh_num && cand != pop->center) {
                ++cand->refs;
                pop->neigh[pop->n_neigh]       = cand;
                pop->neigh_delta[pop->n_neigh] = (pass == 0) ? (int32_t)d : -(int32_t)d;
                ++pop->n_neigh;
            }
        }
    }
    pop->me_scratch = mcd_me_pool_take(s);
    ++s->in_flight;
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

/* Pop the holding-queue head, denoise it (outside the mutex) and emit it. */
static void mcd_release_head(SvtMcdState* s, EbObjectWrapper** out_wrappers,
                             struct PictureParentControlSet** out_pcs, uint32_t* n_out);

/* Release held heads whose temporal neighbours have all settled. Never
 * emits more than the caller's output array can hold (see svt_mcd_pa_arrival). */
static void mcd_release_ready_heads(SvtMcdState* s, EbObjectWrapper** out_wrappers,
                                    struct PictureParentControlSet** out_pcs, uint32_t* n_out) {
    while (s->hold_count > 0 && *n_out < SVT_MCD_MAX_EMITTED && s->watermark_valid &&
           s->hold[0].picture_number + s->cfg.tr <= s->watermark) {
        mcd_release_head(s, out_wrappers, out_pcs, n_out);
    }
}

/* Pop the holding-queue head into pop and take its references. State mutex
 * held; queue must be non-empty. */
static void mcd_hold_pop(SvtMcdState* s, McdPopped* pop) {
    McdHold* head = &s->hold[0];
    pop->wrapper         = head->wrapper;
    pop->pcs             = head->pcs;
    pop->picture_number  = head->picture_number;
    --s->hold_count;
    memmove(&s->hold[0], &s->hold[1], sizeof(McdHold) * s->hold_count);
    mcd_pop_resolve(s, pop);
}

/* Run the denoise of a popped job outside the state mutex and finish it
 * under the mutex (release in-flight bookkeeping, drop references, emit the
 * pcs to the caller's output arrays). */
static void mcd_release_popped(SvtMcdState* s, McdPopped* pop, EbObjectWrapper** out_wrappers,
                               struct PictureParentControlSet** out_pcs, uint32_t* n_out) {
    svt_release_mutex(s->mutex);

    int denoised = 0;
    if (pop->center && pop->pcs && pop->pcs->enhanced_pic) {
        const int rc = mcd_denoise_center(s, pop);
        /* ERR_PARAM = nothing to denoise (start of stream) / geometry
         * mismatch: the picture passes through undenoised and is never
         * retried, so its pixel copies are recyclable on the same terms */
        denoised     = (rc == MCD_SUCCESS || rc == MCD_ERR_PARAM);
        if (rc != MCD_SUCCESS && rc != MCD_ERR_PARAM) {
            SVT_ERROR("MC denoiser: failed to denoise picture %llu (rc=%d), passing it through undenoised\n",
                      (unsigned long long)pop->picture_number,
                      rc);
        }
    }

    svt_block_on_mutex(s->mutex);

    mcd_me_pool_put(s, pop->me_scratch);
    if (pop->center) {
        /* ERR_PARAM = "nothing to denoise / geometry mismatch": the picture
         * passes through undenoised and is never retried, so its pixel
         * copies become recyclable on the same terms (once its caches are
         * complete). Real failures keep everything owned by the buffer. */
        pop->center->denoised = denoised;
        mcd_bufs_maybe_recycle_pix(s, pop->center);
    }

    out_wrappers[*n_out] = pop->wrapper;
    out_pcs[*n_out]      = pop->pcs;
    ++(*n_out);
    --s->in_flight;
    svt_post_semaphore(s->done_semaphore);
    if (pop->center) {
        mcd_bufs_unref(s, pop->center);
    }
    for (uint32_t i = 0; i < pop->n_neigh; ++i) {
        mcd_bufs_unref(s, pop->neigh[i]);
    }
}

/* Pop the holding-queue head, denoise it (outside the mutex) and emit it. */
static void mcd_release_head(SvtMcdState* s, EbObjectWrapper** out_wrappers,
                             struct PictureParentControlSet** out_pcs, uint32_t* n_out) {
    McdPopped pop;
    mcd_hold_pop(s, &pop);
    mcd_release_popped(s, &pop, out_wrappers, out_pcs, n_out);
}

/* End-of-stream flush: pure drain of the holding queue (no waiting). The
 * only trigger is the arrival of the pcs carrying end_of_sequence_flag,
 * which is FIFO-ordered after every real picture, so by the time this runs
 * every temporal neighbour that will ever exist has settled -- any missing
 * ones are legitimately absent tail neighbours and the neighbour sets match
 * a serial arrival exactly. */
static void mcd_eos_flush(SvtMcdState* s, EbObjectWrapper** out_wrappers,
                          struct PictureParentControlSet** out_pcs, uint32_t* n_out) {
    /* Pure drain; see the header comment above. */
    uint32_t quota = SVT_MCD_MAX_EMITTED - *n_out;
    while (s->hold_count > 0 && quota > 0) {
        mcd_release_head(s, out_wrappers, out_pcs, n_out);
        --quota;
    }
}

/* Settle a picture number that went through the hook without a successful
 * copy. Arrivals are ordered, so this can only advance the watermark by the
 * one step (or start it at 0); anything else keeps the watermark unchanged
 * and the ordered-arrival wait of later pictures absorbs the gap. */
static void mcd_settle_ordered(SvtMcdState* s, uint64_t picture_number) {
    if (!s->watermark_valid) {
        if (picture_number == 0) {
            s->watermark       = 0;
            s->watermark_valid = true;
            /* drain anything that settled before picture 0 completed */
            while (s->settled_bits & 1) {
                ++s->watermark;
                s->settled_bits >>= 1;
            }
        } else if (picture_number < 64) {
            s->settled_bits |= 1ull << picture_number;
        }
        return;
    }
    if (picture_number <= s->watermark) {
        return; /* already covered (duplicate/overlay settle) */
    }
    const uint64_t off = picture_number - (s->watermark + 1);
    if (off < 64) {
        s->settled_bits |= 1ull << off;
    }
    /* advance across every consecutively settled picture */
    while (s->settled_bits & 1) {
        ++s->watermark;
        s->settled_bits >>= 1;
    }
}

uint32_t svt_mcd_pa_arrival(void* state,
                            struct PictureParentControlSet* pcs,
                            EbObjectWrapper* pcs_wrapper,
                            EbObjectWrapper** out_wrappers,
                            struct PictureParentControlSet** out_pcs) {
    SvtMcdState*        s       = (SvtMcdState*)state;
    SequenceControlSet* scs     = pcs->scs;
    /* EncodeContext* unused since the terminating-flag belt removal */
    uint32_t            n_out   = 0;

    svt_block_on_mutex(s->mutex);

    /* End of stream: the pcs carrying end_of_sequence_flag (the last real
     * picture with overlays off, the flagged overlay pcs with overlays on)
     * is FIFO-ordered after every real picture, so flushing on ITS arrival
     * is deterministic. There is deliberately no terminating-flag fallback
     * trigger: that flag is set while real pictures may still be in flight,
     * and a fallback flush from it races their arrivals (this exact race
     * shipped nondeterministic output and encoder deadlocks before it was
     * removed). A stream in which no flagged pcs ever reaches this stage
     * would hang rather than silently corrupt. */
    const bool eos = pcs->end_of_sequence_flag;
    /* Low-delay / all-intra: never hold pictures (see file header comment). */
    const bool no_hold = (scs->static_config.pred_structure == LOW_DELAY) || scs->allintra;

    if (!pcs->is_overlay) {
        /* Ordered arrival: fill the ring slots in picture order so the
         * release decisions (and therefore every neighbour set) are
         * independent of PA thread timing. Normally the predecessor is
         * already settled and this loop does not spin at all; it only
         * absorbs genuine reordering between PA threads. */
        if (!no_hold) {
            /* Wait for the predecessor to settle, WITHOUT a poll cap: RC
             * posts pictures through the same FIFO in order, so the
             * predecessor is always en route and the earliest unprocessed
             * picture never waits — no deadlock, just ordered progress.
             * The old poll-cap break let an arrival proceed out of order,
             * evicting ring slots still needed as neighbours (and, via the
             * lost settle, freezing the watermark). */
            while (s->watermark_valid
                       ? s->watermark + 1 < pcs->picture_number
                       : pcs->picture_number != 0) {
                svt_release_mutex(s->mutex);
                mcd_wait_short();
                svt_block_on_mutex(s->mutex);
            }
        }
        /* copy the pixels into the ring slot for this picture number */
        McdSlot* sl = &s->ring[pcs->picture_number % s->ring_len];
        /* Acquire the slot's buffer object: reuse it when nothing but the
         * slot itself holds it, otherwise install a fresh one and let the
         * old one die with its last reader (never recycle data under a live
         * denoiser). */
        McdBufs* b = sl->buf;
        if (b && b->refs == 1) {
            mcd_bufs_reset(b);
        } else {
            if (b) {
                mcd_bufs_unref(s, b);
                sl->buf = NULL; /* owning ref dropped: do not leave the slot
                                   pointing at a buffer the last reader may
                                   free while the alloc below can fail */
            }
            b = (McdBufs*)mcd_alloc_zeroed(sizeof(*b));
            if (!b) {
                /* out of memory: settle and pass the picture through
                 * undenoised rather than dropping it */
                while (!no_hold && s->hold_count >= SVT_MCD_MAX_RELEASED &&
                       n_out < SVT_MCD_MAX_EMITTED) {
                    mcd_release_head(s, out_wrappers, out_pcs, &n_out);
                }
                if (!no_hold && s->hold_count < SVT_MCD_MAX_RELEASED) {
                    mcd_hold_push(s, pcs_wrapper, pcs, pcs->picture_number);
                } else {
                    out_wrappers[n_out] = pcs_wrapper;
                    out_pcs[n_out]      = pcs;
                    ++n_out;
                }
                mcd_settle_ordered(s, pcs->picture_number);
                mcd_release_ready_heads(s, out_wrappers, out_pcs, &n_out);
                if (eos && s->hold_count > 0) {
                    mcd_eos_flush(s, out_wrappers, out_pcs, &n_out);
                }
                svt_release_mutex(s->mutex);
                return n_out;
            }
            b->refs        = 1;
            b->build_mutex = svt_create_mutex();
            EB_NO_THROW_ADD_MEM(b->build_mutex, 1, EB_MUTEX);
            if (!b->build_mutex) {
                /* The state mutex is still held. Do not publish a buffer
                 * whose lazy cache-build lock could not be created. */
                free(b); /* fresh zeroed allocation owns no subordinate data */
                while (!no_hold && s->hold_count >= SVT_MCD_MAX_RELEASED &&
                       n_out < SVT_MCD_MAX_EMITTED) {
                    mcd_release_head(s, out_wrappers, out_pcs, &n_out);
                }
                if (!no_hold && s->hold_count < SVT_MCD_MAX_RELEASED) {
                    mcd_hold_push(s, pcs_wrapper, pcs, pcs->picture_number);
                } else {
                    out_wrappers[n_out] = pcs_wrapper;
                    out_pcs[n_out]      = pcs;
                    ++n_out;
                }
                mcd_settle_ordered(s, pcs->picture_number);
                mcd_release_ready_heads(s, out_wrappers, out_pcs, &n_out);
                if (eos && s->hold_count > 0) {
                    mcd_eos_flush(s, out_wrappers, out_pcs, &n_out);
                }
                svt_release_mutex(s->mutex);
                return n_out;
            }
            sl->buf        = b;
        }
        /* Pre-take the recycled copy buffers for this geometry (cheap,
         * under the mutex), then run the copy OUTSIDE the mutex: slot
         * exclusivity is guaranteed by the ordered arrival (no other
         * picture can touch this slot while ours is unsettled) and
         * pop_resolve skips invalid buffers, so nothing can observe the
         * half-copied state or mutate it concurrently. */
        {
            McdGeom      g;
            const size_t esz = pcs->enhanced_pic->bit_depth > 8 ? sizeof(uint16_t)
                                                                : sizeof(uint8_t);
            mcd_geom_from_pic(&g, pcs->enhanced_pic);
            const uint32_t np = g.has_chroma ? 3 : 1;
            for (uint32_t p = 0; p < np; ++p) {
                if (b->pix_buf[p] && b->pix_bytes[p] == (size_t)g.w[p] * g.h[p] * esz) {
                    continue; /* already owned from the reset path */
                }
                free(b->pix_buf[p]);
                b->pix_buf[p]   = mcd_pix_pool_take(s, p, (size_t)g.w[p] * g.h[p] * esz);
                b->pix_bytes[p] = b->pix_buf[p] ? (size_t)g.w[p] * g.h[p] * esz : 0;
            }
        }
        /* The out-of-mutex copy is safe only in hold mode, where the
         * unbreakable ordered-arrival wait serializes any picture that
         * could touch this slot (number +/- ring_len). no_hold skips that
         * wait entirely, so two PA threads can hold number N and
         * N + ring_len concurrently: an out-of-mutex copy there races the
         * later arrival's slot reset (heap UAF). Keep the copy under the
         * mutex in no_hold mode -- its cost is bounded by one memcpy and
         * no_hold denoises inline on this thread anyway. */
        int copy_rc;
        if (no_hold) {
            copy_rc = mcd_bufs_copy_pixels(s, b, pcs->enhanced_pic);
        } else {
            ++s->in_copy;
            svt_release_mutex(s->mutex);
            copy_rc = mcd_bufs_copy_pixels(s, b, pcs->enhanced_pic);
            svt_block_on_mutex(s->mutex);
            --s->in_copy;
        }

        if (copy_rc == MCD_SUCCESS) {
            b->picture_number = pcs->picture_number;
            b->valid          = true;
            if (!s->newest_valid || pcs->picture_number > s->newest) {
                s->newest       = pcs->picture_number;
                s->newest_valid = true;
            }
            /* settle this picture: arrivals are ordered, so the watermark
             * simply steps onto it */
            mcd_settle_ordered(s, pcs->picture_number);

            /* Eager cache build: the pixels are final and the frame will
             * later be read as a neighbour strictly through its caches, so
             * hand the buffer to the helper pool now — the build overlaps
             * the current head's ME/degrain instead of serializing in
             * front of a future ME pass. The posted job carries an extra
             * reference, so the slot cannot recycle the buffer under the
             * helpers. In no-hold mode the arriving picture is denoised
             * inline on this very thread below (its search planes are
             * built right away); queueing it would only risk making that
             * denoise wait on a helper, so it is skipped there. */
            if (!no_hold) {
                mcd_build_post(s, b);
            }

            if (no_hold) {
                /* denoise immediately with whatever neighbours exist */
                McdPopped pop;
                pop.wrapper        = pcs_wrapper;
                pop.pcs            = pcs;
                pop.picture_number = pcs->picture_number;
                mcd_pop_resolve(s, &pop);
                mcd_release_popped(s, &pop, out_wrappers, out_pcs, &n_out);
            } else {
                /* capacity guard: ordered arrivals hold at most tr + 1
                 * pictures, far below the output capacity; release the head
                 * early in case that assumption is ever violated (never
                 * emits past the caller's output capacity) */
                while (s->hold_count >= SVT_MCD_MAX_RELEASED && n_out < SVT_MCD_MAX_EMITTED) {
                    mcd_release_head(s, out_wrappers, out_pcs, &n_out);
                }
                if (s->hold_count <= SVT_MCD_MAX_RELEASED - 1) {
                    mcd_hold_push(s, pcs_wrapper, pcs, pcs->picture_number);
                    /* release the head once all its temporal neighbours
                     * settled */
                    mcd_release_ready_heads(s, out_wrappers, out_pcs, &n_out);
                } else {
                    out_wrappers[n_out] = pcs_wrapper;
                    out_pcs[n_out]      = pcs;
                    ++n_out;
                }
            }
        } else {
            /* could not copy (out of memory): pass the picture through
             * undenoised rather than dropping it */
            out_wrappers[n_out] = pcs_wrapper;
            out_pcs[n_out]      = pcs;
            ++n_out;
            mcd_settle_ordered(s, pcs->picture_number);
            mcd_release_ready_heads(s, out_wrappers, out_pcs, &n_out);
        }
    } else {
        /* overlay pictures are not denoised (they do not carry the input
         * pixels at this stage); forward them untouched. They never occupy
         * a ring slot, but they DO settle their picture number: later
         * arrivals must not wait for an overlay in their ordered-arrival
         * check, and the overlay's number is never a valid neighbour (its
         * slot stays empty), exactly like the original design.
         *
         * The overlay SHARES its base real picture's number, and the real
         * twin is FIFO-ordered ahead of us. Wait for the twin to settle the
         * shared number before settling/flushing here: without this gate an
         * overlay arrival (in particular the EOS-flagged overlay that RC
         * stamps at end of stream when overlays are on) could settle and
         * flush while the real twin's pixel copy is still running outside
         * the state mutex -- popping heads with a mid-copy neighbour
         * (divergent output) or draining the flush before the twin is held
         * (stranding it in the hold queue forever: encoder deadlock). */
        if (!no_hold) {
            while (s->watermark_valid ? s->watermark < pcs->picture_number
                                      : pcs->picture_number != 0) {
                svt_release_mutex(s->mutex);
                mcd_wait_short();
                svt_block_on_mutex(s->mutex);
            }
        }
        out_wrappers[n_out] = pcs_wrapper;
        out_pcs[n_out]      = pcs;
        ++n_out;
        mcd_settle_ordered(s, pcs->picture_number);
        mcd_release_ready_heads(s, out_wrappers, out_pcs, &n_out);
    }

    if (eos && s->hold_count > 0) {
        /* flush (part of) the holding queue; see mcd_eos_flush */
        mcd_eos_flush(s, out_wrappers, out_pcs, &n_out);
    }

    svt_release_mutex(s->mutex);
    return n_out;
}
