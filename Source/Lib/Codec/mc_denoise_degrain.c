/*
 * mc_denoise_degrain.c — motion-compensated weighted temporal aggregation.
 *
 * Clean-room MDegrain reimplementation (Fizick scheme): every reference frame
 * is motion-compensated block-wise on the vector grid produced by the motion
 * analysis pass, each compensated block is SAD-weighted against the source,
 * and the weighted blocks are accumulated over the overlapped grid with
 * raised-cosine^2 windows.  The per-sample window sum is exactly 2048, so the
 * fixed-point finalization is out = (acc + 1024) >> 11.
 *
 * Window exactness: raw 2D windows are quantized as
 *   q2d[y][x] = (q[y] * q[x] + 1024) >> 11,
 *   q[j] = round(2048 * cos^2(pi * (j - overlap + 0.5) / (2 * overlap)))
 * (when blk = 2*overlap this is the usual 2048 * sin^2(pi*(j + 0.5)/blk)).
 * Rounding alone does not guarantee exact overlap sums, and frame borders are
 * covered by fewer blocks than the interior.  Therefore every visible sample
 * has a unique "designated" block — the covering block with the largest grid
 * index, i.e. the block owning the sample's grid column (luma-local offsets
 * < step), or, past the last full grid step, the final block of the row or
 * column.  The designated block applies the residual window
 *   2048 - (sum of the other covering blocks' raw windows)
 * which makes the per-sample window sum exactly 2048 everywhere, independent
 * of rounding ties and border coverage.
 *
 * Chroma planes use the same luma-grid windows evaluated at the corresponding
 * luma-sample offsets (a chroma sample at (Xc, Yc) with subsampling shifts
 * x_ss/y_ss maps to luma (Xc << x_ss, Yc << y_ss)), so chroma samples inherit
 * the exact-2048 property.  Chroma sample columns whose luma-local offset is
 * negative (possible for odd luma block origins) get window 0, which keeps
 * them consistent with the luma covering set.
 *
 * dst planes are written fully; center planes are only read (block loop and
 * finalization) before any dst write happens, so dst may alias center->planes
 * as long as the refs' pel caches were built beforehand (guaranteed by the
 * glue).  dst is never read.
 */

#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#include "mc_denoise.h"
#include "mc_denoise_simd.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MCD_WIN_Q 2048 /* raised-cosine^2 window quantization scale */
#define MCD_W_ONE 256  /* unity degrain weight of a single ref       */

/* Floor-dividing right shift for signed values (the behavior of >> on
 * negatives is implementation-defined in C11; keep the math explicit).
 * s is 0 or 1.                                                                */
static inline int32_t mcd_floor_shr(int32_t v, uint32_t s) {
    const int32_t m = (int32_t)(1u << s) - 1;
    return (v >= 0) ? (v >> s) : -((m - v) >> s);
}

static inline uint32_t mcd_plane_load(const McdPlane *p, uint32_t y, uint32_t x) {
    if (p->highbd)
        return ((const uint16_t *)p->data)[(size_t)y * p->stride + x];
    return ((const uint8_t *)p->data)[(size_t)y * p->stride + x];
}

static inline void mcd_plane_store(const McdPlane *p, uint32_t y, uint32_t x, uint32_t v) {
    if (p->highbd)
        ((uint16_t *)p->data)[(size_t)y * p->stride + x] = (uint16_t)v;
    else
        ((uint8_t *)p->data)[(size_t)y * p->stride + x] = (uint8_t)v;
}

/* 1D raised-cosine^2 window, quantized 0..2048:
 * q[j] = round(2048 * cos^2(pi*(j - overlap + 0.5) / (2*overlap))).           */
static void mcd_window_1d(uint32_t blk, uint32_t overlap, uint32_t *q) {
    uint32_t j;
    if (overlap == 0) {
        for (j = 0; j < blk; j++) q[j] = MCD_WIN_Q;
        return;
    }
    for (j = 0; j < blk; j++) {
        const double arg = M_PI * ((double)j - (double)overlap + 0.5) /
                           (2.0 * (double)overlap);
        const double v   = (double)MCD_WIN_Q * cos(arg) * cos(arg);
        uint32_t r       = (uint32_t)(v + 0.5);
        if (r > (uint32_t)MCD_WIN_Q) r = MCD_WIN_Q;
        q[j] = r;
    }
}

static inline uint32_t mcd_win_raw(const uint32_t *qx, const uint32_t *qy,
                                   uint32_t ox, uint32_t oy) {
    return (qx[ox] * qy[oy] + (MCD_WIN_Q >> 1)) >> 11;
}

/* Window value applied by block (bx,by) at luma-local offsets (ox,oy).
 * Returns 0 when the block does not cover the sample.  The designated block
 * of a sample is the covering block with the largest grid index: the one
 * owning the sample's grid column (offsets < step) or, beyond the last full
 * grid step, the final block of the grid row/column.  There the window is
 * the residual that makes the sum over all covering blocks exactly 2048.   */
static inline uint32_t mcd_block_window(const uint32_t *qx, const uint32_t *qy,
                                        uint32_t blk_w, uint32_t blk_h,
                                        uint32_t step_w, uint32_t step_h,
                                        uint32_t nbx, uint32_t nby,
                                        int32_t ox, int32_t oy,
                                        uint32_t bx, uint32_t by) {
    if (ox < 0 || oy < 0 || (uint32_t)ox >= blk_w || (uint32_t)oy >= blk_h)
        return 0;
    if (((uint32_t)ox < step_w || bx == nbx - 1) &&
        ((uint32_t)oy < step_h || by == nby - 1)) {
        int32_t s = MCD_WIN_Q;
        uint32_t i, j;
        for (i = 0; i <= bx; i++) {
            const uint32_t wx = (uint32_t)ox + i * step_w;
            if (wx >= blk_w) break;
            for (j = 0; j <= by; j++) {
                const uint32_t wy = (uint32_t)oy + j * step_h;
                if (wy >= blk_h) break;
                if (i == 0 && j == 0) continue; /* the designated block itself */
                s -= (int32_t)mcd_win_raw(qx, qy, wx, wy);
            }
        }
        if (s < 0) s = 0;
        if (s > MCD_WIN_Q) s = MCD_WIN_Q;
        return (uint32_t)s;
    }
    return mcd_win_raw(qx, qy, (uint32_t)ox, (uint32_t)oy);
}

static void mcd_copy_planes(const McdFrame *center, McdPlane *dst_planes,
                            uint32_t n_planes) {
    uint32_t p, y, x;
    for (p = 0; p < n_planes; p++) {
        const McdPlane *sp = &center->planes[p];
        for (y = 0; y < sp->h; y++)
            for (x = 0; x < sp->w; x++)
                mcd_plane_store(&dst_planes[p], y, x, mcd_plane_load(sp, y, x));
    }
}

int mcd_degrain_frame(const McdFrame *center, const McdFrame *const *refs,
                      const McdMeResult *const *results, uint32_t n_refs,
                      const McdConfig *cfg, McdPlane *dst_planes) {
    uint32_t p, i, bx, by, lx, ly;
    uint32_t blk_w = 0, blk_h = 0, overlap = 0, step_w = 0, step_h = 0;
    uint32_t nbx = 0, nby = 0;
    uint32_t n_planes, bd, pmax, w_l, h_l;
    size_t blk_area;
    uint32_t *qx = NULL, *qy = NULL, *bacc = NULL, *wts = NULL, *win = NULL;
    uint16_t *win_luma = NULL;
    uint32_t *acc[3] = { NULL, NULL, NULL };
    uint16_t *comp = NULL;
    int32_t *coloff = NULL;
    int ret = MCD_SUCCESS;

    if (!center || !cfg || !dst_planes) return MCD_ERR_PARAM;
    if (n_refs != 0 && (!refs || !results)) return MCD_ERR_PARAM;

    n_planes = center->num_planes;
    if (n_planes != 1 && n_planes != 3) return MCD_ERR_PARAM;
    bd = center->bitdepth;
    if (bd < 8 || bd > 16) return MCD_ERR_PARAM;
    if (center->x_ss > 1 || center->y_ss > 1) return MCD_ERR_PARAM;
    pmax = (1u << bd) - 1;

    /* center and dst plane sanity */
    for (p = 0; p < n_planes; p++) {
        const McdPlane *sp = &center->planes[p];
        const McdPlane *dp = &dst_planes[p];
        if (!sp->w || !sp->h || sp->stride < sp->w || !sp->data) return MCD_ERR_PARAM;
        if (dp->w != sp->w || dp->h != sp->h || dp->stride < dp->w || !dp->data)
            return MCD_ERR_PARAM;
        if (sp->highbd != (bd > 8)) return MCD_ERR_PARAM;
        if (bd > 8 && !dp->highbd) return MCD_ERR_PARAM;
    }

    if (n_refs == 0) { /* nothing to merge against: plain copy */
        mcd_copy_planes(center, dst_planes, n_planes);
        return MCD_SUCCESS;
    }

    /* reference frames: matching format, pel caches, identical final grid */
    for (i = 0; i < n_refs; i++) {
        const McdFrame *r = refs[i];
        const McdMeResult *g = results[i];
        if (!r || !g) return MCD_ERR_PARAM;
        if (r->num_planes != n_planes || r->bitdepth != bd) return MCD_ERR_PARAM;
        if (r->x_ss != center->x_ss || r->y_ss != center->y_ss) return MCD_ERR_PARAM;
        for (p = 0; p < n_planes; p++) {
            const McdPlane *sp = &center->planes[p];
            const McdPlane *rp = &r->planes[p];
            if (rp->w != sp->w || rp->h != sp->h || rp->stride < rp->w || !rp->data)
                return MCD_ERR_PARAM;
            if (rp->highbd != (bd > 8)) return MCD_ERR_PARAM;
        }
        if (!r->pel_valid || !r->pel) return MCD_ERR_PARAM;
        if (!g->vecs) return MCD_ERR_PARAM;
        if (i == 0) {
            blk_w = g->blk_w;
            blk_h = g->blk_h;
            overlap = g->overlap;
            nbx = g->n_blks_x;
            nby = g->n_blks_y;
            if (!blk_w || !blk_h || !nbx || !nby) return MCD_ERR_PARAM;
            if (overlap >= blk_w || overlap >= blk_h) return MCD_ERR_PARAM;
            step_w = blk_w - overlap;
            step_h = blk_h - overlap;
            w_l = center->planes[0].w;
            h_l = center->planes[0].h;
            /* the grid must cover the visible luma plane completely, since
             * every dst sample is produced from the window accumulation */
            if ((uint64_t)(nbx - 1) * step_w + blk_w < w_l) return MCD_ERR_PARAM;
            if ((uint64_t)(nby - 1) * step_h + blk_h < h_l) return MCD_ERR_PARAM;
        } else {
            if (g->blk_w != blk_w || g->blk_h != blk_h || g->overlap != overlap ||
                g->n_blks_x != nbx || g->n_blks_y != nby)
                return MCD_ERR_PARAM;
        }
    }

    blk_area = (size_t)blk_w * blk_h;
    qx     = (uint32_t *)malloc((size_t)blk_w * sizeof(*qx));
    qy     = (uint32_t *)malloc((size_t)blk_h * sizeof(*qy));
    comp   = (uint16_t *)malloc(blk_area * sizeof(*comp));
    bacc   = (uint32_t *)malloc(blk_area * sizeof(*bacc));
    coloff = (int32_t *)malloc((size_t)blk_w * sizeof(*coloff));
    wts    = (uint32_t *)malloc((size_t)n_refs * sizeof(*wts));
    win    = (uint32_t *)malloc(blk_w * sizeof(*win));
    /* Cache each block's exact luma-grid window while processing luma. Chroma
     * maps to a subset of these same luma offsets, so it can reuse the values
     * instead of repeating the designated-block residual calculation. */
    if (n_planes > 1)
        win_luma = (uint16_t *)malloc(blk_area * sizeof(*win_luma));
    for (p = 0; p < n_planes; p++)
        acc[p] = (uint32_t *)calloc((size_t)center->planes[p].w * center->planes[p].h,
                                    sizeof(*acc[p]));
    if (!qx || !qy || !comp || !bacc || !coloff || !wts || !win ||
        !acc[0] || (n_planes == 3 && (!acc[1] || !acc[2])) ||
        (n_planes > 1 && !win_luma)) {
        ret = MCD_ERR_NOMEM;
        goto cleanup;
    }

    mcd_window_1d(blk_w, overlap, qx);
    mcd_window_1d(blk_h, overlap, qy);

    for (by = 0; by < nby; by++) {
        const uint32_t y0_l = by * step_h;
        if (y0_l >= h_l) break;
        for (bx = 0; bx < nbx; bx++) {
            const uint32_t x0_l = bx * step_w;
            uint32_t wsum = 0, wsrc, tot, wsum_n;
            if (x0_l >= w_l) break;

            /* per-ref degrain weights for this block (0 on scene change) */
            for (i = 0; i < n_refs; i++) {
                const McdMeResult *g = results[i];
                const McdVector *v = &g->vecs[(size_t)by * nbx + bx];
                const uint32_t w = g->scene_change ? 0
                                                   : mcd_degrain_weight(v->sad_norm,
                                                                        cfg->thsad);
                wts[i] = w;
                wsum += w;
            }
            /* mvtools normalizeWeights semantics: the center keeps its full
             * weight and the combined set is scaled to sum to exactly 256:
             * tot = 256 + sum(w); WRef_i = w_i * 256 / tot;
             * WSrc = 256 - sum(WRef_i). The source is therefore never fully
             * replaced by the references (e.g. one perfect ref blends 50/50). */
            tot    = MCD_W_ONE + wsum;
            wsum_n = 0;
            for (i = 0; i < n_refs; i++) {
                const uint32_t wn = (wts[i] * MCD_W_ONE + tot / 2) / tot;
                wts[i] = wn;
                wsum_n += wn;
            }
            if (wsum_n > MCD_W_ONE) { /* unreachable; defensive clamp */
                uint32_t ex = wsum_n - MCD_W_ONE;
                for (i = 0; i < n_refs && ex; i++) {
                    const uint32_t d = wts[i] < ex ? wts[i] : ex;
                    wts[i] -= d;
                    ex -= d;
                }
                wsum_n = MCD_W_ONE;
            }
            wsrc = MCD_W_ONE - wsum_n;

            for (p = 0; p < n_planes; p++) {
                const uint32_t ssx = p ? center->x_ss : 0;
                const uint32_t ssy = p ? center->y_ss : 0;
                const McdPlane *sp = &center->planes[p];
                const uint32_t x0_p = x0_l >> ssx;
                const uint32_t y0_p = y0_l >> ssy;
                uint32_t x1_l = x0_l + blk_w;
                uint32_t y1_l = y0_l + blk_h;
                uint32_t bw, bh, n;
                const uint32_t ss_mx = (1u << ssx) - 1;
                const uint32_t ss_my = (1u << ssy) - 1;

                if (x1_l > w_l) x1_l = w_l;
                if (y1_l > h_l) y1_l = h_l;
                /* chroma block extent: round the clipped luma edge UP so the
                 * last chroma column/row of odd-size planes stays covered    */
                bw = ((x1_l + ss_mx) >> ssx) - x0_p;
                bh = ((y1_l + ss_my) >> ssy) - y0_p;
                if (bw > sp->w - x0_p) bw = sp->w - x0_p; /* defensive clip */
                if (bh > sp->h - y0_p) bh = sp->h - y0_p;
                if (!bw || !bh) continue;

                /* B = (128 + WSrc*src + sum_i w_i*comp_i) >> 8 */
                const int simd = mcd_use_avx2();
                for (ly = 0; ly < bh; ly++) {
                    const uint32_t Y = y0_p + ly;
                    if (simd)
                        mcd_simd_degrain_init_row(bacc + (size_t)ly * bw, sp, Y, x0_p,
                                                  wsrc, bw);
                    else
                        for (lx = 0; lx < bw; lx++)
                            bacc[(size_t)ly * bw + lx] =
                                128 + wsrc * mcd_plane_load(sp, Y, x0_p + lx);
                }
                for (i = 0; i < n_refs; i++) {
                    const McdVector *v;
                    const uint32_t w = wts[i];
                    int32_t fx, fy;
                    size_t l;
                    if (!w) continue;
                    v = &results[i]->vecs[(size_t)by * nbx + bx];
                    /* luma: origin*2 + vector (half-pel units); chroma: both
                     * origin and vector divided by 2^ss, flooring            */
                    fx = 2 * (int32_t)x0_p + mcd_floor_shr(v->x, ssx);
                    fy = 2 * (int32_t)y0_p + mcd_floor_shr(v->y, ssy);
                    mcd_pelset_fetch((const McdPelSet *)refs[i]->pel, p,
                                     fx, fy, bw, bh, comp);
                    if (simd)
                        mcd_simd_degrain_add_ref(bacc, comp, w, (uint32_t)(bw * bh));
                    else
                        for (l = 0; l < (size_t)bw * bh; l++)
                            bacc[l] += (uint32_t)w * comp[l];
                }
                n = bw * bh;
                if (simd)
                    mcd_simd_degrain_shift(bacc, n);
                else
                    for (lx = 0; lx < n; lx++) bacc[lx] >>= 8;

                /* accumulate with exact-sum windows at luma-grid offsets */
                for (lx = 0; lx < bw; lx++)
                    coloff[lx] = (int32_t)((x0_p + lx) << ssx) - (int32_t)x0_l;
                for (ly = 0; ly < bh; ly++) {
                    const int32_t oy = (int32_t)((y0_p + ly) << ssy) - (int32_t)y0_l;
                    uint32_t *accrow = acc[p] + (size_t)(y0_p + ly) * sp->w + x0_p;
                    const uint32_t *brow = bacc + (size_t)ly * bw;
                    for (lx = 0; lx < bw; lx++) {
                        uint32_t w;
                        if (p == 0) {
                            w = mcd_block_window(qx, qy, blk_w, blk_h, step_w, step_h,
                                                 nbx, nby, coloff[lx], oy, bx, by);
                            if (win_luma) {
                                /* Luma offsets are nonnegative and within the
                                 * block for this plane's visible rectangle. */
                                win_luma[(size_t)ly * blk_w + lx] = (uint16_t)w;
                            }
                        } else if (coloff[lx] < 0 || oy < 0 ||
                                   (uint32_t)coloff[lx] >= blk_w ||
                                   (uint32_t)oy >= blk_h) {
                            w = 0;
                        } else {
                            w = win_luma[(size_t)(uint32_t)oy * blk_w +
                                         (uint32_t)coloff[lx]];
                        }
                        win[lx] = w;
                    }
                    if (simd)
                        mcd_simd_degrain_acc_row(accrow, brow, win, bw);
                    else
                        for (lx = 0; lx < bw; lx++)
                            accrow[lx] += brow[lx] * win[lx];
                }
            }
        }
    }

    /* finalization: out = (acc + 1024) >> 11, clamped, then limit-clamped
     * against the source (center planes are still untouched here, so this is
     * safe even when dst aliases them)                                          */
    for (p = 0; p < n_planes; p++) {
        const McdPlane *sp = &center->planes[p];
        const int32_t lim = cfg->limit
                                ? (int32_t)((uint64_t)cfg->limit * pmax / 255)
                                : 0;
        uint32_t x, y;
        const int simd_fin = mcd_use_avx2();
        for (y = 0; y < sp->h; y++) {
            if (simd_fin &&
                mcd_simd_degrain_finalize_row(&dst_planes[p], y, acc[p], sp, lim, pmax))
                continue;
            for (x = 0; x < sp->w; x++) {
                uint32_t o = (acc[p][(size_t)y * sp->w + x] + 1024) >> 11;
                if (o > pmax) o = pmax;
                if (lim) {
                    const int32_t s = (int32_t)mcd_plane_load(sp, y, x);
                    int32_t lo = s - lim;
                    int32_t hi = s + lim;
                    if (lo < 0) lo = 0;
                    if (hi > (int32_t)pmax) hi = (int32_t)pmax;
                    if ((int32_t)o < lo) o = (uint32_t)lo;
                    if ((int32_t)o > hi) o = (uint32_t)hi;
                }
                mcd_plane_store(&dst_planes[p], y, x, o);
            }
        }
    }

cleanup:
    for (p = 0; p < 3; p++) {
        if (acc[p]) {
            free(acc[p]);
            acc[p] = NULL;
        }
    }
    free(qx);
    free(qy);
    free(comp);
    free(bacc);
    free(coloff);
    free(wts);
    free(win);
    free(win_luma);
    return ret;
}
