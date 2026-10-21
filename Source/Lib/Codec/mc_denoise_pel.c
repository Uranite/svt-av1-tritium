/*
 * mc_denoise_pel.c — picture pyramids, half-pel phase planes and block
 * fetch for the motion-compensated denoiser (see mc_denoise.h).
 *
 * Clean-room reimplementation of the sample-handling scheme of the
 * mvtools motion-compensated denoiser (MVTools, Alexander Fizick et al.),
 * from the published filter formulas only; no GPL code was copied.
 *   - Hierarchical pyramid reduction with the separable bilinear kernel
 *     (1,3,3,1)/8 (mvtools "rfilter=1"): a horizontal (1,3,3,1)/8 pass at
 *     the even columns kept by the 2:1 decimation, then the same kernel
 *     vertically.  Edge handling clamps source indices, which is exactly
 *     equivalent to filtering an edge-replicated image.
 *   - Half-pel phase planes with the 6-tap Wiener filter
 *     (1,-5,20,20,-5,1)/32 (mvtools "sharp=2").  The xy phase is the
 *     horizontal 6-tap applied to the vertically interpolated phase.
 *     Intermediates and results are clamped to the pixel range.
 *   - Every malloc'd plane buffer is edge-replicated by `pad` samples on
 *     all four sides so clamped block fetches never read out of bounds.
 *
 * Sample layout of every malloc'd buffer in this module:
 *   logical sample (x, y), x in [-pad, w+pad), y in [-pad, h+pad)
 *   lives at data[(y + pad) * stride + (x + pad)];
 *   stride is in SAMPLES (uint8 units for 8-bit planes, uint16 units for
 *   highbd planes), rounded up to a multiple of 16.
 *
 * McdPelSet phase order: [0] integer, [1] x-half, [2] y-half, [3] xy-half.
 * Phase plane sample (ix, iy) holds the interpolated value at full
 * resolution coordinate (ix + fx/2, iy + fy/2), fx/fy in {0,1}.
 */

#include <stdlib.h>
#include <string.h>

#include "mc_denoise.h"
#include "mc_denoise_simd.h"

#define MCD_PYR_MAX_LEVELS  12  /* hard cap on pyramid depth                */
#define MCD_STRIDE_ALIGN    16  /* samples per row alignment                */
#define MCD_PYR_MIN_DIM      8  /* stop reducing below this dimension       */

/* ---------------------------------------------------------------------- */
/* small helpers                                                          */

static int32_t mcd_clampi(int32_t v, int32_t lo, int32_t hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

static uint32_t mcd_align_stride(uint32_t samples)
{
    return (samples + (MCD_STRIDE_ALIGN - 1)) & ~(uint32_t)(MCD_STRIDE_ALIGN - 1);
}

static size_t mcd_plane_bytes(uint32_t stride, uint32_t rows, uint32_t highbd)
{
    return (size_t)stride * rows * (highbd ? 2u : 1u);
}

/* ---------------------------------------------------------------------- */
/* edge-replication padding                                               */

/* Fill the `pad` margin of a plane whose core [w x h] samples already sit
 * at offset (pad, pad); replicates edges (corners included). */
static void mcd_pad_fill_u8(uint8_t *data, uint32_t stride,
                            uint32_t w, uint32_t h, uint32_t pad)
{
    const uint32_t full = w + 2 * pad;
    uint32_t x, y;

    for (y = 0; y < h; y++) {
        uint8_t *row = data + (size_t)(pad + y) * stride;
        const uint8_t left = row[pad];
        const uint8_t right = row[pad + w - 1];
        for (x = 0; x < pad; x++) {
            row[x] = left;
            row[pad + w + x] = right;
        }
    }
    for (y = 0; y < pad; y++) {
        memcpy(data + (size_t)y * stride,
               data + (size_t)pad * stride, full);
        memcpy(data + (size_t)(pad + h + y) * stride,
               data + (size_t)(pad + h - 1) * stride, full);
    }
}

static void mcd_pad_fill_u16(uint16_t *data, uint32_t stride,
                             uint32_t w, uint32_t h, uint32_t pad)
{
    const uint32_t full = w + 2 * pad;
    uint32_t y;

    for (y = 0; y < h; y++) {
        uint16_t *row = data + (size_t)(pad + y) * stride;
        const uint16_t left = row[pad];
        const uint16_t right = row[pad + w - 1];
        uint32_t x;
        for (x = 0; x < pad; x++) {
            row[x] = left;
            row[pad + w + x] = right;
        }
    }
    for (y = 0; y < pad; y++) {
        memcpy(data + (size_t)y * stride,
               data + (size_t)pad * stride, full * sizeof(uint16_t));
        memcpy(data + (size_t)(pad + h + y) * stride,
               data + (size_t)(pad + h - 1) * stride, full * sizeof(uint16_t));
    }
}

/* Copy a source plane into the core of a padded destination buffer. */
static void mcd_plane_copy_in_u8(const uint8_t *src, uint32_t sstride,
                                 uint32_t w, uint32_t h,
                                 uint8_t *dst, uint32_t dstride, uint32_t pad)
{
    uint32_t y;
    for (y = 0; y < h; y++)
        memcpy(dst + (size_t)(pad + y) * dstride + pad,
               src + (size_t)y * sstride, w);
}

static void mcd_plane_copy_in_u16(const uint16_t *src, uint32_t sstride,
                                  uint32_t w, uint32_t h,
                                  uint16_t *dst, uint32_t dstride, uint32_t pad)
{
    uint32_t y;
    for (y = 0; y < h; y++)
        memcpy(dst + (size_t)(pad + y) * dstride + pad,
               src + (size_t)y * sstride, w * sizeof(uint16_t));
}

/* ---------------------------------------------------------------------- */
/* pyramid reduction: separable (1,3,3,1)/8, edges clamped                */

static void mcd_reduce_u8(const McdPlane *src, uint8_t *dst,
                          uint32_t w2, uint32_t h2, uint32_t dstride,
                          uint8_t *tmp, uint32_t tstride)
{
    const uint32_t w = src->w, h = src->h, sstride = src->stride;
    const uint8_t *sdata = (const uint8_t *)src->data;
    uint32_t y, j, i;

    /* horizontal pass at the even columns kept by the decimation */
    for (y = 0; y < h; y++) {
        uint8_t *trow = tmp + (size_t)y * tstride;
        const int32_t yc = (int32_t)y;
        for (j = 0; j < w2; j++) {
            const int32_t xc = (int32_t)(2 * j);
            const int32_t t =
                sdata[(size_t)yc * sstride + (uint32_t)mcd_clampi(xc - 1, 0, (int32_t)w - 1)] +
                3 * sdata[(size_t)yc * sstride + (uint32_t)mcd_clampi(xc, 0, (int32_t)w - 1)] +
                3 * sdata[(size_t)yc * sstride + (uint32_t)mcd_clampi(xc + 1, 0, (int32_t)w - 1)] +
                sdata[(size_t)yc * sstride + (uint32_t)mcd_clampi(xc + 2, 0, (int32_t)w - 1)];
            trow[j] = (uint8_t)((t + 4) >> 3);
        }
    }
    /* vertical pass with 2:1 decimation */
    for (i = 0; i < h2; i++) {
        uint8_t *drow = dst + (size_t)i * dstride;
        const int32_t yc = (int32_t)(2 * i);
        const uint32_t y0 = (uint32_t)mcd_clampi(yc - 1, 0, (int32_t)h - 1);
        const uint32_t y1 = (uint32_t)mcd_clampi(yc, 0, (int32_t)h - 1);
        const uint32_t y2 = (uint32_t)mcd_clampi(yc + 1, 0, (int32_t)h - 1);
        const uint32_t y3 = (uint32_t)mcd_clampi(yc + 2, 0, (int32_t)h - 1);
        for (j = 0; j < w2; j++) {
            const int32_t t = tmp[(size_t)y0 * tstride + j] +
                              3 * tmp[(size_t)y1 * tstride + j] +
                              3 * tmp[(size_t)y2 * tstride + j] +
                              tmp[(size_t)y3 * tstride + j];
            drow[j] = (uint8_t)((t + 4) >> 3);
        }
    }
}

static void mcd_reduce_u16(const McdPlane *src, uint16_t *dst,
                           uint32_t w2, uint32_t h2, uint32_t dstride,
                           uint16_t *tmp, uint32_t tstride)
{
    const uint32_t w = src->w, h = src->h, sstride = src->stride;
    const uint16_t *sdata = (const uint16_t *)src->data;
    uint32_t y, j, i;

    for (y = 0; y < h; y++) {
        uint16_t *trow = tmp + (size_t)y * tstride;
        const int32_t yc = (int32_t)y;
        for (j = 0; j < w2; j++) {
            const int32_t xc = (int32_t)(2 * j);
            const int32_t t =
                sdata[(size_t)yc * sstride + (uint32_t)mcd_clampi(xc - 1, 0, (int32_t)w - 1)] +
                3 * sdata[(size_t)yc * sstride + (uint32_t)mcd_clampi(xc, 0, (int32_t)w - 1)] +
                3 * sdata[(size_t)yc * sstride + (uint32_t)mcd_clampi(xc + 1, 0, (int32_t)w - 1)] +
                sdata[(size_t)yc * sstride + (uint32_t)mcd_clampi(xc + 2, 0, (int32_t)w - 1)];
            trow[j] = (uint16_t)((t + 4) >> 3);
        }
    }
    for (i = 0; i < h2; i++) {
        uint16_t *drow = dst + (size_t)i * dstride;
        const int32_t yc = (int32_t)(2 * i);
        const uint32_t y0 = (uint32_t)mcd_clampi(yc - 1, 0, (int32_t)h - 1);
        const uint32_t y1 = (uint32_t)mcd_clampi(yc, 0, (int32_t)h - 1);
        const uint32_t y2 = (uint32_t)mcd_clampi(yc + 1, 0, (int32_t)h - 1);
        const uint32_t y3 = (uint32_t)mcd_clampi(yc + 2, 0, (int32_t)h - 1);
        for (j = 0; j < w2; j++) {
            const int32_t t = tmp[(size_t)y0 * tstride + j] +
                              3 * tmp[(size_t)y1 * tstride + j] +
                              3 * tmp[(size_t)y2 * tstride + j] +
                              tmp[(size_t)y3 * tstride + j];
            drow[j] = (uint16_t)((t + 4) >> 3);
        }
    }
}

/* ---------------------------------------------------------------------- */
/* pyramid object                                                         */

struct McdPyramid {
    uint32_t n_planes;
    uint32_t n_levels;
    McdPlane planes[MCD_MAX_PLANES][MCD_PYR_MAX_LEVELS];
    void    *bufs[MCD_MAX_PLANES][MCD_PYR_MAX_LEVELS]; /* NULL at level 0  */
};

int mcd_pyramid_create(const McdPlane *src_planes, uint32_t n_planes,
                       uint32_t pad, uint32_t max_levels, McdPyramid **out)
{
    McdPyramid *p;
    uint32_t i, level, n_levels;
    uint32_t cur_w, cur_h;
    size_t max_tmp;
    void *scratch;

    if (!out)
        return MCD_ERR_PARAM;
    *out = NULL;
    if (!src_planes || n_planes < 1 || n_planes > MCD_MAX_PLANES ||
        pad < 1 || max_levels < 1)
        return MCD_ERR_PARAM;
    for (i = 0; i < n_planes; i++)
        if (!src_planes[i].data || src_planes[i].w < 1 || src_planes[i].h < 1 ||
            src_planes[i].stride < src_planes[i].w)
            return MCD_ERR_PARAM;
    if (max_levels > MCD_PYR_MAX_LEVELS)
        max_levels = MCD_PYR_MAX_LEVELS;

    p = calloc(1, sizeof(*p));
    if (!p)
        return MCD_ERR_NOMEM;
    p->n_planes = n_planes;

    /* level count is driven by the luma plane dimensions; all planes share
     * the same level grid so callers can index level L on every plane */
    cur_w = src_planes[0].w;
    cur_h = src_planes[0].h;
    n_levels = 1;
    while (n_levels < max_levels && cur_w >= MCD_PYR_MIN_DIM && cur_h >= MCD_PYR_MIN_DIM) {
        cur_w = (cur_w + 1) >> 1;
        cur_h = (cur_h + 1) >> 1;
        n_levels++;
    }
    p->n_levels = n_levels;

    for (i = 0; i < n_planes; i++)
        p->planes[i][0] = src_planes[i]; /* level 0 references the source */

    /* scratch for the horizontal pass: worst case is level 0 -> 1
     * (h rows of w/2 samples); u16-sized so both paths share it */
    max_tmp = 0;
    for (i = 0; i < n_planes; i++) {
        const size_t s = (size_t)src_planes[i].h * ((src_planes[i].w + 1) >> 1);
        if (s > max_tmp)
            max_tmp = s;
    }
    scratch = malloc(max_tmp * 2);
    if (!scratch) {
        mcd_pyramid_destroy(&p);
        return MCD_ERR_NOMEM;
    }

    for (level = 1; level < n_levels; level++) {
        for (i = 0; i < n_planes; i++) {
            const McdPlane *sp = &p->planes[i][level - 1];
            const uint32_t w2 = (sp->w + 1) >> 1;
            const uint32_t h2 = (sp->h + 1) >> 1;
            const uint32_t stride = mcd_align_stride(w2 + 2 * pad);
            const uint32_t rows = h2 + 2 * pad;
            uint8_t *buf = malloc(mcd_plane_bytes(stride, rows, sp->highbd));
            McdPlane *dp;

            if (!buf) {
                free(scratch);
                mcd_pyramid_destroy(&p);
                return MCD_ERR_NOMEM;
            }
            dp = &p->planes[i][level];
            dp->w = w2;
            dp->h = h2;
            dp->stride = stride;
            dp->highbd = sp->highbd;
            /* data points at the visible origin; the edge-replicated pad is
             * reachable via negative row/column offsets (buf keeps ownership) */
            dp->data = sp->highbd ? (void *)((uint16_t *)buf + (size_t)pad * stride + pad)
                                  : (void *)(buf + (size_t)pad * stride + pad);
            p->bufs[i][level] = buf;

            if (!sp->highbd) {
                mcd_reduce_u8(sp, buf + (size_t)pad * stride + pad,
                              w2, h2, stride, (uint8_t *)scratch, w2);
                mcd_pad_fill_u8(buf, stride, w2, h2, pad);
            } else {
                mcd_reduce_u16(sp, (uint16_t *)buf + (size_t)pad * stride + pad,
                               w2, h2, stride, (uint16_t *)scratch, w2);
                mcd_pad_fill_u16((uint16_t *)buf, stride, w2, h2, pad);
            }
        }
    }
    free(scratch);
    *out = p;
    return MCD_SUCCESS;
}

void mcd_pyramid_destroy(McdPyramid **p)
{
    uint32_t i, l;

    if (!p || !*p)
        return;
    for (i = 0; i < (*p)->n_planes; i++)
        for (l = 1; l < (*p)->n_levels; l++)
            free((*p)->bufs[i][l]);
    free(*p);
    *p = NULL;
}

const McdPlane *mcd_pyramid_get(const McdPyramid *p, uint32_t plane, uint32_t level)
{
    if (!p || plane >= p->n_planes || level >= p->n_levels)
        return NULL;
    return &p->planes[plane][level];
}

uint32_t mcd_pyramid_levels(const McdPyramid *p)
{
    return p ? p->n_levels : 0;
}

/* ---------------------------------------------------------------------- */
/* half-pel phase planes: 6-tap Wiener (1,-5,20,20,-5,1)/32               */
/*                                                                        */
/* All filters read a padded plane and clamp tap indices into the CORE    */
/* [0, dim-1]; since the padded plane replicates its edges, this is       */
/* identical to filtering an edge-replicated source (mvtools behavior)    */
/* and stays in bounds for every padded output position.                  */

static void mcd_halfpel_h_u8(const uint8_t *src, uint32_t sstride,
                             uint32_t w, uint32_t h, uint32_t pad,
                             uint8_t *dst, uint32_t dstride, uint32_t maxval)
{
    if (mcd_use_avx2()) {
        mcd_simd_halfpel_h_u8(src, sstride, w, h, pad, dst, dstride, maxval);
        return;
    }
    const uint32_t tw = w + 2 * pad;
    const uint32_t th = h + 2 * pad;
    const int32_t wm1 = (int32_t)w - 1;
    uint32_t X, Y;

    for (Y = 0; Y < th; Y++) {
        const uint8_t *srow = src + (size_t)Y * sstride;
        uint8_t *drow = dst + (size_t)Y * dstride;
        for (X = 0; X < tw; X++) {
            const int32_t x = (int32_t)X - (int32_t)pad;
            const int32_t v = (srow[mcd_clampi(x - 2, 0, wm1) + (int32_t)pad] -
                               5 * srow[mcd_clampi(x - 1, 0, wm1) + (int32_t)pad] +
                              20 * srow[mcd_clampi(x, 0, wm1) + (int32_t)pad] +
                              20 * srow[mcd_clampi(x + 1, 0, wm1) + (int32_t)pad] -
                               5 * srow[mcd_clampi(x + 2, 0, wm1) + (int32_t)pad] +
                                   srow[mcd_clampi(x + 3, 0, wm1) + (int32_t)pad] + 16) >> 5;
            drow[X] = (uint8_t)(v < 0 ? 0 : (v > (int32_t)maxval ? (int32_t)maxval : v));
        }
    }
}

static void mcd_halfpel_v_u8(const uint8_t *src, uint32_t sstride,
                             uint32_t w, uint32_t h, uint32_t pad,
                             uint8_t *dst, uint32_t dstride, uint32_t maxval)
{
    if (mcd_use_avx2()) {
        mcd_simd_halfpel_v_u8(src, sstride, w, h, pad, dst, dstride, maxval);
        return;
    }
    const uint32_t tw = w + 2 * pad;
    const uint32_t th = h + 2 * pad;
    const int32_t hm1 = (int32_t)h - 1;
    uint32_t X, Y;

    for (Y = 0; Y < th; Y++) {
        uint8_t *drow = dst + (size_t)Y * dstride;
        const int32_t y = (int32_t)Y - (int32_t)pad;
        const uint32_t r0 = (uint32_t)(mcd_clampi(y - 2, 0, hm1) + (int32_t)pad);
        const uint32_t r1 = (uint32_t)(mcd_clampi(y - 1, 0, hm1) + (int32_t)pad);
        const uint32_t r2 = (uint32_t)(mcd_clampi(y, 0, hm1) + (int32_t)pad);
        const uint32_t r3 = (uint32_t)(mcd_clampi(y + 1, 0, hm1) + (int32_t)pad);
        const uint32_t r4 = (uint32_t)(mcd_clampi(y + 2, 0, hm1) + (int32_t)pad);
        const uint32_t r5 = (uint32_t)(mcd_clampi(y + 3, 0, hm1) + (int32_t)pad);
        for (X = 0; X < tw; X++) {
            const int32_t v = (src[(size_t)r0 * sstride + X] -
                               5 * src[(size_t)r1 * sstride + X] +
                              20 * src[(size_t)r2 * sstride + X] +
                              20 * src[(size_t)r3 * sstride + X] -
                               5 * src[(size_t)r4 * sstride + X] +
                                   src[(size_t)r5 * sstride + X] + 16) >> 5;
            drow[X] = (uint8_t)(v < 0 ? 0 : (v > (int32_t)maxval ? (int32_t)maxval : v));
        }
    }
}

static void mcd_halfpel_h_u16(const uint16_t *src, uint32_t sstride,
                              uint32_t w, uint32_t h, uint32_t pad,
                              uint16_t *dst, uint32_t dstride, uint32_t maxval)
{
    if (mcd_use_avx512()) {
        mcd_simd_halfpel_h_u16(src, sstride, w, h, pad, dst, dstride, maxval);
        return;
    }
    const uint32_t tw = w + 2 * pad;
    const uint32_t th = h + 2 * pad;
    const int32_t wm1 = (int32_t)w - 1;
    uint32_t X, Y;

    for (Y = 0; Y < th; Y++) {
        const uint16_t *srow = src + (size_t)Y * sstride;
        uint16_t *drow = dst + (size_t)Y * dstride;
        for (X = 0; X < tw; X++) {
            const int32_t x = (int32_t)X - (int32_t)pad;
            const int32_t v = (srow[mcd_clampi(x - 2, 0, wm1) + (int32_t)pad] -
                               5 * srow[mcd_clampi(x - 1, 0, wm1) + (int32_t)pad] +
                              20 * srow[mcd_clampi(x, 0, wm1) + (int32_t)pad] +
                              20 * srow[mcd_clampi(x + 1, 0, wm1) + (int32_t)pad] -
                               5 * srow[mcd_clampi(x + 2, 0, wm1) + (int32_t)pad] +
                                   srow[mcd_clampi(x + 3, 0, wm1) + (int32_t)pad] + 16) >> 5;
            drow[X] = (uint16_t)(v < 0 ? 0 : (v > (int32_t)maxval ? (int32_t)maxval : v));
        }
    }
}

static void mcd_halfpel_v_u16(const uint16_t *src, uint32_t sstride,
                              uint32_t w, uint32_t h, uint32_t pad,
                              uint16_t *dst, uint32_t dstride, uint32_t maxval)
{
    if (mcd_use_avx512()) {
        mcd_simd_halfpel_v_u16(src, sstride, w, h, pad, dst, dstride, maxval);
        return;
    }
    const uint32_t tw = w + 2 * pad;
    const uint32_t th = h + 2 * pad;
    const int32_t hm1 = (int32_t)h - 1;
    uint32_t X, Y;

    for (Y = 0; Y < th; Y++) {
        uint16_t *drow = dst + (size_t)Y * dstride;
        const int32_t y = (int32_t)Y - (int32_t)pad;
        const uint32_t r0 = (uint32_t)(mcd_clampi(y - 2, 0, hm1) + (int32_t)pad);
        const uint32_t r1 = (uint32_t)(mcd_clampi(y - 1, 0, hm1) + (int32_t)pad);
        const uint32_t r2 = (uint32_t)(mcd_clampi(y, 0, hm1) + (int32_t)pad);
        const uint32_t r3 = (uint32_t)(mcd_clampi(y + 1, 0, hm1) + (int32_t)pad);
        const uint32_t r4 = (uint32_t)(mcd_clampi(y + 2, 0, hm1) + (int32_t)pad);
        const uint32_t r5 = (uint32_t)(mcd_clampi(y + 3, 0, hm1) + (int32_t)pad);
        for (X = 0; X < tw; X++) {
            const int32_t v = (src[(size_t)r0 * sstride + X] -
                               5 * src[(size_t)r1 * sstride + X] +
                              20 * src[(size_t)r2 * sstride + X] +
                              20 * src[(size_t)r3 * sstride + X] -
                               5 * src[(size_t)r4 * sstride + X] +
                                   src[(size_t)r5 * sstride + X] + 16) >> 5;
            drow[X] = (uint16_t)(v < 0 ? 0 : (v > (int32_t)maxval ? (int32_t)maxval : v));
        }
    }
}

/* ---------------------------------------------------------------------- */
/* pelset object                                                          */

struct McdPelSet {
    uint32_t n_planes;
    uint32_t pad;
    McdPlane ph[MCD_MAX_PLANES][4]; /* [plane][0=int 1=xhalf 2=yhalf 3=xy] */
    void    *bufs[MCD_MAX_PLANES];  /* one malloc holding the 4 phases;
                                     * NULL for an adopted (borrowed) plane */
};

/* Build one plane's four phases. When `borrow` is non-NULL it must be a
 * padded plane identical to what phase 0 would be (same w/h/stride/highbd,
 * edge-replicated margin of exactly `pad`, data pointer at the visible
 * origin): its storage is adopted for phase 0 and only phases 1..3 are
 * allocated (offsets 0..2pb). Otherwise all four phases live in one malloc
 * (offsets 0..3pb). Returns MCD_SUCCESS or MCD_ERR_NOMEM/PARAM; on failure
 * the pelset is left consistent for mcd_pelset_destroy. */
static int mcd_pelset_build_plane(McdPelSet *s, uint32_t i,
                                  const McdPlane *src, uint32_t pad,
                                  uint32_t maxval, const McdPlane *borrow)
{
    const uint32_t w = src->w, h = src->h;
    const uint32_t stride = mcd_align_stride(w + 2 * pad);
    const uint32_t rows = h + 2 * pad;
    const size_t pb = mcd_plane_bytes(stride, rows, src->highbd);
    const size_t esz = src->highbd ? 2u : 1u;
    uint8_t *buf;         /* phases 1..3 when borrowing, 0..3 otherwise */
    uint32_t k;

    if (borrow) {
        if (borrow->w != w || borrow->h != h || borrow->stride != stride ||
            borrow->highbd != src->highbd)
            return MCD_ERR_PARAM;
        buf = malloc(pb * 3); /* xhalf + yhalf + xyhalf */
        if (!buf)
            return MCD_ERR_NOMEM;
        s->bufs[i] = buf;
        s->ph[i][0].data = (uint8_t *)borrow->data -
                           ((size_t)pad * stride + pad) * esz;
    } else {
        buf = malloc(pb * 4); /* integer + xhalf + yhalf + xyhalf */
        if (!buf)
            return MCD_ERR_NOMEM;
        s->bufs[i] = buf;
        s->ph[i][0].data = buf;
    }
    /* phases 1..3 in buf; when borrowing, buf holds phases 1..3 at offsets
     * 0..2pb, else all four phases sit in buf at offsets 0..3pb */
    const size_t wbase = borrow ? 0 : pb; /* buf offset of phase 1 */
    for (k = 0; k < 4; k++) {
        McdPlane *ph = &s->ph[i][k];
        ph->w = w;
        ph->h = h;
        ph->stride = stride;
        ph->highbd = src->highbd;
        /* k == 0 with borrow is the adopted phase 0, data set above */
        if (k)
            ph->data = buf + (k - 1) * pb + wbase;
    }

    /* phase 0 storage: the adopted plane when borrowing (its buffer origin;
     * NOT inside buf), else buf itself */
    uint8_t *src0;

    if (!borrow) {
        if (!src->highbd) {
            mcd_plane_copy_in_u8((const uint8_t *)src->data, src->stride,
                                 w, h, buf, stride, pad);
            mcd_pad_fill_u8(buf, stride, w, h, pad);
        } else {
            mcd_plane_copy_in_u16((const uint16_t *)src->data, src->stride,
                                  w, h, (uint16_t *)buf, stride, pad);
            mcd_pad_fill_u16((uint16_t *)buf, stride, w, h, pad);
        }
        src0 = buf;
    } else {
        src0 = (uint8_t *)borrow->data - ((size_t)pad * stride + pad) * esz;
    }
    if (!src->highbd) {
        mcd_halfpel_h_u8(src0, stride, w, h, pad,
                         buf + wbase, stride, maxval);
        mcd_halfpel_v_u8(src0, stride, w, h, pad,
                         buf + wbase + pb, stride, maxval);
        /* xy: horizontal 6-tap over the vertically interpolated phase */
        mcd_halfpel_h_u8(buf + wbase + pb, stride, w, h, pad,
                         buf + wbase + 2 * pb, stride, maxval);
    } else {
        uint16_t *s0 = (uint16_t *)src0;
        uint16_t *b = (uint16_t *)buf;
        const size_t wb16 = wbase / 2; /* wbase is even: stride*rows*esq */
        mcd_halfpel_h_u16(s0, stride, w, h, pad,
                          b + wb16, stride, maxval);
        mcd_halfpel_v_u16(s0, stride, w, h, pad,
                          b + wb16 + (size_t)pb / 2, stride, maxval);
        mcd_halfpel_h_u16(b + wb16 + (size_t)pb / 2, stride, w, h, pad,
                          b + wb16 + (size_t)pb, stride, maxval);
    }
    return MCD_SUCCESS;
}

int mcd_pelset_create(const McdPlane *src_planes, uint32_t n_planes,
                      uint32_t bitdepth, uint32_t pad, McdPelSet **out)
{
    return mcd_pelset_create_borrow0(src_planes, n_planes, bitdepth, pad, NULL, out);
}

int mcd_pelset_create_borrow0(const McdPlane *src_planes, uint32_t n_planes,
                              uint32_t bitdepth, uint32_t pad,
                              const McdPlane *borrow0, McdPelSet **out)
{
    McdPelSet *s;
    uint32_t i;
    uint32_t maxval;

    if (!out)
        return MCD_ERR_PARAM;
    *out = NULL;
    if (!src_planes || n_planes < 1 || n_planes > MCD_MAX_PLANES || pad < 1 ||
        (bitdepth != 8 && bitdepth != 10 && bitdepth != 12))
        return MCD_ERR_PARAM;
    for (i = 0; i < n_planes; i++)
        if (!src_planes[i].data || src_planes[i].w < 1 || src_planes[i].h < 1 ||
            src_planes[i].stride < src_planes[i].w)
            return MCD_ERR_PARAM;
    for (i = 0; i < n_planes; i++)
        if (src_planes[i].highbd != (bitdepth > 8))
            return MCD_ERR_PARAM;
    if (borrow0 && n_planes >= 1 && !borrow0->data)
        return MCD_ERR_PARAM;

    maxval = (1u << bitdepth) - 1u;

    s = calloc(1, sizeof(*s));
    if (!s)
        return MCD_ERR_NOMEM;
    s->n_planes = n_planes;
    s->pad = pad;

    for (i = 0; i < n_planes; i++) {
        const int rc = mcd_pelset_build_plane(s, i, &src_planes[i], pad, maxval,
                                              (i == 0) ? borrow0 : NULL);
        if (rc != MCD_SUCCESS) {
            mcd_pelset_destroy(&s);
            return rc;
        }
    }
    *out = s;
    return MCD_SUCCESS;
}

void mcd_pelset_destroy(McdPelSet **p)
{
    uint32_t i;

    if (!p || !*p)
        return;
    for (i = 0; i < (*p)->n_planes; i++)
        free((*p)->bufs[i]);
    free(*p);
    *p = NULL;
}

void mcd_pelset_locate(const McdPelSet *s, uint32_t plane,
                       int32_t x_halfpel, int32_t y_halfpel,
                       uint32_t w, uint32_t h, uint32_t eff_pad,
                       const void **row0, int32_t *stride)
{
    const McdPlane *p;
    int32_t lo_x, hi_x, lo_y, hi_y, ix, iy;
    uint32_t fx, fy, phase;

    if (!row0 || !stride) {
        if (row0)
            *row0 = NULL;
        if (stride)
            *stride = 0;
        return;
    }
    if (!s || w == 0 || h == 0 || plane >= s->n_planes) {
        *row0  = NULL;
        *stride = 0;
        return;
    }
    if (eff_pad > s->pad)
        eff_pad = s->pad; /* never reach outside the built margin */

    /* identical clamp + phase selection to mcd_pelset_fetch_pad */
    lo_x = -(int32_t)eff_pad;
    hi_x = (int32_t)s->ph[plane][0].w + (int32_t)eff_pad - (int32_t)w;
    lo_y = -(int32_t)eff_pad;
    hi_y = (int32_t)s->ph[plane][0].h + (int32_t)eff_pad - (int32_t)h;
    if (hi_x < lo_x) /* degenerate: padded region narrower than the block */
        hi_x = lo_x;
    if (hi_y < lo_y)
        hi_y = lo_y;
    x_halfpel = mcd_clampi(x_halfpel, 2 * lo_x, 2 * hi_x + 1);
    y_halfpel = mcd_clampi(y_halfpel, 2 * lo_y, 2 * hi_y + 1);

    /* half-pel split: x = 2*ix + fx (arithmetic shift, GCC/Clang/MSVC) */
    ix = x_halfpel >> 1;
    iy = y_halfpel >> 1;
    fx = (uint32_t)(x_halfpel & 1);
    fy = (uint32_t)(y_halfpel & 1);
    phase = fx + 2 * fy;

    p = &s->ph[plane][phase];
    *stride = (int32_t)p->stride;
    *row0 = (const uint8_t *)p->data +
            ((size_t)(iy + (int32_t)s->pad) * p->stride + (uint32_t)(ix + (int32_t)s->pad)) *
                (p->highbd ? 2u : 1u);
}

void mcd_pelset_fetch(const McdPelSet *s, uint32_t plane,
                      int32_t x_halfpel, int32_t y_halfpel,
                      uint32_t w, uint32_t h, uint16_t *dst)
{
    if (!s)
        return;
    mcd_pelset_fetch_pad(s, plane, x_halfpel, y_halfpel, w, h, s->pad, dst);
}

void mcd_pelset_fetch_pad(const McdPelSet *s, uint32_t plane,
                          int32_t x_halfpel, int32_t y_halfpel,
                          uint32_t w, uint32_t h, uint32_t eff_pad,
                          uint16_t *dst)
{
    const McdPlane *p;
    int32_t lo_x, hi_x, lo_y, hi_y, ix, iy;
    uint32_t fx, fy, phase, r, c;

    if (!s || !dst || w == 0 || h == 0 || plane >= s->n_planes)
        return;
    if (eff_pad > s->pad)
        eff_pad = s->pad; /* never reach outside the built margin */

    /* clamp into the valid padded region so any vector is safe */
    lo_x = -(int32_t)eff_pad;
    hi_x = (int32_t)s->ph[plane][0].w + (int32_t)eff_pad - (int32_t)w;
    lo_y = -(int32_t)eff_pad;
    hi_y = (int32_t)s->ph[plane][0].h + (int32_t)eff_pad - (int32_t)h;
    if (hi_x < lo_x) /* degenerate: padded region narrower than the block */
        hi_x = lo_x;
    if (hi_y < lo_y)
        hi_y = lo_y;
    x_halfpel = mcd_clampi(x_halfpel, 2 * lo_x, 2 * hi_x + 1);
    y_halfpel = mcd_clampi(y_halfpel, 2 * lo_y, 2 * hi_y + 1);

    /* half-pel split: x = 2*ix + fx (arithmetic shift, GCC/Clang/MSVC) */
    ix = x_halfpel >> 1;
    iy = y_halfpel >> 1;
    fx = (uint32_t)(x_halfpel & 1);
    fy = (uint32_t)(y_halfpel & 1);
    phase = fx + 2 * fy;

    p = &s->ph[plane][phase];
    {
        const uint32_t sx = (uint32_t)(ix + (int32_t)s->pad);
        const uint32_t sy = (uint32_t)(iy + (int32_t)s->pad);
        if (!p->highbd) {
            const uint8_t *data = (const uint8_t *)p->data;
            const int simd = mcd_use_avx2();
            for (r = 0; r < h; r++) {
                const uint8_t *row = data + (size_t)(sy + r) * p->stride + sx;
                uint16_t *out = dst + (size_t)r * w;
                if (simd)
                    mcd_simd_fetch_u8_row(row, out, w);
                else
                    for (c = 0; c < w; c++)
                        out[c] = (uint16_t)row[c];
            }
        } else {
            const uint16_t *data = (const uint16_t *)p->data;
            for (r = 0; r < h; r++) {
                const uint16_t *row = data + (size_t)(sy + r) * p->stride + sx;
                uint16_t *out = dst + (size_t)r * w;
                memcpy(out, row, w * sizeof(uint16_t));
            }
        }
    }
}
