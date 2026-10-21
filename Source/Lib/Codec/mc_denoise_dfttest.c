/*
 * mc_denoise_dfttest.c — dfttest spatial prefilter + luma-mask merge for the
 * motion-compensated denoiser (contract in mc_denoise.h).
 *
 * Clean-room implementation: written from the published algorithm description
 * only — overlapped block-DFT denoising (16x16 blocks, 4-sample stepping,
 * overlap-normalized Hann analysis/synthesis windows, type-0 Wiener
 * shrinkage, zmean DC restoration) with a textbook radix-2 Cooley-Tukey FFT
 * of length 16.  No code was copied from GPL-licensed references (dfttest,
 * vs-dfttest2).
 *
 * Normalization convention (unity gain by construction):
 *   - forward FFT unnormalized, inverse kernel uses conjugated twiddles and
 *     is also unnormalized, so fwd-then-inv == 256 * input;
 *   - analysis = synthesis window win[i][j] = w'[i] * w'[j] / 16 where the
 *     /16 carries exactly that 1/256, so no extra synthesis factor is needed;
 *   - w'[i] = w[i] / sqrt(sum of w[j]^2 for j == i (mod 4)) makes the sum of
 *     win^2 over the 4x4 blocks covering any pixel equal 1/256, hence
 *     overlap-add reproduces the input when sigma_pct == 0.
 *
 * Implementation notes / contract gaps:
 *  - mcd_dfttest_plane allocates dst->data with malloc when it is NULL and
 *    fills the dst geometry from src; an already-allocated dst with matching
 *    geometry is reused.  The caller (glue) owns and frees the buffer.
 *  - Bit depth is supplied by the caller (McdPlane only distinguishes 8-bit
 *    from highbd storage), and must be 8, 10, or 12 with storage matching it.
 *  - Frame borders are padded by symmetric mirroring (edge pixel
 *    duplicated), which makes every 4-stepped block position computable.
 */

#include "mc_denoise.h"
#include "mc_denoise_simd.h"
#include "mc_denoise_dfttest_ctx.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Symmetric mirror reflection of i into [0, w). */
static inline int32_t mcd_mirror(int32_t i, uint32_t w)
{
    if (w == 1)
        return 0;
    while (i < 0 || i >= (int32_t)w) {
        if (i < 0)
            i = -i - 1;
        else
            i = 2 * (int32_t)w - 1 - i;
    }
    return i;
}

/* ---------------------------------------------------------------------- */
/* length-16 radix-2 decimation-in-time complex FFT, in place.             */
/* forward: X[k] = sum x[n] * e^{-2*pi*i*kn/16} (unnormalized).            */
/* inverse: same kernel with conjugated twiddles (unnormalized), so        */
/* forward followed by inverse equals 256 * input.                         */

static void mcd_fft16(const McdDftCtx *c, float *re, float *im, int inverse)
{
    for (uint32_t i = 0; i < MCD_DFT_N; i++) {
        const uint32_t j = c->brev[i];
        if (j > i) {
            float t  = re[i]; re[i] = re[j]; re[j] = t;
            t        = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
    for (uint32_t len = 2; len <= MCD_DFT_N; len <<= 1) {
        const uint32_t half  = len >> 1;
        const uint32_t tstep = MCD_DFT_N / len;
        for (uint32_t ib = 0; ib < MCD_DFT_N; ib += len) {
            for (uint32_t j = 0; j < half; j++) {
                const float wr = c->twr[j * tstep];
                const float wi = inverse ? -c->twi[j * tstep] : c->twi[j * tstep];
                const float ar = re[ib + j];
                const float ai = im[ib + j];
                const float br = re[ib + j + half];
                const float bi = im[ib + j + half];
                const float tr = br * wr - bi * wi;
                const float ti = br * wi + bi * wr;
                re[ib + j]         = ar + tr;
                im[ib + j]         = ai + ti;
                re[ib + j + half]  = ar - tr;
                im[ib + j + half]  = ai - ti;
            }
        }
    }
}

/* 2D transform: rows in place, columns via gather/scatter scratch. */
static void mcd_fft2d(const McdDftCtx *c, float *re, float *im, int inverse)
{
    float cr[MCD_DFT_N], ci[MCD_DFT_N];

    for (uint32_t r = 0; r < MCD_DFT_N; r++)
        mcd_fft16(c, re + r * MCD_DFT_N, im + r * MCD_DFT_N, inverse);
    for (uint32_t col = 0; col < MCD_DFT_N; col++) {
        for (uint32_t r = 0; r < MCD_DFT_N; r++) {
            cr[r] = re[r * MCD_DFT_N + col];
            ci[r] = im[r * MCD_DFT_N + col];
        }
        mcd_fft16(c, cr, ci, inverse);
        for (uint32_t r = 0; r < MCD_DFT_N; r++) {
            re[r * MCD_DFT_N + col] = cr[r];
            im[r * MCD_DFT_N + col] = ci[r];
        }
    }
}

/* 1D noise variance profile (8-bit^2 units), piecewise-linear through
 * (0.0, 4), (0.2, 9), (1.0, 15) in the triangular frequency location
 * loc(k) = k <= 8 ? k/8 : (16-k)/8.                                        */
static inline float mcd_sigma_profile(float loc)
{
    return (loc <= 0.2f) ? (4.0f + 25.0f * loc)
                         : (9.0f + 7.5f * (loc - 0.2f));
}

static void mcd_dft_tables(McdDftCtx *c, uint32_t sigma_pct)
{
    /* twiddles and bit-reversal permutation */
    for (uint32_t k = 0; k < 8; k++) {
        const float a = (float)(2.0 * MCD_PI * (double)k / (double)MCD_DFT_N);
        c->twr[k] = cosf(a);
        c->twi[k] = -sinf(a);
    }
    for (uint32_t i = 0; i < MCD_DFT_N; i++) {
        uint32_t r = 0, v = i;
        for (uint32_t b = 0; b < 4; b++) {
            r = (r << 1) | (v & 1u);
            v >>= 1;
        }
        c->brev[i] = (uint8_t)r;
    }

    /* Hann window, overlap-normalized for the 4-sample stepping */
    float w[MCD_DFT_N], wn[MCD_DFT_N], s[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    for (uint32_t i = 0; i < MCD_DFT_N; i++)
        w[i] = 0.5f * (1.0f - cosf((float)(2.0 * MCD_PI * (double)i / (double)MCD_DFT_N)));
    for (uint32_t i = 0; i < MCD_DFT_N; i++)
        s[i & 3u] += w[i] * w[i];
    for (uint32_t i = 0; i < MCD_DFT_N; i++)
        wn[i] = w[i] / sqrtf(s[i & 3u]);

    /* analysis (= synthesis) window: the 1/16 per axis carries the
     * inverse-FFT normalization (see file header) */
    c->wscale = 0.0f;
    for (uint32_t i = 0; i < MCD_DFT_N; i++) {
        for (uint32_t j = 0; j < MCD_DFT_N; j++) {
            const float winv = wn[i] * wn[j] * (1.0f / 16.0f);
            c->win[i * MCD_DFT_N + j] = winv;
            c->wscale += winv * winv;
        }
    }

    /* zmean profile: 2D DFT of the window */
    memcpy(c->wfre, c->win, sizeof(c->wfre));
    memset(c->wfim, 0, sizeof(c->wfim));
    mcd_fft2d(c, c->wfre, c->wfim, 0);

    /* per-coefficient noise variance: profile(ky) * profile(kx) * wscale,
     * scaled linearly by sigma_pct/100 */
    const float sig = (float)sigma_pct / 100.0f;
    float p1[MCD_DFT_N];
    for (uint32_t k = 0; k < MCD_DFT_N; k++) {
        const float loc = (float)(k <= 8 ? (int32_t)k : 16 - (int32_t)k) / 8.0f;
        p1[k] = mcd_sigma_profile(loc) * sig;
    }
    for (uint32_t ky = 0; ky < MCD_DFT_N; ky++)
        for (uint32_t kx = 0; kx < MCD_DFT_N; kx++)
            c->sigma2[ky * MCD_DFT_N + kx] = p1[ky] * p1[kx] * c->wscale;
}

/* ---------------------------------------------------------------------- */
/* dfttest spatial denoise of one plane                                    */

int mcd_dfttest_ctx_create(uint32_t sigma_pct, McdDftCtx **out)
{
    if (!out || sigma_pct > 100000u)
        return MCD_ERR_PARAM;
    *out = NULL;
    McdDftCtx *ctx = calloc(1, sizeof(*ctx));
    if (!ctx)
        return MCD_ERR_NOMEM;
    mcd_dft_tables(ctx, sigma_pct);
    ctx->sigma_pct = sigma_pct;
    *out = ctx;
    return MCD_SUCCESS;
}

void mcd_dfttest_ctx_destroy(McdDftCtx **ctx)
{
    if (!ctx || !*ctx)
        return;
    free(*ctx);
    *ctx = NULL;
}

int mcd_dfttest_plane_with_ctx(const McdDftCtx *ctx, const McdPlane *src,
                               McdPlane *dst, uint32_t bitdepth)
{
    if (mcd_use_avx512()) {
        /* bit-identical 16-block vector path (see mc_denoise_simd.c) */
        return mcd_simd_dfttest_plane_with_ctx(ctx, src, dst, bitdepth);
    }
    if (!ctx || !src || !dst || !src->data ||
        (bitdepth != 8 && bitdepth != 10 && bitdepth != 12) ||
        src->highbd != (bitdepth > 8))
        return MCD_ERR_PARAM;
    const uint32_t w = src->w;
    const uint32_t h = src->h;
    if (w == 0 || h == 0 || src->stride < w)
        return MCD_ERR_PARAM;

    const uint32_t pmax   = (1u << bitdepth) - 1;
    const float    iscale = 1.0f / (float)(1u << (bitdepth - 8));
    const float    oscale = (float)(1u << (bitdepth - 8));

    /* destination: allocate, or reuse a compatible caller buffer */
    if (!dst->data) {
        dst->data = malloc((size_t)w * h * (src->highbd ? sizeof(uint16_t)
                                                        : sizeof(uint8_t)));
        if (!dst->data)
            return MCD_ERR_NOMEM;
        dst->w      = w;
        dst->h      = h;
        dst->stride = w;
        dst->highbd = src->highbd;
    } else if (dst->w != w || dst->h != h || dst->stride < w ||
               dst->highbd != src->highbd) {
        return MCD_ERR_PARAM;
    }

    const uint32_t sigma_pct = ctx->sigma_pct;

    const uint32_t pw = w + 2 * MCD_DFT_PAD;
    const uint32_t ph = h + 2 * MCD_DFT_PAD;

    int32_t *xmap = malloc(sizeof(int32_t) * pw);
    int32_t *ymap = malloc(sizeof(int32_t) * ph);
    float   *pad  = malloc(sizeof(float) * pw * ph);
    float   *acc  = calloc((size_t)w * h, sizeof(float));
    int      ret  = MCD_SUCCESS;

    if (!xmap || !ymap || !pad || !acc) {
        ret = MCD_ERR_NOMEM;
        goto cleanup;
    }

    for (uint32_t x = 0; x < pw; x++)
        xmap[x] = mcd_mirror((int32_t)x - MCD_DFT_PAD, w);
    for (uint32_t y = 0; y < ph; y++)
        ymap[y] = mcd_mirror((int32_t)y - MCD_DFT_PAD, h);

    /* mirror-padded copy in the internal 8-bit-range float scale */
    for (uint32_t y = 0; y < ph; y++) {
        const size_t srow = (size_t)ymap[y] * src->stride;
        float *pr = pad + (size_t)y * pw;
        if (src->highbd) {
            const uint16_t *s = src->data;
            for (uint32_t x = 0; x < pw; x++)
                pr[x] = (float)s[srow + xmap[x]] * iscale;
        } else {
            const uint8_t *s = src->data;
            for (uint32_t x = 0; x < pw; x++)
                pr[x] = (float)s[srow + xmap[x]] * iscale;
        }
    }

    {
        const int32_t xmax = (int32_t)w - 1;
        const int32_t ymax = (int32_t)h - 1;
        float re[MCD_DFT_N2], im[MCD_DFT_N2];

        /* block grid: every multiple-of-4 start with the 16x16 block
         * intersecting the visible frame, from -(N-STEP) up to w-1/h-1 */
        for (int32_t by = -(MCD_DFT_N - MCD_DFT_STEP); by <= ymax; by += MCD_DFT_STEP) {
            for (int32_t bx = -(MCD_DFT_N - MCD_DFT_STEP); bx <= xmax; bx += MCD_DFT_STEP) {
                const float *pp = pad + (size_t)(by + MCD_DFT_PAD) * pw +
                                  (bx + MCD_DFT_PAD);
                const float *wp = ctx->win;
                for (uint32_t i = 0; i < MCD_DFT_N; i++, pp += pw)
                    for (uint32_t j = 0; j < MCD_DFT_N; j++)
                        re[i * MCD_DFT_N + j] = pp[j] * wp[i * MCD_DFT_N + j];
                memset(im, 0, sizeof(im));

                mcd_fft2d(ctx, re, im, 0);

                if (sigma_pct) {
                    /* zmean: remove the windowed local mean before the
                     * spectral shrinkage, restore it afterwards */
                    const float gf = re[0] / ctx->wfre[0];
                    for (uint32_t k = 0; k < MCD_DFT_N2; k++) {
                        re[k] -= gf * ctx->wfre[k];
                        im[k] -= gf * ctx->wfim[k];
                    }
                    /* type-0 Wiener shrinkage */
                    const float *s2 = ctx->sigma2;
                    for (uint32_t k = 0; k < MCD_DFT_N2; k++) {
                        const float psd = re[k] * re[k] + im[k] * im[k];
                        const float num = psd - s2[k];
                        const float mult = num > 0.0f ? num / (psd + 1e-15f)
                                                      : 0.0f;
                        re[k] *= mult;
                        im[k] *= mult;
                    }
                    for (uint32_t k = 0; k < MCD_DFT_N2; k++) {
                        re[k] += gf * ctx->wfre[k];
                        im[k] += gf * ctx->wfim[k];
                    }
                }

                /* inverse transform: with the 1/16^2 carried by win there
                 * is no extra normalization anywhere */
                mcd_fft2d(ctx, re, im, 1);

                /* overlap-add synthesis */
                for (uint32_t i = 0; i < MCD_DFT_N; i++) {
                    const int32_t oy = by + (int32_t)i;
                    if (oy < 0 || oy > ymax)
                        continue;
                    float       *ar = acc + (size_t)oy * w;
                    const float *wr = ctx->win + i * MCD_DFT_N;
                    const float *rr = re + i * MCD_DFT_N;
                    for (uint32_t j = 0; j < MCD_DFT_N; j++) {
                        const int32_t ox = bx + (int32_t)j;
                        if (ox >= 0 && ox <= xmax)
                            ar[ox] += rr[j] * wr[j];
                    }
                }
            }
        }
    }

    /* back to native scale, rounded and clamped */
    if (dst->highbd) {
        uint16_t *d = dst->data;
        for (uint32_t y = 0; y < h; y++) {
            uint16_t   *dr = d + (size_t)y * dst->stride;
            const float *ar = acc + (size_t)y * w;
            for (uint32_t x = 0; x < w; x++) {
                int32_t v = (int32_t)(ar[x] * oscale + 0.5f);
                if (v < 0)
                    v = 0;
                if (v > (int32_t)pmax)
                    v = (int32_t)pmax;
                dr[x] = (uint16_t)v;
            }
        }
    } else {
        uint8_t *d = dst->data;
        for (uint32_t y = 0; y < h; y++) {
            uint8_t    *dr = d + (size_t)y * dst->stride;
            const float *ar = acc + (size_t)y * w;
            for (uint32_t x = 0; x < w; x++) {
                int32_t v = (int32_t)(ar[x] * oscale + 0.5f);
                if (v < 0)
                    v = 0;
                if (v > (int32_t)pmax)
                    v = (int32_t)pmax;
                dr[x] = (uint8_t)v;
            }
        }
    }

cleanup:
    free(xmap);
    free(ymap);
    free(pad);
    free(acc);
    return ret;
}

int mcd_dfttest_plane(const McdPlane *src, McdPlane *dst, uint32_t sigma_pct,
                      uint32_t bitdepth)
{
    McdDftCtx *ctx = NULL;
    int rc = mcd_dfttest_ctx_create(sigma_pct, &ctx);
    if (rc != MCD_SUCCESS) {
        return rc;
    }
    rc = mcd_dfttest_plane_with_ctx(ctx, src, dst, bitdepth);
    mcd_dfttest_ctx_destroy(&ctx);
    return rc;
}

/* ---------------------------------------------------------------------- */
/* vsdenoise luma-masked merge of source and prefiltered planes            */

static inline float mcd_plane_getf(const McdPlane *p, uint32_t x, uint32_t y)
{
    return p->highbd
               ? (float)((const uint16_t *)p->data)[(size_t)y * p->stride + x]
               : (float)((const uint8_t *)p->data)[(size_t)y * p->stride + x];
}

/* m = clamp((Y8 - 16) / 59, 0, 1), Y8 = Y scaled to the 8-bit range */
static inline float mcd_luma_mask(const McdPlane *yp, uint32_t x, uint32_t y,
                                  float inv_pmax)
{
    const float y8 = mcd_plane_getf(yp, x, y) * (255.0f * inv_pmax);
    const float m  = (y8 - 16.0f) / 59.0f;
    return m < 0.0f ? 0.0f : (m > 1.0f ? 1.0f : m);
}

/* luma mask box-averaged over the chroma pixel's footprint */
static inline float mcd_mask_box(const McdPlane *yp, uint32_t cx, uint32_t cy,
                                 uint32_t x_ss, uint32_t y_ss, float inv_pmax)
{
    const uint32_t span_x = 1u << x_ss;
    const uint32_t span_y = 1u << y_ss;
    float sum = 0.0f;
    for (uint32_t dy = 0; dy < span_y; dy++) {
        uint32_t yy = cy * span_y + dy;
        if (yy >= yp->h)
            yy = yp->h - 1;
        for (uint32_t dx = 0; dx < span_x; dx++) {
            uint32_t xx = cx * span_x + dx;
            if (xx >= yp->w)
                xx = yp->w - 1;
            sum += mcd_luma_mask(yp, xx, yy, inv_pmax);
        }
    }
    return sum / (float)(span_x * span_y);
}

int mcd_prefilter_mask_merge(const McdPlane *src_planes,
                             const McdPlane *filt_planes,
                             McdPlane *out_planes, uint32_t n_planes,
                             uint32_t x_ss, uint32_t y_ss, uint32_t bitdepth)
{
    if (!src_planes || !filt_planes || !out_planes)
        return MCD_ERR_PARAM;
    if (n_planes != 1 && n_planes != 3)
        return MCD_ERR_PARAM;
    if (x_ss > 1 || y_ss > 1)
        return MCD_ERR_PARAM;
    if (bitdepth < 8 || bitdepth > 16)
        return MCD_ERR_PARAM;

    const uint32_t pmax     = (1u << bitdepth) - 1;
    const float    inv_pmax = 1.0f / (float)pmax;
    const bool     hb       = bitdepth != 8;

    for (uint32_t p = 0; p < n_planes; p++) {
        const McdPlane *s = &src_planes[p];
        const McdPlane *f = &filt_planes[p];
        McdPlane       *o = &out_planes[p];
        if (!s->data || !f->data || !o->data)
            return MCD_ERR_PARAM;
        if (s->highbd != hb || f->highbd != hb || o->highbd != hb)
            return MCD_ERR_PARAM;
        if (!s->w || !s->h || s->stride < s->w ||
            f->w != s->w || f->h != s->h || f->stride < f->w ||
            o->w != s->w || o->h != s->h || o->stride < o->w)
            return MCD_ERR_PARAM;
        /* Plane samples may be read and written in place. Luma is snapshotted
         * below when aliased because chroma masks read neighboring luma. */
    }

    /* Chroma masks read neighboring luma samples. If the luma output aliases
     * its source, the luma pass below would otherwise change those samples
     * before the chroma pass reads them. Snapshot luma for the whole merge. */
    McdPlane src_snapshot[MCD_MAX_PLANES];
    memcpy(src_snapshot, src_planes, n_planes * sizeof(*src_snapshot));
    void *luma_snapshot = NULL;
    if (out_planes[0].data == src_planes[0].data) {
        const McdPlane *s = &src_planes[0];
        const size_t elem_size = s->highbd ? sizeof(uint16_t) : sizeof(uint8_t);
        const size_t stride = s->w;
        if ((size_t)s->h > SIZE_MAX / stride / elem_size)
            return MCD_ERR_NOMEM;
        luma_snapshot = malloc(stride * s->h * elem_size);
        if (!luma_snapshot)
            return MCD_ERR_NOMEM;
        for (uint32_t y = 0; y < s->h; y++)
            memcpy((uint8_t *)luma_snapshot + (size_t)y * stride * elem_size,
                   (const uint8_t *)s->data + (size_t)y * s->stride * elem_size,
                   stride * elem_size);
        src_snapshot[0].data = luma_snapshot;
        src_snapshot[0].stride = (uint32_t)stride;
    }
    const McdPlane *merge_src = luma_snapshot ? src_snapshot : src_planes;
    const McdPlane *yl = &merge_src[0];

    /* SIMD paths (bit-identical lane-wise float ops, see mc_denoise_simd.c);
     * on any unsupported geometry they decline before writing anything and
     * the scalar loop below runs instead. */
    if (mcd_use_avx512() || mcd_use_avx2()) {
        if (mcd_simd_mask_merge(merge_src, filt_planes, out_planes, n_planes,
                                x_ss, y_ss, bitdepth) == MCD_SUCCESS) {
            free(luma_snapshot);
            return MCD_SUCCESS;
        }
    }

    for (uint32_t p = 0; p < n_planes; p++) {
        const McdPlane *s = &merge_src[p];
        const McdPlane *f = &filt_planes[p];
        McdPlane       *o = &out_planes[p];

        /* The luma snapshot keeps the chroma mask independent of in-place
         * luma writes; each plane's source and filtered samples are read
         * before its output sample is written. */
        for (uint32_t y = 0; y < s->h; y++) {
            for (uint32_t x = 0; x < s->w; x++) {
                const float m = (p == 0)
                                    ? mcd_luma_mask(yl, x, y, inv_pmax)
                                    : mcd_mask_box(yl, x, y, x_ss, y_ss,
                                                   inv_pmax);
                const float sv = mcd_plane_getf(s, x, y);
                const float fv = mcd_plane_getf(f, x, y);
                const float v  = sv + m * (fv - sv);
                uint32_t iv = (v <= 0.0f) ? 0 : (uint32_t)(v + 0.5f);
                if (iv > pmax)
                    iv = pmax;
                if (o->highbd)
                    ((uint16_t *)o->data)[(size_t)y * o->stride + x] = (uint16_t)iv;
                else
                    ((uint8_t *)o->data)[(size_t)y * o->stride + x] = (uint8_t)iv;
            }
        }
    }
    free(luma_snapshot);
    return MCD_SUCCESS;
}
