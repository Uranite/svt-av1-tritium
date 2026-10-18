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
 *  - McdPlane carries no bit depth, so mcd_dfttest_plane derives the 8-bit
 *    internal scale factor 2^(bd-8) from src->highbd, assuming bd = 10 for
 *    highbd planes (AV1 main-10).  A 12-bit source is therefore filtered
 *    with the sigma profile scaled as 10-bit (slightly weak, never
 *    oversmoothing); fixing that needs a bitdepth parameter in the contract.
 *  - Frame borders are padded by symmetric mirroring (edge pixel
 *    duplicated), which makes every 4-stepped block position computable.
 */

#include "mc_denoise.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define MCD_DFT_N    16                     /* transform block size        */
#define MCD_DFT_STEP  4                     /* block stepping (75% overlap)*/
#define MCD_DFT_PAD  16                     /* mirror padding per side     */
#define MCD_DFT_N2   (MCD_DFT_N * MCD_DFT_N)

#define MCD_PI 3.14159265358979323846

/* ---------------------------------------------------------------------- */
/* per-call context: all expensive tables precomputed exactly once         */

typedef struct McdDftCtx {
    float   twr[8], twi[8];        /* fwd twiddles e^{-2*pi*i*k/16}        */
    uint8_t brev[MCD_DFT_N];       /* bit-reversal permutation             */
    float   win[MCD_DFT_N2];       /* analysis = synthesis window          */
    float   wscale;                /* sum of win^2 over the block          */
    float   wfre[MCD_DFT_N2];      /* DFT(win): zmean profile, real part   */
    float   wfim[MCD_DFT_N2];      /* DFT(win): imaginary part             */
    float   sigma2[MCD_DFT_N2];    /* per-coefficient noise variance       */
} McdDftCtx;

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

int mcd_dfttest_plane(const McdPlane *src, McdPlane *dst, uint32_t sigma_pct)
{
    if (!src || !dst || !src->data)
        return MCD_ERR_PARAM;
    const uint32_t w = src->w;
    const uint32_t h = src->h;
    if (w == 0 || h == 0 || src->stride < w)
        return MCD_ERR_PARAM;

    /* effective bit depth: McdPlane has no depth field, assume 10-bit for
     * highbd planes (see file header note) */
    const uint32_t bd     = src->highbd ? 10 : 8;
    const uint32_t pmax   = (1u << bd) - 1;
    const float    iscale = 1.0f / (float)(1u << (bd - 8));
    const float    oscale = (float)(1u << (bd - 8));

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

    McdDftCtx ctx;
    mcd_dft_tables(&ctx, sigma_pct);

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
                const float *wp = ctx.win;
                for (uint32_t i = 0; i < MCD_DFT_N; i++, pp += pw)
                    for (uint32_t j = 0; j < MCD_DFT_N; j++)
                        re[i * MCD_DFT_N + j] = pp[j] * wp[i * MCD_DFT_N + j];
                memset(im, 0, sizeof(im));

                mcd_fft2d(&ctx, re, im, 0);

                if (sigma_pct) {
                    /* zmean: remove the windowed local mean before the
                     * spectral shrinkage, restore it afterwards */
                    const float gf = re[0] / ctx.wfre[0];
                    for (uint32_t k = 0; k < MCD_DFT_N2; k++) {
                        re[k] -= gf * ctx.wfre[k];
                        im[k] -= gf * ctx.wfim[k];
                    }
                    /* type-0 Wiener shrinkage */
                    const float *s2 = ctx.sigma2;
                    for (uint32_t k = 0; k < MCD_DFT_N2; k++) {
                        const float psd = re[k] * re[k] + im[k] * im[k];
                        const float num = psd - s2[k];
                        const float mult = num > 0.0f ? num / (psd + 1e-15f)
                                                      : 0.0f;
                        re[k] *= mult;
                        im[k] *= mult;
                    }
                    for (uint32_t k = 0; k < MCD_DFT_N2; k++) {
                        re[k] += gf * ctx.wfre[k];
                        im[k] += gf * ctx.wfim[k];
                    }
                }

                /* inverse transform: with the 1/16^2 carried by win there
                 * is no extra normalization anywhere */
                mcd_fft2d(&ctx, re, im, 1);

                /* overlap-add synthesis */
                for (uint32_t i = 0; i < MCD_DFT_N; i++) {
                    const int32_t oy = by + (int32_t)i;
                    if (oy < 0 || oy > ymax)
                        continue;
                    float       *ar = acc + (size_t)oy * w;
                    const float *wr = ctx.win + i * MCD_DFT_N;
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
        if (f->data == o->data)        /* out may alias src, never filt */
            return MCD_ERR_PARAM;
    }

    const McdPlane *yl = &src_planes[0];

    for (uint32_t p = 0; p < n_planes; p++) {
        const McdPlane *s = &src_planes[p];
        const McdPlane *f = &filt_planes[p];
        McdPlane       *o = &out_planes[p];

        /* strictly pixel-by-pixel: every read of a plane happens before the
         * write to the same coordinate, so out == src aliasing is safe */
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
    return MCD_SUCCESS;
}
