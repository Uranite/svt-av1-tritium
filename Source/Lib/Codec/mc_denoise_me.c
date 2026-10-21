/*
 * mc_denoise_me.c — hierarchical EPZ motion analysis + recalculate pass
 * for the motion-compensated temporal denoiser (contract: mc_denoise.h).
 *
 * Clean-room notice: this module reimplements the well-known EPZ-style
 * (enhanced predictive zonal) hierarchical block-matching scheme used by
 * mvtools-class motion plugins, from published algorithm descriptions only:
 *   - coarse-to-fine search over a bilinear-reduction image pyramid,
 *   - small exhaustive integer-pel search at the coarsest level,
 *   - at finer levels a predictor set {parent-scaled, zero, left, top,
 *     top-right, median} evaluated before 8-point hexagon refinement
 *     followed by a full +-1 ring,
 *   - lambda-regularized SAD cost (SATD luma at the finest level when
 *     requested) and half-pel refinement on interpolated reference phases.
 * No GPL-licensed code was consulted or copied.
 *
 * Internal scratch is plain malloc/free, allocated once per call; there are
 * no per-block allocations. All failures return MCD_ERR_NOMEM/MCD_ERR_PARAM.
 */
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "mc_denoise.h"
#include "mc_denoise_simd.h"

/* Side padding, in samples, that every pyramid plane and pelset is guaranteed
 * to carry (the glue builds pyramids with pad >= 32 + blksize). Candidate
 * reference blocks are clamped so they stay inside the visible plane extended
 * by this margin, which is always within the padded region. */
#define MCD_ME_EDGE_PAD 32

/* -------------------------------------------------------------------------- */
/* small helpers                                                              */

static inline int32_t mcd_me_clamp(int32_t v, int32_t lo, int32_t hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static inline int32_t mcd_me_imax(int32_t a, int32_t b) { return a > b ? a : b; }

static inline int32_t mcd_me_imin(int32_t a, int32_t b) { return a < b ? a : b; }

/* Floor division by two; vectors may be negative (half-pel units). */
static inline int32_t mcd_me_half(int32_t v) { return (v - (v & 1)) >> 1; }

/* Number of grid blocks covering a visible dimension: n = ceil((dim - ov)/step),
 * at least 1 so degenerate small frames still get one block. */
static uint32_t mcd_me_grid(uint32_t dim, uint32_t ov, uint32_t step) {
    const uint64_t d = (dim > ov) ? (uint64_t)(dim - ov) : 0ull;
    if (d == 0)
        return 1;
    return (uint32_t)((d + step - 1) / step);
}

/* -------------------------------------------------------------------------- */
/* SAD kernels (strides in samples, native depth, uint64 accumulation)         */

/* Process-lifetime cached dispatch flags. CPU features never change at
 * runtime, so a benign idempotent race (same pattern as mcd_simd_disabled)
 * is safe; this keeps the per-SAD-candidate dispatch cost down to two
 * branch-on-global tests instead of __builtin_cpu_supports calls inside the
 * candidate search loops. */
static int g_me_avx2 = -1;
static int g_me_avx512 = -1;

static int mcd_me_use_avx2(void) {
    int v = g_me_avx2;
    if (v < 0) {
        v = mcd_use_avx2();
        g_me_avx2 = v;
    }
    return v;
}

static int mcd_me_use_avx512(void) {
    int v = g_me_avx512;
    if (v < 0) {
        v = mcd_use_avx512();
        g_me_avx512 = v;
    }
    return v;
}

static uint64_t mcd_sad_scalar_u8u8(
    const uint8_t *a, int32_t sa, const uint8_t *b, int32_t sb, uint32_t w, uint32_t h, uint64_t bound) {
    uint64_t sum = 0;
    for (uint32_t y = 0; y < h; y++) {
        for (uint32_t x = 0; x < w; x++) {
            const int d = (int)a[x] - (int)b[x];
            sum += (uint64_t)(d < 0 ? -d : d);
        }
        if (bound && sum >= bound)
            return sum; /* partial-row abort: bound semantics above */
        a += sa;
        b += sb;
    }
    return sum;
}

static uint64_t mcd_sad_u8u8(
    const uint8_t *a, int32_t sa, const uint8_t *b, int32_t sb, uint32_t w, uint32_t h, uint64_t bound) {
    if (mcd_me_use_avx2())
        return mcd_simd_sad_u8u8(a, sa, b, sb, w, h, bound);
    return mcd_sad_scalar_u8u8(a, sa, b, sb, w, h, bound);
}

static uint64_t mcd_sad_scalar_u16u16(
    const uint16_t *a, int32_t sa, const uint16_t *b, int32_t sb, uint32_t w, uint32_t h, uint64_t bound) {
    uint64_t sum = 0;
    for (uint32_t y = 0; y < h; y++) {
        for (uint32_t x = 0; x < w; x++) {
            const int d = (int)a[x] - (int)b[x];
            sum += (uint64_t)(d < 0 ? -d : d);
        }
        if (bound && sum >= bound)
            return sum; /* partial-row abort: bound semantics above */
        a += sa;
        b += sb;
    }
    return sum;
}

static uint64_t mcd_sad_u16u16(
    const uint16_t *a, int32_t sa, const uint16_t *b, int32_t sb, uint32_t w, uint32_t h, uint64_t bound) {
    if (mcd_me_use_avx2() && (uint64_t)w * h * 65535ull < 0x7fffffffull)
        return mcd_simd_sad_u16u16(a, sa, b, sb, w, h, bound);
    return mcd_sad_scalar_u16u16(a, sa, b, sb, w, h, bound);
}

static uint64_t mcd_sad_u8u16(
    const uint8_t *a, int32_t sa, const uint16_t *b, int32_t sb, uint32_t w, uint32_t h, uint64_t bound) {
    if (mcd_me_use_avx2() && (uint64_t)w * h * 65535ull < 0x7fffffffull)
        return mcd_simd_sad_u8u16(a, sa, b, sb, w, h, bound);
    uint64_t sum = 0;
    for (uint32_t y = 0; y < h; y++) {
        for (uint32_t x = 0; x < w; x++) {
            const int d = (int)a[x] - (int)b[x];
            sum += (uint64_t)(d < 0 ? -d : d);
        }
        if (bound && sum >= bound)
            return sum; /* partial-row abort: bound semantics above */
        a += sa;
        b += sb;
    }
    return sum;
}

static uint64_t mcd_sad_u16u8(
    const uint16_t *a, int32_t sa, const uint8_t *b, int32_t sb, uint32_t w, uint32_t h, uint64_t bound) {
    if (mcd_me_use_avx2() && (uint64_t)w * h * 65535ull < 0x7fffffffull)
        return mcd_simd_sad_u16u8(a, sa, b, sb, w, h, bound);
    uint64_t sum = 0;
    for (uint32_t y = 0; y < h; y++) {
        for (uint32_t x = 0; x < w; x++) {
            const int d = (int)a[x] - (int)b[x];
            sum += (uint64_t)(d < 0 ? -d : d);
        }
        if (bound && sum >= bound)
            return sum; /* partial-row abort: bound semantics above */
        a += sa;
        b += sb;
    }
    return sum;
}

/* Block SAD between two planes; (ax,ay)/(bx,by) are signed sample coordinates
 * into the padded plane buffers (negative = inside the replicated margin). */
static uint64_t mcd_plane_sad(const McdPlane *pa, int32_t ax, int32_t ay,
                              const McdPlane *pb, int32_t bx, int32_t by,
                              int32_t w, int32_t h, uint64_t bound) {
    const int32_t sa = (int32_t)pa->stride;
    const int32_t sb = (int32_t)pb->stride;

    if (pa->highbd && pb->highbd) {
        const uint16_t *a = (const uint16_t *)pa->data + (ptrdiff_t)ay * sa + ax;
        const uint16_t *b = (const uint16_t *)pb->data + (ptrdiff_t)by * sb + bx;
        return mcd_sad_u16u16(a, sa, b, sb, (uint32_t)w, (uint32_t)h, bound);
    }
    if (!pa->highbd && !pb->highbd) {
        const uint8_t *a = (const uint8_t *)pa->data + (ptrdiff_t)ay * sa + ax;
        const uint8_t *b = (const uint8_t *)pb->data + (ptrdiff_t)by * sb + bx;
        return mcd_sad_u8u8(a, sa, b, sb, (uint32_t)w, (uint32_t)h, bound);
    }
    if (!pa->highbd) {
        const uint8_t *a = (const uint8_t *)pa->data + (ptrdiff_t)ay * sa + ax;
        const uint16_t *b = (const uint16_t *)pb->data + (ptrdiff_t)by * sb + bx;
        return mcd_sad_u8u16(a, sa, b, sb, (uint32_t)w, (uint32_t)h, bound);
    }
    {
        const uint16_t *a = (const uint16_t *)pa->data + (ptrdiff_t)ay * sa + ax;
        const uint8_t *b = (const uint8_t *)pb->data + (ptrdiff_t)by * sb + bx;
        return mcd_sad_u16u8(a, sa, b, sb, (uint32_t)w, (uint32_t)h, bound);
    }
}

/* -------------------------------------------------------------------------- */
/* SATD: sum over 4x4 sub-blocks of the absolute sums of a (butterfly) 2D      */
/* Hadamard transform of the prediction error. Unnormalized, so a uniform     */
/* error of d over a 4x4 block yields 16*d, matching the SAD scale.           */

static inline int32_t mcd_satd_4x4(const int32_t d[16]) {
    int32_t t[16];
    int32_t sum = 0;

    for (int r = 0; r < 4; r++) {
        const int32_t *x = d + 4 * r;
        const int32_t a0 = x[0] + x[3];
        const int32_t a1 = x[1] + x[2];
        const int32_t a2 = x[1] - x[2];
        const int32_t a3 = x[0] - x[3];
        t[4 * r + 0] = a0 + a1;
        t[4 * r + 1] = a2 + a3;
        t[4 * r + 2] = a0 - a1;
        t[4 * r + 3] = a3 - a2;
    }
    for (int c = 0; c < 4; c++) {
        const int32_t a0 = t[c] + t[12 + c];
        const int32_t a1 = t[4 + c] + t[8 + c];
        const int32_t a2 = t[4 + c] - t[8 + c];
        const int32_t a3 = t[c] - t[12 + c];
        const int32_t s0 = a0 + a1;
        const int32_t s1 = a2 + a3;
        const int32_t s2 = a0 - a1;
        const int32_t s3 = a3 - a2;
        sum += (s0 < 0 ? -s0 : s0) + (s1 < 0 ? -s1 : s1) +
               (s2 < 0 ? -s2 : s2) + (s3 < 0 ? -s3 : s3);
    }
    return sum;
}

static uint64_t mcd_satd_scalar_u8u8(
    const uint8_t *a, int32_t sa, const uint8_t *b, int32_t sb, uint32_t w, uint32_t h,
    uint64_t bound) {
    uint64_t sum = 0;
    for (uint32_t y = 0; y < h; y += 4) {
        for (uint32_t x = 0; x < w; x += 4) {
            int32_t d[16];
            for (int r = 0; r < 4; r++)
                for (int c = 0; c < 4; c++)
                    d[4 * r + c] = (int)a[(y + r) * (uint32_t)sa + x + c] -
                                   (int)b[(y + r) * (uint32_t)sb + x + c];
            sum += (uint64_t)mcd_satd_4x4(d);
            if (bound && sum >= bound)
                return sum; /* partial-sub-block abort: bound semantics above */
        }
    }
    return sum;
}

static uint64_t mcd_satd_u8u8(
    const uint8_t *a, int32_t sa, const uint8_t *b, int32_t sb, uint32_t w, uint32_t h,
    uint64_t bound) {
    if (mcd_me_use_avx512() && w >= 16 && h >= 16 && (w & 15) == 0 && (h & 15) == 0)
        return mcd_simd_satd_u8u8(a, sa, b, sb, w, h, bound);
    return mcd_satd_scalar_u8u8(a, sa, b, sb, w, h, bound);
}

static uint64_t mcd_satd_scalar_u16u16(
    const uint16_t *a, int32_t sa, const uint16_t *b, int32_t sb, uint32_t w, uint32_t h,
    uint64_t bound) {
    uint64_t sum = 0;
    for (uint32_t y = 0; y < h; y += 4) {
        for (uint32_t x = 0; x < w; x += 4) {
            int32_t d[16];
            for (int r = 0; r < 4; r++)
                for (int c = 0; c < 4; c++)
                    d[4 * r + c] = (int)a[(y + r) * (uint32_t)sa + x + c] -
                                   (int)b[(y + r) * (uint32_t)sb + x + c];
            sum += (uint64_t)mcd_satd_4x4(d);
            if (bound && sum >= bound)
                return sum; /* partial-sub-block abort: bound semantics above */
        }
    }
    return sum;
}

static uint64_t mcd_satd_u16u16(
    const uint16_t *a, int32_t sa, const uint16_t *b, int32_t sb, uint32_t w, uint32_t h,
    uint64_t bound) {
    if (mcd_me_use_avx512() && w >= 16 && h >= 16 && (w & 15) == 0 && (h & 15) == 0)
        return mcd_simd_satd_u16u16(a, sa, b, sb, w, h, bound);
    return mcd_satd_scalar_u16u16(a, sa, b, sb, w, h, bound);
}

/* SATD between two native planes; falls back to SAD for (impossible in
 * practice) mixed-depth pairs. */
static uint64_t mcd_plane_satd(const McdPlane *pa, int32_t ax, int32_t ay,
                               const McdPlane *pb, int32_t bx, int32_t by,
                               int32_t w, int32_t h, uint64_t bound) {
    const int32_t sa = (int32_t)pa->stride;
    const int32_t sb = (int32_t)pb->stride;

    if (pa->highbd != pb->highbd)
        return mcd_plane_sad(pa, ax, ay, pb, bx, by, w, h, bound);
    if (pa->highbd) {
        const uint16_t *a = (const uint16_t *)pa->data + (ptrdiff_t)ay * sa + ax;
        const uint16_t *b = (const uint16_t *)pb->data + (ptrdiff_t)by * sb + bx;
        return mcd_satd_u16u16(a, sa, b, sb, (uint32_t)w, (uint32_t)h, bound);
    }
    {
        const uint8_t *a = (const uint8_t *)pa->data + (ptrdiff_t)ay * sa + ax;
        const uint8_t *b = (const uint8_t *)pb->data + (ptrdiff_t)by * sb + bx;
        return mcd_satd_u8u8(a, sa, b, sb, (uint32_t)w, (uint32_t)h, bound);
    }
}

/* -------------------------------------------------------------------------- */
/* cost pieces shared by analyze / recalculate                                 */

/* Luma SAD of the current block at (X,Y) against the reference phase plane
 * at half-pel position (2*X + hx, 2*Y + hy), computed DIRECTLY on the pelset
 * phase plane: mcd_pelset_locate applies the same coordinate clamp and phase
 * selection the fetch would, so the summed samples are exactly the ones
 * mcd_pelset_fetch_pad would have staged into a scratch block (u8 values are
 * summed as u8 vs the identical u16-widened values) — no staging copy. */
static uint64_t mcd_hp_luma_sad(const McdPlane *cur0, const McdPelSet *pel,
                                int32_t X, int32_t Y, int32_t b,
                                int32_t hx, int32_t hy, uint32_t pel_pad,
                                uint64_t bound) {
    const void *row0;
    int32_t     st;
    mcd_pelset_locate(pel, 0, 2 * X + hx, 2 * Y + hy, (uint32_t)b, (uint32_t)b,
                      pel_pad, &row0, &st);
    {
        const int32_t ct = (int32_t)cur0->stride;
        if (cur0->highbd) {
            const uint16_t *a = (const uint16_t *)cur0->data + (ptrdiff_t)Y * ct + X;
            return mcd_sad_u16u16(a, ct, (const uint16_t *)row0, st, (uint32_t)b, (uint32_t)b, bound);
        }
        {
            const uint8_t *a = (const uint8_t *)cur0->data + (ptrdiff_t)Y * ct + X;
            return mcd_sad_u8u8(a, ct, (const uint8_t *)row0, st, (uint32_t)b, (uint32_t)b, bound);
        }
    }
}

/* Chroma SADs summed over planes 1..n-1. The reference offset (offx,offy) is
 * in integer LUMA samples (an integer candidate, or floor(halfpel/2)); chroma
 * positions derive from it with the subsampling shifts. Reference blocks are
 * clamped into the padded region independently of luma. */
/* bound: early-abort bound on the chroma SUM (0 = compute fully); once the
 * accumulated sum reaches bound, return early with a value >= bound (the
 * callers only compare the TOTAL against the same bound, so the outcome is
 * unchanged). */
static uint64_t mcd_chroma_sad(const McdPlane *const *curp, const McdPlane *const *refp,
                               uint32_t n_planes, uint32_t x_ss, uint32_t y_ss,
                               int32_t X, int32_t Y, int32_t bblk,
                               int32_t offx, int32_t offy, uint64_t bound) {
    uint64_t sum = 0;
    for (uint32_t p = 1; p < n_planes; p++) {
        const McdPlane *a = curp[p];
        const McdPlane *b = refp[p];
        const int32_t bw = mcd_me_imax(1, bblk >> x_ss);
        const int32_t bh = mcd_me_imax(1, bblk >> y_ss);
        int32_t rx = (X + offx) >> x_ss;
        int32_t ry = (Y + offy) >> y_ss;
        rx = mcd_me_clamp(rx, -MCD_ME_EDGE_PAD, (int32_t)b->w + MCD_ME_EDGE_PAD - bw);
        ry = mcd_me_clamp(ry, -MCD_ME_EDGE_PAD, (int32_t)b->h + MCD_ME_EDGE_PAD - bh);
        sum += mcd_plane_sad(a, X >> x_ss, Y >> y_ss, b, rx, ry, bw, bh,
                             bound > sum ? bound - sum : 0);
        if (bound && sum >= bound)
            return sum;
    }
    return sum;
}

/* Per-level, per-block search context. */
typedef struct McdMeLevel {
    const McdPlane *curp[3];         /* current-frame pyramid planes          */
    const McdPlane *refp[3];         /* reference-frame pyramid planes        */
    uint32_t n_planes;
    uint32_t x_ss, y_ss;
    int32_t  bblk;                   /* luma block size at this level         */
    int32_t  X, Y;                   /* block origin at this level            */
    int32_t  px, py;                 /* primary predictor (this level's units)*/
    uint64_t lambda;                 /* motion cost regularizer               */
    int      use_satd;               /* SATD luma instead of SAD (level 0)    */
} McdMeLevel;

/* Raw block cost (no motion penalty). luma_satd selects the SATD luma variant
 * when the level requests it; pass 0 to force plain SAD (used to obtain the
 * true sad_raw of a vector found under SATD).
 *
 * bound: EARLY-ABORT bound on the raw cost (0 = compute fully). The kernels
 * return as soon as the accumulated cost reaches bound, with a value that is
 * both >= bound and a lower bound of the full raw cost. Exactness: a candidate
 * whose raw cost >= bound has full cost raw + lambda_term >= bound +
 * lambda_term, i.e. it cannot beat an incumbent of `bound + lambda_term`;
 * callers pass bound = incumbent - lambda_term (0 when that underflows), so
 * aborted and full evaluation produce the SAME comparison outcome (both
 * lose; ties lose under the strict `<`). Only the winning candidate's exact
 * value is ever consumed (it is never aborted, since its partial sums stay
 * below the bound). */
static uint64_t mcd_me_sad_raw(const McdMeLevel *lv, int32_t cx, int32_t cy, int luma_satd,
                               uint64_t bound) {
    const McdPlane *a = lv->curp[0];
    const McdPlane *b = lv->refp[0];
    int32_t rx = mcd_me_clamp(lv->X + cx, -MCD_ME_EDGE_PAD,
                              (int32_t)b->w + MCD_ME_EDGE_PAD - lv->bblk);
    int32_t ry = mcd_me_clamp(lv->Y + cy, -MCD_ME_EDGE_PAD,
                              (int32_t)b->h + MCD_ME_EDGE_PAD - lv->bblk);
    uint64_t cost;

    if (lv->use_satd && luma_satd)
        cost = mcd_plane_satd(a, lv->X, lv->Y, b, rx, ry, lv->bblk, lv->bblk, bound);
    else
        cost = mcd_plane_sad(a, lv->X, lv->Y, b, rx, ry, lv->bblk, lv->bblk, bound);

    if (bound && cost >= bound)
        return cost; /* luma alone already past the bound: candidate loses */
    cost += mcd_chroma_sad(lv->curp, lv->refp, lv->n_planes, lv->x_ss, lv->y_ss,
                           lv->X, lv->Y, lv->bblk, cx, cy,
                           bound > cost ? bound - cost : 0);
    return cost;
}

static inline uint64_t mcd_me_lambda_term(const McdMeLevel *lv, int32_t cx, int32_t cy) {
    const int64_t dx = (int64_t)cx - lv->px;
    const int64_t dy = (int64_t)cy - lv->py;
    return (lv->lambda * (uint64_t)(dx * dx + dy * dy)) >> 8;
}

/* Full candidate cost: raw cost + (lambda * dist^2 to primary predictor)>>8.
 * incumbent (0 = none): the raw-cost kernels may abort early as soon as the
 * candidate provably cannot beat it (see mcd_me_sad_raw). */
static uint64_t mcd_me_cost_bounded(const McdMeLevel *lv, int32_t cx, int32_t cy,
                                    uint64_t incumbent) {
    const uint64_t lterm = mcd_me_lambda_term(lv, cx, cy);
    const uint64_t bound = incumbent > lterm ? incumbent - lterm : 0;
    uint64_t cost = mcd_me_sad_raw(lv, cx, cy, 1, bound);
    cost += lterm;
    return cost;
}

static uint64_t mcd_me_cost(const McdMeLevel *lv, int32_t cx, int32_t cy) {
    return mcd_me_cost_bounded(lv, cx, cy, 0);
}

static inline void mcd_me_try(const McdMeLevel *lv, int32_t cx, int32_t cy,
                              int32_t *bx, int32_t *by, uint64_t *bc) {
    /* *bc is a complete incumbent cost (never 0 here): candidates that
     * cannot beat it may abort early — the loser set is unchanged. */
    const uint64_t c = mcd_me_cost_bounded(lv, cx, cy, *bc);
    if (c < *bc) {
        *bc = c;
        *bx = cx;
        *by = cy;
    }
}

/* -------------------------------------------------------------------------- */
/* reusable scratch (predictor fields)                                         */

struct McdMeScratch {
    int32_t *fld[2];      /* grow-to-fit predictor fields                     */
    uint32_t cap;         /* capacity in int32 elements per field             */
};

/* -------------------------------------------------------------------------- */
/* level-0 fused window evaluator                                              */

/* The finest-level integer search (predictor set + hexagon + +-1 ring) used
 * to evaluate every candidate through the full per-candidate chain
 * (mcd_me_cost -> mcd_me_sad_raw -> mcd_plane_sad/satd + mcd_chroma_sad),
 * re-deriving clamps, depth dispatch and plane pointers for every candidate.
 * The fused evaluator resolves all of that ONCE per block into a small
 * window state (plane aliases, strides, clamp bounds, selected kernels) and
 * then scores every candidate with a single function over that state.
 *
 * Design note -- deliberate NO window staging: the planned variant copied
 * the current block and a predictor-centered reference window into scratch
 * once per block. On 1080p content that measured as a net REGRESSION
 * (+4..7% ME stage time across staging radii 8..24): the per-candidate
 * chain overhead such a copy removes is already far below 1 ns/eval (the
 * chain inlines into the search loop and the SIMD kernels dominate), while
 * the copy itself costs ~90 ns/block in scattered reference reads.
 * Aliasing the planes keeps every property that does pay: one fused 3-plane
 * cost function, no per-candidate clamp/dispatch layers, and kernels
 * selected once per block instead of per candidate.
 *
 * Exactness contract (bit-identical chosen vectors):
 *   - candidates keep their unclamped (cx, cy) for the lambda term and the
 *     exact mcd_me_clamp results for the reference position;
 *   - the window aliases the padded plane with its origin at sample
 *     (-MCD_ME_EDGE_PAD, -MCD_ME_EDGE_PAD), so its bounds cover the FULL
 *     clamp range and every candidate sums the identical samples;
 *   - the kernel flags replicate the wrappers' dispatch conditions exactly
 *     (mcd_use_avx2/avx512 honor the MCD_NO_SIMD hook), so each plane cost
 *     comes from the same kernel body the direct chain would pick;
 *   - bound (early-abort) bookkeeping is replicated line for line, so
 *     fused and direct values agree for winners and aborted losers alike.
 * mcd_me_level_search below mirrors the reference candidate order and the
 * strict `<` tie-breaking exactly; it is pure code motion of the sequence
 * previously inlined in mcd_me_analyze_s. */

/* Per-block fused-evaluator state (a stack local of plain aliases into the
 * caller's planes -- no shared mutable state, nothing to clean up). */
typedef struct McdMeL0Win {
    int      staged;                    /* 0 = direct path for every candidate */
    int      highbd;
    uint32_t n_planes;
    uint32_t ssx, ssy;
    int32_t  bblk, bw, bh;
    int32_t  X, Y;
    int32_t  lw, lh;                    /* ref luma dims   */
    int32_t  cw, ch;                    /* ref chroma dims */
    uint8_t *cur[3];                    /* current-block aliases (u8 or u16 samples) */
    uint8_t *win[3];                    /* reference window aliases                  */
    int32_t  cur_str[3], win_str[3];    /* strides in samples                        */
    int32_t  wx[3], wy[3];              /* window origins (ref sample coordinates)   */
    int32_t  ww[3], wh[3];              /* window extents                            */
    int      sad_simd, satd_simd;       /* luma kernel selection                     */
    int      csad_simd;                 /* chroma kernel selection                   */
} McdMeL0Win;

/* Resolve, once per block, everything the per-candidate cost needs. Sets
 * wn->staged = 0 for geometry the fused path does not handle (the search
 * then runs the direct path, which is bit-identical). */
static void mcd_me_l0_setup(const McdMeLevel *lv, McdMeL0Win *wn) {
    memset(wn, 0, sizeof(*wn));
    const uint32_t npl = lv->n_planes;
    if (npl != 1 && npl != 3)
        return;
    const McdPlane *ref0 = lv->refp[0];
    const McdPlane *cur0 = lv->curp[0];
    const int32_t   b    = lv->bblk;
    if (cur0->highbd != ref0->highbd)
        return; /* mixed depths: direct path */

    wn->staged   = 1;
    wn->highbd   = ref0->highbd ? 1 : 0;
    wn->n_planes = npl;
    wn->ssx      = lv->x_ss;
    wn->ssy      = lv->y_ss;
    wn->bblk     = b;
    wn->bw       = mcd_me_imax(1, b >> (int32_t)lv->x_ss);
    wn->bh       = mcd_me_imax(1, b >> (int32_t)lv->y_ss);
    wn->X        = lv->X;
    wn->Y        = lv->Y;
    wn->lw       = (int32_t)ref0->w;
    wn->lh       = (int32_t)ref0->h;
    if (npl == 3) {
        const McdPlane *r1 = lv->refp[1];
        const McdPlane *r2 = lv->refp[2];
        if (r1->highbd != ref0->highbd || r2->highbd != ref0->highbd ||
            r1->w != r2->w || r1->h != r2->h) {
            wn->staged = 0;
            return;
        }
        wn->cw = (int32_t)r1->w;
        wn->ch = (int32_t)r1->h;
    }

    {
        const int32_t   P  = MCD_ME_EDGE_PAD;
        const ptrdiff_t es = wn->highbd ? 2 : 1;
        wn->cur[0]     = (uint8_t *)cur0->data +
                     ((ptrdiff_t)lv->Y * (ptrdiff_t)cur0->stride + lv->X) * es;
        wn->cur_str[0] = (int32_t)cur0->stride;
        wn->win[0]     = (uint8_t *)ref0->data -
                     ((ptrdiff_t)P * (ptrdiff_t)ref0->stride + P) * es;
        wn->win_str[0] = (int32_t)ref0->stride;
        wn->wx[0]      = -P;
        wn->wy[0]      = -P;
        wn->ww[0]      = wn->lw + 2 * P;
        wn->wh[0]      = wn->lh + 2 * P;
        if (npl == 3) {
            const int32_t   cx = lv->X >> (int32_t)lv->x_ss;
            const int32_t   cy = lv->Y >> (int32_t)lv->y_ss;
            const McdPlane *c1 = lv->curp[1];
            const McdPlane *c2 = lv->curp[2];
            wn->cur[1]     = (uint8_t *)c1->data + ((ptrdiff_t)cy * (ptrdiff_t)c1->stride + cx) * es;
            wn->cur_str[1] = (int32_t)c1->stride;
            wn->cur[2]     = (uint8_t *)c2->data + ((ptrdiff_t)cy * (ptrdiff_t)c2->stride + cx) * es;
            wn->cur_str[2] = (int32_t)c2->stride;
            for (uint32_t p = 1; p < 3; p++) {
                const McdPlane *rp = lv->refp[p];
                wn->win[p]     = (uint8_t *)rp->data -
                             ((ptrdiff_t)P * (ptrdiff_t)rp->stride + P) * es;
                wn->win_str[p] = (int32_t)rp->stride;
                wn->wx[p]      = -P;
                wn->wy[p]      = -P;
                wn->ww[p]      = wn->cw + 2 * P;
                wn->wh[p]      = wn->ch + 2 * P;
            }
        }
    }

    /* kernel selection: replicates the dispatch conditions of mcd_sad_* /
     * mcd_satd_* exactly (w == h == bblk for luma; bw x bh for chroma) */
    if (wn->highbd) {
        const uint64_t lum = (uint64_t)b * (uint64_t)b * 65535ull;
        const uint64_t chr = (uint64_t)wn->bw * (uint64_t)wn->bh * 65535ull;
        wn->sad_simd  = (mcd_me_use_avx2() && lum < 0x7fffffffull) ? 1 : 0;
        wn->csad_simd = (mcd_me_use_avx2() && chr < 0x7fffffffull) ? 1 : 0;
    } else {
        wn->sad_simd  = mcd_me_use_avx2() ? 1 : 0;
        wn->csad_simd = wn->sad_simd;
    }
    wn->satd_simd = (mcd_me_use_avx512() && b >= 16 && (b & 15) == 0) ? 1 : 0;
}

/* Fused twin of mcd_me_sad_raw (identical bound semantics, see there) over
 * the resolved window state. The per-plane window test falls back to the
 * exact direct call for anything outside the window (unreachable with the
 * full-plane alias, kept as the safety net). */
static uint64_t mcd_l0_raw_u8(const McdMeLevel *lv, const McdMeL0Win *wn,
                              int32_t cx, int32_t cy, uint64_t bound) {
    const int32_t b  = wn->bblk;
    const int32_t rx = mcd_me_clamp(wn->X + cx, -MCD_ME_EDGE_PAD,
                                    wn->lw + MCD_ME_EDGE_PAD - b);
    const int32_t ry = mcd_me_clamp(wn->Y + cy, -MCD_ME_EDGE_PAD,
                                    wn->lh + MCD_ME_EDGE_PAD - b);
    uint64_t cost;

    {
        const int32_t ox = rx - wn->wx[0];
        const int32_t oy = ry - wn->wy[0];
        if (ox >= 0 && oy >= 0 && ox + b <= wn->ww[0] && oy + b <= wn->wh[0]) {
            const uint8_t *r = wn->win[0] + (size_t)oy * (size_t)wn->win_str[0] + (size_t)ox;
            if (lv->use_satd)
                cost = wn->satd_simd
                    ? mcd_simd_satd_u8u8(wn->cur[0], wn->cur_str[0], r, wn->win_str[0],
                                         (uint32_t)b, (uint32_t)b, bound)
                    : mcd_satd_scalar_u8u8(wn->cur[0], wn->cur_str[0], r, wn->win_str[0],
                                           (uint32_t)b, (uint32_t)b, bound);
            else
                cost = wn->sad_simd
                    ? mcd_simd_sad_u8u8(wn->cur[0], wn->cur_str[0], r, wn->win_str[0],
                                        (uint32_t)b, (uint32_t)b, bound)
                    : mcd_sad_scalar_u8u8(wn->cur[0], wn->cur_str[0], r, wn->win_str[0],
                                          (uint32_t)b, (uint32_t)b, bound);
        } else {
            cost = lv->use_satd
                ? mcd_plane_satd(lv->curp[0], lv->X, lv->Y, lv->refp[0], rx, ry, b, b, bound)
                : mcd_plane_sad(lv->curp[0], lv->X, lv->Y, lv->refp[0], rx, ry, b, b, bound);
        }
    }
    if (bound && cost >= bound)
        return cost; /* luma alone already past the bound: candidate loses */

    uint64_t sum = cost;
    for (uint32_t p = 1; p < wn->n_planes; p++) {
        const int32_t rxc = mcd_me_clamp((wn->X + cx) >> (int32_t)wn->ssx,
                                         -MCD_ME_EDGE_PAD, wn->cw + MCD_ME_EDGE_PAD - wn->bw);
        const int32_t ryc = mcd_me_clamp((wn->Y + cy) >> (int32_t)wn->ssy,
                                         -MCD_ME_EDGE_PAD, wn->ch + MCD_ME_EDGE_PAD - wn->bh);
        const uint64_t rbound = bound > sum ? bound - sum : 0;
        const int32_t ox = rxc - wn->wx[p];
        const int32_t oy = ryc - wn->wy[p];
        if (ox >= 0 && oy >= 0 && ox + wn->bw <= wn->ww[p] && oy + wn->bh <= wn->wh[p]) {
            const uint8_t *r = wn->win[p] + (size_t)oy * (size_t)wn->win_str[p] + (size_t)ox;
            sum += wn->csad_simd
                ? mcd_simd_sad_u8u8(wn->cur[p], wn->cur_str[p], r, wn->win_str[p],
                                    (uint32_t)wn->bw, (uint32_t)wn->bh, rbound)
                : mcd_sad_scalar_u8u8(wn->cur[p], wn->cur_str[p], r, wn->win_str[p],
                                      (uint32_t)wn->bw, (uint32_t)wn->bh, rbound);
        } else {
            sum += mcd_plane_sad(lv->curp[p], wn->X >> (int32_t)wn->ssx, wn->Y >> (int32_t)wn->ssy,
                                 lv->refp[p], rxc, ryc, wn->bw, wn->bh, rbound);
        }
        if (bound && sum >= bound)
            return sum;
    }
    return sum;
}

static uint64_t mcd_l0_raw_u16(const McdMeLevel *lv, const McdMeL0Win *wn,
                               int32_t cx, int32_t cy, uint64_t bound) {
    const int32_t b  = wn->bblk;
    const int32_t rx = mcd_me_clamp(wn->X + cx, -MCD_ME_EDGE_PAD,
                                    wn->lw + MCD_ME_EDGE_PAD - b);
    const int32_t ry = mcd_me_clamp(wn->Y + cy, -MCD_ME_EDGE_PAD,
                                    wn->lh + MCD_ME_EDGE_PAD - b);
    uint64_t cost;

    {
        const int32_t ox = rx - wn->wx[0];
        const int32_t oy = ry - wn->wy[0];
        if (ox >= 0 && oy >= 0 && ox + b <= wn->ww[0] && oy + b <= wn->wh[0]) {
            const uint16_t *r = (const uint16_t *)wn->win[0] +
                                (size_t)oy * (size_t)wn->win_str[0] + (size_t)ox;
            if (lv->use_satd)
                cost = wn->satd_simd
                    ? mcd_simd_satd_u16u16((const uint16_t *)wn->cur[0], wn->cur_str[0], r,
                                           wn->win_str[0], (uint32_t)b, (uint32_t)b, bound)
                    : mcd_satd_scalar_u16u16((const uint16_t *)wn->cur[0], wn->cur_str[0], r,
                                             wn->win_str[0], (uint32_t)b, (uint32_t)b, bound);
            else
                cost = wn->sad_simd
                    ? mcd_simd_sad_u16u16((const uint16_t *)wn->cur[0], wn->cur_str[0], r,
                                          wn->win_str[0], (uint32_t)b, (uint32_t)b, bound)
                    : mcd_sad_scalar_u16u16((const uint16_t *)wn->cur[0], wn->cur_str[0], r,
                                            wn->win_str[0], (uint32_t)b, (uint32_t)b, bound);
        } else {
            cost = lv->use_satd
                ? mcd_plane_satd(lv->curp[0], lv->X, lv->Y, lv->refp[0], rx, ry, b, b, bound)
                : mcd_plane_sad(lv->curp[0], lv->X, lv->Y, lv->refp[0], rx, ry, b, b, bound);
        }
    }
    if (bound && cost >= bound)
        return cost;

    uint64_t sum = cost;
    for (uint32_t p = 1; p < wn->n_planes; p++) {
        const int32_t rxc = mcd_me_clamp((wn->X + cx) >> (int32_t)wn->ssx,
                                         -MCD_ME_EDGE_PAD, wn->cw + MCD_ME_EDGE_PAD - wn->bw);
        const int32_t ryc = mcd_me_clamp((wn->Y + cy) >> (int32_t)wn->ssy,
                                         -MCD_ME_EDGE_PAD, wn->ch + MCD_ME_EDGE_PAD - wn->bh);
        const uint64_t rbound = bound > sum ? bound - sum : 0;
        const int32_t ox = rxc - wn->wx[p];
        const int32_t oy = ryc - wn->wy[p];
        if (ox >= 0 && oy >= 0 && ox + wn->bw <= wn->ww[p] && oy + wn->bh <= wn->wh[p]) {
            const uint16_t *r = (const uint16_t *)wn->win[p] +
                                (size_t)oy * (size_t)wn->win_str[p] + (size_t)ox;
            sum += wn->csad_simd
                ? mcd_simd_sad_u16u16((const uint16_t *)wn->cur[p], wn->cur_str[p], r,
                                      wn->win_str[p], (uint32_t)wn->bw, (uint32_t)wn->bh, rbound)
                : mcd_sad_scalar_u16u16((const uint16_t *)wn->cur[p], wn->cur_str[p], r,
                                        wn->win_str[p], (uint32_t)wn->bw, (uint32_t)wn->bh, rbound);
        } else {
            sum += mcd_plane_sad(lv->curp[p], wn->X >> (int32_t)wn->ssx, wn->Y >> (int32_t)wn->ssy,
                                 lv->refp[p], rxc, ryc, wn->bw, wn->bh, rbound);
        }
        if (bound && sum >= bound)
            return sum;
    }
    return sum;
}

/* Full candidate cost with fused-or-direct selection; mirrors
 * mcd_me_cost_bounded exactly (which passes luma_satd = 1). */
static inline uint64_t mcd_l0_cost(const McdMeLevel *lv, const McdMeL0Win *wn,
                                   int32_t cx, int32_t cy, uint64_t incumbent) {
    const uint64_t lterm = mcd_me_lambda_term(lv, cx, cy);
    const uint64_t bound = incumbent > lterm ? incumbent - lterm : 0;
    uint64_t cost;
    if (wn && wn->staged)
        cost = wn->highbd ? mcd_l0_raw_u16(lv, wn, cx, cy, bound)
                          : mcd_l0_raw_u8(lv, wn, cx, cy, bound);
    else
        cost = mcd_me_sad_raw(lv, cx, cy, 1, bound);
    return cost + lterm;
}

static inline void mcd_l0_try(const McdMeLevel *lv, const McdMeL0Win *wn,
                              int32_t cx, int32_t cy,
                              int32_t *bx, int32_t *by, uint64_t *bc) {
    const uint64_t c = mcd_l0_cost(lv, wn, cx, cy, *bc);
    if (c < *bc) {
        *bc = c;
        *bx = cx;
        *by = cy;
    }
}

/* One non-coarsest-level block search: predictor set {parent-scaled, zero,
 * left, top, top-right, median}, 8-point hexagon refinement iterated to
 * convergence, then a full +-1 ring. This is pure code motion of the
 * sequence previously inlined in mcd_me_analyze_s; candidate order and the
 * strict `<` tie-breaking are unchanged. wn != NULL with wn->staged selects
 * the fused evaluator (level 0); otherwise the direct chain runs. */
static void mcd_me_level_search(McdMeLevel *lv, const McdMeL0Win *wn,
                                const int32_t *fldp, const int32_t *fldc,
                                uint32_t idx, uint32_t gbx, uint32_t gby, uint32_t nbx,
                                int32_t *out_x, int32_t *out_y) {
    static const int32_t hexo[8][2] = {
        { 1, 2 }, { 2, 1 }, { 2, -1 }, { 1, -2 },
        { -1, -2 }, { -2, -1 }, { -2, 1 }, { -1, 2 }
    };

    /* primary predictor: parent vector scaled x2 */
    lv->px = fldp[2 * idx] * 2;
    lv->py = fldp[2 * idx + 1] * 2;
    int32_t  vxc = lv->px;
    int32_t  vyc = lv->py;
    uint64_t cc  = mcd_l0_cost(lv, wn, vxc, vyc, 0);

    /* additional predictors, all penalized vs the primary */
    mcd_l0_try(lv, wn, 0, 0, &vxc, &vyc, &cc);

    int32_t lx = 0, ly = 0, tx = 0, ty = 0, rx = 0, ry = 0;
    const int has_l = (gbx > 0);
    const int has_t = (gby > 0);
    const int has_r = (gby > 0 && gbx + 1 < nbx);
    if (has_l) {
        lx = fldc[2 * (idx - 1)];
        ly = fldc[2 * (idx - 1) + 1];
        mcd_l0_try(lv, wn, lx, ly, &vxc, &vyc, &cc);
    }
    if (has_t) {
        tx = fldc[2 * (idx - nbx)];
        ty = fldc[2 * (idx - nbx) + 1];
        mcd_l0_try(lv, wn, tx, ty, &vxc, &vyc, &cc);
    }
    if (has_r) {
        rx = fldc[2 * (idx - nbx + 1)];
        ry = fldc[2 * (idx - nbx + 1) + 1];
        mcd_l0_try(lv, wn, rx, ry, &vxc, &vyc, &cc);
    }
    if (has_l && has_t && has_r) {
        const int32_t mx = lx + tx + rx -
                           mcd_me_imax(lx, mcd_me_imax(tx, rx)) -
                           mcd_me_imin(lx, mcd_me_imin(tx, rx));
        const int32_t my = ly + ty + ry -
                           mcd_me_imax(ly, mcd_me_imax(ty, ry)) -
                           mcd_me_imin(ly, mcd_me_imin(ty, ry));
        mcd_l0_try(lv, wn, mx, my, &vxc, &vyc, &cc);
    }

    /* 8-point hexagon refinement iterated to convergence, then a full +-1
     * ring. The center commits only after each full sweep; the incumbent
     * tightens immediately. */
    for (;;) {
        int32_t nx = vxc, ny = vyc;
        for (int k = 0; k < 8; k++) {
            const int32_t  cx = vxc + hexo[k][0];
            const int32_t  cy = vyc + hexo[k][1];
            const uint64_t c  = mcd_l0_cost(lv, wn, cx, cy, cc);
            if (c < cc) {
                cc = c;
                nx = cx;
                ny = cy;
            }
        }
        if (nx == vxc && ny == vyc)
            break;
        vxc = nx;
        vyc = ny;
    }
    for (int32_t dy = -1; dy <= 1; dy++)
        for (int32_t dx = -1; dx <= 1; dx++)
            if (dx | dy)
                mcd_l0_try(lv, wn, vxc + dx, vyc + dy, &vxc, &vyc, &cc);

    *out_x = vxc;
    *out_y = vyc;
}

/* Scene-change decision: at least thscd2_pct percent of blocks exceed
 * thscd1 (integer arithmetic). A zero threshold still requires one changed
 * block, while 100 requires every block to be changed. */
static int mcd_me_scene(const McdVector *v, uint32_t n, const McdConfig *cfg) {
    uint32_t changed = 0;
    for (uint32_t i = 0; i < n; i++)
        if ((uint32_t)v[i].sad_norm > cfg->thscd1)
            changed++;
    if (!changed)
        return 0;
    return ((uint64_t)changed * 100 >= (uint64_t)cfg->thscd2_pct * n);
}

/* Validation shared by analyze and recalculate. */
static int mcd_me_frames_ok(const McdFrame *cur, const McdFrame *ref) {
    if (!cur->pyr || !ref->pyr)
        return 0;
    if (cur->num_planes != ref->num_planes)
        return 0;
    if (cur->num_planes != 1 && cur->num_planes != 3)
        return 0;
    if (cur->planes[0].w != ref->planes[0].w || cur->planes[0].h != ref->planes[0].h)
        return 0;
    if (cur->bitdepth != ref->bitdepth || cur->x_ss != ref->x_ss || cur->y_ss != ref->y_ss)
        return 0;
    return cur->planes[0].w != 0 && cur->planes[0].h != 0;
}

/* Pelset source plane for a frame: the prefiltered luma when available. */
static const McdPlane *mcd_me_pel_src(const McdFrame *f) {
    return f->prefiltered ? &f->fplanes[0] : &f->planes[0];
}

/* Half-pel pelset for the reference frame at margin `want_pad`. Uses the
 * glue-built shared cache (ref->me_pel) when present — its phase-plane
 * content is identical to a locally built pelset (content does not depend on
 * the margin) and fetches are clamped to `want_pad`, matching a local build
 * bit-for-bit. Otherwise builds a temporary pelset the caller must destroy.
 * *own tells the caller whether it owns the returned pelset. */
static McdPelSet *mcd_me_ref_pel(const McdFrame *ref, uint32_t want_pad, int *own) {
    *own = 0;
    if (ref->me_pel_valid && ref->me_pel) {
        return (McdPelSet *)ref->me_pel;
    }
    McdPelSet *pel = NULL;
    if (mcd_pelset_create(mcd_me_pel_src(ref), 1, ref->bitdepth,
                          want_pad, &pel) != MCD_SUCCESS) {
        return NULL;
    }
    *own = 1;
    return pel;
}

/* -------------------------------------------------------------------------- */
/* reusable scratch (predictor fields)                                         */

int mcd_me_scratch_create(McdMeScratch **out) {
    if (!out)
        return MCD_ERR_PARAM;
    *out = NULL;
    McdMeScratch *sc = (McdMeScratch *)calloc(1, sizeof(*sc));
    if (!sc)
        return MCD_ERR_NOMEM;
    *out = sc;
    return MCD_SUCCESS;
}

void mcd_me_scratch_destroy(McdMeScratch **s) {
    if (!s || !*s)
        return;
    free((*s)->fld[0]);
    free((*s)->fld[1]);
    free(*s);
    *s = NULL;
}

/* Ensure the scratch holds two int32 fields of n entries. Returns
 * MCD_SUCCESS or MCD_ERR_NOMEM. */
static int mcd_me_scratch_fit(McdMeScratch *sc, uint32_t n) {
    if (!sc)
        return MCD_SUCCESS;
    if (sc->cap >= n)
        return MCD_SUCCESS;
    for (int i = 0; i < 2; i++) {
        int32_t *p = (int32_t *)realloc(sc->fld[i], (size_t)n * sizeof(int32_t));
        if (!p)
            return MCD_ERR_NOMEM;
        sc->fld[i] = p;
    }
    sc->cap = n;
    return MCD_SUCCESS;
}

/* -------------------------------------------------------------------------- */
/* mcd_me_analyze                                                              */

int mcd_me_analyze(const McdFrame *cur, const McdFrame *ref, int32_t delta,
                   const McdConfig *cfg, McdMeResult **out) {
    return mcd_me_analyze_s(cur, ref, delta, cfg, NULL, out);
}

int mcd_me_analyze_s(const McdFrame *cur, const McdFrame *ref, int32_t delta,
                     const McdConfig *cfg, McdMeScratch *scratch, McdMeResult **out) {
    McdMeResult *res  = NULL;
    McdVector   *vecs = NULL;
    int32_t     *fld_a = NULL, *fld_b = NULL;
    McdPelSet   *pel   = NULL;
    int          ret   = MCD_SUCCESS;
    int32_t     *own_fld[2] = { NULL, NULL };

    if (out)
        *out = NULL;
    if (!cur || !ref || !cfg || !out)
        return MCD_ERR_PARAM;
    (void)delta; /* the scheme uses fixed search parameters for every distance */

    const uint32_t blk = cfg->blksize;
    if (blk != 8 && blk != 16 && blk != 32)
        return MCD_ERR_PARAM;
    if (!mcd_me_frames_ok(cur, ref))
        return MCD_ERR_PARAM;

    const uint32_t w    = cur->planes[0].w;
    const uint32_t h    = cur->planes[0].h;
    const uint32_t ov   = blk >> 1;
    const uint32_t step = blk - ov;
    const uint32_t nbx  = mcd_me_grid(w, ov, step);
    const uint32_t nby  = mcd_me_grid(h, ov, step);
    if (nbx > UINT32_MAX / nby)
        return MCD_ERR_PARAM;
    const uint32_t n_blks = nbx * nby;

    uint32_t lev = mcd_pyramid_levels(cur->pyr);
    {
        const uint32_t levr = mcd_pyramid_levels(ref->pyr);
        if (levr < lev)
            lev = levr;
    }
    if (lev == 0)
        return MCD_ERR_PARAM;
    while (lev > 1 && (blk >> (lev - 1)) == 0) /* keep block size >= 1/pel */
        lev--;

    res = (McdMeResult *)calloc(1, sizeof(*res));
    vecs = (McdVector *)calloc(n_blks, sizeof(*vecs));
    /* initialized BEFORE any goto fail: the fail: path reads it (jumping
     * over an initializer would leave it indeterminate -- clang
     * -Wsometimes-uninitialized, a real OOM-path hazard) */
    int pel_own = 0;
    if (scratch) {
        /* grow-to-fit pooled predictor fields; the fields are written
         * before any read at every level, so no per-call zeroing is needed */
        if (mcd_me_scratch_fit(scratch, (uint32_t)n_blks * 2) != MCD_SUCCESS) {
            ret = MCD_ERR_NOMEM;
            goto fail;
        }
        fld_a = scratch->fld[0];
        fld_b = scratch->fld[1];
    } else {
        fld_a = (int32_t *)calloc((size_t)n_blks * 2, sizeof(int32_t));
        fld_b = (int32_t *)calloc((size_t)n_blks * 2, sizeof(int32_t));
        own_fld[0] = fld_a;
        own_fld[1] = fld_b;
    }
    if (!res || !vecs || !fld_a || !fld_b) {
        ret = MCD_ERR_NOMEM;
        goto fail;
    }

    /* Half-pel phase planes over the REFERENCE prefiltered luma: the sub-pel
     * stage interpolates reference blocks. With the glue's shared cache the
     * pelset is built once per reference frame instead of once per call. */
    const uint32_t want_pad = (uint32_t)MCD_ME_EDGE_PAD + (blk >> 1);
    pel = mcd_me_ref_pel(ref, want_pad, &pel_own);
    if (pel == NULL) {
        ret = MCD_ERR_NOMEM;
        goto fail;
    }

    {
        const uint64_t pmax   = (1ull << cur->bitdepth) - 1;
        const uint64_t lambda = 100ull * blk * blk * pmax / (64ull * 255ull);
        const uint32_t ssx    = cur->x_ss;
        const uint32_t ssy    = cur->y_ss;
        const uint32_t npl    = cur->num_planes;

        int32_t  level = (int32_t)lev - 1;
        int32_t *fldp  = NULL; /* coarser-level vectors (parent predictors)   */
        int32_t *fldc  = fld_a;

        for (; level >= 0; level--) {
            McdMeLevel lv;
            memset(&lv, 0, sizeof(lv));
            for (uint32_t p = 0; p < npl; p++) {
                lv.curp[p] = mcd_pyramid_get(cur->pyr, p, (uint32_t)level);
                lv.refp[p] = mcd_pyramid_get(ref->pyr, p, (uint32_t)level);
            }
            lv.n_planes  = npl;
            lv.x_ss      = ssx;
            lv.y_ss      = ssy;
            lv.bblk      = mcd_me_imax(1, (int32_t)blk >> level);
            lv.lambda    = lambda;
            lv.use_satd  = (level == 0 && cfg->satd) ? 1 : 0;

            const int coarsest = (level == (int32_t)lev - 1);

            for (uint32_t by = 0; by < nby; by++) {
                for (uint32_t bx = 0; bx < nbx; bx++) {
                    const uint32_t idx = by * nbx + bx;
                    const int32_t  X   = (int32_t)((bx * step) >> level);
                    const int32_t  Y   = (int32_t)((by * step) >> level);
                    int32_t        vxc, vyc;
                    uint64_t       cc;

                    lv.X = X;
                    lv.Y = Y;

                    if (coarsest) {
                        /* exhaustive integer-pel search, radius 2 around (0,0) */
                        lv.px = 0;
                        lv.py = 0;
                        vxc   = 0;
                        vyc   = 0;
                        cc    = mcd_me_cost(&lv, 0, 0);
                        for (int32_t dy = -2; dy <= 2; dy++)
                            for (int32_t dx = -2; dx <= 2; dx++)
                                if (dx | dy)
                                    mcd_me_try(&lv, dx, dy, &vxc, &vyc, &cc);
                    } else {
                        /* predictor set + hexagon + +-1 ring (see
                         * mcd_me_level_search); at the finest level the
                         * candidate costs come from the fused window
                         * evaluator (bit-identical, direct fallback per
                         * candidate), elsewhere from the direct chain. */
                        McdMeL0Win wn;
                        if (level == 0) {
                            mcd_me_l0_setup(&lv, &wn);
                            mcd_me_level_search(&lv, &wn, fldp, fldc, idx, bx, by, nbx,
                                                &vxc, &vyc);
                        } else {
                            mcd_me_level_search(&lv, NULL, fldp, fldc, idx, bx, by, nbx,
                                                &vxc, &vyc);
                        }
                    }

                    fldc[2 * idx]     = vxc;
                    fldc[2 * idx + 1] = vyc;

                    if (level == 0) {
                        /* half-pel refinement: 8 neighbors around the best
                         * integer vector, luma-only sub-pel cost against the
                         * interpolated reference (the integer-rounded vector
                         * position is used for chroma). */
                        int32_t  hx = vxc * 2;
                        int32_t  hy = vyc * 2;
                        uint64_t bl = mcd_hp_luma_sad(lv.curp[0], pel, X, Y,
                                                      (int32_t)blk, hx, hy, want_pad, 0);
                        static const int32_t off8[8][2] = {
                            { -1, -1 }, { 0, -1 }, { 1, -1 }, { -1, 0 },
                            { 1, 0 }, { -1, 1 }, { 0, 1 }, { 1, 1 }
                        };
                        for (int k = 0; k < 8; k++) {
                            const int32_t cx = vxc * 2 + off8[k][0];
                            const int32_t cy = vyc * 2 + off8[k][1];
                            const uint64_t l = mcd_hp_luma_sad(lv.curp[0], pel, X, Y,
                                                               (int32_t)blk, cx, cy, want_pad, bl);
                            if (l < bl) {
                                bl = l;
                                hx = cx;
                                hy = cy;
                            }
                        }

                        const uint64_t sad_raw =
                            bl + mcd_chroma_sad(lv.curp, lv.refp, npl, ssx, ssy,
                                                X, Y, (int32_t)blk,
                                                mcd_me_half(hx), mcd_me_half(hy), 0);
                        vecs[idx].x        = hx;
                        vecs[idx].y        = hy;
                        vecs[idx].sad_norm = mcd_norm_sad(sad_raw, blk * blk,
                                                          ssx, ssy, cur->bitdepth);
                    }
                }
            }

            /* descend: the finished level becomes the parent predictor field */
            fldp = fldc;
            fldc = (fldc == fld_a) ? fld_b : fld_a;
        }
    }

    res->blk_w        = blk;
    res->blk_h        = blk;
    res->overlap      = ov;
    res->n_blks_x     = nbx;
    res->n_blks_y     = nby;
    res->vecs         = vecs;
    res->scene_change = mcd_me_scene(vecs, n_blks, cfg);
    vecs              = NULL;

fail:
    if (pel_own) {
        mcd_pelset_destroy(&pel);
    }
    free(own_fld[0]);
    free(own_fld[1]);
    free(vecs);
    if (ret != MCD_SUCCESS) {
        free(res);
        return ret;
    }
    *out = res;
    return MCD_SUCCESS;
}

/* -------------------------------------------------------------------------- */
/* mcd_me_recalculate                                                          */

int mcd_me_recalculate(const McdFrame *cur, const McdFrame *ref,
                       const McdConfig *cfg, McdMeResult *inout) {
    return mcd_me_recalculate_s(cur, ref, cfg, NULL, inout);
}

int mcd_me_recalculate_s(const McdFrame *cur, const McdFrame *ref,
                         const McdConfig *cfg, McdMeScratch *scratch,
                         McdMeResult *inout) {
    /* scratch is unused here today: the only surviving allocation of this
     * pass is the vector array itself, which escapes into inout->vecs */
    (void)scratch;
    McdVector *nv    = NULL;
    McdPelSet *pel   = NULL;
    const McdPlane *cp[3] = { 0 };
    const McdPlane *rp[3] = { 0 };
    int ret = MCD_SUCCESS;

    if (!cur || !ref || !cfg || !inout || !inout->vecs)
        return MCD_ERR_PARAM;
    if (!mcd_me_frames_ok(cur, ref))
        return MCD_ERR_PARAM;

    const uint32_t blk = inout->blk_w;
    if (blk < 4 || blk != inout->blk_h || (blk & (blk - 1)) != 0)
        return MCD_ERR_PARAM;
    const uint32_t oov   = inout->overlap;
    const uint32_t ostep = blk - oov;
    if (oov >= blk || ostep == 0 || (blk % ostep) != 0)
        return MCD_ERR_PARAM;
    const uint32_t onbx = inout->n_blks_x;
    const uint32_t onby = inout->n_blks_y;
    if (onbx == 0 || onby == 0)
        return MCD_ERR_PARAM;

    const uint32_t w     = cur->planes[0].w;
    const uint32_t h     = cur->planes[0].h;
    const uint32_t blk2  = blk >> 1;
    const uint32_t ov2   = blk2 >> 1;
    const uint32_t step2 = blk2 - ov2;
    const uint32_t nbx2  = mcd_me_grid(w, ov2, step2);
    const uint32_t nby2  = mcd_me_grid(h, ov2, step2);
    if (nbx2 > UINT32_MAX / nby2)
        return MCD_ERR_PARAM;
    const uint32_t n2 = nbx2 * nby2;

    const uint32_t ssx = cur->x_ss;
    const uint32_t ssy = cur->y_ss;
    const uint32_t npl = cur->num_planes;
    for (uint32_t p = 0; p < npl; p++) {
        cp[p] = mcd_pyramid_get(cur->pyr, p, 0);
        rp[p] = mcd_pyramid_get(ref->pyr, p, 0);
    }

    /* initialized BEFORE any goto fail: the fail: path reads it (jumping
     * over an initializer would leave it indeterminate -- clang
     * -Wsometimes-uninitialized, a real OOM-path hazard) */
    int pel_own = 0;
    nv    = (McdVector *)calloc(n2, sizeof(*nv));
    if (!nv) {
        ret = MCD_ERR_NOMEM;
        goto fail;
    }

    /* half-pel phases over the reference prefiltered luma (level 0). With
     * the glue's shared cache (built at the analyze-grid margin, which is
     * WIDER than what this pass clamps to) the fetches clamp to this pass's
     * own expected margin, matching a local pelset build bit-for-bit. */
    const uint32_t want_pad = (uint32_t)MCD_ME_EDGE_PAD + (blk2 >> 1);
    pel = mcd_me_ref_pel(ref, want_pad, &pel_own);
    if (pel == NULL) {
        ret = MCD_ERR_NOMEM;
        goto fail;
    }

    for (uint32_t by2 = 0; by2 < nby2; by2++) {
        for (uint32_t bx2 = 0; bx2 < nbx2; bx2++) {
            const uint32_t idx2 = by2 * nbx2 + bx2;
            const int32_t  X    = (int32_t)(bx2 * step2);
            const int32_t  Y    = (int32_t)(by2 * step2);

            /* parent block (old grid) containing this subblock */
            const int32_t pobx = mcd_me_imin((int32_t)(X / ostep), (int32_t)onbx - 1);
            const int32_t poby = mcd_me_imin((int32_t)(Y / ostep), (int32_t)onby - 1);
            const McdVector old = inout->vecs[(uint32_t)poby * onbx + (uint32_t)pobx];

            int32_t  hx = old.x;
            int32_t  hy = old.y;

            /* SAD at the inherited vector: luma at half-pel precision via the
             * pelset, chroma at the integer-rounded position. */
            uint64_t bl = mcd_hp_luma_sad(cp[0], pel, X, Y, (int32_t)blk2, hx, hy, want_pad, 0);
            uint64_t raw = bl + mcd_chroma_sad(cp, rp, npl, ssx, ssy, X, Y,
                                               (int32_t)blk2,
                                               mcd_me_half(hx), mcd_me_half(hy), 0);
            int32_t sn = mcd_norm_sad(raw, blk2 * blk2, ssx, ssy, cur->bitdepth);

            if ((uint32_t)sn * 2u > cfg->thsad) {
                /* small bounded refinement: hexagon + +-1 ring in half-pel
                 * units, candidates clamped to a +-4 half-pel box around the
                 * initial vector; luma-only cost, chroma follows the integer
                 * part of the winner. */
                static const int32_t hexo[8][2] = {
                    { 1, 2 }, { 2, 1 }, { 2, -1 }, { 1, -2 },
                    { -1, -2 }, { -2, -1 }, { -2, 1 }, { -1, 2 }
                };
                const int32_t xlo = hx - 4, xhi = hx + 4;
                const int32_t ylo = hy - 4, yhi = hy + 4;

                for (;;) {
                    int32_t nx = hx, ny = hy;
                    for (int k = 0; k < 8; k++) {
                        const int32_t cx = mcd_me_clamp(hx + hexo[k][0], xlo, xhi);
                        const int32_t cy = mcd_me_clamp(hy + hexo[k][1], ylo, yhi);
                        const uint64_t l = mcd_hp_luma_sad(cp[0], pel, X, Y,
                                                           (int32_t)blk2, cx, cy, want_pad, bl);
                        if (l < bl) {
                            bl = l;
                            nx = cx;
                            ny = cy;
                        }
                    }
                    if (nx == hx && ny == hy)
                        break;
                    hx = nx;
                    hy = ny;
                }
                for (int32_t dy = -1; dy <= 1; dy++) {
                    for (int32_t dx = -1; dx <= 1; dx++) {
                        if (!(dx | dy))
                            continue;
                        const int32_t cx = mcd_me_clamp(hx + dx, xlo, xhi);
                        const int32_t cy = mcd_me_clamp(hy + dy, ylo, yhi);
                        const uint64_t l = mcd_hp_luma_sad(cp[0], pel, X, Y,
                                                           (int32_t)blk2, cx, cy, want_pad, bl);
                        if (l < bl) {
                            bl = l;
                            hx = cx;
                            hy = cy;
                        }
                    }
                }

                raw = bl + mcd_chroma_sad(cp, rp, npl, ssx, ssy, X, Y, (int32_t)blk2,
                                          mcd_me_half(hx), mcd_me_half(hy), 0);
                sn = mcd_norm_sad(raw, blk2 * blk2, ssx, ssy, cur->bitdepth);
            }

            nv[idx2].x        = hx;
            nv[idx2].y        = hy;
            nv[idx2].sad_norm = sn;
        }
    }

    free(inout->vecs);
    inout->vecs     = nv;
    nv              = NULL;
    inout->blk_w    = blk2;
    inout->blk_h    = blk2;
    inout->overlap  = ov2;
    inout->n_blks_x = nbx2;
    inout->n_blks_y = nby2;
    inout->scene_change = mcd_me_scene(inout->vecs, n2, cfg);

fail:
    if (pel_own) {
        mcd_pelset_destroy(&pel);
    }
    free(nv);
    return ret;
}

/* -------------------------------------------------------------------------- */

void mcd_me_result_free(McdMeResult **r) {
    if (!r || !*r)
        return;
    free((*r)->vecs);
    free(*r);
    *r = NULL;
}
