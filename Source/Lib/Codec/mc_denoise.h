/*
 * mc_denoise.h — motion-compensated temporal denoiser (mc_degrain port)
 *
 * Clean-room reimplementation, from published algorithms and formulas only, of
 * the VapourSynth workflow:
 *   mc_degrain(clip, prefilter=Prefilter.DFTTEST,
 *              preset=MVToolsPreset.HQ_SAD, thsad=400)
 * i.e. dfttest spatial prefilter + mvtools-style hierarchical block-matching
 * motion estimation + SAD-weighted motion-compensated temporal aggregation.
 *
 * No code was copied from GPL-licensed reference projects (mvutensils /
 * vs-dfttest2); only the underlying math is reproduced.
 *
 * Files:
 *   mc_denoise.h          this contract (shared types + module interfaces)
 *   mc_denoise_pel.c      picture pyramids, half-pel phase planes, block fetch
 *   mc_denoise_dfttest.c  dfttest spatial prefilter + luma mask merge
 *   mc_denoise_me.c       hierarchical EPZ motion analysis + recalculate pass
 *   mc_denoise_degrain.c  motion-compensated weighted temporal aggregation
 *   mc_denoise.c          encoder glue: frame queue, orchestration, write-back
 *
 * Conventions fixed by this contract:
 *   - Pixel data is planar. 8-bit planes are uint8_t, 10/12-bit planes are
 *     uint16_t native-endian (highbd flag selects).
 *   - All vectors are in HALF-PEL LUMA SAMPLE units at full resolution
 *     relative to the block grid origin; (0,0) = identical blocks.
 *   - All SADs exposed to weighting/thresholds are NORMALIZED mean absolute
 *     residuals across the represented plane samples, scaled to 64 samples
 *     at 8-bit range (see mcd_norm_sad). Thresholds (thsad, thscd1) use this
 *     implementation-specific unit; they are not directly interchangeable
 *     with thresholds from other denoisers.
 *   - Internal scratch memory uses plain malloc/free; every create/destroy
 *     pair is balanced. Functions return MCD_* codes.
 */

#ifndef MC_DENOISE_H
#define MC_DENOISE_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MCD_MAX_TR 3

#define MCD_SUCCESS 0
#define MCD_ERR_NOMEM 1
#define MCD_ERR_PARAM 2

/* ---------------------------------------------------------------------- */
/* configuration (mirrors the --mc-denoise* encoder parameters)            */

typedef struct McdConfig {
    uint8_t  tr;           /* temporal radius, 1..MCD_MAX_TR                    */
    uint32_t thsad;        /* SAD soft threshold, normalized units, default 50  */
    uint32_t blksize;      /* analysis block size (square), 8/16/32, default 16 */
    uint8_t  prefilter;    /* 0 = none, 1 = dfttest                             */
    uint32_t sigma_pct;    /* prefilter sigma scale, percent, default 50         */
    uint32_t thscd1;       /* per-block scene-change SAD, default 400            */
    uint32_t thscd2_pct;   /* % changed blocks => scene change, default 51       */
    uint32_t limit;        /* max output delta vs source (normalized 8-bit
                              scale), 0 = unlimited                              */
    uint8_t  satd;         /* use SATD instead of SAD at finest ME level         */
} McdConfig;

/* ---------------------------------------------------------------------- */
/* picture representation                                                  */

typedef struct McdPlane {
    uint32_t w, h;         /* visible samples                                   */
    uint32_t stride;       /* samples per row (NOT bytes; uint16 units if highbd)*/
    bool     highbd;       /* 0: uint8_t, 1: uint16_t                           */
    void    *data;         /* module-visible buffer, owns no memory semantics:
                              for input planes it points at encoder memory
                              (read-only), for module planes at malloc'd data   */
} McdPlane;

typedef struct McdFrame {
    McdPlane planes[3];    /* [0]=Y, [1]=Cb, [2]=Cr (2,3 unused if mono)        */
    uint32_t num_planes;   /* 1 or 3                                             */
    uint32_t x_ss, y_ss;   /* chroma subsampling shifts (0 or 1)                 */
    uint32_t bitdepth;     /* 8, 10 or 12                                        */
    /* module caches (owned by mc_denoise.c glue) */
    McdPlane fplanes[3];   /* prefiltered planes (prefilter==1)                  */
    bool     prefiltered;
    void    *pyr;          /* McdPyramid* over fplanes, for ME                   */
    bool     pyr_valid;
    void    *pel;          /* McdPelSet* over planes (ORIGINAL pixels), for
                              degrain compensation                               */
    bool     pel_valid;
    void    *me_pel;       /* McdPelSet* over the PREFILTERED luma, shared by
                              every analyze/recalculate call against this
                              frame as reference; built by the glue, read-only
                              for the modules (may be NULL)                      */
    bool     me_pel_valid;
} McdFrame;

/* ---------------------------------------------------------------------- */
/* shared inline math                                                      */

/* Normalize a raw SAD (luma + chroma block SADs summed, native depth) to
 * 8-bit-range equivalent-per-64-luma-samples units.
 * ss_factor as a fraction: 1 + 2/((1<<x_ss)*(1<<y_ss)) = (2^s + 2) / 2^s.   */
static inline int32_t mcd_norm_sad(uint64_t sad_raw, uint32_t blk_area_luma,
                                   uint32_t x_ss, uint32_t y_ss, uint32_t bitdepth) {
    const uint64_t s      = (uint64_t)1 << (x_ss + y_ss);
    const uint64_t num    = s + 2;                   /* ss_factor numerator   */
    const uint64_t den    = s;                       /* ss_factor denominator */
    const uint64_t pmax   = (1ull << bitdepth) - 1;
    /* sad_norm = sad_raw * 255 * 64 * den / (pmax * area * num), rounded     */
    return (int32_t)((sad_raw * 255ull * 64ull * den + (pmax * blk_area_luma * num) / 2) /
                     (pmax * (uint64_t)blk_area_luma * num));
}

/* mvtools degrain weight: with r = sad/thsad in Q16 (0..65536),
 * w = 256 * (1 - r^2) / (1 + r^2), i.e. 256 at sad=0, 0 at sad>=thsad.      */
static inline uint32_t mcd_degrain_weight(int32_t sad_norm, uint32_t thsad) {
    if (thsad == 0 || sad_norm >= (int32_t)thsad || sad_norm < 0)
        return 0;
    const uint64_t r  = ((uint64_t)(uint32_t)sad_norm << 16) / thsad; /* 0..65536 */
    const uint64_t r2 = (r * r) >> 16;                                /* 0..65536 */
    const uint64_t n  = 65536ull - r2;
    const uint64_t d  = 65536ull + r2;
    return (uint32_t)((256ull * n + d / 2) / d);                      /* 0..256 */
}

/* ---------------------------------------------------------------------- */
/* mc_denoise_pel.c — pyramids, half-pel planes, block fetch               */

typedef struct McdPyramid McdPyramid;   /* opaque */

/* Build a gaussian-free bilinear-reduction pyramid over src_planes.
 * Reduction filter is the separable (1,3,3,1)/8 (mvtools rfilter=1).
 * Level 0 references the SOURCE planes (no copy); levels >= 1 are allocated.
 * pad: extra margin replicated at each level's borders (>= blk + 8).
 * max_levels: cap on reduction levels (typ. 4).                              */
int  mcd_pyramid_create(const McdPlane *src_planes, uint32_t n_planes,
                        uint32_t pad, uint32_t max_levels, McdPyramid **out);
void mcd_pyramid_destroy(McdPyramid **p);
const McdPlane *mcd_pyramid_get(const McdPyramid *p, uint32_t plane, uint32_t level);
uint32_t        mcd_pyramid_levels(const McdPyramid *p);

typedef struct McdPelSet McdPelSet;     /* opaque */

/* Half-pel (pel=2) phase planes over the ORIGINAL pixels, with Wiener 6-tap
 * (1,-5,20,20,-5,1)/32 interpolation for non-integer phases (mvtools
 * sharp=2), edge-replicated beyond borders by `pad` (pad >= 32).
 * bitdepth is the owning frame's native depth (8, 10 or 12); interpolation
 * clamps to that depth's sample range, including for uint16-backed planes.
 * Fetch clamps requested coordinates into the valid padded region, so any
 * in-range motion vector is safe to fetch.                                   */
int  mcd_pelset_create(const McdPlane *src_planes, uint32_t n_planes,
                       uint32_t bitdepth, uint32_t pad, McdPelSet **out);

/* Create variant for callers that already hold an identical padded copy of
 * src_planes[0] (same w/h/stride/highbd, edge-replicated margin of exactly
 * `pad`, data at the visible origin): the pelset ADOPTS that storage for
 * plane 0's integer phase instead of rebuilding it (the creator keeps
 * owning/freeing the borrowed buffer, which must outlive the pelset). All
 * other phases and planes are built exactly as mcd_pelset_create would;
 * fetches are bit-identical. borrow0 == NULL behaves like
 * mcd_pelset_create. */
int  mcd_pelset_create_borrow0(const McdPlane *src_planes, uint32_t n_planes,
                               uint32_t bitdepth, uint32_t pad,
                               const McdPlane *borrow0, McdPelSet **out);
void mcd_pelset_destroy(McdPelSet **p);

/* Copy a w x h block whose top-left corner is at (x_halfpel, y_halfpel) in
 * half-pel luma-sample units (for chroma planes the caller passes the same
 * vector's chroma coordinates: x_halfpel >> x_ss is NOT done here — the
 * caller pre-shifts, see mc_denoise_degrain.c). Interpolated with the
 * 6-tap Wiener filter for fractional phases. dst is uint16_t[w * h] and
 * holds native-depth values regardless of plane bit width.                   */
void mcd_pelset_fetch(const McdPelSet *s, uint32_t plane,
                      int32_t x_halfpel, int32_t y_halfpel,
                      uint32_t w, uint32_t h, uint16_t *dst);

/* Same fetch, with the coordinate clamp applied as if the pelset carried the
 * margin `eff_pad` instead of its own (clamped to the pelset's own pad, so
 * reads stay in bounds). The phase-plane CONTENT of a pelset does not depend
 * on its pad — only the clamp range does — so fetching from a wider pelset
 * with a smaller eff_pad is bit-identical to fetching from a pelset built
 * with that smaller pad. Callers use this to share one cached pelset across
 * users that were built expecting different margins. */
void mcd_pelset_fetch_pad(const McdPelSet *s, uint32_t plane,
                          int32_t x_halfpel, int32_t y_halfpel,
                          uint32_t w, uint32_t h, uint32_t eff_pad,
                          uint16_t *dst);

/* Direct-access twin of mcd_pelset_fetch_pad: applies the exact same
 * coordinate clamp and phase selection, but instead of copying a w x h
 * block it returns a pointer to the FIRST ROW of the selected phase plane
 * (row r of the block sits at (const uint8_t*)row0 + (ptrdiff_t)r*stride
 * *element_size, element_size 2 for highbd pelsets / 1 otherwise, samples
 * in their native depth). Callers can therefore run block kernels (e.g.
 * SAD) straight over the phase plane with no staging copy. */
void mcd_pelset_locate(const McdPelSet *s, uint32_t plane,
                       int32_t x_halfpel, int32_t y_halfpel,
                       uint32_t w, uint32_t h, uint32_t eff_pad,
                       const void **row0, int32_t *stride);

/* ---------------------------------------------------------------------- */
/* mc_denoise_dfttest.c — spatial prefilter                                */

/* dfttest spatial denoise of one plane, vsdenoise Prefilter.DFTTEST:
 *   block 16x16, step 4 (75% overlap), Hann window normalized for overlap,
 *   type-0 Wiener shrink  mult = max((psd - sigma)/psd, 0),
 *   sigma(f) piecewise-linear through (0.0,4),(0.2,9),(1.0,15) [variance,
 *   8-bit^2 units] scaled by sigma_pct/100, zmean on.
 * Internal pixel scale is the 8-bit range (native/2^(bd-8)) so the sigma
 * profile keeps its 8-bit meaning at any depth. dst may not alias src.      */
int mcd_dfttest_plane(const McdPlane *src, McdPlane *dst, uint32_t sigma_pct);

/* Reusable constant-table context (window, frequency response, twiddles):
 * depends only on sigma_pct. Build once and share across calls and threads
 * (read-only after creation); saves rebuilding the tables per call. */
typedef struct McdDftCtx McdDftCtx;
int  mcd_dfttest_ctx_create(uint32_t sigma_pct, McdDftCtx **out);
void mcd_dfttest_ctx_destroy(McdDftCtx **ctx);
/* Same filter as mcd_dfttest_plane, using the caller's context (which must
 * have been created with the same sigma_pct the plain call would pass). */
int  mcd_dfttest_plane_with_ctx(const McdDftCtx *ctx, const McdPlane *src,
                                McdPlane *dst);

/* vsdenoise luma-masked merge, applied after the dfttest pass:
 *   m = clamp((Y8 - 16) / (75 - 16), 0, 1)  from the SOURCE luma
 *   out = src + m * (filt - src)            on every plane
 * (keeps dark, noise-dominant regions closer to source). For subsampled
 * chroma the luma mask is box-filtered down. src/filt/out are the caller's
 * plane triplets; out may alias src and/or filt (every output pixel is
 * computed from reads at its own coordinate before it is written). */
int mcd_prefilter_mask_merge(const McdPlane *src_planes, const McdPlane *filt_planes,
                             McdPlane *out_planes, uint32_t n_planes,
                             uint32_t x_ss, uint32_t y_ss, uint32_t bitdepth);

/* ---------------------------------------------------------------------- */
/* mc_denoise_me.c — motion analysis                                        */

typedef struct McdVector {
    int32_t x, y;          /* half-pel luma samples, relative to block origin  */
    int32_t sad_norm;      /* normalized SAD (see mcd_norm_sad)                */
} McdVector;

typedef struct McdMeResult {
    uint32_t blk_w, blk_h, overlap;   /* final grid geometry (post-recalc)     */
    uint32_t n_blks_x, n_blks_y;      /* block grid covering the frame         */
    McdVector *vecs;                  /* n_blks_x*n_blks_y, row-major          */
    int      scene_change;            /* 1 if vs ref judged a scene change     */
} McdMeResult;

/* Hierarchical EPZ analysis of cur vs ref (both must be PREFILTERED frames
 * with pyr caches built by the glue: cur->pyr/ref->pyr over fplanes).
 * Search: coarsest level small exhaustive, finer levels hexagon refinement
 * around predictors {parent-scaled, left, top, top-right, median, zero},
 * cost = SAD(Y)+SAD(U)+SAD(V) [SATD(Y) at level 0 when cfg->satd] +
 *        (lambda * dist^2) >> 8, lambda = 100 * blkArea/64 * pmax/255,
 * half-pel refinement at level 0 (via the prefiltered luma half-pel phases
 * built internally by this module). delta is signed frame distance
 * (positive = ref is in the past). Grid geometry in *out matches
 * cfg->blksize with overlap blksize/2.                                      */
int  mcd_me_analyze(const McdFrame *cur, const McdFrame *ref, int32_t delta,
                    const McdConfig *cfg, McdMeResult **out);

/* Reusable per-caller scratch for the motion analysis passes (predictor
 * fields; a few hundred KB at 1080p). Create one per calling thread context
 * and share it across calls that are serialized on the caller's side (the
 * scratch is owned by one analyze call at a time). The plain entry points
 * above behave exactly like the _s variants with a NULL scratch (they
 * allocate internally and free before returning). */
typedef struct McdMeScratch McdMeScratch;
int  mcd_me_scratch_create(McdMeScratch **out);
void mcd_me_scratch_destroy(McdMeScratch **s);
int  mcd_me_analyze_s(const McdFrame *cur, const McdFrame *ref, int32_t delta,
                      const McdConfig *cfg, McdMeScratch *scratch, McdMeResult **out);

/* Refinement pass on the same pair (prefiltered frames): re-grids the
 * vectors to blksize/2 with overlap blksize/4, recomputes every normalized
 * SAD from the prefiltered pixels at half-pel precision, and re-searches
 * (small hexagon, radius 2 half-pel) only blocks whose SAD exceeds
 * thsad/2. inout is mutated (vecs reallocated). */
int  mcd_me_recalculate(const McdFrame *cur, const McdFrame *ref,
                        const McdConfig *cfg, McdMeResult *inout);
/* Scratch-sharing variant of mcd_me_recalculate (same semantics). */
int  mcd_me_recalculate_s(const McdFrame *cur, const McdFrame *ref,
                          const McdConfig *cfg, McdMeScratch *scratch,
                          McdMeResult *inout);
void mcd_me_result_free(McdMeResult **r);

/* ---------------------------------------------------------------------- */
/* mc_denoise_degrain.c — temporal aggregation                              */

/* Motion-compensated degrain of center against n_refs neighbors.
 * For each ref i, results[i] supplies the final (post-recalc) vector grid;
 * refs[i] must have pel caches (refs[i]->pel over ORIGINAL planes).
 * Per block: w_i = mcd_degrain_weight(sad_norm, thsad) (0 if
 * results[i]->scene_change). Weights are then normalized the mvtools way:
 * the center keeps its full weight 256 and the total is scaled to exactly
 * 256, i.e. WSum = 256 + sum(w_i);  WRef_i = w_i * 256 / WSum;
 * WSrc = 256 - sum(WRef_i) (>= 0, residual absorbs rounding).
 * Block output B = (bias + WSrc*src + sum(WRef_i*comp_i)) >> 8  per pixel,
 * accumulated over the overlapped grid with raised-cosine^2 windows
 * quantized 0..2048 (sum 2048 per sample): out = (acc + 1024) >> 11.
 * limit>0 clamps |out - src| <= limit (scaled to bitdepth).
 * dst planes are written fully (luma + chroma); dst must not alias any
 * ref plane it reads; dst MAY alias center->planes ONLY if the caller
 * guarantees refs' pel sets were built before (they are, by the glue).      */
int mcd_degrain_frame(const McdFrame *center, const McdFrame *const *refs,
                      const McdMeResult *const *results, uint32_t n_refs,
                      const McdConfig *cfg, McdPlane *dst_planes);

#ifdef __cplusplus
}
#endif

#endif /* MC_DENOISE_H */
