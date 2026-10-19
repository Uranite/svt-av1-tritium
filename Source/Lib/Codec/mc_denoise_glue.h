/*
 * mc_denoise_glue.h — encoder integration of the motion-compensated
 * temporal denoiser (see mc_denoise.h for the algorithm modules).
 *
 * The glue owns one state object per encoder instance, hung off the shared
 * EncodeContext (SCS objects are struct-copied by the pipeline, EncodeContext
 * is not). The state buffers copies of the most recent input pictures and
 * delays the release of PictureParentControlSets into the Picture Analysis
 * kernel until enough temporal neighbours have arrived (or end of stream is
 * signalled), denoises them, and writes the result back into the pcs input
 * buffers before the regular PA body runs.
 */
#ifndef MC_DENOISE_GLUE_H
#define MC_DENOISE_GLUE_H

#include "EbSvtAv1Enc.h"
#include "mc_denoise.h"

#ifdef __cplusplus
extern "C" {
#endif

struct EbObjectWrapper;
struct PictureParentControlSet;

/* Maximum number of pictures a single PA iteration can get back from the
 * denoiser: the whole holding queue plus the just-arrived picture. */
#define SVT_MCD_MAX_RELEASED (2 * MCD_MAX_TR + 2)

/* Size of the caller-side output arrays passed to svt_mcd_pa_arrival. One
 * arrival call can emit its own pass-through plus every head it drains; the
 * holding queue never holds more than SVT_MCD_MAX_RELEASED heads (the
 * capacity guard keeps it there), and draining may be interleaved with other
 * threads' pushes, so the emit bound is one larger than the queue bound. */
#define SVT_MCD_MAX_EMITTED (2 * MCD_MAX_TR + 3)

/* Create the per-instance denoiser state from the encoder configuration.
 * Only the mc_denoise* fields of cfg are read. *state_ptr must be NULL. */
EbErrorType svt_mcd_state_create(void** state_ptr, const EbSvtAv1EncConfiguration* cfg);

/* Destroy the state created above (safe to call with *state_ptr == NULL). */
void svt_mcd_state_destroy(void** state_ptr);

/* Picture Analysis kernel arrival hook.
 *
 * Hands over the freshly arrived picture (its pixel planes are copied
 * immediately, so the caller may proceed with other work). Returns the number
 * of pictures that may proceed through the regular PA body (possibly 0 when
 * the arrived picture itself is still waiting for neighbours, 1 in the common
 * case, and up to SVT_MCD_MAX_RELEASED when end of stream flushes the holding
 * queue). out_wrappers/out_pcs receive the released pictures; the caller must
 * run the normal PA body on each of them and forward exactly those wrappers to
 * the PA results fifo.
 *
 * Overlay pictures are never denoised (they do not carry the input pixels at
 * this stage); they are passed through untouched. */
uint32_t svt_mcd_pa_arrival(void* state,
                            struct PictureParentControlSet* pcs,
                            struct EbObjectWrapper* pcs_wrapper,
                            struct EbObjectWrapper** out_wrappers,
                            struct PictureParentControlSet** out_pcs);

#ifdef __cplusplus
}
#endif

#endif // MC_DENOISE_GLUE_H
