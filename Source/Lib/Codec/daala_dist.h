/*
* Copyright (c) 2016, Alliance for Open Media. All rights reserved
*
* This source code is subject to the terms of the BSD 2 Clause License and
* the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
* was not distributed with this source code in the LICENSE file, you can
* obtain it at www.aomedia.org/license/software. If the Alliance for Open
* Media Patent License 1.0 was not distributed with this source code in the
* PATENTS file, you can obtain it at www.aomedia.org/license/patent.
*/

#ifndef EbDaalaDist_h
#define EbDaalaDist_h

#include "definitions.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef int32_t od_coeff;

double svt_aom_od_compute_dist(uint16_t *x, uint16_t *y, int bsize_w, int bsize_h, int qindex, int activity_masking);

#ifdef __cplusplus
}
#endif

#endif // EbDaalaDist_h
