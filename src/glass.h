// SPDX-License-Identifier: Apache-2.0 AND LicenseRef-Commons-Clause-1.0
// Radiance glass, ported from rad5R4 src/rt/m_glass.c (see THIRD_PARTY_NOTICES.md): a pane as two parallel faces with
// the internal reflections summed, refractive index 1.52, unpolarised light. Shared by the device program and the host.
#pragma once
#include <math.h>
#include <vector_types.h>   // __host__ __device__ for host compilers too (crt/host_defines.h)

// Transmissivity tn from the transmittance at normal incidence Tn (m_glass.c header comment; n = 1.52). Double: the
// difference of near-equal terms loses ~5 digits in float.
__host__ __device__ inline double tn_from_Tn(double Tn) {
  return (sqrt(.8402528435 + .0072522239 * Tn * Tn) - .9166530661) / .0036261119 / Tn;
}

// Transmittance T and reflectance R of the pane at incidence cosine c (0..1) for transmissivity tn (< 1). m_glass.c with
// its 1/pdot terms multiplied out (r1m = ((c2 - n c) / (c2 + n c))^2), so grazing c = 0 gives T 0 instead of NaN.
__host__ __device__ inline void glass_TR(float c, float tn, float& T, float& R) {
  const float n = 1.52f;
  const float c2 = sqrtf((1.f - 1.f / (n * n)) + c * c / (n * n));   // cosine inside the glass
  const float d = powf(tn, 1.f / c2), d2 = d * d;                     // transmissivity along the slanted path
  float rs = (c - n * c2) / (c + n * c2), rp = (c2 - n * c) / (c2 + n * c);
  rs *= rs; rp *= rp;
  T = .5f * (1.f - rs) * (1.f - rs) * d / (1.f - rs * rs * d2) + .5f * (1.f - rp) * (1.f - rp) * d / (1.f - rp * rp * d2);
  R = .5f * rs * (1.f + (1.f - 2.f * rs) * d2) / (1.f - rs * rs * d2) + .5f * rp * (1.f + (1.f - 2.f * rp) * d2) / (1.f - rp * rp * d2);
}
