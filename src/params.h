// SPDX-License-Identifier: Apache-2.0 AND LicenseRef-Commons-Clause-1.0
// Launch parameters and hit-record data shared by the OptiX device program (trace_dev.cu) and the host (gpu.cu, run.cpp).
#pragma once
#include <vector_types.h>

// Optics of one global slot (mesh base + face slot), the data of its hit record. Rewritten every run.
struct Slot {
  float rho = 0.2f;       // diffuse reflectance
  float tau = 0.f;        // transmittance at normal incidence; > 0 = translucent (any-hit), 0 = opaque
  float tn = 0.f;         // glass transmissivity (tn_from_Tn(tau)), used with kSlotGlass
  unsigned flags = 0;     // kSlot*
};
enum : unsigned {
  kSlotGlass = 1,         // transmittance by angle (Radiance glass); else tau at every angle (porous, --fresnel off)
  kSlotThin = 2,          // closed thin solid (a pane modelled as a box): the exit face passes everything
  kSlotInward = 4,        // ... wound inward: entry is the back face
};

struct Params {
  unsigned long long gas; // OptixTraversableHandle
  const float3* P;        // sensor origins of this launch (device, already offset by i0)
  const float3* N;        // sensor normals
  const float3* D;        // direction table
  unsigned nDir;
  unsigned nc;            // sensors in the chunk (row stride of bits / bytes)
  unsigned rayFlags;      // OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT [| DISABLE_ANYHIT] [| CULL_BACK_FACING_TRIANGLES]
  unsigned pad;
  unsigned* bits;         // word-major: bits[w*nc + i], bit k of word w = direction w*32+k visible (null in byte mode)
  unsigned char* bytes;   // byte mode (any-hit on): transmittance bytes[d*nc + i], 255 = 1; null = bits
};
