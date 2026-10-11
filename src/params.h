// SPDX-License-Identifier: Apache-2.0 AND LicenseRef-Commons-Clause-1.0
// Launch parameters shared by the OptiX device program (trace_dev.cu) and the host (gpu.cu).
#pragma once
#include <optix_types.h>
#include <vector_types.h>

struct Params {
  OptixTraversableHandle gas;
  const float3* P;        // sensor origins of this launch (device, already offset by i0)
  const float3* N;        // sensor normals
  const float3* D;        // direction table
  unsigned nDir;
  unsigned nc;            // sensors in this launch
  unsigned rayFlags;      // OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | DISABLE_ANYHIT [| CULL_BACK_FACING_TRIANGLES]
  unsigned pad;
  unsigned* bits;         // word-major: bits[w*nc + i], bit k of word w = direction w*32+k visible
};
