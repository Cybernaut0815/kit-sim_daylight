// SPDX-License-Identifier: MIT
// OptiX device program, compiled to OptiX-IR and embedded. One launch index per (sensor i, 32-direction word w):
// bit k of bits[w*nc + i] = direction w*32+k is in the sensor's hemisphere and unoccluded.
#include <optix.h>
#include <cfloat>
#include "params.h"

extern "C" { __constant__ Params params; }

// Waechter & Binder 2019 (Ray Tracing Gems ch. 6): nudge the origin along the normal by a few ulps, scale-aware.
static __forceinline__ __device__ float3 offset_ray(float3 p, float3 n) {
  const float origin = 1.f / 32.f, floatScale = 1.f / 65536.f, intScale = 256.f;
  const int ox = int(intScale * n.x), oy = int(intScale * n.y), oz = int(intScale * n.z);
  const float px = __int_as_float(__float_as_int(p.x) + (p.x < 0.f ? -ox : ox));
  const float py = __int_as_float(__float_as_int(p.y) + (p.y < 0.f ? -oy : oy));
  const float pz = __int_as_float(__float_as_int(p.z) + (p.z < 0.f ? -oz : oz));
  return make_float3(fabsf(p.x) < origin ? p.x + floatScale * n.x : px,
                     fabsf(p.y) < origin ? p.y + floatScale * n.y : py,
                     fabsf(p.z) < origin ? p.z + floatScale * n.z : pz);
}

extern "C" __global__ void __raygen__bits() {
  const uint3 li = optixGetLaunchIndex();
  const unsigned i = li.x, w = li.y;
  const float3 n = params.N[i];
  const float3 o = offset_ray(params.P[i], n);
  unsigned word = 0;
  for (unsigned k = 0; k < 32; ++k) {
    const unsigned d = w * 32 + k;
    if (d >= params.nDir) break;
    const float3 dir = params.D[d];
    if (n.x * dir.x + n.y * dir.y + n.z * dir.z <= 0.f) continue;   // hemisphere cull
    // tmin 1e-5 m: ignore hits at the origin (sensor exactly in an occluder plane), as Radiance does; scenes are in metres
    optixTraverse(params.gas, o, dir, 1e-5f, FLT_MAX, 0.f, OptixVisibilityMask(1), params.rayFlags, 0, 1, 0);
    if (!optixHitObjectIsHit()) word |= 1u << k;
  }
  params.bits[w * params.nc + i] = word;
}

extern "C" __global__ void __miss__ms() {}         // never invoked (optixTraverse without shader invocation)
extern "C" __global__ void __closesthit__ch() {}   // ditto; the SBT needs the records
