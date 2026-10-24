// SPDX-License-Identifier: Apache-2.0 AND LicenseRef-Commons-Clause-1.0
// OptiX device program, compiled to OptiX-IR and embedded. One launch index per (sensor i, 32-direction word w):
// bit k of bits[w*nc + i] = direction w*32+k is in the sensor's hemisphere and unoccluded; in byte mode (a slot
// transmits) bytes[d*nc + i] = the transmittance along direction d instead (x255, 0 = culled or blocked).
#include <optix.h>
#include <cfloat>
#include "glass.h"
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
    unsigned t = 0;   // payload 0: transmittance along the ray (float bits); stays 0 when culled or blocked
    if (n.x * dir.x + n.y * dir.y + n.z * dir.z > 0.f) {   // hemisphere cull
      t = __float_as_uint(1.f);
      // tmin 1e-5 m: ignore hits at the origin (sensor exactly in an occluder plane), as Radiance does; scenes are in metres
      optixTraverse(params.gas, o, dir, 1e-5f, FLT_MAX, 0.f, OptixVisibilityMask(1), params.rayFlags, 0, 1, 0, t);
      if (optixHitObjectIsHit()) t = 0;
    }
    if (params.bytes) params.bytes[d * params.nc + i] = (unsigned char)__float2uint_rn(__uint_as_float(t) * 255.f);
    else word |= unsigned(t != 0) << k;
  }
  if (!params.bytes) params.bits[w * params.nc + i] = word;
}

// A translucent slot's triangle (REQUIRE_SINGLE_ANYHIT_CALL, so once per crossing): payload 0 *= what passes, tau
// (porous, or glass with --fresnel off), Radiance glass T(cos) of the hit triangle, or 1 on a thin solid's exit face.
extern "C" __global__ void __anyhit__tr() {
  const Slot s = *reinterpret_cast<const Slot*>(optixGetSbtDataPointer());
  float T = s.tau;
  if ((s.flags & kSlotThin) && optixIsFrontFaceHit() == ((s.flags & kSlotInward) != 0)) T = 1.f;   // entry = front face unless inward
  else if (s.flags & kSlotGlass) {
    float3 v[3];
    optixGetTriangleVertexData(v);   // the current hit: no ALLOW_RANDOM_VERTEX_ACCESS needed (OptiX 9)
    const float ax = v[1].x - v[0].x, ay = v[1].y - v[0].y, az = v[1].z - v[0].z;
    const float bx = v[2].x - v[0].x, by = v[2].y - v[0].y, bz = v[2].z - v[0].z;
    const float nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx;
    const float3 d = optixGetWorldRayDirection();   // unit (direction table); one GAS: object = world
    float R;
    glass_TR(fabsf(nx * d.x + ny * d.y + nz * d.z) * rsqrtf(nx * nx + ny * ny + nz * nz), s.tn, T, R);
  }
  T *= __uint_as_float(optixGetPayload_0());
  optixSetPayload_0(__float_as_uint(T));
  if (!(T >= 1.f / 510.f)) optixTerminateRay();   // would round to byte 0 (or NaN): blocked
  optixIgnoreIntersection();
}

extern "C" __global__ void __miss__ms() {}         // never invoked (optixTraverse without shader invocation)
extern "C" __global__ void __closesthit__ch() {}   // ditto; the SBT needs the records
