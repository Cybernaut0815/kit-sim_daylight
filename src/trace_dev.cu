// SPDX-License-Identifier: Apache-2.0 AND LicenseRef-Commons-Clause-1.0
// OptiX device program, compiled to OptiX-IR and embedded. __raygen__bits: one launch index per (sensor i, 32-direction
// word w), bit k of bits[w*nc + i] = direction w*32+k is in the sensor's hemisphere and unoccluded; __raygen__bytes (a
// slot transmits): the same launch, bytes[d*nc + i] = the transmittance along direction d instead (x255, 0 = culled or
// blocked); __raygen__bounce: one launch index per sensor, the paths of reflected light (--bounces) binned into
// dc[b*nc + i].
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

static __forceinline__ __device__ float3 operator+(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
static __forceinline__ __device__ float3 operator-(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
static __forceinline__ __device__ float3 operator*(float s, float3 a) { return make_float3(s * a.x, s * a.y, s * a.z); }
static __forceinline__ __device__ float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

// The triangle of the current hit (one GAS: object = world): hit point p, unit normal n on the side the ray came from,
// ext = its largest edge coordinate from vertex 0, and lift = 2^-20 x ext along n (2 mm at 2 km), what new origins
// leave it by. The hardware intersects relative to the ray origin, ~2^-25 x that extent off: without the lift a tenth
// of the rays leaving a 2 km quad near the origin met it, or its coplanar neighbour, again. False when degenerate.
struct Hit { float3 p, n, lift; float ext; bool front; };
static __forceinline__ __device__ bool hit_triangle(Hit& h) {
  float3 tv[3];
  optixHitObjectGetTriangleVertexData(tv);
  const float2 bc = optixHitObjectGetTriangleBarycentrics();
  const float3 e1 = tv[1] - tv[0], e2 = tv[2] - tv[0];
  const float3 ng = make_float3(e1.y * e2.z - e1.z * e2.y, e1.z * e2.x - e1.x * e2.z, e1.x * e2.y - e1.y * e2.x);
  const float l2 = dot(ng, ng);
  if (!(l2 > 0.f)) return false;
  h.front = optixIsFrontFaceHit(optixHitObjectGetHitKind());
  h.n = (h.front ? rsqrtf(l2) : -rsqrtf(l2)) * ng;
  h.p = tv[0] + bc.x * e1 + bc.y * e2;
  h.ext = fmaxf(fmaxf(fmaxf(fabsf(e1.x), fabsf(e1.y)), fmaxf(fabsf(e1.z), fabsf(e2.x))), fmaxf(fabsf(e2.y), fabsf(e2.z)));
  h.lift = (h.ext * 9.5367432e-7f) * h.n;
  return true;
}
// The hit's plane lies within 2^-22 x its triangle's extent of the ray origin (8x the hardware's error): from a sensor,
// its own plane (a large occluder's, near the origin), which its rays meet only through that error. Such hits pass, as
// tmin means them to.
static __forceinline__ __device__ bool own_plane(const Hit& h, float3 d) {
  return optixHitObjectGetRayTmax() * fabsf(dot(d, h.n)) < h.ext * 2.3841858e-7f;
}
static __forceinline__ __device__ float3 through(const Hit& h) { return offset_ray(h.p, -1.f * h.n) - h.lift; }   // the origin past it

// A visibility ray's hit lies under 1/32 m above the plane of the sensor (normal n), as its own plane does for triangles
// up to 2^17 m; higher hits cost nothing more. Read from the hit object: one more value live across the visibility
// traversals (a mask, the direction) cost 1-2 % on --bench.
static __forceinline__ __device__ bool low_hit(float3 n) {
  return optixHitObjectGetRayTmax() * dot(n, optixHitObjectGetWorldRayDirection()) < 1.f / 32.f;
}
// ... then the ray is walked again by closest hits (the first hit reported need not be the nearest), on its own line,
// past the sensor's own plane(s): the tmin the rest of the ray goes on from when nothing comes next or, with
// translucent, a translucent slot; 0 = blocked.
static __forceinline__ __device__ float past_own_plane(float3 o, float3 d, bool translucent) {
  const unsigned flags = (params.rayFlags & ~OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT) | OPTIX_RAY_FLAG_DISABLE_ANYHIT;
  float tmin = 1e-5f;
  for (unsigned seg = 0; seg < 4; ++seg) {
    optixTraverse(params.gas, o, d, tmin, FLT_MAX, 0.f, OptixVisibilityMask(1), flags, 0, 1, 0);
    if (!optixHitObjectIsHit()) return tmin;
    Hit h;
    if (!hit_triangle(h)) return 0.f;
    if (!own_plane(h, d)) return translucent && reinterpret_cast<const Slot*>(optixHitObjectGetSbtDataPointer())->tau > 0.f ? tmin : 0.f;
    tmin = __uint_as_float(__float_as_uint(optixHitObjectGetRayTmax()) + 1);   // the next float past it
  }
  return 0.f;
}

// Every slot opaque: no payload, no any-hit (the hot path of default runs; byte mode has its own raygen).
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
    else if (low_hit(n) && past_own_plane(o, params.D[d], false) > 0.f) word |= 1u << k;   // D reloaded: see low_hit
  }
  params.bits[w * params.nc + i] = word;
}

// Byte mode (a slot transmits): payload 0 = the transmittance along the ray (float bits), multiplied by __anyhit__tr at
// every translucent crossing; stays 0 when culled or blocked. Past the sensor's own plane the ray is traced on again.
// ponytail: a translucent own plane still passes its tau where float error meets it (any-hit, no hit reported); test
// own_plane in __anyhit__tr (a vertex fetch per porous crossing) if sensors lie on large translucent sheets
extern "C" __global__ void __raygen__bytes() {
  const uint3 li = optixGetLaunchIndex();
  const unsigned i = li.x, w = li.y;
  const float3 n = params.N[i];
  const float3 o = offset_ray(params.P[i], n);
  for (unsigned k = 0; k < 32; ++k) {
    const unsigned d = w * 32 + k;
    if (d >= params.nDir) break;
    const float3 dir = params.D[d];
    unsigned t = 0;
    if (n.x * dir.x + n.y * dir.y + n.z * dir.z > 0.f) {   // hemisphere cull
      t = __float_as_uint(1.f);
      optixTraverse(params.gas, o, dir, 1e-5f, FLT_MAX, 0.f, OptixVisibilityMask(1), params.rayFlags, 0, 1, 0, t);
      if (optixHitObjectIsHit()) {
        t = 0;
        if (low_hit(n)) {
          const float3 dd = params.D[d];   // reloaded: see low_hit
          const float t0 = past_own_plane(o, dd, true);
          if (t0 > 0.f) {
            t = __float_as_uint(1.f);
            optixTraverse(params.gas, o, dd, t0, FLT_MAX, 0.f, OptixVisibilityMask(1), params.rayFlags, 0, 1, 0, t);
            if (optixHitObjectIsHit()) t = 0;
          }
        }
      }
    }
    params.bytes[d * params.nc + i] = (unsigned char)__float2uint_rn(__uint_as_float(t) * 255.f);
  }
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

// Jarzynski & Olano 2020 (JCGT 9(3), "Hash Functions for GPU Rendering"): counter-based, 3 x 32 random bits per call.
static __forceinline__ __device__ uint3 pcg3d(unsigned x, unsigned y, unsigned z) {
  x = x * 1664525u + 1013904223u; y = y * 1664525u + 1013904223u; z = z * 1664525u + 1013904223u;
  x += y * z; y += z * x; z += x * y;
  x ^= x >> 16; y ^= y >> 16; z ^= z >> 16;
  x += y * z; y += z * x; z += x * y;
  return make_uint3(x, y, z);
}
static __forceinline__ __device__ float u01(unsigned v) { return float(v >> 8) * (1.f / 16777216.f); }   // [0, 1)

// Cosine-weighted direction about unit n: sin^2 theta = u, azimuth 2 pi v (basis: Duff et al. 2017, JCGT 6(1)).
static __forceinline__ __device__ float3 cosine_dir(float3 n, float u, float v) {
  const float sg = copysignf(1.f, n.z), a = -1.f / (sg + n.z), b = n.x * n.y * a;
  const float3 t = make_float3(1.f + sg * n.x * n.x * a, sg * b, -sg * n.x), w = make_float3(b, sg + n.y * n.y * a, -n.y);
  float s, c;
  sincospif(2.f * v, &s, &c);
  const float r = sqrtf(u);
  return (r * c) * t + (r * s) * w + sqrtf(fmaxf(0.f, 1.f - u)) * n;
}

__constant__ unsigned kTnaz[7] = {30, 30, 24, 24, 18, 12, 6};   // Tregenza patches per row (x mf rows of x mf patches)

// Bin of an escaping direction (model frame): 0 below the horizon (the ground), else 1 + the Reinhart patch (gendaymtx
// order) of the direction turned back to true north, as reinhart.cal's rbin.
static __forceinline__ __device__ unsigned sky_bin(float3 d) {
  if (!(d.z >= 0.f)) return 0;
  const unsigned mf = params.mf, r = min(unsigned(asinf(fminf(d.z, 1.f)) / params.alpha), 7 * mf);
  if (r == 7 * mf) return params.nBins - 1;   // zenith cap
  const unsigned n = kTnaz[r / mf] * mf;
  unsigned b = 1 + (r % mf) * n;
  for (unsigned q = 0; q < r / mf; ++q) b += kTnaz[q] * mf * mf;
  float az = atan2f(d.x * params.cosN + d.y * params.sinN, d.y * params.cosN - d.x * params.sinN);   // clockwise from north
  if (az < 0.f) az += 6.28318531f;
  return b + unsigned(az * (float(n) / 6.28318531f) + .5f) % n;
}

// Reflected light, one launch index per sensor i of the chunk, its samples [s0, s1) of M. Sample s leaves the sensor
// cosine-weighted (Hammersley point s, Cranley-Patterson rotated by a hash of the global sensor index) and goes from
// closest hit to closest hit: the sensor's own plane and a thin solid's exit face pass it unchanged; elsewhere it is
// transmitted with probability T / (T + R) (translucent slots: porous tau, Radiance glass T(cos)), weight x (T + R), else
// reflected, by the mirror on glass (also opaque glass: T 0), else diffusely (rho; these count toward params.bounces,
// Russian roulette from the second on: survival = the weight). A path that reflected and escapes adds weight x pi / M
// to its bin; transmit-only paths are the direct pass's.
// Random numbers: pcg3d(global sensor, sample, event), so rows do not depend on how sensors or samples are launched.
// Thread i alone adds to row i, in sample order: no atomics, deterministic.
// ponytail: rows sum in float; a bin fed by all of M = 65536 paths rounds by ~5e-4 (a few hundred paths per bin, the
// usual case, by ~1e-6); double rows (2x the chunk memory) if single-bin sensors need better
extern "C" __global__ void __raygen__bounce() {
  const unsigned i = optixGetLaunchIndex().x, gi = params.i0 + i;
  const float3 n = params.N[i], o0 = offset_ray(params.P[i], n);
  const uint3 rot = pcg3d(gi, ~0u, ~0u);
  const float r0 = u01(rot.x), r1 = u01(rot.y), weight = 3.14159265f / float(params.M);
  float* row = params.dc + i;
  for (unsigned s = params.s0; s < params.s1; ++s) {
    const float u = float(s) / float(params.M) + r0, v = u01(__brev(s)) + r1;
    float3 o = o0, d = cosine_dir(n, u - floorf(u), v - floorf(v));
    float w = 1.f;
    unsigned diffuse = 0, k = 0;   // diffuse reflections, random draws (events)
    bool reflected = false;
    for (unsigned seg = 0; seg < params.bounces + 16; ++seg) {
      optixTraverse(params.gas, o, d, 1e-5f, FLT_MAX, 0.f, OptixVisibilityMask(1), params.rayFlags, 0, 1, 0);
      if (!optixHitObjectIsHit()) { if (reflected) row[sky_bin(d) * params.nc] += w * weight; break; }
      const Slot sl = *reinterpret_cast<const Slot*>(optixHitObjectGetSbtDataPointer());
      Hit h;
      if (!hit_triangle(h)) break;
      if (seg == 0 && own_plane(h, d)) { o = through(h); continue; }   // the first segment passes the sensor's own plane
      if ((sl.flags & (kSlotThin | kSlotTwin)) == kSlotThin && h.front == ((sl.flags & kSlotInward) != 0)) { o = through(h); continue; }   // exit face
      const bool mirror = (sl.flags & kSlotGlass) != 0;
      float T = sl.tau, R = sl.rho;   // opaque: tau 0 (every slot with --transmission off)
      if (mirror) { glass_TR(fabsf(dot(d, h.n)), sl.tn, T, R); if (!(sl.tau > 0.f)) T = 0.f; }   // off: mirrors R, absorbs T
      else if (diffuse == params.bounces) R = 0.f;   // no diffuse reflection left: transmitted or absorbed
      if (!(T + R > 0.f)) break;
      const uint3 r = pcg3d(gi, s, k++);
      w *= T + R;
      if (u01(r.x) * (T + R) < T) { o = through(h); continue; }   // transmitted
      reflected = true;
      if (mirror) d = d - (2.f * dot(d, h.n)) * h.n;
      else {
        d = cosine_dir(h.n, u01(r.y), u01(r.z));
        if (++diffuse >= 2) {   // Russian roulette, survival = w (<= 1)
          if (u01(pcg3d(gi, s, k++).x) >= w) break;
          w = 1.f;
        }
      }
      o = offset_ray(h.p, h.n) + h.lift;
    }
  }
}

extern "C" __global__ void __miss__ms() {}         // never invoked (optixTraverse without shader invocation)
extern "C" __global__ void __closesthit__ch() {}   // ditto; the SBT needs the records
