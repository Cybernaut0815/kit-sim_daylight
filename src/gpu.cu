// SPDX-License-Identifier: Apache-2.0 AND LicenseRef-Commons-Clause-1.0
// OptiX host side (context, pipeline from the embedded OptiX-IR, SBT, compacted GAS, TDR-safe sub-launches of the
// visibility and the reflected-light pass) and the plain CUDA reduce / series kernels. Everything runs on the legacy
// default stream with synchronous copies.
// ponytail: one stream + synchronous D2H; double-buffer the series copy if it ever exceeds ~1 s per 1e6 sensors.
#include "internal.h"
#include "params.h"
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>
#include "trace_optixir.h"
#include <algorithm>
#include <cstring>

namespace daylight {

struct Gpu {
  OptixDeviceContext ctx = nullptr;
  OptixModule module = nullptr;
  OptixProgramGroup pg[5] = {};                     // raygen bits, bytes, bounce, miss, hitgroup
  OptixPipeline pipeline = nullptr;
  OptixShaderBindingTable sbt[3] = {};              // per raygen (kBits, kBytes, kBounce); the same miss and hit records
  char hdr[5][OPTIX_SBT_RECORD_HEADER_SIZE] = {};   // packed headers of pg (the hitgroup's for every slot's record)
  unsigned maxSbt = 0;                              // OPTIX_DEVICE_PROPERTY_LIMIT_MAX_SBT_RECORDS_PER_GAS
  OptixTraversableHandle gas = 0;
  void *dGas = nullptr, *dSbt = nullptr; size_t sbtCap = 0;
  Params* dParams = nullptr;
  float3 *dP = nullptr, *dN = nullptr, *dD = nullptr;
  size_t nSensors = 0, nDir = 0, nWords = 0;
  size_t i0 = 0;                            // first sensor of the last gpu_trace chunk
  bool bytes = false;                       // the last gpu_trace wrote transmittance bytes (any-hit on), not bits
  void* dVis = nullptr; size_t visCap = 0;  // its bits or bytes
  void* dDc = nullptr; size_t dcCap = 0;    // the last gpu_bounce's rows [nBins x dcN], sensors dcI0..
  size_t nBins = 0, dcN = 0, dcI0 = 0;
  void* dScratch = nullptr; size_t scratchCap = 0;   // reduce / series inputs
  void* dOut = nullptr; size_t outCap = 0;           // reduce / series outputs
  cudaEvent_t ev[2] = {};
  GpuStats stats;
};

enum { kBits, kBytes, kBounce };   // raygen programs: pg / sbt index

#define CU(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { err = std::string(#call) + ": " + cudaGetErrorString(e_); return false; } } while (0)
#define OX(call) do { OptixResult r_ = (call); if (r_ != OPTIX_SUCCESS) { err = std::string(#call) + ": " + optixGetErrorString(r_); return false; } } while (0)
#define OXL(call) do { OptixResult r_ = (call); if (r_ != OPTIX_SUCCESS) { err = std::string(#call) + ": " + optixGetErrorString(r_) + "\n" + log; return false; } } while (0)

static bool grow(void** p, size_t* cap, size_t bytes, std::string& err) {   // device buffer, grown as needed
  if (bytes <= *cap) return true;
  cudaFree(*p); *p = nullptr; *cap = 0;
  CU(cudaMalloc(p, bytes)); *cap = bytes;
  return true;
}
static bool upload(void** d, const void* h, size_t bytes, std::string& err) {   // fresh device copy of a host array
  cudaFree(*d); *d = nullptr;
  if (!bytes) return true;
  CU(cudaMalloc(d, bytes)); CU(cudaMemcpy(*d, h, bytes, cudaMemcpyHostToDevice));
  return true;
}

// ------------------------------------------------------------------ kernels
// Rounded per operation (no fma contraction) so the host's float dot product reproduces c bit for bit.
__device__ __forceinline__ float dotf(float3 a, float3 b) {
  return __fadd_rn(__fadd_rn(__fmul_rn(a.x, b.x), __fmul_rn(a.y, b.y)), __fmul_rn(a.z, b.z));
}

// k_reduce / k_series read visibility through an accessor V: mask(w, i) = the visible directions 32w..32w+31 of sensor
// i as bits, bit(d, i) = one of them, at(d, i) = the fraction of visible direction d that arrives. at() scales the first
// factor of each product, where the exact 1.0f of Bits leaves the rounding and fma contraction of the bit walk unchanged.
struct Bits {   // gpu_trace's word-major bits[w*nc + i]
  const unsigned* b; unsigned nc;
  __device__ __forceinline__ unsigned mask(unsigned w, unsigned i) const { return b[w * nc + i]; }
  __device__ __forceinline__ unsigned bit(unsigned d, unsigned i) const { return (b[(d >> 5) * nc + i] >> (d & 31)) & 1u; }
  __device__ __forceinline__ float at(unsigned, unsigned) const { return 1.f; }
};
struct Bytes {   // byte mode: transmittance bytes[d*nc + i], x255 (255 = 1.0f, 0 = culled or blocked)
  const unsigned char* b; unsigned nc, nDir;
  __device__ __forceinline__ unsigned mask(unsigned w, unsigned i) const {
    unsigned m = 0;
    for (unsigned k = 0; k < 32 && w * 32 + k < nDir; ++k) m |= unsigned(b[(w * 32 + k) * nc + i] != 0) << k;
    return m;
  }
  __device__ __forceinline__ unsigned bit(unsigned d, unsigned i) const { return b[d * nc + i] != 0; }
  __device__ __forceinline__ float at(unsigned d, unsigned i) const { return b[d * nc + i] / 255.f; }
};
struct Dc {   // gpu_bounce's rows c[b*nc + i]: every bin walked (no cosine), at() = the coefficient
  const float* c; unsigned nc, nBins;
  __device__ __forceinline__ unsigned mask(unsigned w, unsigned) const { return w * 32 + 32 <= nBins ? ~0u : (1u << (nBins - w * 32)) - 1; }
  __device__ __forceinline__ float at(unsigned b, unsigned i) const { return c[b * nc + i]; }
};

template <class V>
__global__ void k_reduce(unsigned nc, unsigned nWords, V vis, const float3* N, const float3* D,
                         const float* W, int useCos, double* out) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nc) return;
  const float3 n = N[i];
  double acc = 0;
  for (unsigned w = 0; w < nWords; ++w) {
    unsigned m = vis.mask(w, i);
    while (m) {
      const unsigned d = w * 32 + (__ffs(m) - 1); m &= m - 1;
      acc += (double)((useCos ? fmaxf(dotf(n, D[d]), 0.f) : 1.f) * vis.at(d, i)) * W[d];
    }
  }
  out[i] = acc;
}

// E[i*T + t] = sum_{sky bits} max(0,cos)*wsky*S[row,t] + G*g[t] + sum_{sun bits} max(0,cos)*sunW at sunStep.
// t in tiles of 32 held in registers; the bit walk is repeated per tile. ponytail: S read through L1 (broadcast),
// shared-memory tiling if the series pass ever dominates.
template <class V>
__global__ void k_series(unsigned nc, V vis, const float3* N, const float3* D, unsigned nSky,
                         const int* row, const float* wsky, const float* S, int T, const float* g, const float* G,
                         unsigned sun0, unsigned nSun, const int* sunStep, const float3* sunVec, const float* sunW,
                         float* E) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nc) return;
  const float3 n = N[i];
  float* Ei = E + (size_t)i * T;
  for (int t0 = 0; t0 < T; t0 += 32) {
    float e[32];
#pragma unroll
    for (int tt = 0; tt < 32; ++tt) e[tt] = 0.f;
    for (unsigned w = 0; w * 32 < nSky; ++w) {
      unsigned m = vis.mask(w, i);
      while (m) {
        const unsigned d = w * 32 + (__ffs(m) - 1); m &= m - 1;
        if (d >= nSky) break;
        const float c = fmaxf(dotf(n, D[d]), 0.f) * vis.at(d, i) * wsky[d];
        const float* s = S + (size_t)row[d] * T + t0;
#pragma unroll
        for (int tt = 0; tt < 32; ++tt) if (t0 + tt < T) e[tt] += c * s[tt];
      }
    }
    for (int tt = 0; tt < 32 && t0 + tt < T; ++tt) Ei[t0 + tt] = e[tt] + G[i] * g[t0 + tt];
  }
  for (unsigned j = 0; j < nSun; ++j) {
    const unsigned d = sun0 + j;
    if (vis.bit(d, i)) Ei[sunStep[j]] += fmaxf(dotf(n, sunVec[j]), 0.f) * vis.at(d, i) * sunW[j];
  }
}

// E[i*T + t] += sum_b dc(b, i) * S[b*T + t]: reflected light per step, t in tiles of 32 as k_series.
__global__ void k_series_dc(unsigned nc, Dc dc, const float* S, int T, float* E) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nc) return;
  float* Ei = E + (size_t)i * T;
  for (int t0 = 0; t0 < T; t0 += 32) {
    float e[32];
#pragma unroll
    for (int tt = 0; tt < 32; ++tt) e[tt] = 0.f;
    for (unsigned b = 0; b < dc.nBins; ++b) {
      const float c = dc.at(b, i);
      if (c == 0.f) continue;
      const float* s = S + (size_t)b * T + t0;
#pragma unroll
      for (int tt = 0; tt < 32; ++tt) if (t0 + tt < T) e[tt] += c * s[tt];
    }
    for (int tt = 0; tt < 32 && t0 + tt < T; ++tt) Ei[t0 + tt] += e[tt];
  }
}

// ------------------------------------------------------------------ init / destroy
static bool init(Gpu& g, std::string& err) {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) { err = "no CUDA device"; return false; }
  CU(cudaFree(0));   // creates the primary context (device 0 or CUDA_VISIBLE_DEVICES) before optixInit
  const OptixResult r = optixInit();
  if (r != OPTIX_SUCCESS) { err = std::string("optixInit failed (driver too old or nvoptix.dll missing): ") + optixGetErrorString(r); return false; }
  OptixDeviceContextOptions co = {};
  OX(optixDeviceContextCreate(nullptr, &co, &g.ctx));
  OX(optixDeviceContextGetProperty(g.ctx, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_SBT_RECORDS_PER_GAS, &g.maxSbt, sizeof g.maxSbt));

  char log[4096] = {}; size_t logSize = sizeof log;
  OptixModuleCompileOptions mo = {};
  mo.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3; mo.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
  OptixPipelineCompileOptions po = {};
  po.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
  po.usesPrimitiveTypeFlags = static_cast<unsigned>(OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);
  po.numPayloadValues = 1; po.numAttributeValues = 2;   // payload 0: transmittance along the ray (any-hit)
  po.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
  po.pipelineLaunchParamsVariableName = "params";
  OXL(optixModuleCreate(g.ctx, &mo, &po, reinterpret_cast<const char*>(trace_optixir), trace_optixir_len, log, &logSize, &g.module));

  OptixProgramGroupDesc pd[5] = {};
  const char* raygen[3] = {"__raygen__bits", "__raygen__bytes", "__raygen__bounce"};
  for (int k = 0; k < 3; ++k) { pd[k].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN; pd[k].raygen.module = g.module; pd[k].raygen.entryFunctionName = raygen[k]; }
  pd[3].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;     pd[3].miss.module = g.module;       pd[3].miss.entryFunctionName = "__miss__ms";
  pd[4].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP; pd[4].hitgroup.moduleCH = g.module; pd[4].hitgroup.entryFunctionNameCH = "__closesthit__ch";
  pd[4].hitgroup.moduleAH = g.module; pd[4].hitgroup.entryFunctionNameAH = "__anyhit__tr";
  OptixProgramGroupOptions pgo = {};
  logSize = sizeof log;
  OXL(optixProgramGroupCreate(g.ctx, pd, 5, &pgo, log, &logSize, g.pg));

  OptixPipelineLinkOptions lo = {};
  lo.maxTraceDepth = 1;
  logSize = sizeof log;
  OXL(optixPipelineCreate(g.ctx, &po, &lo, g.pg, 5, log, &logSize, &g.pipeline));
  OptixStackSizes ss = {};
  for (OptixProgramGroup p : g.pg) OX(optixUtilAccumulateStackSizes(p, &ss, g.pipeline));
  unsigned trav = 0, state = 0, cont = 0;
  OX(optixUtilComputeStackSizes(&ss, 1, 0, 0, &trav, &state, &cont));
  OX(optixPipelineSetStackSize(g.pipeline, trav, state, cont, 1));

  for (int k = 0; k < 5; ++k) OX(optixSbtRecordPackHeader(g.pg[k], g.hdr[k]));
  CU(cudaMalloc(reinterpret_cast<void**>(&g.dParams), sizeof(Params)));
  CU(cudaEventCreate(&g.ev[0])); CU(cudaEventCreate(&g.ev[1]));
  return gpu_set_slots(g, {Slot{}}, err);   // one opaque slot until a scene brings its own
}

Gpu* gpu_init(std::string& err) {
  Gpu* g = new Gpu;
  if (init(*g, err)) return g;
  gpu_destroy(g);
  return nullptr;
}

void gpu_destroy(Gpu* g) {
  if (!g) return;
  if (g->pipeline) optixPipelineDestroy(g->pipeline);
  for (OptixProgramGroup p : g->pg) if (p) optixProgramGroupDestroy(p);
  if (g->module) optixModuleDestroy(g->module);
  if (g->ctx) optixDeviceContextDestroy(g->ctx);
  for (void* p : {g->dGas, g->dSbt, (void*)g->dParams, (void*)g->dP, (void*)g->dN, (void*)g->dD, g->dVis, g->dDc, g->dScratch, g->dOut}) cudaFree(p);
  for (cudaEvent_t e : g->ev) if (e) cudaEventDestroy(e);
  delete g;
}

// ------------------------------------------------------------------ scene
// tmp[0..4] = verts, indices, build temp, uncompacted output, slot per triangle; the caller frees whatever is left.
// Per-triangle SBT offsets only when a slot transmits or perSlot: they grow the GAS ~18 % and cost ~6 % GRays/s
// (--bench), and with every slot opaque (and, for the bounce pass, of one rho) one record behaves the same.
static bool build(Gpu& g, const Tris& t, const std::vector<Slot>& slots, bool perSlot, void** tmp, std::string& err) {
  if (t.slot.size() != t.i.size() || std::any_of(t.slot.begin(), t.slot.end(), [&](uint32_t s) { return s >= slots.size(); })) {
    err = "gpu_build: a triangle's slot has no Slot"; return false;
  }
  const bool slotted = perSlot || std::any_of(slots.begin(), slots.end(), [](const Slot& s) { return s.tau > 0; });
  if (slotted && slots.size() > g.maxSbt) { err = std::to_string(slots.size()) + " optics slots exceed the GPU's " + std::to_string(g.maxSbt) + " per scene"; return false; }
  if (!upload(&tmp[0], t.v.data(), t.v.size() * sizeof(float3), err) || !upload(&tmp[1], t.i.data(), t.i.size() * sizeof(uint3), err) ||
      (slotted && !upload(&tmp[4], t.slot.data(), t.slot.size() * sizeof(uint32_t), err))) return false;
  std::vector<unsigned> flags(slotted ? slots.size() : 1, OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT);   // any-hit only where light passes,
  for (size_t s = 0; slotted && s < slots.size(); ++s)                                          // once per crossing
    if (slots[s].tau > 0) flags[s] = OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL;
  const CUdeviceptr vb = reinterpret_cast<CUdeviceptr>(tmp[0]);
  OptixBuildInput bi = {};
  bi.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
  bi.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
  bi.triangleArray.vertexStrideInBytes = sizeof(float3);
  bi.triangleArray.numVertices = static_cast<unsigned>(t.v.size());
  bi.triangleArray.vertexBuffers = &vb;
  bi.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
  bi.triangleArray.indexStrideInBytes = sizeof(uint3);
  bi.triangleArray.numIndexTriplets = static_cast<unsigned>(t.i.size());
  bi.triangleArray.indexBuffer = reinterpret_cast<CUdeviceptr>(tmp[1]);
  bi.triangleArray.flags = flags.data();
  bi.triangleArray.numSbtRecords = static_cast<unsigned>(flags.size());
  if (slotted) {
    bi.triangleArray.sbtIndexOffsetBuffer = reinterpret_cast<CUdeviceptr>(tmp[4]);
    bi.triangleArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
    bi.triangleArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);
  }
  OptixAccelBuildOptions bo = {};
  bo.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
  bo.operation = OPTIX_BUILD_OPERATION_BUILD;
  OptixAccelBufferSizes sz = {};
  OX(optixAccelComputeMemoryUsage(g.ctx, &bo, &bi, 1, &sz));
  const size_t outBytes = (sz.outputSizeInBytes + 7) & ~size_t(7);
  CU(cudaMalloc(&tmp[2], sz.tempSizeInBytes));
  CU(cudaMalloc(&tmp[3], outBytes + sizeof(size_t)));
  OptixAccelEmitDesc emit = {};
  emit.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
  emit.result = reinterpret_cast<CUdeviceptr>(tmp[3]) + outBytes;
  CU(cudaEventRecord(g.ev[0]));
  OX(optixAccelBuild(g.ctx, 0, &bo, &bi, 1, reinterpret_cast<CUdeviceptr>(tmp[2]), sz.tempSizeInBytes,
                     reinterpret_cast<CUdeviceptr>(tmp[3]), sz.outputSizeInBytes, &g.gas, &emit, 1));
  size_t compacted = 0;
  CU(cudaMemcpy(&compacted, reinterpret_cast<void*>(emit.result), sizeof compacted, cudaMemcpyDeviceToHost));
  for (int k : {0, 1, 2, 4}) { cudaFree(tmp[k]); tmp[k] = nullptr; }   // inputs + temp freed before compaction
  if (compacted < sz.outputSizeInBytes) {
    CU(cudaMalloc(&g.dGas, compacted));
    OX(optixAccelCompact(g.ctx, 0, g.gas, reinterpret_cast<CUdeviceptr>(g.dGas), compacted, &g.gas));
    g.stats.gasBytes = compacted;
  } else {
    g.dGas = tmp[3]; tmp[3] = nullptr;
    g.stats.gasBytes = sz.outputSizeInBytes;
  }
  CU(cudaEventRecord(g.ev[1])); CU(cudaEventSynchronize(g.ev[1]));
  float ms = 0; CU(cudaEventElapsedTime(&ms, g.ev[0], g.ev[1]));
  g.stats.buildMs += ms;
  return true;
}

bool gpu_build(Gpu& g, const Tris& t, const std::vector<Slot>& slots, std::string& err, bool perSlot) {
  cudaFree(g.dGas); g.dGas = nullptr; g.gas = 0; g.stats.gasBytes = 0;
  if (t.i.empty()) return true;   // empty scene: traversable 0 = every ray misses
  void* tmp[5] = {};
  const bool ok = build(g, t, slots, perSlot, tmp, err);
  for (void* p : tmp) cudaFree(p);
  if (!ok) { cudaFree(g.dGas); g.dGas = nullptr; g.gas = 0; }
  return ok;
}

bool gpu_set_slots(Gpu& g, const std::vector<Slot>& slots, std::string& err) {
  struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) Rec { char header[OPTIX_SBT_RECORD_HEADER_SIZE]; Slot s; };
  std::vector<Rec> rec(4 + std::max<size_t>(1, slots.size()));   // the three raygens, miss, one hit record per slot
  for (size_t k = 0; k < rec.size(); ++k) {
    std::memcpy(rec[k].header, g.hdr[std::min<size_t>(k, 4)], sizeof rec[k].header);
    rec[k].s = k >= 4 && k - 4 < slots.size() ? slots[k - 4] : Slot{};
  }
  if (!grow(&g.dSbt, &g.sbtCap, rec.size() * sizeof(Rec), err)) return false;
  CU(cudaMemcpy(g.dSbt, rec.data(), rec.size() * sizeof(Rec), cudaMemcpyHostToDevice));
  const CUdeviceptr base = reinterpret_cast<CUdeviceptr>(g.dSbt);
  for (int k = 0; k < 3; ++k) {
    OptixShaderBindingTable& s = g.sbt[k];
    s.raygenRecord = base + k * sizeof(Rec);
    s.missRecordBase = base + 3 * sizeof(Rec);     s.missRecordStrideInBytes = sizeof(Rec);     s.missRecordCount = 1;
    s.hitgroupRecordBase = base + 4 * sizeof(Rec); s.hitgroupRecordStrideInBytes = sizeof(Rec); s.hitgroupRecordCount = unsigned(rec.size() - 4);
  }
  return true;
}

static_assert(sizeof(OptixTraversableHandle) == sizeof(unsigned long long), "GasSlot::gas holds an OptixTraversableHandle");
GasSlot gpu_detach(Gpu& g) {
  GasSlot s{g.dGas, g.gas, g.stats.gasBytes};
  g.dGas = nullptr; g.gas = 0; g.stats.gasBytes = 0;
  return s;
}
void gpu_attach(Gpu& g, GasSlot s) {
  cudaFree(g.dGas);
  g.dGas = s.dGas; g.gas = s.gas; g.stats.gasBytes = s.bytes;
}
void gpu_free(GasSlot& s) { cudaFree(s.dGas); s = GasSlot{}; }
size_t gpu_free_bytes() { size_t f = 0, t = 0; return cudaMemGetInfo(&f, &t) == cudaSuccess ? f : 0; }

bool gpu_set_scene(Gpu& g, const Sensors& s, const std::vector<float3>& dirs, std::string& err) {
  g.nSensors = s.p.size(); g.nDir = dirs.size(); g.nWords = (g.nDir + 31) / 32;
  return upload((void**)&g.dP, s.p.data(), s.p.size() * sizeof(float3), err) &&
         upload((void**)&g.dN, s.n.data(), s.n.size() * sizeof(float3), err) &&
         upload((void**)&g.dD, dirs.data(), dirs.size() * sizeof(float3), err);
}

// ------------------------------------------------------------------ trace / reduce / series
bool gpu_trace(Gpu& g, size_t i0, size_t nc, unsigned rayFlags, std::string& err) {
  if (i0 + nc > g.nSensors) { err = "gpu_trace: sensor range exceeds the scene"; return false; }
  g.bytes = !(rayFlags & OPTIX_RAY_FLAG_DISABLE_ANYHIT);   // any-hit runs: fractional transmittance needs bytes
  if (!grow(&g.dVis, &g.visCap, g.bytes ? g.nDir * nc : g.nWords * nc * sizeof(unsigned), err)) return false;
  g.i0 = i0;
  if (nc == 0 || g.nWords == 0) return true;
  // Each sub-launch is sized from the previous one's time, well under the 2 s Windows TDR: ~150 ms, first 4096 sensors,
  // at least 1024. With any-hit a ray can cost thousands of opaque ones (a call per crossing, no early exit near tau 1):
  // byte mode starts at 2^13 threads (2^18 rays, about a full GPU: a thread traces its 32 directions in turn, so a
  // smaller launch is hardly shorter), never goes below that, aims at 30 ms and at most doubles.
  // ponytail: the --bench boxes all at tau 1 take 0.29 s per 2^13 threads on the 2060, and sensors > ~60x costlier than
  // the previous launch's can pass the TDR; split a thread's directions over launches if that shows up
  const bool any = g.bytes;
  const double targetMs = any ? 30 : 150;
  const size_t nMin = any ? std::max<size_t>(1, (size_t(1) << 13) / g.nWords) : 1024;
  size_t done = 0, n = std::min(any ? nMin : 4096, nc);
  while (done < nc) {
    // params.nc is the row stride of the whole chunk; the sub-launch writes its slice bits[w*nc + done + i] / bytes[d*nc + done + i].
    unsigned* bits = g.bytes ? nullptr : static_cast<unsigned*>(g.dVis) + done;
    unsigned char* bytes = g.bytes ? static_cast<unsigned char*>(g.dVis) + done : nullptr;
    const Params p = {g.gas, g.dP + i0 + done, g.dN + i0 + done, g.dD, (unsigned)g.nDir, (unsigned)nc, rayFlags, 0, bits, bytes};
    CU(cudaMemcpy(g.dParams, &p, sizeof p, cudaMemcpyHostToDevice));
    CU(cudaEventRecord(g.ev[0]));
    OX(optixLaunch(g.pipeline, 0, reinterpret_cast<CUdeviceptr>(g.dParams), sizeof(Params), &g.sbt[g.bytes ? kBytes : kBits], (unsigned)n, (unsigned)g.nWords, 1));
    CU(cudaEventRecord(g.ev[1])); CU(cudaEventSynchronize(g.ev[1]));
    float ms = 0; CU(cudaEventElapsedTime(&ms, g.ev[0], g.ev[1]));
    g.stats.traceMs += ms; g.stats.rays += n * g.nDir; g.stats.launches++;
    done += n;
    const double want = double(n) * targetMs / std::max<double>(ms, 1e-3);
    n = std::min({std::max<size_t>(want > 1e12 ? size_t(1) << 40 : size_t(want), nMin), any ? 2 * n : nc, nc - done});
  }
  return true;
}

bool gpu_bounce(Gpu& g, size_t i0, size_t nc, unsigned rayFlags, const Bounce& b, std::string& err) {
  if (i0 + nc > g.nSensors) { err = "gpu_bounce: sensor range exceeds the scene"; return false; }
  const Patches pa = make_patches(b.mf);
  g.nBins = size_t(pa.P) + 1; g.dcN = nc; g.dcI0 = i0;
  if (!grow(&g.dDc, &g.dcCap, g.nBins * nc * sizeof(float), err)) return false;
  if (nc == 0) return true;
  CU(cudaMemset(g.dDc, 0, g.nBins * nc * sizeof(float)));
  // Every launch runs the whole chunk over the samples [done, done + n): a path costs a few closest hits, varying with
  // the scene and N, so n starts at 2^13 paths (1 to 64 samples: a thread's run in turn), aims at 100 ms, at most doubles.
  // ponytail: one thread per sensor keeps the row sums ordered without atomics, so small chunks (sky density 4: 29k
  // sensors) idle the GPU in launch tails where path costs vary widely (--bench's boxes, not a city); give each thread a
  // slice of a sensor's samples, with per-slice rows summed in order, if that matters
  const double rad = b.north * 3.14159265358979323846 / 180;
  Params p = {};
  p.gas = g.gas; p.P = g.dP + i0; p.N = g.dN + i0; p.nc = unsigned(nc); p.rayFlags = rayFlags;
  p.dc = static_cast<float*>(g.dDc); p.i0 = unsigned(i0); p.M = b.M; p.bounces = b.N; p.mf = unsigned(b.mf); p.nBins = unsigned(g.nBins);
  p.alpha = pa.alpha; p.cosN = float(std::cos(rad)); p.sinN = float(std::sin(rad));
  for (size_t done = 0, n = std::clamp<size_t>((size_t(1) << 13) / nc, 1, 64); done < b.M;) {
    if (b.cancel && b.cancel->load()) { err = "cancelled"; return false; }
    n = std::min<size_t>(n, b.M - done);
    p.s0 = unsigned(done); p.s1 = unsigned(done + n);
    CU(cudaMemcpy(g.dParams, &p, sizeof p, cudaMemcpyHostToDevice));
    CU(cudaEventRecord(g.ev[0]));
    OX(optixLaunch(g.pipeline, 0, reinterpret_cast<CUdeviceptr>(g.dParams), sizeof(Params), &g.sbt[kBounce], (unsigned)nc, 1, 1));
    CU(cudaEventRecord(g.ev[1])); CU(cudaEventSynchronize(g.ev[1]));
    float ms = 0; CU(cudaEventElapsedTime(&ms, g.ev[0], g.ev[1]));
    g.stats.bounceMs += ms; g.stats.paths += n * nc; g.stats.bounceLaunches++;
    done += n;
    const double want = double(n) * 100 / std::max<double>(ms, 1e-3);
    n = std::max<size_t>(1, std::min<size_t>(2 * n, want > 1e9 ? size_t(1e9) : size_t(want)));
  }
  return true;
}

bool gpu_reduce(Gpu& g, size_t nc, const float* W, bool useCos, double* out, std::string& err) {
  if (nc == 0) return true;
  if (!grow(&g.dScratch, &g.scratchCap, g.nDir * sizeof(float), err) || !grow(&g.dOut, &g.outCap, nc * sizeof(double), err)) return false;
  CU(cudaMemcpy(g.dScratch, W, g.nDir * sizeof(float), cudaMemcpyHostToDevice));
  const auto launch = [&](auto vis) {
    k_reduce<<<unsigned((nc + 255) / 256), 256>>>((unsigned)nc, (unsigned)g.nWords, vis, g.dN + g.i0, g.dD,
                                                   static_cast<const float*>(g.dScratch), useCos, static_cast<double*>(g.dOut));
  };
  if (g.bytes) launch(Bytes{static_cast<const unsigned char*>(g.dVis), (unsigned)nc, (unsigned)g.nDir});
  else launch(Bits{static_cast<const unsigned*>(g.dVis), (unsigned)nc});
  CU(cudaGetLastError());
  CU(cudaMemcpy(out, g.dOut, nc * sizeof(double), cudaMemcpyDeviceToHost));
  return true;
}

bool gpu_reduce_dc(Gpu& g, size_t nc, const float* W, double* out, std::string& err) {
  if (nc == 0) return true;
  if (nc != g.dcN) { err = "gpu_reduce_dc: no gpu_bounce rows for this chunk"; return false; }
  if (!grow(&g.dScratch, &g.scratchCap, g.nBins * sizeof(float), err) || !grow(&g.dOut, &g.outCap, nc * sizeof(double), err)) return false;
  CU(cudaMemcpy(g.dScratch, W, g.nBins * sizeof(float), cudaMemcpyHostToDevice));
  k_reduce<<<unsigned((nc + 255) / 256), 256>>>((unsigned)nc, unsigned((g.nBins + 31) / 32), Dc{static_cast<const float*>(g.dDc), (unsigned)nc, (unsigned)g.nBins},
                                                 g.dN + g.dcI0, g.dD, static_cast<const float*>(g.dScratch), 0, static_cast<double*>(g.dOut));
  CU(cudaGetLastError());
  CU(cudaMemcpy(out, g.dOut, nc * sizeof(double), cudaMemcpyDeviceToHost));
  return true;
}

bool gpu_series(Gpu& g, size_t nc, const SeriesIn& in, float* E, std::string& err) {
  if (nc == 0 || in.T <= 0) return true;
  if (in.Sdc && nc != g.dcN) { err = "gpu_series: no gpu_bounce rows for this chunk"; return false; }
  const size_t T = size_t(in.T);
  // one packed upload: row | wsky | S | g | G | sunStep | sunVec | sunW | Sdc (each 256-byte aligned)
  const void* src[9] = {in.row, in.wsky, in.S, in.g, in.G, in.sunStep, in.sunVec, in.sunW, in.Sdc};
  const size_t len[9] = {in.nSky * 4, in.nSky * 4, size_t(in.P) * T * 4, T * 4, nc * 4, in.nSun * 4, in.nSun * sizeof(float3), in.nSun * 4,
                         in.Sdc ? g.nBins * T * 4 : 0};
  size_t off[9], total = 0;
  for (int k = 0; k < 9; ++k) { off[k] = total; total += (len[k] + 255) & ~size_t(255); }
  std::vector<char> h(total);
  for (int k = 0; k < 9; ++k) if (len[k]) std::memcpy(h.data() + off[k], src[k], len[k]);
  if (!grow(&g.dScratch, &g.scratchCap, total, err) || !grow(&g.dOut, &g.outCap, nc * T * sizeof(float), err)) return false;
  CU(cudaMemcpy(g.dScratch, h.data(), total, cudaMemcpyHostToDevice));
  const char* b = static_cast<const char*>(g.dScratch);
  const auto launch = [&](auto vis) {
    k_series<<<unsigned((nc + 255) / 256), 256>>>((unsigned)nc, vis, g.dN + g.i0, g.dD, in.nSky,
        reinterpret_cast<const int*>(b + off[0]), reinterpret_cast<const float*>(b + off[1]), reinterpret_cast<const float*>(b + off[2]),
        in.T, reinterpret_cast<const float*>(b + off[3]), reinterpret_cast<const float*>(b + off[4]), in.sun0, in.nSun,
        reinterpret_cast<const int*>(b + off[5]), reinterpret_cast<const float3*>(b + off[6]), reinterpret_cast<const float*>(b + off[7]),
        static_cast<float*>(g.dOut));
  };
  if (g.bytes) launch(Bytes{static_cast<const unsigned char*>(g.dVis), (unsigned)nc, (unsigned)g.nDir});
  else launch(Bits{static_cast<const unsigned*>(g.dVis), (unsigned)nc});
  if (in.Sdc)
    k_series_dc<<<unsigned((nc + 255) / 256), 256>>>((unsigned)nc, Dc{static_cast<const float*>(g.dDc), (unsigned)nc, (unsigned)g.nBins},
                                                     reinterpret_cast<const float*>(b + off[8]), in.T, static_cast<float*>(g.dOut));
  CU(cudaGetLastError());
  CU(cudaMemcpy(E, g.dOut, nc * T * sizeof(float), cudaMemcpyDeviceToHost));
  return true;
}

bool gpu_download_bits(Gpu& g, size_t nc, std::vector<uint32_t>& out, std::string& err) {
  if (g.bytes) { err = "gpu_download_bits: the last trace wrote bytes"; return false; }
  out.resize(g.nWords * nc);
  if (!out.empty()) CU(cudaMemcpy(out.data(), g.dVis, out.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  return true;
}

bool gpu_download_bytes(Gpu& g, size_t nc, std::vector<uint8_t>& out, std::string& err) {
  if (!g.bytes) { err = "gpu_download_bytes: the last trace wrote bits"; return false; }
  out.resize(g.nDir * nc);
  if (!out.empty()) CU(cudaMemcpy(out.data(), g.dVis, out.size(), cudaMemcpyDeviceToHost));
  return true;
}

bool gpu_download_dc(Gpu& g, size_t nc, std::vector<float>& out, std::string& err) {
  if (nc != g.dcN) { err = "gpu_download_dc: no gpu_bounce rows for this chunk"; return false; }
  out.resize(g.nBins * nc);
  if (!out.empty()) CU(cudaMemcpy(out.data(), g.dDc, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
  return true;
}

GpuStats gpu_stats(const Gpu& g) { return g.stats; }

unsigned gpu_ray_flags(bool cullBackfaces, bool translucent) {
  return OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | (translucent ? 0u : OPTIX_RAY_FLAG_DISABLE_ANYHIT) |
         (cullBackfaces ? OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES : 0u);
}

unsigned gpu_bounce_flags(bool cullBackfaces) {
  return OPTIX_RAY_FLAG_DISABLE_ANYHIT | (cullBackfaces ? OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES : 0u);
}

}  // namespace daylight
