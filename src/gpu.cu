// SPDX-License-Identifier: Apache-2.0 AND LicenseRef-Commons-Clause-1.0
// OptiX host side (context, pipeline from the embedded OptiX-IR, SBT, compacted GAS, TDR-safe sub-launches) and the
// plain CUDA reduce / series kernels. Everything runs on the legacy default stream with synchronous copies.
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
  OptixProgramGroup pg[3] = {};
  OptixPipeline pipeline = nullptr;
  OptixShaderBindingTable sbt = {};
  OptixTraversableHandle gas = 0;
  void *dGas = nullptr, *dSbt = nullptr;
  Params* dParams = nullptr;
  float3 *dP = nullptr, *dN = nullptr, *dD = nullptr;
  size_t nSensors = 0, nDir = 0, nWords = 0;
  size_t i0 = 0;                            // first sensor of the last gpu_trace chunk
  unsigned* dBits = nullptr; size_t bitsCap = 0;
  void* dScratch = nullptr; size_t scratchCap = 0;   // reduce / series inputs
  void* dOut = nullptr; size_t outCap = 0;           // reduce / series outputs
  cudaEvent_t ev[2] = {};
  GpuStats stats;
};

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

__global__ void k_reduce(unsigned nc, unsigned nWords, const unsigned* bits, const float3* N, const float3* D,
                         const float* W, int useCos, double* out) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nc) return;
  const float3 n = N[i];
  double acc = 0;
  for (unsigned w = 0; w < nWords; ++w) {
    unsigned m = bits[w * nc + i];
    while (m) {
      const unsigned d = w * 32 + (__ffs(m) - 1); m &= m - 1;
      acc += (double)(useCos ? fmaxf(dotf(n, D[d]), 0.f) : 1.f) * W[d];
    }
  }
  out[i] = acc;
}

// E[i*T + t] = sum_{sky bits} max(0,cos)*wsky*S[row,t] + G*g[t] + sum_{sun bits} max(0,cos)*sunW at sunStep.
// t in tiles of 32 held in registers; the bit walk is repeated per tile. ponytail: S read through L1 (broadcast),
// shared-memory tiling if the series pass ever dominates.
__global__ void k_series(unsigned nc, const unsigned* bits, const float3* N, const float3* D, unsigned nSky,
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
      unsigned m = bits[w * nc + i];
      while (m) {
        const unsigned d = w * 32 + (__ffs(m) - 1); m &= m - 1;
        if (d >= nSky) break;
        const float c = fmaxf(dotf(n, D[d]), 0.f) * wsky[d];
        const float* s = S + (size_t)row[d] * T + t0;
#pragma unroll
        for (int tt = 0; tt < 32; ++tt) if (t0 + tt < T) e[tt] += c * s[tt];
      }
    }
    for (int tt = 0; tt < 32 && t0 + tt < T; ++tt) Ei[t0 + tt] = e[tt] + G[i] * g[t0 + tt];
  }
  for (unsigned j = 0; j < nSun; ++j) {
    const unsigned d = sun0 + j;
    if ((bits[(d >> 5) * nc + i] >> (d & 31)) & 1u) Ei[sunStep[j]] += fmaxf(dotf(n, sunVec[j]), 0.f) * sunW[j];
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

  char log[4096] = {}; size_t logSize = sizeof log;
  OptixModuleCompileOptions mo = {};
  mo.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3; mo.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
  OptixPipelineCompileOptions po = {};
  po.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
  po.usesPrimitiveTypeFlags = static_cast<unsigned>(OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);
  po.numPayloadValues = 0; po.numAttributeValues = 2;
  po.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
  po.pipelineLaunchParamsVariableName = "params";
  OXL(optixModuleCreate(g.ctx, &mo, &po, reinterpret_cast<const char*>(trace_optixir), trace_optixir_len, log, &logSize, &g.module));

  OptixProgramGroupDesc pd[3] = {};
  pd[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;   pd[0].raygen.module = g.module;     pd[0].raygen.entryFunctionName = "__raygen__bits";
  pd[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;     pd[1].miss.module = g.module;       pd[1].miss.entryFunctionName = "__miss__ms";
  pd[2].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP; pd[2].hitgroup.moduleCH = g.module; pd[2].hitgroup.entryFunctionNameCH = "__closesthit__ch";
  OptixProgramGroupOptions pgo = {};
  logSize = sizeof log;
  OXL(optixProgramGroupCreate(g.ctx, pd, 3, &pgo, log, &logSize, g.pg));

  OptixPipelineLinkOptions lo = {};
  lo.maxTraceDepth = 1;
  logSize = sizeof log;
  OXL(optixPipelineCreate(g.ctx, &po, &lo, g.pg, 3, log, &logSize, &g.pipeline));
  OptixStackSizes ss = {};
  for (OptixProgramGroup p : g.pg) OX(optixUtilAccumulateStackSizes(p, &ss, g.pipeline));
  unsigned trav = 0, state = 0, cont = 0;
  OX(optixUtilComputeStackSizes(&ss, 1, 0, 0, &trav, &state, &cont));
  OX(optixPipelineSetStackSize(g.pipeline, trav, state, cont, 1));

  struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) Rec { char header[OPTIX_SBT_RECORD_HEADER_SIZE]; } rec[3];
  for (int k = 0; k < 3; ++k) OX(optixSbtRecordPackHeader(g.pg[k], &rec[k]));
  if (!upload(&g.dSbt, rec, sizeof rec, err)) return false;
  const CUdeviceptr base = reinterpret_cast<CUdeviceptr>(g.dSbt);
  g.sbt.raygenRecord = base;
  g.sbt.missRecordBase = base + sizeof(Rec);     g.sbt.missRecordStrideInBytes = sizeof(Rec);     g.sbt.missRecordCount = 1;
  g.sbt.hitgroupRecordBase = base + 2 * sizeof(Rec); g.sbt.hitgroupRecordStrideInBytes = sizeof(Rec); g.sbt.hitgroupRecordCount = 1;

  CU(cudaMalloc(reinterpret_cast<void**>(&g.dParams), sizeof(Params)));
  CU(cudaEventCreate(&g.ev[0])); CU(cudaEventCreate(&g.ev[1]));
  return true;
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
  for (void* p : {g->dGas, g->dSbt, (void*)g->dParams, (void*)g->dP, (void*)g->dN, (void*)g->dD, (void*)g->dBits, g->dScratch, g->dOut}) cudaFree(p);
  for (cudaEvent_t e : g->ev) if (e) cudaEventDestroy(e);
  delete g;
}

// ------------------------------------------------------------------ scene
// tmp[0..3] = verts, indices, build temp, uncompacted output; the caller frees whatever is left.
static bool build(Gpu& g, const Tris& t, void** tmp, std::string& err) {
  if (!upload(&tmp[0], t.v.data(), t.v.size() * sizeof(float3), err) || !upload(&tmp[1], t.i.data(), t.i.size() * sizeof(uint3), err)) return false;
  const unsigned flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
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
  bi.triangleArray.flags = &flags;
  bi.triangleArray.numSbtRecords = 1;
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
  for (int k = 0; k < 3; ++k) { cudaFree(tmp[k]); tmp[k] = nullptr; }   // inputs + temp freed before compaction
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

bool gpu_build(Gpu& g, const Tris& t, std::string& err) {
  cudaFree(g.dGas); g.dGas = nullptr; g.gas = 0; g.stats.gasBytes = 0;
  if (t.i.empty()) return true;   // empty scene: traversable 0 = every ray misses
  void* tmp[4] = {};
  const bool ok = build(g, t, tmp, err);
  for (void* p : tmp) cudaFree(p);
  if (!ok) { cudaFree(g.dGas); g.dGas = nullptr; g.gas = 0; }
  return ok;
}

bool gpu_set_scene(Gpu& g, const Sensors& s, const std::vector<float3>& dirs, std::string& err) {
  g.nSensors = s.p.size(); g.nDir = dirs.size(); g.nWords = (g.nDir + 31) / 32;
  return upload((void**)&g.dP, s.p.data(), s.p.size() * sizeof(float3), err) &&
         upload((void**)&g.dN, s.n.data(), s.n.size() * sizeof(float3), err) &&
         upload((void**)&g.dD, dirs.data(), dirs.size() * sizeof(float3), err);
}

// ------------------------------------------------------------------ trace / reduce / series
bool gpu_trace(Gpu& g, size_t i0, size_t nc, unsigned rayFlags, std::string& err) {
  if (i0 + nc > g.nSensors) { err = "gpu_trace: sensor range exceeds the scene"; return false; }
  if (!grow((void**)&g.dBits, &g.bitsCap, g.nWords * nc * sizeof(unsigned), err)) return false;
  g.i0 = i0;
  if (nc == 0 || g.nWords == 0) return true;
  const double targetMs = 150;   // per sub-launch, well under the 2 s Windows TDR
  size_t done = 0, n = std::min<size_t>(4096, nc);
  while (done < nc) {
    // params.nc is the word stride of the whole chunk; the sub-launch writes its slice bits[w*nc + done + i].
    const Params p = {g.gas, g.dP + i0 + done, g.dN + i0 + done, g.dD, (unsigned)g.nDir, (unsigned)nc, rayFlags, 0, g.dBits + done};
    CU(cudaMemcpy(g.dParams, &p, sizeof p, cudaMemcpyHostToDevice));
    CU(cudaEventRecord(g.ev[0]));
    OX(optixLaunch(g.pipeline, 0, reinterpret_cast<CUdeviceptr>(g.dParams), sizeof(Params), &g.sbt, (unsigned)n, (unsigned)g.nWords, 1));
    CU(cudaEventRecord(g.ev[1])); CU(cudaEventSynchronize(g.ev[1]));
    float ms = 0; CU(cudaEventElapsedTime(&ms, g.ev[0], g.ev[1]));
    g.stats.traceMs += ms; g.stats.rays += n * g.nDir; g.stats.launches++;
    done += n;
    const double want = double(n) * targetMs / std::max<double>(ms, 1e-3);
    n = std::min(std::max<size_t>(want > 1e12 ? size_t(1) << 40 : size_t(want), 1024), nc - done);
  }
  return true;
}

bool gpu_reduce(Gpu& g, size_t nc, const float* W, bool useCos, double* out, std::string& err) {
  if (nc == 0) return true;
  if (!grow(&g.dScratch, &g.scratchCap, g.nDir * sizeof(float), err) || !grow(&g.dOut, &g.outCap, nc * sizeof(double), err)) return false;
  CU(cudaMemcpy(g.dScratch, W, g.nDir * sizeof(float), cudaMemcpyHostToDevice));
  k_reduce<<<unsigned((nc + 255) / 256), 256>>>((unsigned)nc, (unsigned)g.nWords, g.dBits, g.dN + g.i0, g.dD,
                                                 static_cast<const float*>(g.dScratch), useCos, static_cast<double*>(g.dOut));
  CU(cudaGetLastError());
  CU(cudaMemcpy(out, g.dOut, nc * sizeof(double), cudaMemcpyDeviceToHost));
  return true;
}

bool gpu_series(Gpu& g, size_t nc, const SeriesIn& in, float* E, std::string& err) {
  if (nc == 0 || in.T <= 0) return true;
  const size_t T = size_t(in.T);
  // one packed upload: row | wsky | S | g | G | sunStep | sunVec | sunW (each 256-byte aligned)
  const void* src[8] = {in.row, in.wsky, in.S, in.g, in.G, in.sunStep, in.sunVec, in.sunW};
  const size_t len[8] = {in.nSky * 4, in.nSky * 4, size_t(in.P) * T * 4, T * 4, nc * 4, in.nSun * 4, in.nSun * sizeof(float3), in.nSun * 4};
  size_t off[8], total = 0;
  for (int k = 0; k < 8; ++k) { off[k] = total; total += (len[k] + 255) & ~size_t(255); }
  std::vector<char> h(total);
  for (int k = 0; k < 8; ++k) if (len[k]) std::memcpy(h.data() + off[k], src[k], len[k]);
  if (!grow(&g.dScratch, &g.scratchCap, total, err) || !grow(&g.dOut, &g.outCap, nc * T * sizeof(float), err)) return false;
  CU(cudaMemcpy(g.dScratch, h.data(), total, cudaMemcpyHostToDevice));
  const char* b = static_cast<const char*>(g.dScratch);
  k_series<<<unsigned((nc + 255) / 256), 256>>>((unsigned)nc, g.dBits, g.dN + g.i0, g.dD, in.nSky,
      reinterpret_cast<const int*>(b + off[0]), reinterpret_cast<const float*>(b + off[1]), reinterpret_cast<const float*>(b + off[2]),
      in.T, reinterpret_cast<const float*>(b + off[3]), reinterpret_cast<const float*>(b + off[4]), in.sun0, in.nSun,
      reinterpret_cast<const int*>(b + off[5]), reinterpret_cast<const float3*>(b + off[6]), reinterpret_cast<const float*>(b + off[7]),
      static_cast<float*>(g.dOut));
  CU(cudaGetLastError());
  CU(cudaMemcpy(E, g.dOut, nc * T * sizeof(float), cudaMemcpyDeviceToHost));
  return true;
}

bool gpu_download_bits(Gpu& g, size_t nc, std::vector<uint32_t>& out, std::string& err) {
  out.resize(g.nWords * nc);
  if (!out.empty()) CU(cudaMemcpy(out.data(), g.dBits, out.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  return true;
}

GpuStats gpu_stats(const Gpu& g) { return g.stats; }

unsigned gpu_ray_flags(bool cullBackfaces) {
  return OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_ANYHIT |
         (cullBackfaces ? OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES : 0u);
}

}  // namespace daylight
