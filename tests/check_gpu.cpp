// SPDX-License-Identifier: Apache-2.0 AND LicenseRef-Commons-Clause-1.0
// GPU module checks: hand-coded ground + box scene, hemisphere cull, Waechter-Binder offset, reduce / series
// against CPU loops over the downloaded bits, adaptive sub-launches. Self-skips without a CUDA device.
#include "internal.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace daylight;

static const double kPi = 3.14159265358979323846;
static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { std::printf("%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #cond); ++failures; } } while (0)

static float3 f3(double x, double y, double z) { return float3{float(x), float(y), float(z)}; }
static float3 unit(double x, double y, double z) { double l = std::sqrt(x * x + y * y + z * z); return f3(x / l, y / l, z / l); }
static float dotf(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }   // same op order as the device
static void quad(Tris& t, float3 a, float3 b, float3 c, float3 d) {   // CCW seen from outside
  unsigned v = unsigned(t.v.size());
  t.v.insert(t.v.end(), {a, b, c, d});
  t.i.push_back(uint3{v, v + 1, v + 2}); t.i.push_back(uint3{v, v + 2, v + 3});
}
// Slab test: ray from o along d against the box [lo,hi]^3 (t > 0).
static bool hits_box(float3 o, float3 d, double lo, double hi) {
  double t0 = 0, t1 = 1e30, oo[3] = {o.x, o.y, o.z}, dd[3] = {d.x, d.y, d.z};
  for (int a = 0; a < 3; ++a) {
    double ta = (lo - oo[a]) / dd[a], tb = (hi - oo[a]) / dd[a];   // dd == 0 -> +-inf, origin strictly inside the slab
    if (ta > tb) std::swap(ta, tb);
    t0 = std::max(t0, ta); t1 = std::min(t1, tb);
  }
  return t0 <= t1;
}

int check_gpu(const char* testsDir) {
  std::string err;
  Gpu* g = gpu_init(err);
  if (!g) { std::printf("skip: no CUDA device (%s)\n", err.c_str()); return 0; }

  CHECK(gpu_ray_flags(false) == 5u);    // TERMINATE_ON_FIRST_HIT (4) | DISABLE_ANYHIT (1)
  CHECK(gpu_ray_flags(true) == 21u);    // + CULL_BACK_FACING_TRIANGLES (16)

  // scene: ground [-2,2]^2 at z = 0, unit box [0.5,1.5]^3, all faces outward
  Tris t;
  const double lo = 0.5, hi = 1.5;
  quad(t, f3(-2, -2, 0), f3(2, -2, 0), f3(2, 2, 0), f3(-2, 2, 0));
  quad(t, f3(lo, lo, lo), f3(lo, hi, lo), f3(hi, hi, lo), f3(hi, lo, lo));   // bottom (-z)
  quad(t, f3(lo, lo, hi), f3(hi, lo, hi), f3(hi, hi, hi), f3(lo, hi, hi));   // top (+z)
  quad(t, f3(lo, lo, lo), f3(lo, lo, hi), f3(lo, hi, hi), f3(lo, hi, lo));   // -x
  quad(t, f3(hi, lo, lo), f3(hi, hi, lo), f3(hi, hi, hi), f3(hi, lo, hi));   // +x
  quad(t, f3(lo, lo, lo), f3(hi, lo, lo), f3(hi, lo, hi), f3(lo, lo, hi));   // -y
  quad(t, f3(lo, hi, lo), f3(lo, hi, hi), f3(hi, hi, hi), f3(hi, hi, lo));   // +y
  CHECK(t.i.size() == 14);
  CHECK(gpu_build(*g, t, err));
  CHECK(gpu_stats(*g).gasBytes > 0);

  // sensors: s0 under the box, s1 open, s2 facing down on the ground, s3 above the box looking down, s4 inside the box
  Sensors s;
  s.p = {f3(1, 1, 0), f3(-1, -1, 0), f3(-1, -1, 0), f3(1, 1, 2), f3(1, 1, 1)};
  s.n = {f3(0, 0, 1), f3(0, 0, 1), f3(0, 0, -1), f3(0, 0, -1), f3(0, 0, 1)};
  const size_t nc = s.p.size();

  // directions: 145 "patches" (7 rows x uniform azimuth + zenith), then zenith (sun), near-horizontal, nadir
  std::vector<float3> D;
  const int naz[7] = {30, 30, 24, 24, 18, 12, 6};
  for (int r = 0; r < 7; ++r) {
    const double alt = (6 + 12 * r) * kPi / 180;
    for (int j = 0; j < naz[r]; ++j) { double az = 2 * kPi * j / naz[r]; D.push_back(f3(std::cos(alt) * std::sin(az), std::cos(alt) * std::cos(az), std::sin(alt))); }
  }
  D.push_back(f3(0, 0, 1));
  const unsigned nSky = 145, Z = 145, H = 146, NA = 147;
  D.push_back(f3(0, 0, 1)); D.push_back(unit(0.999, 0, 0.0447)); D.push_back(f3(0, 0, -1));
  const unsigned nDir = unsigned(D.size()), nWords = (nDir + 31) / 32;
  CHECK(nDir == 148 && nWords == 5);
  CHECK(gpu_set_scene(*g, s, D, err));

  const unsigned flags = gpu_ray_flags(false);
  CHECK(gpu_trace(*g, 0, nc, flags, err));
  std::vector<uint32_t> bits;
  CHECK(gpu_download_bits(*g, nc, bits, err));
  CHECK(bits.size() == nWords * nc);
  auto bit = [&](size_t i, unsigned d) { return (bits[(d / 32) * nc + i] >> (d % 32)) & 1u; };
  CHECK(bit(0, Z) == 0);    // box bottom above s0
  CHECK(bit(0, H) == 1);    // grazing ray passes under the box
  CHECK(bit(1, Z) == 1);
  CHECK(bit(1, NA) == 0);   // culled
  CHECK(bit(2, NA) == 1);   // origin nudged below the ground plane along -z
  CHECK(bit(2, Z) == 0);    // culled
  CHECK(bit(3, NA) == 0);   // box top
  CHECK(bit(4, Z) == 0);    // inside the box: hits the top from its back side (two-sided occluders)
  for (unsigned d = 0; d < nDir; ++d) {   // every bit of s0 / s1 against the CPU slab test (s1 never sees the box)
    CHECK(bit(0, d) == (dotf(s.n[0], D[d]) > 0 && !hits_box(s.p[0], D[d], lo, hi) ? 1u : 0u));
    CHECK(bit(1, d) == (dotf(s.n[1], D[d]) > 0 && !hits_box(s.p[1], D[d], lo, hi) ? 1u : 0u));
  }

  // reduce: counts (W = 1, no cos), then cosine-weighted against a CPU loop over the same bits
  std::vector<float> W(nDir, 1.f);
  std::vector<double> out(nc);
  CHECK(gpu_reduce(*g, nc, W.data(), false, out.data(), err));
  int expect0 = 0, expect1 = 0, upward = 0;   // the box also clips a few low NE directions from s1 (alt 18, azi 45: z 0.69 at x 0.5)
  for (unsigned d = 0; d < nDir; ++d) {
    upward += dotf(s.n[0], D[d]) > 0;
    expect0 += dotf(s.n[0], D[d]) > 0 && !hits_box(s.p[0], D[d], lo, hi);
    expect1 += dotf(s.n[1], D[d]) > 0 && !hits_box(s.p[1], D[d], lo, hi);
  }
  CHECK(upward == 147);   // 145 + zenith + near-horizontal; nadir culled
  CHECK(out[1] == expect1 && expect1 < 147 && expect1 > expect0);
  CHECK(out[0] == expect0);
  std::printf("gpu: visible dirs s0 %d, s1 %d of %d upward\n", expect0, expect1, upward);
  CHECK(out[2] == 1 && out[3] == 0 && out[4] == 0);   // s2: nadir only; s3: nadir blocked; s4: everything blocked
  for (unsigned d = 0; d < nDir; ++d) W[d] = 0.04f * (1.f + 0.01f * d);
  CHECK(gpu_reduce(*g, nc, W.data(), true, out.data(), err));
  for (size_t i = 0; i < nc; ++i) {
    double acc = 0;
    for (unsigned d = 0; d < nDir; ++d) if (bit(i, d)) acc += (double)std::max(dotf(s.n[i], D[d]), 0.f) * W[d];
    CHECK(std::fabs(out[i] - acc) <= 1e-9);
  }

  // series: T = 3, S = 1, g = {0.5, 0.25, 0}, one sun (zenith) at step 1 with 100
  const int T = 3;
  std::vector<int> row(nSky);
  std::vector<float> wsky(nSky), S(nSky * T, 1.f), G(nc);
  for (unsigned d = 0; d < nSky; ++d) { row[d] = int(d); wsky[d] = W[d]; }
  for (size_t i = 0; i < nc; ++i) G[i] = 0.1f * float(i + 1);
  const float gt[3] = {0.5f, 0.25f, 0.f};
  const int sunStep[1] = {1}; const float3 sunVec[1] = {f3(0, 0, 1)}; const float sunW[1] = {100.f};
  SeriesIn in;
  in.nSky = nSky; in.row = row.data(); in.wsky = wsky.data(); in.S = S.data(); in.P = int(nSky); in.T = T;
  in.g = gt; in.G = G.data(); in.sun0 = Z; in.nSun = 1; in.sunStep = sunStep; in.sunVec = sunVec; in.sunW = sunW;
  std::vector<float> E(nc * T);
  CHECK(gpu_series(*g, nc, in, E.data(), err));
  std::vector<double> sky(nc);
  for (size_t i = 0; i < nc; ++i) {
    for (unsigned d = 0; d < nSky; ++d) if (bit(i, d)) sky[i] += std::max(dotf(s.n[i], D[d]), 0.f) * wsky[d];
    for (int tt = 0; tt < T; ++tt) {
      const double e = sky[i] + G[i] * gt[tt] + (tt == 1 && bit(i, Z) ? std::max(s.n[i].z, 0.f) * 100.0 : 0.0);
      CHECK(std::fabs(E[i * T + tt] - e) <= 1e-4 * e + 1e-6);
    }
  }
  CHECK(E[1 * T + 1] > 100 && E[0 * T + 1] < 100);   // s1 sees the sun, s0 does not
  // sum_t E[i][t] == reduce with the equivalent weights (3*wsky on sky, 100 on the sun direction) + G*sum g
  std::vector<float> W2(nDir, 0.f);
  for (unsigned d = 0; d < nSky; ++d) W2[d] = 3 * wsky[d];
  W2[Z] = 100.f;
  CHECK(gpu_reduce(*g, nc, W2.data(), true, out.data(), err));
  for (size_t i = 0; i < nc; ++i) {
    const double sum = double(E[i * T]) + E[i * T + 1] + E[i * T + 2] - G[i] * 0.75;
    CHECK(std::fabs(sum - out[i]) <= 1e-4 * std::fabs(out[i]) + 1e-6);
  }

  // chunk at i0 = 1 (sensors s1..s3): bits, reduce and series must use the offset sensor arrays
  {
    std::vector<uint32_t> sub; std::vector<double> o2(3); std::vector<float> E2(3 * T);
    SeriesIn in2 = in; in2.G = G.data() + 1;
    CHECK(gpu_trace(*g, 1, 3, flags, err) && gpu_download_bits(*g, 3, sub, err) && sub.size() == nWords * 3);
    for (unsigned w = 0; w < nWords; ++w) for (size_t i = 0; i < 3; ++i) CHECK(sub[w * 3 + i] == bits[w * nc + 1 + i]);
    CHECK(gpu_reduce(*g, 3, W2.data(), true, o2.data(), err) && gpu_series(*g, 3, in2, E2.data(), err));
    for (size_t i = 0; i < 3; ++i) { CHECK(o2[i] == out[i + 1]); for (int tt = 0; tt < T; ++tt) CHECK(E2[i * T + tt] == E[(i + 1) * T + tt]); }
  }

  // back-face culling flag: s4 inside the box now sees the zenith (box top hit from behind is culled)
  CHECK(gpu_trace(*g, 0, nc, gpu_ray_flags(true), err));
  CHECK(gpu_download_bits(*g, nc, bits, err));
  CHECK(bit(4, Z) == 1 && bit(0, Z) == 0 && bit(3, NA) == 0);

  // 5000 copies of s1: adaptive sub-launch path (4096 + 904), all words equal to s1's
  CHECK(gpu_trace(*g, 0, nc, flags, err));
  CHECK(gpu_download_bits(*g, nc, bits, err));
  std::vector<uint32_t> ref(nWords);
  for (unsigned w = 0; w < nWords; ++w) ref[w] = bits[w * nc + 1];
  Sensors many;
  many.p.assign(5000, s.p[1]); many.n.assign(5000, s.n[1]);
  CHECK(gpu_set_scene(*g, many, D, err));
  const GpuStats before = gpu_stats(*g);
  CHECK(gpu_trace(*g, 0, 5000, flags, err));
  const GpuStats after = gpu_stats(*g);
  CHECK(after.launches - before.launches == 2);
  CHECK(after.rays - before.rays == 5000u * nDir);
  CHECK(gpu_download_bits(*g, 5000, bits, err));
  size_t bad = 0;
  for (unsigned w = 0; w < nWords; ++w) for (size_t i = 0; i < 5000; ++i) bad += bits[w * 5000 + i] != ref[w];
  CHECK(bad == 0);
  std::printf("gpu: 5000 sensors x %u dirs: %.3f ms, %.3f GRays/s (build %.2f ms, GAS %zu B)\n", nDir,
              after.traceMs - before.traceMs, double(after.rays - before.rays) / ((after.traceMs - before.traceMs) * 1e6),
              after.buildMs, after.gasBytes);

  // empty scene: traversable 0 = every hemisphere direction visible
  CHECK(gpu_build(*g, Tris{}, err));
  CHECK(gpu_set_scene(*g, s, D, err));
  CHECK(gpu_trace(*g, 0, nc, flags, err));
  CHECK(gpu_reduce(*g, nc, std::vector<float>(nDir, 1.f).data(), false, out.data(), err));
  CHECK(out[0] == 147 && out[2] == 1);

  if (!err.empty()) std::printf("gpu: last error: %s\n", err.c_str());
  gpu_destroy(g);

  // run() keeps its GPU context for the process and, with Options::sceneKey, the resident GAS: same numbers, no rebuild
  {
    std::vector<Mesh> meshes;
    std::string e2;
    CHECK(read_obj(std::string(testsDir) + "/box.obj", meshes, e2));
    Options o;
    o.quantities = {Quantity::SunHours};
    o.lat = 42.35; o.lon = -71.07; o.tz = -5;
    CHECK(parse_hoy("06-21T12:00", false, o.t0, e2));
    auto ms = [](std::chrono::steady_clock::duration d) { return std::chrono::duration<double, std::milli>(d).count(); };
    Result r1, r2, r3;
    o.sceneKey = "box";
    const auto t0 = std::chrono::steady_clock::now();
    CHECK(run(meshes, nullptr, o, r1, e2));
    const auto t1 = std::chrono::steady_clock::now();
    CHECK(run(meshes, nullptr, o, r2, e2));   // same key: scene reused
    const auto t2 = std::chrono::steady_clock::now();
    o.sceneKey = "box-2";
    CHECK(run(meshes, nullptr, o, r3, e2));   // new key: rebuilt, identical values
    size_t faces = 0;
    for (const Mesh& m : meshes) faces += m.numFaces();
    CHECK(r1.values.size() == faces && faces > 0);
    CHECK(r1.values == r2.values);
    CHECK(r1.values == r3.values);
    if (!e2.empty()) std::printf("run: %s\n", e2.c_str());
    std::printf("run: first %.0f ms (context + GAS), reused scene %.0f ms\n", ms(t1 - t0), ms(t2 - t1));

    // several resident scenes: A, B, A again reuses A's parked GAS with identical values; one slot only keeps the last
    o.residentScenes = 2;
    Result ra, rb, ra2;
    o.sceneKey = "scene-A"; CHECK(run(meshes, nullptr, o, ra, e2));
    o.sceneKey = "scene-B"; CHECK(run(meshes, nullptr, o, rb, e2));
    CHECK(scene_resident("scene-A") && scene_resident("scene-B") && !scene_resident("scene-C") && !scene_resident(""));
    o.sceneKey = "scene-A"; CHECK(run(meshes, nullptr, o, ra2, e2));
    CHECK(ra2.values == ra.values && ra.values == r1.values);
    o.residentScenes = 1;
    o.sceneKey = "scene-C"; CHECK(run(meshes, nullptr, o, ra2, e2));
    CHECK(scene_resident("scene-C") && !scene_resident("scene-A") && !scene_resident("scene-B"));

    // cancel: a set flag stops the run with "cancelled"
    std::atomic<bool> stop{true};
    o.cancel = &stop;
    e2.clear();
    CHECK(!run(meshes, nullptr, o, ra2, e2) && e2 == "cancelled");
  }
  return failures;
}
