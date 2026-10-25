// SPDX-License-Identifier: Apache-2.0 AND LicenseRef-Commons-Clause-1.0
// GPU module checks: hand-coded ground + box scene, hemisphere cull, Waechter-Binder offset, reduce / series
// against CPU loops over the downloaded bits or transmittance bytes, adaptive sub-launches, the Radiance glass port,
// the reflected-light pass against closed forms (bins, events, chunks), and run() end to end with optics (slots, thin
// solids, culling, residency, the dump) and bounces. Self-skips without a CUDA device.
#include "glass.h"
#include "internal.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace daylight;

static const double kPi = 3.14159265358979323846;
static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { std::printf("%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #cond); ++failures; } } while (0)

Mesh box_mesh(const char* name, double x0, double y0, double z0, double x1, double y1, double z1, bool soup);   // check_mesh.cpp

static float3 f3(double x, double y, double z) { return float3{float(x), float(y), float(z)}; }
static float3 unit(double x, double y, double z) { double l = std::sqrt(x * x + y * y + z * z); return f3(x / l, y / l, z / l); }
static float dotf(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }   // same op order as the device
static void quad(Tris& t, float3 a, float3 b, float3 c, float3 d, uint32_t slot = 0) {   // CCW seen from outside
  unsigned v = unsigned(t.v.size());
  t.v.insert(t.v.end(), {a, b, c, d});
  t.i.push_back(uint3{v, v + 1, v + 2}); t.i.push_back(uint3{v, v + 2, v + 3});
  t.slot.insert(t.slot.end(), {slot, slot});
}
// One square face (2 half x 2 half, facing +z) per x, centred at (x, 0, z).
static Mesh squares(const char* name, bool sensor, double z, double half, std::initializer_list<double> xs) {
  Mesh m; m.name = name; m.sensor = sensor;
  for (double x : xs) {
    const int b = int(m.numPoints());
    m.points.insert(m.points.end(), {x - half, -half, z, x + half, -half, z, x + half, half, z, x - half, half, z});
    m.counts.push_back(4); m.indices.insert(m.indices.end(), {b, b + 1, b + 2, b + 3});
  }
  return m;
}

// Radiance glass port (glass.h) against rad5R4, run once: rtrace -ab 0 through a glass pane at z = 1 with tn =
// tn_from_Tn(0.64) = 0.6975761815, a glow sky behind it, rays from the origin at 0 / 30 / 60 / 75 / 85 deg:
//   glass.rad: void glass pane_mat 0 0 3 0.6975761815 0.6975761815 0.6975761815
//              pane_mat polygon pane 0 0 12 -100 -100 1 100 -100 1 100 100 1 -100 100 1
//              void glow skyglow 0 0 4 1 1 1 0     skyglow source sky 0 0 4 0 0 1 180
//   oconv glass.rad > glass.oct; printf '0 0 0 sin(a) 0 cos(a)\n' ... | rtrace -ab 0 -h glass.oct
static const double kGlassDeg[5] = {0, 30, 60, 75, 85};
static const double kGlassT[5] = {6.400000e-01, 6.248004e-01, 5.397539e-01, 3.696623e-01, 1.153073e-01};

static void check_glass() {
  const float tn = float(tn_from_Tn(0.64));
  float T, R;
  for (int k = 0; k < 5; ++k) { glass_TR(float(std::cos(kGlassDeg[k] * kPi / 180)), tn, T, R); CHECK(std::fabs(T - kGlassT[k]) <= 1e-5); }
  glass_TR(1.f, tn, T, R);
  CHECK(std::fabs(T - 0.64) <= 1e-6);
  for (double a : {0., 30., 60., 89.}) { glass_TR(float(std::cos(a * kPi / 180)), 1.f, T, R); CHECK(std::fabs(T + R - 1) <= 1e-6); }   // no absorption: T + R = 1
  glass_TR(0.f, tn, T, R);
  CHECK(T == 0 && R == 1);   // grazing: everything reflected, no NaN
  for (double Tn : {1e-4, 1e-9})   // below ~1.1e-4 the tn constants turn negative: clamped to 0, blocked, no NaN
    for (float c : {1.f, 0.5f}) { glass_TR(c, float(tn_from_Tn(Tn)), T, R); CHECK(T >= 0 && T < 1.f / 510 && R >= 0 && R <= 1); }
}
// What a pane of tn passes and mirrors of a diffuse field: glass_TR's cosine-weighted means (weight 2 cos, midpoint rule)
static void glass_means(float tn, double& mT, double& mR) {
  mT = mR = 0;
  for (int k = 0; k < 100000; ++k) { const double c = (k + 0.5) / 1e5; float T, R; glass_TR(float(c), tn, T, R); mT += 2e-5 * c * T; mR += 2e-5 * c * R; }
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
  check_glass();
  std::string err;
  Gpu* g = gpu_init(err);
  if (!g) { std::printf("skip: no CUDA device (%s)\n", err.c_str()); return failures; }

  CHECK(gpu_ray_flags(false, false) == 5u);    // TERMINATE_ON_FIRST_HIT (4) | DISABLE_ANYHIT (1)
  CHECK(gpu_ray_flags(true, false) == 21u);    // + CULL_BACK_FACING_TRIANGLES (16)
  CHECK(gpu_ray_flags(false, true) == 4u);     // a slot transmits: any-hit on
  CHECK(gpu_ray_flags(true, true) == 20u);

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
  CHECK(gpu_build(*g, t, {Slot{}}, err));
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

  const unsigned flags = gpu_ray_flags(false, false);
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
  CHECK(gpu_trace(*g, 0, nc, gpu_ray_flags(true, false), err));
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

  // byte mode: porous 0.5 panes (slot 1 at z = 1 over x in [-50, 50], slot 2 at z = 2 over x in [0, 50]), an opaque lid
  // (slot 0) at z = 0.5 over (30, 30); s0 under one pane, s1 under both, s2 under the lid, s3 looking down
  {
    Tris tp;
    quad(tp, f3(-50, -50, 1), f3(50, -50, 1), f3(50, 50, 1), f3(-50, 50, 1), 1);
    quad(tp, f3(0, -50, 2), f3(50, -50, 2), f3(50, 50, 2), f3(0, 50, 2), 2);
    quad(tp, f3(25, 25, 0.5), f3(35, 25, 0.5), f3(35, 35, 0.5), f3(25, 35, 0.5), 0);
    std::vector<Slot> sl(3);
    sl[1].tau = sl[2].tau = 0.5f;
    Sensors sp;
    sp.p = {f3(-10, 0, 0), f3(10, 0, 0), f3(30, 30, 0), f3(-10, 0, 0)};
    sp.n = {f3(0, 0, 1), f3(0, 0, 1), f3(0, 0, 1), f3(0, 0, -1)};
    const size_t np = sp.p.size();
    CHECK(gpu_build(*g, tp, sl, err) && gpu_set_slots(*g, sl, err) && gpu_set_scene(*g, sp, D, err));
    const unsigned tf = gpu_ray_flags(false, true);
    std::vector<uint8_t> by;
    CHECK(gpu_trace(*g, 0, np, tf, err) && gpu_download_bytes(*g, np, by, err) && by.size() == nDir * np);
    auto byte = [&](size_t i, unsigned d) { return by[d * np + i]; };
    CHECK(byte(0, Z) == 128 && byte(1, Z) == 64 && byte(2, Z) == 0 && byte(3, Z) == 0 && byte(3, NA) == 255 && byte(0, NA) == 0);
    // reduce / series against CPU loops over the bytes (T = byte / 255)
    for (unsigned d = 0; d < nDir; ++d) W[d] = 0.04f * (1.f + 0.01f * d);
    std::vector<double> ob(np);
    CHECK(gpu_reduce(*g, np, W.data(), true, ob.data(), err));
    for (size_t i = 0; i < np; ++i) {
      double acc = 0;
      for (unsigned d = 0; d < nDir; ++d) if (byte(i, d)) acc += (double)(std::max(dotf(sp.n[i], D[d]), 0.f) * (byte(i, d) / 255.f)) * W[d];
      CHECK(std::fabs(ob[i] - acc) <= 1e-9 && acc > 0);
    }
    std::vector<float> Gp(np, 0.25f), Ep(np * T);
    SeriesIn ip = in; ip.G = Gp.data();
    CHECK(gpu_series(*g, np, ip, Ep.data(), err));
    for (size_t i = 0; i < np; ++i) {
      double sk = 0;
      for (unsigned d = 0; d < nSky; ++d) sk += std::max(dotf(sp.n[i], D[d]), 0.f) * (byte(i, d) / 255.f) * wsky[d];
      for (int tt = 0; tt < T; ++tt) {
        const double e = sk + 0.25 * gt[tt] + (tt == 1 ? std::max(sp.n[i].z, 0.f) * (byte(i, Z) / 255.f) * 100.0 : 0.0);
        CHECK(std::fabs(Ep[i * T + tt] - e) <= 1e-4 * e + 1e-6);
      }
    }
    // chunk at i0 = 1 (s1, s2): the same bytes, reduce and series
    std::vector<uint8_t> sub; std::vector<double> o2(2); std::vector<float> E2(2 * T);
    SeriesIn ip2 = ip; ip2.G = Gp.data() + 1;
    CHECK(gpu_trace(*g, 1, 2, tf, err) && gpu_download_bytes(*g, 2, sub, err) && sub.size() == nDir * 2);
    for (unsigned d = 0; d < nDir; ++d) for (size_t i = 0; i < 2; ++i) CHECK(sub[d * 2 + i] == byte(1 + i, d));
    CHECK(gpu_reduce(*g, 2, W.data(), true, o2.data(), err) && gpu_series(*g, 2, ip2, E2.data(), err));
    for (size_t i = 0; i < 2; ++i) { CHECK(o2[i] == ob[i + 1]); for (int tt = 0; tt < T; ++tt) CHECK(E2[i * T + tt] == Ep[(i + 1) * T + tt]); }
    CHECK(!gpu_download_bits(*g, 2, bits, err));   // bytes were written, not bits
    // 20000 copies of s1: byte-mode sub-launches start at 2^13 threads (1638 sensors x 5 words) and at most double, so
    // at least 4 (1638 + 3276 + 6552 + 8534; bits: 4096 + 15904), each writing its slice; every byte equals s1's
    Sensors copies;
    copies.p.assign(20000, sp.p[1]); copies.n.assign(20000, sp.n[1]);
    const GpuStats b0 = gpu_stats(*g);
    std::vector<uint8_t> cb;
    CHECK(gpu_set_scene(*g, copies, D, err) && gpu_trace(*g, 0, 20000, tf, err) && gpu_download_bytes(*g, 20000, cb, err));
    CHECK(gpu_stats(*g).launches - b0.launches >= 4 && gpu_stats(*g).rays - b0.rays == 20000u * nDir && cb.size() == 20000u * nDir);
    size_t badB = 0;
    for (unsigned d = 0; d < nDir && cb.size() == 20000u * nDir; ++d) for (size_t i = 0; i < 20000; ++i) badB += cb[d * 20000 + i] != byte(1, d);
    CHECK(badB == 0);
    CHECK(gpu_set_scene(*g, sp, D, err));
    // transmission off: the same GAS traced without any-hit gives the bits of an all-opaque build
    std::vector<uint32_t> offBits, opaqueBits;
    CHECK(gpu_trace(*g, 0, np, gpu_ray_flags(false, false), err) && gpu_download_bits(*g, np, offBits, err));
    std::vector<Slot> opaque(3);
    CHECK(gpu_build(*g, tp, opaque, err) && gpu_set_slots(*g, opaque, err));
    CHECK(gpu_trace(*g, 0, np, gpu_ray_flags(false, false), err) && gpu_download_bits(*g, np, opaqueBits, err));
    CHECK(offBits == opaqueBits && ((offBits[(Z / 32) * np] >> (Z % 32)) & 1u) == 0);
    std::vector<Slot> few(2);
    CHECK(!gpu_build(*g, tp, few, err) && err.find("slot") != std::string::npos);   // a triangle in slot 2 needs 3 records
    err.clear();
  }

  // glass pane (slot 0) at z = 1: bytes at the five rtrace angles within half a step; --fresnel off (no kSlotGlass) = tau
  {
    Tris tg;
    quad(tg, f3(-100, -100, 1), f3(100, -100, 1), f3(100, 100, 1), f3(-100, 100, 1));
    Slot glass; glass.rho = 0; glass.tau = 0.64f; glass.tn = float(tn_from_Tn(0.64)); glass.flags = kSlotGlass;
    Sensors so; so.p = {f3(0, 0, 0)}; so.n = {f3(0, 0, 1)};
    std::vector<float3> D5;
    for (double a : kGlassDeg) D5.push_back(f3(std::sin(a * kPi / 180), 0, std::cos(a * kPi / 180)));
    std::vector<uint8_t> gb;
    CHECK(gpu_build(*g, tg, {glass}, err) && gpu_set_slots(*g, {glass}, err) && gpu_set_scene(*g, so, D5, err));
    CHECK(gpu_trace(*g, 0, 1, gpu_ray_flags(false, true), err) && gpu_download_bytes(*g, 1, gb, err) && gb.size() == 5);
    for (int k = 0; k < 5 && gb.size() == 5; ++k) CHECK(std::fabs(gb[size_t(k)] - 255 * kGlassT[k]) <= 0.5);
    if (gb.size() == 5) std::printf("gpu: glass bytes %d %d %d %d %d (rtrace x 255: 163.2 159.3 137.6 94.3 29.4)\n", gb[0], gb[1], gb[2], gb[3], gb[4]);
    glass.flags = 0;
    CHECK(gpu_set_slots(*g, {glass}, err) && gpu_trace(*g, 0, 1, gpu_ray_flags(false, true), err) && gpu_download_bytes(*g, 1, gb, err));
    CHECK(gb == std::vector<uint8_t>(5, 163));   // 0.64 x 255 = 163.2 at every angle
  }

  // reflected light (gpu_bounce): a sensor's row holds its coefficient per bin (0 ground, 1 + p the patches of the sky
  // density, gendaymtx order), pi / M per path, so a row without random events sums to pi x the weight of the reflected paths
  {
    CHECK(gpu_bounce_flags(false) == 1u && gpu_bounce_flags(true) == 17u);   // DISABLE_ANYHIT (1) | CULL_BACK_FACING_TRIANGLES (16)
    const Patches pa = make_patches(1);
    const size_t nb = size_t(pa.P) + 1;
    std::vector<float> dc;   // the last bounce's rows [bins x n]
    const auto bounce = [&](const Tris& tr, const std::vector<Slot>& sl, const Sensors& se, Bounce b, size_t i0 = 0, bool cull = false) {
      const size_t n = se.p.size() - i0, bins = size_t(make_patches(b.mf).P) + 1;
      const bool ok = gpu_build(*g, tr, sl, err, true) && gpu_set_slots(*g, sl, err) && gpu_set_scene(*g, se, D, err) &&
                      gpu_bounce(*g, i0, n, gpu_bounce_flags(cull), b, err) && gpu_download_dc(*g, n, dc, err) && dc.size() == bins * n;
      CHECK(ok);
      return ok;
    };
    // a 2 km plane (rho 0.5): sensors looking down 1 m and 1 cm above it reflect every path once, deterministically, the
    // reflections spread over the patch rows of every sky density as cos^2 says (5 sigma; the zenith cap is the last bin);
    // one looking up sees nothing reflected
    Tris plane, under;   // under: wound away from the sensors
    quad(plane, f3(-1000, -1000, 0), f3(1000, -1000, 0), f3(1000, 1000, 0), f3(-1000, 1000, 0));
    quad(under, f3(-1000, -1000, 0), f3(-1000, 1000, 0), f3(1000, 1000, 0), f3(1000, -1000, 0));
    Slot half; half.rho = 0.5f;
    Sensors sa;
    sa.p = {f3(0, 0, 1), f3(3, 0, 0.01), f3(0, 0, 1)};
    sa.n = {f3(0, 0, -1), f3(0, 0, -1), f3(0, 0, 1)};
    const unsigned M = 65536;
    for (int mf : {1, 2, 4}) {
      if (!bounce(plane, {half}, sa, Bounce{1, M, mf})) continue;
      const Patches pm = make_patches(mf);
      const size_t bins = size_t(pm.P) + 1;
      for (size_t i = 0; i < 2; ++i) {
        double sum = 0;
        for (size_t b = 0; b < bins; ++b) sum += dc[b * 3 + i];
        CHECK(std::fabs(sum / (0.5 * kPi) - 1) <= 1e-4 && dc[i] == 0);
      }
      CHECK(pm.row[size_t(pm.P) - 1] == 7 * mf);
      for (int r = 0; r <= 7 * mf; ++r) {   // row r covers altitudes [r alpha, (r + 1) alpha), the zenith cap 7 mf alpha .. 90 deg
        double c = 0;
        for (size_t p = 0; p < size_t(pm.P); ++p) if (pm.row[p] == r) c += dc[(p + 1) * 3];
        const double s0 = std::sin(pm.alpha * r), s1 = r == 7 * mf ? 1.0 : std::sin(pm.alpha * (r + 1)), q = s1 * s1 - s0 * s0;
        CHECK(std::fabs(c - 0.5 * kPi * q) <= 5 * 0.5 * kPi * std::sqrt(q * (1 - q) / M));
      }
      double open = 0;
      for (size_t b = 0; b < bins; ++b) open += dc[b * 3 + 2];
      CHECK(open == 0);
    }
    // the plane wound away from the sensors gives the same rows (the normal is taken on the ray's side), none when back
    // faces are culled; wound towards them, culling changes nothing
    if (bounce(plane, {half}, sa, Bounce{1, 4096})) {
      const std::vector<float> rows = dc;
      CHECK(bounce(under, {half}, sa, Bounce{1, 4096}) && dc == rows);
      CHECK(bounce(under, {half}, sa, Bounce{1, 4096}, 0, true) && dc == std::vector<float>(nb * 3, 0.f));
      CHECK(bounce(plane, {half}, sa, Bounce{1, 4096}, 0, true) && dc == rows);
    }
    // sensors in the plane of a 10 km quad at the origin, looking up and down: the hardware errs there by ~20x their offset,
    // so a first hit on that plane is the sensor's own; nothing above or below them reflects
    {
      Tris wide;
      quad(wide, f3(-5000, -5000, 0), f3(5000, -5000, 0), f3(5000, 5000, 0), f3(-5000, 5000, 0));
      Sensors so; so.p = {f3(0, 0, 0), f3(0.01, 0.02, 0)}; so.n = {f3(0, 0, 1), f3(0, 0, -1)};
      CHECK(bounce(wide, {half}, so, Bounce{1, 4096}) && dc == std::vector<float>(nb * 2, 0.f));
    }
    // a chunk gives the rows its sensors have in the whole run: seeds from the global sensor index, sums in sample order,
    // whatever the launches (300 sensors start at 27 samples per launch, the chunk of the last 3 at 64)
    {
      Sensors sm;
      for (int k = 0; k < 297; ++k) { sm.p.push_back(f3(0.01 * k, 1, 0.5)); sm.n.push_back(f3(0, 0, -1)); }
      sm.p.insert(sm.p.end(), sa.p.begin(), sa.p.end()); sm.n.insert(sm.n.end(), sa.n.begin(), sa.n.end());
      if (bounce(plane, {half}, sm, Bounce{1, M})) {
        const std::vector<float> all = dc;
        if (bounce(plane, {half}, sm, Bounce{1, M}, 297))
          for (size_t b = 0; b < nb; ++b) for (size_t i = 0; i < 3; ++i) CHECK(dc[b * 3 + i] == all[b * 300 + 297 + i]);
      }
    }
    // reduce = CPU loop over the rows; the series' reflected part summed over its steps = the reduce of the summed radiances
    {
      const int T3 = 3;
      std::vector<float> Sd(nb * T3), Wd(nb, 0.f), zS(nSky * T3, 0.f), zg(T3, 0.f), zG(3, 0.f), E3(3 * T3);
      for (size_t b = 0; b < nb; ++b)
        for (int t3 = 0; t3 < T3; ++t3) { Sd[b * T3 + t3] = float(1 + (7 * b + 3 * t3) % 11); Wd[b] += Sd[b * T3 + t3]; }
      SeriesIn sd;
      sd.nSky = nSky; sd.row = row.data(); sd.wsky = wsky.data(); sd.S = zS.data(); sd.P = int(nSky); sd.T = T3; sd.g = zg.data(); sd.G = zG.data(); sd.Sdc = Sd.data();
      std::vector<double> od(3);
      CHECK(bounce(plane, {half}, sa, Bounce{1, 1024}) && gpu_trace(*g, 0, 3, flags, err) && gpu_series(*g, 3, sd, E3.data(), err) && gpu_reduce_dc(*g, 3, Wd.data(), od.data(), err));
      for (size_t i = 0; i < 3 && dc.size() == nb * 3; ++i) {
        double cpu = 0;
        for (size_t b = 0; b < nb; ++b) cpu += double(dc[b * 3 + i]) * Wd[b];
        CHECK(std::fabs(od[i] - cpu) <= 1e-12 * cpu && (i == 2 ? cpu == 0 : cpu > 0));
        CHECK(std::fabs(double(E3[i * T3]) + E3[i * T3 + 1] + E3[i * T3 + 2] - od[i]) <= 1e-5 * od[i]);
      }
    }
    // a 2 km ceiling (rho 0.4) over a sensor looking up: everything reflected goes down, into the ground bin
    Tris ceiling;
    quad(ceiling, f3(-1000, -1000, 1), f3(-1000, 1000, 1), f3(1000, 1000, 1), f3(1000, -1000, 1));
    Slot c4; c4.rho = 0.4f;
    Sensors sb; sb.p = {f3(0, 0, 0)}; sb.n = {f3(0, 0, 1)};
    if (bounce(ceiling, {c4}, sb, Bounce{1, 4096})) {
      double up = 0;
      for (size_t b = 1; b < nb; ++b) up += dc[b];
      CHECK(std::fabs(dc[0] / (0.4 * kPi) - 1) <= 1e-4 && up == 0);
    }
    // sensors between a ground (rho_g, z 0) and a porous ceiling (tau, rho_c, z 2), both 200 km wide. Looking down, light
    // leaves through the ceiling after an odd number of reflections, sum_sky = pi rho_g tau sum_{k <= (N-1)/2} (rho_c rho_g)^k;
    // looking up, after an even number, the first off the ceiling: pi tau rho_c rho_g sum_{k <= (N-2)/2} (rho_c rho_g)^k,
    // so N 20 adds pi tau (rho_c rho_g)^10 to N 19 (11 sigma). The ground bin stays empty. 64 sensors of each (independent
    // seeds): their mean within 5 sigma of their spread, N 19 and 20 at high albedo (Russian roulette)
    Tris sw;
    quad(sw, f3(-1e5, -1e5, 0), f3(1e5, -1e5, 0), f3(1e5, 1e5, 0), f3(-1e5, 1e5, 0), 0);
    quad(sw, f3(-1e5, -1e5, 2), f3(1e5, -1e5, 2), f3(1e5, 1e5, 2), f3(-1e5, 1e5, 2), 1);
    Sensors sc;
    for (int k = 0; k < 128; ++k) { sc.p.push_back(f3(0.1 * (k % 64), 0, 1)); sc.n.push_back(f3(0, 0, k < 64 ? -1 : 1)); }
    const struct { float rg, rc, tau; unsigned N; } cases[] = {{0.5f, 0.4f, 0.5f, 1}, {0.5f, 0.4f, 0.5f, 2}, {0.5f, 0.4f, 0.5f, 3},
                                                               {0.9f, 0.9f, 0.1f, 19}, {0.9f, 0.9f, 0.1f, 20}};
    for (const auto& cs : cases) {
      std::vector<Slot> sl(2);
      sl[0].rho = cs.rg; sl[1].rho = cs.rc; sl[1].tau = cs.tau;
      const auto t0 = std::chrono::steady_clock::now();
      if (!bounce(sw, sl, sc, Bounce{cs.N, 4096})) continue;
      const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      for (unsigned up = 0; up < 2; ++up) {
        double expect = 0, f = 1, mean = 0, var = 0, gnd = 0;
        for (unsigned k = 0; 2 * k + 1 + up <= cs.N; ++k, f *= double(cs.rc) * cs.rg) expect += f;
        expect *= kPi * cs.tau * (up ? double(cs.rc) * cs.rg : cs.rg);
        std::vector<double> esc(64, 0.0);   // what escaped to the sky, per sensor
        for (size_t i = 0; i < 64; ++i) {
          for (size_t b = 1; b < nb; ++b) esc[i] += dc[b * 128 + 64 * up + i];
          gnd += dc[64 * up + i]; mean += esc[i] / 64;
        }
        for (double x : esc) var += (x - mean) * (x - mean) / 63;
        const double sigma = std::sqrt(var / 64);
        CHECK(gnd == 0 && std::fabs(mean - expect) <= 5 * sigma + 1e-4 * expect);
        std::printf("gpu: bounce sandwich N %u looking %s: %.5f (expected %.5f, sigma %.5f), %.0f ms\n", cs.N, up ? "up" : "down", mean, expect, sigma, ms);
      }
    }
    // glass (tau 0.64) 1 m over a ground (rho 0.5), 64 sensors under it looking up: light leaves through the glass after
    // the glass mirrored it and the ground reflected it, sum_sky = pi <R> rho_g <T> (1 + [N 2] rho_g <R>) (the mirror does
    // not count toward N; <T>, <R> glass_TR's cosine-weighted means). A 1 cm pane box (thin solid, entry face = the
    // sheet) gives the sheet's rows, its exit face being no event, and so does the sheet doubled with its twin (kSlotTwin)
    // in either order; --fresnel off (no kSlotGlass): no mirror, every path is transmit-only, nothing counts. Opaque
    // (--transmission off: tau 0) the sheet alone mirrors every path to the ground bin: pi <R>
    {
      double mT, mR;
      glass_means(float(tn_from_Tn(0.64)), mT, mR);
      Tris sheet, box, twin, twin2, pane;
      const double a0 = -1000, a1 = 1000, z0 = 1, z1 = 1.01;   // 2 km: no path from near the sensors clips the 1 cm edges
      const auto lower = [&](Tris& tr) { quad(tr, f3(a0, a0, z0), f3(a0, a1, z0), f3(a1, a1, z0), f3(a1, a0, z0), 1); };   // -z
      const auto upper = [&](Tris& tr, double z) { quad(tr, f3(a0, a0, z), f3(a1, a0, z), f3(a1, a1, z), f3(a0, a1, z), 1); };   // +z
      for (Tris* tr : {&sheet, &box, &twin, &twin2}) quad(*tr, f3(-1e4, -1e4, 0), f3(1e4, -1e4, 0), f3(1e4, 1e4, 0), f3(-1e4, 1e4, 0), 0);
      lower(sheet); lower(box); lower(twin); upper(twin, z0); upper(twin2, z0); lower(twin2); lower(pane);
      upper(box, z1);   // then the sides, outward
      quad(box, f3(a0, a0, z0), f3(a0, a0, z1), f3(a0, a1, z1), f3(a0, a1, z0), 1);
      quad(box, f3(a1, a0, z0), f3(a1, a1, z0), f3(a1, a1, z1), f3(a1, a0, z1), 1);
      quad(box, f3(a0, a0, z0), f3(a1, a0, z0), f3(a1, a0, z1), f3(a0, a0, z1), 1);
      quad(box, f3(a0, a1, z0), f3(a0, a1, z1), f3(a1, a1, z1), f3(a1, a1, z0), 1);
      std::vector<Slot> sl(2);
      sl[0].rho = 0.5f; sl[1].rho = 0; sl[1].tau = 0.64f; sl[1].tn = float(tn_from_Tn(0.64)); sl[1].flags = kSlotGlass;
      Sensors sg;
      for (int k = 0; k < 64; ++k) { sg.p.push_back(f3(0.3 * (k % 8) - 1, 0.3 * (k / 8) - 1, 0.1 + 0.0125 * k)); sg.n.push_back(f3(0, 0, 1)); }
      const auto mean = [&](size_t b0, size_t b1, double& sigma) {   // over the sensors of their sum over bins [b0, b1)
        std::vector<double> x(64, 0.0);
        double m = 0, var = 0;
        for (size_t i = 0; i < 64; ++i) { for (size_t b = b0; b < b1; ++b) x[i] += dc[b * 64 + i]; m += x[i] / 64; }
        for (double v : x) var += (v - m) * (v - m) / 63;
        sigma = std::sqrt(var / 64);
        return m;
      };
      std::vector<float> rowsSheet;
      double sigma = 0, s0 = 0;
      for (unsigned N : {1u, 2u}) {
        if (!bounce(sheet, sl, sg, Bounce{N, 4096})) continue;
        const double expect = kPi * mR * 0.5 * mT * (N == 2 ? 1 + 0.5 * mR : 1), m = mean(1, nb, sigma);
        CHECK(std::fabs(m - expect) <= 5 * sigma + 1e-4 * expect && mean(0, 1, s0) == 0);
        std::printf("gpu: bounce under glass N %u: %.5f (expected %.5f, sigma %.5f)\n", N, m, expect, sigma);
        rowsSheet = dc;
      }
      sl[1].flags = kSlotGlass | kSlotThin;
      CHECK(bounce(box, sl, sg, Bounce{2, 4096}) && dc == rowsSheet);
      sl[1].flags = kSlotGlass | kSlotThin | kSlotTwin;
      CHECK(bounce(twin, sl, sg, Bounce{2, 4096}) && dc == rowsSheet);
      CHECK(bounce(twin2, sl, sg, Bounce{2, 4096}) && dc == rowsSheet);
      sl[1].flags = 0;
      CHECK(bounce(sheet, sl, sg, Bounce{2, 4096}) && dc == std::vector<float>(nb * 64, 0.f));
      sl[1].flags = kSlotGlass; sl[1].tau = 0;
      if (bounce(pane, sl, sg, Bounce{1, 4096})) {
        const double m = mean(0, 1, sigma);
        CHECK(std::fabs(m - kPi * mR) <= 5 * sigma + 1e-4 * kPi * mR && mean(1, nb, s0) == 0);
        std::printf("gpu: opaque glass mirrors %.5f (pi <R> %.5f, sigma %.5f)\n", m, kPi * mR, sigma);
      }
    }
    // a black wall (rho 0) 1 m east of a sensor looking down 1 cm over a ground (rho 0.5): no reflected light reaches a
    // patch east of it below 72 deg, some reaches every patch west of it; --north 90 turns that east to true south. The
    // same at sky density 2
    {
      Tris tw;
      quad(tw, f3(-1000, -1000, 0), f3(1000, -1000, 0), f3(1000, 1000, 0), f3(-1000, 1000, 0), 0);
      quad(tw, f3(1, -1000, 0), f3(1, -1000, 1000), f3(1, 1000, 1000), f3(1, 1000, 0), 1);
      std::vector<Slot> sl(2);
      sl[0].rho = 0.5f; sl[1].rho = 0;
      Sensors st; st.p = {f3(0, 0, 0.01)}; st.n = {f3(0, 0, -1)};
      for (int mf : {1, 2})
        for (double north : {0.0, 90.0}) {
          if (!bounce(tw, sl, st, Bounce{1, mf == 1 ? 16384u : 65536u, mf, north})) continue;
          const Patches pm = make_patches(mf);
          int dark = 0, lit = 0, wrong = 0;
          for (size_t p = 0; p < size_t(pm.P); ++p) {
            if (pm.alt[p] > 72 * kPi / 180) continue;
            const double az = std::fmod(pm.azi[p] * 180 / kPi - north + 720, 360);   // the patch's azimuth in the model
            if (az >= 30 && az <= 150) { ++dark; wrong += dc[p + 1] != 0; }
            if (az >= 210 && az <= 330) { ++lit; wrong += !(dc[p + 1] > 0); }
          }
          CHECK(dark > 30 * mf && lit > 30 * mf && wrong == 0);
        }
    }
  }

  // empty scene: traversable 0 = every hemisphere direction visible
  CHECK(gpu_build(*g, Tris{}, {}, err) && gpu_set_slots(*g, {Slot{}}, err));
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
    CHECK(run(meshes, nullptr, o, r1, e2) && !r1.sceneReused);
    const auto t1 = std::chrono::steady_clock::now();
    CHECK(run(meshes, nullptr, o, r2, e2) && r2.sceneReused);    // same key: scene reused
    const auto t2 = std::chrono::steady_clock::now();
    o.sceneKey = "box-2";
    CHECK(run(meshes, nullptr, o, r3, e2) && !r3.sceneReused);   // new key: rebuilt, identical values
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
    o.sceneKey = "scene-A"; CHECK(run(meshes, nullptr, o, ra, e2) && !ra.sceneReused);
    o.sceneKey = "scene-B"; CHECK(run(meshes, nullptr, o, rb, e2) && !rb.sceneReused);
    o.sceneKey = "scene-A"; CHECK(run(meshes, nullptr, o, ra2, e2) && ra2.sceneReused);   // A's parked GAS
    CHECK(ra2.values == ra.values && ra.values == r1.values);
    o.residentScenes = 1;
    o.sceneKey = "scene-C"; CHECK(run(meshes, nullptr, o, ra2, e2) && !ra2.sceneReused);
    o.sceneKey = "scene-C"; CHECK(run(meshes, nullptr, o, ra2, e2) && ra2.sceneReused);
    o.sceneKey = "scene-B"; CHECK(run(meshes, nullptr, o, ra2, e2) && !ra2.sceneReused);   // one slot: A and B were freed
    o.sceneKey = "scene-A"; CHECK(run(meshes, nullptr, o, ra2, e2) && !ra2.sceneReused);

    // cancel: a set flag stops the run with "cancelled"
    std::atomic<bool> stop{true};
    o.cancel = &stop;
    e2.clear();
    CHECK(!run(meshes, nullptr, o, ra2, e2) && e2 == "cancelled");

    // one clock per run, end to end. The same instant in another --tz (or no --tz) gives the same numbers; a site 15 deg
    // east reading this weather pairs each row with the same solar time (12:00 UTC-5 there is 13:00 UTC-5 in Boston).
    o.cancel = nullptr;
    o.sceneKey.clear();
    o.quantities = {Quantity::Irradiance};
    Weather cs;
    CHECK(clear_sky_weather(42.35, -71.07, -5, 3, 0, cs, e2));
    Result h5, h4, hd, far, near13;
    o.tz = -5; CHECK(parse_hoy("06-21T12:00", false, o.t0, e2)); CHECK(run(meshes, &cs, o, h5, e2));
    o.tz = -4; CHECK(parse_hoy("06-21T13:00", false, o.t0, e2)); CHECK(run(meshes, &cs, o, h4, e2));
    o.tz = NAN; CHECK(parse_hoy("06-21T12:00", false, o.t0, e2)); CHECK(run(meshes, &cs, o, hd, e2));
    CHECK(!h5.values.empty() && h5.values == h4.values && h5.values == hd.values);
    o.tz = -5; o.lon = -56.07; CHECK(run(meshes, &cs, o, far, e2));
    o.lon = -71.07; CHECK(parse_hoy("06-21T13:00", false, o.t0, e2)); CHECK(run(meshes, &cs, o, near13, e2));
    double worst = 0;
    for (size_t i = 0; i < far.values.size() && i < near13.values.size(); ++i)
      worst = std::max(worst, std::fabs(far.values[i] - near13.values[i]) / std::max(1.0, std::fabs(near13.values[i])));
    CHECK(far.values.size() == near13.values.size() && worst < 2e-3);   // the sun differs only by one hour of declination
  }

  // run() with optics, end to end: sun hours at 06-21T12:00 Boston (the sun ~71 deg up, just west of south: a ray leaves
  // its sensor ~0.34 m sideways per metre of height), one 1 m sensor square per case, the cases 100 m apart
  {
    Options o;
    o.quantities = {Quantity::SunHours};
    o.lat = 42.35; o.lon = -71.07; o.tz = -5;
    o.transmission = true;
    std::string e2;
    CHECK(parse_hoy("06-21T12:00", false, o.t0, e2));
    const auto sun = [&](const std::vector<Mesh>& ms, const Options& op) {
      Result r; std::string e;
      if (!run(ms, nullptr, op, r, e)) { std::printf("run: %s\n", e.c_str()); ++failures; }
      return r;
    };
    const auto b255 = [](int b) { return double(float(b) / 255.f); };   // what a byte adds: dt (1 h) x byte / 255

    // two meshes x two slots: A/0 tau 1, A/1 tau 0.5, B/0 tau 0.25 (NAME/SLOT over NAME), B/1 opaque (NAME)
    Mesh A = squares("A", false, 1, 20, {0, 100}), B = squares("B", false, 1, 20, {200, 300});
    A.slot = {0, 1}; B.slot = {0, 1};
    const std::vector<Mesh> ab{squares("ground", true, 0, 0.5, {0, 100, 200, 300}), A, B};
    o.optics = {{"A", 0, 0, 1, false}, {"A", 1, 0.2, 0.5, false}, {"B", -1, 0.2, 0, false}, {"B", 0, 0.2, 0.25, false}};
    Result r = sun(ab, o);
    CHECK(r.values == std::vector<double>({1, b255(128), b255(64), 0}));
    // --transmission off: every slot opaque, the values of a run without optics
    Options off = o; off.transmission = false;
    Options plain = off; plain.optics.clear();
    const Result rOff = sun(ab, off), rPlain = sun(ab, plain);
    CHECK(rOff.values == rPlain.values && rOff.values == std::vector<double>(4, 0.0));

    // glass wall south of a sensor, turned 30 deg about z (normal (-sin 30, cos 30, 0), so every term of the device's
    // cosine counts): --fresnel off passes tau like porous; on passes glass_TR at the sun's incidence (~75 deg)
    const double s30 = 0.5, c30 = std::sqrt(0.75);
    Mesh wall; wall.name = "wall"; wall.sensor = false;
    wall.points = {-20 * c30, -1 - 20 * s30, 0, 20 * c30, -1 + 20 * s30, 0, 20 * c30, -1 + 20 * s30, 10, -20 * c30, -1 - 20 * s30, 10};
    wall.counts = {4}; wall.indices = {0, 1, 2, 3};
    const std::vector<Mesh> gw{squares("ground", true, 0, 0.5, {0}), wall};
    Options gl = o; gl.optics = {{"wall", -1, 0, 0.64, true}};
    Options gloff = gl; gloff.fresnel = false;
    Options por = o; por.optics = {{"wall", -1, 0, 0.64, false}};
    const Result rg = sun(gw, gl), rgoff = sun(gw, gloff), rpor = sun(gw, por);
    CHECK(rgoff.values == rpor.values && rpor.values == std::vector<double>({b255(163)}));
    const Sun sn = sun_noaa(42.35, -71.07, -5, 6, 21, 12.0, false);   // run()'s sun for the step
    const double cw = std::fabs(-s30 * sn.vec.x + c30 * sn.vec.y);
    float Tw, Rw;
    glass_TR(float(cw), float(tn_from_Tn(0.64)), Tw, Rw);
    CHECK(rg.values.size() == 1 && std::fabs(rg.values[0] - b255(int(std::lround(255 * Tw)))) <= 1.0 / 255 + 1e-6);
    if (rg.values.size() == 1) std::printf("run: glass wall at %.1f deg incidence passes %.3f (glass_TR %.3f, tau 0.64)\n", std::acos(cw) * 180 / kPi, rg.values[0], Tw);

    // thin solids: a 20 x 20 x 0.01 m pane box (indexed, a soup, and wound inward) passes tau once; the 3 m cube twice; a
    // double-faced sheet once. --cull-backfaces: every closed solid is crossed through one front face, tau once
    Mesh flip = box_mesh("flip", 290, -10, 1, 310, 10, 1.01, false);
    for (size_t f = 0; f < 6; ++f) std::swap(flip.indices[4 * f + 1], flip.indices[4 * f + 3]);
    Mesh twin = squares("twin", false, 1, 10, {400});
    twin.counts.push_back(4); twin.indices.insert(twin.indices.end(), {3, 2, 1, 0});
    std::vector<Mesh> th{squares("ground", true, 0, 0.5, {0, 100, 200, 300, 400}), box_mesh("pane", -10, -10, 1, 10, 10, 1.01, false),
                         box_mesh("soup", 90, -10, 1, 110, 10, 1.01, true), box_mesh("cube", 198.5, -1.5, 1, 201.5, 1.5, 4, false), flip, twin};
    for (size_t k = 1; k < th.size(); ++k) th[k].sensor = false;
    Options tho = o;
    tho.optics = {{"pane", -1, 0.2, 0.5, false}, {"soup", -1, 0.2, 0.5, false}, {"cube", -1, 0.2, 0.5, false}, {"flip", -1, 0.2, 0.5, false},
                  {"twin", -1, 0.2, 0.5, false}};
    const std::vector<double> thinV{b255(128), b255(128), b255(64), b255(128), b255(128)}, culledV(5, b255(128));
    CHECK(sun(th, tho).values == thinV);
    Options thc = tho; thc.cullBackfaces = true;
    CHECK(sun(th, thc).values == culledV);
    // a resident scene keeps its thin solids: reused as built, with culling, after a tau edit
    Options thk = tho; thk.sceneKey = "thin";
    r = sun(th, thk); CHECK(!r.sceneReused && r.values == thinV);
    r = sun(th, thk); CHECK(r.sceneReused && r.values == thinV);
    thk.cullBackfaces = true; r = sun(th, thk); CHECK(r.sceneReused && r.values == culledV);
    thk.cullBackfaces = false; thk.optics[0].tau = 0.25; r = sun(th, thk);
    CHECK(r.sceneReused && r.values == std::vector<double>({b255(64), b255(128), b255(64), b255(128), b255(128)}));

    // --dump-bits with a transmitting slot: uint8 rows [sensors x directions], here three sun steps of the two-mesh case;
    // a transmitting slot without faces in the GAS (the sensor mesh under --no-self-occlude) keeps the uint32 bits
    const auto npy = [](const std::string& path, std::string& head, std::vector<uint8_t>& data) {
      std::FILE* f = std::fopen(path.c_str(), "rb");
      if (!f) return false;
      std::vector<uint8_t> b;
      for (int ch; (ch = std::fgetc(f)) != EOF;) b.push_back(uint8_t(ch));
      std::fclose(f);
      const size_t n = b.size() < 10 ? 0 : 10 + size_t(b[8] | (b[9] << 8));   // magic, version, header length, header
      if (n == 0 || b.size() < n) return false;
      head.assign(b.begin() + 10, b.begin() + std::ptrdiff_t(n)); data.assign(b.begin() + std::ptrdiff_t(n), b.end());
      return true;
    };
    Options od = o; od.t1 = od.t0 + 3; od.dumpBitsPath = std::string(testsDir) + "/_check_gpu_tmp.npy";
    std::string head;
    std::vector<uint8_t> data;
    r = sun(ab, od);
    CHECK(r.values == std::vector<double>({3, 3 * b255(128), 3 * b255(64), 0}));
    CHECK(npy(od.dumpBitsPath, head, data) && head.find("'descr': '<u1'") != std::string::npos && head.find("'shape': (4, 3)") != std::string::npos);
    CHECK(data == std::vector<uint8_t>({255, 255, 255, 128, 128, 128, 64, 64, 64, 0, 0, 0}));
    Options ou = od; ou.selfOcclude = false; ou.optics = {{"ground", -1, 0.2, 0.5, false}};
    r = sun(ab, ou);
    CHECK(r.values == std::vector<double>(4, 0.0));
    CHECK(npy(ou.dumpBitsPath, head, data) && head.find("'descr': '<u4'") != std::string::npos && head.find("'shape': (4, 1)") != std::string::npos);
    std::remove(od.dumpBitsPath.c_str());

    // residency: a tau edit on the same key reuses the scene; a slot that starts to transmit rebuilds; no key never reuses
    const std::vector<Mesh> lid{squares("ground", true, 0, 0.5, {0}), squares("lid", false, 1, 20, {0})};
    Options k1 = o; k1.sceneKey = "lid";
    k1.optics = {{"lid", -1, 0.2, 0.3, false}}; r = sun(lid, k1);
    CHECK(!r.sceneReused && r.values.size() == 1 && std::fabs(r.values[0] - 0.3) <= 0.5 / 255 + 1e-6);
    k1.optics = {{"lid", -1, 0.2, 0.5, false}}; r = sun(lid, k1);
    CHECK(r.sceneReused && r.values == std::vector<double>({b255(128)}));
    k1.sceneKey = "lid-2"; k1.optics = {{"lid", -1, 0.2, 0, false}}; r = sun(lid, k1);
    CHECK(!r.sceneReused && r.values == std::vector<double>({0.0}));
    k1.optics = {{"lid", -1, 0.2, 0.5, false}}; r = sun(lid, k1);
    CHECK(!r.sceneReused && r.values == std::vector<double>({b255(128)}));
    k1.optics = {{"lid", -1, 0.2, 0.25, false}}; r = sun(lid, k1);
    CHECK(r.sceneReused && r.values == std::vector<double>({b255(64)}));
    k1.sceneKey.clear(); r = sun(lid, k1);
    CHECK(!r.sceneReused); r = sun(lid, k1);
    CHECK(!r.sceneReused && r.values == std::vector<double>({b255(64)}));

    // reflected light end to end (June, clear sky, sensors 1 m up, not blocking): a face looking down over a ground mesh
    // with one bounce gets about what the virtual ground gives it without the mesh; one looking up gets nothing reflected;
    // direct and sun hours stay; the series sums to the cumulative value, in W/m2 or, illuminance first, in lux. Ground
    // halves of rho 0.3 (x > 0) / 0.1 read their own slot: the face looking down 1 m east of their border sees the west
    // half with F = (1 - 1 / sqrt 2) / 2 of its view (one record would give 0.3 or 0.1 everywhere, swapped records the
    // other's value); the per-slot layout keys a resident scene of its own
    Weather cs;
    CHECK(clear_sky_weather(42.35, -71.07, -5, 3, 0, cs, e2));
    const auto go = [&](const std::vector<Mesh>& ms, const Options& op) {
      Result rr; std::string e;
      if (!run(ms, &cs, op, rr, e)) { std::printf("run: %s\n", e.c_str()); ++failures; }
      return rr;
    };
    Mesh s2; s2.name = "s2";
    s2.points = {0.5, -0.5, 1, 1.5, -0.5, 1, 1.5, 0.5, 1, 0.5, 0.5, 1, -0.5, 4.5, 1, 0.5, 4.5, 1, 0.5, 5.5, 1, -0.5, 5.5, 1};
    s2.counts = {4, 4}; s2.indices = {3, 2, 1, 0, 4, 5, 6, 7};   // face 0 looks down at (1, 0, 1), face 1 up at (0, 5, 1)
    const std::vector<Mesh> halves{s2, squares("east", false, 0, 500, {500}), squares("west", false, 0, 500, {-500})};
    Options ob;
    ob.quantities = {Quantity::Radiation, Quantity::Direct, Quantity::SunHours};
    ob.selfOcclude = false; ob.bounceSamples = 65536;
    CHECK(parse_hoy("06-01T00:00", false, ob.t0, e2) && parse_hoy("07-01T00:00", false, ob.t1, e2));
    const Result virt = go({s2}, ob);   // no ground mesh: the virtual ground (rho 0.2)
    ob.bounces = 1; ob.seriesPath = std::string(testsDir) + "/_check_gpu_series.npy";
    const Result mesh = go(halves, ob);   // ground mesh of the default rho 0.2: one hit record
    CHECK(virt.values.size() == 6 && mesh.values.size() == 6);
    if (virt.values.size() == 6 && mesh.values.size() == 6) {
      CHECK(std::fabs(mesh.values[0] / virt.values[0] - 1) <= 0.03 && mesh.values[1] == 0 && mesh.values[2] == 0);
      CHECK(std::vector<double>(mesh.values.begin() + 3, mesh.values.end()) == std::vector<double>(virt.values.begin() + 3, virt.values.end()));
      std::printf("run: face looking down, June: ground mesh + 1 bounce %.3f kWh/m2, virtual ground %.3f\n", mesh.values[0], virt.values[0]);
      CHECK(npy(ob.seriesPath, head, data) && data.size() % 8 == 0);
      const size_t tup = data.size() / 8;   // float32 [2 x Tup], W/m2 at 1 h steps
      for (size_t e = 0; e < 2 && tup; ++e) {
        double kwh = 0;
        for (size_t j = 0; j < tup; ++j) { float x; std::memcpy(&x, &data[4 * (e * tup + j)], 4); kwh += x / 1000.0; }
        CHECK(std::fabs(kwh / mesh.values[3 * e] - 1) <= 1e-5);
      }
    }
    Options ol = ob; ol.quantities = {Quantity::Illuminance};
    const Result lux = go(halves, ol);
    CHECK(lux.values.size() == 2 && npy(ol.seriesPath, head, data) && data.size() % 8 == 0);
    for (size_t e = 0, tup = data.size() / 8; e < 2 && tup && lux.values.size() == 2; ++e) {
      double sum = 0;   // lux at 1 h steps
      for (size_t j = 0; j < tup; ++j) { float x; std::memcpy(&x, &data[4 * (e * tup + j)], 4); sum += x; }
      CHECK(lux.values[e] > 0 && std::fabs(sum / lux.durationHours / lux.values[e] - 1) <= 1e-5);
    }
    std::remove(ob.seriesPath.c_str());
    std::remove((ob.seriesPath + ".hoys.txt").c_str());
    Options hk = ob; hk.seriesPath.clear(); hk.sceneKey = "halves";
    hk.optics = {{"east", -1, 0.3, 0, false}, {"west", -1, 0.1, 0, false}};
    const Result h1 = go(halves, hk);
    hk.optics = {{"east", -1, 0.1, 0, false}, {"west", -1, 0.3, 0, false}};
    const Result h2 = go(halves, hk);
    hk.bounces = 0;
    const Result h3 = go(halves, hk);
    hk.bounces = 1; hk.optics = {{"*", -1, 0.2, 0, false}};
    const Result h4 = go(halves, hk);
    CHECK(h1.values.size() == 6 && h2.values.size() == 6 && h3.values.size() == 6 && mesh.values.size() == 6);
    if (h1.values.size() == 6 && h2.values.size() == 6 && h3.values.size() == 6 && mesh.values.size() == 6) {
      const double F = (1 - std::sqrt(0.5)) / 2, x1 = (0.3 * (1 - F) + 0.1 * F) / 0.2, x2 = (0.1 * (1 - F) + 0.3 * F) / 0.2;
      CHECK(!h1.sceneReused && std::fabs(h1.values[0] / (x1 * mesh.values[0]) - 1) <= 0.02);
      CHECK(h2.sceneReused && std::fabs(h2.values[0] / (x2 * mesh.values[0]) - 1) <= 0.02);
      CHECK(!h3.sceneReused && h3.values[0] == 0);   // bounces 0: the one-record layout, another scene
      CHECK(h4.sceneReused && h4.values == mesh.values);   // one rho again: h3's scene, its one record at 0.2
      std::printf("run: ground halves 0.3 / 0.1 and 0.1 / 0.3: %.4f, %.4f x all 0.2 (expected %.4f, %.4f)\n", h1.values[0] / mesh.values[0],
                  h2.values[0] / mesh.values[0], x1, x2);
    }
    // glass is opaque with --transmission off but still mirrors R(cos): a face looking up under a 2 km pane gets <R> x
    // what the virtual ground gives it looking down (64 samples per patch: its hemisphere sums to pi). The pane is the only
    // slot in the GAS, so its one hit record must carry the glass
    double mT, mR;
    glass_means(float(tn_from_Tn(0.64)), mT, mR);
    const std::vector<Mesh> gp{squares("face", true, 0.5, 0.5, {0}), squares("pane", false, 3, 1000, {0})};
    Options og = ob; og.quantities = {Quantity::Radiation}; og.seriesPath.clear(); og.bounceSamples = 4096; og.optics = {{"pane", -1, 0, 0.64, true}};
    Options ov = og; ov.bounces = 0; ov.optics.clear(); ov.sides = Sides::Back; ov.patchSamples = 64;
    const Result rm = go(gp, og), rv = go({gp[0]}, ov);
    CHECK(rm.values.size() == 1 && rv.values.size() == 1);
    if (rm.values.size() == 1 && rv.values.size() == 1) {
      CHECK(std::fabs(rm.values[0] / (mR * rv.values[0]) - 1) <= 2e-3);
      std::printf("run: opaque glass mirrors %.5f of the virtual ground (<R> %.5f)\n", rm.values[0] / rv.values[0], mR);
    }
  }
  return failures;
}
