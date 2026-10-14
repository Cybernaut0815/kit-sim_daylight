// SPDX-License-Identifier: Apache-2.0 AND LicenseRef-Commons-Clause-1.0
// Composition: direction table, per-quantity weights (physical and ladybug modes), chunked GPU loop, series, dumps.
#include "internal.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>

namespace daylight {
namespace {

using File = std::unique_ptr<std::FILE, decltype(&std::fclose)>;

constexpr double kPi = 3.14159265358979323846;
constexpr double kDeg = kPi / 180.0;
constexpr double kRf = 0.265074126, kGf = 0.670114631, kBf = 0.064811243;   // Radiance color.h CIE_rf/gf/bf
constexpr double kWhtEfficacy = 179.0;                                        // lm/W, Radiance WHTEFFICACY
constexpr double kNight = 1e-4;                                               // gendaymtx: dir+dif <= 1e-4

inline double bright(const float* rgb) { return kRf * rgb[0] + kGf * rgb[1] + kBf * rgb[2]; }
inline float3 neg(float3 v) { return float3{-v.x, -v.y, -v.z}; }

// Process-lifetime GPU state. Created on first use and never destroyed: CUDA / OptiX teardown from a static
// destructor runs after the runtime has unloaded and can crash at exit, and the OS reclaims the device anyway.
// gpu_build / upload / grow free or reuse their previous buffers, so one Gpu serves every run of the process.
// `key` is the Options::sceneKey whose occluder triangles / GAS are resident (empty = none). Not thread-safe:
// run() is one-at-a-time, as the CLI and the server (one GPU, a mutex) use it.
// `cur` is the scene installed in the Gpu; `parked` holds up to Options::residentScenes - 1 other scenes' GAS in VRAM,
// most recent first, so users who alternate between scenes do not rebuild each other's.
struct Scene {
  std::string key;
  double c[3] = {0, 0, 0};   // bbox centre the scene's triangles were recentred by
  size_t nTris = 0;
  int skipped = 0;
  GasSlot gas;               // parked scenes only
};
struct Resident {
  Gpu* gpu = nullptr;
  Scene cur;
  std::vector<Scene> parked;
};
Resident& resident() { static Resident r; return r; }

}  // namespace

// NumPy .npy v1.0 header (little-endian, C order). descr "<f4", "<f8" or "<u4". Data rows follow.
bool npy_write_header(std::FILE* f, size_t rows, size_t cols, const char* descr) {
  char dict[256];
  int n = std::snprintf(dict, sizeof dict, "{'descr': '%s', 'fortran_order': False, 'shape': (%zu, %zu), }", descr, rows, cols);
  if (n <= 0) return false;
  size_t total = 10 + static_cast<size_t>(n) + 1;                  // magic(6) + version(2) + len(2) + dict + '\n'
  size_t pad = (64 - total % 64) % 64;
  uint16_t hlen = static_cast<uint16_t>(static_cast<size_t>(n) + pad + 1);
  if (std::fwrite("\x93NUMPY\x01\x00", 1, 8, f) != 8) return false;
  if (std::fwrite(&hlen, 2, 1, f) != 1) return false;
  if (std::fwrite(dict, 1, static_cast<size_t>(n), f) != static_cast<size_t>(n)) return false;
  for (size_t i = 0; i < pad; ++i) std::fputc(' ', f);
  std::fputc('\n', f);
  return true;
}

bool scene_resident(const std::string& key) {
  const Resident& R = resident();
  if (key.empty() || !R.gpu) return false;
  return key == R.cur.key || std::any_of(R.parked.begin(), R.parked.end(), [&](const Scene& s) { return s.key == key; });
}

bool run(const std::vector<Mesh>& meshes, const Weather* w, const Options& opt, Result& out, std::string& err) {
  const auto cancelled = [&] { if (opt.cancel && opt.cancel->load()) { err = "cancelled"; return true; } return false; };
  // ---- validation --------------------------------------------------------------------------------------------
  const bool ladybug = opt.mode == Mode::Ladybug;
  bool needWeather = false, wantVis = false, wantRad = false;
  for (Quantity q : opt.quantities)
    if (q != Quantity::SunHours) { needWeather = true; if (q == Quantity::Illuminance) wantVis = true; else wantRad = true; }
  if (needWeather && !w) { err = "weather (--epw) required for radiation/irradiance/illuminance"; return false; }
  if (ladybug && wantVis) { err = "illuminance is not available in ladybug mode (RadiationStudy is radiometric)"; return false; }
  if (ladybug && opt.mf != 1 && opt.mf != 2) { err = "ladybug mode supports --sky-density 1 or 2 only"; return false; }
  if (!opt.dumpSky.empty() && !(ladybug && needWeather)) { err = "--dump-sky is ladybug-mode only (with weather)"; return false; }
  if (opt.mf != 1 && opt.mf != 2 && opt.mf != 4) { err = "--sky-density must be 1, 2 or 4"; return false; }
  const int K = ladybug ? 1 : std::max(1, opt.patchSamples);
  {
    int m = static_cast<int>(std::lround(std::sqrt(static_cast<double>(K))));
    if (m * m != K) { err = "--patch-samples must be a square number (1, 4, 9, 16, ...)"; return false; }
  }
  double lat = w ? w->lat : opt.lat, lon = w ? w->lon : opt.lon, tz = w ? w->tz : opt.tz;
  if (!std::isnan(opt.lat)) lat = opt.lat;
  if (!std::isnan(opt.lon)) lon = opt.lon;
  if (!std::isnan(opt.tz)) tz = opt.tz;
  if (std::isnan(lat) || std::isnan(lon) || std::isnan(tz)) { err = "location unknown: give --epw or --lat --lon --tz"; return false; }
  const bool leap = w ? w->leap : false;

  // ---- steps ---------------------------------------------------------------------------------------------------
  std::vector<Step> steps;
  if (!make_steps(opt, w, steps, err)) return false;
  if (steps.empty()) { err = "no timesteps selected"; return false; }
  const size_t T = steps.size();
  double durationHours = 0;
  for (const Step& s : steps) durationHours += s.dt;
  out.durationHours = durationHours;

  // ---- geometry ------------------------------------------------------------------------------------------------
  Resident& R = resident();
  const std::string& key = opt.sceneKey;
  const bool installed = !key.empty() && R.gpu && key == R.cur.key;   // same meshes + selfOcclude as the scene in the Gpu
  const auto parkedAt = [&] { return std::find_if(R.parked.begin(), R.parked.end(), [&](const Scene& s) { return s.key == key; }); };
  const bool reuse = installed || (!key.empty() && R.gpu && parkedAt() != R.parked.end());
  double c[3];
  Tris tris;
  int skipped = 0;
  if (reuse) { const Scene& s = installed ? R.cur : *parkedAt(); std::copy(s.c, s.c + 3, c); skipped = s.skipped; }
  else {
    bbox_centre(meshes, c);
    for (const Mesh& m : meshes)
      if (!m.sensor || opt.selfOcclude) triangulate(m, c, tris, skipped);
  }
  Sensors sens;
  std::string warn;
  out.ranges.clear();
  for (const Mesh& m : meshes) {
    if (!m.sensor) continue;
    std::string wmsg;
    size_t n = make_sensors(m, opt, c, sens, wmsg);
    if (wmsg.rfind("error:", 0) == 0) { err = wmsg; return false; }
    if (!wmsg.empty()) warn += (warn.empty() ? "" : "; ") + wmsg;
    out.ranges.emplace_back(m.name, n);
  }
  const int nS = opt.sides == Sides::Both ? 2 : 1;
  const size_t nSens = sens.p.size();
  if (nSens == 0) { err = "no sensor elements (no sensor meshes, or all faces degenerate)"; return false; }
  if (opt.verbose) {
    std::fprintf(stderr, "info: %zu sensors, %zu triangles%s, %zu steps (%.1f h)\n", nSens, reuse ? (installed ? R.cur : *parkedAt()).nTris : tris.i.size(),
                 reuse ? " (scene reused)" : "", T, durationHours);
    if (skipped) std::fprintf(stderr, "warn: %d degenerate faces skipped in occluders\n", skipped);
    if (!warn.empty()) std::fprintf(stderr, "warn: %s\n", warn.c_str());
  }

  // ---- sun positions per step (NOAA, ladybug semantics) ----------------------------------------------------------
  std::vector<Sun> suns(T);
  for (size_t t = 0; t < T; ++t) {
    suns[t] = sun_noaa(lat, lon, tz, steps[t].month, steps[t].day, steps[t].hour, leap);
    suns[t].vec = rotate_z(suns[t].vec, opt.north);
  }

  // ---- direction table: [sky P*K | ground P*K (= -sky) | sun] ------------------------------------------------------
  if (cancelled()) return false;
  Patches pa = make_patches(opt.mf);
  const int P = pa.P;
  std::vector<float3> D;
  std::vector<int> rowOf;                        // patch index per sky direction
  size_t nSky = 0;
  if (needWeather) {
    for (int p = 0; p < P; ++p)
      for (int k = 0; k < K; ++k) {
        float3 v = ladybug ? pa.lb[static_cast<size_t>(p)] : patch_sample(pa, p, k, K);
        D.push_back(rotate_z(v, opt.north));
        rowOf.push_back(p);
      }
    nSky = D.size();
    for (size_t d = 0; d < nSky; ++d) D.push_back(neg(D[d]));
  }
  const size_t sun0 = D.size();
  std::vector<int> sunStep;                      // step index per sun direction
  for (size_t t = 0; t < T; ++t)
    if (suns[t].up) { D.push_back(suns[t].vec); sunStep.push_back(static_cast<int>(t)); }
  const size_t nSun = sunStep.size();
  const size_t nDir = D.size();
  out.nSky = nSky; out.nSun = nSun;

  // ---- sky model per step ----------------------------------------------------------------------------------------
  // Series columns: non-night steps. S is radiometric (W/m2/sr) unless the first quantity is illuminance (then lux/sr).
  const bool series = !opt.seriesPath.empty();
  const bool seriesVis = series && opt.quantities[0] == Quantity::Illuminance;
  std::vector<int> colOf(T, -1);
  int Tup = 0;
  for (size_t t = 0; t < T; ++t)
    if (steps[t].dni + steps[t].dhi > kNight) colOf[t] = Tup++;
  const size_t Pz = static_cast<size_t>(P);
  std::vector<float> S;                          // [P x Tup]
  std::vector<float> gSeries;                    // [Tup]
  if (series) { S.assign(Pz * static_cast<size_t>(Tup), 0.f); gSeries.assign(static_cast<size_t>(Tup), 0.f); }
  // per-direction cumulative weights (double): sky radiometric/visible, ground scalar, sun per direction
  std::vector<double> skyRad(Pz, 0.0), skyVis(Pz, 0.0), skyDirLB(Pz, 0.0), skyDifLB(Pz, 0.0);
  double gndRad = 0, gndVis = 0, gndLB = 0;
  std::vector<double> sunRad(nSun, 0.0), sunVis(nSun, 0.0);
  std::vector<float> buf(3 * (Pz + 1)), buf2(3 * (Pz + 1));
  // ladybug emulation state (gendaymtx -A float accumulation over all rows, wea-rounded location/irradiance)
  std::vector<float> accDir(Pz, 0.f), accDif(3 * Pz, 0.f);
  std::vector<float> lbDif3g;                    // for --dump-sky: averaged RGB (ground row first), dif then dir
  const double latW = round_fmt(lat, "%.2f"), lonWestW = round_fmt(-lon, "%.2f"), meridW = static_cast<double>(static_cast<int>(-tz * 15.0));
  std::vector<float> lbGroundDif(3, 0.f), lbGroundDir(3, 0.f);   // float accumulators of the ground row (gendaymtx row 0)
  double droppedDni = 0;
  std::vector<size_t> sunIdxOfStep(T, SIZE_MAX);
  for (size_t j = 0; j < nSun; ++j) sunIdxOfStep[static_cast<size_t>(sunStep[j])] = j;

  if (needWeather) {
    for (size_t t = 0; t < T; ++t) {
      const Step& s = steps[t];
      const double dt = s.dt;
      const int col = series ? colOf[t] : -1;   // series columns only when a series file is written
      if (ladybug) {
        // gendaymtx reads the ladybug-written .wea: lat/lon %.2f (lon west-positive), time_zone (int)(-tz*15), DNI/DHI %d.
        double altS, aziS;
        sun_radiance(latW, lonWestW, meridW, s.jd, s.hour, altS, aziS);
        const double dniW = static_cast<double>(static_cast<int>(s.dni)), dhiW = static_cast<double>(static_cast<int>(s.dhi));
        gendaymtx_step(pa, s.jd, altS, aziS, dniW, dhiW, opt.ground, true, false, false, buf.data());    // -d
        gendaymtx_step(pa, s.jd, altS, aziS, dniW, dhiW, opt.ground, false, true, false, buf2.data());   // -s
        for (int ch = 0; ch < 3; ++ch) { lbGroundDir[static_cast<size_t>(ch)] += buf[static_cast<size_t>(ch)]; lbGroundDif[static_cast<size_t>(ch)] += buf2[static_cast<size_t>(ch)]; }
        double meanPatch = 0;
        for (size_t p = 0; p < Pz; ++p) {
          const float* dr = &buf[3 * (p + 1)];
          const float* df = &buf2[3 * (p + 1)];
          accDir[p] += dr[0];
          for (int ch = 0; ch < 3; ++ch) accDif[3 * p + static_cast<size_t>(ch)] += df[ch];
          if (col >= 0) {
            const double Lp = bright(df) + dr[0];
            S[p * static_cast<size_t>(Tup) + static_cast<size_t>(col)] = static_cast<float>(Lp);
            meanPatch += Lp * pa.omega[p];
          }
        }
        if (col >= 0) gSeries[static_cast<size_t>(col)] = static_cast<float>(opt.ground * meanPatch / static_cast<double>(P));
        continue;
      }
      // physical: Perez sky at the NOAA sun; sun kept exact (not binned)
      const double alt = suns[t].alt * kDeg, azi = suns[t].azi * kDeg;
      if (wantRad || (series && !seriesVis)) {
        gendaymtx_step(pa, s.jd, alt, azi, s.dni, s.dhi, opt.ground, false, true, false, buf.data());
        const double g = bright(buf.data());                        // ground radiance W/m2/sr (incl. direct on ground)
        gndRad += g * dt / 1000.0;
        for (size_t p = 0; p < Pz; ++p) {
          const double Lp = bright(&buf[3 * (p + 1)]);
          skyRad[p] += Lp * dt / 1000.0;
          if (col >= 0 && !seriesVis) S[p * static_cast<size_t>(Tup) + static_cast<size_t>(col)] = static_cast<float>(Lp);
        }
        if (col >= 0 && !seriesVis) gSeries[static_cast<size_t>(col)] = static_cast<float>(g);
      }
      if (wantVis || seriesVis) {
        gendaymtx_step(pa, s.jd, alt, azi, s.dni, s.dhi, opt.ground, false, true, true, buf.data());
        const double g = kWhtEfficacy * bright(buf.data());        // ground luminance-equivalent lux/sr
        gndVis += g * dt;
        for (size_t p = 0; p < Pz; ++p) {
          const double Lp = kWhtEfficacy * bright(&buf[3 * (p + 1)]);
          skyVis[p] += Lp * dt;
          if (col >= 0 && seriesVis) S[p * static_cast<size_t>(Tup) + static_cast<size_t>(col)] = static_cast<float>(Lp);
        }
        if (col >= 0 && seriesVis) gSeries[static_cast<size_t>(col)] = static_cast<float>(g);
        // direct illuminance: AddDirect conserves energy, sum_p out*omega = dir_illum/179
        gendaymtx_step(pa, s.jd, alt, azi, s.dni, s.dhi, opt.ground, true, false, true, buf2.data());
        double dirIll = 0;
        for (size_t p = 0; p < Pz; ++p) dirIll += buf2[3 * (p + 1) + 1] * pa.omega[p];
        if (sunIdxOfStep[t] != SIZE_MAX) sunVis[sunIdxOfStep[t]] = kWhtEfficacy * dirIll * dt;
      }
      if (sunIdxOfStep[t] != SIZE_MAX) sunRad[sunIdxOfStep[t]] = s.dni * dt / 1000.0;
      else if (s.dni > 0) droppedDni += s.dni * dt;
    }
    if (ladybug) {
      // gendaymtx -A: float average over ALL rows, ASCII %.3g, ladybug: broadband * omega * hours / 1000 (kWh/m2)
      const float inv = static_cast<float>(1.0 / static_cast<double>(T));
      lbDif3g.assign(3 * (Pz + 1) * 2, 0.f);
      for (int ch = 0; ch < 3; ++ch) {
        lbDif3g[static_cast<size_t>(ch)] = static_cast<float>(round_fmt(T > 1 ? lbGroundDif[static_cast<size_t>(ch)] * inv : lbGroundDif[static_cast<size_t>(ch)], "%.3g"));
        lbDif3g[3 * (Pz + 1) + static_cast<size_t>(ch)] = static_cast<float>(round_fmt(T > 1 ? lbGroundDir[static_cast<size_t>(ch)] * inv : lbGroundDir[static_cast<size_t>(ch)], "%.3g"));
      }
      for (size_t p = 0; p < Pz; ++p) {
        float avgDir = T > 1 ? accDir[p] * inv : accDir[p];
        float r = T > 1 ? accDif[3 * p] * inv : accDif[3 * p];
        float gch = T > 1 ? accDif[3 * p + 1] * inv : accDif[3 * p + 1];
        float b = T > 1 ? accDif[3 * p + 2] * inv : accDif[3 * p + 2];
        const double rd = round_fmt(avgDir, "%.3g"), rr = round_fmt(r, "%.3g"), rg = round_fmt(gch, "%.3g"), rb = round_fmt(b, "%.3g");
        lbDif3g[3 * (p + 1)] = static_cast<float>(rr); lbDif3g[3 * (p + 1) + 1] = static_cast<float>(rg); lbDif3g[3 * (p + 1) + 2] = static_cast<float>(rb);
        lbDif3g[3 * (Pz + 1) + 3 * (p + 1)] = lbDif3g[3 * (Pz + 1) + 3 * (p + 1) + 1] = lbDif3g[3 * (Pz + 1) + 3 * (p + 1) + 2] = static_cast<float>(rd);
        skyDirLB[p] = rd * pa.omega[p] * durationHours / 1000.0;
        skyDifLB[p] = (kRf * rr + kGf * rg + kBf * rb) * pa.omega[p] * durationHours / 1000.0;
        skyRad[p] = skyDirLB[p] + skyDifLB[p];
      }
      double mean = 0;
      for (size_t p = 0; p < Pz; ++p) mean += skyRad[p];
      gndLB = opt.ground * mean / static_cast<double>(P);
    }
    if (opt.verbose && droppedDni > 0)
      std::fprintf(stderr, "warn: %.1f Wh/m2 of DNI at steps with the sun below the horizon dropped (gendaymtx would bin it)\n", droppedDni);
  }

  // ---- weight vector per quantity --------------------------------------------------------------------------------
  const size_t nQ = opt.quantities.size();
  std::vector<std::vector<float>> Wq(nQ, std::vector<float>(nDir, 0.f));
  std::vector<float> Wg(nDir, 0.f);              // ground visibility weights for the series term
  for (size_t d = nSky; d < 2 * nSky; ++d)
    Wg[d] = ladybug ? 1.f : static_cast<float>(pa.omega[static_cast<size_t>(rowOf[d - nSky])] / K);
  for (size_t qi = 0; qi < nQ; ++qi) {
    const Quantity q = opt.quantities[qi];
    std::vector<float>& W = Wq[qi];
    if (q == Quantity::SunHours) {
      for (size_t j = 0; j < nSun; ++j) W[sun0 + j] = static_cast<float>(steps[static_cast<size_t>(sunStep[j])].dt);
      continue;
    }
    const bool vis = q == Quantity::Illuminance;
    const bool wantSky = q != Quantity::Direct || ladybug;
    const bool wantSun = q != Quantity::Diffuse;
    for (size_t d = 0; d < nSky; ++d) {
      const size_t p = static_cast<size_t>(rowOf[d]);
      double v;
      if (ladybug) v = q == Quantity::Direct ? skyDirLB[p] : q == Quantity::Diffuse ? skyDifLB[p] : skyRad[p];
      else v = wantSky ? (vis ? skyVis[p] : skyRad[p]) * pa.omega[p] / K : 0.0;
      W[d] = static_cast<float>(v);
    }
    if (q != Quantity::Direct)
      for (size_t d = nSky; d < 2 * nSky; ++d) {
        const size_t p = static_cast<size_t>(rowOf[d - nSky]);
        W[d] = static_cast<float>(ladybug ? gndLB : (vis ? gndVis : gndRad) * pa.omega[p] / K);
      }
    if (!ladybug && wantSun)
      for (size_t j = 0; j < nSun; ++j) W[sun0 + j] = static_cast<float>(vis ? sunVis[j] : sunRad[j]);
  }

  // ---- GPU -----------------------------------------------------------------------------------------------------
  std::vector<double> acc(nSens * nQ, 0.0);
  std::vector<uint32_t> bitsBuf, bitRows;
  const size_t nWords = (nDir + 31) / 32;
  File sf(nullptr, std::fclose), bf(nullptr, std::fclose);   // series .npy [nSens x Tup] f4; bits .npy [nSens x nWords] u4
  std::vector<double> seriesHoys;
  if (series) {
    sf.reset(std::fopen(opt.seriesPath.c_str(), "wb"));
    if (!sf || !npy_write_header(sf.get(), nSens, static_cast<size_t>(Tup), "<f4")) { err = "cannot write " + opt.seriesPath; return false; }
    for (size_t t = 0; t < T; ++t) if (colOf[t] >= 0) seriesHoys.push_back(steps[t].hoy);
  }
  if (!opt.dumpBitsPath.empty()) {
    bf.reset(std::fopen(opt.dumpBitsPath.c_str(), "wb"));
    if (!bf || !npy_write_header(bf.get(), nSens, nWords, "<u4")) { err = "cannot write " + opt.dumpBitsPath; return false; }
  }
  if (nDir > 0) {
    if (cancelled()) return false;
    if (!R.gpu && !(R.gpu = gpu_init(err))) return false;
    Gpu* g = R.gpu;
    if (!installed) {
      // park the installed keyed scene when more than one may stay resident; an unparked GAS is freed by attach / build
      if (!R.cur.key.empty() && opt.residentScenes > 1) { Scene s = R.cur; s.gas = gpu_detach(*g); R.parked.insert(R.parked.begin(), s); }
      R.cur = Scene{};   // a failed build leaves nothing installed
      if (auto it = parkedAt(); reuse && it != R.parked.end()) {
        Scene s = *it;
        R.parked.erase(it);
        gpu_attach(*g, s.gas);
        s.gas = GasSlot{};
        R.cur = s;
      } else {
        if (!gpu_build(*g, tris, err)) return false;
        R.cur.key = key; std::copy(c, c + 3, R.cur.c); R.cur.nTris = tris.i.size(); R.cur.skipped = skipped;
      }
      // ponytail: 1 GB VRAM floor for parked scenes; derive it from the run's own needs if large scenes start to fail
      while (!R.parked.empty() && (R.parked.size() + 1 > size_t(std::max(1, opt.residentScenes)) || gpu_free_bytes() < (size_t(1) << 30))) {
        gpu_free(R.parked.back().gas);
        R.parked.pop_back();
      }
    }
    if (!gpu_set_scene(*g, sens, D, err)) return false;
    // ponytail: 256 MB cap on both the bits buffer and the series tile; make it a device-memory fraction if it ever thrashes
    const size_t ncMax = std::max<size_t>(1, (size_t(256) << 20) / (4 * std::max<size_t>(1, std::max(nWords, series ? size_t(Tup) : size_t(0)))));
    const unsigned flags = gpu_ray_flags(opt.cullBackfaces);
    std::vector<double> tmp;
    std::vector<float> G, E;
    std::vector<float3> sunVec(nSun);
    std::vector<float> sunW(nSun, 0.f);
    std::vector<int> sunCol(nSun, 0);
    for (size_t j = 0; j < nSun; ++j) {
      sunVec[j] = D[sun0 + j];
      const int cidx = colOf[static_cast<size_t>(sunStep[j])];
      sunCol[j] = cidx < 0 ? 0 : cidx;
      const double dtj = steps[static_cast<size_t>(sunStep[j])].dt;
      // per-step sun strength (W/m2 or lux) = cumulative weight / dt; ladybug mode: sun already inside the patches
      sunW[j] = (ladybug || cidx < 0 || dtj <= 0) ? 0.f : static_cast<float>((seriesVis ? sunVis[j] : sunRad[j] * 1000.0) / dtj);
    }
    for (size_t i0 = 0; i0 < nSens; i0 += ncMax) {
      if (cancelled()) return false;
      const size_t nc = std::min(ncMax, nSens - i0);
      if (!gpu_trace(*g, i0, nc, flags, err)) return false;
      tmp.assign(nc, 0.0);
      for (size_t qi = 0; qi < nQ; ++qi) {
        if (!gpu_reduce(*g, nc, Wq[qi].data(), opt.quantities[qi] != Quantity::SunHours, tmp.data(), err)) return false;
        for (size_t i = 0; i < nc; ++i) acc[(i0 + i) * nQ + qi] = tmp[i];
      }
      if (bf) {   // word-major on the device -> row-major [nc x nWords] rows
        if (!gpu_download_bits(*g, nc, bitsBuf, err)) return false;
        bitRows.resize(nc * nWords);
        for (size_t wi = 0; wi < nWords; ++wi)
          for (size_t i = 0; i < nc; ++i) bitRows[i * nWords + wi] = bitsBuf[wi * nc + i];
        if (std::fwrite(bitRows.data(), 4, bitRows.size(), bf.get()) != bitRows.size()) { err = "write failed: " + opt.dumpBitsPath; return false; }
      }
      if (series) {
        if (!gpu_reduce(*g, nc, Wg.data(), true, tmp.data(), err)) return false;
        G.resize(nc);
        for (size_t i = 0; i < nc; ++i) G[i] = static_cast<float>(tmp[i]);
        SeriesIn in;
        in.nSky = static_cast<unsigned>(nSky); in.row = rowOf.data(); in.P = P; in.T = Tup;
        std::vector<float> wsky(nSky);
        for (size_t d = 0; d < nSky; ++d) wsky[d] = ladybug ? pa.omega[static_cast<size_t>(rowOf[d])] : static_cast<float>(pa.omega[static_cast<size_t>(rowOf[d])] / K);
        in.wsky = wsky.data(); in.S = S.data(); in.g = gSeries.data(); in.G = G.data();
        in.sun0 = static_cast<unsigned>(sun0); in.nSun = static_cast<unsigned>(nSun);
        in.sunStep = sunCol.data(); in.sunVec = sunVec.data(); in.sunW = sunW.data();
        E.assign(nc * static_cast<size_t>(Tup), 0.f);
        if (Tup > 0 && !gpu_series(*g, nc, in, E.data(), err)) return false;
        if (std::fwrite(E.data(), sizeof(float), E.size(), sf.get()) != E.size()) { err = "write failed: " + opt.seriesPath; return false; }
      }
    }
    if (opt.verbose) {
      GpuStats st = gpu_stats(*g);
      std::fprintf(stderr, "info: GAS %.1f MB build %.0f ms, %zu launches, %.3g rays in %.0f ms (%.2f GRays/s)\n",
                   st.gasBytes / 1048576.0, st.buildMs, st.launches, static_cast<double>(st.rays), st.traceMs,
                   st.traceMs > 0 ? static_cast<double>(st.rays) / st.traceMs / 1e6 : 0.0);
    }
  }
  if (sf) {
    File hf(std::fopen((opt.seriesPath + ".hoys.txt").c_str(), "w"), std::fclose);
    if (hf) for (double h : seriesHoys) std::fprintf(hf.get(), "%.4f\n", h);
  }

  // ---- values: N elements x (quantities x sides), quantity-major ----------------------------------------------------
  const size_t nElem = nSens / static_cast<size_t>(nS);
  const size_t C = nQ * static_cast<size_t>(nS);
  out.values.assign(nElem * C, 0.0);
  out.columns.clear();
  for (size_t qi = 0; qi < nQ; ++qi)
    for (int s = 0; s < nS; ++s)
      out.columns.push_back(std::string(kQuantityNames[static_cast<size_t>(opt.quantities[qi])]) + ((opt.sides == Sides::Back || (opt.sides == Sides::Both && s == 1)) ? "_back" : "_front"));
  for (size_t e = 0; e < nElem; ++e)
    for (size_t qi = 0; qi < nQ; ++qi)
      for (int s = 0; s < nS; ++s) {
        double v = acc[(e * static_cast<size_t>(nS) + static_cast<size_t>(s)) * nQ + qi];
        const Quantity q = opt.quantities[qi];
        if (q == Quantity::Irradiance) v = durationHours > 0 ? v * 1000.0 / durationHours : 0.0;
        else if (q == Quantity::Illuminance) v = durationHours > 0 ? v / durationHours : 0.0;
        out.values[e * C + qi * static_cast<size_t>(nS) + static_cast<size_t>(s)] = v;
      }

  // ---- dumps -----------------------------------------------------------------------------------------------------
  if (!opt.dumpSun.empty()) {   // columns 7-9: Radiance sun.c at the exact location, for the gendaymtx -D comparison
    File f(std::fopen(opt.dumpSun.c_str(), "w"), std::fclose);
    if (!f) { err = "cannot write " + opt.dumpSun; return false; }
    std::fprintf(f.get(), "# hoy month day hour alt_deg azi_deg up alt_radiance_deg azi_radiance_deg up_radiance dni dhi   (north %.3f)\n", opt.north);
    for (size_t t = 0; t < T; ++t) {
      double altR, aziR;
      sun_radiance(lat, -lon, -15.0 * tz, steps[t].jd, steps[t].hour, altR, aziR);
      std::fprintf(f.get(), "%.4f %d %d %.4f %.6f %.6f %d %.6f %.6f %d %.4f %.4f\n", steps[t].hoy, steps[t].month, steps[t].day, steps[t].hour,
                   suns[t].alt, suns[t].azi, suns[t].up ? 1 : 0, altR / kDeg, aziR / kDeg, altR > -0.2665 * kDeg ? 1 : 0, steps[t].dni, steps[t].dhi);
    }
  }
  if (!opt.dumpSky.empty()) {   // the two gendaymtx -A ASCII blocks (ground row first)
    File f(std::fopen(opt.dumpSky.c_str(), "w"), std::fclose);
    if (!f) { err = "cannot write " + opt.dumpSky; return false; }
    const char* head[2] = {"# gendaymtx -s -A ascii rows (ground row first), %.3g\n", "# gendaymtx -d -A ascii rows (ground row first), %.3g\n"};
    for (size_t b = 0; b < 2; ++b) {
      std::fputs(head[b], f.get());
      for (size_t r = 0; r <= Pz; ++r) {
        const float* v = &lbDif3g[b * 3 * (Pz + 1) + 3 * r];
        std::fprintf(f.get(), "%.3g %.3g %.3g\n", v[0], v[1], v[2]);
      }
    }
  }
  return true;
}

}  // namespace daylight
