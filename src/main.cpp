// SPDX-License-Identifier: Apache-2.0 AND LicenseRef-Commons-Clause-1.0
// daylight CLI: meshes + location + time spec in, one value list per element out.
#include "daylight.h"
#include "internal.h"
#ifdef DAYLIGHT_USD
#include "daylight_usd.h"
#endif

#include <algorithm>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <random>
#include <string>
#include <vector>

using namespace daylight;

namespace {

const char* kUsage =
    "usage: daylight <mesh>... [flags]\n"
    "  meshes: .obj (one o/g group per mesh), .dmesh (raw arrays, layout in daylight.h) or .usd/.usda/.usdc/.usdz\n"
    "          (-DDAYLIGHT_USD=ON builds only); world metres, Z-up; positional = analysed (sensor) meshes\n"
    "  --context FILE           occluder-only mesh (repeatable)\n"
    "  --on faces|vertices      sensor placement (default faces)\n"
    "  --mode physical|ladybug  default physical\n"
    "  --quantity Q[,Q...]      radiation|direct|diffuse|irradiance|illuminance|sunhours (default radiation)\n"
    "  --sides front|back|both  receiving side(s) of each element (default front)\n"
    "  --cull-backfaces         occluders block only from their front side (default two-sided)\n"
    "  --epw FILE               weather + location; --lat --lon --tz override or replace it (sunhours only)\n"
    "  --clear-sky              synthetic cloudless weather (Ineichen) at --lat --lon instead of --epw (upper bound)\n"
    "  --turbidity TL           Linke turbidity for --clear-sky (default 3)   --altitude M  site altitude (default 0)\n"
    "  --tz H                   defaults to round(lon / 15) when --lat --lon are given without --tz\n"
    "  --time [YYYY-]MM-DDTHH:MM | --from TS --to TS [--step MIN]   default: annual over the EPW rows\n"
    "  --sky-density 1|2|4  --patch-samples K  --ground RHO  --north DEG  --offset M  --no-self-occlude\n"
    "  --out FILE[.npy]         values (text rows, or float64 [N x C] NumPy when .npy) + FILE.json manifest\n"
    "  --timeseries FILE.npy  --dump-bits FILE.npy  --dump-sky FILE  --dump-sun FILE\n"
    "  --bench [NSENS NTRIS]  -v          exit code: 1 usage, 2 runtime error\n";

struct Args {
  std::vector<std::string> sensors, context;
  Options opt;
  std::string epw, out, timeStr, fromStr, toStr;
  bool bench = false, clearSky = false;
  double turbidity = 3, altitude = 0;
  size_t benchSens = 1000000, benchTris = 10000000;
};

bool parse_quantities(const std::string& s, std::vector<Quantity>& out) {
  out.clear();
  for (size_t b = 0, e; b <= s.size(); b = e + 1) {
    e = std::min(s.find(',', b), s.size());
    size_t q = 0;
    while (q < std::size(kQuantityNames) && s.compare(b, e - b, kQuantityNames[q])) ++q;
    if (q == std::size(kQuantityNames)) return false;
    out.push_back(static_cast<Quantity>(q));
  }
  return true;
}

int usage_error(const char* msg) {
  std::fprintf(stderr, "error: %s\n%s", msg, kUsage);
  return 1;
}

bool parse_args(int argc, char** argv, Args& a, int& rc) {
  rc = 0;
  auto need = [&](int& i, const char* flag) -> const char* {
    if (i + 1 >= argc) { rc = usage_error((std::string("missing value for ") + flag).c_str()); return nullptr; }
    return argv[++i];
  };
  for (int i = 1; i < argc; ++i) {
    std::string f = argv[i];
    const char* v = nullptr;
    if (f == "-h" || f == "--help") { std::fputs(kUsage, stdout); rc = 0; return false; }
    else if (f == "-v") a.opt.verbose = true;
    else if (f == "--context") { if (!(v = need(i, "--context"))) return false; a.context.push_back(v); }
    else if (f == "--on") { if (!(v = need(i, "--on"))) return false; if (!std::strcmp(v, "faces")) a.opt.onVertices = false; else if (!std::strcmp(v, "vertices")) a.opt.onVertices = true; else { rc = usage_error("--on faces|vertices"); return false; } }
    else if (f == "--mode") { if (!(v = need(i, "--mode"))) return false; if (!std::strcmp(v, "physical")) a.opt.mode = Mode::Physical; else if (!std::strcmp(v, "ladybug")) a.opt.mode = Mode::Ladybug; else { rc = usage_error("--mode physical|ladybug"); return false; } }
    else if (f == "--quantity") { if (!(v = need(i, "--quantity"))) return false; if (!parse_quantities(v, a.opt.quantities)) { rc = usage_error("unknown quantity"); return false; } }
    else if (f == "--sides") { if (!(v = need(i, "--sides"))) return false; if (!std::strcmp(v, "front")) a.opt.sides = Sides::Front; else if (!std::strcmp(v, "back")) a.opt.sides = Sides::Back; else if (!std::strcmp(v, "both")) a.opt.sides = Sides::Both; else { rc = usage_error("--sides front|back|both"); return false; } }
    else if (f == "--cull-backfaces") a.opt.cullBackfaces = true;
    else if (f == "--no-self-occlude") a.opt.selfOcclude = false;
    else if (f == "--epw") { if (!(v = need(i, "--epw"))) return false; a.epw = v; }
    else if (f == "--clear-sky") a.clearSky = true;
    else if (f == "--turbidity") { if (!(v = need(i, "--turbidity"))) return false; a.turbidity = std::atof(v); }
    else if (f == "--altitude") { if (!(v = need(i, "--altitude"))) return false; a.altitude = std::atof(v); }
    else if (f == "--lat") { if (!(v = need(i, "--lat"))) return false; a.opt.lat = std::atof(v); }
    else if (f == "--lon") { if (!(v = need(i, "--lon"))) return false; a.opt.lon = std::atof(v); }
    else if (f == "--tz") { if (!(v = need(i, "--tz"))) return false; a.opt.tz = std::atof(v); }
    else if (f == "--time") { if (!(v = need(i, "--time"))) return false; a.timeStr = v; }
    else if (f == "--from") { if (!(v = need(i, "--from"))) return false; a.fromStr = v; }
    else if (f == "--to") { if (!(v = need(i, "--to"))) return false; a.toStr = v; }
    else if (f == "--step") { if (!(v = need(i, "--step"))) return false; a.opt.stepMin = std::atoi(v); if (a.opt.stepMin <= 0) { rc = usage_error("--step must be positive minutes"); return false; } }
    else if (f == "--sky-density") { if (!(v = need(i, "--sky-density"))) return false; a.opt.mf = std::atoi(v); }
    else if (f == "--patch-samples") { if (!(v = need(i, "--patch-samples"))) return false; a.opt.patchSamples = std::atoi(v); }
    else if (f == "--ground") { if (!(v = need(i, "--ground"))) return false; a.opt.ground = std::atof(v); }
    else if (f == "--north") { if (!(v = need(i, "--north"))) return false; a.opt.north = std::atof(v); }
    else if (f == "--offset") { if (!(v = need(i, "--offset"))) return false; a.opt.offset = std::atof(v); }
    else if (f == "--out") { if (!(v = need(i, "--out"))) return false; a.out = v; }
    else if (f == "--timeseries") { if (!(v = need(i, "--timeseries"))) return false; a.opt.seriesPath = v; }
    else if (f == "--dump-bits") { if (!(v = need(i, "--dump-bits"))) return false; a.opt.dumpBitsPath = v; }
    else if (f == "--dump-sky") { if (!(v = need(i, "--dump-sky"))) return false; a.opt.dumpSky = v; }
    else if (f == "--dump-sun") { if (!(v = need(i, "--dump-sun"))) return false; a.opt.dumpSun = v; }
    else if (f == "--bench") {
      a.bench = true;
      if (i + 2 < argc && std::isdigit(static_cast<unsigned char>(argv[i + 1][0])) && std::isdigit(static_cast<unsigned char>(argv[i + 2][0]))) {
        a.benchSens = static_cast<size_t>(std::atoll(argv[++i]));
        a.benchTris = static_cast<size_t>(std::atoll(argv[++i]));
      }
    }
    else if (f.size() > 1 && f[0] == '-' && !std::isdigit(static_cast<unsigned char>(f[1]))) { rc = usage_error((std::string("unknown flag ") + f).c_str()); return false; }
    else a.sensors.push_back(f);
  }
  return true;
}

std::string lower_ext(const std::string& path) {
  std::string ext = path.substr(std::min(path.find_last_of('.'), path.size()));
  for (char& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return ext;
}

bool load(const std::string& path, bool sensor, bool verbose, std::vector<Mesh>& meshes, std::string& err) {
  const size_t n = meshes.size();
  const std::string ext = lower_ext(path);
  const auto t0 = std::chrono::steady_clock::now();
  bool ok = false;
  if (ext == ".obj") ok = read_obj(path, meshes, err);
  else if (ext == ".dmesh") ok = read_dmesh(path, meshes, err);
  else if (ext == ".usd" || ext == ".usda" || ext == ".usdc" || ext == ".usdz") {
#ifdef DAYLIGHT_USD
    ok = read_usd(path, 'A', -1, meshes, err);
#else
    err = path + ": built without USD: configure with -DDAYLIGHT_USD=ON (preset vs2022-usd) or use bindings/python";
#endif
  } else err = path + ": unknown extension; .obj, .dmesh or .usd/.usda/.usdc/.usdz";
  if (!ok) return false;
  size_t F = 0, P = 0;
  for (size_t i = n; i < meshes.size(); ++i) { meshes[i].sensor = sensor; F += meshes[i].numFaces(); P += meshes[i].numPoints(); }
  if (verbose)
    std::fprintf(stderr, "info: read %s: %zu meshes, %zu faces, %zu points, %.3f s\n", path.c_str(), meshes.size() - n, F, P,
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  return true;
}

// Synthetic city for --bench: NSENS quads on a flat grid plus random boxes (12 triangles each) as context.
void bench_scene(size_t nSens, size_t nTris, std::vector<Mesh>& meshes) {
  const size_t side = static_cast<size_t>(std::ceil(std::sqrt(static_cast<double>(nSens))));
  const double extent = 1000.0, cell = extent / static_cast<double>(side);
  Mesh grid;
  grid.name = "bench_grid";
  for (size_t j = 0; j <= side; ++j)
    for (size_t i = 0; i <= side; ++i) {
      grid.points.push_back(-extent / 2 + cell * static_cast<double>(i));
      grid.points.push_back(-extent / 2 + cell * static_cast<double>(j));
      grid.points.push_back(0.0);
    }
  for (size_t j = 0; j < side && grid.counts.size() < nSens; ++j)
    for (size_t i = 0; i < side && grid.counts.size() < nSens; ++i) {
      const int a = static_cast<int>(j * (side + 1) + i);
      const int w = static_cast<int>(side + 1);
      grid.counts.push_back(4);
      grid.indices.insert(grid.indices.end(), {a, a + 1, a + w + 1, a + w});
    }
  meshes.push_back(std::move(grid));
  Mesh boxes;
  boxes.name = "bench_boxes";
  boxes.sensor = false;
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> pos(-extent / 2, extent / 2), fp(5.0, 30.0), ht(5.0, 60.0);
  const size_t nBox = nTris / 12;
  for (size_t b = 0; b < nBox; ++b) {
    const double x0 = pos(rng), y0 = pos(rng), w = fp(rng), d = fp(rng), h = ht(rng);
    const int base = static_cast<int>(boxes.points.size() / 3);
    for (int k = 0; k < 8; ++k) {
      boxes.points.push_back(x0 + ((k & 1) ? w : 0.0));
      boxes.points.push_back(y0 + ((k & 2) ? d : 0.0));
      boxes.points.push_back(0.1 + ((k & 4) ? h : 0.0));
    }
    // corner k: bit0 -> +x, bit1 -> +y, bit2 -> +z; quads wound for outward normals
    static const int quads[6][4] = {{0, 2, 3, 1}, {4, 5, 7, 6}, {0, 1, 5, 4}, {1, 3, 7, 5}, {3, 2, 6, 7}, {2, 0, 4, 6}};
    for (const int* q : quads) {
      boxes.counts.push_back(3); boxes.indices.insert(boxes.indices.end(), {base + q[0], base + q[1], base + q[2]});
      boxes.counts.push_back(3); boxes.indices.insert(boxes.indices.end(), {base + q[0], base + q[2], base + q[3]});
    }
  }
  meshes.push_back(std::move(boxes));
}

}  // namespace

int main(int argc, char** argv) {
  Args a;
  int rc = 0;
  if (!parse_args(argc, argv, a, rc)) return rc;
  if (a.sensors.empty() && !a.bench) return usage_error("no mesh given");
  if (a.clearSky && !a.epw.empty()) return usage_error("--clear-sky and --epw are exclusive");
  if (a.clearSky && (std::isnan(a.opt.lat) || std::isnan(a.opt.lon))) return usage_error("--clear-sky needs --lat and --lon");
  if (!std::isnan(a.opt.lat) && !std::isnan(a.opt.lon) && std::isnan(a.opt.tz)) {
    a.opt.tz = std::round(a.opt.lon / 15);
    if (a.opt.verbose) std::fprintf(stderr, "info: tz defaulted to %g\n", a.opt.tz);
  }

  Weather weather;
  bool haveWeather = false;
  std::string err;
  if (!a.epw.empty()) {
    if (!read_epw(a.epw, weather, err)) { std::fprintf(stderr, "error: %s\n", err.c_str()); return 2; }
    haveWeather = true;
    if (a.opt.verbose && weather.missing) std::fprintf(stderr, "warn: %d EPW rows with missing radiation treated as 0\n", weather.missing);
  } else if (a.clearSky) {
    if (!clear_sky_weather(a.opt.lat, a.opt.lon, a.opt.tz, a.turbidity, a.altitude, weather, err)) return usage_error(err.c_str());
    haveWeather = true;
  }
  const bool leap = haveWeather && weather.leap;
  if (!a.timeStr.empty()) {
    if (!a.fromStr.empty() || !a.toStr.empty()) return usage_error("--time cannot be combined with --from/--to");
    if (!parse_hoy(a.timeStr, leap, a.opt.t0, err)) return usage_error(err.c_str());
  } else if (!a.fromStr.empty()) {
    if (a.toStr.empty()) return usage_error("--from needs --to");
    if (!parse_hoy(a.fromStr, leap, a.opt.t0, err) || !parse_hoy(a.toStr, leap, a.opt.t1, err)) return usage_error(err.c_str());
  } else if (!a.toStr.empty()) return usage_error("--to needs --from");

  std::vector<Mesh> meshes;
  if (a.bench) {
    bench_scene(a.benchSens, a.benchTris, meshes);
    a.opt.verbose = true;
    if (!haveWeather) { a.opt.quantities = {Quantity::SunHours}; if (std::isnan(a.opt.lat)) { a.opt.lat = 42.35; a.opt.lon = -71.07; a.opt.tz = -5; } }
    else a.opt.quantities = {Quantity::Radiation, Quantity::SunHours};
    std::fprintf(stderr, "bench: %zu sensors, %zu triangles\n", a.benchSens, a.benchTris);
  } else {
    for (const std::string& p : a.sensors)
      if (!load(p, true, a.opt.verbose, meshes, err)) { std::fprintf(stderr, "error: %s\n", err.c_str()); return 2; }
    for (const std::string& p : a.context)
      if (!load(p, false, a.opt.verbose, meshes, err)) { std::fprintf(stderr, "error: %s\n", err.c_str()); return 2; }
  }

  const auto t0 = std::chrono::steady_clock::now();
  Result res;
  if (!run(meshes, haveWeather ? &weather : nullptr, a.opt, res, err)) { std::fprintf(stderr, "error: %s\n", err.c_str()); return 2; }
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  if (a.opt.verbose || a.bench) std::fprintf(stderr, "info: run %.2f s total (incl. sky model, GPU, output)\n", secs);
  if (a.bench) return 0;

  // values: one line per element, one column per quantity x side (or a float64 [N x C] .npy); FILE.json manifest
  const auto t1 = std::chrono::steady_clock::now();
  const bool npy = lower_ext(a.out) == ".npy";
  std::FILE* f = a.out.empty() ? stdout : std::fopen(a.out.c_str(), npy ? "wb" : "w");
  if (!f) { std::fprintf(stderr, "error: cannot write %s\n", a.out.c_str()); return 2; }
  const size_t C = res.columns.size(), N = C ? res.values.size() / C : 0;
  if (npy) {
    if (!npy_write_header(f, N, C, "<f8") || std::fwrite(res.values.data(), 8, res.values.size(), f) != res.values.size()) {
      std::fprintf(stderr, "error: cannot write %s\n", a.out.c_str()); return 2;
    }
  } else {
    static char buf[1 << 16];
    char* p = buf;
    for (size_t e = 0; e < N; ++e) {
      for (size_t c = 0; c < C; ++c) {
        if (p + 32 > std::end(buf)) { std::fwrite(buf, 1, static_cast<size_t>(p - buf), f); p = buf; }
        if (c) *p++ = ' ';
        p = std::to_chars(p, std::end(buf), res.values[e * C + c], std::chars_format::general, 10).ptr;   // == "%.10g"
      }
      *p++ = '\n';
    }
    std::fwrite(buf, 1, static_cast<size_t>(p - buf), f);
  }
  const auto output_secs = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count(); };
  if (f == stdout) { std::fflush(f); if (a.opt.verbose) std::fprintf(stderr, "info: output %.3f s\n", output_secs()); return 0; }
  std::fclose(f);
  std::FILE* m = std::fopen((a.out + ".json").c_str(), "w");
  if (!m) { std::fprintf(stderr, "error: cannot write %s.json\n", a.out.c_str()); return 2; }
  std::fprintf(m, "{\n  \"columns\": [");
  for (size_t c = 0; c < C; ++c) std::fprintf(m, "%s\"%s\"", c ? ", " : "", res.columns[c].c_str());
  std::fprintf(m, "],\n  \"duration_h\": %.6f,\n  \"dirs\": {\"sky\": %zu, \"ground\": %zu, \"sun\": %zu},\n  \"meshes\": [",
               res.durationHours, res.nSky, res.nSky, res.nSun);
  size_t first = 0;
  for (size_t i = 0; i < res.ranges.size(); ++i) {
    std::fprintf(m, "%s\n    {\"name\": \"%s\", \"first\": %zu, \"count\": %zu}", i ? "," : "", res.ranges[i].first.c_str(), first, res.ranges[i].second);
    first += res.ranges[i].second;
  }
  std::fprintf(m, "\n  ]\n}\n");
  std::fclose(m);
  if (a.opt.verbose) std::fprintf(stderr, "info: output %.3f s\n", output_secs());
  return 0;
}
