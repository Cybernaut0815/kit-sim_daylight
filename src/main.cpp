// SPDX-License-Identifier: Apache-2.0 AND LicenseRef-Commons-Clause-1.0
// daylight CLI: meshes + location + time spec in, one value list per element out. Flags and rules live in cli.cpp.
#include "cli.h"
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
#include <iterator>
#include <random>
#include <string>
#include <vector>

using namespace daylight;

namespace {

int usage_error(const char* msg) {
  std::fprintf(stderr, "error: %s\n%s", msg, cli::kUsage);
  return 1;
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
  cli::Args a;
  std::string err;
  if (cli::parse(argc, argv, a, err)) return usage_error(err.c_str());
  if (a.help) { std::fputs(cli::kUsage, stdout); return 0; }
  if (a.sensors.empty() && !a.bench) return usage_error("no mesh given");

  Weather weather;
  bool haveWeather = false;
  if (const int rc = cli::prepare(a, weather, haveWeather, err)) {
    if (rc == 1) return usage_error(err.c_str());
    std::fprintf(stderr, "error: %s\n", err.c_str());
    return rc;
  }

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
  const std::string manifest = cli::manifest_json(res);
  std::fwrite(manifest.data(), 1, manifest.size(), m);
  std::fclose(m);
  if (a.opt.verbose) std::fprintf(stderr, "info: output %.3f s\n", output_secs());
  return 0;
}
