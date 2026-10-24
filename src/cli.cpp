// SPDX-License-Identifier: Apache-2.0 AND LicenseRef-Commons-Clause-1.0
// CLI flag parsing, the post-parse weather / time rules and the output manifest, shared by main.cpp and the server.
#include "cli.h"
#include "internal.h"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>

namespace daylight::cli {

const char* const kUsage =
    "usage: daylight <mesh>... [flags]\n"
    "  meshes: .obj (one o/g group per mesh), .dmesh (raw arrays, layout in daylight.h) or .usd/.usda/.usdc/.usdz\n"
    "          (-DDAYLIGHT_USD=ON builds only); world metres, Z-up; positional = analysed (sensor) meshes\n"
    "  --context FILE           occluder-only mesh (repeatable)\n"
    "  --on faces|vertices      sensor placement (default faces)\n"
    "  --mode physical|ladybug  default physical\n"
    "  --quantity Q[,Q...]      radiation|direct|diffuse|irradiance|illuminance|sunhours (default radiation)\n"
    "  --sides front|back|both  receiving side(s) of each element (default front)\n"
    "  --cull-backfaces         occluders block only from their front side (default two-sided)\n"
    "  --epw FILE               weather + location; --lat --lon override its location (or replace it: sunhours)\n"
    "  --clear-sky              synthetic cloudless weather (Ineichen) at --lat --lon instead of --epw (upper bound)\n"
    "  --turbidity TL           Linke turbidity for --clear-sky (default 3)   --altitude M  site altitude (default 0)\n"
    "  --tz H                   clock of the timestamps, standard time (default: the EPW's zone, else round(lon / 15))\n"
    "  --time [YYYY-]MM-DDTHH:MM | --from TS --to TS [--step MIN]   default: annual over the EPW rows\n"
    "  --hours HH:MM-HH:MM      with --from/--to: only these hours of every day (analysis period)\n"
    "  --sky-density 1|2|4  --patch-samples K  --ground RHO  --north DEG  --offset M  --no-self-occlude\n"
    "  --optics SPEC[;SPEC...]  NAME[/SLOT]=RHO[,TAU[,glass|porous]] per mesh / face slot, * = every mesh (repeatable;\n"
    "                           default rho 0.2, tau 0)\n"
    "  --transmission off|on    light through glass / porous slots (default off: every surface opaque)\n"
    "  --fresnel on|off         glass transmittance by angle (default on; off: tau at every angle)\n"
    "  --bounces N              reflected light: up to N = 1..20 diffuse reflections by RHO (physical; default 0: none)\n"
    "  --bounce-samples M       paths per sensor for --bounces, 64..65536 (default 1024; noise falls as 1 / sqrt(M))\n"
    "  --out FILE[.npy]         values (text rows, or float64 [N x C] NumPy when .npy) + FILE.json manifest\n"
    "  --timeseries FILE.npy  --dump-bits FILE.npy  --dump-sky FILE  --dump-sun FILE\n"
    "  --bench [NSENS NTRIS]  -v          exit code: 1 usage, 2 runtime error\n";

namespace {

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

// SPEC[;SPEC...], SPEC = NAME[/SLOT]=RHO[,TAU[,glass|porous]]: split at the last '=', /SLOT only as a trailing '/' + digits
// (Facade/East is a name), kind porous when TAU > 0 unless given. Ranges are optics_ok's.
// ponytail: names containing ';' cannot be addressed (app mesh names are ids)
bool parse_optics(const std::string& s, std::vector<Optics>& out, std::string& err) {
  for (size_t b = 0, e; b <= s.size(); b = e + 1) {
    e = std::min(s.find(';', b), s.size());
    const std::string spec = s.substr(b, e - b);
    const auto bad = [&](const std::string& what) { err = "--optics " + spec + ": " + what; return false; };
    const size_t eq = spec.rfind('=');
    if (eq == std::string::npos || eq == 0) return bad("NAME[/SLOT]=RHO[,TAU[,glass|porous]]");
    Optics o;
    o.mesh = spec.substr(0, eq);
    const size_t sl = o.mesh.rfind('/');
    if (sl != std::string::npos && sl + 1 < o.mesh.size() && o.mesh.find_first_not_of("0123456789", sl + 1) == std::string::npos) {
      unsigned long v = 0;
      const auto r = std::from_chars(o.mesh.data() + sl + 1, o.mesh.data() + o.mesh.size(), v);
      if (r.ec != std::errc() || v > 65535) return bad("SLOT 0..65535");
      o.slot = int(v);
      o.mesh.resize(sl);
      if (o.mesh.empty() || o.mesh == "*") return bad("/SLOT needs a mesh NAME");
    }
    std::vector<std::string> v;   // RHO [TAU [kind]]
    for (size_t f = eq + 1, c; f <= spec.size(); f = c + 1) { c = std::min(spec.find(',', f), spec.size()); v.push_back(spec.substr(f, c - f)); }
    if (v.size() > 3) return bad("NAME[/SLOT]=RHO[,TAU[,glass|porous]]");
    double x[2] = {0, 0};
    for (size_t k = 0; k < v.size() && k < 2; ++k) {
      char* end = nullptr;
      x[k] = std::strtod(v[k].c_str(), &end);
      if (v[k].empty() || *end) return bad("bad number '" + v[k] + "'");
    }
    if (v.size() == 3 && v[2] != "glass" && v[2] != "porous") return bad("kind glass or porous");
    o.rho = x[0]; o.tau = x[1];
    o.glass = v.size() == 3 && v[2] == "glass";   // porous by default (opaque when TAU = 0)
    out.push_back(o);
  }
  return true;
}

}  // namespace

int parse(int argc, char** argv, Args& a, std::string& err) {
  auto need = [&](int& i, const char* flag) -> const char* {
    if (i + 1 >= argc) { err = std::string("missing value for ") + flag; return nullptr; }
    return argv[++i];
  };
  for (int i = 1; i < argc; ++i) {
    std::string f = argv[i];
    const char* v = nullptr;
    if (f == "-h" || f == "--help") { a.help = true; return 0; }
    else if (f == "-v") a.opt.verbose = true;
    else if (f == "--context") { if (!(v = need(i, "--context"))) return 1; a.context.push_back(v); }
    else if (f == "--on") { if (!(v = need(i, "--on"))) return 1; if (!std::strcmp(v, "faces")) a.opt.onVertices = false; else if (!std::strcmp(v, "vertices")) a.opt.onVertices = true; else { err = "--on faces|vertices"; return 1; } }
    else if (f == "--mode") { if (!(v = need(i, "--mode"))) return 1; if (!std::strcmp(v, "physical")) a.opt.mode = Mode::Physical; else if (!std::strcmp(v, "ladybug")) a.opt.mode = Mode::Ladybug; else { err = "--mode physical|ladybug"; return 1; } }
    else if (f == "--quantity") { if (!(v = need(i, "--quantity"))) return 1; if (!parse_quantities(v, a.opt.quantities)) { err = "unknown quantity"; return 1; } }
    else if (f == "--sides") { if (!(v = need(i, "--sides"))) return 1; if (!std::strcmp(v, "front")) a.opt.sides = Sides::Front; else if (!std::strcmp(v, "back")) a.opt.sides = Sides::Back; else if (!std::strcmp(v, "both")) a.opt.sides = Sides::Both; else { err = "--sides front|back|both"; return 1; } }
    else if (f == "--cull-backfaces") a.opt.cullBackfaces = true;
    else if (f == "--no-self-occlude") a.opt.selfOcclude = false;
    else if (f == "--epw") { if (!(v = need(i, "--epw"))) return 1; a.epw = v; }
    else if (f == "--clear-sky") a.clearSky = true;
    else if (f == "--turbidity") { if (!(v = need(i, "--turbidity"))) return 1; a.turbidity = std::atof(v); }
    else if (f == "--altitude") { if (!(v = need(i, "--altitude"))) return 1; a.altitude = std::atof(v); }
    else if (f == "--lat") { if (!(v = need(i, "--lat"))) return 1; a.opt.lat = std::atof(v); }
    else if (f == "--lon") { if (!(v = need(i, "--lon"))) return 1; a.opt.lon = std::atof(v); }
    else if (f == "--tz") { if (!(v = need(i, "--tz"))) return 1; a.opt.tz = std::atof(v); }
    else if (f == "--time") { if (!(v = need(i, "--time"))) return 1; a.timeStr = v; }
    else if (f == "--from") { if (!(v = need(i, "--from"))) return 1; a.fromStr = v; }
    else if (f == "--to") { if (!(v = need(i, "--to"))) return 1; a.toStr = v; }
    else if (f == "--step") { if (!(v = need(i, "--step"))) return 1; a.opt.stepMin = std::atoi(v); if (a.opt.stepMin <= 0) { err = "--step must be positive minutes"; return 1; } }
    else if (f == "--hours") {
      if (!(v = need(i, "--hours"))) return 1;
      int h0, m0, h1, m1;
      char end;
      if (std::sscanf(v, "%d:%d-%d:%d%c", &h0, &m0, &h1, &m1, &end) != 4 || h0 < 0 || m0 < 0 || m0 > 59 || m1 < 0 || m1 > 59 ||
          h0 * 60 + m0 >= h1 * 60 + m1 || h1 * 60 + m1 > 1440) { err = "--hours HH:MM-HH:MM with start < end <= 24:00"; return 1; }
      a.opt.dayFrom = h0 * 60 + m0;
      a.opt.dayTo = h1 * 60 + m1;
    }
    else if (f == "--sky-density") { if (!(v = need(i, "--sky-density"))) return 1; a.opt.mf = std::atoi(v); }
    else if (f == "--patch-samples") { if (!(v = need(i, "--patch-samples"))) return 1; a.opt.patchSamples = std::atoi(v); }
    else if (f == "--ground") { if (!(v = need(i, "--ground"))) return 1; a.opt.ground = std::atof(v); }
    else if (f == "--optics") { if (!(v = need(i, "--optics"))) return 1; if (!parse_optics(v, a.opt.optics, err)) return 1; }
    else if (f == "--transmission") { if (!(v = need(i, "--transmission"))) return 1; if (!std::strcmp(v, "on")) a.opt.transmission = true; else if (!std::strcmp(v, "off")) a.opt.transmission = false; else { err = "--transmission off|on"; return 1; } }
    else if (f == "--fresnel") { if (!(v = need(i, "--fresnel"))) return 1; if (!std::strcmp(v, "on")) a.opt.fresnel = true; else if (!std::strcmp(v, "off")) a.opt.fresnel = false; else { err = "--fresnel on|off"; return 1; } }
    else if (f == "--bounces" || f == "--bounce-samples") {   // ranges: optics_ok below
      if (!(v = need(i, f.c_str()))) return 1;
      int& n = f == "--bounces" ? a.opt.bounces : a.opt.bounceSamples;
      const char* e = v + std::strlen(v);
      if (const auto r = std::from_chars(v, e, n); r.ec != std::errc() || r.ptr != e) { err = f + " needs a whole number"; return 1; }
    }
    else if (f == "--north") { if (!(v = need(i, "--north"))) return 1; a.opt.north = std::atof(v); }
    else if (f == "--offset") { if (!(v = need(i, "--offset"))) return 1; a.opt.offset = std::atof(v); }
    else if (f == "--out") { if (!(v = need(i, "--out"))) return 1; a.out = v; }
    else if (f == "--timeseries") { if (!(v = need(i, "--timeseries"))) return 1; a.opt.seriesPath = v; }
    else if (f == "--dump-bits") { if (!(v = need(i, "--dump-bits"))) return 1; a.opt.dumpBitsPath = v; }
    else if (f == "--dump-sky") { if (!(v = need(i, "--dump-sky"))) return 1; a.opt.dumpSky = v; }
    else if (f == "--dump-sun") { if (!(v = need(i, "--dump-sun"))) return 1; a.opt.dumpSun = v; }
    else if (f == "--bench") {
      a.bench = true;
      if (i + 2 < argc && std::isdigit(static_cast<unsigned char>(argv[i + 1][0])) && std::isdigit(static_cast<unsigned char>(argv[i + 2][0]))) {
        a.benchSens = static_cast<size_t>(std::atoll(argv[++i]));
        a.benchTris = static_cast<size_t>(std::atoll(argv[++i]));
      }
    }
    else if (f.size() > 1 && f[0] == '-' && !std::isdigit(static_cast<unsigned char>(f[1]))) { err = "unknown flag " + f; return 1; }
    else a.sensors.push_back(f);
  }
  return optics_ok(a.opt, err) ? 0 : 1;
}

int prepare(Args& a, Weather& weather, bool& haveWeather, std::string& err) {
  if (a.clearSky && !a.epw.empty()) { err = "--clear-sky and --epw are exclusive"; return 1; }
  if (a.clearSky && (std::isnan(a.opt.lat) || std::isnan(a.opt.lon))) { err = "--clear-sky needs --lat and --lon"; return 1; }
  // Without --tz an EPW's own zone applies (run() reads it); only runs without an EPW guess it from the longitude.
  if (!std::isnan(a.opt.lat) && !std::isnan(a.opt.lon) && std::isnan(a.opt.tz) && a.epw.empty() && !haveWeather) {
    a.opt.tz = std::round(a.opt.lon / 15);
    if (a.opt.verbose) std::fprintf(stderr, "info: tz defaulted to %g\n", a.opt.tz);
  }
  if (!a.epw.empty()) {
    if (haveWeather) { err = "--epw cannot be combined with weather supplied by the caller"; return 1; }
    if (!read_epw(a.epw, weather, err)) return 2;
    haveWeather = true;
    if (a.opt.verbose && weather.missing) std::fprintf(stderr, "warn: %d EPW rows with missing radiation treated as 0\n", weather.missing);
  } else if (a.clearSky) {
    if (haveWeather) { err = "--clear-sky cannot be combined with weather supplied by the caller"; return 1; }
    if (!clear_sky_weather(a.opt.lat, a.opt.lon, a.opt.tz, a.turbidity, a.altitude, weather, err)) return 1;
    haveWeather = true;
  }
  const bool leap = haveWeather && weather.leap;
  if (!a.timeStr.empty()) {
    if (!a.fromStr.empty() || !a.toStr.empty()) { err = "--time cannot be combined with --from/--to"; return 1; }
    if (!parse_hoy(a.timeStr, leap, a.opt.t0, err)) return 1;
  } else if (!a.fromStr.empty()) {
    if (a.toStr.empty()) { err = "--from needs --to"; return 1; }
    if (!parse_hoy(a.fromStr, leap, a.opt.t0, err) || !parse_hoy(a.toStr, leap, a.opt.t1, err)) return 1;
  } else if (!a.toStr.empty()) { err = "--to needs --from"; return 1; }
  if ((a.opt.dayFrom > 0 || a.opt.dayTo < 1440) && a.fromStr.empty()) { err = "--hours needs --from/--to"; return 1; }
  return 0;
}

std::string manifest_json(const Result& r, const std::string& extraFields) {
  std::string m = "{\n  \"columns\": [";
  for (size_t c = 0; c < r.columns.size(); ++c) m += (c ? ", \"" : "\"") + r.columns[c] + "\"";
  char buf[160];
  std::snprintf(buf, sizeof buf, "],\n  \"duration_h\": %.6f,\n  \"dirs\": {\"sky\": %zu, \"ground\": %zu, \"sun\": %zu},\n  \"meshes\": [",
                r.durationHours, r.nSky, r.nSky, r.nSun);
  m += buf;
  size_t first = 0;
  for (size_t i = 0; i < r.ranges.size(); ++i) {
    std::snprintf(buf, sizeof buf, ", \"first\": %zu, \"count\": %zu}", first, r.ranges[i].second);
    m += std::string(i ? ",\n    {\"name\": \"" : "\n    {\"name\": \"") + r.ranges[i].first + "\"" + buf;
    first += r.ranges[i].second;
  }
  m += "\n  ]";
  if (!extraFields.empty()) m += ",\n  " + extraFields;
  m += "\n}\n";
  return m;
}

}  // namespace daylight::cli
