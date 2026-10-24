// SPDX-License-Identifier: Apache-2.0 AND LicenseRef-Commons-Clause-1.0
// OBJ / .dmesh readers, triangulation (earcut for n-gons), sensor derivation (ladybug / physical rules), optics slots
// (row checks, resolution per global slot, thin solids).
#include "internal.h"
#include "glass.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <charconv>
#include <cstring>
#include <unordered_map>
#include <mapbox/earcut.hpp>

namespace daylight {
namespace {

struct V { double x, y, z; };
V operator+(V a, V b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
V operator-(V a, V b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
V operator*(V a, double s) { return {a.x * s, a.y * s, a.z * s}; }
V cross(V a, V b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
double dot(V a, V b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
double len(V a) { return std::sqrt(dot(a, a)); }
V unit(V a) { double l = len(a); return l > 0 ? a * (1 / l) : V{0, 0, 0}; }
V pt(const Mesh& m, int i) { return {m.points[3 * i], m.points[3 * i + 1], m.points[3 * i + 2]}; }
float3 f3(V a) { return float3{(float)a.x, (float)a.y, (float)a.z}; }

// Newell normal of polygon idx[0..n-1]: |N| = 2 * area, direction by winding.
V newell(const Mesh& m, const int* idx, int n) {
  V N{0, 0, 0};
  for (int i = 0; i < n; ++i) {
    V a = pt(m, idx[i]), b = pt(m, idx[(i + 1) % n]);
    N.x += (a.y - b.y) * (a.z + b.z); N.y += (a.z - b.z) * (a.x + b.x); N.z += (a.x - b.x) * (a.y + b.y);
  }
  return N;
}

// Unit normal, area and centroid of one face by the mode rules. Degenerate (area < 1e-12) -> n = 0.
void face_geom(const Mesh& m, const int* idx, int n, bool ladybug, V& nrm, double& area, V& cen) {
  nrm = {0, 0, 0}; area = 0; cen = {0, 0, 0};
  if (n < 3) return;
  V P0 = pt(m, idx[0]);
  if (ladybug) {
    for (int i = 0; i < n; ++i) cen = cen + pt(m, idx[i]);
    cen = cen * (1.0 / n);
    if (n == 3) { V c = cross(pt(m, idx[1]) - P0, pt(m, idx[2]) - P0); nrm = c; area = len(c) / 2; }
    else if (n == 4) {
      V P1 = pt(m, idx[1]), P2 = pt(m, idx[2]), P3 = pt(m, idx[3]);
      V n1 = cross(P1 - P0, P2 - P0), n2 = cross(P3 - P2, P1 - P2);
      nrm = n1 + n2; area = (len(n1) + len(n2)) / 2;
    }
  } else {
    V N = newell(m, idx, n), Nu = unit(N);
    nrm = N; area = len(N) / 2;
    double sum = 0;
    for (int i = 1; i + 1 < n; ++i) {              // signed-area-weighted fan centroid: exact for concave planar n-gons
      V a = pt(m, idx[i]), b = pt(m, idx[i + 1]);
      double w = dot(cross(a - P0, b - P0), Nu) / 2;
      cen = cen + (P0 + a + b) * (w / 3); sum += w;
    }
    if (sum > 1e-12) cen = cen * (1 / sum);
    else { for (int i = 0; i < n; ++i) cen = cen + pt(m, idx[i]); cen = cen * (1.0 / n); }
  }
  nrm = area < 1e-12 ? V{0, 0, 0} : unit(nrm);
}

std::string stem(const std::string& path) {
  size_t s = path.find_last_of("/\\"), b = s == std::string::npos ? 0 : s + 1, d = path.rfind('.');
  return path.substr(b, d == std::string::npos || d < b ? std::string::npos : d - b);
}

}  // namespace

void bbox_centre(const std::vector<Mesh>& meshes, double c[3]) {
  double lo[3] = {INFINITY, INFINITY, INFINITY}, hi[3] = {-INFINITY, -INFINITY, -INFINITY};
  for (const Mesh& m : meshes)
    for (size_t i = 0; i < m.points.size(); ++i) {
      lo[i % 3] = std::fmin(lo[i % 3], m.points[i]); hi[i % 3] = std::fmax(hi[i % 3], m.points[i]);
    }
  for (int k = 0; k < 3; ++k) c[k] = lo[k] <= hi[k] ? (lo[k] + hi[k]) / 2 : 0;
}

void triangulate(const Mesh& m, const double c[3], Tris& out, int& skipped, uint32_t slotBase) {
  unsigned base = (unsigned)out.v.size();
  V C{c[0], c[1], c[2]};
  for (size_t i = 0; i < m.numPoints(); ++i) out.v.push_back(f3(pt(m, (int)i) - C));
  uint32_t slot = slotBase;
  auto emit = [&](int a, int b, int d) { out.i.push_back(uint3{base + (unsigned)a, base + (unsigned)b, base + (unsigned)d}); out.slot.push_back(slot); };
  size_t off = 0;
  for (size_t f = 0; f < m.counts.size(); ++f) {
    const int n = m.counts[f], *idx = m.indices.data() + off;
    off += n;
    slot = slotBase + (m.slot.empty() ? 0u : m.slot[f]);
    if (n == 3) emit(idx[0], idx[1], idx[2]);
    else if (n == 4) { emit(idx[0], idx[1], idx[2]); emit(idx[0], idx[2], idx[3]); }  // ponytail: no convexity check (matches Tydra)
    else if (n >= 5) {
      V N = newell(m, idx, n);
      if (len(N) == 0) { ++skipped; continue; }
      N = unit(N);
      V ax = std::fabs(N.x) <= std::fabs(N.y) && std::fabs(N.x) <= std::fabs(N.z) ? V{1, 0, 0} : std::fabs(N.y) <= std::fabs(N.z) ? V{0, 1, 0} : V{0, 0, 1};
      V u = unit(cross(N, ax)), v = cross(N, u);   // (u, v, N) right-handed: polygon winding is CCW in (u, v)
      std::vector<std::vector<std::array<double, 2>>> poly(1);
      for (int k = 0; k < n; ++k) { V p = pt(m, idx[k]); poly[0].push_back({dot(p, u), dot(p, v)}); }
      std::vector<unsigned> t = mapbox::earcut<unsigned>(poly);
      if (t.empty()) { ++skipped; continue; }
      for (size_t k = 0; k + 2 < t.size(); k += 3) emit(idx[t[k]], idx[t[k + 1]], idx[t[k + 2]]);
    } else ++skipped;
  }
}

size_t slot_count(const Mesh& m) { return m.slot.empty() ? 1 : size_t(*std::max_element(m.slot.begin(), m.slot.end())) + 1; }

bool optics_ok(const Options& opt, std::string& err) {
  for (const Optics& o : opt.optics) {
    const std::string row = "--optics " + o.mesh + (o.slot >= 0 ? "/" + std::to_string(o.slot) : "") + ": ";
    if (o.mesh.empty() || o.slot < -1 || o.slot > 65535 || (o.mesh == "*" && o.slot >= 0)) { err = row + "NAME (or *) with an optional /SLOT 0..65535"; return false; }
    if (!(o.rho >= 0 && o.rho <= 1 && o.tau >= 0 && o.tau <= 1)) { err = row + "RHO and TAU must be in 0..1"; return false; }
    if (o.rho + o.tau > 1 + 1e-9) { err = row + "RHO + TAU must not exceed 1"; return false; }
    if (o.glass && o.rho != 0) { err = row + "glass reflects by angle: RHO must be 0"; return false; }
    if (o.glass && !(o.tau > 0 && o.tau <= 0.918)) { err = row + "glass TAU must be in (0, 0.918] (n 1.52 passes at most 0.918)"; return false; }
  }
  if (opt.transmission && opt.mode == Mode::Ladybug) { err = "--transmission on needs --mode physical (ladybug: every surface opaque and black)"; return false; }
  return true;
}

bool resolve_optics(const std::vector<Mesh>& meshes, const std::vector<Optics>& rows, std::vector<Slot>& out, std::string& err) {
  std::vector<size_t> base(meshes.size() + 1, 0);
  std::unordered_map<std::string, size_t> byName;   // mesh index, SIZE_MAX when several meshes share the name
  for (size_t k = 0; k < meshes.size(); ++k) {
    const Mesh& m = meshes[k];
    if (!m.slot.empty() && m.slot.size() != m.numFaces()) { err = "mesh " + m.name + ": " + std::to_string(m.slot.size()) + " slots for " + std::to_string(m.numFaces()) + " faces"; return false; }
    base[k + 1] = base[k] + slot_count(m);
    if (!byName.emplace(m.name, k).second) byName[m.name] = SIZE_MAX;
  }
  std::vector<std::pair<std::string, int>> keys;
  for (const Optics& o : rows) keys.emplace_back(o.mesh, o.slot);
  std::sort(keys.begin(), keys.end());
  if (auto d = std::adjacent_find(keys.begin(), keys.end()); d != keys.end()) {
    err = "--optics " + d->first + (d->second >= 0 ? "/" + std::to_string(d->second) : "") + " given twice"; return false;
  }
  // ponytail: one Slot (16 B, host) per id up to each mesh's largest, so sparse high ids (65535) cost ~1 MB per mesh;
  // a sparse table if such meshes appear (only a transmitting run uploads a record per slot)
  out.assign(base.back(), Slot{});
  for (int rank = 0; rank < 3; ++rank)   // * < NAME < NAME/SLOT: later ranks overwrite
    for (const Optics& o : rows) {
      if ((o.mesh == "*" ? 0 : o.slot < 0 ? 1 : 2) != rank) continue;
      const std::string row = "--optics " + o.mesh + (o.slot >= 0 ? "/" + std::to_string(o.slot) : "") + ": ";
      size_t b = 0, e = out.size();
      if (rank) {
        const auto it = byName.find(o.mesh);
        if (it == byName.end()) { err = row + "no mesh of that name"; return false; }
        if (it->second == SIZE_MAX) { err = row + "several meshes have that name"; return false; }
        b = base[it->second]; e = base[it->second + 1];
        if (rank == 2 && size_t(o.slot) >= e - b) { err = row + "slot out of range (the mesh has slots 0.." + std::to_string(e - b - 1) + ")"; return false; }
        if (rank == 2) { b += size_t(o.slot); e = b + 1; }
      }
      const Slot s{float(o.rho), float(o.tau), o.glass ? float(tn_from_Tn(o.tau)) : 0.f, o.glass ? kSlotGlass : 0u};
      std::fill(out.begin() + std::ptrdiff_t(b), out.begin() + std::ptrdiff_t(e), s);
    }
  return true;
}

unsigned thin_solid(const Mesh& m, unsigned s, const double c[3]) {
  using Key = std::array<long long, 3>;
  const V C{c[0], c[1], c[2]};
  std::vector<Key> keys;   // welded position of every corner of the slot's faces, face after face
  std::vector<int> n;      // corners per face
  double vol = 0, area = 0;
  size_t off = 0;
  for (size_t f = 0; f < m.counts.size(); ++f) {
    const int k = m.counts[f], *idx = m.indices.data() + off;
    off += k;
    if ((m.slot.empty() ? 0u : m.slot[f]) != s) continue;
    const V N = newell(m, idx, k);   // |N| = 2 area: volume = sum N . p / 6 over closed planar faces
    area += len(N) / 2; vol += dot(N, pt(m, idx[0]) - C) / 6;
    for (int j = 0; j < k; ++j) { const V p = pt(m, idx[j]) - C; keys.push_back({std::llround(p.x * 1e5), std::llround(p.y * 1e5), std::llround(p.z * 1e5)}); }
    n.push_back(k);
  }
  if (!(area > 0) || 2 * std::fabs(vol) / area >= 0.1) return 0;
  std::vector<Key> u = keys;
  std::sort(u.begin(), u.end()); u.erase(std::unique(u.begin(), u.end()), u.end());
  const auto id = [&](size_t j) { return std::lower_bound(u.begin(), u.end(), keys[j]) - u.begin(); };
  std::vector<std::pair<std::ptrdiff_t, std::ptrdiff_t>> fwd, rev;   // directed edges between welded points, and reversed
  for (size_t f = 0, j0 = 0; f < n.size(); j0 += size_t(n[f++]))
    for (int j = 0; j < n[f]; ++j) {
      const std::ptrdiff_t a = id(j0 + size_t(j)), b = id(j0 + size_t((j + 1) % n[f]));
      if (a != b) { fwd.emplace_back(a, b); rev.emplace_back(b, a); }
    }
  std::sort(fwd.begin(), fwd.end()); std::sort(rev.begin(), rev.end());
  return fwd == rev ? kSlotThin | (vol < 0 ? kSlotInward : 0u) : 0u;
}

size_t make_sensors(const Mesh& m, const Options& opt, const double c[3], Sensors& out, std::string& warn) {
  const bool ladybug = opt.mode == Mode::Ladybug;
  const V C{c[0], c[1], c[2]};
  int degenerate = 0, zeroVn = 0;
  std::vector<V> vn;
  if (opt.onVertices) vn.assign(m.numPoints(), V{0, 0, 0});
  auto push = [&](V p, V n) {
    if (opt.sides != Sides::Back) { out.p.push_back(f3(p + n * opt.offset - C)); out.n.push_back(f3(n)); }
    if (opt.sides != Sides::Front) { out.p.push_back(f3(p - n * opt.offset - C)); out.n.push_back(f3(n * -1)); }
  };
  if (ladybug)
    for (size_t f = 0; f < m.counts.size(); ++f)
      if (m.counts[f] > 4) {
        warn = "error: ladybug mode supports tri/quad faces only (mesh " + m.name + ", face " + std::to_string(f) + ")";
        return 0;
      }
  size_t off = 0;
  for (size_t f = 0; f < m.counts.size(); ++f) {
    const int n = m.counts[f], *idx = m.indices.data() + off;
    off += n;
    V nrm, cen; double area;
    face_geom(m, idx, n, ladybug, nrm, area, cen);
    if (area < 1e-12) ++degenerate;
    if (opt.onVertices) { for (int k = 0; k < n; ++k) vn[idx[k]] = vn[idx[k]] + nrm * area; }
    else push(cen, nrm);
  }
  for (size_t v = 0; v < vn.size(); ++v) {
    V n = vn[v];
    if (len(n) == 0) { n = {0, 0, 1}; ++zeroVn; }
    push(pt(m, (int)v), unit(n));
  }
  if (degenerate) warn += "mesh " + m.name + ": " + std::to_string(degenerate) + " degenerate face(s)";
  if (zeroVn) warn += std::string(degenerate ? "; " : "mesh " + m.name + ": ") + std::to_string(zeroVn) + " zero-length vertex normal(s) set to (0,0,1)";
  return opt.onVertices ? m.numPoints() : m.numFaces();
}

bool read_obj(const std::string& path, std::vector<Mesh>& out, std::string& err) {
  // ponytail: whole file in RAM, memory-map if a > RAM OBJ ever matters.
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) { err = "cannot open " + path; return false; }
  std::fseek(f, 0, SEEK_END);
  std::string buf((size_t)_ftelli64(f), '\0');   // buf[size()] == '\0' terminates the last line
  std::rewind(f);
  const bool ok = std::fread(buf.data(), 1, buf.size(), f) == buf.size();
  std::fclose(f);
  if (!ok) { err = "cannot read " + path; return false; }
  std::vector<double> V3;              // global v
  std::vector<int> remap, touched;     // global -> local point index per current mesh
  Mesh cur; cur.name = stem(path);
  auto finish = [&]() {
    if (!cur.counts.empty()) {
      std::sort(touched.begin(), touched.end());   // original vertex order within the mesh (per-vertex outputs stay 1:1)
      cur.points.resize(3 * touched.size());
      for (size_t k = 0; k < touched.size(); ++k) {
        remap[touched[k]] = (int)k;
        const double* p = &V3[3 * (size_t)touched[k]];
        cur.points[3 * k] = p[0]; cur.points[3 * k + 1] = p[1]; cur.points[3 * k + 2] = p[2];
      }
      for (int& k : cur.indices) k = remap[k];
      out.push_back(std::move(cur));
    }
    for (int g : touched) remap[g] = -1;
    touched.clear();
    cur = Mesh();
  };
  auto ws = [](const char*& s) { while (*s == ' ' || *s == '\t' || *s == '\r') ++s; };
  auto tok = [](const char*& s) { while (*s && *s != ' ' && *s != '\t' && *s != '\r' && *s != '\n') ++s; };   // to the token end (skips /vt/vn)
  auto bad = [&](size_t ln, const char* what) { err = path + ":" + std::to_string(ln) + ": " + what; return false; };
  // Clinger fast path: <= 15 digits, no exponent -> m / 10^k is exact in both operands, so the division is correctly
  // rounded like from_chars; anything else goes to from_chars.
  auto num = [](const char*& s, const char* e, double& q) {
    static const double P[] = {1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15};
    const bool neg = *s == '-';
    const char* p = s + (neg || *s == '+'); uint64_t m = 0; int digits = 0, frac = 0; bool dot = false;
    for (;; ++p) {
      if (*p >= '0' && *p <= '9') { m = m * 10 + (uint64_t)(*p - '0'); ++digits; frac += dot; }
      else if (*p == '.' && !dot) dot = true;
      else break;
    }
    if (digits && digits <= 15 && *p != 'e' && *p != 'E') { q = neg ? -(double)m / P[frac] : (double)m / P[frac]; s = p; return true; }
    if (*s == '+') ++s;   // from_chars rejects a leading '+'
    const auto r = std::from_chars(s, e, q);
    s = r.ptr; return r.ec == std::errc();
  };
  const char* s = buf.c_str();
  const char* const end = s + buf.size();
  for (size_t lineNo = 1; s < end; ++lineNo) {
    const char* nl = (const char*)std::memchr(s, '\n', (size_t)(end - s));
    const char* const e = nl ? nl : end;   // line [s, e); *e is '\n' or the final '\0'
    ws(s);
    if (s[0] == 'v' && (s[1] == ' ' || s[1] == '\t')) {
      ++s; double p[3];
      for (double& q : p) { ws(s); if (!num(s, e, q)) return bad(lineNo, "bad vertex"); }
      V3.insert(V3.end(), p, p + 3); remap.push_back(-1);
    } else if (s[0] == 'f' && (s[1] == ' ' || s[1] == '\t')) {
      ++s; int n = 0;
      for (;;) {
        ws(s);
        if (s >= e || *s == '#') break;
        long a;
        if (*s == '+') ++s;
        const auto r = std::from_chars(s, e, a);
        if (r.ec != std::errc()) return bad(lineNo, "bad face token");
        s = r.ptr; tok(s);
        const long nv = (long)remap.size();
        a = a < 0 ? nv + a : a - 1;
        if (a < 0 || a >= nv) return bad(lineNo, "vertex index out of range");
        if (remap[a] < 0) { remap[a] = (int)touched.size(); touched.push_back((int)a); }   // mark seen; points assembled in finish()
        cur.indices.push_back((int)a); ++n;
      }
      if (n) cur.counts.push_back(n);
    } else if ((s[0] == 'o' || s[0] == 'g') && (s[1] == ' ' || s[1] == '\t')) {
      finish();
      s += 2; ws(s);
      const char* t = s; tok(t);
      cur.name = t > s ? std::string(s, t) : stem(path);
    }
    s = e + 1;
  }
  finish();
  return true;
}

namespace {

// One .dmesh image in memory -> Mesh named `name`; `label` names the source in error messages (the path for files).
bool parse_dmesh(const char* data, size_t size, const std::string& name, const std::string& label, std::vector<Mesh>& out, std::string& err) {
  struct Header { char magic[4]; uint32_t flags; uint64_t nPoints, nFaces, nIndices; };   // flags bit 0: slot block
  static_assert(sizeof(Header) == 32 && sizeof(int) == 4, "dmesh layout");
  Header h{};
  if (size < sizeof h) { err = label + ": truncated"; return false; }
  std::memcpy(&h, data, sizeof h);
  if (std::memcmp(h.magic, "DLM1", 4)) { err = label + ": not a .dmesh file (magic DLM1)"; return false; }
  if (h.flags & ~1u) { err = label + ": unsupported header flags " + std::to_string(h.flags) + " (only bit 0, the slot block, is defined)"; return false; }
  if (!(h.nPoints <= size / 24 && h.nFaces <= size / 4 && h.nIndices <= size / 4 &&   // no overflow in the sums below
        32 + 24 * h.nPoints + 4 * h.nFaces + 4 * h.nIndices <= size)) { err = label + ": truncated"; return false; }
  const uint64_t nSlots = (h.flags & 1) ? h.nFaces : 0;
  if (32 + 24 * h.nPoints + 4 * h.nFaces + 4 * h.nIndices + 2 * nSlots > size) { err = label + ": truncated slot block"; return false; }
  Mesh m; m.name = name;
  m.points.resize(3 * h.nPoints); m.counts.resize(h.nFaces); m.indices.resize(h.nIndices); m.slot.resize(nSlots);
  const char* p = data + sizeof h;
  std::memcpy(m.points.data(), p, 24 * h.nPoints); p += 24 * h.nPoints;
  std::memcpy(m.counts.data(), p, 4 * h.nFaces); p += 4 * h.nFaces;
  std::memcpy(m.indices.data(), p, 4 * h.nIndices); p += 4 * h.nIndices;
  std::memcpy(m.slot.data(), p, 2 * nSlots);
  uint64_t sum = 0;
  for (int c : m.counts) { if (c < 3) { err = label + ": face with fewer than 3 vertices"; return false; } sum += (uint64_t)c; }
  if (sum != h.nIndices) { err = label + ": sum(counts) != n_indices"; return false; }
  for (int i : m.indices) if (i < 0 || (uint64_t)i >= h.nPoints) { err = label + ": vertex index out of range"; return false; }
  if (h.nFaces) out.push_back(std::move(m));   // empty meshes dropped like empty OBJ groups
  return true;
}

}  // namespace

bool read_dmesh(const char* data, size_t size, const std::string& name, std::vector<Mesh>& out, std::string& err) {
  return parse_dmesh(data, size, name, name, out, err);
}

bool read_dmesh(const std::string& path, std::vector<Mesh>& out, std::string& err) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) { err = "cannot open " + path; return false; }
  std::fseek(f, 0, SEEK_END);
  std::vector<char> buf(static_cast<size_t>(_ftelli64(f)));
  std::rewind(f);
  const bool ok = std::fread(buf.data(), 1, buf.size(), f) == buf.size();
  std::fclose(f);
  if (!ok) { err = path + ": truncated"; return false; }
  return parse_dmesh(buf.data(), buf.size(), stem(path), path, out, err);
}

}  // namespace daylight
