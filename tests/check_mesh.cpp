// SPDX-License-Identifier: Apache-2.0 AND LicenseRef-Commons-Clause-1.0
// Checks for mesh.cpp: OBJ / .dmesh readers, triangulation, sensor rules (ladybug / physical), sides, offset, optics
// slots (thin solids, --optics / --bounces grammar via cli::parse, resolution and its errors).
#include "cli.h"
#include "glass.h"
#include "internal.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

using namespace daylight;

// Axis box [x0,x1] x [y0,y1] x [z0,z1], 6 outward quads; indexed (8 points) or a soup (4 points per face, as app
// meshes arrive). Also used by check_gpu.cpp.
Mesh box_mesh(const char* name, double x0, double y0, double z0, double x1, double y1, double z1, bool soup) {
  Mesh m; m.name = name;
  const double X[2] = {x0, x1}, Y[2] = {y0, y1}, Z[2] = {z0, z1};
  static const int q[6][4] = {{0, 2, 3, 1}, {4, 5, 7, 6}, {0, 1, 5, 4}, {1, 3, 7, 5}, {3, 2, 6, 7}, {2, 0, 4, 6}};   // corner k: bit0 x, bit1 y, bit2 z
  const auto corner = [&](int k) { m.points.insert(m.points.end(), {X[k & 1], Y[(k >> 1) & 1], Z[k >> 2]}); };
  for (int k = 0; !soup && k < 8; ++k) corner(k);
  for (const int* f : q) {
    m.counts.push_back(4);
    for (int j = 0; j < 4; ++j) { if (soup) corner(f[j]); m.indices.push_back(soup ? int(m.points.size() / 3) - 1 : f[j]); }
  }
  return m;
}
static Mesh merged(const Mesh& a, const Mesh& b) {   // b's faces after a's, one mesh
  Mesh m = a;
  const int base = int(a.numPoints());
  m.points.insert(m.points.end(), b.points.begin(), b.points.end());
  m.counts.insert(m.counts.end(), b.counts.begin(), b.counts.end());
  for (int i : b.indices) m.indices.push_back(base + i);
  return m;
}
static int parse(std::initializer_list<const char*> args, cli::Args& a, std::string& err) {   // daylight <args>
  std::vector<char*> argv{const_cast<char*>("daylight")};
  for (const char* s : args) argv.push_back(const_cast<char*>(s));
  a = cli::Args{}; err.clear();
  return cli::parse(int(argv.size()), argv.data(), a, err);
}

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { std::printf("%s:%d: CHECK(%s)\n", __FILE__, __LINE__, #cond); ++failures; } } while (0)

static bool near3(float3 a, double x, double y, double z, double tol = 1e-6) {
  return std::fabs(a.x - x) < tol && std::fabs(a.y - y) < tol && std::fabs(a.z - z) < tol;
}
static Mesh polygon(std::initializer_list<double> xyz) {   // one face over all given points
  Mesh m; m.name = "poly"; m.points = xyz; m.counts = {(int)(xyz.size() / 3)};
  for (int i = 0; i < m.counts[0]; ++i) m.indices.push_back(i);
  return m;
}
static double tri_area(const Tris& t, size_t k) {
  float3 a = t.v[t.i[k].x], b = t.v[t.i[k].y], c = t.v[t.i[k].z];
  double ux = b.x - a.x, uy = b.y - a.y, uz = b.z - a.z, vx = c.x - a.x, vy = c.y - a.y, vz = c.z - a.z;
  double cx = uy * vz - uz * vy, cy = uz * vx - ux * vz, cz = ux * vy - uy * vx;
  return std::sqrt(cx * cx + cy * cy + cz * cz) / 2;
}

int check_mesh(const char* testsDir) {
  const double zero[3] = {0, 0, 0};
  std::string err, warn;
  Options phys, lb;
  lb.mode = Mode::Ladybug;

  // --- box.obj: two meshes, counts, bbox centre, ground face 0 sensor in both modes
  std::vector<Mesh> meshes;
  CHECK(read_obj(std::string(testsDir) + "/box.obj", meshes, err));
  CHECK(meshes.size() == 2);
  if (meshes.size() == 2) {
    CHECK(meshes[0].name == "ground" && meshes[0].numPoints() == 9 && meshes[0].numFaces() == 4);
    CHECK(meshes[1].name == "box" && meshes[1].numPoints() == 8 && meshes[1].numFaces() == 12);
    CHECK(meshes[1].indices[0] == 0 && meshes[1].indices[1] == 3 && meshes[1].points[3] == 1.5 && meshes[1].points[4] == 0.5);   // remapped per mesh, ORIGINAL v order kept (f 10 13 12 -> 0 3 2)
    double c[3]; bbox_centre(meshes, c);
    CHECK(std::fabs(c[0]) < 1e-12 && std::fabs(c[1]) < 1e-12 && std::fabs(c[2] - 0.75) < 1e-12);
    for (const Options* o : {&phys, &lb}) {
      Sensors s; warn.clear();
      CHECK(make_sensors(meshes[0], *o, zero, s, warn) == 4 && s.p.size() == 4 && warn.empty());
      CHECK(near3(s.p[0], -1, -1, 0) && near3(s.n[0], 0, 0, 1));
    }
    Sensors s; warn.clear();                                   // recentring: origin = centroid - c
    CHECK(make_sensors(meshes[0], phys, c, s, warn) == 4 && near3(s.p[0], -1, -1, -0.75));
    Tris t; int skipped = 0;
    triangulate(meshes[1], c, t, skipped);
    CHECK(t.i.size() == 12 && skipped == 0 && near3(t.v[0], 0.5, 0.5, -0.25));
  }
  CHECK(!read_obj(std::string(testsDir) + "/does_not_exist.obj", meshes, err) && !err.empty());

  // --- non-planar quad: ladybug n = normalize(n1 + n2) by hand, differs from Newell (physical)
  {
    Mesh q = polygon({0, 0, 0, 1, 0, 0, 1, 1, 0.2, 0, 1, 0});
    // n1 = (P1-P0)x(P2-P0) = (1,0,0)x(1,1,0.2) = (0,-0.2,1); n2 = (P3-P2)x(P1-P2) = (-1,0,-0.2)x(0,-1,-0.2) = (-0.2,-0.2,1)
    double sx = -0.2, sy = -0.4, sz = 2, l = std::sqrt(sx * sx + sy * sy + sz * sz);
    Sensors a, b; warn.clear();
    CHECK(make_sensors(q, lb, zero, a, warn) == 1 && near3(a.n[0], sx / l, sy / l, sz / l));
    CHECK(near3(a.p[0], 0.5, 0.5, 0.05));                                   // plain vertex mean
    CHECK(make_sensors(q, phys, zero, b, warn) == 1);
    double nx = -0.2, ny = -0.2, nz = 2, ln = std::sqrt(nx * nx + ny * ny + nz * nz);   // Newell by hand
    CHECK(near3(b.n[0], nx / ln, ny / ln, nz / ln));
    CHECK(std::fabs(a.n[0].y - b.n[0].y) > 1e-6);
  }

  // --- concave L polygon: earcut -> 4 triangles, area 3
  {
    Mesh L = polygon({0, 0, 0, 2, 0, 0, 2, 1, 0, 1, 1, 0, 1, 2, 0, 0, 2, 0});
    L.counts = {3, 6}; L.indices = {0, 1, 2, 0, 1, 2, 3, 4, 5};   // a leading triangle, then the L
    Tris t; int skipped = 0;
    triangulate(L, zero, t, skipped);
    CHECK(t.i.size() == 5 && skipped == 0);
    double area = 0;
    for (size_t k = 1; k < t.i.size(); ++k) area += tri_area(t, k);
    CHECK(std::fabs(area - 3) < 1e-12);
    for (size_t k = 1; k < t.i.size(); ++k) {   // winding preserved: every earcut triangle faces +Z like the input
      float3 a = t.v[t.i[k].x], b = t.v[t.i[k].y], c = t.v[t.i[k].z];
      CHECK((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x) > 0);
    }
    Sensors s; warn.clear();                    // physical: Newell area 3, area-weighted centroid of the fan (2/3, 2/3)... by hand:
    CHECK(make_sensors(L, phys, zero, s, warn) == 2 && near3(s.n[1], 0, 0, 1));
    // fan (0,i,i+1): tris (0,1,2) a=1 c=(4/3,1/3); (0,2,3) a=0.5 c=(1,2/3); (0,3,4) a=0.5 c=(2/3,1); (0,4,5) a=1 c=(1/3,4/3)
    CHECK(near3(s.p[1], (4.0 / 3 + 0.5 + 1.0 / 3 + 1.0 / 3) / 3, (1.0 / 3 + 1.0 / 3 + 0.5 + 4.0 / 3) / 3, 0));
    // concave chevron: fan triangle (0,2,3) lies outside -> signed weights; exact centroid by shoelace = (1, 0.7)
    Mesh ch = polygon({0, 0, 0, 2, 0, 0, 2, 2, 0, 1, 0.5, 0, 0, 2, 0});
    Sensors sc; warn.clear();
    CHECK(make_sensors(ch, phys, zero, sc, warn) == 1 && near3(sc.p[0], 1, 0.7, 0) && near3(sc.n[0], 0, 0, 1));
    Sensors e; warn.clear();                    // ladybug: n-gon is an error, nothing emitted
    CHECK(make_sensors(L, lb, zero, e, warn) == 0 && e.p.empty() && warn.rfind("error:", 0) == 0);
    int sk = 0; Tris d; Mesh deg = polygon({0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4});   // collinear 5-gon
    triangulate(deg, zero, d, sk);
    CHECK(sk == 1 && d.i.empty());
  }

  // --- unit cube of 6 quads: corner (1,1,1) vertex normal == normalize(1,1,1) (area-weighted) in physical mode
  {
    Mesh cube; cube.name = "cube";
    cube.points = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1};
    cube.counts = {4, 4, 4, 4, 4, 4};
    cube.indices = {0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7};   // outward
    Options ov = phys; ov.onVertices = true;
    Sensors s; warn.clear();
    CHECK(make_sensors(cube, ov, zero, s, warn) == 8 && s.p.size() == 8 && warn.empty());
    double k = 1 / std::sqrt(3.0);
    CHECK(near3(s.n[6], k, k, k) && near3(s.n[0], -k, -k, -k) && near3(s.p[6], 1, 1, 1));
    Options lv = lb; lv.onVertices = true; Sensors sl; warn.clear();   // same cube under ladybug quad rules
    CHECK(make_sensors(cube, lv, zero, sl, warn) == 8 && near3(sl.n[6], k, k, k));
    // unequal areas prove the weighting: tri +Z area 0.5, tri +Y area 2 -> vn0 = normalize(0, 2, 0.5) in both modes
    Mesh two; two.points = {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 2, 2, 0, 0}; two.counts = {3, 3}; two.indices = {0, 1, 2, 0, 3, 4};
    double w = std::sqrt(4.25);
    for (const Options* o : {&ov, &lv}) { Sensors sw; warn.clear(); CHECK(make_sensors(two, *o, zero, sw, warn) == 5 && near3(sw.n[0], 0, 2 / w, 0.5 / w)); }
  }

  // --- sides and offset
  {
    Mesh tri = polygon({0, 0, 0, 2, 0, 0, 0, 2, 0});
    Options both = phys; both.sides = Sides::Both; both.offset = 0.1;
    Sensors s; warn.clear();
    CHECK(make_sensors(tri, both, zero, s, warn) == 1 && s.p.size() == 2 && s.n.size() == 2);
    CHECK(near3(s.n[0], 0, 0, 1) && near3(s.n[1], 0, 0, -1));
    CHECK(near3(s.p[0], 2.0 / 3, 2.0 / 3, 0.1) && near3(s.p[1], 2.0 / 3, 2.0 / 3, -0.1));
    Options back = phys; back.sides = Sides::Back; Sensors b; warn.clear();
    CHECK(make_sensors(tri, back, zero, b, warn) == 1 && b.p.size() == 1 && near3(b.n[0], 0, 0, -1) && near3(b.p[0], 2.0 / 3, 2.0 / 3, 0));
    Mesh deg = polygon({0, 0, 0, 1, 1, 1, 2, 2, 2});   // degenerate: normal 0, warned
    Sensors d; warn.clear();
    CHECK(make_sensors(deg, phys, zero, d, warn) == 1 && near3(d.n[0], 0, 0, 0) && warn.find("1 degenerate") != std::string::npos);
    Options dv = phys; dv.onVertices = true; Sensors dd; warn.clear();
    CHECK(make_sensors(deg, dv, zero, dd, warn) == 3 && near3(dd.n[0], 0, 0, 1) && warn.find("3 zero-length") != std::string::npos);
  }

  // --- OBJ: negative indices, a/b/c and a//c tokens (vt/vn skipped), g groups, default name, exponents / '+',
  //     CRLF, comments, blank lines, no final newline; bad token -> error with line number
  {
    std::string tmp = std::string(testsDir) + "/_check_mesh_tmp.obj";
    FILE* f = std::fopen(tmp.c_str(), "wb");
    CHECK(f != nullptr);
    if (f) {
      std::fputs("# comment\n\nv 0 0 0\nv 1 0 0\nv 1 1 0\nvt 0 0\nvn 0 0 1\nvn 0 0 1\nvn 0 0 1\n"
                 "f -3/1/-3 -2/1/-2 -1/1/-1\n"              // relative v indices
                 "g second\r\nv +5e0 5.0E0 5\r\nvn 0 0 1\nf 2//2 3//3 4//4\r\n"
                 "g third\nf 1/1/3 2/1/2 3/1/1\n\n"
                 "g fourth\nf 1 2 3 # tail\nf 1 2\n"        // a 2-vertex face is kept and skipped later
                 "g fifth\nv 1e-3 -2.5E-1 +0\nv 0.1234567890123456 0 0\nf -2 +1 2\nf -1 1 2", f);   // 16 digits: from_chars path; no final newline
      std::fclose(f);
      std::vector<Mesh> ms;
      CHECK(read_obj(tmp, ms, err));
      CHECK(ms.size() == 5);
      if (ms.size() == 5) {
        CHECK(ms[0].name == "_check_mesh_tmp" && ms[0].numPoints() == 3 && ms[0].indices == std::vector<int>({0, 1, 2}));
        CHECK(ms[1].name == "second" && ms[1].numPoints() == 3 && ms[1].points[6] == 5 && ms[1].points[7] == 5 && ms[1].points[8] == 5);
        CHECK(ms[1].indices == std::vector<int>({0, 1, 2}) && ms[1].points[0] == 1 && ms[1].points[1] == 0);
        CHECK(ms[2].name == "third" && ms[2].numPoints() == 3 && ms[2].indices == std::vector<int>({0, 1, 2}));
        CHECK(ms[3].name == "fourth" && ms[3].counts == std::vector<int>({3, 2}));
        CHECK(ms[4].name == "fifth" && ms[4].counts == std::vector<int>({3, 3}) && ms[4].indices == std::vector<int>({2, 0, 1, 3, 0, 1}));
        CHECK(ms[4].numPoints() == 4 && ms[4].points[6] == 1e-3 && ms[4].points[7] == -0.25 && ms[4].points[8] == 0 && ms[4].points[9] == 0.1234567890123456);
        Tris t; int skipped = 0;
        triangulate(ms[3], zero, t, skipped);
        CHECK(t.i.size() == 1 && skipped == 1);
      }
      f = std::fopen(tmp.c_str(), "wb");
      if (f) { std::fputs("v 0 0 0\nf 1 x 1\n", f); std::fclose(f); }
      err.clear();
      CHECK(!read_obj(tmp, ms, err) && err.find(":2: bad face token") != std::string::npos);
      std::remove(tmp.c_str());
    }
  }

  // --- .dmesh: quad + triangle round trip; bad magic, truncated (header only), index out of range -> false + err;
  //     slot block (flags bit 0), truncated block, unknown flag bits
  {
    std::string tmp = std::string(testsDir) + "/_check_mesh_tmp.dmesh";
    const double P[15] = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 2, 0, 0};
    const int C[2] = {4, 3};
    int I[7] = {0, 1, 2, 3, 1, 4, 2};
    auto write = [&](const char* magic, int lastIndex, bool arrays) {
      FILE* f = std::fopen(tmp.c_str(), "wb");
      CHECK(f != nullptr);
      if (!f) return;
      const uint32_t zero = 0; const uint64_t n[3] = {5, 2, 7};
      std::fwrite(magic, 1, 4, f); std::fwrite(&zero, 4, 1, f); std::fwrite(n, 8, 3, f);
      I[6] = lastIndex;
      if (arrays) { std::fwrite(P, 8, 15, f); std::fwrite(C, 4, 2, f); std::fwrite(I, 4, 7, f); }
      std::fclose(f);
    };
    std::vector<Mesh> ms;
    write("DLM1", 2, true);
    CHECK(read_dmesh(tmp, ms, err) && ms.size() == 1);
    if (ms.size() == 1) {
      CHECK(ms[0].name == "_check_mesh_tmp" && ms[0].points == std::vector<double>(P, P + 15));
      CHECK(ms[0].counts == std::vector<int>({4, 3}) && ms[0].indices == std::vector<int>({0, 1, 2, 3, 1, 4, 2}));
    }
    write("DLM9", 2, true); err.clear(); CHECK(!read_dmesh(tmp, ms, err) && !err.empty());
    write("DLM1", 2, false); err.clear(); CHECK(!read_dmesh(tmp, ms, err) && !err.empty());
    write("DLM1", 5, true); err.clear(); CHECK(!read_dmesh(tmp, ms, err) && !err.empty());
    std::remove(tmp.c_str());
    // memory overload: the same image from a buffer, name given by the caller, label in errors
    std::string buf;
    {
      const uint32_t flags = 0; const uint64_t counts[3] = {5, 2, 7};
      I[6] = 2;
      buf.append("DLM1", 4); buf.append(reinterpret_cast<const char*>(&flags), 4); buf.append(reinterpret_cast<const char*>(counts), 24);
      buf.append(reinterpret_cast<const char*>(P), 120); buf.append(reinterpret_cast<const char*>(C), 8); buf.append(reinterpret_cast<const char*>(I), 28);
    }
    ms.clear();
    CHECK(read_dmesh(buf.data(), buf.size(), "upload", ms, err) && ms.size() == 1 && ms[0].name == "upload" && ms[0].indices.size() == 7 && ms[0].points == std::vector<double>(P, P + 15));
    CHECK(ms.size() == 1 && ms[0].slot.empty());
    err.clear(); CHECK(!read_dmesh(buf.data(), 32, "upload", ms, err) && err.find("upload") != std::string::npos);
    // slot block: flags bit 0 -> uint16 slot[n_faces] after the indices; truncated block, other bits -> false + err
    std::string sb = buf;
    const uint16_t S[2] = {7, 65535};
    sb[4] = 1; sb.append(reinterpret_cast<const char*>(S), 4);
    ms.clear(); err.clear();
    CHECK(read_dmesh(sb.data(), sb.size(), "upload", ms, err) && ms.size() == 1 && ms[0].slot == std::vector<uint16_t>({7, 65535}));
    CHECK(ms.size() == 1 && ms[0].counts == std::vector<int>({4, 3}) && ms[0].indices == std::vector<int>({0, 1, 2, 3, 1, 4, 2}));
    for (size_t n : {sb.size() - 1, buf.size()}) {   // half a block, no block
      ms.clear(); err.clear();
      CHECK(!read_dmesh(sb.data(), n, "upload", ms, err) && ms.empty() && err.find("truncated slot block") != std::string::npos);
    }
    for (uint32_t bad : {3u, 2u, 0x80000000u}) {
      std::memcpy(&sb[4], &bad, 4); ms.clear(); err.clear();
      CHECK(!read_dmesh(sb.data(), sb.size(), "upload", ms, err) && ms.empty() && err.find("flags") != std::string::npos);
    }
    std::memset(&sb[4], 0, 4); ms.clear();   // flags 0: trailing bytes ignored, as by readers before the slot block
    CHECK(read_dmesh(sb.data(), sb.size(), "upload", ms, err) && ms.size() == 1 && ms[0].slot.empty());
    buf[3] = '9'; err.clear(); CHECK(!read_dmesh(buf.data(), buf.size(), "upload", ms, err) && !err.empty());
  }

  // --- global slots: triangulate() tags each triangle with mesh base + face slot
  {
    Mesh m = polygon({0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0});
    m.counts = {3, 4}; m.indices = {0, 1, 2, 0, 1, 2, 3}; m.slot = {2, 0};
    Tris t; int skipped = 0;
    triangulate(m, zero, t, skipped, 5);
    CHECK(t.slot == std::vector<uint32_t>({7, 5, 5}) && slot_count(m) == 3);
    m.slot.clear(); t = Tris{};
    triangulate(m, zero, t, skipped);
    CHECK(t.slot == std::vector<uint32_t>({0, 0, 0}) && slot_count(m) == 1);
  }

  // --- thin solids: a 20 x 20 x 0.01 m pane box, indexed and as a soup (welded), flipped (inward), open, the 3 m cube,
  //     two touching panes in one slot, a frame (slot 0) sharing the glass box's (slot 1) corners, double-faced meshes
  {
    const Mesh pane = box_mesh("pane", 0, 0, 1, 20, 20, 1.01, false), soup = box_mesh("pane", 0, 0, 1, 20, 20, 1.01, true);
    CHECK(soup.numPoints() == 24 && pane.numPoints() == 8);
    CHECK(thin_solid(pane, 0, zero) == kSlotThin && thin_solid(soup, 0, zero) == kSlotThin);
    const double far[3] = {1e5, -3e5, 50};   // recentring changes nothing
    CHECK(thin_solid(soup, 0, far) == kSlotThin);
    Mesh flipped = soup;
    for (size_t f = 0; f < 6; ++f) std::swap(flipped.indices[4 * f + 1], flipped.indices[4 * f + 3]);
    CHECK(thin_solid(flipped, 0, zero) == (kSlotThin | kSlotInward));
    Mesh open = pane; open.counts.pop_back(); open.indices.resize(20);
    CHECK(thin_solid(open, 0, zero) == 0 && thin_solid(pane, 1, zero) == 0);   // open; a slot without faces
    Mesh bent = pane; std::swap(bent.indices[1], bent.indices[3]);              // the bottom wound against the others,
    const double mid[3] = {10, 10, 1.005};                                     // centred so the volume still reads thin
    CHECK(thin_solid(bent, 0, mid) == 0);
    CHECK(thin_solid(box_mesh("cube", -1.5, -1.5, 1, 1.5, 1.5, 4, false), 0, zero) == 0);   // closed, 2V/S = 1 m
    CHECK(thin_solid(merged(box_mesh("a", 0, 0, 0, 1, 1, 0.01, true), box_mesh("b", 1, 0, 0, 2, 1, 0.01, true)), 0, zero) == kSlotThin);
    Mesh win = box_mesh("window", 0, 0, 0, 1, 1, 0.01, false);   // glass: corners 0..7; frame ring in z = 0 around it
    win.points.insert(win.points.end(), {-0.1, -0.1, 0, 1.1, -0.1, 0, 1.1, 1.1, 0, -0.1, 1.1, 0});
    win.counts.insert(win.counts.end(), {4, 4, 4, 4});
    win.indices.insert(win.indices.end(), {8, 9, 1, 0, 9, 10, 3, 1, 10, 11, 2, 3, 11, 8, 0, 2});
    win.slot = {1, 1, 1, 1, 1, 1, 0, 0, 0, 0};
    CHECK(thin_solid(win, 1, zero) == kSlotThin && thin_solid(win, 0, zero) == 0);
    // double-faced: every face repeated with the opposite winding (SketchUp's two-sided export), its twin after or
    // before it, indexed or a soup; a pane box exported so (volumes cancel): closed, no volume -> twin, never inward
    Mesh twin = polygon({0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 2, 1});
    twin.counts.push_back(4); twin.indices.insert(twin.indices.end(), {3, 2, 1, 0});
    Mesh twinFirst = twin; twinFirst.indices = {3, 2, 1, 0, 0, 1, 2, 3};
    Mesh twinSoup = twin; twinSoup.points.insert(twinSoup.points.end(), {0, 2, 1, 2, 2, 1, 2, 0, 1, 0, 0, 1}); twinSoup.indices = {0, 1, 2, 3, 4, 5, 6, 7};
    for (const Mesh* m : {&twin, &twinFirst, &twinSoup}) CHECK(thin_solid(*m, 0, zero) == (kSlotThin | kSlotTwin) && thin_solid(*m, 0, far) == (kSlotThin | kSlotTwin));
    CHECK(thin_solid(merged(pane, flipped), 0, zero) == (kSlotThin | kSlotTwin));
  }

  // --- --optics grammar (cli::parse): split at the last '=', /SLOT only as trailing digits, kind, ';' packing, repeats
  {
    cli::Args a;
    CHECK(parse({"m.obj", "--optics", "Facade/East=0.3", "--optics", "/World/Roof/2=0.1,0.5;a=b=0,0.64,glass;*=0.25;w/=0.2,0.3,porous;x/07=0.4,0"}, a, err) == 0);
    const std::vector<Optics>& o = a.opt.optics;
    CHECK(o.size() == 6 && err.empty());
    if (o.size() == 6) {
      CHECK(o[0].mesh == "Facade/East" && o[0].slot == -1 && o[0].rho == 0.3 && o[0].tau == 0 && !o[0].glass);
      CHECK(o[1].mesh == "/World/Roof" && o[1].slot == 2 && o[1].rho == 0.1 && o[1].tau == 0.5 && !o[1].glass);   // porous by default
      CHECK(o[2].mesh == "a=b" && o[2].slot == -1 && o[2].rho == 0 && o[2].tau == 0.64 && o[2].glass);
      CHECK(o[3].mesh == "*" && o[3].slot == -1 && o[3].rho == 0.25);
      CHECK(o[4].mesh == "w/" && o[4].slot == -1 && o[4].tau == 0.3 && !o[4].glass);
      CHECK(o[5].mesh == "x" && o[5].slot == 7 && o[5].rho == 0.4 && o[5].tau == 0);
    }
    CHECK(!a.opt.transmission && a.opt.fresnel);   // defaults
    CHECK(parse({"m.obj", "--transmission", "on", "--fresnel", "off"}, a, err) == 0 && a.opt.transmission && !a.opt.fresnel);
    CHECK(parse({"m.obj", "--transmission", "off", "--fresnel", "on"}, a, err) == 0 && !a.opt.transmission && a.opt.fresnel);
    const char* bad[][2] = {   // every grammar / range / mode error: SPEC, a piece of the message
        {"wall", "NAME[/SLOT]="}, {"=0.2", "NAME[/SLOT]="}, {"a=0.2;", "NAME[/SLOT]="}, {"a=0.1,0.2,glass,1", "NAME[/SLOT]="},
        {"*/1=0.2", "/SLOT needs a mesh NAME"}, {"/3=0.2", "/SLOT needs a mesh NAME"}, {"a/65536=0.2", "SLOT 0..65535"},
        {"a=x", "bad number 'x'"}, {"a=0.2,", "bad number ''"}, {"a=0.2 ", "bad number"}, {"a=0.1,0.2,metal", "kind glass or porous"},
        {"a=1.2", "0..1"}, {"a=-0.1", "0..1"}, {"a=0.2,1.5", "0..1"}, {"a=nan", "0..1"}, {"a=0.6,0.6", "must not exceed 1"},
        {"a=0.1,0.5,glass", "RHO must be 0"}, {"a=0,0.95,glass", "(0, 0.918]"}, {"a=0,0,glass", "(0, 0.918]"}};
    for (const auto& b : bad) {
      const int rc = parse({"m.obj", "--optics", b[0]}, a, err);
      CHECK(rc == 1 && err.find(b[1]) != std::string::npos);
      if (rc != 1 || err.find(b[1]) == std::string::npos) std::printf("  --optics %s: rc %d, '%s'\n", b[0], rc, err.c_str());
    }
    CHECK(parse({"m.obj", "--optics", "a=0.7,0.3"}, a, err) == 0 && parse({"m.obj", "--optics", "a=0,0.918,glass"}, a, err) == 0);   // limits included
    CHECK(parse({"m.obj", "--transmission", "yes"}, a, err) == 1 && err == "--transmission off|on");
    CHECK(parse({"m.obj", "--fresnel", "1"}, a, err) == 1 && err == "--fresnel on|off");
    CHECK(parse({"m.obj", "--optics"}, a, err) == 1 && err == "missing value for --optics");
    CHECK(parse({"m.obj", "--transmission", "on", "--mode", "ladybug"}, a, err) == 1 && err.find("physical") != std::string::npos);
    CHECK(parse({"m.obj", "--mode", "ladybug", "--optics", "a=0.2,0.5", "--transmission", "off"}, a, err) == 0);   // optics: a no-op there
    Options lbt; lbt.mode = Mode::Ladybug; lbt.transmission = true;
    CHECK(!optics_ok(lbt, err));   // run() checks the same
    // --bounces 0..20, --bounce-samples 64..65536: whole numbers; ladybug refuses bounces (0 is a no-op there)
    CHECK(parse({"m.obj"}, a, err) == 0 && a.opt.bounces == 0 && a.opt.bounceSamples == 1024);
    CHECK(parse({"m.obj", "--bounces", "20", "--bounce-samples", "64"}, a, err) == 0 && a.opt.bounces == 20 && a.opt.bounceSamples == 64);
    CHECK(parse({"m.obj", "--bounces", "0", "--bounce-samples", "65536"}, a, err) == 0 && a.opt.bounces == 0 && a.opt.bounceSamples == 65536);
    const char* badB[][3] = {{"--bounces", "21", "0..20"}, {"--bounces", "-1", "0..20"}, {"--bounces", "1.5", "whole number"},
                             {"--bounces", "two", "whole number"}, {"--bounces", "", "whole number"}, {"--bounces", "99999999999", "whole number"},
                             {"--bounce-samples", "63", "64..65536"}, {"--bounce-samples", "65537", "64..65536"}, {"--bounce-samples", "1e3", "whole number"}};
    for (const auto& b : badB) {
      const int rc = parse({"m.obj", b[0], b[1]}, a, err);
      CHECK(rc == 1 && err.find(b[2]) != std::string::npos);
      if (rc != 1 || err.find(b[2]) == std::string::npos) std::printf("  %s '%s': rc %d, '%s'\n", b[0], b[1], rc, err.c_str());
    }
    CHECK(parse({"m.obj", "--bounces"}, a, err) == 1 && err == "missing value for --bounces");
    CHECK(parse({"m.obj", "--mode", "ladybug", "--bounces", "1"}, a, err) == 1 && err.find("physical") != std::string::npos);
    CHECK(parse({"m.obj", "--mode", "ladybug", "--bounces", "0", "--bounce-samples", "4096"}, a, err) == 0);
    Options lbb; lbb.mode = Mode::Ladybug; lbb.bounces = 2;
    Options few; few.bounceSamples = 0;
    CHECK(!optics_ok(lbb, err) && !optics_ok(few, err));   // run() checks the same
  }

  // --- resolution: precedence NAME/SLOT > NAME > * > built-in (rho 0.2, tau 0), glass tn, and every error
  {
    Mesh fe = polygon({0, 0, 0, 1, 0, 0, 0, 1, 0}); fe.name = "Facade/East";
    Mesh wall = polygon({0, 0, 0, 1, 0, 0, 0, 1, 0}); wall.name = "wall";
    wall.counts = {3, 3, 3}; wall.indices = {0, 1, 2, 0, 1, 2, 0, 1, 2}; wall.slot = {0, 2, 0};   // slots 0..2 (1 unused)
    Mesh roof = polygon({0, 0, 0, 1, 0, 0, 0, 1, 0}); roof.name = "roof";
    const std::vector<Mesh> ms{fe, wall, roof};
    std::vector<Slot> s;
    CHECK(resolve_optics(ms, {}, s, err) && s.size() == 5);
    for (const Slot& x : s) CHECK(x.rho == 0.2f && x.tau == 0 && x.flags == 0);
    std::vector<Optics> rows = {{"wall", 2, 0, 0.64, true}, {"*", -1, 0.3, 0, false}, {"wall", -1, 0.4, 0.1, false}, {"Facade/East", -1, 0.5, 0, false}};
    CHECK(resolve_optics(ms, rows, s, err) && s.size() == 5);
    if (s.size() == 5) {
      CHECK(s[0].rho == 0.5f && s[0].tau == 0);                                             // Facade/East: NAME over *
      CHECK(s[1].rho == 0.4f && s[1].tau == 0.1f && s[2].rho == 0.4f && s[1].flags == 0);  // wall/0, wall/1: NAME
      CHECK(s[3].rho == 0 && s[3].tau == 0.64f && s[3].flags == kSlotGlass && std::fabs(s[3].tn - 0.6975761815) < 1e-7);   // wall/2: NAME/SLOT
      CHECK(s[4].rho == 0.3f && s[4].tau == 0);                                             // roof: *
    }
    Mesh twin = roof;
    const struct { std::vector<Mesh> m; std::vector<Optics> r; const char* what; } bad[] = {
        {ms, {{"Facade", -1, 0.2, 0, false}}, "no mesh of that name"},
        {{fe, roof, twin}, {{"roof", -1, 0.2, 0, false}}, "several meshes have that name"},
        {ms, {{"wall", 3, 0.2, 0, false}}, "slot out of range (the mesh has slots 0..2)"},
        {ms, {{"roof", 1, 0.2, 0, false}}, "slot out of range (the mesh has slots 0..0)"},
        {ms, {{"wall", 1, 0.2, 0, false}, {"wall", 1, 0.3, 0, false}}, "--optics wall/1 given twice"},
        {ms, {{"*", -1, 0.2, 0, false}, {"*", -1, 0.3, 0, false}}, "--optics * given twice"}};
    for (const auto& b : bad) {
      err.clear();
      CHECK(!resolve_optics(b.m, b.r, s, err) && err.find(b.what) != std::string::npos);
    }
    CHECK(resolve_optics({fe, roof, twin}, {{"fe", -1, 0.2, 0, false}}, s, err) == false);   // unknown, with a repeated name elsewhere
    Mesh odd = wall; odd.slot.pop_back();                                                 // Mesh::slot not one per face
    err.clear(); CHECK(!resolve_optics({odd}, {}, s, err) && err.find("2 slots for 3 faces") != std::string::npos);
  }
  if (failures) std::printf("check_mesh: %d failure(s)\n", failures);
  return failures;
}
