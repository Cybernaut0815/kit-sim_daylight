// SPDX-License-Identifier: MIT
// Checks for mesh.cpp: OBJ / .dmesh readers, triangulation, sensor rules (ladybug / physical), sides, offset.
#include "internal.h"
#include <cmath>
#include <cstdio>
#include <string>

using namespace daylight;

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

  // --- .dmesh: quad + triangle round trip; bad magic, truncated (header only), index out of range -> false + err
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
  }
  if (failures) std::printf("check_mesh: %d failure(s)\n", failures);
  return failures;
}
