// SPDX-License-Identifier: Apache-2.0 AND LicenseRef-Commons-Clause-1.0
// read_usd checks against box.usda: hand-computed world Z-up metres, winding flips, pruning. Run: check_usd <dir>.
#include "daylight_usd.h"
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#define CHECK(cond) \
  do { if (!(cond)) { std::printf("%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #cond); ++fails; } } while (0)

namespace {
using daylight::Mesh;

const Mesh* find(const std::vector<Mesh>& ms, const char* name) {
  for (const Mesh& m : ms) if (m.name == name) return &m;
  return nullptr;
}
bool near3(const double* a, const double* b, double eps) {
  return std::fabs(a[0] - b[0]) < eps && std::fabs(a[1] - b[1]) < eps && std::fabs(a[2] - b[2]) < eps;
}
// cross(p1-p0, p2-p0) of face f (first three points) and the face centroid; returns dot(n, centroid - c).
double outward(const Mesh& m, size_t f, const double c[3], double n[3]) {
  const int* I = m.indices.data() + 4 * f;
  const double *p0 = &m.points[3 * size_t(I[0])], *p1 = &m.points[3 * size_t(I[1])], *p2 = &m.points[3 * size_t(I[2])];
  double e1[3], e2[3], ctr[3] = {0, 0, 0};
  for (int j = 0; j < 3; ++j) { e1[j] = p1[j] - p0[j]; e2[j] = p2[j] - p0[j]; }
  n[0] = e1[1] * e2[2] - e1[2] * e2[1]; n[1] = e1[2] * e2[0] - e1[0] * e2[2]; n[2] = e1[0] * e2[1] - e1[1] * e2[0];
  for (int k = 0; k < 4; ++k) for (int j = 0; j < 3; ++j) ctr[j] += m.points[3 * size_t(I[k]) + size_t(j)] / 4;
  return n[0] * (ctr[0] - c[0]) + n[1] * (ctr[1] - c[1]) + n[2] * (ctr[2] - c[2]);
}
}  // namespace

int main(int argc, char** argv) {
  int fails = 0;
  const std::string dir = argc > 1 ? argv[1] : ".";
  const std::string path = dir + "/box.usda";
  std::vector<Mesh> ms;
  std::string err;
  CHECK(!daylight::read_usd(dir + "/does_not_exist.usda", 'A', 0, ms, err));
  CHECK(!err.empty());
  err.clear();
  CHECK(daylight::read_usd(path, 'A', 0, ms, err));
  CHECK(err.empty());
  CHECK(ms.size() == 3);   // Hidden (invisible), Guide (purpose), Inactive (active = false), HiddenGroup/Child (inherited) pruned
  const Mesh *box = find(ms, "/World/Box"), *left = find(ms, "/World/BoxLeft"), *floor = find(ms, "/Floor");
  CHECK(box && left && floor);
  if (!box || !left || !floor) return 1;
  CHECK(ms[0].name == "/World/Box" && ms[1].name == "/World/BoxLeft" && ms[2].name == "/Floor");
  CHECK(box->sensor && left->sensor && floor->sensor);

  // /World: rotateY(90) then translate (100,200,300) cm: local (x,y,z) -> (z,y,-x) + t; * 0.01; Y-up -> Z-up (x,-z,y).
  const double c[3] = {1, -3, 2};   // box centre in Z-up metres
  const double boxPts[24] = {0.5, -3.5, 1.5, 0.5, -2.5, 1.5, 0.5, -2.5, 2.5, 0.5, -3.5, 2.5,
                             1.5, -3.5, 1.5, 1.5, -2.5, 1.5, 1.5, -2.5, 2.5, 1.5, -3.5, 2.5};
  const int boxIdx[24] = {0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4, 3, 7, 6, 2, 0, 4, 7, 3, 1, 2, 6, 5};
  // BoxLeft: scale (-1,1,1) under the same Xform: mirrored x before the rotation. Authored leftHanded (reversed runs)
  // and det < 0: the double flip cancels, indices stay as authored.
  const double leftPts[24] = {0.5, -2.5, 1.5, 0.5, -3.5, 1.5, 0.5, -3.5, 2.5, 0.5, -2.5, 2.5,
                              1.5, -2.5, 1.5, 1.5, -3.5, 1.5, 1.5, -3.5, 2.5, 1.5, -2.5, 2.5};
  const int leftIdx[24] = {1, 2, 3, 0, 7, 6, 5, 4, 4, 5, 1, 0, 2, 6, 7, 3, 3, 7, 4, 0, 5, 6, 2, 1};
  const Mesh* boxes[2] = {box, left};
  const double* expPts[2] = {boxPts, leftPts};
  const int* expIdx[2] = {boxIdx, leftIdx};
  for (int b = 0; b < 2; ++b) {
    const Mesh& m = *boxes[b];
    CHECK(m.numPoints() == 8 && m.numFaces() == 6 && m.indices.size() == 24);
    if (m.numPoints() != 8 || m.indices.size() != 24) continue;
    for (size_t i = 0; i < 8; ++i) CHECK(near3(&m.points[3 * i], expPts[b] + 3 * i, 1e-9));
    for (size_t f = 0; f < 6; ++f) CHECK(m.counts[f] == 4);
    for (size_t k = 0; k < 24; ++k) CHECK(m.indices[k] == expIdx[b][k]);
    for (size_t f = 0; f < 6; ++f) {   // right-hand rule on the delivered winding points outward on every face
      double n[3];
      const double d = outward(m, f, c, n);
      CHECK(d > 0.49 && d < 0.51);   // unit faces: |n| = 1, face centre 0.5 m from the centre
    }
  }
  // Floor: root prim, no xform; authored leftHanded (clockwise seen from +Y) -> single flip, right-hand normal up (+Z).
  const double floorPts[12] = {-10, 10, 0, 10, 10, 0, 10, -10, 0, -10, -10, 0};
  CHECK(floor->numPoints() == 4 && floor->numFaces() == 1 && floor->counts[0] == 4 && floor->indices.size() == 4);
  if (floor->numPoints() == 4 && floor->indices.size() == 4) {
    for (size_t i = 0; i < 4; ++i) CHECK(near3(&floor->points[3 * i], floorPts + 3 * i, 1e-9));
    CHECK(floor->indices[0] == 3 && floor->indices[1] == 2 && floor->indices[2] == 1 && floor->indices[3] == 0);
    double n[3];
    const double origin[3] = {0, 0, -1};
    CHECK(outward(*floor, 0, origin, n) > 0 && n[2] > 399 && n[2] < 401 && std::fabs(n[0]) < 1e-9 && std::fabs(n[1]) < 1e-9);
  }

  // Argument overrides: Z-up and metres as-is -> raw composed coordinates (cm, Y-up layout untouched).
  ms.clear();
  CHECK(daylight::read_usd(path, 'Z', 1.0, ms, err) && ms.size() == 3);
  if (ms.size() == 3) {
    const double raw0[3] = {50, 150, 350};
    CHECK(near3(&ms[0].points[0], raw0, 1e-9));
  }
  std::printf("%s: %d failure(s)\n", fails ? "FAIL" : "OK", fails);
  return fails ? 1 : 0;
}
