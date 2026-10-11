// SPDX-License-Identifier: MIT
// Checks for sky.cpp: patch geometry, gendaymtx port identities, AddDirect, sun.c, round_fmt.
#include "internal.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <cstdio>
#include <cstring>

namespace daylight { double tsin_radiance(double x); }   // sky.cpp test hook: Radiance table sine used for normalisation
using namespace daylight;

#define CHECK(cond) do { if (!(cond)) { std::printf("%s:%d: CHECK(%s)\n", __FILE__, __LINE__, #cond); ++fails; } } while (0)

static const double PI = 3.14159265358979323846;
static double deg(double r) { return r * 180 / PI; }
static bool near(double a, double b, double tol) { return std::fabs(a - b) <= tol; }
static bool rel(double a, double b, double tol) { return std::fabs(a - b) <= tol * std::fabs(b); }

int check_sky(const char*) {
  int fails = 0;
  // ---- patches
  const int counts[3][2] = {{1, 145}, {2, 577}, {4, 2305}};
  for (auto& c : counts) {
    const Patches pa = make_patches(c[0]);
    CHECK(pa.P == c[1]);
    CHECK((int)pa.centre.size() == c[1] && (int)pa.lb.size() == c[1] && (int)pa.omega.size() == c[1]);
    double sum = 0; for (float w : pa.omega) sum += w;
    CHECK(near(sum, 2 * PI, 1e-6));
    CHECK(pa.row.back() == 7 * c[0] && pa.ninrow.back() == 1 && pa.row[0] == 0 && pa.ninrow[0] == 30 * c[0]);
  }
  const Patches t = make_patches(1);
  const double altC[7] = {6, 18, 30, 42, 54, 66, 78};
  const double altLB[7] = {5.967369, 17.907707, 29.787295, 41.755320, 53.581926, 65.253386, 76.209964};
  int p0 = 0;
  for (int i = 0; i < 7; i++) {
    CHECK(near(deg(t.alt[p0]), altC[i], 1e-4));
    CHECK(near(deg(std::asin(t.lb[p0].z)), altLB[i], 1e-5));
    CHECK(near(std::atan2(t.lb[p0].x, t.lb[p0].y), t.azi[p0], 1e-6));   // same azimuth as the centre
    p0 += t.ninrow[p0];
  }
  CHECK(p0 == 144);
  const Patches r = make_patches(2);
  int p12 = 0; for (int i = 0; i < 12; i++) p12 += r.ninrow[p12];
  CHECK(r.row[p12] == 12 && near(deg(r.alt[p12]) - deg(std::asin(r.lb[p12].z)), 0.424, 0.01));
  // vectors: patch 0 due north, patch 1 clockwise toward east, last = zenith
  CHECK(t.centre[0].x == 0.f && near(t.centre[0].y, std::cos(6 * PI / 180), 1e-6) && near(t.centre[0].z, std::sin(6 * PI / 180), 1e-6));
  CHECK(t.centre[1].x > 0.f && t.centre[1].y > 0.f);
  CHECK(near(t.centre[144].x, 0, 1e-6) && near(t.centre[144].y, 0, 1e-6) && near(t.centre[144].z, 1, 1e-6));
  CHECK(t.lb[144].x == 0.f && t.lb[144].y == 0.f && t.lb[144].z == 1.f);
  // samples: K == 1 is the centre, K == 4 strata lie inside the patch
  for (int p = 0; p < t.P; p++) {
    const float3 c = patch_sample(t, p, 0, 1);
    CHECK(std::memcmp(&c, &t.centre[p], sizeof c) == 0);
    for (int k = 0; k < 4; k++) {
      const float3 s = patch_sample(t, p, k, 4);
      const double alt = std::asin((double)s.z), a = t.alpha, i = t.row[p];
      CHECK(near(s.x * s.x + s.y * s.y + s.z * s.z, 1, 1e-6));
      if (p == t.P - 1) { CHECK(alt >= PI / 2 - a / 2 - 1e-6); continue; }
      CHECK(alt >= a * i - 1e-6 && alt <= a * (i + 1) + 1e-6);
      double daz = std::fmod(std::atan2((double)s.x, (double)s.y) - t.azi[p] + 3 * PI, 2 * PI) - PI;
      CHECK(std::fabs(daz) <= PI / t.ninrow[p] + 1e-6);
      CHECK(std::memcmp(&s, &c, sizeof s) != 0);
    }
  }
  // dir_from_altaz / rotate_z: azimuth clockwise from +Y; north rotation counterclockwise about +Z
  const float3 e = dir_from_altaz(0, PI / 2);
  CHECK(near(e.x, 1, 1e-7) && near(e.y, 0, 1e-7) && e.z == 0.f);
  const float3 w = rotate_z(float3{0.f, 1.f, 0.f}, 90);
  CHECK(near(w.x, -1, 1e-7) && near(w.y, 0, 1e-7) && w.z == 0.f);

  // ---- gendaymtx_step
  std::vector<float> out(3 * (t.P + 1));
  const double alt45 = PI / 4, south = PI;
  // night
  gendaymtx_step(t, 172, alt45, south, 5e-5, 5e-5, 0.2, true, true, false, out.data());
  { bool zero = true; for (float v : out) zero = zero && v == 0.f; CHECK(zero); }
  // overcast-ish sky only: DHI = 100, DNI = 0, -O1. Expected: ComputeSky scales L_p so that
  //   sum_p out_G[p] * rh_cos(p) * omega_p = diff_illum / 179 * skycolor_G = (100 * 179) / 179 * 1.004f = 100.39999
  // (W/m2/sr integrating to the DHI; the -O1 "*179" and the output "/179" cancel), where rh_cos is Radiance's table
  // sine tsin (tcos.c, 1024 entries, no interpolation). With the true sine the identity only holds to the table's
  // accuracy (2.3e-4 here), so both are checked at their own tolerance.
  gendaymtx_step(t, 172, alt45, south, 0, 100, 0.2, true, true, false, out.data());
  {
    const double expect = 100.0 * (double)1.004f;
    double sTab = 0, sTrue = 0;
    for (int p = 0; p < t.P; p++) {
      sTab += out[3 * (p + 1) + 1] * tsin_radiance(t.alt[p]) * t.omega[p];
      sTrue += out[3 * (p + 1) + 1] * std::sin((double)t.alt[p]) * t.omega[p];
    }
    CHECK(rel(sTab, expect, 1e-6));
    CHECK(rel(sTrue, expect, 1e-3));
    CHECK(rel(out[4] / out[3], 1.004 / 0.96, 1e-6) && rel(out[5] / out[3], 1.118 / 0.96, 1e-6));   // skycolor
    CHECK(rel(out[0], 0.2 * 100 / PI, 1e-5) && out[1] == out[0] && out[2] == out[0]);              // ground row
    bool pos = true; for (int p = 0; p < t.P; p++) pos = pos && out[3 * (p + 1)] > 0.f; CHECK(pos);
  }
  // Perez parameters: hand-evaluated relative luminance ratios (independent of the normalisation) from Table 1 rows
  // 1 and 8 of Perez 1993, incl. the category-0 c/d special forms. Patches: 0 = north horizon, 15 = south horizon
  // (sun side), P-1 = zenith; sun at 45 deg south. delta = DHI*airmass/(1367*E0), no 0.2 clamp (eps = 1 and 6.32).
  {
    const double z = PI / 4, da = 171 * 2 * PI / 365;
    const double E0 = 1.00011 + 0.034221 * std::cos(da) + 0.00128 * std::sin(da) + 0.000719 * std::cos(2 * da) + 0.000077 * std::sin(2 * da);
    const double D = 100 / (std::cos(z) + 0.15 * std::pow(93.885 - 45, -1.253)) / (1367 * E0);
    auto L = [&](const double* c, int p) {
      const double zeta = PI / 2 - t.alt[p], g = std::acos(std::cos(z) * std::cos(zeta) + std::sin(z) * std::sin(zeta) * std::cos(std::fabs(t.azi[p] - south)));
      return (1 + c[0] * std::exp(c[1] / std::cos(zeta))) * (1 + c[2] * std::exp(c[3] * g) + c[4] * std::cos(g) * std::cos(g));
    };
    const double c0[5] = {1.3525 - 0.2576 * z + D * (-0.2690 - 1.4366 * z), -0.7670 + 0.0007 * z + D * (1.2734 - 0.1233 * z),
                          std::exp(std::pow(D * (2.8 + 0.6004 * z), 1.2375)) - 1.0, -std::exp(D * (1.8734 + 0.6297 * z)) + 0.9738 + D * 0.2809,
                          0.0356 - 0.1246 * z + D * (-0.5718 + 0.9938 * z)};
    const double c7[5] = {-1.0500 + 0.0289 * z + D * (0.4260 + 0.3590 * z), -0.3250 + 0.1156 * z + D * (0.7781 + 0.0025 * z),
                          31.0625 - 14.5 * z + D * (-46.1148 + 55.375 * z), -7.2312 + 0.4050 * z + D * (13.35 + 0.6234 * z),
                          1.5 - 0.6426 * z + D * (1.8564 + 0.5636 * z)};
    gendaymtx_step(t, 172, alt45, south, 0, 100, 0.2, false, true, false, out.data());      // eps = 1 -> category 0
    CHECK(rel(out[3 * t.P + 1] / out[4], L(c0, t.P - 1) / L(c0, 0), 1e-5) && rel(out[3 * 16 + 1] / out[4], L(c0, 15) / L(c0, 0), 1e-5));
    gendaymtx_step(t, 172, alt45, south, 800, 100, 0.2, false, true, false, out.data());    // eps = 6.32 -> category 7
    CHECK(rel(out[3 * t.P + 1] / out[4], L(c7, t.P - 1) / L(c7, 0), 1e-5) && rel(out[3 * 16 + 1] / out[4], L(c7, 15) / L(c7, 0), 1e-5));
  }
  // ground row: sun-only mode zeroes grefl (-d); sky-only keeps the direct term (dir_illum sin alt)
  gendaymtx_step(t, 172, alt45, south, 800, 100, 0.2, true, false, false, out.data());
  CHECK(out[0] == 0.f && out[1] == 0.f && out[2] == 0.f);
  gendaymtx_step(t, 172, alt45, south, 800, 100, 0.2, false, true, false, out.data());
  CHECK(rel(out[0], 0.2 * (100 + 800 * std::sin(alt45)) / PI, 1e-5));
  // AddDirect: sun only, DNI = 800 -> sum_p out_G omega_p = dir_illum/179 = DNI (-O1) over exactly the 4 nearest patches
  {
    const double alt = 35 * PI / 180, azi = 200 * PI / 180;
    gendaymtx_step(t, 172, alt, azi, 800, 100, 0.2, true, false, false, out.data());
    double sum = 0; int nz = 0; bool rgbEqual = true;
    std::vector<double> dot(t.P), sorted; const float3 sv = dir_from_altaz(alt, azi);
    for (int p = 0; p < t.P; p++) {
      const float g = out[3 * (p + 1) + 1];
      sum += g * t.omega[p]; nz += g != 0.f ? 1 : 0;
      rgbEqual = rgbEqual && out[3 * (p + 1)] == g && out[3 * (p + 1) + 2] == g;
      dot[p] = (double)t.centre[p].x * sv.x + (double)t.centre[p].y * sv.y + (double)t.centre[p].z * sv.z;
    }
    CHECK(rel(sum, 800.0, 1e-5) && nz == 4 && rgbEqual && out[0] == 0.f);
    sorted = dot; std::sort(sorted.begin(), sorted.end(), std::greater<double>());
    bool nearest = true; for (int p = 0; p < t.P; p++) nearest = nearest && ((out[3 * (p + 1) + 1] != 0.f) == (dot[p] >= sorted[3]));
    CHECK(nearest);
    // sun below the 1e-4 lux threshold adds nothing; sky-only mode adds nothing
    gendaymtx_step(t, 172, alt, azi, 800, 100, 0.2, false, true, false, out.data());
    double sky = 0; for (int p = 0; p < t.P; p++) sky += out[3 * (p + 1) + 1] * tsin_radiance(t.alt[p]) * t.omega[p];
    CHECK(rel(sky, 100.0 * (double)1.004f, 1e-6));
  }
  // DHI = 0, DNI = 800: eps = inf -> 11.9 (category 7), delta -> 0.01; nothing NaN/inf on either output path
  for (int vis = 0; vis < 2; vis++) {
    gendaymtx_step(t, 172, alt45, south, 800, 0, 0.2, true, true, vis != 0, out.data());
    bool fin = true; for (float v : out) fin = fin && std::isfinite(v); CHECK(fin);
    CHECK(out[0] > 0.f);
  }
  // -O0 efficacy: clear sky at Jun 21 noon, DNI 800 DHI 100; category 7 (eps > 6.2); hand evaluation of Eqns 7/8
  {
    gendaymtx_step(t, 172, alt45, south, 800, 100, 0.2, true, true, true, out.data());
    const double z = PI / 4, W = std::exp(0.07 * 11 - 0.075), da = 171 * 2 * PI / 365;
    const double E0 = 1.00011 + 0.034221 * std::cos(da) + 0.00128 * std::sin(da) + 0.000719 * std::cos(2 * da) + 0.000077 * std::sin(2 * da);
    const double m = 1 / (std::cos(z) + 0.15 * std::pow(93.885 - 45, -1.253)), delta = 100 * m / (1367 * E0);
    // eps = (9 + 1.041 z^3) / (1 + 1.041 z^3) = 6.32 > 6.2 -> category 7 (the coefficients below); checked via the results
    const double dif = 100 * (152.23 + 0.35 * W - 45.27 * std::cos(z) - 7.98 * std::log(delta));
    const double dir = 800 * (101.18 + 1.58 * W - 1.10 * std::exp(5.73 * z - 5) - 8.29 * delta);
    CHECK(rel(out[0], 0.2 * (dif + dir * std::sin(z)) / PI / 179, 1e-5));
    gendaymtx_step(t, 172, alt45, south, 800, 100, 0.2, false, true, true, out.data());   // sky only: normalised to dif
    double sTab = 0; for (int p = 0; p < t.P; p++) sTab += out[3 * (p + 1) + 1] * tsin_radiance(t.alt[p]) * t.omega[p];
    CHECK(rel(sTab, (dif / 179) * (double)1.004f, 1e-5));
    gendaymtx_step(t, 172, alt45, south, 800, 100, 0.2, true, false, true, out.data());   // sun only: AddDirect of dir
    double sSun = 0; for (int p = 0; p < t.P; p++) sSun += out[3 * (p + 1) + 1] * t.omega[p];
    CHECK(rel(sSun, dir / 179, 1e-5));
  }

  // ---- round_fmt
  CHECK(round_fmt(0.123456, "%.3g") == 0.123 && round_fmt(1234.5, "%.3g") == 1230 && round_fmt(0, "%.3g") == 0 && round_fmt(42.346, "%.2f") == 42.35);

  // ---- sun.c
  double alt, azi;
  sun_radiance(42.35, 71.07, 75, 172, 11.75, alt, azi);            // Boston, ~solar noon
  CHECK(deg(alt) > 70 && deg(alt) < 72 && near(deg(azi), 180, 30));
  sun_radiance(42.35, 71.07, 75, 172, 12.5, alt, azi);             // afternoon: sun west of south
  CHECK(near(deg(alt), 68.97980, 1e-3) && deg(azi) > 180 && near(deg(azi), 209.4386, 1e-3));
  sun_radiance(42.35, 71.07, 75, 355, 12, alt, azi);               // Dec 21 noon, low sun, still due south-ish
  CHECK(deg(alt) > 23 && deg(alt) < 26 && near(deg(azi), 180, 10));
  // hand-derived: at the pole alt == sdec(jd); sdec(172) = 0.4093 sin(2 pi 91/365) = 0.409296209774023
  sun_radiance(90, 0, 0, 172, 0, alt, azi);
  CHECK(near(alt, 0.409296209774023, 1e-12));
  // hand-derived: lat 0, jd 81 (sdec 0), hour 0 -> alt = |st| pi/12 - pi/2 with st = stadj(81) = -0.118294008300896 h
  sun_radiance(0, 0, 0, 81, 0, alt, azi);
  CHECK(near(alt + PI / 2, 0.118294008300896 * PI / 12, 1e-12));
  // meridian 15 deg west of the site adds exactly +1 h of solar time
  sun_radiance(0, 0, 15, 81, 0, alt, azi);
  CHECK(near(alt + PI / 2, (1 - 0.118294008300896) * PI / 12, 1e-12));
  return fails;
}
