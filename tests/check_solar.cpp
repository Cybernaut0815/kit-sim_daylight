// SPDX-License-Identifier: MIT
// Checks for solar.cpp: NOAA sun vs ladybug Sunpath goldens, EPW reader, steps, timestamp parsing.
#include "internal.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>

using namespace daylight;
static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("%s:%d: CHECK(%s)\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

// Golden: ladybug-core 0.44 Sunpath(lat, lon, tz).calculate_sun_from_hoy(hoy) (leap: is_leap_year = True), altitude,
// azimuth, is_during_day. Generated with uv run --with ladybug-core (build/tmp/solar/golden.py).
struct Gold { double lat, lon, tz, hoy; int leap; double alt, azi; int up; };
static const Gold kGold[] = {
  {42.35, -71.07, -5, 12.5, 0, 24.0339386184163, 190.570642204051, 1},
  {42.35, -71.07, -5, 4000.5, 0, 29.08381246151125, 276.17896241480855, 1},
  {42.35, -71.07, -5, 6000.25, 0, -41.44263283012595, 10.983536987101843, 0},
  {42.35, -71.07, -5, 8759.5, 0, -70.30471699994148, 347.8405810050913, 0},
  {42.35, -71.07, -5, 1428.5, 1, 39.48797980663118, 190.7228271557164, 1},   // Feb 29 12:30 leap
  {42.35, -71.07, -5, 6000.25, 1, -41.17039406533562, 10.845617532167182, 0},
  {-33.87, 151.21, 10, 12.5, 0, 77.13561721832852, 325.6862040558454, 1},
  {-33.87, 151.21, 10, 4000.5, 0, 3.5430411612210313, 301.1471210914584, 1},
  {-33.87, 151.21, 10, 6000.25, 0, -61.52147685262607, 168.5136689766449, 0},
  {-33.87, 151.21, 10, 8759.5, 0, -32.671122980132544, 187.7313007570311, 0},
  {-33.87, 151.21, 10, 1428.5, 1, 63.47570158069742, 347.5427234050982, 1},
  {-33.87, 151.21, 10, 6000.25, 1, -61.81145969247512, 168.54733691494903, 0},
  {69.65, 18.96, 1, 12.5, 0, -2.8060492434552717, 189.71332252273297, 0},
  {69.65, 18.96, 1, 4000.5, 0, 28.348217517311056, 260.943393834801, 1},
  {69.65, 18.96, 1, 6000.25, 0, -14.404862608925326, 8.487049213244063, 0},
  {69.65, 18.96, 1, 8759.5, 0, -43.295128829433864, 354.4791363604088, 0},
  {69.65, 18.96, 1, 1428.5, 1, 12.506761401540151, 188.4741942527532, 1},
  {69.65, 18.96, 1, 6000.25, 1, -14.124331495896154, 8.405926941386554, 0},
  {1.35, 103.82, 8, 12.5, 0, 63.953180935220225, 159.62552457750758, 1},
  {1.35, 103.82, 8, 4000.5, 0, 35.840247685373875, 298.1647603729691, 1},
  {1.35, 103.82, 8, 6000.25, 0, -76.11471777161485, 301.2009173781879, 0},
  {1.35, 103.82, 8, 8759.5, 0, -57.832628764322145, 225.72183675574692, 0},
  {1.35, 103.82, 8, 1428.5, 1, 75.0873589975341, 128.06493639024552, 1},
  {1.35, 103.82, 8, 6000.25, 1, -75.91337880896288, 302.0675774002865, 0},
  {28.6, 77.2, 5.5, 12.5, 0, 38.42996071686319, 181.53487192212, 1},
  {28.6, 77.2, 5.5, 4000.5, 0, 34.6367900660623, 279.9266392744926, 1},
  {28.6, 77.2, 5.5, 6000.25, 0, -55.60136715411337, 358.2081230414227, 0},
  {28.6, 77.2, 5.5, 8759.5, 0, -76.56423773821679, 291.3410151747154, 0},
  {28.6, 77.2, 5.5, 1428.5, 1, 53.62972517352374, 178.46450224504974, 1},
  {28.6, 77.2, 5.5, 6000.25, 1, -55.31611991426416, 358.10944649135683, 0},
  // refraction branches (Boston): > 5 deg, 0..5 deg polynomial, below -0.575 deg, exact x.0 hours
  {42.35, -71.07, -5, 4001.0, 0, 23.61253947328314, 280.82961400006536, 1},
  {42.35, -71.07, -5, 4003.0, 0, 3.079793521974994, 299.4863519386687, 1},
  {42.35, -71.07, -5, 4747.5, 0, -2.6631840530198203, 302.01819322052427, 0},
  {42.35, -71.07, -5, 4747.7, 0, -4.570041547695895, 304.11659632398306, 0},
  {42.35, -71.07, -5, 100.0, 0, -34.9456080460268, 89.79282732386088, 0},
};

static void write_epw(const char* path, int rows, const char* leapFlag) {
  std::ofstream f(path);
  f << "LOCATION,Synth,ST,CC,src,000000,1.5,2.5,3.0,4.0\nDESIGN CONDITIONS,0\nTYPICAL/EXTREME PERIODS,0\nGROUND TEMPERATURES,0\n"
    << "HOLIDAYS/DAYLIGHT SAVINGS," << leapFlag << ",0,0,0\nCOMMENTS 1\nCOMMENTS 2\nDATA PERIODS,1,1,Data,Sunday, 1/ 1,12/31\n";
  for (int r = 0; r < rows; ++r) {   // DNI = r % 1000, DHI = r % 500; row 100 DHI missing, row 200 DNI missing
    int dni = r == 200 ? 9999 : r % 1000, dhi = r == 100 ? 9999 : r % 500;
    f << "2016,1,1," << r % 24 + 1 << ",60,flags,0,0,0,0,0,0,0,0," << dni << "," << dhi << ",0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n";
  }
}

int check_solar(const char*) {
  // ---- NOAA sun vs ladybug (1e-5 deg; identical is_during_day)
  for (const Gold& g : kGold) {
    int month, day; double hour;
    hoy_to_date(g.hoy, g.leap != 0, month, day, hour);
    Sun s = sun_noaa(g.lat, g.lon, g.tz, month, day, hour, g.leap != 0);
    CHECK(std::fabs(s.alt - g.alt) < 1e-5);
    CHECK(std::fabs(s.azi - g.azi) < 1e-5);
    CHECK(s.up == (g.up != 0));
    if (std::fabs(s.alt - g.alt) >= 1e-5 || std::fabs(s.azi - g.azi) >= 1e-5)
      std::printf("  hoy %g leap %d: got %.9f %.9f want %.9f %.9f\n", g.hoy, g.leap, s.alt, s.azi, g.alt, g.azi);
  }
  {   // sun vector (ladybug sun_vector_reversed), north 0 and north 30 (counterclockwise, applied by rotate_z)
    Sun s = sun_noaa(42.35, -71.07, -5, 6, 16, 16.5, false);   // hoy 4000.5
    CHECK(std::fabs(s.vec.x - -0.8688326578327976) < 1e-6 && std::fabs(s.vec.y - 0.0940626653218609) < 1e-6 && std::fabs(s.vec.z - 0.4860884977817768) < 1e-6);
    const float3 n = rotate_z(s.vec, 30);
    CHECK(std::fabs(n.x - -0.799462485981686) < 1e-6 && std::fabs(n.y - -0.35295567119999366) < 1e-6 && std::fabs(n.z - s.vec.z) < 1e-7);
  }

  // ---- clear sky: pvlib 0.16.1 goldens (ineichen without Perez enhancement, dni_extra 1366.1; Spencer)
  struct CsGold { double z, TL, alt, ghi, dni, dhi; };
  static const CsGold kCs[] = {
    {0, 3, 0, 1055.832800, 943.707724, 112.125077}, {30, 3, 0, 898.145558, 917.861089, 103.254538},
    {60, 3, 0, 470.345298, 789.020222, 75.835187},  {80, 3, 0, 107.650399, 413.343189, 35.874107},
    {85, 3, 0, 31.236607, 176.747515, 15.832046},   {89, 3, 0, 0.975523, 9.913176, 0.802514},
    {30, 4, 1500, 949.498908, 906.503037, 164.444250}, {70, 4, 1500, 293.039303, 611.234749, 83.984706},
  };
  auto rel = [](double got, double want) { return std::fabs(got - want) / want < 1e-6; };
  for (const CsGold& g : kCs) {
    double ghi, dni, dhi;
    ineichen(g.z, g.TL, g.alt, 1366.1, ghi, dni, dhi);
    CHECK(rel(ghi, g.ghi) && rel(dni, g.dni) && rel(dhi, g.dhi));
    if (!(rel(ghi, g.ghi) && rel(dni, g.dni) && rel(dhi, g.dhi))) std::printf("  z %g TL %g: got %.6f %.6f %.6f\n", g.z, g.TL, ghi, dni, dhi);
  }
  CHECK(rel(extraterrestrial_irradiance(1), 1413.981805) && rel(extraterrestrial_irradiance(172), 1321.623593) && rel(extraterrestrial_irradiance(355), 1412.708564));
  std::string err;
  {   // Boston TL 3: annual sums vs pvlib get_clearsky (SPA positions; NOAA differs ~0.01 deg)
    Weather w;
    CHECK(!clear_sky_weather(42.35, -71.07, -5, 0.5, 0, w, err));
    CHECK(clear_sky_weather(42.35, -71.07, -5, 3, 0, w, err) && w.dni.size() == 8760 && w.dhi.size() == 8760 && !w.leap && w.tz == -5);
    CHECK(w.dni[11] > 0 && w.dhi[11] > 0 && w.dni[0] == 0 && w.dhi[0] == 0 && w.dni[23] == 0 && w.dhi[23] == 0);
    double sdni = 0, sdhi = 0;
    for (size_t r = 0; r < w.dni.size(); ++r) { sdni += w.dni[r]; sdhi += w.dhi[r]; }
    CHECK(std::fabs(sdni / 1000 - 2960.51) < 0.005 * 2960.51 && std::fabs(sdhi / 1000 - 294.55) < 0.01 * 294.55);
    std::printf("  clear-sky Boston TL3: dni %.2f dhi %.2f kWh/m2 (pvlib 2960.51 / 294.55)\n", sdni / 1000, sdhi / 1000);
  }

  // ---- EPW: Boston TMY
  Weather bos;
  CHECK(read_epw("C:/DAYSIM/wea/USA_MA_Boston-City.WSO_TMY.epw", bos, err));
  CHECK(bos.lat == 42.35 && bos.lon == -71.07 && bos.tz == -5 && !bos.leap);
  CHECK(bos.dni.size() == 8760 && bos.dhi.size() == 8760 && bos.missing == 0);
  CHECK(bos.dni.size() > 3991 && bos.dni[11] == 2 && bos.dhi[11] == 67 && bos.dni[3991] == 532 && bos.dhi[3991] == 191);   // Jan 1 h12; Jun 16 h8
  // ---- EPW: synthetic leap file with 9999 values
  std::string tmp = std::getenv("TEMP") ? std::getenv("TEMP") : ".";   // scratch EPWs; cwd-independent
  std::string leapPath = tmp + "/daylight_check_leap.epw", badPath = tmp + "/daylight_check_bad.epw";
  write_epw(leapPath.c_str(), 8784, "Yes");
  Weather lp;
  CHECK(read_epw(leapPath, lp, err));
  CHECK(lp.leap && lp.dni.size() == 8784 && lp.lat == 1.5 && lp.lon == 2.5 && lp.tz == 3);
  CHECK(lp.missing == 2 && lp.dni[100] == 100 && lp.dhi[100] == 0 && lp.dni[200] == 0 && lp.dhi[200] == 200 && lp.dni[8783] == 783);
  write_epw(badPath.c_str(), 8000, "No");
  CHECK(!read_epw(badPath, lp, err) && err.find("8000") != std::string::npos);
  CHECK(!read_epw(tmp + "/daylight_check_does_not_exist.epw", lp, err));
  write_epw(leapPath.c_str(), 8784, "No");   // leap inferred from the row count alone
  CHECK(read_epw(leapPath, lp, err) && lp.leap);

  // ---- steps: annual
  Options opt;
  std::vector<Step> st;
  CHECK(make_steps(opt, &bos, st, err) && st.size() == 8760);
  if (st.size() == 8760) {
    CHECK(st[0].hoy == 0.5 && st[0].hour == 0.5 && st[0].month == 1 && st[0].day == 1 && st[0].jd == 1 && st[0].dt == 1);
    CHECK(st[8759].hoy == 8759.5 && st[8759].hour == 23.5 && st[8759].month == 12 && st[8759].day == 31 && st[8759].jd == 365);
    CHECK(st[1415].month == 2 && st[1415].day == 28 && st[1415].jd == 59 && st[1416].month == 3 && st[1416].day == 1 && st[1416].jd == 60);
    CHECK(st[3991].dni == 532 && st[3991].dhi == 191 && st[3991].hour == 7.5 && st[3991].month == 6 && st[3991].day == 16);
  }
  CHECK(make_steps(opt, &lp, st, err) && st.size() == 8784);
  if (st.size() == 8784) {
    CHECK(st[1428].month == 2 && st[1428].day == 29 && st[1428].jd == 60 && st[1428].hour == 12.5);
    CHECK(st[1452].month == 3 && st[1452].day == 1 && st[1452].jd == 61);
    CHECK(st[8783].month == 12 && st[8783].day == 31 && st[8783].jd == 366 && st[8783].hoy == 8783.5);
  }
  CHECK(!make_steps(opt, nullptr, st, err));   // nothing to evaluate
  // ---- steps: range / single time
  CHECK(parse_hoy("06-21T06:00", false, opt.t0, err) && opt.t0 == 4110);
  opt.t1 = 4113; opt.stepMin = 30;
  CHECK(make_steps(opt, &bos, st, err) && st.size() == 6);
  if (st.size() == 6) {
    for (int k = 0; k < 6; ++k) CHECK(st[k].hour == 6 + 0.5 * k && st[k].hoy == 4110 + 0.5 * k && st[k].dt == 0.5 && st[k].month == 6 && st[k].day == 21 && st[k].jd == 172);
    CHECK(st[0].dni == 286 && st[0].dhi == 144 && st[1].dni == 286 && st[2].dni == bos.dni[4111]);
  }
  opt.stepMin = 10; opt.t1 = 4111;
  CHECK(make_steps(opt, &bos, st, err) && st.size() == 6 && std::fabs(st[5].hour - (6 + 50 / 60.0)) < 1e-9);   // no fp extra step at 7:00
  opt.t1 = NAN; opt.stepMin = 60;
  CHECK(make_steps(opt, &bos, st, err) && st.size() == 1 && st[0].hour == 6 && st[0].dt == 1);   // --time: one exact step
  CHECK(make_steps(opt, nullptr, st, err) && st.size() == 1 && st[0].dni == 0 && st[0].dhi == 0);   // no weather: zero irradiance
  opt.t0 = 8759.5;
  CHECK(make_steps(opt, &bos, st, err) && st.size() == 1 && st[0].hour == 23.5);   // t1 clamped to the end of year
  opt.t0 = 10; opt.t1 = 5;
  CHECK(!make_steps(opt, &bos, st, err));
  opt.t1 = 20; opt.stepMin = 0;
  CHECK(!make_steps(opt, &bos, st, err));

  // ---- parse_hoy / hoy_to_date round trips
  double hoy; int month, day; double hour;
  CHECK(parse_hoy("2017-06-21T12:30", false, hoy, err) && hoy == 4116.5);
  hoy_to_date(hoy, false, month, day, hour);
  CHECK(month == 6 && day == 21 && hour == 12.5);
  CHECK(parse_hoy("12-31T23:59", false, hoy, err) && std::fabs(hoy - (8759 + 59 / 60.0)) < 1e-12);
  hoy_to_date(hoy, false, month, day, hour);
  CHECK(month == 12 && day == 31 && std::fabs(hour - (23 + 59 / 60.0)) < 1e-12);
  CHECK(parse_hoy("01-01T00:00", true, hoy, err) && hoy == 0);
  CHECK(parse_hoy("02-29T00:00", true, hoy, err) && hoy == 1416);
  hoy_to_date(1416, true, month, day, hour);
  CHECK(month == 2 && day == 29 && hour == 0);
  CHECK(parse_hoy("03-01T00:00", true, hoy, err) && hoy == 1440 && parse_hoy("03-01T00:00", false, hoy, err) && hoy == 1416);
  CHECK(parse_hoy("12-31T00:00", true, hoy, err) && hoy == 365 * 24);
  CHECK(!parse_hoy("02-29T00:00", false, hoy, err));
  CHECK(!parse_hoy("13-01T00:00", false, hoy, err) && !parse_hoy("06-21T24:00", false, hoy, err) && !parse_hoy("06-21T12:60", false, hoy, err));
  CHECK(!parse_hoy("garbage", false, hoy, err) && !parse_hoy("06-21", false, hoy, err) && !parse_hoy("", false, hoy, err));
  for (int r = 0; r < 8784; r += 7) {   // every hoy survives hoy_to_date -> format -> parse_hoy
    hoy_to_date(r + 0.25, true, month, day, hour);
    char buf[32];
    std::snprintf(buf, sizeof buf, "%02d-%02dT%02d:%02d", month, day, (int)hour, (int)std::lround((hour - (int)hour) * 60));
    CHECK(parse_hoy(buf, true, hoy, err) && hoy == r + 0.25);
  }
  return fails;
}
