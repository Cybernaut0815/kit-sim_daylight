// SPDX-License-Identifier: Apache-2.0 AND LicenseRef-Commons-Clause-1.0
// NOAA solar position with ladybug Sunpath semantics, EPW reader, Ineichen clear sky, evaluation steps. Equations are
// the public-domain NOAA solar calculator (Meeus) chain, written from the NOAA spreadsheet; no ladybug code.
#include "internal.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace daylight {
namespace {
const double kPi = 3.14159265358979323846;
double rad(double d) { return d * kPi / 180; }
double deg(double r) { return r * 180 / kPi; }
double pymod(double a, double m) { double r = std::fmod(a, m); return r < 0 ? r + m : r; }   // Python %, result >= 0
int days_in_month(int m, bool leap) {
  static const int d[12] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
  return d[m - 1] + (leap && m == 2 ? 1 : 0);
}
// Days in the Gregorian years 1900..year-1 (1900 is not leap); + doy + 1 gives the Excel serial the NOAA spreadsheet
// (and ladybug) use: 2017-01-01 -> 42736 = JD 2457754.5.
constexpr int leaps_before(int y) { --y; return y / 4 - y / 100 + y / 400; }
constexpr int days_from_1900(int year) { return (year - 1900) * 365 + leaps_before(year) - leaps_before(1900); }
static_assert(days_from_1900(2017) == 42734 && days_from_1900(2016) == 42368, "ladybug anchors");
}  // namespace

Sun sun_noaa(double lat, double lon, double tz, int month, int day, double hourFrac, bool leap) {
  int hour = (int)hourFrac, minute = (int)std::lround((hourFrac - hour) * 60);   // ladybug DateTime: minute resolution
  if (minute == 60) { ++hour; minute = 0; }
  int doy = day;
  for (int m = 1; m < month; ++m) doy += days_in_month(m, leap);
  // round_fmt "%.2f" == Python round(x, 2) (exact-value ties-to-even)
  double jd = days_from_1900(leap ? 2016 : 2017) + doy + 1 + 2415018.5 + round_fmt((minute + hour * 60) / 1440.0, "%.2f") - tz / 24;
  double jc = (jd - 2451545) / 36525;
  double L0 = pymod(280.46646 + jc * (36000.76983 + jc * 0.0003032), 360);        // geometric mean longitude (deg)
  double M = 357.52911 + jc * (35999.05029 - 0.0001537 * jc);                      // geometric mean anomaly (deg)
  double e = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc);                 // eccentricity
  double C = std::sin(rad(M)) * (1.914602 - jc * (0.004817 + 0.000014 * jc)) +
             std::sin(rad(2 * M)) * (0.019993 - 0.000101 * jc) + std::sin(rad(3 * M)) * 0.000289;   // equation of centre
  double omega = rad(125.04 - 1934.136 * jc);
  double lambda = rad(L0 + C - 0.00569 - 0.00478 * std::sin(omega));              // apparent longitude
  double eps0 = 23 + (26 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60) / 60;
  double eps = rad(eps0 + 0.00256 * std::cos(omega));                              // corrected obliquity
  double dec = std::asin(std::sin(eps) * std::sin(lambda));
  double y = std::tan(eps / 2);
  y *= y;
  double eot = 4 * deg(y * std::sin(2 * rad(L0)) - 2 * e * std::sin(rad(M)) + 4 * e * y * std::sin(rad(M)) * std::cos(2 * rad(L0)) -
                       0.5 * y * y * std::sin(4 * rad(L0)) - 1.25 * e * e * std::sin(2 * rad(M)));   // minutes
  double tst = pymod(hourFrac * 60 + eot + 4 * lon - 60 * tz, 1440);              // true solar time (minutes)
  double H = tst < 0 ? tst / 4 + 180 : tst / 4 - 180;                              // hour angle (deg)
  double la = rad(lat);
  double zen = std::acos(std::sin(la) * std::sin(dec) + std::cos(la) * std::cos(dec) * std::cos(rad(H)));
  double alt = 90 - deg(zen), t = std::tan(rad(alt)), r;                           // refraction in arcseconds
  if (alt > 85) r = 0;
  else if (alt > 5) r = 58.1 / t - 0.07 / (t * t * t) + 0.000086 / (t * t * t * t * t);
  else if (alt > -0.575) r = 1735 + alt * (-518.2 + alt * (103.4 + alt * (-12.79 + alt * 0.711)));
  else r = -20.772 / t;
  Sun s;
  s.alt = alt + r / 3600;
  double a = (std::sin(la) * std::cos(zen) - std::sin(dec)) / (std::cos(la) * std::sin(zen));   // unrefracted zenith
  if (a >= -1 && a <= 1) s.azi = pymod(H > 0 ? deg(std::acos(a)) + 180 : 540 - deg(std::acos(a)), 360);
  else s.azi = a > 0 ? 180 : 0;   // exact solar noon: acos domain error in ladybug
  s.up = s.alt >= 0;
  s.vec = dir_from_altaz(rad(s.alt), rad(s.azi));
  return s;
}

bool read_epw(const std::string& path, Weather& w, std::string& err) {
  std::ifstream f(path);
  if (!f) { err = "cannot open EPW " + path; return false; }
  return read_epw(f, path, w, err);
}

bool read_epw(std::istream& f, const std::string& path, Weather& w, std::string& err) {
  std::string line;
  std::vector<std::string> cols;
  auto split = [&](const std::string& l) {
    cols.clear();
    std::stringstream ss(l);
    for (std::string c; std::getline(ss, c, ',');) cols.push_back(c);
  };
  bool hdrLeap = false;
  for (int i = 0; i < 8; ++i) {
    if (!std::getline(f, line)) { err = "EPW header truncated: " + path; return false; }
    split(line);
    if (i == 0) {
      if (cols.size() < 10 || cols[0] != "LOCATION") { err = "EPW: bad LOCATION line: " + path; return false; }
      w.lat = std::atof(cols[6].c_str()); w.lon = std::atof(cols[7].c_str()); w.tz = std::atof(cols[8].c_str());
    }
    if (i == 4 && cols.size() > 1 && cols[1] == "Yes") hdrLeap = true;
  }
  w.dni.clear(); w.dhi.clear(); w.missing = 0;
  while (std::getline(f, line)) {
    if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
    split(line);
    if (cols.size() < 16) { err = "EPW: short data row " + std::to_string(w.dni.size() + 1) + ": " + path; return false; }
    double dni = std::atof(cols[14].c_str()), dhi = std::atof(cols[15].c_str());
    if (dni >= 9999 || dhi >= 9999) { ++w.missing; if (dni >= 9999) dni = 0; if (dhi >= 9999) dhi = 0; }
    w.dni.push_back((float)dni); w.dhi.push_back((float)dhi);
  }
  size_t n = w.dni.size();
  w.leap = hdrLeap || n == 8784;
  // ponytail: hourly 8760/8784 rows only; sub-hourly or partial-year EPWs would need their own step mapping.
  if (n != (w.leap ? 8784u : 8760u)) {
    err = "EPW: " + std::to_string(n) + " data rows, expected " + (w.leap ? "8784" : "8760") + ": " + path;
    return false;
  }
  return true;
}

double extraterrestrial_irradiance(int doy) {   // Spencer 1971 as in pvlib
  double B = 2 * kPi * (doy - 1) / 365;
  return 1366.1 * (1.00011 + 0.034221 * std::cos(B) + 0.00128 * std::sin(B) + 0.000719 * std::cos(2 * B) + 0.000077 * std::sin(2 * B));
}

void ineichen(double z, double TL, double altitude, double dniExtra, double& ghi, double& dni, double& dhi) {
  z = std::min(std::max(z, 0.0), 90.0);
  double cosz = std::cos(rad(z));
  double am = 1 / (cosz + 0.50572 * std::pow(6.07995 + (90 - z), -1.6364));              // Kasten-Young 1989
  double pressure = 100 * std::pow((44331.514 - altitude) / 11880.516, 1 / 0.1902632);   // sea-level standard
  double ama = am * pressure / 101325;
  double fh1 = std::exp(-altitude / 8000), fh2 = std::exp(-altitude / 1250);
  double cg1 = 5.09e-5 * altitude + 0.868, cg2 = 3.92e-5 * altitude + 0.0387;
  ghi = std::max(0.0, cg1 * dniExtra * cosz * std::exp(-cg2 * ama * (fh1 + fh2 * (TL - 1))));
  double b = 0.664 + 0.163 / fh1;
  double bnci = dniExtra * std::max(0.0, b * std::exp(-0.09 * ama * (TL - 1)));
  double bnci2 = ghi * std::min(std::max((1 - (0.1 - 0.2 * std::exp(-TL)) / (0.1 + 0.882 / fh1)) / cosz, 0.0), 1e20);
  dni = std::min(bnci, bnci2);
  dhi = ghi - dni * cosz;
}

bool clear_sky_weather(double lat, double lon, double tz, double turbidity, double altitude, Weather& out, std::string& err) {
  if (turbidity < 1) { err = "turbidity must be >= 1"; return false; }
  if (std::fabs(lat) > 90) { err = "latitude out of range"; return false; }
  out.lat = lat; out.lon = lon; out.tz = tz; out.leap = false; out.missing = 0;
  out.dni.assign(8760, 0.f); out.dhi.assign(8760, 0.f);
  for (int r = 0; r < 8760; ++r) {
    int month, day; double hour;
    hoy_to_date(r + 0.5, false, month, day, hour);
    Sun s = sun_noaa(lat, lon, tz, month, day, hour, false);
    if (!s.up) continue;
    double ghi, dni, dhi;
    ineichen(90 - s.alt, turbidity, altitude, extraterrestrial_irradiance(r / 24 + 1), ghi, dni, dhi);
    out.dni[r] = (float)dni; out.dhi[r] = (float)dhi;
  }
  return true;
}

void hoy_to_date(double hoy, bool leap, int& month, int& day, double& hour) {
  int d = (int)std::floor(hoy / 24);
  hour = hoy - d * 24.0;
  month = 1;
  while (month < 12 && d >= days_in_month(month, leap)) d -= days_in_month(month++, leap);
  day = d + 1;
}

bool parse_hoy(const std::string& s, bool leap, double& hoy, std::string& err) {
  int y, m, d, h, mi;
  if (std::sscanf(s.c_str(), "%d-%d-%dT%d:%d", &y, &m, &d, &h, &mi) != 5 && std::sscanf(s.c_str(), "%d-%dT%d:%d", &m, &d, &h, &mi) != 4) {
    err = "bad timestamp '" + s + "', expected [YYYY-]MM-DDTHH:MM"; return false;
  }
  if (m < 1 || m > 12 || d < 1 || d > days_in_month(m, leap) || h < 0 || h > 23 || mi < 0 || mi > 59) {
    err = "timestamp out of range '" + s + "'"; return false;
  }
  hoy = 24.0 * (d - 1) + h + mi / 60.0;
  for (int k = 1; k < m; ++k) hoy += 24 * days_in_month(k, leap);
  return true;
}

bool make_steps(const Options& opt, const Weather* w, std::vector<Step>& out, std::string& err) {
  out.clear();
  bool leap = w && w->leap;
  if (std::isnan(opt.t0)) {   // annual: one step per EPW row at the row midpoint
    if (!w) { err = "no weather and no time range: nothing to evaluate"; return false; }
    for (size_t r = 0; r < w->dni.size(); ++r) {
      Step s;
      s.hoy = r + 0.5; s.jd = (int)(r / 24) + 1; s.dt = 1; s.dni = w->dni[r]; s.dhi = w->dhi[r];
      hoy_to_date(s.hoy, leap, s.month, s.day, s.hour);
      out.push_back(s);
    }
    return true;
  }
  if (opt.stepMin <= 0) { err = "step must be a positive number of minutes"; return false; }
  double step = opt.stepMin / 60.0, nh = leap ? 8784 : 8760;
  double t1 = std::isnan(opt.t1) ? opt.t0 + step : opt.t1;
  if (t1 > nh) t1 = nh;
  if (opt.t0 < 0 || opt.t0 >= nh || t1 <= opt.t0) { err = "time range must satisfy 0 <= from < to <= end of year"; return false; }
  for (int k = 0;; ++k) {
    double hoy = opt.t0 + k * step;
    if (hoy >= t1 - 1e-9) break;
    double hod = (hoy - 24 * std::floor(hoy / 24)) * 60;   // minutes into the day: the --hours window
    if (hod >= 1440 - 1e-6) hod -= 1440;                    // a midnight step that lands at 24n - eps (step 10 min)
    if (hod < opt.dayFrom - 1e-6 || hod >= opt.dayTo - 1e-6) continue;
    Step s;
    s.hoy = hoy; s.jd = (int)(hoy / 24) + 1; s.dt = step;
    hoy_to_date(hoy, leap, s.month, s.day, s.hour);
    size_t r = (size_t)hoy;   // ponytail: containing EPW row, no sub-hourly interpolation
    if (w && r < w->dni.size()) { s.dni = w->dni[r]; s.dhi = w->dhi[r]; }
    out.push_back(s);
  }
  if (out.empty()) { err = "no step falls inside the --hours window"; return false; }
  return true;
}

}  // namespace daylight
