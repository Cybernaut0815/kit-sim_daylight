// SPDX-License-Identifier: Apache-2.0 AND LicenseRef-Commons-Clause-1.0
// daylight: GPU annual daylight / solar simulation. Public API: generic face-vertex mesh in, value list out.
#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <utility>
#include <vector>

namespace daylight {

// Generic face-vertex mesh. Readers deliver world coordinates in metres, Z-up.
struct Mesh {
  std::string name;                   // OBJ group / file stem, or prim path
  std::vector<double> points;         // xyz interleaved
  std::vector<int> counts;            // vertices per face (faceVertexCounts)
  std::vector<int> indices;           // faceVertexIndices, right-handed winding (normals derived from geometry)
  bool sensor = true;                 // false = context: blocks rays only
  size_t numFaces() const { return counts.size(); }
  size_t numPoints() const { return points.size() / 3; }
};

// Hourly weather (EPW). dni/dhi: one value per row (8760 or 8784), W/m2 averaged over the hour ending at the row.
struct Weather {
  double lat = 0, lon = 0, tz = 0;   // degrees N+, degrees E+, hours E+
  bool leap = false;
  std::vector<float> dni, dhi;
  int missing = 0;                             // rows with missing (>= 9999) radiation, treated as 0
};

// One evaluation step. hour: local standard decimal hour at which the sun is evaluated (EPW-driven: row%24 + 0.5).
// jd: day of year as gendaymtx computes it (Feb 29 -> 60, later days +1 in leap years). dt: hours represented.
// hoy: hour of year (0 = Jan 1 00:00). dni/dhi: W/m2 of the EPW row containing the step (0 without weather).
struct Step { int month = 1, day = 1, jd = 1; double hour = 0, hoy = 0, dt = 1; float dni = 0, dhi = 0; };

enum class Mode { Physical, Ladybug };
enum class Quantity { Radiation, Direct, Diffuse, Irradiance, Illuminance, SunHours };
inline constexpr const char* kQuantityNames[] = {"radiation", "direct", "diffuse", "irradiance", "illuminance", "sunhours"};
enum class Sides { Front, Back, Both };

struct Options {
  Mode mode = Mode::Physical;
  std::vector<Quantity> quantities = {Quantity::Radiation};
  Sides sides = Sides::Front;
  bool cullBackfaces = false;         // occluders one-sided (winding); default two-sided like Radiance
  bool onVertices = false;            // sensors on points instead of faces
  bool selfOcclude = true;            // sensor meshes also block rays
  int mf = 1;                        // sky density: 1 (145 patches), 2 (577), 4 (2305)
  int patchSamples = 1;               // physical: K = m*m stratified rays per patch
  double ground = 0.2;                // ground reflectance
  double north = 0;                   // degrees, counterclockwise from +Y (ladybug convention; 90 = west)
  double offset = 0;                  // sensor offset along the normal, metres
  double lat = NAN, lon = NAN, tz = NAN;        // override EPW header / no-EPW runs
  double t0 = NAN, t1 = NAN;          // hour-of-year range [t0, t1); NAN = annual over EPW rows (midpoints)
  int stepMin = 60;                   // step for ranges, minutes
  std::string seriesPath;             // .npy time series sink (+ .hoys.txt); empty = off
  std::string dumpBitsPath, dumpSky, dumpSun;   // validation dumps; empty = off (GPU selection: CUDA_VISIBLE_DEVICES)
  // Scene identity for long-lived callers (a server). Non-empty and equal to the previous run's key: the occluder
  // triangles and the acceleration structure of that run stay resident and are reused (the caller guarantees the
  // meshes and selfOcclude are unchanged). Empty (the CLI): rebuilt every run. The GPU context is kept either way.
  std::string sceneKey;
  bool verbose = false;
};

struct Result {
  std::vector<double> values;                            // N x C row-major; N elements in input order, C = columns
  std::vector<std::string> columns;                      // e.g. "radiation_front", "sunhours_front"
  std::vector<std::pair<std::string, size_t>> ranges;    // (mesh name, element count) in input order
  double durationHours = 0;
  size_t nSky = 0, nSun = 0;                             // directions: [sky nSky | ground nSky (mirrored) | sun nSun]
};

// OBJ contract: world metres, Z-up, right-handed winding; one o/g group per mesh; 1-based global indices (v, f only).
bool read_obj(const std::string& path, std::vector<Mesh>& out, std::string& err);
// .dmesh contract: one mesh per file (name = file stem), little-endian, no padding after the 32-byte header:
//   char[4] "DLM1" | uint32 0 | uint64 n_points | uint64 n_faces | uint64 n_indices |
//   float64 xyz[3*n_points] (metres, Z-up) | int32 counts[n_faces] (>= 3, right-handed) | int32 indices[n_indices] (0-based)
bool read_dmesh(const std::string& path, std::vector<Mesh>& out, std::string& err);
// The same from memory (an upload); name is the mesh name (what the file stem is for the path overload).
bool read_dmesh(const char* data, size_t size, const std::string& name, std::vector<Mesh>& out, std::string& err);
bool read_epw(const std::string& path, Weather& out, std::string& err);
bool read_epw(std::istream& in, const std::string& label, Weather& out, std::string& err);   // label names the source in errors
// Synthetic 8760-row weather from the Ineichen-Perez clear-sky model (Linke turbidity, site altitude in metres):
// every hour cloudless, so radiation results are an upper bound, not a climate estimate. Sun at each hour midpoint
// (NOAA position); turbidity >= 1 (3 = clean temperate air, 5-7 = urban/humid).
bool clear_sky_weather(double lat, double lon, double tz, double turbidity, double altitude, Weather& out, std::string& err);
// weather may be null only when every quantity is SunHours.
bool run(const std::vector<Mesh>& meshes, const Weather* weather, const Options& opt, Result& out, std::string& err);

}  // namespace daylight
